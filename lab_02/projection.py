"""Проекция дорожки и вращающегося ромба на видео с движущейся камеры."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np


LAB_DIR = Path(__file__).resolve().parent
MIN_TRACKS = 12


class Camera:
    def __init__(self, calibration):
        storage = cv2.FileStorage(str(calibration), cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise OSError(f"Не удалось открыть калибровку: {calibration}")
        try:
            self.K = storage.getNode("K").mat()
            self.D = storage.getNode("D").mat()
            size = storage.getNode("sz")
            self.size = (int(size.at(0).real()), int(size.at(1).real()))
        finally:
            storage.release()
        if (self.K is None or self.K.shape != (3, 3)
                or self.D is None or self.D.size != 5 or min(self.size) <= 0):
            raise ValueError("Ожидаются K (3×3), пять коэффициентов D и разрешение")

    def project(self, points, pose):
        rotation, translation = pose
        pixels, _ = cv2.projectPoints(points, rotation, translation, self.K, self.D)
        return pixels.reshape(-1, 2)

    @staticmethod
    def center(pose):
        rotation, translation = pose
        matrix, _ = cv2.Rodrigues(rotation)
        return (-matrix.T @ translation).ravel()


def read_scene(path):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise OSError(f"Не удалось открыть настройки сцены: {path}")
    try:
        scene = {}
        for key in ("start_frame", "end_frame", "frame_step"):
            scene[key] = int(storage.getNode(key).real())
        for key in (
            "tile_size_m", "way_width_m", "way_start_m", "way_end_m",
            "diamond_radius_m", "diamond_speed_deg_s",
        ):
            scene[key] = storage.getNode(key).real()
        center = storage.getNode("diamond_center_m")
        scene["center"] = np.array([center.at(i).real() for i in range(3)])
        scene["anchors"] = storage.getNode("anchors_px").mat()
    finally:
        storage.release()
    if not (0 <= scene["start_frame"] < scene["end_frame"] and scene["frame_step"] > 0):
        raise ValueError("Проверьте диапазон и шаг кадров в настройках сцены")
    if (scene["tile_size_m"] <= 0 or scene["anchors"] is None
            or scene["anchors"].shape != (24, 2)):
        raise ValueError("Нужны положительная сторона плитки и 24 стыка плитки")
    if not (scene["way_width_m"] > 0 and scene["way_end_m"] > scene["way_start_m"]):
        raise ValueError("Проверьте ширину и границы дорожки")
    if scene["diamond_radius_m"] <= 0 or scene["center"][2] <= 0:
        raise ValueError("Ромб должен иметь положительный радиус и высоту")
    return scene


def reprojection_rms(camera, world, pixels, pose):
    delta = camera.project(world, pose) - pixels.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def reference_pose(frame, camera, scene):
    pixels = scene["anchors"].astype(np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.02)
    pixels = cv2.cornerSubPix(gray, pixels, (6, 6), (-1, -1), criteria)
    world = np.array([(x, y, 0) for y in range(6) for x in range(-1, 3)], np.float32)
    world *= scene["tile_size_m"]
    ok, rotation, translation = cv2.solvePnP(world, pixels, camera.K, camera.D)
    if not ok:
        raise ValueError("Не удалось оценить позу по стыкам плитки")
    pose = (rotation, translation)
    return pose, reprojection_rms(camera, world, pixels, pose)


class GroundTracker:
    """Переносит координаты ориентиров на плоскости Z=0 между кадрами."""

    def __init__(self, camera, frame, pose, way_width):
        self.camera = camera
        self.pose = pose
        self.way_width = way_width
        self.gray = self.tracking_image(frame)
        self.pixels = np.empty((0, 1, 2), dtype=np.float32)
        self.world = np.empty((0, 3), dtype=np.float32)
        self.add_features()
        if len(self.pixels) < MIN_TRACKS:
            raise ValueError("На опорном кадре слишком мало точек на тротуаре")

    @staticmethod
    def tracking_image(frame):
        return cv2.pyrDown(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def add_features(self):
        center = self.camera.center(self.pose)
        near, far = center[1] + 2.7, center[1] + 12.0
        left = -self.way_width / 2 - 0.13
        right = self.way_width / 2 + 0.35
        region = np.array([[left, near, 0], [right, near, 0],
                           [right, far, 0], [left, far, 0]])
        outline = np.rint(self.camera.project(region, self.pose) / 2).astype(np.int32)
        mask = np.zeros_like(self.gray)
        cv2.fillConvexPoly(mask, outline, 255)
        height = mask.shape[0]
        mask[:round(height * 0.406)] = 0
        mask[round(height * 0.802):] = 0
        for point in np.rint(self.pixels[:, 0]).astype(int):
            cv2.circle(mask, tuple(point), 9, 0, -1)
        new_pixels = cv2.goodFeaturesToTrack(
            self.gray, maxCorners=300 - len(self.pixels), qualityLevel=0.006,
            minDistance=10, mask=mask,
        )
        if new_pixels is None:
            return

        # Из исправленных пиксельных координат строим лучи и пересекаем их с Z=0.
        normalized = cv2.undistortPoints(new_pixels * 2, self.camera.K, self.camera.D)
        rotation, _ = cv2.Rodrigues(self.pose[0])
        rays = np.column_stack((normalized.reshape(-1, 2), np.ones(len(new_pixels)))) @ rotation
        valid = rays[:, 2] < -0.02
        rays = rays[valid]
        ground = center - rays * (center[2] / rays[:, 2:3])
        ground[:, 2] = 0
        self.pixels = np.concatenate((self.pixels, new_pixels[valid]))
        self.world = np.concatenate((self.world, ground.astype(np.float32)))

    def update(self, frame):
        gray = self.tracking_image(frame)
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.pixels, None, winSize=(21, 21), maxLevel=3
        )
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, moved, None, winSize=(21, 21), maxLevel=3
        )
        distance = np.linalg.norm((back - self.pixels).reshape(-1, 2), axis=1)
        valid = (status[:, 0] != 0) & (back_status[:, 0] != 0) & (distance < 0.8)
        pixels = moved[valid]
        world = self.world[valid]
        if len(world) < MIN_TRACKS:
            raise ValueError("Потеряны опорные точки на плоскости")

        corrected = cv2.undistortPoints(
            pixels * 2, self.camera.K, self.camera.D, P=self.camera.K
        )
        _, inliers = cv2.findHomography(world[:, :2], corrected, cv2.RANSAC, 2.5)
        if inliers is None or np.count_nonzero(inliers) < MIN_TRACKS:
            raise ValueError("Слишком мало согласованных точек на плоскости")
        valid = inliers[:, 0].astype(bool)
        self.world, self.pixels = world[valid], pixels[valid]

        rotation, translation = self.pose
        ok, rotation, translation = cv2.solvePnP(
            self.world, self.pixels * 2, self.camera.K, self.camera.D,
            rotation, translation, useExtrinsicGuess=True,
        )
        if not ok:
            raise ValueError("Не удалось восстановить позу камеры")
        self.pose = (rotation, translation)
        rms = reprojection_rms(self.camera, self.world, self.pixels * 2, self.pose)
        self.gray = gray
        tracked = len(self.pixels)
        if tracked < 130:
            self.add_features()
        return tracked, rms


class WayEstimator:
    def __init__(self, width, start, end):
        y = np.linspace(start, end, max(2, int(np.ceil((end - start) / 0.10)) + 1))
        self.lines = [np.column_stack((np.full_like(y, x), y, np.zeros_like(y)))
                      for x in (-width / 2, width / 2)]

    def draw(self, frame, camera, pose):
        rotation, _ = cv2.Rodrigues(pose[0])
        for line in self.lines:
            depth = line @ rotation[2] + pose[1][2, 0]
            visible = line[depth > 1.2]
            if len(visible) < 2:
                continue
            pixels = np.rint(camera.project(visible, pose)).astype(np.int32)
            cv2.polylines(frame, [pixels], False, (35, 45, 45), 11, cv2.LINE_AA)
            cv2.polylines(frame, [pixels], False, (227, 231, 70), 5, cv2.LINE_AA)


class RotatingDiamond:
    def __init__(self, center, radius, degrees_per_second):
        self.center = center
        self.speed = np.deg2rad(degrees_per_second)
        height = center[2]
        self.vertices = np.array([[radius, 0, 0], [0, radius, 0], [-radius, 0, 0],
                                  [0, -radius, 0], [0, 0, height], [0, 0, -height]])
        self.faces = [(4, i, (i + 1) % 4) for i in range(4)]
        self.faces += [(5, (i + 1) % 4, i) for i in range(4)]

    def draw(self, frame, camera, pose, elapsed):
        half_angle = (self.speed * elapsed + np.pi / 7) / 2
        axis_part = np.array([0, 0, np.sin(half_angle)])
        twice_cross = 2 * np.cross(axis_part, self.vertices)
        # Векторная форма поворота единичным кватернионом q p q⁻¹.
        vertices = self.vertices + np.cos(half_angle) * twice_cross
        vertices += np.cross(axis_part, twice_cross)
        vertices += self.center
        rotation, _ = cv2.Rodrigues(pose[0])
        in_camera = vertices @ rotation.T + pose[1].ravel()
        if np.min(in_camera[:, 2]) <= 1.2:
            return
        pixels = np.rint(camera.project(vertices, pose)).astype(np.int32)
        surface = frame.copy()
        outlines = []
        for index, face in enumerate(self.faces):
            a, b, c = in_camera[list(face)]
            if np.dot(np.cross(b - a, c - a), a) < 0:
                contour = pixels[list(face)]
                color = (43, 191 if index % 2 else 139, 255)
                cv2.fillConvexPoly(surface, contour, color, cv2.LINE_AA)
                outlines.append(contour)
        cv2.addWeighted(surface, 0.34, frame, 0.66, 0, dst=frame)
        cv2.polylines(frame, outlines, True, (36, 53, 75), 8, cv2.LINE_AA)
        cv2.polylines(frame, outlines, True, (85, 209, 255), 4, cv2.LINE_AA)


def save_poses(path, camera, scene, poses, metrics):
    poses = np.asarray(poses)
    metrics = np.asarray(metrics)
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise OSError(f"Не удалось сохранить параметры: {path}")
    try:
        storage.write("K", camera.K)
        storage.write("D", camera.D)
        storage.write("r", poses[0, 1:4].reshape(3, 1))
        storage.write("t", poses[0, 4:7].reshape(3, 1))
        storage.write("pose_convention", "world_to_camera_Rodrigues_meters")
        storage.write("tile_size_m", scene["tile_size_m"])
        storage.write("reference_rms_px", float(metrics[0, 1]))
        storage.startWriteStruct("sz", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
        for dimension in camera.size:
            storage.write("", dimension)
        storage.endWriteStruct()
        storage.write("poses", poses)
        storage.write("tracking", metrics)
    finally:
        storage.release()


def read_frame(video, index):
    ok, frame = video.read()
    if not ok:
        raise OSError(f"Не удалось прочитать кадр {index}")
    return frame


def render(source, camera, scene, destination):
    video = cv2.VideoCapture(str(source))
    if not video.isOpened():
        raise OSError(f"Не удалось открыть видео: {source}")
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        start, end, step = (scene[key] for key in ("start_frame", "end_frame", "frame_step"))
        if fps <= 0 or not video.set(cv2.CAP_PROP_POS_FRAMES, start):
            raise ValueError("Не удалось установить начальный кадр и частоту видео")
        frame = read_frame(video, start)
        if (frame.shape[1], frame.shape[0]) != camera.size:
            raise ValueError("Разрешение ролика должно совпадать с калибровкой")

        pose, reference_error = reference_pose(frame, camera, scene)
        tracker = GroundTracker(camera, frame, pose, scene["way_width_m"])
        way = WayEstimator(scene["way_width_m"], scene["way_start_m"], scene["way_end_m"])
        diamond = RotatingDiamond(scene["center"], scene["diamond_radius_m"],
                                  scene["diamond_speed_deg_s"])
        poses = []
        metrics = []
        preview_index = start + ((end - start) // (2 * step)) * step
        video_path = destination / "overlay.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"avc1"),
                                 fps / step, camera.size)
        if not writer.isOpened():
            writer.release()
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / step, camera.size)
        if not writer.isOpened():
            raise OSError("Не удалось открыть файл выходного видео")

        try:
            for index in range(start, end, step):
                if index == start:
                    tracked, rms = len(tracker.pixels), reference_error
                else:
                    for _ in range(step - 1):
                        if not video.grab():
                            raise OSError(f"Не удалось перейти к кадру {index}")
                    frame = read_frame(video, index)
                    try:
                        tracked, rms = tracker.update(frame)
                    except ValueError as error:
                        raise ValueError(f"Кадр {index}: {error}") from error

                rotation, translation = tracker.pose
                poses.append([index, *rotation.ravel(), *translation.ravel()])
                metrics.append([tracked, rms])
                way.draw(frame, camera, tracker.pose)
                diamond.draw(frame, camera, tracker.pose, (index - start) / fps)
                writer.write(frame)
                if index == preview_index and not cv2.imwrite(str(destination / "preview.jpg"), frame):
                    raise OSError("Не удалось сохранить кадр для отчёта")
        finally:
            writer.release()
    finally:
        video.release()
    save_poses(destination / "leftImage.yml", camera, scene, poses, metrics)
    return np.asarray(poses), np.asarray(metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=LAB_DIR / "footage.avi")
    parser.add_argument("--calibration", type=Path, default=LAB_DIR.parent / "lab_01/results/calibration.yml")
    parser.add_argument("--scene", type=Path, default=LAB_DIR / "scene.yml")
    parser.add_argument("--output-dir", type=Path, default=LAB_DIR / "results")
    args = parser.parse_args()

    try:
        camera = Camera(args.calibration)
        scene = read_scene(args.scene)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output_dir) as directory:
            pending = Path(directory)
            poses, metrics = render(args.video, camera, scene, pending)
            for name in ("overlay.mp4", "preview.jpg", "leftImage.yml"):
                (pending / name).replace(args.output_dir / name)
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"{error}\n")

    print(f"Сохранено кадров: {len(poses)}. Ошибка опорной позы: {metrics[0, 1]:.2f} px.")
    if len(poses) > 1:
        print(f"Медианная ошибка трекинга: {np.median(metrics[1:, 1]):.2f} px, "
              f"минимум точек: {int(metrics[1:, 0].min())}.")
    print(f"Результаты: {args.output_dir}")


if __name__ == "__main__":
    main()
