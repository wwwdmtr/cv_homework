"""Калибровка камеры по видео с шахматной доской 8×8."""

import argparse
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np


PATTERN = (7, 7)
LAB_DIR = Path(__file__).resolve().parent


def collect_observations(video, step):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video}")

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    frame_indices = []
    image_points = []
    example = None
    try:
        for index in count():
            if not capture.grab():
                break
            if index % step:
                continue

            ok, frame = capture.retrieve()
            if not ok:
                raise ValueError(f"Не удалось прочитать кадр {index}")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, PATTERN, flags)
            if not found:
                continue

            if example is None:
                example = frame.copy()
            if frame.shape[:2] != example.shape[:2]:
                raise ValueError("В видео изменилось разрешение кадров")

            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            frame_indices.append(index)
            image_points.append(refined)
    finally:
        capture.release()

    if len(image_points) < 3:
        raise ValueError(f"Шаблон найден на {len(image_points)} кадрах, нужно хотя бы три")

    return index, frame_indices, image_points, example


def comparison_images(frame, corners, camera_matrix, distortion, image_size):
    original = cv2.drawChessboardCorners(frame, PATTERN, corners, True)

    map_xy, map_frac = cv2.initUndistortRectifyMap(
        camera_matrix, distortion, None, camera_matrix, image_size, cv2.CV_16SC2
    )
    remapped = cv2.remap(original, map_xy, map_frac, cv2.INTER_LINEAR)

    optimal_matrix, (x, y, width, height) = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion, image_size, 1, image_size
    )
    corrected = cv2.undistort(original, camera_matrix, distortion, None, optimal_matrix)
    if width > 0 and height > 0:
        cv2.rectangle(corrected, (x, y), (x + width - 1, y + height - 1), (0, 0, 255), 2)

    return {
        "01-original.jpg": original,
        "02-remap.jpg": remapped,
        "03-undistort.jpg": corrected,
    }


def save_results(output_dir, image_size, parameters, images):
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as directory:
        pending = Path(directory)
        yaml_path = pending / "calibration.yml"
        storage = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_WRITE)
        if not storage.isOpened():
            raise OSError(f"Не удалось открыть файл для записи: {yaml_path}")
        try:
            for key, value in parameters.items():
                storage.write(key, value)
            storage.startWriteStruct("sz", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
            for dimension in image_size:
                storage.write("", dimension)
            storage.endWriteStruct()
        finally:
            storage.release()

        for filename, image in images.items():
            if not cv2.imwrite(str(pending / filename), image):
                raise OSError(f"Не удалось сохранить изображение: {filename}")

        for filename in (*images, yaml_path.name):
            (pending / filename).replace(output_dir / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=LAB_DIR / "checkerboard.avi")
    parser.add_argument("--square-mm", type=float, default=16.25, help="Сторона клетки в мм")
    parser.add_argument("--step", type=int, default=50, help="Шаг выборки кадров")
    parser.add_argument("--output-dir", type=Path, default=LAB_DIR / "results")
    args = parser.parse_args()
    if args.step < 1 or not np.isfinite(args.square_mm) or args.square_mm <= 0:
        parser.error("Шаг выборки и размер клетки должны быть положительными")

    try:
        frame_count, frame_indices, image_points, example = collect_observations(
            args.video, args.step
        )
        height, width = example.shape[:2]
        image_size = (width, height)
        board_points = np.zeros((PATTERN[0] * PATTERN[1], 3), dtype=np.float32)
        board_points[:, :2] = np.mgrid[:PATTERN[0], :PATTERN[1]].T.reshape(-1, 2)
        board_points *= args.square_mm

        rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
            [board_points] * len(image_points), image_points, image_size, None, None
        )
        parameters = {
            "K": camera_matrix,
            "D": distortion,
            "rms_px": rms,
            "square_size_mm": args.square_mm,
            "frame_step": args.step,
            "frame_count": frame_count,
            "frame_indices": np.asarray(frame_indices, dtype=np.int32).reshape(-1, 1),
            "rvecs": np.asarray(rotations).reshape(-1, 3),
            "tvecs_mm": np.asarray(translations).reshape(-1, 3),
        }
        images = comparison_images(example, image_points[0], camera_matrix, distortion, image_size)
        save_results(args.output_dir, image_size, parameters, images)
    except (ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"{error}\n")

    sampled = (frame_count + args.step - 1) // args.step
    print(f"Прочитано кадров: {frame_count}. Доска найдена на {len(image_points)} из {sampled}.")
    print(f"Разрешение: {width} × {height}. Кадр для сравнения: {frame_indices[0]}.")
    print(f"K =\n{camera_matrix}")
    print(f"D = {distortion.ravel()}")
    print(f"Ошибка репроекции (RMS): {rms:.6f} px")
    print(f"Результаты: {args.output_dir}")


if __name__ == "__main__":
    main()
