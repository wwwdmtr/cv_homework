"""Сравнение ключевых точек двух видео одного объекта: SuperPoint против SIFT.

Сеть и сопоставление берутся из репозитория magicleap/SuperPointPretrainedNetwork
(SuperPointFrontend и двусторонний поиск ближайшего соседа PointTracker.nn_match_two_way).
Видео идут синхронно: кадр i первого сопоставляется с кадром i второго
(--offset задаёт сдвиг второго видео в кадрах). Выбросы
отсекаются RANSAC-гомографией.

Основа — Module_4b/superpoint_video_match.py из репозитория курса:
https://github.com/sadekovlar/cv_book/blob/master/Module_4b/superpoint_video_match.py
Добавлены локальные пути, проверки входов и сохранение кадров со статистикой.

Запуск из корня репозитория:
    python lab_03/setup_superpoint.py
    python lab_03/prepare_videos.py
    python lab_03/compare_videos.py
Результат: lab_03/results/video/
"""
import argparse
import csv
import os
import sys
import json

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, 'superpoint')
sys.path.insert(0, REPO)
from demo_superpoint import PointTracker, SuperPointFrontend  # noqa: E402


# ---------------------------------------------------------------- детекторы
class SuperPoint:
    name = 'SuperPoint'

    def __init__(self, weights, conf_thresh, nn_thresh, cuda):
        self.fe = SuperPointFrontend(weights_path=weights, nms_dist=4,
                                     conf_thresh=conf_thresh, nn_thresh=nn_thresh,
                                     cuda=cuda)
        self.matcher = PointTracker(max_length=2, nn_thresh=nn_thresh)
        self.nn_thresh = nn_thresh

    def detect(self, gray):
        with torch.inference_mode():
            pts, desc, _ = self.fe.run(gray.astype(np.float32) / 255.0)
        if desc is None:
            return np.zeros((0, 2), np.float32), np.zeros((256, 0), np.float32)
        return pts[:2].T.astype(np.float32), desc

    def match(self, d1, d2):
        m = self.matcher.nn_match_two_way(d1, d2, self.nn_thresh)
        return m[0].astype(int), m[1].astype(int)


class Sift:
    name = 'SIFT'

    def __init__(self, max_kp):
        self.sift = cv2.SIFT_create(nfeatures=max_kp)
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def detect(self, gray):
        kps, des = self.sift.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kps]).reshape(-1, 2)
        return pts, des

    def match(self, d1, d2):
        if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
            return np.zeros(0, int), np.zeros(0, int)
        ms = self.bf.match(d1, d2)
        return np.array([m.queryIdx for m in ms], int), np.array([m.trainIdx for m in ms], int)


# ---------------------------------------------------------------- сопоставление кадров
def compare(method, g1, g2):
    p1, d1 = method.detect(g1)
    p2, d2 = method.detect(g2)
    i1, i2 = method.match(d1, d2)
    inl = np.zeros(len(i1), bool)
    if len(i1) >= 4:
        _, mask = cv2.findHomography(p1[i1], p2[i2], cv2.RANSAC, 3.0)
        if mask is not None:
            inl = mask.ravel().astype(bool)
    return p1, p2, i1, i2, inl


def draw(img1, img2, p1, p2, i1, i2, inl, title):
    h, w = img1.shape[:2]
    vis = np.hstack([cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR),
                     cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)])
    for x, y in p1:
        cv2.circle(vis, (int(x), int(y)), 2, (255, 160, 0), -1)
    for x, y in p2:
        cv2.circle(vis, (int(x) + w, int(y)), 2, (255, 160, 0), -1)
    for a, b, ok in zip(i1, i2, inl):
        if not ok:
            continue
        pa = (int(p1[a][0]), int(p1[a][1]))
        pb = (int(p2[b][0]) + w, int(p2[b][1]))
        cv2.line(vis, pa, pb, (0, 220, 0), 1, cv2.LINE_AA)
    # cv2.putText не рисует кириллицу, поэтому подписи латиницей
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(vis, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    stats = f'kp {len(p1)}/{len(p2)}  matches {len(i1)}  inliers {int(inl.sum())}'
    cv2.putText(vis, stats, (200, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video1', default=os.path.join(HERE, 'videos', '01.avi'))
    ap.add_argument('--video2', default=os.path.join(HERE, 'videos', '02.avi'))
    ap.add_argument('--offset', type=int, default=0,
                    help='кадр i первого видео сравнивается с кадром i+offset второго')
    ap.add_argument('--step', type=int, default=6, help='брать каждый step-й кадр')
    ap.add_argument('--width', type=int, default=640, help='ширина кадра для обработки')
    ap.add_argument('--weights', default=os.path.join(REPO, 'superpoint_v1.pth'))
    ap.add_argument('--conf-thresh', type=float, default=0.015)
    ap.add_argument('--nn-thresh', type=float, default=0.7)
    ap.add_argument('--max-kp', type=int, default=2000, help='лимит точек SIFT')
    ap.add_argument('--cuda', action='store_true')
    ap.add_argument('--out', default=None, help='по умолчанию lab_03/results/video')
    ap.add_argument('--show', action='store_true')
    args = ap.parse_args()
    if args.step < 1 or args.offset < 0 or args.width < 8 or args.width % 8:
        ap.error('Нужны положительный шаг, неотрицательный offset и ширина, кратная 8')
    cv2.setRNGSeed(0)
    torch.set_num_threads(4)

    out_dir = args.out or os.path.join(HERE, 'results', 'video')
    os.makedirs(out_dir, exist_ok=True)

    def read_all(path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f'Не удалось открыть {path}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            s = args.width / g.shape[1]
            frames.append(cv2.resize(g, (args.width, int(round(g.shape[0] * s / 8)) * 8),
                                     interpolation=cv2.INTER_AREA))
        cap.release()
        if not frames or fps <= 0:
            raise ValueError(f'Нет кадров или частоты видео в {path}')
        return frames, fps

    (f1, fps1), (f2, fps2) = read_all(args.video1), read_all(args.video2)
    if abs(fps1 - fps2) > 0.01 or f1[0].shape != f2[0].shape:
        ap.error('У двух видео должны совпадать частота и размер кадра')
    n = min(len(f1), len(f2) - args.offset)
    assert n > 0, 'offset больше длины видео'
    print(f'Кадров: {len(f1)} и {len(f2)}, сравниваем {len(range(0, n, args.step))} пар, '
          f'offset={args.offset}')

    methods = [SuperPoint(args.weights, args.conf_thresh, args.nn_thresh, args.cuda),
               Sift(args.max_kp)]

    writer = None
    rows = []
    selected = list(range(0, n, args.step))
    snapshots = {selected[0]: 'first', selected[len(selected) // 2]: 'middle', selected[-1]: 'last'}
    for i in range(0, n, args.step):
        g1, g2 = f1[i], f2[i + args.offset]
        panels, row = [], {'frame1': i, 'frame2': i + args.offset}
        for m in methods:
            p1, p2, i1, i2, inl = compare(m, g1, g2)
            panels.append(draw(g1, g2, p1, p2, i1, i2, inl, m.name))
            row[f'{m.name}_kp1'], row[f'{m.name}_kp2'] = len(p1), len(p2)
            row[f'{m.name}_matches'], row[f'{m.name}_inliers'] = len(i1), int(inl.sum())
            row[f'{m.name}_ratio'] = round(inl.sum() / max(len(i1), 1), 3)
        rows.append(row)
        frame = np.vstack(panels)
        if writer is None:
            writer = cv2.VideoWriter(os.path.join(out_dir, 'superpoint_vs_sift.mp4'),
                                     cv2.VideoWriter_fourcc(*'avc1'), fps1 / args.step,
                                     (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                writer.release()
                writer = cv2.VideoWriter(os.path.join(out_dir, 'superpoint_vs_sift.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'), fps1 / args.step,
                                         (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                raise OSError('Не удалось открыть выходное видео')
        if i in snapshots:
            filename = f'superpoint_vs_sift_{snapshots[i]}.jpg'
            if not cv2.imwrite(os.path.join(out_dir, filename), frame):
                raise OSError(f'Не удалось сохранить {filename}')
        writer.write(frame)
        print(f"[{i:3d}->{i + args.offset:3d}] "
              f"SuperPoint {row['SuperPoint_inliers']:4d}/{row['SuperPoint_matches']:4d}  "
              f"SIFT {row['SIFT_inliers']:4d}/{row['SIFT_matches']:4d}")
        if args.show:
            cv2.imshow('SuperPoint vs SIFT', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    with open(os.path.join(out_dir, 'superpoint_vs_sift.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print('\nСреднее по кадрам:')
    summary = {
        'source1_frames': len(f1), 'source2_frames': len(f2),
        'fps': fps1, 'processed_pairs': len(rows), 'step': args.step,
        'offset': args.offset, 'processing_width': args.width,
        'processing_height': f1[0].shape[0], 'methods': {},
    }
    for m in methods:
        k = m.name
        kp = np.mean([r[f'{k}_kp1'] for r in rows])
        mt = np.mean([r[f'{k}_matches'] for r in rows])
        il = np.mean([r[f'{k}_inliers'] for r in rows])
        print(f'  {k:10s} точек {kp:7.1f}  совпадений {mt:7.1f}  инлайеров {il:7.1f}  '
              f'доля инлайеров {il / max(mt, 1):.2f}')
        summary['methods'][k] = {
            'kp1': float(kp), 'kp2': float(np.mean([r[f'{k}_kp2'] for r in rows])),
            'matches': float(mt), 'inliers': float(il), 'ratio': float(il / max(mt, 1)),
        }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f'Результат -> {out_dir}/superpoint_vs_sift.mp4, .csv, _first.jpg')


if __name__ == '__main__':
    main()
