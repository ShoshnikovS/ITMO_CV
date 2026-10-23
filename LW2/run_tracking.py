"""Запуск трекинга:
python run_tracking.py вход.avi выход.mp4 [--detector sift|orb] [--no-flow] [--label Текст]
Первый кадр видео - эталон объекта. Рядом с выходным видео пишется csv с результатами по кадрам.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

from tracker import ObjectTracker, draw_box


def process(video_in, video_out, detector="sift", use_flow=True, label="Mona Lisa", csv_path=None):
    cap = cv2.VideoCapture(str(video_in))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    ok, ref = cap.read()
    assert ok, "не удалось прочитать первый кадр"
    tracker = ObjectTracker(ref, detector, use_flow)
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rows, frame, idx = [], ref, 0
    while ok:
        t = time.perf_counter()
        corners, points, how = tracker.track(frame)
        dt = time.perf_counter() - t
        rows.append([idx, int(corners is not None), points, how or "", round(dt * 1000, 1)]
                    + (corners.flatten().round(1).tolist() if corners is not None else [""] * 8))
        if corners is not None:
            draw_box(frame, corners, label)
        writer.write(frame)
        idx += 1
        ok, frame = cap.read()
    cap.release()
    writer.release()
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["frame", "found", "points", "method", "ms",
                         "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"])
            wr.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video_in")
    ap.add_argument("video_out")
    ap.add_argument("--detector", default="sift", choices=["sift", "orb"])
    ap.add_argument("--no-flow", action="store_true", help="только детекция на каждом кадре")
    ap.add_argument("--label", default="Mona Lisa")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rows = process(a.video_in, a.video_out, a.detector, not a.no_flow, a.label,
                   Path(a.video_out).with_suffix(".csv"))
    print(f"объект найден на {sum(r[1] for r in rows)} из {len(rows)} кадров")
