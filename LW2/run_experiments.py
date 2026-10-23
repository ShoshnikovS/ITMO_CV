"""Эксперименты ЛР2: прогон всех видео и конфигураций, сводная таблица, графики, контрольные кадры."""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_tracking import process

ROOT = Path(__file__).parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

VIDEOS = ["mona-lisa", "mona-lisa-blur", "mona-lisa-blur-extra-credit"]
if (ROOT / "videos" / "my_video.mp4").exists():
    VIDEOS.append("my_video")

CONFIGS = {  # имя: (детектор, использовать поток)
    "sift_flow": ("sift", True),
    "sift_detect": ("sift", False),
    "orb_flow": ("orb", True),
    "orb_detect": ("orb", False),
}


def find_video(name):
    for ext in (".avi", ".mp4"):
        if (ROOT / "videos" / (name + ext)).exists():
            return ROOT / "videos" / (name + ext)


def contact_sheet(video, path, count=12, cols=4):
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ims = []
    for i in np.linspace(0, n - 1, count).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        f = cv2.resize(f, (480, 270))
        cv2.putText(f, f"frame {i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        ims.append(f)
    rows = [np.hstack(ims[j:j + cols]) for j in range(0, count, cols)]
    Path(path).write_bytes(cv2.imencode(".png", np.vstack(rows))[1].tobytes())


def sample_vs_mine(path, frames=(40, 120, 200, 440)):
    """Сверху - образец (simple_track.avi, рамка нарисована авторами задания), снизу - наш результат."""
    a = cv2.VideoCapture(str(ROOT / "videos" / "simple_track.avi"))
    b = cv2.VideoCapture(str(OUT / "mona-lisa_sift_flow.mp4"))
    top, bottom = [], []
    for i in frames:
        for cap, lst in ((a, top), (b, bottom)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            lst.append(cv2.resize(cap.read()[1], (480, 270)))
    Path(path).write_bytes(cv2.imencode(".png", np.vstack([np.hstack(top), np.hstack(bottom)]))[1].tobytes())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    table = ["| Видео | Конфигурация | Найден, кадров | Доля | Среднее время, мс/кадр |", "|---|---|---|---|---|"]
    timelines = {}
    for v in VIDEOS:
        for cfg, (det, flow) in CONFIGS.items():
            main_cfg = cfg == "sift_flow"
            out = OUT / f"{v}_{cfg}.mp4"
            rows = process(find_video(v), out, det, flow, "Mona Lisa" if v != "my_video" else "Object",
                           OUT / f"{v}_{cfg}.csv")
            found = np.array([r[1] for r in rows])
            ms = np.mean([r[4] for r in rows])
            timelines[(v, cfg)] = found
            table.append(f"| {v} | {cfg} | {found.sum()} / {len(found)} | {found.mean() * 100:.1f}% | {ms:.1f} |")
            print(table[-1], flush=True)
            if main_cfg:
                contact_sheet(out, OUT / f"{v}_sheet.png")
            else:
                out.unlink()  # в репозитории оставляем видео только основной конфигурации
    sample_vs_mine(OUT / "sample_vs_mine.png")
    (OUT / "summary.md").write_text("\n".join(table), encoding="utf-8")

    fig, axes = plt.subplots(len(VIDEOS), 1, figsize=(10, 2.2 * len(VIDEOS)), sharex=True)
    for ax, v in zip(np.atleast_1d(axes), VIDEOS):
        for k, cfg in enumerate(CONFIGS):
            f = timelines[(v, cfg)]
            ax.fill_between(range(len(f)), k, k + 0.8 * f, step="mid", label=cfg)
        ax.set_yticks([k + 0.4 for k in range(len(CONFIGS))])
        ax.set_yticklabels(list(CONFIGS), fontsize=7)
        ax.set_title(v, fontsize=9)
    axes[-1].set_xlabel("номер кадра (закрашено - объект найден)")
    plt.tight_layout()
    plt.savefig(OUT / "timeline.png", dpi=110)


if __name__ == "__main__":
    main()
