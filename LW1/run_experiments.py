"""Эксперименты ЛР1: проверка корректности, замеры времени, картинки и графики для отчёта."""
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptive_threshold import threshold_native_integral, threshold_native_naive, threshold_opencv

ROOT = Path(__file__).parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def read_gray(video, index):
    cap = cv2.VideoCapture(str(ROOT / "videos" / video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"не удалось прочитать кадр {index} из {video}"
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def timeit(fn, *args, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


def save_row(name, images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 3.2))
    for ax, im, t in zip(np.atleast_1d(axes), images, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=110)
    plt.close()


def main():
    log = []
    frames = {"mona-lisa": read_gray("mona-lisa.avi", 200),
              "simple_track": read_gray("simple_track.avi", 100)}

    # 1. Корректность: сравнение с OpenCV на полном кадре
    log.append("== Проверка корректности (число отличающихся пикселей от OpenCV) ==")
    for name, g in frames.items():
        for k, c in [(3, 2), (11, 2), (21, 5), (11, -5)]:
            ref = threshold_opencv(g, k, c)
            d1 = int((ref != threshold_native_naive(g, k, c)).sum())
            d2 = int((ref != threshold_native_integral(g, k, c)).sum())
            log.append(f"{name:13s} k={k:2d} C={c:3d}: naive={d1}, integral={d2}")

    # 2. Визуальные результаты
    for name, g in frames.items():
        cv2.imwrite(str(OUT / f"{name}_gray.png"), g)
        glob = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        save_row(f"{name}_compare.png",
                 [g, glob, threshold_opencv(g, 11, 2), threshold_native_integral(g, 11, 2)],
                 ["Исходный (gray)", "Глобальный порог (Оцу)", "OpenCV adaptive mean",
                  "Нативный (integral)"])
        save_row(f"{name}_blocksize.png",
                 [threshold_native_integral(g, k, 5) for k in (3, 11, 31, 101)],
                 [f"blockSize={k}, C=5" for k in (3, 11, 31, 101)])
        save_row(f"{name}_C.png",
                 [threshold_native_integral(g, 31, c) for c in (-10, 0, 5, 15)],
                 [f"blockSize=31, C={c}" for c in (-10, 0, 5, 15)])

    # 3. Время от размера окна (кадр 640x360 уменьшен до 160x90 для naive)
    g = frames["mona-lisa"]
    small = cv2.resize(g, (160, 90))
    ks = [3, 5, 9, 15, 21, 31, 41]
    t_cv, t_naive, t_int = [], [], []
    log.append("\n== Время от blockSize (кадр 160x90), мс ==")
    for k in ks:
        t_cv.append(timeit(threshold_opencv, small, k, 2) * 1e3)
        t_naive.append(timeit(threshold_native_naive, small, k, 2) * 1e3)
        t_int.append(timeit(threshold_native_integral, small, k, 2) * 1e3)
        log.append(f"k={k:2d}: opencv={t_cv[-1]:.4f}  naive={t_naive[-1]:.1f}  integral={t_int[-1]:.1f}")
    plt.figure(figsize=(6, 4))
    plt.plot(ks, t_cv, "o-", label="OpenCV")
    plt.plot(ks, t_naive, "s-", label="Python naive O(k²)")
    plt.plot(ks, t_int, "^-", label="Python integral O(1)")
    plt.yscale("log")
    plt.xlabel("blockSize")
    plt.ylabel("время, мс (лог. шкала)")
    plt.title("Время vs размер окна (160x90)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "time_vs_block.png", dpi=110)
    plt.close()

    # 4. Время от размера изображения (k=11)
    sizes = [(80, 45), (160, 90), (320, 180), (640, 360)]
    r_cv, r_naive, r_int = [], [], []
    log.append("\n== Время от размера изображения (blockSize=11), мс ==")
    for w, h in sizes:
        im = cv2.resize(g, (w, h))
        r_cv.append(timeit(threshold_opencv, im, 11, 2) * 1e3)
        r_naive.append(timeit(threshold_native_naive, im, 11, 2, repeats=1) * 1e3)
        r_int.append(timeit(threshold_native_integral, im, 11, 2) * 1e3)
        log.append(f"{w}x{h}: opencv={r_cv[-1]:.4f}  naive={r_naive[-1]:.1f}  integral={r_int[-1]:.1f}  "
                   f"(naive/opencv={r_naive[-1] / r_cv[-1]:.0f}x, integral/opencv={r_int[-1] / r_cv[-1]:.0f}x)")
    px = [w * h for w, h in sizes]
    plt.figure(figsize=(6, 4))
    plt.plot(px, r_cv, "o-", label="OpenCV")
    plt.plot(px, r_naive, "s-", label="Python naive")
    plt.plot(px, r_int, "^-", label="Python integral")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("число пикселей")
    plt.ylabel("время, мс (лог. шкала)")
    plt.title("Время vs размер изображения (k=11)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "time_vs_size.png", dpi=110)
    plt.close()

    # 5. Обработка видео: 60 кадров simple_track, среднее время кадра 640x360
    cap = cv2.VideoCapture(str(ROOT / "videos" / "simple_track.avi"))
    n, tc, ti = 20, 0.0, 0.0
    writer = cv2.VideoWriter(str(OUT / "simple_track_binary.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                             30, (640, 360), isColor=False)
    for _ in range(n):
        ok, f = cap.read()
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        t = time.perf_counter(); a = threshold_opencv(gray, 11, 2); tc += time.perf_counter() - t
        t = time.perf_counter(); b = threshold_native_integral(gray, 11, 2); ti += time.perf_counter() - t
        assert (a == b).all()
        writer.write(b)
    writer.release()
    log.append(f"\n== Видео simple_track ({n} кадров 640x360, k=11) ==")
    log.append(f"OpenCV: {tc / n * 1e3:.3f} мс/кадр ({n / tc:.0f} FPS); "
               f"Python integral: {ti / n * 1e3:.1f} мс/кадр ({n / ti:.1f} FPS); "
               f"кадры идентичны во всех {n} случаях")

    (OUT / "log.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
