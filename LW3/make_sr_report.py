"""ЛР3.2. По results_sr/metrics.csv и картинкам строит сводные таблицы и сравнительные рисунки."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).parent / "results_sr"
DEGS = ["bicubic", "unknown", "realesrgan"]
MASKS = ["zeros", "ones", "half"]


def tile(path, title, size=256):
    im = Image.open(path).convert("RGB").resize((size, size), Image.NEAREST)
    canvas = Image.new("RGB", (size, size + 18), "white")
    canvas.paste(im, (0, 18))
    ImageDraw.Draw(canvas).text((4, 3), title, fill="black")
    return canvas


def main():
    df = pd.read_csv(OUT / "metrics.csv", dtype={"image": str})
    lines = []
    mean = df.groupby(["degradation", "mask"])[["psnr", "lpips", "clip_iqa"]].mean().reindex(
        [(d, m) for d in DEGS for m in MASKS])
    lines += ["Средние метрики по 4 изображениям (PSNR выше - лучше, LPIPS ниже - лучше, CLIP-IQA выше - лучше):",
              "", "| Деградация | Маска | PSNR, дБ | LPIPS | CLIP-IQA |", "|---|---|---|---|---|"]
    names = {"zeros": "все 0 (bicubic)", "ones": "все 1", "half": "левая половина 1"}
    for (d, m), r in mean.iterrows():
        lines.append(f"| {d} | {names[m]} | {r.psnr:.2f} | {r.lpips:.3f} | {r.clip_iqa:.3f} |")
    half = df[df["mask"] == "half"].groupby("degradation")[["psnr_left", "psnr_right"]].mean().reindex(DEGS)
    zeros = df[df["mask"] == "zeros"].groupby("degradation")[["psnr_left", "psnr_right"]].mean().reindex(DEGS)
    ones = df[df["mask"] == "ones"].groupby("degradation")[["psnr_left", "psnr_right"]].mean().reindex(DEGS)
    lines += ["", "PSNR по половинам для комбинированной маски (левая половина улучшается, правая - нет), дБ:", "",
              "| Деградация | левая: bicubic | левая: SR (маска 1) | левая: комб. маска | правая: bicubic | правая: комб. маска |",
              "|---|---|---|---|---|---|"]
    for d in DEGS:
        lines.append(f"| {d} | {zeros.loc[d, 'psnr_left']:.2f} | {ones.loc[d, 'psnr_left']:.2f} | "
                     f"{half.loc[d, 'psnr_left']:.2f} | {zeros.loc[d, 'psnr_right']:.2f} | {half.loc[d, 'psnr_right']:.2f} |")
    lines += ["", f"Среднее время одного прогона диффузии на CPU: {df[df['mask'] != 'zeros']['sec'].mean():.0f} с"]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    for name in sorted(df["image"].unique()):
        rows = []
        for d in DEGS:
            tiles = [tile(OUT / f"{name}_{d}_zeros.png", f"mask 0 = bicubic ({d})"),
                     tile(OUT / f"{name}_{d}_ones.png", "mask 1"), tile(OUT / f"{name}_{d}_half.png", "left half = 1"),
                      tile(OUT / f"{name}_gt.png", "ground truth")]
            rows.append(np.hstack([np.asarray(t) for t in tiles]))
        Image.fromarray(np.vstack(rows)).save(OUT / f"compare_{name}.png")


if __name__ == "__main__":
    main()
