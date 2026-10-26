"""ЛР3.2. Эксперименты: 3 изображения DIV2K x 3 деградации x 3 маски (всё 1 / всё 0 / комбинированная).
Считает PSNR, LPIPS, CLIP-IQA, сохраняет картинки в results_sr/. Запуск: python run_sr_experiments.py"""
import csv
import io
import sys
import time
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
from PIL import Image
from torchmetrics.multimodal import CLIPImageQualityAssessment

from sr_inpaint import bicubic_up, load_pipe, sr_inpaint

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).parent
OUT = ROOT / "results_sr"
OUT.mkdir(exist_ok=True)
HR = 256
CROPS = {"0001": (300, 400), "0002": (1400, 950), "0003": (1100, 200), "0004": (1100, 560)}
STEPS = 15


def jpeg(arr, q):
    return cv2.imdecode(cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)


def noise(arr, sigma, rng):
    return np.clip(arr.astype(np.float32) + rng.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)


def degrade(hr_rgb, kind, seed=0):
    """hr_rgb: uint8 (256,256,3) RGB -> LR uint8 (64,64,3) RGB."""
    rng = np.random.default_rng(seed)
    bgr = hr_rgb[:, :, ::-1].copy()
    if kind == "bicubic":
        lr = cv2.resize(bgr, (HR // 4, HR // 4), interpolation=cv2.INTER_CUBIC)
    elif kind == "unknown":  # размытие + шум + JPEG
        x = cv2.GaussianBlur(bgr, (0, 0), 2.0)
        x = cv2.resize(x, (HR // 4, HR // 4), interpolation=cv2.INTER_CUBIC)
        lr = jpeg(noise(x, 6, rng), 60)
    elif kind == "realesrgan":  # упрощённая имитация деградаций Real-ESRGAN: два прохода размытие/ресайз/шум/JPEG
        x = cv2.GaussianBlur(bgr, (0, 0), 1.2)
        x = cv2.resize(x, (HR * 3 // 4, HR * 3 // 4), interpolation=cv2.INTER_LINEAR)
        x = jpeg(noise(x, 4, rng), 70)
        x = cv2.GaussianBlur(x, (0, 0), 1.0)
        x = cv2.resize(x, (HR // 4, HR // 4), interpolation=cv2.INTER_CUBIC)
        lr = jpeg(noise(x, 3, rng), 65)
    else:
        raise ValueError(kind)
    return lr[:, :, ::-1].copy()


def make_masks():
    half = np.zeros((HR, HR), np.float32)
    half[:, :HR // 2] = 1
    return {"ones": np.ones((HR, HR), np.float32), "zeros": np.zeros((HR, HR), np.float32), "half": half}


def psnr(a, b, region=None):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if region is not None:
        d = d[region]
    return float(10 * np.log10(255 ** 2 / d.mean()))


def to_t(a):  # uint8 HWC -> [-1,1] NCHW
    return torch.from_numpy(a).permute(2, 0, 1)[None].float() / 127.5 - 1


def main():
    pipe = load_pipe()
    lp = lpips.LPIPS(net="alex", verbose=False)
    clip_iqa = CLIPImageQualityAssessment(model_name_or_path="clip_iqa", data_range=1.0)
    masks = make_masks()
    rows = []
    for name, (x0, y0) in CROPS.items():
        hr = np.asarray(Image.open(ROOT / "data" / "div2k" / f"{name}.png").convert("RGB"))[y0:y0 + HR, x0:x0 + HR]
        Image.fromarray(hr).save(OUT / f"{name}_gt.png")
        for kind in ("bicubic", "unknown", "realesrgan"):
            lr = degrade(hr, kind)
            lr_img = Image.fromarray(lr)
            lr_img.save(OUT / f"{name}_{kind}_lr.png")
            for mname, mask in masks.items():
                path = OUT / f"{name}_{kind}_{mname}.png"
                t = time.time()
                if path.exists():
                    out = np.asarray(Image.open(path).convert("RGB"))
                else:
                    out = np.asarray(sr_inpaint(pipe, lr_img, mask, steps=STEPS, seed=0))
                    Image.fromarray(out).save(path)
                sec = time.time() - t
                with torch.no_grad():
                    lpips_v = lp(to_t(out), to_t(hr)).item()
                    iqa = clip_iqa(torch.from_numpy(out).permute(2, 0, 1)[None].float() / 255).item()
                row = dict(image=name, degradation=kind, mask=mname, psnr=psnr(out, hr), lpips=lpips_v,
                           clip_iqa=iqa, psnr_left=psnr(out, hr, (slice(None), slice(0, HR // 2))),
                           psnr_right=psnr(out, hr, (slice(None), slice(HR // 2, HR))), sec=sec)
                rows.append(row)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
                with open(OUT / "metrics.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    w.writeheader()
                    w.writerows(rows)


if __name__ == "__main__":
    main()
