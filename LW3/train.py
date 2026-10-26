"""ЛР3.1. Классификация цветов (5 классов): дообучение ResNet18 и обучение маленькой CNN с нуля.
Запуск: python train.py
Данные: data/flowers.parquet (датасет Falah/Flowers_Classification = tf_flowers, 3670 фото).
"""
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image

ROOT = Path(__file__).parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
SIZE, PAD = 160, 16  # изображения хранятся 176x176, при обучении вырезается случайный кроп 160x160
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SEED = 0


def load_dataset():
    """Читает parquet, приводит все фото к 176x176 (uint8) и делит на train/val 80/20 (стратифицировано)."""
    df = pd.read_parquet(ROOT / "data" / "flowers.parquet")
    imgs = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB").resize((SIZE + PAD, SIZE + PAD)))
                     for d in df["image"]])
    labels = df["label"].to_numpy()
    rng = np.random.default_rng(SEED)
    train_idx, val_idx = [], []
    for c in range(len(CLASSES)):
        idx = rng.permutation(np.where(labels == c)[0])
        k = int(0.8 * len(idx))
        train_idx += idx[:k].tolist()
        val_idx += idx[k:].tolist()
    to_t = lambda ix: torch.from_numpy(imgs[ix]).permute(0, 3, 1, 2).contiguous()
    return (to_t(train_idx), torch.from_numpy(labels[train_idx]).long(),
            to_t(val_idx), torch.from_numpy(labels[val_idx]).long())


def normalize(x):
    return (x.float() / 255 - MEAN) / STD


def augment(x):
    """Случайный кроп 160x160 и случайное горизонтальное отражение (на уровне батча по одному смещению на картинку)."""
    n = x.shape[0]
    out = torch.empty(n, 3, SIZE, SIZE, dtype=x.dtype)
    for i in range(n):
        dy, dx = np.random.randint(0, PAD + 1, 2)
        im = x[i, :, dy:dy + SIZE, dx:dx + SIZE]
        out[i] = im.flip(2) if np.random.rand() < 0.5 else im
    return out


def center_crop(x):
    return x[:, :, PAD // 2:PAD // 2 + SIZE, PAD // 2:PAD // 2 + SIZE]


def make_resnet():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


def make_small_cnn():
    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    return nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, len(CLASSES)))


@torch.no_grad()
def evaluate(model, x, y, bs=128):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss, correct = 0.0, 0
    for i in range(0, len(x), bs):
        out = model(normalize(center_crop(x[i:i + bs])))
        loss += loss_fn(out, y[i:i + bs]).item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return loss / len(x), correct / len(x)


def fit(name, model, data, epochs, lr, bs=32):
    xtr, ytr, xva, yva = data
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(xtr) + bs - 1) // bs))
    loss_fn = nn.CrossEntropyLoss()
    hist = []
    for ep in range(epochs):
        model.train()
        t = time.time()
        perm = torch.randperm(len(xtr))
        tr_loss = tr_correct = 0
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            xb, yb = normalize(augment(xtr[idx])), ytr[idx]
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tr_loss += loss.item() * len(idx)
            tr_correct += (out.argmax(1) == yb).sum().item()
        va_loss, va_acc = evaluate(model, xva, yva)
        hist.append(dict(epoch=ep + 1, train_loss=tr_loss / len(xtr), train_acc=tr_correct / len(xtr),
                         val_loss=va_loss, val_acc=va_acc, sec=time.time() - t))
        print(name, hist[-1], flush=True)
    return hist


def main(skip_resnet=False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(8)
    data = load_dataset()
    print("train", tuple(data[0].shape), "val", tuple(data[2].shape), flush=True)
    results = {}
    model = make_resnet()
    if skip_resnet and (OUT / "resnet18.pt").exists():  # уже обученная модель: загружаем историю и веса
        model.load_state_dict(torch.load(OUT / "resnet18.pt"))
        results["resnet18_finetune"] = json.load(open(OUT / "history_resnet18.json"))
    else:
        results["resnet18_finetune"] = fit("resnet18", model, data, epochs=6, lr=1e-3)
        torch.save(model.state_dict(), OUT / "resnet18.pt")
        json.dump(results["resnet18_finetune"], open(OUT / "history_resnet18.json", "w"), indent=1)
    small = make_small_cnn()
    results["small_cnn_scratch"] = fit("small_cnn", small, data, epochs=15, lr=3e-3)
    torch.save(small.state_dict(), OUT / "small_cnn.pt")
    json.dump(results, open(OUT / "history.json", "w"), indent=1)
    # индексы валидации нужны для confusion matrix - сохраним предсказания
    for name, m in (("resnet18", model), ("small_cnn", small)):
        m.eval()
        with torch.no_grad():
            pred = torch.cat([m(normalize(center_crop(data[2][i:i + 128]))).argmax(1)
                              for i in range(0, len(data[2]), 128)])
        np.save(OUT / f"val_pred_{name}.npy", pred.numpy())
    np.save(OUT / "val_true.npy", data[3].numpy())


if __name__ == "__main__":
    import sys
    main(skip_resnet="--skip-resnet" in sys.argv)
