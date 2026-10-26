"""ЛР3.1. Оценка обученных моделей: кривые обучения, метрики на val и на собственном test, confusion matrix, ошибки.
Запуск (после train.py): python evaluate.py"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import CLASSES, OUT, PAD, ROOT, SIZE, center_crop, make_resnet, make_small_cnn, normalize

sys.stdout.reconfigure(encoding="utf-8")


def load_test():
    files, imgs, labels = [], [], []
    for c, cls in enumerate(CLASSES):
        for f in sorted((ROOT / "test_images" / cls).glob("*.jpg")):
            files.append(f"{cls}/{f.name}")
            imgs.append(np.asarray(Image.open(f).convert("RGB").resize((SIZE + PAD, SIZE + PAD))))
            labels.append(c)
    return files, torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous(), np.array(labels)


@torch.no_grad()
def predict(model, x):
    model.eval()
    probs = torch.cat([torch.softmax(model(normalize(center_crop(x[i:i + 64]))), 1) for i in range(0, len(x), 64)])
    return probs.numpy()


def plot_confusion(ax, y, p, title):
    cm = confusion_matrix(y, p, labels=range(len(CLASSES)))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("предсказано")
    ax.set_ylabel("истина")
    ax.set_title(title)


def main():
    hist = json.load(open(OUT / "history.json"))
    lines = []

    # 1. кривые обучения
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in hist.items():
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_loss"] for r in h], "--", label=f"{name} train")
        axes[0].plot(ep, [r["val_loss"] for r in h], "-", label=f"{name} val")
        axes[1].plot(ep, [r["train_acc"] for r in h], "--", label=f"{name} train")
        axes[1].plot(ep, [r["val_acc"] for r in h], "-", label=f"{name} val")
    axes[0].set_title("Loss (кросс-энтропия)")
    axes[1].set_title("Accuracy")
    for a in axes:
        a.set_xlabel("эпоха")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "curves.png", dpi=110)
    plt.close()

    # 2. метрики на val
    yv = np.load(OUT / "val_true.npy")
    models = {"resnet18": make_resnet(), "small_cnn": make_small_cnn()}
    for n, m in models.items():
        m.load_state_dict(torch.load(OUT / f"{n if n == 'resnet18' else 'small_cnn'}.pt"))
    files, xt, yt = load_test()
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    summary = {}
    test_probs = {}
    for r, (n, m) in enumerate(models.items()):
        pv = np.load(OUT / f"val_pred_{n}.npy")
        probs = predict(m, xt)
        test_probs[n] = probs
        pt = probs.argmax(1)
        summary[n] = dict(val_acc=accuracy_score(yv, pv), val_f1=f1_score(yv, pv, average="macro"),
                          test_acc=accuracy_score(yt, pt), test_f1=f1_score(yt, pt, average="macro"))
        plot_confusion(axes[r, 0], yv, pv, f"{n}: val ({len(yv)} фото)")
        plot_confusion(axes[r, 1], yt, pt, f"{n}: test ({len(yt)} фото)")
        for split, y, p in (("val", yv, pv), ("test", yt, pt)):
            pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(len(CLASSES)), zero_division=0)
            lines.append(f"\n{n} / {split}: accuracy={accuracy_score(y, p):.3f}, macro-F1={f1_score(y, p, average='macro'):.3f}")
            lines.append("| класс | precision | recall | F1 | число |\n|---|---|---|---|---|")
            lines += [f"| {c} | {pr[i]:.2f} | {rc[i]:.2f} | {f1[i]:.2f} | {sup[i]} |" for i, c in enumerate(CLASSES)]
    plt.tight_layout()
    plt.savefig(OUT / "confusion.png", dpi=110)
    plt.close()

    # 3. ошибки resnet18 на test
    probs = test_probs["resnet18"]
    pt = probs.argmax(1)
    wrong = np.where(pt != yt)[0]
    with open(OUT / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "true", "resnet18", "conf", "ok"])
        for i, fl in enumerate(files):
            w.writerow([fl, CLASSES[yt[i]], CLASSES[pt[i]], f"{probs[i].max():.2f}", int(pt[i] == yt[i])])
    if len(wrong):
        cols = min(4, len(wrong))
        rows_n = (len(wrong) + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(3.3 * cols, 3.1 * rows_n))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
            ax.imshow(xt[i].permute(1, 2, 0).numpy())
            ax.set_title(f"{files[i].split('/')[1][:-4]}\nистина: {CLASSES[yt[i]]}\nответ: {CLASSES[pt[i]]} ({probs[i].max():.2f})",
                         fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT / "test_errors.png", dpi=110)
        plt.close()

    lines.insert(0, "Сводка: " + json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in summary.items()},
                                           ensure_ascii=False))
    lines.append(f"\nОшибок resnet18 на test: {len(wrong)} из {len(yt)}")
    (OUT / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
