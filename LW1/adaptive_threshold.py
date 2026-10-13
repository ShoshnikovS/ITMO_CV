"""Адаптивная бинаризация (mean) в трёх реализациях.

Правило для каждого пикселя (как в cv2.adaptiveThreshold, THRESH_BINARY):
    dst(x, y) = 255, если src(x, y) > T(x, y), иначе 0
    T(x, y)   = mean(окно blockSize x blockSize вокруг (x, y)) - C
"""
import cv2
import numpy as np


def threshold_opencv(gray, block_size=11, c=2):
    """Вариант 1: встроенная функция OpenCV."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, block_size, c)


def _pad_replicate(img, r):
    """Дополняет изображение (список строк) на r пикселей, повторяя крайние значения."""
    h = len(img)
    rows = [[row[0]] * r + row + [row[-1]] * r for row in img]
    return [rows[0][:] for _ in range(r)] + rows + [rows[-1][:] for _ in range(r)]


def threshold_native_naive(gray, block_size=11, c=2):
    """Вариант 2а: чистый Python, окно суммируется напрямую. Сложность O(W*H*k^2)."""
    r = block_size // 2
    area = block_size * block_size
    img = _pad_replicate(gray.tolist(), r)
    h, w = gray.shape
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            s = 0
            for yy in range(y, y + block_size):
                s += sum(img[yy][x:x + block_size])
            mean = (2 * s + area) // (2 * area)  # округление, как у OpenCV
            row.append(255 if img[y + r][x + r] > mean - c else 0)
        out.append(row)
    return np.array(out, dtype=np.uint8)


def threshold_native_integral(gray, block_size=11, c=2):
    """Вариант 2б: чистый Python + интегральное изображение. Сложность O(W*H), не зависит от k."""
    r = block_size // 2
    area = block_size * block_size
    img = _pad_replicate(gray.tolist(), r)
    ph, pw = len(img), len(img[0])
    # integral[y][x] = сумма img[0:y][0:x]
    integral = [[0] * (pw + 1)]
    for y in range(ph):
        acc, prev, row = 0, integral[-1], [0]
        for x in range(pw):
            acc += img[y][x]
            row.append(prev[x + 1] + acc)
        integral.append(row)
    h, w = gray.shape
    out = []
    for y in range(h):
        top, bot = integral[y], integral[y + block_size]
        src_row = img[y + r]
        row = []
        for x in range(w):
            s = bot[x + block_size] - bot[x] - top[x + block_size] + top[x]
            mean = (2 * s + area) // (2 * area)
            row.append(255 if src_row[x + r] > mean - c else 0)
        out.append(row)
    return np.array(out, dtype=np.uint8)
