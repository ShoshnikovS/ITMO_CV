"""Сбор тестовой выборки с Wikimedia Commons: кандидаты (по 20 на класс; затем лишние убираются вручную после просмотра), разметка - по названию категории Commons.
Сохраняет test_images/<класс>/*.jpg и test_images/sources.csv (страница файла, автор, лицензия)."""
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "test_images"
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "ITMO-CV-lab3/1.0 (educational; https://github.com/ShoshnikovS/ITMO_CV)"}

CATEGORIES = {  # класс датасета -> категории Commons
    "daisy": ["Bellis perennis", "Leucanthemum vulgare"],
    "dandelion": ["Taraxacum officinale"],
    "roses": ["Red roses", "Pink roses", "White roses", "Rosa 'Black Magic'", "search:rose flower blossom garden"],
    "sunflowers": ["Helianthus annuus", "Sunflowers"],
    "tulips": ["Tulipa gesneriana", "Tulipa (cultivars)", "Tulips"],
}
REMOVE = {  # ручная чистка после просмотра кандидатов: цветок не виден, чужой цветок, марки, гравюры и т.п.
    "daisy": [2, 7, 8, 10, 11, 12, 17],
    "dandelion": [1, 3, 9, 11, 13, 16],
    "roses": list(range(10, 20)),
    "sunflowers": [0, 1, 5, 8, 9, 10, 13, 14, 17, 18, 19],
    "tulips": [0, 1, 3, 12, 17, 18],
}
PER_CLASS = 20
MIN_SIDE = 400


def get(url):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=40).read()
        except Exception:
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(url)


def api(**params):
    return json.loads(get(API + "?" + urllib.parse.urlencode({"format": "json", **params})))


def category_files(cat):
    if cat.startswith("search:"):
        data = api(action="query", generator="search", gsrnamespace="6", gsrsearch=cat[7:], gsrlimit="60",
                   prop="imageinfo", iiprop="url|size|mime|extmetadata", iiurlwidth="500")
        return [p for p in data.get("query", {}).get("pages", {}).values() if "imageinfo" in p]
    data = api(action="query", generator="categorymembers", gcmtitle="Category:" + cat, gcmtype="file",
               gcmlimit="60", prop="imageinfo", iiprop="url|size|mime|extmetadata", iiurlwidth="500")
    return [p for p in data.get("query", {}).get("pages", {}).values() if "imageinfo" in p]


def main():
    rows = []
    for cls, cats in CATEGORIES.items():
        (OUT / cls).mkdir(parents=True, exist_ok=True)
        n = 0
        for cat in cats:
            pages = category_files(cat)
            pages.sort(key=lambda p: p["title"])
            for p in pages:
                if n >= PER_CLASS:
                    break
                info = p["imageinfo"][0]
                if re.search(r"stamp|marka|postage|painting|drawing|illustration|logo", p["title"], re.I):
                    continue
                if info["mime"] != "image/jpeg" or min(info["width"], info["height"]) < MIN_SIDE:
                    continue
                meta = info.get("extmetadata", {})
                author = re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", "?"))[:80]
                lic = meta.get("LicenseShortName", {}).get("value", "?")
                name = f"{cls}_{n:02d}.jpg"
                try:
                    (OUT / cls / name).write_bytes(get(info["thumburl"]))
                except RuntimeError:
                    continue
                rows.append([cls, name, p["title"], info["descriptionurl"], author, lic])
                n += 1
                time.sleep(1)
        print(cls, n)
    for cls, idxs in REMOVE.items():
        for i in idxs:
            (OUT / cls / f"{cls}_{i:02d}.jpg").unlink(missing_ok=True)
    rows = [r for r in rows if (OUT / r[0] / r[1]).exists()]
    with open(OUT / "sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "file", "commons_title", "page", "author", "license"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
