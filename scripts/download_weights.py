# scripts/download_weights.py
"""
Скачивает веса:
  - yolo11n-face.pt  (YOLOv11 nano face; берём community-чекпойнт)
  - glintr100.onnx   (ArcFace-совместимая модель эмбеддингов)

Использование:
  python scripts/download_weights.py               # в папку weights
  python scripts/download_weights.py --dir weithts # в папку weithts
  WEIGHTS_DIR=weithts python scripts/download_weights.py

Зависимости: только стандартная библиотека Python (urllib). PEP8-сумасшедший :)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# --- Константы и источники (resolve-ссылки HuggingFace) ----------------------

YOLO_CANDIDATES: list[str] = [
    # AdamCodd/YOLOv11n-face-detection (model.pt)
    "https://huggingface.co/AdamCodd/YOLOv11n-face-detection/resolve/main/model.pt?download=true",
    # deepghs/yolo-face (yolov11n-face/model.pt)
    "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov11n-face/model.pt?download=true",
]

GLINTR_CANDIDATES: list[str] = [
    # Несколько зеркал glintr100.onnx (одинаковый файл ~261 МБ)
    "https://huggingface.co/fofr/comfyui/resolve/main/insightface/models/antelopev2/glintr100.onnx?download=true",
    "https://huggingface.co/rupeshs/antelopev2/resolve/main/glintr100.onnx?download=true",
    "https://huggingface.co/LPDoctor/insightface/resolve/main/models/antelopev2/glintr100.onnx?download=true",
]

# Минимальные размеры (для валидации, чтобы не сохранить HTML страницу)
MIN_SIZE_BYTES = {
    "yolo11n-face.pt": 4 * 1024 * 1024,    # >= 4 МБ
    "glintr100.onnx": 200 * 1024 * 1024,   # >= 200 МБ
}


# --- Утилиты -----------------------------------------------------------------

def human(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    s = float(n)
    for u in units:
        if s < 1024.0 or u == units[-1]:
            return f"{s:.1f} {u}"
        s /= 1024.0
    return f"{n} B"


def sha256_of(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def get_content_length(url: str, timeout: int = 20) -> Optional[int]:
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "weights-downloader/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            clen = resp.headers.get("Content-Length")
            return int(clen) if clen and clen.isdigit() else None
    except Exception:
        return None


def stream_download(url: str, dest: Path, resume: bool = True, timeout: int = 30) -> None:
    """
    Скачивает url в dest с поддержкой докачки (HTTP Range) и простым прогрессом.
    Пишет во временный файл dest.with_suffix(dest.suffix + ".part"), затем переименовывает.
    """
    temp = dest.with_suffix(dest.suffix + ".part")
    downloaded = temp.stat().st_size if temp.exists() else 0

    headers = {"User-Agent": "weights-downloader/1.0"}
    if resume and downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            total = resp.headers.get("Content-Length")
            # Если сервер поддерживает Range, Content-Length = оставшиеся байты
            # Для прогресса попробуем получить полный размер
            server_total = get_content_length(url) or 0
            full_size = max(server_total, downloaded + (int(total) if total and total.isdigit() else 0))

            mode = "ab" if resume and downloaded > 0 else "wb"
            with temp.open(mode) as f:
                start = time.time()
                got = downloaded
                last_print = 0.0
                while True:
                    chunk = resp.read(1024 * 256)  # 256KB
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    now = time.time()
                    if now - last_print >= 0.25:
                        pct = (got / full_size * 100.0) if full_size > 0 else 0.0
                        speed = (got - downloaded) / max(1e-6, (now - start))
                        sys.stdout.write(
                            f"\r  -> {human(got)} / {human(full_size)} ({pct:5.1f}%) @ {human(int(speed))}/s"
                        )
                        sys.stdout.flush()
                        last_print = now
                sys.stdout.write("\n")
    except HTTPError as e:
        # Если сервер не поддержал Range — пробуем с нуля
        if e.code in (416, 400, 403) and temp.exists():
            temp.unlink(missing_ok=True)
            headers.pop("Range", None)
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                with temp.open("wb") as f:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
        else:
            raise
    except URLError as e:
        raise RuntimeError(f"URL error for {url}: {e}") from e

    temp.rename(dest)


def try_download_many(urls: Iterable[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(list(urls))}] Скачивание из: {url}")
        try:
            stream_download(url, dest, resume=True)
            print(f"✔ Готово: {dest} ({human(dest.stat().st_size)})")
            return
        except Exception as e:
            print(f"✖ Не удалось: {e}")
            last_err = e
    if last_err:
        raise last_err


def ensure_folder(preferred: Path) -> Path:
    # Приоритет: CLI/ENV -> если существует 'weithts' рядом — использовать её -> иначе 'weights'
    env_dir = os.getenv("WEIGHTS_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    if preferred.exists():
        return preferred.resolve()

    typo_dir = preferred.parent / "weithts"
    if typo_dir.exists():
        print("⚠️  Найдена папка 'weithts' — используем её.")
        return typo_dir.resolve()

    return preferred.resolve()


# --- Основная логика ---------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Скачивание весов YOLOv11 face и glintr100.onnx")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "weights",
        help="Папка для сохранения (по умолчанию: ./weights, можно указать ./weithts)",
    )
    parser.add_argument("--no-hash", action="store_true", help="Не считать sha256 после скачивания (быстрее).")
    args = parser.parse_args()

    dest_root = ensure_folder(args.dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    targets = [
        ("yolo11n-face.pt", YOLO_CANDIDATES),
        ("glintr100.onnx", GLINTR_CANDIDATES),
    ]

    for fname, urls in targets:
        dest = dest_root / fname
        print(f"\n=== {fname} ===")
        if dest.exists() and dest.stat().st_size >= MIN_SIZE_BYTES.get(fname, 0):
            print(f"⏭ Уже существует и выглядит валидным: {dest} ({human(dest.stat().st_size)})")
        else:
            try_download_many(urls, dest)

        if not args.no_hash:
            print("  Вычисляем sha256… (может занять время)")
            digest = sha256_of(dest)
            print(f"  sha256: {digest}")

        # Простая проверка размера (страховка от HTML/ошибок)
        min_size = MIN_SIZE_BYTES.get(fname, 0)
        size = dest.stat().st_size
        if size < min_size:
            raise RuntimeError(
                f"Размер файла {fname} слишком мал ({human(size)} < {human(min_size)}). "
                f"Вероятно, не тот ресурс был скачан."
            )

    print("\n✅ Все веса скачаны успешно.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n❌ Ошибка: {exc}")
        sys.exit(1)
