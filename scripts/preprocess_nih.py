"""Resize NIH ChestX-ray14 images to a fixed square size as 8-bit greyscale PNG.

Decoding a 1024-pixel PNG for every sample every epoch is the main data-loading
cost, so training reads these smaller copies instead. PNG keeps it lossless.
Safe to re-run: files that already exist are skipped, so it can also be run
while the download is still going.

Usage:
    python scripts/preprocess_nih.py --src data/raw/nih/images --dst data/processed/nih512
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def to_gray8(im: Image.Image) -> Image.Image:
    """8-bit greyscale. A few NIH files are 16-bit or RGBA, which ``convert`` mishandles."""
    if im.mode in ("I;16", "I;16B", "I"):
        arr = np.asarray(im, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / max(hi - lo, 1.0) * 255.0
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    return im.convert("L")


def convert(job: tuple[Path, Path, int]) -> int:
    src, dst, size = job
    if dst.exists():
        return 0
    with Image.open(src) as im:
        out = to_gray8(im)
        if out.size != (size, size):
            out = out.resize((size, size), Image.Resampling.LANCZOS, reducing_gap=2.0)
    part = dst.with_name(dst.name + ".part")
    out.save(part, format="PNG")
    part.replace(dst)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Resize NIH ChestX-ray14 images.")
    parser.add_argument("--src", type=Path, default=Path("data/raw/nih/images"))
    parser.add_argument("--dst", type=Path, default=Path("data/processed/nih512"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    jobs = [
        (src, args.dst / src.name, args.size)
        for src in sorted(args.src.glob("*.png"))
        if not (args.dst / src.name).exists()
    ]
    print(f"{len(jobs)} images to convert with {args.workers} workers", flush=True)

    done, start = 0, time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n in pool.map(convert, jobs, chunksize=64):
            done += n
            if done and done % 5000 == 0:
                rate = done / (time.monotonic() - start)
                print(f"{done}/{len(jobs)} converted, {rate:.0f} images/s", flush=True)
    print(f"finished: {done} converted", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
