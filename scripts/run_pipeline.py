"""Run everything that follows the download, unattended.

Waits for all 112,120 images to land, then preprocesses, trains, and produces the
three results tables. Steps whose output already exists are skipped, so this is
safe to restart after an interruption.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-wait    # data is already complete
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXPECTED_IMAGES = 112_120
#: How long to go without new images before saying so. supervise_download.py
#: restarts a downloader that has died, so this only reports; it never gives up.
STALL_MINUTES = 30


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def count_png(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.png")) if directory.exists() else 0


def wait_for_images(raw_images: Path) -> bool:
    """Block until every image is on disk. False if progress stops."""
    last_count, last_change = count_png(raw_images), time.monotonic()
    log(f"waiting for images: {last_count} of {EXPECTED_IMAGES}")

    while last_count < EXPECTED_IMAGES:
        time.sleep(60)
        count = count_png(raw_images)
        if count != last_count:
            if count // 5000 != last_count // 5000:
                log(f"images: {count} of {EXPECTED_IMAGES}")
            last_count, last_change = count, time.monotonic()
        elif time.monotonic() - last_change > STALL_MINUTES * 60:
            log(f"no new images for {STALL_MINUTES} min, still waiting at {count}")
            last_change = time.monotonic()
    log(f"all {last_count} images present")
    return True


def run(step: str, args: list[str]) -> bool:
    log(f"start {step}")
    started = time.monotonic()
    result = subprocess.run([sys.executable, *args], check=False)
    minutes = (time.monotonic() - started) / 60
    if result.returncode != 0:
        log(f"FAILED {step} after {minutes:.1f} min (exit {result.returncode})")
        return False
    log(f"done {step} in {minutes:.1f} min")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Chain preprocessing, training and evaluation.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/nih"))
    parser.add_argument("--images", type=Path, default=Path("data/processed/nih512"))
    parser.add_argument("--run", type=Path, default=Path("runs/densenet121_512"))
    parser.add_argument("--xai-out", type=Path, default=Path("runs/xai"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    if not args.skip_wait and not wait_for_images(args.raw / "images"):
        return 1

    if count_png(args.images) < EXPECTED_IMAGES:
        if not run("preprocessing", [
            "scripts/preprocess_nih.py",
            "--src", str(args.raw / "images"),
            "--dst", str(args.images),
            "--size", "512",
        ]):
            return 1
    else:
        log("preprocessing already complete, skipping")

    processed = count_png(args.images)
    if processed != EXPECTED_IMAGES:
        log(f"stopping: {processed} preprocessed images, expected {EXPECTED_IMAGES}")
        return 1

    checkpoint = args.run / "best.pt"
    if checkpoint.exists():
        log(f"{checkpoint} already exists, skipping training")
    elif not run("training", [
        "scripts/train.py",
        "--data", str(args.raw),
        "--images", str(args.images),
        "--out", str(args.run),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
    ]):
        return 1

    # Evaluation steps are independent of each other, so a failure in one still
    # leaves the others' tables on disk to look at.
    ok = True
    ok &= run("classification metrics", ["scripts/evaluate.py", "--run", str(args.run)])
    ok &= run("localisation vs boxes", [
        "scripts/evaluate_xai.py",
        "--checkpoint", str(checkpoint),
        "--out", str(args.xai_out),
    ])
    ok &= run("faithfulness and sanity check", [
        "scripts/evaluate_faithfulness.py",
        "--checkpoint", str(checkpoint),
        "--out", str(args.xai_out),
    ])

    log("pipeline finished" if ok else "pipeline finished with failures")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
