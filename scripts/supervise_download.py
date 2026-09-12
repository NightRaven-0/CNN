"""Keep the dataset download running until every image is on disk.

The connection here goes through a VPN on college wifi, which truncates
transfers near the end and sometimes cuts out entirely. ``download_nih.py``
already resumes after a truncated read and retries with backoff, but once it
exhausts its attempts it exits and nothing brings it back. This does.

It will not start a second downloader on top of a running one, which would
corrupt the partial file both were writing. The downloader logs on every
attempt and its longest backoff is 20 minutes, so a log that has been silent
for 30 belongs to a downloader that has stopped.

Usage:
    python scripts/supervise_download.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXPECTED_IMAGES = 112_120
IDLE_MINUTES = 30


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), f"[supervisor] {msg}", flush=True)


def count_png(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.png")) if directory.exists() else 0


def minutes_since_touched(path: Path) -> float:
    """Minutes since the log last grew. Large when it does not exist yet."""
    return (time.time() - path.stat().st_mtime) / 60 if path.exists() else 1e9


def main() -> int:
    parser = argparse.ArgumentParser(description="Restart the downloader until it finishes.")
    parser.add_argument("--out", type=Path, default=Path("data/raw/nih"))
    parser.add_argument("--idle-minutes", type=float, default=IDLE_MINUTES)
    args = parser.parse_args()

    images = args.out / "images"
    log_path = args.out / "download.log"
    fruitless = 0

    while True:
        count = count_png(images)
        if count >= EXPECTED_IMAGES:
            log(f"all {count} images present, nothing left to supervise")
            return 0

        idle = minutes_since_touched(log_path)
        if idle < args.idle_minutes:
            time.sleep(120)
            continue

        log(f"download log quiet for {idle:.0f} min at {count} images, restarting downloader")
        started = time.monotonic()
        with open(log_path, "a", encoding="utf-8") as sink:
            result = subprocess.run(
                [sys.executable, "scripts/download_nih.py", "--out", str(args.out)],
                stdout=sink,
                stderr=subprocess.STDOUT,
                check=False,
            )
        gained = count_png(images) - count
        log(
            f"downloader exited {result.returncode} after "
            f"{(time.monotonic() - started) / 60:.1f} min, {gained} new images"
        )

        if gained:
            fruitless = 0
            continue

        # Nothing came back. Could be the VPN being down, so wait longer each
        # time rather than hammering a connection that is not there.
        fruitless += 1
        wait = min(30, 5 * fruitless)
        log(f"no progress on that attempt, waiting {wait} min before the next")
        time.sleep(wait * 60)


if __name__ == "__main__":
    sys.exit(main())
