"""Download and unpack the NIH ChestX-ray14 release.

Every file comes from the NIH Clinical Center's own Box share
(https://nihcc.app.box.com/v/ChestXray-NIHCC). The file ids in
``nih_manifest.json`` were read from that share's folder listing.

Image archives are downloaded (resumably), unpacked into ``images/``, checked,
and then deleted, so peak disk use stays about one archive above the unpacked
images. Re-running is safe: finished archives are recorded and skipped.

Usage:
    python scripts/download_nih.py --out data/raw/nih
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import shutil
import sys
import tarfile
import time
import urllib.request
import zlib
from pathlib import Path

DOWNLOAD_URL = (
    "https://nihcc.app.box.com/index.php?rm=box_download_shared_file"
    "&shared_name={shared}&file_id={file_id}"
)
MANIFEST = Path(__file__).with_name("nih_manifest.json")
CHUNK = 1 << 20
DONE_MARKER = ".done_archives"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def source_url(entry: dict, shared: str) -> str:
    """The entry's direct NIH link if it has one, else the Box shared-link endpoint.

    The shared-link endpoint on nihcc.app.box.com started answering every request
    with 404 after about two hours of steady downloading, its folder page
    included. The direct links on nihcc.box.com are the ones NIH publishes in
    batch_download_zips.py; they support range requests, and each one's
    Content-Length matches the size in the manifest.
    """
    url = entry.get("url")
    return url if isinstance(url, str) else DOWNLOAD_URL.format(shared=shared, file_id=entry["file_id"])


def fetch(url: str, dest: Path, *, retries: int = 20) -> None:
    """Download to ``dest`` through a ``.part`` file, resuming after dropped connections."""
    part = dest.with_name(dest.name + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as resp:
                if "text/html" in resp.headers.get("Content-Type", ""):
                    # Box answers errors with a web page; never save that as data.
                    raise RuntimeError(f"{dest.name}: server sent an HTML page, not the file")
                if have and resp.status != 206:
                    log(f"{dest.name}: server ignored the resume request, starting over")
                    have = 0
                length = int(resp.headers.get("Content-Length") or 0)
                total = have + length if length else 0

                started = last = time.monotonic()
                done = start_bytes = have
                with open(part, "ab" if have else "wb") as f:
                    while chunk := resp.read(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if now - last >= 60:
                            rate = (done - start_bytes) / (now - started) / 1e6
                            pct = f"{100 * done / total:.0f}%" if total else "?"
                            log(f"{dest.name}: {done / 1e9:.2f} GB ({pct}), {rate:.1f} MB/s")
                            last = now

            if total and part.stat().st_size != total:
                raise ConnectionError(f"short read: {part.stat().st_size} of {total} bytes")
            part.replace(dest)
            return
        except (OSError, http.client.HTTPException) as exc:
            # After a couple of hours of steady downloading, Box starts answering
            # every shared-link request with 404 and then clears on its own. Back
            # off hard instead of burning attempts a minute apart.
            wait = min(1200, 30 * 2 ** (attempt - 1))
            log(f"{dest.name}: attempt {attempt}/{retries} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"{dest.name}: gave up after {retries} attempts")


def unpack(archive: Path, images_dir: Path) -> int:
    """Stream PNGs out of ``archive`` into ``images_dir``. Returns the number written."""
    images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not (member.isfile() and member.name.lower().endswith(".png")):
                continue
            # Keep only the base name. This flattens the archive's images/ folder
            # and means no member path can write outside images_dir.
            target = images_dir / Path(member.name).name
            if target.exists() and target.stat().st_size == member.size:
                count += 1
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            count += 1
    return count


def missing_images(out: Path, images_dir: Path) -> list[str]:
    with open(out / "Data_Entry_2017_v2020.csv", newline="") as f:
        expected = {row["Image Index"] for row in csv.DictReader(f)}
    present = {p.name for p in images_dir.glob("*.png")}
    return sorted(expected - present)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download NIH ChestX-ray14.")
    parser.add_argument("--out", type=Path, default=Path("data/raw/nih"))
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--only", nargs="*", help="process only these archive names")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    shared = manifest["shared_name"]
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    images_dir = out / "images"
    done_file = out / DONE_MARKER
    done = set(done_file.read_text().split()) if done_file.exists() else set()

    for entry in manifest["metadata"]:
        dest = out / entry["name"]
        if not (dest.exists() and dest.stat().st_size > 0):
            log(f"fetching {entry['name']}")
            fetch(source_url(entry, shared), dest)

    archives = manifest["archives"]
    if args.only:
        archives = [a for a in archives if a["name"] in set(args.only)]

    for entry in archives:
        name = entry["name"]
        if name in done:
            log(f"{name}: already unpacked, skipping")
            continue

        # The archive plus its unpacked contents, with some room to spare.
        need = entry["size"] * 2.2
        free = shutil.disk_usage(out).free
        if free < need:
            log(f"stopping: {free / 1e9:.1f} GB free, {name} needs about {need / 1e9:.1f} GB")
            return 1

        archive = out / name
        for attempt in (1, 2):
            if not archive.exists():
                log(f"{name}: downloading {entry['size'] / 1e9:.2f} GB")
                fetch(source_url(entry, shared), archive)
            if archive.stat().st_size != entry["size"]:
                log(f"{name}: size {archive.stat().st_size} != expected {entry['size']}")
                archive.unlink()
                if attempt == 2:
                    return 1
                continue
            log(f"{name}: unpacking")
            try:
                n = unpack(archive, images_dir)
                break
            except (tarfile.TarError, EOFError, zlib.error) as exc:
                log(f"{name}: archive is corrupt ({exc}), deleting it")
                archive.unlink(missing_ok=True)
                if attempt == 2:
                    return 1

        if n == 0:
            log(f"{name}: contained no PNG files, stopping")
            return 1
        log(f"{name}: {n} images unpacked")
        with open(done_file, "a") as f:
            f.write(name + "\n")
        if not args.keep_archives:
            archive.unlink()
            log(f"{name}: archive deleted")

    missing = missing_images(out, images_dir)
    log(f"{sum(1 for _ in images_dir.glob('*.png'))} images on disk, {len(missing)} missing")
    if missing and not args.only:
        log(f"first missing: {missing[:5]}")
        return 1
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
