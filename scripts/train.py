"""Train the DenseNet-121 baseline on NIH ChestX-ray14, evaluated on the official test split.

Usage:
    python scripts/train.py --data data/raw/nih --images data/processed/nih512 --out runs/densenet121_512

Smoke test on a partial download (random init, downloads nothing):
    python scripts/train.py --allow-missing --limit 256 --epochs 1 --size 256 --no-pretrained --out runs/smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from arc.data import (
    IMAGE_COL,
    LABELS,
    PATIENT_COL,
    ChestXray14,
    build_transforms,
    load_metadata,
    positive_weights,
)
from arc.model import DenseNet121Classifier
from arc.splits import summarise


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def per_class_auroc(targets: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """AUROC per label, NaN where the split holds only one class for that label."""
    scores = {}
    for j, label in enumerate(LABELS):
        y = targets[:, j]
        scores[label] = float(roc_auc_score(y, probs[:, j])) if 0 < y.sum() < len(y) else math.nan
    return scores


def mean_of(scores: dict[str, float]) -> float:
    values = [v for v in scores.values() if not math.isnan(v)]
    return float(np.mean(values)) if values else math.nan


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype | None
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, targets = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu())
        targets.append(y)
    return torch.cat(targets).numpy(), torch.cat(probs).numpy()


def make_loader(ds: Dataset, *, batch_size: int, workers: int, train: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=workers,
        pin_memory=True,
        # Windows starts workers by spawning fresh processes, which is slow, so
        # keep them alive between epochs.
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the DenseNet-121 baseline.")
    p.add_argument("--data", type=Path, default=Path("data/raw/nih"))
    p.add_argument("--images", type=Path, default=Path("data/processed/nih512"))
    p.add_argument("--out", type=Path, default=Path("runs/densenet121_512"))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--pos-weight-power", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=3, help="epochs without val gain before stopping")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-pretrained", action="store_true", help="random init, downloads nothing")
    p.add_argument("--allow-missing", action="store_true", help="skip absent images (smoke tests)")
    p.add_argument("--limit", type=int, default=0, help="cap images per split (smoke tests)")
    p.add_argument("--fresh", action="store_true", help="ignore any last.pt and start over")
    return p.parse_args()


def save_atomic(payload: dict, path: Path) -> None:
    """Write a checkpoint through a temporary file so a crash cannot truncate it.

    A power cut partway through torch.save leaves a half-written file that will
    not load, which would defeat the point of keeping a resume checkpoint.
    """
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    df = load_metadata(args.data, seed=args.seed)
    present = {p.name for p in args.images.glob("*.png")}
    missing = ~df[IMAGE_COL].isin(present)
    if missing.any():
        if not args.allow_missing:
            log(
                f"{int(missing.sum())} images are not in {args.images}. Finish preprocessing, "
                "or pass --allow-missing for a smoke test."
            )
            return 1
        df = df[~missing]

    frames = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    if args.limit:
        frames = {
            s: f.sample(min(len(f), args.limit), random_state=args.seed) for s, f in frames.items()
        }
    used = pd.concat(frames.values())
    used[[IMAGE_COL, PATIENT_COL, "split"]].to_csv(args.out / "splits.csv", index=False)
    summarise(used, LABELS).to_csv(args.out / "split_summary.csv")
    for s, f in frames.items():
        log(f"{s}: {len(f)} images from {f[PATIENT_COL].nunique()} patients")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    train_loader = make_loader(
        ChestXray14(frames["train"], args.images, build_transforms(args.size, train=True)),
        batch_size=args.batch_size, workers=args.workers, train=True,
    )
    val_loader = make_loader(
        ChestXray14(frames["val"], args.images, build_transforms(args.size, train=False)),
        batch_size=args.batch_size * 2, workers=args.workers, train=False,
    )
    if len(train_loader) == 0:
        log("training set is smaller than one batch")
        return 1

    model = DenseNet121Classifier(len(LABELS), pretrained=not args.no_pretrained)
    model = model.to(device, memory_format=torch.channels_last)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weights(frames["train"], args.pos_weight_power).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.05
    )

    checkpoint = args.out / "best.pt"
    resume_point = args.out / "last.pt"
    history: list[dict[str, float]] = []
    best, stale, first_epoch = -math.inf, 0, 1

    # Mains power has interrupted this project three times in a day, so a run
    # that dies mid-training continues from the last finished epoch rather than
    # throwing away two hours of GPU time.
    if resume_point.exists() and not args.fresh:
        saved = torch.load(resume_point, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best, stale, history = saved["best"], saved["stale"], saved["history"]
        first_epoch = saved["epoch"] + 1
        torch.set_rng_state(saved["rng"].cpu())
        if torch.cuda.is_available() and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        for key in ("epochs", "batch_size", "size"):
            was, now = saved["args"].get(key), getattr(args, key)
            if str(was) != str(now):
                log(f"warning: --{key} was {was} in the interrupted run and is {now} now")
        log(f"resuming at epoch {first_epoch}, best val AUROC so far {best:.4f}")

    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        start, running = time.monotonic(), 0.0
        for step, (x, y, _) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
            loss = criterion(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 250 == 0:
                log(f"epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}")

        val_scores = per_class_auroc(*predict(model, val_loader, device, amp_dtype))
        val_auc = mean_of(val_scores)
        minutes = (time.monotonic() - start) / 60
        train_loss = running / len(train_loader)
        log(f"epoch {epoch}: train loss {train_loss:.4f}, val mean AUROC {val_auc:.4f}, {minutes:.1f} min")
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_mean_auroc": val_auc, "minutes": minutes}
            | {f"val_{k}": v for k, v in val_scores.items()}
        )
        pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)

        if val_auc > best or not checkpoint.exists():
            best, stale = (val_auc if not math.isnan(val_auc) else best), 0
            save_atomic(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_mean_auroc": val_auc,
                    "args": {k: str(v) for k, v in vars(args).items()},
                },
                checkpoint,
            )
        else:
            stale += 1

        # Written every epoch, whether or not the model improved, since this is
        # what an interrupted run comes back to.
        save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best": best,
                "stale": stale,
                "history": history,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "args": {k: str(v) for k, v in vars(args).items()},
            },
            resume_point,
        )

        if stale >= args.patience:
            log(f"no improvement for {args.patience} epochs, stopping")
            break

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    test_ds = ChestXray14(frames["test"], args.images, build_transforms(args.size, train=False))
    test_loader = make_loader(test_ds, batch_size=args.batch_size * 2, workers=args.workers, train=False)
    y_test, p_test = predict(model, test_loader, device, amp_dtype)
    test_scores = per_class_auroc(y_test, p_test)

    predictions = pd.DataFrame(p_test, columns=LABELS)
    predictions.insert(0, IMAGE_COL, test_ds.frame[IMAGE_COL].to_numpy())
    predictions.to_csv(args.out / "test_predictions.csv", index=False)

    metrics = {
        "best_epoch": state["epoch"],
        "val_mean_auroc": state["val_mean_auroc"],
        "test_mean_auroc": mean_of(test_scores),
        "test_auroc": test_scores,
        "n_test_images": len(test_ds),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"test mean AUROC {metrics['test_mean_auroc']:.4f} over {len(test_ds)} images")
    for label, score in test_scores.items():
        log(f"  {label:<20} {score:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
