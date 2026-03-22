# train.py — Training loop with mAP logging, early stopping, and curve saving

import os
import json
import time
import argparse

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score

from config import (
    DEVICE, EPOCHS, EARLY_STOP_PAT,
    RESULTS_DIR, CHECKPOINTS_DIR, VOC_CLASSES
)
from dataset import get_dataloaders
from models import get_model, get_optimizer


# ── mAP computation ────────────────────────────────────────────────────────────

def compute_map(all_targets: np.ndarray, all_scores: np.ndarray) -> tuple[float, dict]:
    """
    Compute mean Average Precision (mAP) for multi-label classification.

    Args:
        all_targets: (N, 20) binary ground-truth matrix
        all_scores:  (N, 20) predicted probability matrix (after sigmoid)

    Returns:
        (mAP, per_class_ap_dict)
    """
    per_class_ap = {}
    aps = []
    for i, cls_name in enumerate(VOC_CLASSES):
        if all_targets[:, i].sum() == 0:
            # No positive examples for this class in this split — skip
            continue
        ap = average_precision_score(all_targets[:, i], all_scores[:, i])
        per_class_ap[cls_name] = round(float(ap), 4)
        aps.append(ap)
    mAP = float(np.mean(aps)) if aps else 0.0
    return mAP, per_class_ap


# ── Evaluation pass ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion) -> tuple[float, float, dict]:
    """
    Returns (val_loss, val_mAP, per_class_ap).
    """
    model.eval()
    running_loss = 0.0
    all_targets  = []
    all_scores   = []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss   = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)

        scores = torch.sigmoid(logits).cpu().numpy()
        all_scores.append(scores)
        all_targets.append(labels.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_scores  = np.concatenate(all_scores,  axis=0)

    avg_loss = running_loss / len(loader.dataset)
    mAP, per_class_ap = compute_map(all_targets, all_scores)
    return avg_loss, mAP, per_class_ap


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one_run(
    model_name:  str,
    pretrained:  bool  = True,
    fraction:    float = 1.0,
    aug_policy:  str   = "standard",
    seed:        int   = 42,
    epochs:      int   = EPOCHS,
    early_stop:  int   = EARLY_STOP_PAT,
    run_tag:     str   = None,
):
    """
    Full training run. Returns history dict with train/val loss and mAP per epoch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Tag for saving ─────────────────────────────────────────────────────────
    tag = run_tag or f"{model_name}_pt{int(pretrained)}_f{int(fraction*100)}_aug{aug_policy}_s{seed}"
    os.makedirs(RESULTS_DIR,     exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Run: {tag}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        fraction=fraction, aug_policy=aug_policy, seed=seed
    )

    # ── Model, loss, optimizer ─────────────────────────────────────────────────
    model     = get_model(model_name, pretrained=pretrained).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()          # correct loss for multi-label
    optimizer = get_optimizer(model, model_name)

    # ── History ────────────────────────────────────────────────────────────────
    history = {
        "tag":        tag,
        "model":      model_name,
        "pretrained": pretrained,
        "fraction":   fraction,
        "aug_policy": aug_policy,
        "seed":       seed,
        "train_loss": [],
        "val_loss":   [],
        "train_map":  [],
        "val_map":    [],
        "per_class_ap_final": {},
        "best_epoch": 0,
        "best_val_map": 0.0,
        "training_time_s": 0.0,
    }

    best_val_map  = -1.0
    patience_ctr  = 0
    best_ckpt     = os.path.join(CHECKPOINTS_DIR, f"{tag}_best.pt")
    t0            = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        running_loss   = 0.0
        train_targets  = []
        train_scores   = []

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_scores.append(torch.sigmoid(logits).detach().cpu().numpy())
            train_targets.append(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_targets_np = np.concatenate(train_targets, axis=0)
        train_scores_np  = np.concatenate(train_scores,  axis=0)
        train_map, _     = compute_map(train_targets_np, train_scores_np)

        # ── Validate ───────────────────────────────────────────────────────────
        val_loss, val_map, per_class_ap = evaluate(model, val_loader, criterion)

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["train_map"].append(round(train_map, 4))
        history["val_map"].append(round(val_map, 4))

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f}  train_mAP={train_map:.4f} | "
            f"val_loss={val_loss:.4f}  val_mAP={val_map:.4f}"
        )

        # ── Early stopping ─────────────────────────────────────────────────────
        if val_map > best_val_map:
            best_val_map  = val_map
            patience_ctr  = 0
            history["best_epoch"]      = epoch
            history["best_val_map"]    = round(best_val_map, 4)
            history["per_class_ap_final"] = per_class_ap
            torch.save(model.state_dict(), best_ckpt)
            print(f"    ✓ New best val mAP: {best_val_map:.4f} — checkpoint saved")
        else:
            patience_ctr += 1
            if patience_ctr >= early_stop:
                print(f"\n  Early stopping triggered (patience={early_stop})")
                break

    history["training_time_s"] = round(time.time() - t0, 1)

    # ── Save history ───────────────────────────────────────────────────────────
    history_path = os.path.join(RESULTS_DIR, f"{tag}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  History saved → {history_path}")
    print(f"  Best val mAP: {best_val_map:.4f} at epoch {history['best_epoch']}")
    print(f"  Total time:   {history['training_time_s']}s")

    return history


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single model run")
    parser.add_argument("--model",      type=str,   default="resnet50",
                        choices=["resnet50", "resnet101", "vit_b16", "deit_s"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--scratch",    action="store_true", default=False,
                        help="Train from random initialization")
    parser.add_argument("--fraction",   type=float, default=0.05,
                        help="Data fraction to use (e.g. 0.05 for 5%%)")
    parser.add_argument("--aug",        type=str,   default="standard",
                        choices=["none", "standard", "strong"])
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    args = parser.parse_args()

    train_one_run(
        model_name  = args.model,
        pretrained  = not args.scratch,
        fraction    = args.fraction,
        aug_policy  = args.aug,
        seed        = args.seed,
        epochs      = args.epochs,
    )
