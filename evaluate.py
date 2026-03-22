# evaluate.py — Per-class AP analysis and training curve plots

import os
import json
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import VOC_CLASSES, RESULTS_DIR


# ── Training curve plots ───────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_dir: str = RESULTS_DIR):
    """Plot loss and mAP curves for a single run."""
    tag    = history["tag"]
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{tag}", fontsize=11, y=1.01)

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
    axes[0].axvline(history["best_epoch"], color="gray", linestyle="--", alpha=0.6, label="Best epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # mAP
    axes[1].plot(epochs, history["train_map"], label="Train mAP", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_map"],   label="Val mAP",   marker="o", markersize=3)
    axes[1].axvline(history["best_epoch"], color="gray", linestyle="--", alpha=0.6, label="Best epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].set_title("Mean Average Precision")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, f"{tag}_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Curves saved → {out}")
    return out


def plot_per_class_ap(history: dict, save_dir: str = RESULTS_DIR):
    """Horizontal bar chart of per-class AP at best checkpoint."""
    per_class = history.get("per_class_ap_final", {})
    if not per_class:
        print("[evaluate] No per-class AP data found in history.")
        return

    # Sort by AP descending
    sorted_items = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
    classes, aps = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3" if ap >= 0.5 else "#FF5722" for ap in aps]
    bars = ax.barh(classes, aps, color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Average Precision")
    ax.set_title(f"Per-Class AP @ best epoch\n{history['tag']}")
    ax.axvline(np.mean(aps), color="black", linestyle="--", alpha=0.7,
               label=f"mAP = {np.mean(aps):.3f}")
    ax.legend()

    # Value labels
    for bar, ap in zip(bars, aps):
        ax.text(ap + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{ap:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    out = os.path.join(save_dir, f"{history['tag']}_per_class_ap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Per-class AP chart saved → {out}")
    return out


def plot_model_comparison(histories: list[dict], save_dir: str = RESULTS_DIR):
    """
    Compare val mAP curves across multiple runs on the same plot.
    Useful for Milestone 2 comparison figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for h in histories:
        label = f"{h['model']} | pt={h['pretrained']} | f={int(h['fraction']*100)}%"
        epochs = range(1, len(h["val_map"]) + 1)
        ax.plot(epochs, h["val_map"], marker="o", markersize=3, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val mAP")
    ax.set_title("Validation mAP Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "model_comparison_val_map.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Comparison plot saved → {out}")
    return out


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary_table(histories: list[dict]):
    """Print a results table to stdout."""
    header = f"{'Tag':<55} {'Best mAP':>9} {'Epoch':>6} {'Time(s)':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for h in sorted(histories, key=lambda x: x["best_val_map"], reverse=True):
        print(
            f"{h['tag']:<55} "
            f"{h['best_val_map']:>9.4f} "
            f"{h['best_epoch']:>6} "
            f"{h['training_time_s']:>8.1f}"
        )
    print("=" * len(header))


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curves and per-class AP")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR,
                        help="Directory containing *_history.json files")
    parser.add_argument("--compare",     action="store_true",
                        help="Also produce multi-model comparison plot")
    args = parser.parse_args()

    json_files = glob.glob(os.path.join(args.results_dir, "*_history.json"))
    if not json_files:
        print(f"No history files found in {args.results_dir}")
        exit(1)

    histories = []
    for path in json_files:
        with open(path) as f:
            h = json.load(f)
        histories.append(h)
        plot_training_curves(h, save_dir=args.results_dir)
        plot_per_class_ap(h,    save_dir=args.results_dir)

    if args.compare and len(histories) > 1:
        plot_model_comparison(histories, save_dir=args.results_dir)

    print_summary_table(histories)
