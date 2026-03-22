# evaluate.py — Per-class AP analysis and training curve plots

import os
import json
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

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

def plot_per_class_ap_comparison(histories: list[dict], save_dir: str = RESULTS_DIR):
    """
    Side-by-side horizontal bar chart comparing per-class AP across two models.
    Bars are grouped by class, sorted by the gap between models.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from config import VOC_CLASS_CATEGORIES

    assert len(histories) == 2, "Pass exactly two model histories for comparison"

    h1, h2 = histories
    ap1 = h1["per_class_ap_final"]
    ap2 = h2["per_class_ap_final"]
    label1 = h1['model']
    label2 = h2['model']

    # ── Sort classes by gap (largest gap at top) ───────────────────────────────
    classes = [c for c in ap1.keys() if c in ap2]
    gaps = {c: ap2[c] - ap1[c] for c in classes}
    sorted_classes = sorted(classes, key=lambda c: gaps[c], reverse=True)

    ap1_vals = [ap1[c] for c in sorted_classes]
    ap2_vals = [ap2[c] for c in sorted_classes]
    gap_vals = [gaps[c] for c in sorted_classes]

    # ── Color bars by category ─────────────────────────────────────────────────
    category_colors = {
        "texture_defined":   "#2196F3",  # blue
        "context_dependent": "#FF5722",  # orange
        "ambiguous":         "#9C27B0",  # purple
    }
    class_to_category = {}
    for cat, cls_list in VOC_CLASS_CATEGORIES.items():
        for c in cls_list:
            class_to_category[c] = cat

    bar_colors = [category_colors[class_to_category[c]] for c in sorted_classes]

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    y = np.arange(len(sorted_classes))
    bar_height = 0.35

    # Left panel — grouped bars
    axes[0].barh(y + bar_height/2, ap2_vals, bar_height,
                 label=label2, color="#FF9800", alpha=0.85)
    axes[0].barh(y - bar_height/2, ap1_vals, bar_height,
                 label=label1, color="#1565C0", alpha=0.85)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([
        f"{c}  [{class_to_category[c].replace('_', ' ')}]"
        for c in sorted_classes
    ], fontsize=9)
    axes[0].set_xlabel("Average Precision")
    axes[0].set_title("Per-Class AP by Model")
    axes[0].set_xlim(0, 1.05)
    axes[0].axvline(h1["best_val_map"], color="#1565C0", linestyle="--", alpha=0.4)
    axes[0].axvline(h2["best_val_map"], color="#FF9800", linestyle="--", alpha=0.4)
    legend_elements_left = [
        plt.Rectangle((0,0), 1, 1, color="#FF9800", alpha=0.85, label=label2),
        plt.Rectangle((0,0), 1, 1, color="#1565C0", alpha=0.85, label=label1),
        Line2D([0], [0], color="#FF9800", linestyle="--", alpha=0.6, label=f"{h2['model']} mAP={h2['best_val_map']:.3f}"),
        Line2D([0], [0], color="#1565C0", linestyle="--", alpha=0.6, label=f"{h1['model']} mAP={h1['best_val_map']:.3f}"),
    ]
    axes[0].legend(handles=legend_elements_left, fontsize=7, loc="lower right")

    axes[0].grid(True, alpha=0.3, axis="x")

    # Right panel — gap bars colored by category
    axes[1].barh(y, gap_vals, color=bar_colors, alpha=0.85)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(sorted_classes, fontsize=9)
    axes[1].set_xlabel(f"AP Gap ({h2['model']} − {h1['model']})")
    axes[1].set_title("Per-Class Gap (sorted largest → smallest)")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].grid(True, alpha=0.3, axis="x")

    # Category legend for right panel
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=cat.replace("_", " "))
        for cat, color in category_colors.items()
    ]
    axes[1].legend(handles=legend_elements, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.4))

    plt.suptitle(
        f"Per-Class AP Comparison\n{label1} vs {label2} | "
        f"fraction={int(h1['fraction']*100)}% | aug={h1['aug_policy']}",
        fontsize=11
    )
    plt.tight_layout()

    out = os.path.join(save_dir, "per_class_ap_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Comparison chart saved → {out}")
    return out

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
    
    if len(histories) == 2:
        plot_per_class_ap_comparison(histories, save_dir=args.results_dir)