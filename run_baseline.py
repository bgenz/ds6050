# run_baseline.py — Quick CPU-friendly baseline run for Milestone 2
#
# PURPOSE: Produces real training curves and mAP metrics for your M2 submission
# using a small data fraction and few epochs so it finishes on CPU.
#
# Usage:
#   python run_baseline.py                  # runs both ResNet-50 and ViT-B/16
#   python run_baseline.py --model resnet50 # runs one model only

import argparse
from train import train_one_run
from evaluate import plot_training_curves, plot_per_class_ap, plot_model_comparison, print_summary_table

# ── CPU-friendly settings ──────────────────────────────────────────────────────
# 5% of VOC train ≈ 286 images, 3 epochs — should complete in ~10-20 min on CPU.
# Increase fraction/epochs when you move to Rivanna GPU.

CPU_FRACTION = 0.05   # bump to 0.10 or 0.20 for more signal
CPU_EPOCHS   = 3      # bump to 30 on GPU with early stopping


def main(models_to_run: list[str]):
    histories = []

    for model_name in models_to_run:
        h = train_one_run(
            model_name  = model_name,
            pretrained  = True,
            fraction    = CPU_FRACTION,
            aug_policy  = "standard",
            seed        = 42,
            epochs      = CPU_EPOCHS,
            early_stop  = 999,      # disable early stopping for short runs
        )
        histories.append(h)
        plot_training_curves(h)
        plot_per_class_ap(h)

    if len(histories) > 1:
        plot_model_comparison(histories)

    print_summary_table(histories)
    print("\nDone! Check ./results/ for plots and JSON history files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="both",
        choices=["resnet50", "vit_b16", "both"],
        help="Which model(s) to run"
    )
    args = parser.parse_args()

    if args.model == "both":
        models = ["resnet50", "vit_b16"]
    else:
        models = [args.model]

    main(models)
