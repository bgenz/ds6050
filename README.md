# CNNs vs Vision Transformers — PASCAL VOC Multi-Label Classification
## DS 6050 Group Project — Milestone 2 Codebase

---

## Project Structure

```
voc_project/
├── config.py          # All hyperparameters and paths — edit this first
├── dataset.py         # VOC DataLoader, transforms, stratified sampling
├── models.py          # ResNet-50, ResNet-101, ViT-B/16, DeiT-S builders
├── train.py           # Training loop with mAP logging + early stopping
├── evaluate.py        # Per-class AP plots + training curve plots
├── run_baseline.py    # Quick CPU run for Milestone 2 baseline
├── results/           # JSON histories + PNG plots (auto-created)
└── checkpoints/       # Best model .pt files (auto-created)
```

---

## Setup

```bash
pip install torch torchvision timm scikit-learn matplotlib
pip install scikit-multilearn   # for multi-label stratified sampling
```

---

## Data Setup

1. Download Pascal VOC 2012 from Kaggle
2. Extract so the structure looks like:
   ```
   data/
   └── VOCdevkit/
       └── VOC2012/
           ├── JPEGImages/
           ├── Annotations/
           └── ImageSets/
               └── Main/
                   ├── train.txt
                   └── val.txt
   ```
3. Update `VOC_ROOT` in `config.py` if your path differs

---

## Running the Milestone 2 Baseline (CPU)

```bash
# Both ResNet-50 and ViT-B/16 at 5% data, 3 epochs
python run_baseline.py

# Single model
python run_baseline.py --model resnet50
```

Outputs in `./results/`:
- `*_history.json`        — epoch-by-epoch loss and mAP
- `*_curves.png`          — training/val loss + mAP plots
- `*_per_class_ap.png`    — per-class AP bar chart (error analysis)
- `model_comparison_val_map.png` — multi-model overlay

---

## Running Full Experiments (Rivanna GPU)

```bash
# Single run via CLI
python train.py --model resnet50 --pretrained --fraction 1.0 --aug standard --seed 42

# From scratch
python train.py --model vit_b16 --scratch --fraction 0.05 --aug none --seed 42
```

### Full experiment grid (40 configs × 3 seeds = 120 runs)
Recommended order on Rivanna:
1. Pretrained ResNet-50 + ViT-B/16 at all 5 data fractions (most informative)
2. Pretrained ResNet-101 + DeiT-S
3. From-scratch ResNet-50 + ViT-B/16 (expect ViT to struggle at low data)

---

## Ablation Study Plan (Milestone 2 Required)

| Experiment | Models | Conditions | Purpose |
|---|---|---|---|
| Core scaling | All 4 | 5 fractions × 2 pretraining × 3 seeds | Main hypothesis: ViT needs more data |
| Augmentation | ResNet-50, ViT-B/16 | none / standard / strong @ 5% and 100% | Does aug help ViT more? |
| DeiT-S vs ViT-B/16 | DeiT-S, ViT-B/16 | Low vs high data fraction | Does distillation help at low data? |
| Per-class AP | Best 2 configs | All 20 classes | Context-dep vs texture-dep classes |
| Efficiency | All 4 | Params, FLOP, time, memory | mAP-per-FLOP comparison |

---

## Key Design Decisions

- **Loss:** `BCEWithLogitsLoss` (multi-label — sigmoid head, not softmax)
- **Metric:** mAP (mean Average Precision across 20 classes)
- **Stratification:** `IterativeStratification` from `scikit-multilearn`
- **ViT checkpoint:** `vit_base_patch16_224.augreg_in21k` (ImageNet-21k pretrained)
- **Differential LRs:** backbone at lower LR than head (see `config.py`)
