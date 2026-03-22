# config.py — Central configuration for VOC CNN vs ViT project

import torch

# ── Dataset ────────────────────────────────────────────────────────────────────
VOC_ROOT        = r"D:\Data Science\Semester 5 (Spring 26)\DS6050\Project\.data\VOCdevkit"   # path to your Kaggle-downloaded VOC data
VOC_YEAR        = "2012"
NUM_CLASSES     = 20
IMAGE_SIZE      = 224

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

VOC_CLASS_CATEGORIES = {
    "texture_defined": [
        "cat",
        "dog",
        "bird",
        "sheep",
        "cow",
        "horse",
        "bottle",
        "tvmonitor",
    ],
    "context_dependent": [
        "diningtable",
        "sofa",
        "chair",
        "pottedplant",
        "boat",
        "aeroplane",
        "train",
    ],
    "ambiguous": [
        "person",
        "car",
        "bus",
        "bicycle",
        "motorbike",
    ],
}

# ── Data fractions (core experiment) ──────────────────────────────────────────
DATA_FRACTIONS  = [0.05, 0.10, 0.20, 0.50, 1.00]

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS          = 30
EARLY_STOP_PAT  = 5          # patience for early stopping on val mAP
BATCH_SIZE      = 16         # reduce to 8 if memory is tight on CPU
NUM_WORKERS     = 0          # 0 is safest on Windows/CPU
SEEDS           = [42, 7, 21]

# ── Optimizer (differential LRs per model) ────────────────────────────────────
LR_CONFIG = {
    "resnet50":  {"backbone_lr": 1e-4, "head_lr": 1e-3},
    "resnet101": {"backbone_lr": 1e-4, "head_lr": 1e-3},
    "vit_b16":   {"backbone_lr": 1e-5, "head_lr": 1e-3},
    "deit_s":    {"backbone_lr": 1e-5, "head_lr": 1e-3},
}
WEIGHT_DECAY    = 1e-2       # AdamW default

# ── Augmentation ───────────────────────────────────────────────────────────────
# Three policies as specified by Prof. Shakeri
AUG_POLICIES = ["none", "standard", "strong"]

IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Output ─────────────────────────────────────────────────────────────────────
RESULTS_DIR     = "./results"
CHECKPOINTS_DIR = "./checkpoints"
