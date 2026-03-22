# dataset.py — Pascal VOC multi-label DataLoader

import os
import xml.etree.ElementTree as ET
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

from config import (
    VOC_ROOT, VOC_YEAR, VOC_CLASSES, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE, NUM_WORKERS
)


# ── Label parsing ──────────────────────────────────────────────────────────────

def parse_voc_annotation(xml_path: str) -> list[int]:
    """Return a 20-dim binary label vector from a VOC annotation XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    label = [0] * len(VOC_CLASSES)
    for obj in root.iter("object"):
        name = obj.find("name").text.strip().lower()
        if name in VOC_CLASSES:
            label[VOC_CLASSES.index(name)] = 1
    return label


# ── Dataset ────────────────────────────────────────────────────────────────────

class VOCMultiLabel(Dataset):
    """
    Pascal VOC multi-label classification dataset.
    Returns (image_tensor, label_tensor) where label is a 20-dim float vector.
    """

    def __init__(self, root: str, year: str, split: str, transform=None):
        """
        Args:
            root:      path to VOCdevkit/
            year:      "2007" or "2012"
            split:     "train", "val", or "trainval"
            transform: torchvision transforms to apply
        """
        self.root      = root
        self.year      = year
        self.split     = split
        self.transform = transform

        self.img_dir = os.path.join(root, f"VOC{year}", "JPEGImages")
        self.ann_dir = os.path.join(root, f"VOC{year}", "Annotations")

        split_file = os.path.join(
            root, f"VOC{year}", "ImageSets", "Main", f"{split}.txt"
        )
        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

        # Pre-parse all labels (fast, done once at init)
        self.labels = {}
        for img_id in self.ids:
            xml_path = os.path.join(self.ann_dir, f"{img_id}.xml")
            self.labels[img_id] = parse_voc_annotation(xml_path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[img_id], dtype=torch.float32)
        return image, label

    def get_label_matrix(self):
        """Return (N, 20) numpy array of all labels — used for stratification."""
        import numpy as np
        return np.array([self.labels[i] for i in self.ids])


# ── Transforms ─────────────────────────────────────────────────────────────────

def get_transforms(split: str, policy: str = "standard"):
    """
    Args:
        split:  "train" or "val"
        policy: "none" | "standard" | "strong"
    """
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "val":
        # Test-time: deterministic resize + center crop
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            normalize,
        ])

    # Train-time augmentation policies
    if policy == "none":
        return T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            normalize,
        ])

    elif policy == "standard":
        return T.Compose([
            T.RandomResizedCrop(IMAGE_SIZE),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
            normalize,
        ])

    elif policy == "strong":
        return T.Compose([
            T.RandomResizedCrop(IMAGE_SIZE),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            normalize,
        ])

    else:
        raise ValueError(f"Unknown augmentation policy: {policy}")


# ── Stratified subset (multi-label) ────────────────────────────────────────────

def make_stratified_subset(dataset: VOCMultiLabel, fraction: float, seed: int = 42):
    """
    Return a Subset of `dataset` containing approximately `fraction` of the data,
    using iterative stratification to preserve multi-label class balance.

    Requires: pip install scikit-multilearn
    Falls back to random sampling if scikit-multilearn is not installed.
    """
    n = len(dataset)
    n_samples = max(1, int(n * fraction))

    try:
        from skmultilearn.model_selection import IterativeStratification
        import numpy as np

        labels = dataset.get_label_matrix()
        dummy_X = np.arange(n).reshape(-1, 1)

        stratifier = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[1.0 - fraction, fraction],
            # random_state=seed,
        )
        # We only need the second fold (the subset)
        for _, subset_idx in stratifier.split(dummy_X, labels):
            return Subset(dataset, subset_idx.tolist())

    except ImportError:
        print(
            "[dataset.py] scikit-multilearn not found — falling back to random sampling.\n"
            "Install with: pip install scikit-multilearn"
        )
        import random
        random.seed(seed)
        indices = random.sample(range(n), n_samples)
        return Subset(dataset, indices)


# ── DataLoader factory ─────────────────────────────────────────────────────────

def get_dataloaders(fraction: float = 1.0, aug_policy: str = "standard", seed: int = 42):
    """
    Returns (train_loader, val_loader) for a given data fraction and aug policy.
    """
    train_dataset = VOCMultiLabel(
        root=VOC_ROOT, year=VOC_YEAR, split="train",
        transform=get_transforms("train", aug_policy)
    )
    val_dataset = VOCMultiLabel(
        root=VOC_ROOT, year=VOC_YEAR, split="val",
        transform=get_transforms("val")
    )

    if fraction < 1.0:
        train_dataset = make_stratified_subset(train_dataset, fraction, seed)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=False
    )

    print(f"[dataset] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    return train_loader, val_loader
