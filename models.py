# models.py — All four model architectures with correct multi-label heads

import torch
import torch.nn as nn
import torchvision.models as tvm
import timm

from config import NUM_CLASSES, LR_CONFIG


# ── Model builders ─────────────────────────────────────────────────────────────

def build_resnet50(pretrained: bool = True) -> nn.Module:
    weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_resnet101(pretrained: bool = True) -> nn.Module:
    weights = tvm.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.resnet101(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_vit_b16(pretrained: bool = True) -> nn.Module:
    # Use augreg_in21k checkpoint as recommended by Prof. Shakeri
    model_name = "vit_base_patch16_224.augreg_in21k" if pretrained else "vit_base_patch16_224"
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=NUM_CLASSES)
    return model


def build_deit_s(pretrained: bool = True) -> nn.Module:
    model = timm.create_model(
        "deit_small_patch16_224", pretrained=pretrained, num_classes=NUM_CLASSES
    )
    return model


# ── Registry ───────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "resnet50":  build_resnet50,
    "resnet101": build_resnet101,
    "vit_b16":   build_vit_b16,
    "deit_s":    build_deit_s,
}


def get_model(name: str, pretrained: bool = True) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](pretrained=pretrained)


# ── Optimizer with differential learning rates ─────────────────────────────────

def get_optimizer(model: nn.Module, model_name: str) -> torch.optim.Optimizer:
    """
    AdamW with differential LRs: lower for pretrained backbone, higher for head.
    Head parameter groups are identified by their name ending with 'fc' or 'head'.
    """
    cfg = LR_CONFIG[model_name]
    backbone_lr = cfg["backbone_lr"]
    head_lr     = cfg["head_lr"]

    head_param_names = {"fc", "head"}  # covers ResNet (.fc) and ViT/DeiT (.head)

    head_params     = []
    backbone_params = []

    for name, param in model.named_parameters():
        # Check if this param belongs to the classification head
        top_module = name.split(".")[0]
        if top_module in head_param_names:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ]

    return torch.optim.AdamW(param_groups, weight_decay=1e-2)


# ── Quick model summary ────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in MODEL_REGISTRY:
        m = get_model(name, pretrained=False)
        print(f"{name:12s}  params: {count_parameters(m)/1e6:.1f}M")
