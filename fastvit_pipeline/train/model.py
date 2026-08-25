
from __future__ import annotations

import timm
import torch
import torch.nn as nn

GROUPS = ["stem", "stages.0", "stages.1", "stages.2", "stages.3", "final_conv"]

PERMANENTLY_FROZEN = {"stem", "stages.0", "stages.1", "stages.2"}
UNLOCKABLE = [g for g in GROUPS if g not in PERMANENTLY_FROZEN]


def build_model(name: str, num_classes: int = 9, drop_path: float = 0.0,
                pretrained: bool = True):
    m = timm.create_model(name, pretrained=pretrained, num_classes=num_classes,
                          drop_path_rate=drop_path)
    cfg = m.pretrained_cfg
    return m, tuple(cfg["mean"]), tuple(cfg["std"])


def _module(model: nn.Module, path: str) -> nn.Module:
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


class FreezeController:

    def __init__(self, model: nn.Module, freeze_epochs: int = 0,
                 mode: str = "all_at_once"):
        assert mode in ("all_at_once", "progressive"), mode
        self.model = model
        self.freeze_epochs = int(freeze_epochs)
        self.mode = mode
        self.frozen: list[nn.Module] = []

    def unfrozen_groups(self, epoch: int) -> list[str]:
        if epoch < self.freeze_epochs:
            return []
        if self.mode == "all_at_once":
            return list(UNLOCKABLE)
        released = epoch - self.freeze_epochs + 1
        return UNLOCKABLE[max(0, len(UNLOCKABLE) - released):]

    def apply(self, epoch: int) -> dict:
        live = set(self.unfrozen_groups(epoch))
        self.frozen = []
        for g in GROUPS:
            mod = _module(self.model, g)
            on = g in live
            for p in mod.parameters():
                p.requires_grad_(on)
            if not on:
                self.frozen.append(mod)
        for p in self.model.head.parameters():
            p.requires_grad_(True)
        return {"unfrozen": sorted(live),
                "n_trainable": count_trainable(self.model)}

    def train_mode(self) -> None:
        self.model.train()
        for mod in self.frozen:
            mod.eval()


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_param_groups(model: nn.Module, base_lr: float, weight_decay: float,
                       layer_decay: float):
    from timm.optim import param_groups_layer_decay

    groups = param_groups_layer_decay(
        model, weight_decay=weight_decay, layer_decay=layer_decay)
    for g in groups:
        g["lr"] = base_lr * g.get("lr_scale", 1.0)
    return groups


def build_optimizer(model, base_lr, weight_decay, layer_decay):
    groups = build_param_groups(model, base_lr, weight_decay, layer_decay)
    opt = torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.999))
    for g, src in zip(opt.param_groups, groups):
        g["base_lr"] = src["lr"]
    return opt


def cosine_lr(opt, epoch_frac: float, warmup: float, total: float,
              min_ratio: float = 0.01) -> float:
    import math
    if epoch_frac < warmup:
        scale = (epoch_frac + 1e-8) / max(warmup, 1e-8)
    else:
        t = (epoch_frac - warmup) / max(total - warmup, 1e-8)
        t = min(max(t, 0.0), 1.0)
        scale = min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * scale
    return scale


def make_ema(model, decay: float | None):
    if not decay:
        return None
    from timm.utils import ModelEmaV3
    return ModelEmaV3(model, decay=decay, use_warmup=True, min_decay=0.0)


def reparameterize(model):
    from timm.utils import reparameterize_model
    return reparameterize_model(model)
