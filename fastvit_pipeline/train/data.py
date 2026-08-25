
from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from imbalance import augment as A

CLINICAL = "clinical: close-up"



def scale_policy(base: A.Policy, geom_scale: float, photo_scale: float) -> A.Policy:
    keep = 1.0 - (1.0 - base.crop_scale_min) * geom_scale
    return replace(
        base,
        crop_scale_min=float(np.clip(keep, 0.35, 0.98)),
        p_colour=base.p_colour * photo_scale,
        p_blur=base.p_blur * photo_scale,
        p_jpeg=base.p_jpeg * photo_scale,
        p_downscale=base.p_downscale * photo_scale,
        p_noise=base.p_noise * photo_scale,
    )


def policy_dict(p: A.Policy) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(p).items()}



def _to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(
        np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).contiguous()


class Normalizer:

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().div_(255.0).sub_(self.mean).div_(self.std)


def eval_view(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    left, top = (w - m) // 2, (h - m) // 2
    return img.crop((left, top, left + m, top + m)).resize((size, size), Image.BICUBIC)



class LesionDataset(Dataset):

    def __init__(self, rows, images_root: Path, *, train: bool,
                 policy: A.Policy | None = None, size: int = 256,
                 seed: int = 0, mixup_alpha: float = 0.0):
        self.paths = [str(Path(images_root) / r) for r in rows.rel_path]
        self.y = rows.y.to_numpy(dtype=np.int64)
        self.train = train
        self.policy = policy
        self.size = size
        self.seed = seed
        self.epoch = 0

        self.mixup_alpha = mixup_alpha
        if train and mixup_alpha > 0:
            self.by_class = [np.flatnonzero(self.y == c)
                             for c in range(int(self.y.max()) + 1)]
            self.by_class = [c for c in self.by_class if len(c)]

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, i: int, rng) -> torch.Tensor:
        img = Image.open(self.paths[i]).convert("RGB")
        if self.train and self.policy is not None:
            img = A.augment(img, self.policy, rng)
            if img.size != (self.size, self.size):
                img = img.resize((self.size, self.size), Image.BICUBIC)
        else:
            img = eval_view(img, self.size)
        return _to_tensor(img)

    def __getitem__(self, i: int):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        out = {"x": self._load(i, rng), "y": int(self.y[i]), "idx": i}
        if self.train and self.mixup_alpha > 0:
            cls = self.by_class[rng.integers(len(self.by_class))]
            j = int(cls[rng.integers(len(cls))])
            out["xb"] = self._load(j, np.random.default_rng(
                [self.seed, self.epoch, 7_000_000 + i]))
            out["yb"] = int(self.y[j])
        return out


def make_loaders(table, images_root, *, train_mask, val_mask,
                 policy, batch_size, workers, seed, mixup_alpha,
                 size=256, val_clinical_only=True):
    from torch.utils.data import DataLoader

    tr_rows = table[train_mask]
    va_rows = table[val_mask]
    if val_clinical_only:
        va_rows = va_rows[va_rows.image_type == CLINICAL]

    tr = LesionDataset(tr_rows, images_root, train=True, policy=policy,
                       size=size, seed=seed, mixup_alpha=mixup_alpha)
    va = LesionDataset(va_rows, images_root, train=False, size=size)

    common = dict(num_workers=workers, pin_memory=True,
                  persistent_workers=workers > 0,
                  prefetch_factor=4 if workers > 0 else None)
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True,
                    **common)
    vl = DataLoader(va, batch_size=max(batch_size, 128), shuffle=False, **common)
    return tl, vl, tr_rows, va_rows


def class_counts(rows, n_classes: int) -> np.ndarray:
    return np.bincount(rows.y.to_numpy(), minlength=n_classes).astype(float)


def base_policy(size: int = 256) -> A.Policy:
    return copy.deepcopy(A.Policy(size=size))
