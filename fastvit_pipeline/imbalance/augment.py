
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter



@dataclass
class Policy:

    size: int | None = 256

    crop_scale_min: float = 0.70
    crop_scale_max: float = 1.00
    crop_ratio: tuple = (0.90, 1.11)
    rotate_deg: float = 180.0
    p_hflip: float = 0.5
    p_vflip: float = 0.5

    brightness: float = 0.20
    contrast: float = 0.20
    saturation: float = 0.20
    hue: float = 0.02
    p_colour: float = 0.80

    p_blur: float = 0.30
    blur_sigma: tuple = (0.1, 1.0)
    p_jpeg: float = 0.30
    jpeg_quality: tuple = (70, 95)
    p_downscale: float = 0.20
    downscale: tuple = (1.1, 1.5)
    p_noise: float = 0.20
    noise_std: float = 0.01

    mixup_alpha: float = 0.2

    logit_tau: float = 1.0



def _largest_inner_rect(w, h, deg):
    a = abs(math.radians(deg)) % math.pi
    if a > math.pi / 2:
        a = math.pi - a
    sin_a, cos_a = abs(math.sin(a)), abs(math.cos(a))
    long_first = w >= h
    s_long, s_short = (w, h) if long_first else (h, w)
    if s_short <= 2.0 * sin_a * cos_a * s_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * s_short
        wr, hr = (x / sin_a, x / cos_a) if long_first else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return max(8, int(wr)), max(8, int(hr))


def op_rotate(img, deg, rng):
    a = rng.uniform(-deg, deg)
    w, h = img.size
    cw, ch = _largest_inner_rect(w, h, a)
    t = math.radians(a)
    cos, sin = math.cos(t), -math.sin(t)
    c = w / 2 - (cos * cw / 2 + sin * ch / 2)
    f = h / 2 - (-sin * cw / 2 + cos * ch / 2)
    return img.transform((cw, ch), Image.AFFINE, (cos, sin, c, -sin, cos, f),
                         resample=Image.BICUBIC)


def op_crop(img, smin, smax, ratio, rng):
    w, h = img.size
    m = min(w, h)
    f = math.sqrt(rng.uniform(smin, smax))
    ar = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
    ch = max(8, min(h, int(round(f * m))))
    cw = max(8, min(w, int(round(ch * ar))))
    x = int(round(rng.beta(2, 2) * (w - cw)))
    y = int(round(rng.beta(2, 2) * (h - ch)))
    return img.crop((x, y, x + cw, y + ch))


def op_colour(img, b, c, s, hu, rng):
    from PIL import ImageEnhance
    order = rng.permutation(4)
    for k in order:
        if k == 0 and b:
            img = ImageEnhance.Brightness(img).enhance(1 + rng.uniform(-b, b))
        elif k == 1 and c:
            img = ImageEnhance.Contrast(img).enhance(1 + rng.uniform(-c, c))
        elif k == 2 and s:
            img = ImageEnhance.Color(img).enhance(1 + rng.uniform(-s, s))
        elif k == 3 and hu:
            h, sat, v = img.convert("HSV").split()
            shift = int(round(rng.uniform(-hu, hu) * 255))
            h = h.point(lambda p, d=shift: (p + d) % 256)
            img = Image.merge("HSV", (h, sat, v)).convert("RGB")
    return img


def op_blur(img, sigma, rng):
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(*sigma)))


def op_jpeg(img, q, rng):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(rng.integers(q[0], q[1] + 1)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def op_downscale(img, f, rng):
    k = rng.uniform(*f)
    w, h = img.size
    small = img.resize((max(8, int(w / k)), max(8, int(h / k))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def op_noise(img, std, rng):
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = a + rng.normal(0, std, a.shape).astype(np.float32)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def augment(img: Image.Image, p: Policy, rng) -> Image.Image:
    img = op_rotate(img, p.rotate_deg, rng)
    img = op_crop(img, p.crop_scale_min, p.crop_scale_max, p.crop_ratio, rng)
    if rng.random() < p.p_hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < p.p_vflip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if rng.random() < p.p_colour:
        img = op_colour(img, p.brightness, p.contrast, p.saturation, p.hue, rng)
    if rng.random() < p.p_blur:
        img = op_blur(img, p.blur_sigma, rng)
    if rng.random() < p.p_downscale:
        img = op_downscale(img, p.downscale, rng)
    if rng.random() < p.p_jpeg:
        img = op_jpeg(img, p.jpeg_quality, rng)
    if rng.random() < p.p_noise:
        img = op_noise(img, p.noise_std, rng)
    return img if p.size is None else img.resize((p.size, p.size), Image.BICUBIC)



def balanced_mixup(x_inst, y_inst, x_bal, y_bal, alpha, rng):
    lam = rng.beta(alpha, 1.0, size=(len(x_inst), 1, 1, 1)).astype("float32")
    x = lam * x_bal + (1.0 - lam) * x_inst
    return x, y_inst, y_bal, lam.reshape(-1)


def expected_lambda(alpha: float) -> float:
    return alpha / (alpha + 1.0)


def effective_prior(counts, alpha: float):
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum()
    k = len(counts)
    el = expected_lambda(alpha)
    return (1 - el) * p + el / k



def sampler_weights(labels, mode="instance", power=0.5):
    labels = np.asarray(labels)
    _, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    if mode == "instance":
        w = np.ones(len(labels))
    elif mode == "balanced":
        w = 1.0 / cnt[inv]
    elif mode == "sqrt":
        w = cnt[inv] ** (-power)
    else:
        raise ValueError(mode)
    return w / w.sum()


def logit_adjust(logits, train_counts, tau=1.0):
    pi = np.asarray(train_counts, dtype=float)
    pi = pi / pi.sum()
    return np.asarray(logits) - tau * np.log(pi)[None, :]
