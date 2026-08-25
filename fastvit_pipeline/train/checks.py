
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from imbalance.labels import HEAD9
from train import data as D
from train import engine as E
from train import model as M

OK, BAD = "  ok  ", " FAIL "
_results: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        def wrapped(*a, **k):
            try:
                msg = fn(*a, **k) or ""
                _results.append((name, True, msg))
                print(f"[{OK}] {name}  {msg}")
            except AssertionError as e:
                _results.append((name, False, str(e)))
                print(f"[{BAD}] {name}  {e}")
            except Exception as e:
                _results.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"[{BAD}] {name}  {type(e).__name__}: {e}")
        return wrapped
    return deco



@check("folds: no lesion straddles a split")
def check_folds(table):
    from train.folds import check as fold_check, fingerprint
    fold_check(table)
    return f"fingerprint {fingerprint(table)}"


@check("folds: train and val sets are disjoint at lesion level")
def check_disjoint(table):
    from train.run_train import split_masks
    for split in ["inner"] + [f"fold{k}" for k in sorted(table.fold.unique())]:
        tr, va = split_masks(table, split)
        a = set(table[tr].lesion_id)
        b = set(table[va].lesion_id)
        assert not (a & b), f"{split}: {len(a & b)} lesions in both sides"
    return f"{table.fold.nunique() + 1} splits clean"


@check("freeze: n_trainable follows the schedule exactly")
def check_freeze_schedule():
    net, _, _ = M.build_model("fastvit_sa12", pretrained=False)
    total = sum(p.numel() for p in net.parameters())
    head = sum(p.numel() for p in net.head.parameters())
    ceiling = head + sum(p.numel() for g in M.UNLOCKABLE
                         for p in M._module(net, g).parameters())
    assert ceiling < total, "permanent floor has no effect -- regression"

    f = M.FreezeController(net, freeze_epochs=2, mode="all_at_once")
    assert f.apply(0)["n_trainable"] == head, "epoch 0 should be head-only"
    assert f.apply(1)["n_trainable"] == head, "epoch 1 should be head-only"
    assert f.apply(2)["n_trainable"] == ceiling, "epoch 2 should hit the ceiling"

    g = M.FreezeController(net, freeze_epochs=1, mode="progressive")
    assert g.apply(0)["n_trainable"] == head, "progressive epoch 0 is head-only"
    seq = [g.apply(e)["n_trainable"] for e in range(1, 8)]
    assert seq == sorted(seq), f"progressive unfreezing is not monotone: {seq}"
    assert seq[-1] == ceiling, "progressive never reaches the ceiling"
    assert max(seq) < total, "progressive breached the permanent floor"
    assert g.unfrozen_groups(1) == ["final_conv"], g.unfrozen_groups(1)
    assert g.unfrozen_groups(2) == ["stages.3", "final_conv"], g.unfrozen_groups(2)
    return (f"head {head / 1e3:.1f}k -> ceiling {ceiling / 1e6:.2f}M "
           f"({ceiling / total:.1%} of {total / 1e6:.2f}M), deepest-first, "
           f"{total - ceiling:>4} params ({1 - ceiling / total:.1%}) permanently frozen")


@check("freeze: a frozen backbone does not drift in train mode")
def check_frozen_bn():
    def probe(net, x):
        net.eval()
        with torch.no_grad():
            return net.forward_features(x).clone()

    def buffers(net):
        return torch.cat([b.flatten() for n, b in net.named_buffers()
                          if "running_" in n])

    torch.manual_seed(0)
    x = torch.randn(4, 3, 256, 256)
    noise = [torch.randn(4, 3, 256, 256) for _ in range(3)]

    ctrl, _, _ = M.build_model("fastvit_sa12", pretrained=False)
    before, b0 = probe(ctrl, x), buffers(ctrl).clone()
    ctrl.train()
    with torch.no_grad():
        for n in noise:
            ctrl.forward_features(n)
    assert (buffers(ctrl) - b0).abs().max() > 0, "control: no BN buffers moved"
    assert (probe(ctrl, x) - before).abs().max() > 0, "control: no visible drift"

    net, _, _ = M.build_model("fastvit_sa12", pretrained=False)
    a, k0 = probe(net, x), buffers(net).clone()
    f = M.FreezeController(net, freeze_epochs=10, mode="all_at_once")
    f.apply(0)
    f.train_mode()
    with torch.no_grad():
        for n in noise:
            net.forward_features(n)
    db = (buffers(net) - k0).abs().max().item()
    assert db == 0.0, f"frozen BatchNorm buffers moved by {db:.3e}"
    dz = (probe(net, x) - a).abs().max().item()
    assert dz == 0.0, f"frozen features moved by {dz:.3e}"
    return "guarded: 0.0 buffer and 0.0 feature drift; control drifts"


@check("optimizer: layer-wise LR decay is monotone and covers every parameter")
def check_llrd():
    net, _, _ = M.build_model("fastvit_sa24", pretrained=False)
    opt = M.build_optimizer(net, base_lr=1e-3, weight_decay=0.05, layer_decay=0.75)
    seen = sum(len(g["params"]) for g in opt.param_groups)
    total = len([p for p in net.parameters() if p.requires_grad])
    assert seen == total, f"{total - seen} parameters missing from the optimizer"
    lrs = [g["base_lr"] for g in opt.param_groups]
    assert min(lrs) < max(lrs), "layer decay had no effect"
    assert abs(max(lrs) - 1e-3) < 1e-9, f"deepest group is not base_lr: {max(lrs)}"
    return f"{len(opt.param_groups)} groups, lr {min(lrs):.2e} … {max(lrs):.2e}"


@check("optimizer: params frozen at epoch 0 are still in the optimizer")
def check_late_unfreeze():
    net, _, _ = M.build_model("fastvit_sa12", pretrained=False)
    opt = M.build_optimizer(net, 1e-3, 0.05, 0.75)
    f = M.FreezeController(net, freeze_epochs=3, mode="all_at_once")
    f.apply(0)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    stage3 = [p for p in M._module(net, "stages.3").parameters()]
    assert all(id(p) in in_opt for p in stage3), "stage3 is absent from the optimizer"
    assert not any(p.requires_grad for p in stage3), "stage3 should be frozen at e0"
    f.apply(3)
    assert all(p.requires_grad for p in stage3), "stage3 never unfroze"

    stem = [p for p in net.stem.parameters()]
    f.apply(100)
    assert not any(p.requires_grad for p in stem), "permanent floor was breached"
    return "stage3 present while frozen, trainable after release; stem stays frozen forever"


@check("policy: search bounds cannot exceed the measured limits")
def check_policy_bounds():
    base = D.base_policy(256)
    off = D.scale_policy(base, 1.0, 0.0)
    assert off.p_colour == off.p_blur == off.p_jpeg == 0.0, "photo_scale=0 leaks"
    assert off.p_hflip == 0.5, "photo_scale must not touch geometry"
    full = D.scale_policy(base, 1.0, 1.0)
    for k in ("p_colour", "p_blur", "p_jpeg", "p_downscale", "p_noise"):
        assert getattr(full, k) == getattr(base, k), f"{k} drifted at scale 1.0"
    for k in ("jpeg_quality", "noise_std", "downscale", "blur_sigma",
              "brightness", "contrast", "saturation", "hue"):
        assert getattr(full, k) == getattr(base, k), f"magnitude {k} was scaled"
    wide = D.scale_policy(base, 1.25, 1.0)
    assert wide.crop_scale_min < base.crop_scale_min, "geom_scale had no effect"
    assert D.scale_policy(base, 9.0, 1.0).crop_scale_min >= 0.35, "crop unbounded"
    return f"crop {wide.crop_scale_min:.3f} … {off.crop_scale_min:.3f}, caps fixed"


@check("data: eval view is deterministic, train view is not")
def check_dataset(table, images):
    rows = table[table.fold == 0].head(8)
    ds_e = D.LesionDataset(rows, images, train=False)
    a, b = ds_e[0]["x"], ds_e[0]["x"]
    assert torch.equal(a, b), "eval view is not deterministic"
    assert a.shape == (3, 256, 256), a.shape

    ds_t = D.LesionDataset(rows, images, train=True,
                           policy=D.base_policy(256), seed=0)
    c = ds_t[0]["x"]
    ds_t.set_epoch(1)
    d = ds_t[0]["x"]
    assert not torch.equal(c, d), "augmentation repeats across epochs"
    ds_t.set_epoch(0)
    assert torch.equal(ds_t[0]["x"], c), "augmentation is not reproducible"
    return "eval stable, train varies per epoch, reproducible by seed"


@check("data: Balanced-MixUp partner stream is class-uniform")
def check_partner(table, images):
    rows = table[(table.fold != 0)]
    ds = D.LesionDataset(rows, images, train=True,
                         policy=D.base_policy(256), seed=0, mixup_alpha=0.2)
    rng_counts = np.zeros(len(HEAD9))
    for i in range(3000):
        r = np.random.default_rng([0, 0, i])
        r.integers(0, 1)
        cls = ds.by_class[np.random.default_rng([0, 0, i]).integers(len(ds.by_class))]
        rng_counts[ds.y[cls[0]]] += 1
    share = rng_counts / rng_counts.sum()
    assert share.min() > 0.06, f"partner stream is not uniform: {share.round(3)}"
    nat = np.bincount(ds.y, minlength=len(HEAD9)) / len(ds.y)
    return (f"partner min {share.min():.3f} vs natural min {nat.min():.4f} "
            f"(uniform target {1 / len(HEAD9):.3f})")


@check("metrics: binary derivation and the always-malignant floor")
def check_metrics():
    comp = {"BCC": 40, "SCCKA": 15, "MEL": 10, "AKIEC": 5,
            "NV": 12, "BKL": 8, "DF": 5, "VASC": 3, "OTH_BEN": 2}
    y = np.array([HEAD9.index(c) for c, n in comp.items() for _ in range(n)])
    assert len(y) == 100 and len(set(y)) == len(HEAD9)
    perfect = np.full((100, len(HEAD9)), -10.0)
    perfect[np.arange(100), y] = 10.0

    m = E.binary_from_multiclass(y, perfect)
    assert abs(m["bin_f1"] - 1.0) < 1e-9, f"bin_f1 {m['bin_f1']}"
    assert abs(m["bin_prevalence"] - 0.70) < 1e-9, f"prev {m['bin_prevalence']}"
    floor = m["bin_always_malignant_f1"]
    assert abs(floor - 2 * .7 / 1.7) < 1e-9, f"floor {floor}"

    mm = E.multiclass_metrics(y, perfect)
    assert abs(mm["macro_f1"] - 1.0) < 1e-9, f"macro_f1 {mm['macro_f1']}"
    assert abs(mm["macro_f1_6"] - 1.0) < 1e-9, f"macro_f1_6 {mm['macro_f1_6']}"

    wrecked = perfect.copy()
    tail = np.isin(y, [HEAD9.index("DF"), HEAD9.index("VASC")])
    wrecked[tail] = -10.0
    wrecked[tail, HEAD9.index("NV")] = 10.0
    w = E.multiclass_metrics(y, wrecked)
    assert w["macro_f1"] < 0.85, f"9-class did not notice the tail: {w['macro_f1']}"
    assert w["macro_f1_6"] > 0.95, f"6-class was contaminated: {w['macro_f1_6']}"
    return (f"floor {floor:.3f} at prevalence 0.70; wrecking DF+VASC costs "
            f"macro_f1 {1 - w['macro_f1']:.3f} but macro_f1_6 only "
            f"{1 - w['macro_f1_6']:.3f}")


@check("metrics: tau=0 leaves the prediction untouched")
def check_tau():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, len(HEAD9)))
    counts = np.array([450, 746, 2522, 303, 544, 52, 47, 473, 94], float)
    from imbalance import augment as A
    assert np.array_equal(A.logit_adjust(logits, counts, 0.0).argmax(1),
                          logits.argmax(1)), "tau=0 changed the argmax"
    adj = A.logit_adjust(logits, counts, 1.0)
    moved = (adj.argmax(1) != logits.argmax(1)).mean()
    assert moved > 0, "tau=1 changed nothing"
    return f"tau=1 moves {moved:.1%} of predictions towards the tail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=Path, default=Path("data/folds_head9.csv"))
    ap.add_argument("--images", type=Path, required=True)
    a = ap.parse_args()

    from train.run_train import load_table
    table = load_table(a.folds)

    check_folds(table)
    check_disjoint(table)
    check_freeze_schedule()
    check_frozen_bn()
    check_llrd()
    check_late_unfreeze()
    check_policy_bounds()
    check_dataset(table, a.images)
    check_partner(table, a.images)
    check_metrics()
    check_tau()

    bad = [n for n, ok, _ in _results if not ok]
    print(f"\n{len(_results) - len(bad)}/{len(_results)} passed")
    if bad:
        print("failed: " + ", ".join(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
