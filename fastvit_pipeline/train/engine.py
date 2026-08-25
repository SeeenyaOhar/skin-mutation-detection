
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from imbalance import augment as A
from imbalance.labels import HEAD9

MAL9 = [HEAD9.index(c) for c in ("MEL", "BCC", "AKIEC", "SCCKA")]

ANALYSABLE = ["MEL", "NV", "BCC", "AKIEC", "BKL", "SCCKA"]
ANALYSABLE_IDX = [HEAD9.index(c) for c in ANALYSABLE]

TAUS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25]


def autocast(device: str, dtype):
    import contextlib
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.split(":")[0], dtype=dtype)



class Loss:

    def __init__(self, counts, mode="ce", label_smoothing=0.0, device="cpu",
                 tau_train=1.0):
        assert mode in ("ce", "logit_adjusted", "weighted"), mode
        self.mode = mode
        self.ls = label_smoothing
        pi = np.asarray(counts, float)
        pi = pi / pi.sum()
        self.log_pi = torch.tensor(np.log(pi), dtype=torch.float32,
                                   device=device)[None, :]
        self.tau_train = tau_train
        w = (1.0 / np.maximum(counts, 1))
        self.weight = torch.tensor(w / w.mean(), dtype=torch.float32,
                                   device=device)

    def __call__(self, logits, target, reduction="mean"):
        if self.mode == "logit_adjusted":
            logits = logits + self.tau_train * self.log_pi
        w = self.weight if self.mode == "weighted" else None
        return F.cross_entropy(logits, target, weight=w,
                               label_smoothing=self.ls, reduction=reduction)



def train_one_epoch(model, freezer, loader, opt, loss_fn, device, *,
                    epoch, total_epochs, warmup, normalize, scaler=None,
                    ema=None, mixup_alpha=0.0, amp_dtype=torch.bfloat16,
                    log_every=0):
    from train.model import cosine_lr

    freezer.train_mode()
    n_steps = max(len(loader), 1)
    running, seen = 0.0, 0

    for step, batch in enumerate(loader):
        lr_scale = cosine_lr(opt, epoch + step / n_steps, warmup, total_epochs)
        x = normalize(batch["x"].to(device, non_blocking=True))
        y = batch["y"].to(device, non_blocking=True)

        if mixup_alpha > 0 and "xb" in batch:
            xb = normalize(batch["xb"].to(device, non_blocking=True))
            yb = batch["yb"].to(device, non_blocking=True)
            lam = torch.from_numpy(
                np.random.default_rng([epoch, step]).beta(
                    mixup_alpha, 1.0, size=len(x)).astype("float32")
            ).to(device)
            x = lam[:, None, None, None] * xb + (1 - lam[:, None, None, None]) * x
        else:
            lam, yb = None, None

        opt.zero_grad(set_to_none=True)
        with autocast(device, amp_dtype):
            out = model(x)
            if lam is None:
                loss = loss_fn(out, y)
            else:
                li = loss_fn(out, y, reduction="none")
                lb = loss_fn(out, yb, reduction="none")
                loss = (lam * lb + (1 - lam) * li).mean()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        if ema is not None:
            ema.update(model, step=epoch * n_steps + step)

        running += float(loss.detach()) * len(x)
        seen += len(x)
        if log_every and step % log_every == 0:
            print(f"    e{epoch} s{step}/{n_steps} loss {running / max(seen,1):.4f}",
                  flush=True)

    return {"train_loss": running / max(seen, 1),
            "lr": opt.param_groups[0]["lr"], "lr_scale": lr_scale}


@torch.no_grad()
def evaluate(model, loader, device, normalize, loss_fn=None,
            amp_dtype=torch.bfloat16):
    model.eval()
    logits, ys = [], []
    loss_sum, n = 0.0, 0
    for batch in loader:
        x = normalize(batch["x"].to(device, non_blocking=True))
        y = batch["y"].to(device, non_blocking=True)
        with autocast(device, amp_dtype):
            out = model(x)
            if loss_fn is not None:
                loss_sum += float(loss_fn(out, y).detach()) * len(y)
                n += len(y)
        logits.append(out.float().cpu().numpy())
        ys.append(batch["y"].numpy())
    val_loss = loss_sum / n if n else float("nan")
    return np.concatenate(logits), np.concatenate(ys), val_loss



def _softmax(s):
    s = np.asarray(s, dtype=float)
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def multiclass_metrics(y, logits, labels=HEAD9, prefix=""):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, precision_score, recall_score,
                                 roc_auc_score)
    pred = np.asarray(logits).argmax(1)
    prob = _softmax(logits)
    out = {
        f"{prefix}acc": float(accuracy_score(y, pred)),
        f"{prefix}macro_f1": float(f1_score(y, pred, average="macro",
                                            zero_division=0)),
        f"{prefix}bal_acc": float(balanced_accuracy_score(y, pred)),
        f"{prefix}weighted_f1": float(f1_score(y, pred, average="weighted",
                                               zero_division=0)),
        f"{prefix}macro_f1_6": float(f1_score(y, pred, labels=ANALYSABLE_IDX,
                                              average="macro", zero_division=0)),
    }
    aurocs = []
    for k, name in enumerate(labels):
        pos = y == k
        if 0 < pos.sum() < len(y):
            aurocs.append(float(roc_auc_score(pos, prob[:, k])))
        out[f"{prefix}f1_{name}"] = float(f1_score(y, pred, labels=[k],
                                                   average="micro",
                                                   zero_division=0))
        out[f"{prefix}recall_{name}"] = float(recall_score(y, pred, labels=[k],
                                                           average="micro",
                                                           zero_division=0))
        out[f"{prefix}precision_{name}"] = float(
            precision_score(y, pred, labels=[k], average="micro",
                            zero_division=0))
    out[f"{prefix}macro_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
    return out


def binary_from_multiclass(y, logits, prefix="bin_"):
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    prob = _softmax(logits)
    score = prob[:, MAL9].sum(1)
    pred = (score >= 0.5).astype(int)
    true = np.isin(y, MAL9).astype(int)

    prev = true.mean()
    floor = 2 * prev / (prev + 1)

    spec95 = float("nan")
    order = np.argsort(-score)
    tp = np.cumsum(true[order])
    if true.sum():
        need = np.searchsorted(tp, 0.95 * true.sum())
        if need < len(order):
            thr = score[order][need]
            neg = true == 0
            spec95 = float((score[neg] < thr).mean()) if neg.any() else float("nan")

    return {
        f"{prefix}acc": float(accuracy_score(true, pred)),
        f"{prefix}precision": float(precision_score(true, pred, zero_division=0)),
        f"{prefix}recall": float(recall_score(true, pred, zero_division=0)),
        f"{prefix}f1": float(f1_score(true, pred, zero_division=0)),
        f"{prefix}auroc": float(roc_auc_score(true, score)) if 0 < true.mean() < 1
        else float("nan"),
        f"{prefix}specificity_at_recall95": spec95,
        f"{prefix}always_malignant_f1": float(floor),
        f"{prefix}prevalence": float(prev),
    }


def select_tau(y, logits, counts, taus=TAUS):
    from sklearn.metrics import f1_score
    best, curve = (0.0, -1.0), {}
    for t in taus:
        pred = A.logit_adjust(logits, counts, t).argmax(1)
        f1 = float(f1_score(y, pred, average="macro", zero_division=0))
        curve[t] = f1
        if f1 > best[1]:
            best = (t, f1)
    return best[0], best[1], curve


def full_report(y, logits, counts, tau=None):
    out = multiclass_metrics(y, logits, prefix="")
    out.update(binary_from_multiclass(y, logits, prefix="bin_"))
    if tau is None:
        tau, _, curve = select_tau(y, logits, counts)
    else:
        _, _, curve = select_tau(y, logits, counts)
    adj = A.logit_adjust(logits, counts, tau)
    out.update(multiclass_metrics(y, adj, prefix="tau_"))
    out.update(binary_from_multiclass(y, adj, prefix="tau_bin_"))
    out["tau"] = float(tau)
    return out, curve
