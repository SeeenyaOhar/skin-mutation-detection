
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score

from imbalance.labels import RAW11, MILK_MALIGNANT
from train import model as M
from train.data import eval_view, Normalizer, _to_tensor

MAL_IDX = [i for i, c in enumerate(RAW11) if c in MILK_MALIGNANT]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


@torch.no_grad()
def infer(ckpt_path: Path, rows: pd.DataFrame, device: str, size: int = 256,
          batch_size: int = 64):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    net, mean, std = M.build_model(cfg["backbone"], len(RAW11), pretrained=False)
    net.load_state_dict(ckpt["model"])
    net.to(device).eval()
    normalize = Normalizer(mean, std).to(device)

    logits = np.zeros((len(rows), len(RAW11)), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        chunk = rows.iloc[start:start + batch_size]
        imgs = [eval_view(Image.open(p).convert("RGB"), size) for p in chunk.path]
        x = torch.stack([_to_tensor(im) for im in imgs]).to(device)
        x = normalize(x)
        logits[start:start + len(chunk)] = net(x).cpu().numpy()
    return logits, float(ckpt["tau"]), cfg


def logit_adjust(logits, counts, tau):
    prior = counts / counts.sum()
    return logits - tau * np.log(np.clip(prior, 1e-12, None))


def per_class_metrics(y_true, y_pred, probs):
    rows = {}
    for i, cls in enumerate(RAW11):
        true_bin = (y_true == i).astype(int)
        pred_bin = (y_pred == i).astype(int)
        n = int(true_bin.sum())
        if n == 0:
            continue
        tp = int(((pred_bin == 1) & (true_bin == 1)).sum())
        fp = int(((pred_bin == 1) & (true_bin == 0)).sum())
        fn = int(((pred_bin == 0) & (true_bin == 1)).sum())
        tn = int(((pred_bin == 0) & (true_bin == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        try:
            auroc = float(roc_auc_score(true_bin, probs[:, i])) if 0 < n < len(y_true) else float("nan")
        except ValueError:
            auroc = float("nan")
        rows[cls] = dict(n=n, precision=prec, recall=rec, specificity=spec, f1=f1, auroc=auroc)
    return rows


def binary_metrics(true, score):
    from sklearn.metrics import roc_curve
    pred = (score >= 0.5).astype(int)
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(true)
    bal_acc = (rec + spec) / 2
    prev = float(true.mean())
    auroc = float(roc_auc_score(true, score)) if 0 < prev < 1 else float("nan")
    fpr, tpr, _ = roc_curve(true, score)
    idx = np.searchsorted(tpr, 0.95)
    spec_at_95 = float(1 - fpr[idx]) if idx < len(fpr) else float("nan")
    return dict(sensitivity=rec, specificity=spec, ppv=prec, npv=npv, f1=f1,
                accuracy=acc, balanced_acc=bal_acc, auroc=auroc,
                spec_at_95_sens=spec_at_95, always_malignant_f1=2 * prev / (prev + 1),
                prevalence=prev, n=len(true))


def eval_split(logits, tau, train_counts, rows):
    from sklearn.metrics import confusion_matrix, roc_curve

    adj = logit_adjust(logits, train_counts, tau)
    probs = softmax(adj)
    y_pred = adj.argmax(1)
    y_true = rows.y.to_numpy()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = per_class_metrics(y_true, y_pred, probs)
    acc = float((y_pred == y_true).mean())
    mal_score = probs[:, MAL_IDX].sum(1)
    mal_true = np.isin(y_true, MAL_IDX).astype(int)
    binm = binary_metrics(mal_true, mal_score)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(RAW11))))
    fpr, tpr, _ = roc_curve(mal_true, mal_score)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int))
    roc = dict(fpr=fpr[idx].tolist(), tpr=tpr[idx].tolist())

    return dict(accuracy=acc, macro_f1=float(macro_f1), per_class=per_class,
                binary=binm, confusion_matrix=cm.tolist(), roc=roc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-before", type=Path, required=True)
    ap.add_argument("--ckpt-after", type=Path, required=True)
    ap.add_argument("--pool-before", type=Path, default=Path("data/pool_raw11_rebuilt.csv"))
    ap.add_argument("--pool-after", type=Path, default=Path("data/pool_raw11_full_synth.csv"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("reports/pool_clinical_vs_derm.json"))
    a = ap.parse_args()

    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    pool = pd.read_csv(a.pool_before)
    pool["y"] = pool.raw11.map({c: i for i, c in enumerate(RAW11)})
    te = pool[pool.split == "test"]
    clin = te[te.image_type == "clinical: close-up"].reset_index(drop=True)
    derm = te[te.image_type == "dermoscopic"].reset_index(drop=True)
    print(f"test: {len(clin)} clinical, {len(derm)} dermoscopic (same {clin.lesion_id.nunique()} lesions)")

    out = {}
    for key, ckpt, pool_path in (("before_synthetic", a.ckpt_before, a.pool_before),
                                  ("after_synthetic", a.ckpt_after, a.pool_after)):
        p = pd.read_csv(pool_path)
        p["y"] = p.raw11.map({c: i for i, c in enumerate(RAW11)})
        train_counts = np.bincount(p[p.split == "train"].y.to_numpy(), minlength=len(RAW11)).astype(float)
        out[key] = {}
        for mod, rows in (("clinical", clin), ("dermoscopic", derm)):
            logits, tau, cfg = infer(ckpt, rows, device)
            res = eval_split(logits, tau, train_counts, rows)
            out[key][mod] = res
            print(f"\n== {key} / {mod} == (tau={tau})")
            print(f"  accuracy={res['accuracy']:.4f}  macro_f1={res['macro_f1']:.4f}")
            for cls, m in res["per_class"].items():
                print(f"    {cls:8s} n={m['n']:4d} P={m['precision']:.3f} R={m['recall']:.3f} "
                      f"Spec={m['specificity']:.3f} F1={m['f1']:.3f} AUROC={m['auroc']:.3f}")
            b = res["binary"]
            print(f"  binary: sens={b['sensitivity']:.3f} spec={b['specificity']:.3f} "
                  f"f1={b['f1']:.3f} acc={b['accuracy']:.3f} bal_acc={b['balanced_acc']:.3f} "
                  f"auroc={b['auroc']:.3f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
