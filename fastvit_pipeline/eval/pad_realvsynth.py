
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score

from imbalance.labels import HEAD9, PAD_TO_HEAD9, PAD_MALIGNANT
from train import model as M
from train.data import eval_view, Normalizer, _to_tensor

MAL_IDX = [i for i, c in enumerate(HEAD9) if c in PAD_MALIGNANT]
PAD_IDX = sorted(HEAD9.index(v) for v in PAD_TO_HEAD9.values())
PAD_CLASSES = [HEAD9[i] for i in PAD_IDX]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load_pad(root: Path) -> pd.DataFrame:
    m = pd.read_csv(root / "metadata.csv")
    m = m[m.diagnostic.isin(PAD_TO_HEAD9)].copy()
    m["head9"] = m.diagnostic.map(PAD_TO_HEAD9)
    m["y"] = m.head9.map(HEAD9.index)
    m["malignant"] = m.diagnostic.isin(PAD_MALIGNANT).astype(int)

    def find(img_id):
        for part in ("imgs_part_1", "imgs_part_2", "imgs_part_3"):
            p = root / part / part / img_id
            if p.exists():
                return str(p)
        return None
    m["path"] = m.img_id.map(find)
    missing = m.path.isna().sum()
    if missing:
        print(f"warning: {missing} images not found on disk, dropped")
    return m.dropna(subset=["path"]).reset_index(drop=True)


@torch.no_grad()
def infer(ckpt_path: Path, df: pd.DataFrame, device: str, size: int = 256,
          batch_size: int = 64):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    net, mean, std = M.build_model(cfg["backbone"], len(HEAD9), pretrained=False)
    net.load_state_dict(ckpt["model"])
    net.to(device).eval()
    normalize = Normalizer(mean, std).to(device)

    logits = np.zeros((len(df), len(HEAD9)), dtype=np.float32)
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        imgs = [eval_view(Image.open(p).convert("RGB"), size) for p in chunk.path]
        x = torch.stack([_to_tensor(im) for im in imgs]).to(device)
        x = normalize(x)
        logits[start:start + len(chunk)] = net(x).cpu().numpy()
    return logits, float(ckpt["tau"])


def logit_adjust(logits, counts, tau):
    prior = counts / counts.sum()
    return logits - tau * np.log(np.clip(prior, 1e-12, None))


def per_class_metrics(y_true, y_pred, probs):
    rows = {}
    for i in PAD_IDX:
        cls = HEAD9[i]
        true_bin = (y_true == i).astype(int)
        pred_bin = (y_pred == i).astype(int)
        n = int(true_bin.sum())
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


def run_one(ckpt_path, folds_path, df, device):
    logits, tau = infer(ckpt_path, df, device)
    table = pd.read_csv(folds_path, keep_default_na=False)
    counts = np.bincount(table[table.fold != 0].y.values, minlength=len(HEAD9))
    adj = logit_adjust(logits, counts, tau)
    probs = softmax(adj)
    y_pred = adj.argmax(1)
    mal_score = probs[:, MAL_IDX].sum(1)

    macro_f1 = f1_score(df.y, y_pred, labels=PAD_IDX, average="macro", zero_division=0)
    per_class = per_class_metrics(df.y.to_numpy(), y_pred, probs)
    macro_precision = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_recall = float(np.mean([v["recall"] for v in per_class.values()]))
    macro_specificity = float(np.mean([v["specificity"] for v in per_class.values()]))
    macro_auroc = float(np.nanmean([v["auroc"] for v in per_class.values()]))
    acc = float((y_pred == df.y.to_numpy()).mean())

    binm = binary_metrics(df.malignant.to_numpy(), mal_score)
    return dict(tau=tau, headline=dict(accuracy=acc, macro_f1=macro_f1,
                macro_precision=macro_precision, macro_sensitivity=macro_recall,
                macro_specificity=macro_specificity, macro_auroc=macro_auroc),
                per_class=per_class, binary=binm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-real", type=Path, required=True)
    ap.add_argument("--folds-real", type=Path, required=True)
    ap.add_argument("--ckpt-synth", type=Path, required=True)
    ap.add_argument("--folds-synth", type=Path, required=True)
    ap.add_argument("--pad-root", type=Path, default=Path.home() / "Downloads/skin-cancer")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("reports/pad_realvsynth.json"))
    a = ap.parse_args()

    device = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    df = load_pad(a.pad_root)
    print(f"PAD lesions usable: {len(df)} (6 of 9 head classes present)")
    print(df.head9.value_counts().to_string(), "\n")

    out = {}
    for key, ckpt, folds in (("real_only_best", a.ckpt_real, a.folds_real),
                             ("real_synth_best", a.ckpt_synth, a.folds_synth)):
        print(f"== {key} ==")
        res = run_one(ckpt, folds, df, device)
        out[key] = res
        print(json.dumps(res["headline"], indent=2))
        for cls, m in res["per_class"].items():
            print(f"  {cls:8s} n={m['n']:4d} P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"Spec={m['specificity']:.3f} F1={m['f1']:.3f} AUROC={m['auroc']:.3f}")
        print(json.dumps(res["binary"], indent=2))
        print()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
