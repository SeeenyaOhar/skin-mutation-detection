
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from eval.probe_milk10k import CLINICAL, load_tables
from imbalance.labels import HEAD9, MILK_MALIGNANT, TO_HEAD9

N_SPLITS = 5
SEED = 0
INNER_VAL_FRAC = 0.20


def build(milk_root: Path, milk_images: Path, n_splits: int = N_SPLITS,
          seed: int = SEED) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold, train_test_split

    df, _ = load_tables(milk_root, milk_images)
    df["head9"] = df.dx.map(TO_HEAD9)
    df["malignant"] = df.dx.isin(MILK_MALIGNANT).astype(int)

    n_before = df.lesion_id.nunique()
    df = df[df.head9.notna()].reset_index(drop=True)
    print(f"dropped {n_before - df.lesion_id.nunique()} MAL_OTH lesions "
          f"-> {df.lesion_id.nunique()} lesions, {len(df)} images")

    clin = df[df.image_type == CLINICAL].reset_index(drop=True)
    assert clin.lesion_id.is_unique, "more than one clinical view per lesion"
    y_les = np.array([HEAD9.index(c) for c in clin.head9])

    fold = np.empty(len(y_les), dtype=int)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(y_les)), y_les)):
        fold[te] = k
    df["fold"] = df.lesion_id.map(dict(zip(clin.lesion_id, fold)))

    dev = clin[fold != 0]
    y_dev = y_les[fold != 0]
    tr_les, va_les = train_test_split(
        dev.lesion_id.values, test_size=INNER_VAL_FRAC,
        stratify=y_dev, random_state=seed)
    inner = pd.Series("", index=df.index, dtype=object)
    inner[df.lesion_id.isin(set(tr_les))] = "train"
    inner[df.lesion_id.isin(set(va_les))] = "val"
    df["inner"] = inner

    df["y"] = [HEAD9.index(c) for c in df.head9]
    df["rel_path"] = [f"{l}/{i}.jpg" for l, i in zip(df.lesion_id, df.isic_id)]

    cols = ["lesion_id", "isic_id", "rel_path", "image_type", "dx", "head9",
            "y", "malignant", "skin_tone_class", "fold", "inner"]
    return df[cols].copy()


def check(t: pd.DataFrame) -> None:
    per = t.groupby("lesion_id").agg(nf=("fold", "nunique"),
                                     ni=("inner", "nunique"),
                                     nv=("image_type", "nunique"))
    assert (per.nf == 1).all(), "a lesion is split across outer folds"
    assert (per.ni == 1).all(), "a lesion is split across the inner split"
    assert (per.nv == 2).all(), "a lesion is missing one of its two views"

    inner_les = set(t[t.inner != ""].lesion_id)
    fold0_les = set(t[t.fold == 0].lesion_id)
    assert not (inner_les & fold0_les), "inner split leaks into fold 0's test set"
    assert inner_les == set(t[t.fold != 0].lesion_id), "inner split is incomplete"

    for k in sorted(t.fold.unique()):
        seen = t[(t.fold == k) & (t.image_type == CLINICAL)].y.nunique()
        assert seen == len(HEAD9), f"fold {k} holds only {seen}/{len(HEAD9)} classes"
    print("fold integrity: ok")


def fingerprint(t: pd.DataFrame) -> str:
    key = t.sort_values("isic_id")[["isic_id", "y", "fold", "inner"]]
    return hashlib.sha1(key.to_csv(index=False).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milk-root", type=Path, required=True)
    ap.add_argument("--milk-images", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/folds_head9.csv"))
    a = ap.parse_args()

    t = build(a.milk_root, a.milk_images)
    check(t)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False)

    clin = t[t.image_type == CLINICAL]
    print(f"\n{len(t)} images, {t.lesion_id.nunique()} lesions")
    print(f"fingerprint {fingerprint(t)}")
    print("\nlesions per fold and class")
    print(pd.crosstab(clin.head9, clin.fold).reindex(HEAD9))
    print(f"\ninner search split: {(t.inner == 'train').sum() // 2} train "
          f"/ {(t.inner == 'val').sum() // 2} val lesions")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
