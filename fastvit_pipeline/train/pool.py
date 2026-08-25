
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from imbalance.labels import RAW11
from imbalance.pool_labels import (DERM7PT_TO_RAW11, DERM12345_TO_RAW11,
                                   HIBA_TO_RAW11, ISIC2019_TO_RAW11)

DL = Path.home() / "Downloads"
COLS = ["dataset", "lesion_id", "isic_id", "path", "image_type", "raw11", "y",
        "split", "fold"]


def milk10k(root: Path, seed: int = 0, n_splits: int = 5,
            inner_frac: float = 0.10) -> pd.DataFrame:
    gt = pd.read_csv(DL / "MILK10k/MILK10k_Training_GroundTruth.csv")
    meta = pd.read_csv(DL / "MILK10k/MILK10k_Training_Metadata.csv")
    gt["raw11"] = gt[RAW11].idxmax(1)

    df = meta.merge(gt[["lesion_id", "raw11"]], on="lesion_id", how="inner")
    df["path"] = df.apply(
        lambda r: str(root / r.lesion_id / f"{r.isic_id}.jpg"), axis=1)
    df = df[df.path.map(lambda p: Path(p).exists())].reset_index(drop=True)

    les = df.drop_duplicates("lesion_id")[["lesion_id", "raw11"]].reset_index(drop=True)
    y = les.raw11.map(RAW11.index).to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.zeros(len(les), dtype=int)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(les)), y)):
        fold[te] = k
    les["fold"] = fold

    tr = les[les.fold != 0]
    vc = tr.raw11.value_counts()
    strat_ok = tr[tr.raw11.isin(vc[vc >= 2].index)]
    _, val_les = train_test_split(strat_ok.lesion_id, test_size=inner_frac,
                                  stratify=strat_ok.raw11, random_state=seed)
    val = set(val_les)

    les["split"] = np.where(les.fold == 0, "test",
                            np.where(les.lesion_id.isin(val), "val", "train"))
    out = df.merge(les[["lesion_id", "fold", "split"]], on="lesion_id")
    out["dataset"] = "milk10k"
    return out


def isic2019() -> pd.DataFrame:
    gt = pd.read_csv(DL / "ISIC_2019_Training_GroundTruth.csv")
    cls = [c for c in gt.columns if c != "image"]
    gt["native"] = gt[cls].idxmax(1)
    gt["raw11"] = gt.native.map(ISIC2019_TO_RAW11)
    gt = gt.dropna(subset=["raw11"])
    root = DL / "ISIC_2019_Training_Input"
    gt["path"] = gt.image.map(lambda i: str(root / f"{i}.jpg"))
    gt = gt[gt.path.map(lambda p: Path(p).exists())]
    return pd.DataFrame({
        "dataset": "isic2019", "lesion_id": gt.image, "isic_id": gt.image,
        "path": gt.path, "image_type": "dermoscopic", "raw11": gt.raw11})


def derm12345(meta_csv: Path, images_root: Path) -> pd.DataFrame:
    d = pd.read_csv(meta_csv)
    d["raw11"] = d.diagnosis_3.map(DERM12345_TO_RAW11)
    d = d.dropna(subset=["raw11"])
    d["path"] = d.isic_id.map(lambda i: str(images_root / f"{i}.jpg"))
    d = d[d.path.map(lambda p: Path(p).exists())]
    return pd.DataFrame({
        "dataset": "derm12345", "lesion_id": d.isic_id, "isic_id": d.isic_id,
        "path": d.path, "image_type": d.image_type, "raw11": d.raw11})


def derm7pt(meta_csv: Path, images_root: Path) -> pd.DataFrame:
    d = pd.read_csv(meta_csv)
    d["raw11"] = d.diagnosis.map(DERM7PT_TO_RAW11)
    d = d.dropna(subset=["raw11"])
    frames = []
    for col, itype in (("clinic", "clinical: close-up"), ("derm", "dermoscopic")):
        sub = d[[col, "case_num", "raw11"]].copy()
        sub["path"] = sub[col].map(lambda r: str(images_root / r))
        sub = sub[sub.path.map(lambda p: Path(p).exists())]
        frames.append(pd.DataFrame({
            "dataset": "derm7pt",
            "lesion_id": "d7pt_" + sub.case_num.astype(str),
            "isic_id": "d7pt_" + sub.case_num.astype(str) + "_" + itype[:4],
            "path": sub.path, "image_type": itype, "raw11": sub.raw11}))
    return pd.concat(frames, ignore_index=True)


def hiba() -> pd.DataFrame:
    d = pd.read_csv(DL / "HIBA/metadata.csv")
    d["raw11"] = d.diagnosis.map(HIBA_TO_RAW11)
    d = d.dropna(subset=["raw11"])
    root = DL / "HIBA/images"
    d["path"] = d.isic_id.map(lambda i: str(root / f"{i}.jpg"))
    d = d[d.path.map(lambda p: Path(p).exists())]
    return pd.DataFrame({
        "dataset": "hiba", "lesion_id": d.lesion_id.fillna(d.isic_id),
        "isic_id": d.isic_id, "path": d.path,
        "image_type": d.image_type, "raw11": d.raw11})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milk-images", type=Path, required=True)
    ap.add_argument("--derm12345-meta", type=Path, required=True)
    ap.add_argument("--derm12345-images", type=Path, required=True)
    ap.add_argument("--derm7pt-meta", type=Path, required=True)
    ap.add_argument("--derm7pt-images", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/pool_raw11.csv"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    milk = milk10k(a.milk_images, seed=a.seed)
    print(f"milk10k: {len(milk)} images, {milk.lesion_id.nunique()} lesions")
    print(milk.drop_duplicates('lesion_id').split.value_counts().to_string())

    ext = []
    for name, frame in (("isic2019", isic2019()),
                        ("derm12345", derm12345(a.derm12345_meta, a.derm12345_images)),
                        ("derm7pt", derm7pt(a.derm7pt_meta, a.derm7pt_images)),
                        ("hiba", hiba())):
        print(f"{name}: {len(frame)} images")
        ext.append(frame)
    ext = pd.concat(ext, ignore_index=True)

    overlap = set(milk.isic_id) & set(ext.isic_id)
    if overlap:
        print(f"dropping {len(overlap)} external images that are MILK10k ids")
        ext = ext[~ext.isic_id.isin(overlap)]
    ext["split"], ext["fold"] = "train", -1

    pool = pd.concat([milk[COLS[:6] + ["split", "fold"]] if False else
                      milk.assign(dataset="milk10k")[
                          ["dataset", "lesion_id", "isic_id", "path",
                           "image_type", "raw11", "split", "fold"]],
                      ext[["dataset", "lesion_id", "isic_id", "path",
                           "image_type", "raw11", "split", "fold"]]],
                     ignore_index=True)
    pool["y"] = pool.raw11.map(RAW11.index)
    pool = pool[COLS]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(a.out, index=False)

    print(f"\npooled: {len(pool)} images")
    print(pd.crosstab(pool.raw11, pool.split).reindex(RAW11).to_string())
    print("\nby dataset (train rows only):")
    print(pool[pool.split == 'train'].dataset.value_counts().to_string())
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
