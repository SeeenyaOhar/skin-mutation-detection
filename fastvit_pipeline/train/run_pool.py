
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

torch.multiprocessing.set_sharing_strategy("file_system")

from imbalance.labels import MILK_MALIGNANT, RAW11
from train import data as D
from train import engine as E
from train import model as M
from train import tracking as T

MAL_IDX = [i for i, c in enumerate(RAW11) if c in MILK_MALIGNANT]

TRIAL003 = dict(
    backbone="fastvit_mci1.apple_mclip", base_lr=0.0007501954443620128,
    layer_decay=0.9512978503519359, weight_decay=0.00019391724470856566,
    warmup_epochs=2, batch_size=96, drop_path=0.1997896410541809,
    label_smoothing=0.06922190433793977, ema=0.999, freeze_epochs=0,
    unfreeze_mode="all_at_once", geom_scale=1.1402645881432276,
    photo_scale=0.11827442586893322, mixup_alpha=0.2, loss="ce", seed=0,
)


def binary_block(y, logits, prefix="bin_"):
    from sklearn.metrics import roc_auc_score
    prob = E._softmax(logits)
    score = prob[:, MAL_IDX].sum(1)
    true = np.isin(y, MAL_IDX).astype(int)
    pred = (score >= 0.5).astype(int)

    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    prev = float(true.mean())

    spec95 = float("nan")
    order = np.argsort(-score)
    hits = np.cumsum(true[order])
    if true.sum():
        need = np.searchsorted(hits, 0.95 * true.sum())
        if need < len(order):
            thr = score[order][need]
            neg = true == 0
            spec95 = float((score[neg] < thr).mean()) if neg.any() else float("nan")

    return {
        f"{prefix}f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        f"{prefix}precision": prec, f"{prefix}recall": rec,
        f"{prefix}specificity": tn / (tn + fp) if tn + fp else 0.0,
        f"{prefix}acc": (tp + tn) / len(true),
        f"{prefix}auroc": float(roc_auc_score(true, score)) if 0 < prev < 1 else float("nan"),
        f"{prefix}specificity_at_recall95": spec95,
        f"{prefix}always_malignant_f1": 2 * prev / (prev + 1),
        f"{prefix}prevalence": prev, f"{prefix}fn": fn,
    }


def save_atomic(payload: dict, path: Path) -> None:
    import os
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def checkpoint(net, ema, opt, ep, cfg, tau, best, mean, std) -> dict:
    return {
        "model": (ema.module if ema is not None else net).state_dict(),
        "model_live": net.state_dict(),
        "ema": ema.module.state_dict() if ema is not None else None,
        "opt": opt.state_dict(),
        "epoch": ep, "cfg": cfg, "tau": tau, "best": best,
        "labels": RAW11, "mean": mean, "std": std,
        "code_sha": T.code_sha(),
    }


def report(y, logits, counts, tau):
    out = E.multiclass_metrics(y, logits, labels=RAW11, prefix="")
    out.update(binary_block(y, logits, prefix="bin_"))
    from imbalance import augment as A
    adj = A.logit_adjust(logits, counts, tau)
    out.update(E.multiclass_metrics(y, adj, labels=RAW11, prefix="tau_"))
    out.update(binary_block(y, adj, prefix="tau_bin_"))
    out["tau"] = float(tau)
    return out


def loaders(pool: pd.DataFrame, cfg: dict, workers: int):
    from torch.utils.data import DataLoader
    pool = pool.assign(rel_path=pool.path)
    tr = pool[pool.split == "train"]
    va = pool[(pool.split == "val") & (pool.image_type == D.CLINICAL)]
    te = pool[(pool.split == "test") & (pool.image_type == D.CLINICAL)]

    policy = D.scale_policy(D.base_policy(256), cfg["geom_scale"], cfg["photo_scale"])
    mk = lambda rows, train: D.LesionDataset(
        rows, Path("/"), train=train, policy=policy if train else None,
        size=256, seed=cfg["seed"],
        mixup_alpha=cfg["mixup_alpha"] if train else 0.0)

    common = dict(num_workers=workers, pin_memory=True,
                  persistent_workers=workers > 0,
                  prefetch_factor=4 if workers > 0 else None)
    tl = DataLoader(mk(tr, True), batch_size=cfg["batch_size"], shuffle=True,
                    drop_last=True, **common)
    vl = DataLoader(mk(va, False), batch_size=128, shuffle=False, **common)
    tel = DataLoader(mk(te, False), batch_size=128, shuffle=False, **common)
    return tl, vl, tel, tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=Path("data/pool_raw11.csv"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--experiment", default="07-pooled-raw11")
    ap.add_argument("--run-name", default="pooled-trial003")
    ap.add_argument("--out", type=Path, default=Path("reports/pooled_raw11"))
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--warmup-epochs", type=int, default=None,
                    help="override trial003's warmup_epochs=2")
    ap.add_argument("--freeze-epochs", type=int, default=None,
                    help="override trial003's freeze_epochs=0")
    ap.add_argument("--unfreeze-mode", choices=["all_at_once", "progressive"],
                    default=None, help="override trial003's all_at_once")
    a = ap.parse_args()

    import mlflow
    cfg = {**TRIAL003, "epochs": a.epochs, "label_space": "RAW11"}
    for k, v in (("warmup_epochs", a.warmup_epochs),
                ("freeze_epochs", a.freeze_epochs),
                ("unfreeze_mode", a.unfreeze_mode)):
        if v is not None:
            cfg[k] = v
    dev = a.device or ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if dev.startswith("cuda") else None
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    a.out.mkdir(parents=True, exist_ok=True)

    pool = pd.read_csv(a.pool)
    tl, vl, tel, tr_rows, va_rows, te_rows = loaders(pool, cfg, a.workers)
    counts = D.class_counts(tr_rows, len(RAW11))
    print(f"train {len(tr_rows)}  val {len(va_rows)}  test {len(te_rows)} "
          f"| device {dev} | {len(tl)} steps/epoch")

    net, mean, std = M.build_model(cfg["backbone"], len(RAW11), cfg["drop_path"],
                                   pretrained=True)
    net = net.to(dev)
    normalize = D.Normalizer(mean, std).to(dev)
    loss_fn = E.Loss(counts, mode=cfg["loss"],
                     label_smoothing=cfg["label_smoothing"], device=dev)
    opt = M.build_optimizer(net, cfg["base_lr"], cfg["weight_decay"],
                            cfg["layer_decay"])
    freezer = M.FreezeController(net, cfg["freeze_epochs"], cfg["unfreeze_mode"])
    ema = M.make_ema(net, cfg["ema"] or None)

    start_ep = 0
    best = {"macro_f1": -1.0, "epoch": -1, "tau": 0.0}
    if a.resume and a.resume.exists():
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model_live"])
        opt.load_state_dict(ck["opt"])
        if ema is not None and ck.get("ema"):
            ema.module.load_state_dict(ck["ema"])
        start_ep = ck["epoch"] + 1
        best = ck.get("best", best)
        print(f"resumed from {a.resume} at epoch {start_ep} "
              f"(best so far {best['macro_f1']:.4f} @ e{best['epoch']})")

    T.setup(a.experiment)
    mlflow.start_run(run_name=a.run_name)
    try:
        T.log_config({**cfg, "device": dev, "n_train": len(tr_rows),
                      "n_val": len(va_rows), "n_test": len(te_rows),
                      "train_counts": counts.astype(int).tolist(),
                      "datasets": tr_rows.dataset.value_counts().to_dict(),
                      "code_sha": T.code_sha()})
        t0 = time.time()
        for ep in range(start_ep, cfg["epochs"]):
            freezer.apply(ep)
            stats = E.train_one_epoch(
                net, freezer, tl, opt, loss_fn, dev, epoch=ep,
                total_epochs=cfg["epochs"], warmup=cfg["warmup_epochs"],
                normalize=normalize, ema=ema, mixup_alpha=cfg["mixup_alpha"],
                amp_dtype=amp, log_every=100)
            tl.dataset.set_epoch(ep + 1)

            scoring = ema.module if ema is not None else net
            logits, yv, vloss = E.evaluate(scoring, vl, dev, normalize,
                                           loss_fn=loss_fn, amp_dtype=amp)
            m = E.multiclass_metrics(yv, logits, labels=RAW11, prefix="val_")
            tau, f1_tau, _ = E.select_tau(yv, logits, counts)
            m.update({"val_macro_f1_tau": f1_tau, "val_tau": tau,
                      "val_loss": vloss, **stats})
            T.log_epoch(m, ep)
            print(f"  e{ep:02d} loss {stats['train_loss']:.4f} "
                  f"val_loss {vloss:.4f} f1 {m['val_macro_f1']:.4f} "
                  f"f1@tau {f1_tau:.4f} acc {m['val_acc']:.4f} "
                  f"[{(time.time()-t0)/60:.1f} min]", flush=True)

            if f1_tau > best["macro_f1"]:
                best = {"macro_f1": f1_tau, "epoch": ep, "tau": tau}
            ck = checkpoint(net, ema, opt, ep, cfg, tau, best, mean, std)
            save_atomic(ck, a.out / "last.pt")
            if best["epoch"] == ep:
                save_atomic(ck, a.out / "best.pt")
                print(f"    new best {f1_tau:.4f} -> best.pt", flush=True)

        ck = torch.load(a.out / "best.pt", map_location=dev, weights_only=False)
        eval_net, _, _ = M.build_model(cfg["backbone"], len(RAW11), 0.0,
                                       pretrained=False)
        eval_net.load_state_dict(ck["model"])
        eval_net.to(dev)
        tlogits, ty, tloss = E.evaluate(eval_net, tel, dev, normalize,
                                        loss_fn=loss_fn, amp_dtype=amp)
        rep = report(ty, tlogits, counts, ck["tau"])
        rep["best_epoch"] = ck["best"]["epoch"]
        rep["test_loss"] = tloss
        T.log_final(rep, prefix="test_")

        pred = pd.DataFrame({"lesion_id": te_rows.lesion_id.values,
                             "isic_id": te_rows.isic_id.values,
                             "image_type": te_rows.image_type.values,
                             "y_true": ty, "raw11_true": [RAW11[i] for i in ty]})
        for i, c in enumerate(RAW11):
            pred[f"logit_{c}"] = tlogits[:, i]
        pred.to_csv(a.out / "test_predictions.csv", index=False)
        (a.out / "test_report.json").write_text(json.dumps(rep, indent=2))

        v = torch.load(a.out / "best.pt", map_location="cpu", weights_only=False)
        vnet, _, _ = M.build_model(cfg["backbone"], len(RAW11), 0.0,
                                   pretrained=False)
        vnet.load_state_dict(v["model"])
        vnet.to(dev).eval()
        one = next(iter(tel))
        with torch.no_grad():
            xa = normalize(one["x"].to(dev))
            ra = eval_net(xa).float().cpu().numpy()
            rb = vnet(xa).float().cpu().numpy()
        assert np.allclose(ra, rb, atol=1e-4), "best.pt does not reproduce logits"
        print(f"checkpoint verified: {a.out / 'best.pt'} reloads and reproduces "
              f"logits (epoch {v['best']['epoch']}, tau {v['tau']})")
        for f in ("best.pt", "last.pt", "test_predictions.csv", "test_report.json"):
            mlflow.log_artifact(str(a.out / f))

        print(json.dumps({k: round(v_, 4) for k, v_ in rep.items()
                          if isinstance(v_, float)}, indent=2))
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()
