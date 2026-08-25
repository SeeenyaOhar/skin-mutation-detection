
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

torch.multiprocessing.set_sharing_strategy("file_system")

from imbalance.labels import HEAD9
from train import data as D
from train import engine as E
from train import model as M
from train import tracking as T

CLINICAL = D.CLINICAL

DEFAULTS = dict(
    split="inner", modality_train="both", size=256,
    backbone="fastvit_sa24.apple_dist_in1k", drop_path=0.05, pretrained=True,
    base_lr=1e-3, layer_decay=0.75, weight_decay=0.05, warmup_epochs=2,
    epochs=40, batch_size=48, label_smoothing=0.0, ema=0.0,
    freeze_epochs=2, unfreeze_mode="all_at_once",
    augment=True, geom_scale=1.0, photo_scale=1.0, mixup_alpha=0.0, loss="ce",
    post_hoc_tau=True,
    seed=0, workers=8, device=None, phase="smoke", treatment="", study="",
    label_space="HEAD9", run_name=None, save_ckpt=False,
)


def pick_device(name=None) -> str:
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tune_backend(dev: str) -> None:
    if not dev.startswith("cuda"):
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def split_masks(table: pd.DataFrame, split: str):
    if split == "inner":
        return table.inner == "train", table.inner == "val"
    assert split.startswith("fold"), split
    k = int(split[4:])
    return table.fold != k, table.fold == k


def run(cfg: dict, table: pd.DataFrame, images_root: Path, *,
        trial=None, mlflow_run=True, out_dir: Path | None = None) -> dict:
    import mlflow

    c = {**DEFAULTS, **cfg}
    dev = pick_device(c["device"])
    tune_backend(dev)
    amp = torch.bfloat16 if dev.startswith("cuda") else None
    torch.manual_seed(c["seed"])
    np.random.seed(c["seed"])

    tr_mask, va_mask = split_masks(table, c["split"])
    if c["modality_train"] == "clinical":
        tr_mask = tr_mask & (table.image_type == CLINICAL)

    net, mean, std = M.build_model(c["backbone"], len(HEAD9), c["drop_path"],
                                   c["pretrained"])
    net = net.to(dev)
    if dev.startswith("cuda"):
        net = net.to(memory_format=torch.channels_last)

    policy = None
    if c["augment"]:
        policy = D.scale_policy(D.base_policy(c["size"]),
                                c["geom_scale"], c["photo_scale"])

    normalize = D.Normalizer(mean, std).to(dev)
    tl, vl, tr_rows, va_rows = D.make_loaders(
        table, images_root, train_mask=tr_mask, val_mask=va_mask,
        policy=policy, batch_size=c["batch_size"], workers=c["workers"],
        seed=c["seed"], mixup_alpha=c["mixup_alpha"], size=c["size"])

    counts = D.class_counts(tr_rows, len(HEAD9))
    loss_fn = E.Loss(counts, mode=c["loss"], label_smoothing=c["label_smoothing"],
                     device=dev)

    opt = M.build_optimizer(net, c["base_lr"], c["weight_decay"], c["layer_decay"])
    freezer = M.FreezeController(net, c["freeze_epochs"], c["unfreeze_mode"])
    ema = M.make_ema(net, c["ema"] or None)

    c = {**c, "n_train_images": int(len(tr_rows)),
         "n_val_lesions": int(len(va_rows)),
         "train_counts": counts.astype(int).tolist(),
         "policy": D.policy_dict(policy) if policy else None,
         "device": dev, "code_sha": T.code_sha(),
         "folds_sha": cfg.get("folds_sha", ""), "fold": c["split"]}

    out_dir = out_dir or (T.ROOT / "reports" / "train_tmp")
    best = {"macro_f1": -1.0}
    status = "FINISHED"

    if mlflow_run:
        mlflow.start_run(run_name=c["run_name"], nested=c.get("nested", False))
    try:
        if mlflow_run:
            T.log_config(c)
        t0 = time.time()
        for ep in range(c["epochs"]):
            state = freezer.apply(ep)
            stats = E.train_one_epoch(
                net, freezer, tl, opt, loss_fn, dev, epoch=ep,
                total_epochs=c["epochs"], warmup=c["warmup_epochs"],
                normalize=normalize, ema=ema, mixup_alpha=c["mixup_alpha"],
                amp_dtype=amp)
            tl.dataset.set_epoch(ep + 1)

            scoring = ema.module if ema is not None else net
            logits, yv, val_loss = E.evaluate(scoring, vl, dev, normalize,
                                              loss_fn=loss_fn, amp_dtype=amp)
            m = E.multiclass_metrics(yv, logits, prefix="val_")
            tau, f1_tau, _ = E.select_tau(yv, logits, counts)
            m["val_macro_f1_tau"] = f1_tau
            m["val_tau"] = tau
            m["val_loss"] = val_loss
            m.update(stats)
            m["n_trainable_params"] = state["n_trainable"]
            objective = f1_tau if c["post_hoc_tau"] else m["val_macro_f1"]

            if mlflow_run:
                T.log_epoch(m, ep)
            print(f"  e{ep:02d} loss {stats['train_loss']:.4f} "
                  f"val_loss {val_loss:.4f} f1 {m['val_macro_f1']:.4f} "
                  f"f1@tau {f1_tau:.4f} acc {m['val_acc']:.4f} "
                  f"trainable {state['n_trainable'] / 1e6:.2f}M "
                  f"[{time.time() - t0:.0f}s]", flush=True)

            if objective > best["macro_f1"]:
                best = {"macro_f1": objective, "epoch": ep, "tau": tau,
                        "logits": logits, "y": yv}

            if trial is not None:
                trial.report(objective, ep)
                if trial.should_prune():
                    import optuna
                    if mlflow_run:
                        mlflow.set_tag("pruned", "true")
                        mlflow.log_metric("pruned_at_epoch", ep)
                        mlflow.log_metric("best_val_macro_f1", best["macro_f1"])
                    status = "KILLED"
                    raise optuna.TrialPruned()

        rep, curve = E.full_report(best["y"], best["logits"], counts, best["tau"])
        rep["best_epoch"] = best["epoch"]
        rep["epoch_seconds"] = (time.time() - t0) / c["epochs"]
        if mlflow_run:
            T.log_final(rep, prefix="test_")
            rd = Path(out_dir) / (mlflow.active_run().info.run_id[:8])
            T.log_predictions(va_rows, best["y"], best["logits"], rd)
            T.log_curves(curve, [{"class": k, "f1": rep[f"f1_{k}"],
                                  "f1_tau": rep[f"tau_f1_{k}"],
                                  "n": int((best["y"] == i).sum())}
                                 for i, k in enumerate(HEAD9)],
                         best["y"], best["logits"], rd, best["tau"])
            (rd / "config.json").write_text(json.dumps(c, indent=2, default=str))
            (rd / "env.txt").write_text(T.env_report())
            mlflow.log_artifact(str(rd / "config.json"))
            mlflow.log_artifact(str(rd / "env.txt"))
            if c["save_ckpt"]:
                torch.save({"model": (ema.module if ema else net).state_dict(),
                            "cfg": c, "tau": best["tau"]}, rd / "best.pt")
                mlflow.log_artifact(str(rd / "best.pt"))
    except BaseException:
        if status == "FINISHED":
            status = "FAILED"
        raise
    finally:
        if mlflow_run:
            mlflow.end_run(status=status)
    return rep


def load_table(folds_csv: Path):
    t = pd.read_csv(folds_csv, keep_default_na=False)
    t["fold"] = t["fold"].astype(int)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=Path, default=Path("data/folds_head9.csv"))
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--experiment", default="smoke")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=None)
        elif v is None:
            ap.add_argument(f"--{k.replace('_', '-')}", default=None)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=None)
    a = ap.parse_args()

    cfg = {k: v for k, v in vars(a).items()
           if k in DEFAULTS and v is not None}
    for k, v in DEFAULTS.items():
        if isinstance(v, bool) and k in cfg:
            cfg[k] = bool(cfg[k])

    from train.folds import fingerprint
    table = load_table(a.folds)
    cfg["folds_sha"] = fingerprint(table)

    T.setup(a.experiment)
    rep = run(cfg, table, a.images)
    print(json.dumps({k: round(v, 4) for k, v in rep.items()
                      if isinstance(v, float)}, indent=2))


if __name__ == "__main__":
    main()
