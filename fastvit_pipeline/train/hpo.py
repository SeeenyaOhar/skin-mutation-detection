
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna

from train import run_train as R
from train import tracking as T

BACKBONES = [
    "fastvit_sa12.apple_dist_in1k",
    "fastvit_sa24.apple_dist_in1k",
    "fastvit_mci1.apple_mclip",
]

SEEDS = [
    dict(backbone="fastvit_sa24.apple_dist_in1k", base_lr=3e-4, layer_decay=0.75,
         freeze_epochs=2, unfreeze_mode="all_at_once", weight_decay=0.05,
         warmup_epochs=2, batch_size=48, drop_path=0.05, label_smoothing=0.05,
         ema="0.9998", geom_scale=1.0, photo_scale=1.0, mixup_alpha=0.0,
         loss="ce"),
    dict(backbone="fastvit_mci1.apple_mclip", base_lr=5e-4, layer_decay=0.80,
         freeze_epochs=1, unfreeze_mode="progressive", weight_decay=0.05,
         warmup_epochs=2, batch_size=48, drop_path=0.05, label_smoothing=0.05,
         ema="0.999", geom_scale=1.0, photo_scale=1.0, mixup_alpha=0.2,
         loss="logit_adjusted"),
    dict(backbone="fastvit_sa12.apple_dist_in1k", base_lr=1e-3, layer_decay=0.65,
         freeze_epochs=0, unfreeze_mode="all_at_once", weight_decay=0.02,
         warmup_epochs=1, batch_size=96, drop_path=0.0, label_smoothing=0.0,
         ema="off", geom_scale=1.15, photo_scale=0.5, mixup_alpha=0.1,
         loss="ce"),
]


def suggest(trial, backbones) -> dict:
    ema = trial.suggest_categorical("ema", ["off", "0.999", "0.9998"])
    return dict(
        backbone=trial.suggest_categorical("backbone", backbones),
        base_lr=trial.suggest_float("base_lr", 1e-4, 1e-2, log=True),
        layer_decay=trial.suggest_float("layer_decay", 0.55, 1.0),
        freeze_epochs=trial.suggest_categorical("freeze_epochs", [0, 1, 2, 3, 5]),
        unfreeze_mode=trial.suggest_categorical(
            "unfreeze_mode", ["all_at_once", "progressive"]),
        weight_decay=trial.suggest_float("weight_decay", 1e-4, 0.2, log=True),
        warmup_epochs=trial.suggest_categorical("warmup_epochs", [1, 2, 3]),
        batch_size=trial.suggest_categorical("batch_size", [48, 96]),
        drop_path=trial.suggest_float("drop_path", 0.0, 0.25),
        label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.15),
        ema=0.0 if ema == "off" else float(ema),
        geom_scale=trial.suggest_float("geom_scale", 0.75, 1.25),
        photo_scale=trial.suggest_float("photo_scale", 0.0, 1.0),
        mixup_alpha=trial.suggest_categorical("mixup_alpha", [0.0, 0.1, 0.2, 0.3]),
        loss=trial.suggest_categorical("loss", ["ce", "logit_adjusted"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=Path, default=Path("data/folds_head9.csv"))
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--study-name", default="hpo-v1")
    ap.add_argument("--storage", default=None,
                    help="default sqlite:///optuna.db next to the package")
    ap.add_argument("--n-trials", type=int, default=90)
    ap.add_argument("--timeout-h", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--min-resource", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--modality-train", default="both")
    ap.add_argument("--backbones", default=",".join(BACKBONES))
    ap.add_argument("--no-seeds", action="store_true")
    a = ap.parse_args()

    import mlflow

    backbones = [b for b in a.backbones.split(",") if b]
    storage = a.storage or f"sqlite:///{T.ROOT / 'optuna.db'}"

    table = R.load_table(a.folds)
    from train.folds import fingerprint
    folds_sha = fingerprint(table)

    study = optuna.create_study(
        study_name=a.study_name, storage=storage, load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=0),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=a.min_resource, max_resource=a.epochs,
            reduction_factor=3),
    )
    done = len(study.trials)
    if done == 0 and not a.no_seeds:
        usable = [s for s in SEEDS if s["backbone"] in backbones]
        for s in usable:
            study.enqueue_trial(s)
        skipped = len(SEEDS) - len(usable)
        print(f"enqueued {len(usable)} seed configurations"
              + (f" ({skipped} skipped: backbone not in --backbones)"
                 if skipped else ""))
    print(f"study '{a.study_name}' at {storage}: {done} trials already recorded")

    T.setup("hpo")
    parent = mlflow.start_run(run_name=a.study_name)
    mlflow.set_tags({"phase": "E1", "study": a.study_name,
                     "folds_sha": folds_sha, "code_sha": T.code_sha()})
    mlflow.log_params({"n_trials": a.n_trials, "epochs": a.epochs,
                       "backbones": ",".join(backbones),
                       "modality_train": a.modality_train,
                       "min_resource": a.min_resource,
                       "search_split": "inner (fold0 train, 80/20)"})
    t_start = time.time()

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest(trial, backbones)
        cfg.update(
            split="inner", modality_train=a.modality_train, epochs=a.epochs,
            workers=a.workers, augment=True, seed=0, phase="E1",
            study=a.study_name, folds_sha=folds_sha, nested=True,
            run_name=f"trial-{trial.number:03d}-{cfg['backbone'].split('.')[0][8:]}",
        )
        print(f"\n=== trial {trial.number} :: {cfg['run_name']} ===")
        print(json.dumps({k: v for k, v in cfg.items()
                          if k in dict(trial.params) or k == "ema"}, default=str))
        rep = R.run(cfg, table, a.images, trial=trial)
        for k in ("acc", "macro_f1", "macro_f1_6", "bal_acc", "tau",
                  "tau_macro_f1", "tau_bin_f1", "epoch_seconds"):
            if k in rep:
                trial.set_user_attr(k, float(rep[k]))
        return float(rep["tau_macro_f1"])

    try:
        study.optimize(objective, n_trials=a.n_trials,
                       timeout=a.timeout_h * 3600 if a.timeout_h else None,
                       gc_after_trial=True,
                       catch=(RuntimeError,))
    finally:
        report(study, a, t_start)
        mlflow.end_run()


def report(study, a, t_start) -> None:
    import mlflow
    states = {}
    for t in study.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    print(f"\nstates: {states}   wall {(time.time() - t_start) / 3600:.2f} h")

    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        print("no completed trials")
        return

    best = study.best_trial
    print(f"\nbest trial {best.number}: tau_macro_f1 {best.value:.4f}")
    print(json.dumps(best.params, indent=2))
    mlflow.log_metrics({"best_value": float(best.value),
                        "n_complete": len(complete),
                        "n_pruned": states.get("PRUNED", 0),
                        "wall_hours": (time.time() - t_start) / 3600})
    mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})

    out = T.ROOT / "reports" / "hpo"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for t in study.trials:
        rows.append({"number": t.number, "state": t.state.name,
                     "value": t.value, **t.params, **t.user_attrs})
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("value", ascending=False,
                                        na_position="last")
    df.to_csv(out / f"{a.study_name}_trials.csv", index=False)

    top = df[df.state == "COMPLETE"].head(5)
    (out / f"{a.study_name}_top5.json").write_text(
        json.dumps([{k: v for k, v in r.items()
                     if k not in ("state", "number", "value")}
                    for _, r in top.iterrows()], indent=2, default=str))
    print(f"\ntop 5 by inner-split tau_macro_f1:")
    cols = [c for c in ("number", "value", "backbone", "base_lr", "layer_decay",
                        "freeze_epochs", "unfreeze_mode", "mixup_alpha",
                        "photo_scale", "loss") if c in top.columns]
    print(top[cols].to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        from optuna.visualization.matplotlib import (
            plot_optimization_history, plot_param_importances)
        for fn, name in ((plot_optimization_history, "history"),
                         (plot_param_importances, "importance")):
            ax = fn(study)
            ax.figure.tight_layout()
            ax.figure.savefig(out / f"{a.study_name}_{name}.png", dpi=110)
            mlflow.log_artifact(str(out / f"{a.study_name}_{name}.png"))
    except Exception as e:
        print(f"[warn] plots skipped: {e}")

    for f in (f"{a.study_name}_trials.csv", f"{a.study_name}_top5.json"):
        mlflow.log_artifact(str(out / f))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
