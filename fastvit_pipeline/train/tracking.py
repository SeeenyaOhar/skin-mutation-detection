
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = {
    "smoke": "00-smoke",
    "hpo": "01-hpo-head9",
    "topk": "02-topk-confirm-head9",
    "final": "03-final-cv-head9",
    "imbalance": "04-ablation-imbalance",
    "modality": "05-ablation-modality",
    "external": "06-external-test",
}


def code_sha() -> str:
    h = hashlib.sha1()
    for rel in sorted(["train", "imbalance", "eval"]):
        d = ROOT / rel
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.py")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def env_report() -> str:
    lines = [f"python {sys.version.split()[0]}",
             f"platform {platform.platform()}"]
    try:
        import torch
        lines.append(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"gpu {torch.cuda.get_device_name(0)}")
    except Exception:
        pass
    try:
        import timm
        lines.append(f"timm {timm.__version__}")
    except Exception:
        pass
    try:
        lines.append(subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                    capture_output=True, text=True,
                                    timeout=60).stdout)
    except Exception:
        pass
    return "\n".join(lines)


def setup(experiment: str, tracking_uri: str | None = None):
    import mlflow
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") \
        or f"sqlite:///{ROOT / 'mlruns.db'}"
    mlflow.set_tracking_uri(uri)
    name = EXPERIMENTS.get(experiment, experiment)
    mlflow.set_experiment(name)
    return name


def log_config(cfg: dict, extra_tags: dict | None = None) -> None:
    import mlflow
    flat = {}
    for k, v in cfg.items():
        if isinstance(v, (dict, list, tuple)):
            flat[k] = json.dumps(v, ensure_ascii=False)[:480]
        else:
            flat[k] = v
    mlflow.log_params(flat)
    mlflow.set_tags({
        "phase": cfg.get("phase", ""),
        "backbone": cfg.get("backbone", ""),
        "label_space": cfg.get("label_space", "HEAD9"),
        "modality_train": cfg.get("modality_train", ""),
        "treatment": cfg.get("treatment", ""),
        "study": cfg.get("study", ""),
        "thesis_section": cfg.get("thesis_section", "ch3"),
        **(extra_tags or {}),
    })


def log_epoch(metrics: dict, epoch: int) -> None:
    import mlflow
    clean = {k: float(v) for k, v in metrics.items()
             if isinstance(v, (int, float, np.floating)) and np.isfinite(v)}
    mlflow.log_metrics(clean, step=epoch)


def log_final(metrics: dict, prefix: str = "test_") -> None:
    import mlflow
    clean = {f"{prefix}{k}": float(v) for k, v in metrics.items()
             if isinstance(v, (int, float, np.floating)) and np.isfinite(v)}
    mlflow.log_metrics(clean)


def log_predictions(rows, y, logits, out_dir: Path, name="oof_predictions.csv"):
    import mlflow
    import pandas as pd
    from imbalance.labels import HEAD9

    out_dir.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame({
        "lesion_id": rows.lesion_id.to_numpy(),
        "isic_id": rows.isic_id.to_numpy(),
        "image_type": rows.image_type.to_numpy(),
        "y_true": y,
        "head9_true": [HEAD9[i] for i in y],
    })
    for k, c in enumerate(HEAD9):
        t[f"logit_{c}"] = logits[:, k]
    p = out_dir / name
    t.to_csv(p, index=False)
    mlflow.log_artifact(str(p))
    return p


def log_curves(curve: dict, per_class: dict, y, logits, out_dir: Path,
               tau: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    from imbalance.labels import HEAD9
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(per_class).to_csv(out_dir / "per_class_f1.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(list(curve), list(curve.values()), marker="o")
    ax.axvline(tau, ls="--", c="k", lw=0.8)
    ax.set_xlabel("tau"), ax.set_ylabel("macro F1"), ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tau_curve.png", dpi=110)
    plt.close(fig)

    cm = confusion_matrix(y, logits.argmax(1), labels=range(len(HEAD9)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues",
              vmin=0, vmax=1)
    ax.set_xticks(range(len(HEAD9)), HEAD9, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(HEAD9)), HEAD9, fontsize=8)
    for i in range(len(HEAD9)):
        for j in range(len(HEAD9)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax.set_xlabel("predicted"), ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=110)
    plt.close(fig)

    for f in ("per_class_f1.csv", "tau_curve.png", "confusion_matrix.png"):
        mlflow.log_artifact(str(out_dir / f))


REQUIRED_PARAMS = ["backbone", "base_lr", "layer_decay", "freeze_epochs",
                   "unfreeze_mode", "folds_sha", "code_sha", "seed", "fold"]


def assert_complete(run) -> list[str]:
    missing = [p for p in REQUIRED_PARAMS if p not in run.data.params]
    if "test_macro_f1" not in run.data.metrics:
        missing.append("metric:test_macro_f1")
    return missing
