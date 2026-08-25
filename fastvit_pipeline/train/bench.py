
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import data as D
from train.run_train import load_table, split_masks

GPU_CEILING = {"fastvit_sa12": 1037, "fastvit_sa24": 624, "fastvit_mci1": 449}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=Path, default=Path("data/folds_head9.csv"))
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--workers", default="12,24,40")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--split", default="inner")
    ap.add_argument("--batches", type=int, default=25)
    a = ap.parse_args()

    table = load_table(a.folds)
    tr, va = split_masks(table, a.split)
    policy = D.scale_policy(D.base_policy(256), 1.0, 1.0)

    print(f"split {a.split}, batch {a.batch_size}, mixup {a.mixup_alpha} "
          f"({'2' if a.mixup_alpha > 0 else '1'} image loads per sample)")
    for w in [int(x) for x in a.workers.split(",")]:
        tl, _, tr_rows, _ = D.make_loaders(
            table, a.images, train_mask=tr, val_mask=va, policy=policy,
            batch_size=a.batch_size, workers=w, seed=0,
            mixup_alpha=a.mixup_alpha)
        it = iter(tl)
        next(it)
        t0, n = time.time(), 0
        for b in it:
            n += len(b["x"])
        secs = time.time() - t0
        rate = n / secs
        line = f"  workers {w:3d}   {rate:7.0f} samples/s   {secs:6.1f} s/epoch"
        gate = [k for k, v in GPU_CEILING.items() if v > rate]
        line += f"   starves: {', '.join(gate) if gate else 'none'}"
        print(line, flush=True)
        del tl, it

    print("\nGPU ceilings (samples/s): "
          + ", ".join(f"{k} {v}" for k, v in GPU_CEILING.items()))


if __name__ == "__main__":
    main()
