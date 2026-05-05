#!/usr/bin/env python3
"""Aggregator: concatenate K=3 LAFTR encoder outputs into a per-(dataset, seed)
union representation [N, 192] for the cross-purpose attack.

Reads:
    results/laftr_benchmark/{dataset}/{purpose}/seed_{s}/encoder.pt
    results/laftr_benchmark/{dataset}/{purpose}/seed_{s}/test_labels.npz
        (any one of the three is used as the canonical label set —
         all three are computed on the same test split & ordering since
         the loader is shuffle=False, but we sanity-check.)

Writes:
    results/laftr_benchmark/{dataset}/seed_{s}/union_reps.npz
        (shape [N, 64*K] = [N, 192])
    results/laftr_benchmark/{dataset}/seed_{s}/union_test_labels.npz
        (canonical task + all-sensitive labels for the test split,
         re-saved at the seed level for cross-purpose-attack convenience)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.models.encoder import StandardEncoder  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from scripts.run_laftr_benchmark import PURPOSE_NAMES, build_datasets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["adult", "hmda", "diabetes"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--bench-root", default="results/laftr_benchmark",
                    help="root containing {dataset}/{purpose}/seed_{s}/encoder.pt")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    bench_root = Path(args.bench_root)
    purpose_names = PURPOSE_NAMES[args.dataset]
    out_dir = bench_root / args.dataset / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reload dataset (test split, shuffle=False) once; re-extract reps for each
    # of the K encoders by loading their state_dicts.
    purposes, _train_ds, _val_ds, test_ds = build_datasets(args.dataset)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0,
    )

    # Pull canonical labels (task + all sensitive attrs).
    all_sensitive_attrs = list(test_ds.info.sensitive_attrs.keys())
    label_buf: dict[str, list[np.ndarray]] = {a: [] for a in all_sensitive_attrs}
    label_buf_task: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for batch in test_loader:
            for a in all_sensitive_attrs:
                label_buf[a].append(batch["sensitive_attrs"][a].numpy())
            for tname, tarr in batch["task_labels"].items():
                label_buf_task.setdefault(tname, []).append(tarr.numpy())

    sensitive_arrs = {f"sensitive_{a}": np.concatenate(v).astype(np.int64)
                      for a, v in label_buf.items()}
    task_arrs = {f"task_{t}": np.concatenate(v).astype(np.int64)
                 for t, v in label_buf_task.items()}
    n_test = next(iter(sensitive_arrs.values())).shape[0]

    # Load each encoder, extract reps on the test split, concatenate.
    reps_per_purpose: list[np.ndarray] = []
    for p_idx, p_name in enumerate(purpose_names):
        ckpt_path = bench_root / args.dataset / p_name / f"seed_{args.seed}" / "encoder.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        enc = StandardEncoder(
            input_dim=payload["input_dim"],
            hidden_dims=payload["hidden_dims"],
            repr_dim=payload["repr_dim"],
            dropout=payload["dropout"],
        )
        enc.load_state_dict(payload["encoder"])
        enc.eval()

        # As an additional sanity check, prefer the cached eval_reps.npz if it
        # exists and has the right N — reps were extracted with the same loader
        # and ordering, so they're identical to what we'd recompute.
        cached_path = ckpt_path.parent / "eval_reps.npz"
        if cached_path.exists():
            cached = np.load(cached_path)["reps"]
            if cached.shape[0] == n_test and cached.shape[1] == payload["repr_dim"]:
                reps_per_purpose.append(cached.astype(np.float32))
                print(f"  [{args.dataset}/{p_name}/s{args.seed}] cached reps {cached.shape}")
                continue

        # Recompute
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["features"]
                h = enc(x, 0)
                chunks.append(h.cpu().numpy())
        reps = np.concatenate(chunks, axis=0).astype(np.float32)
        if reps.shape[0] != n_test:
            raise RuntimeError(f"row mismatch: reps {reps.shape[0]} vs labels {n_test}")
        reps_per_purpose.append(reps)
        print(f"  [{args.dataset}/{p_name}/s{args.seed}] reps {reps.shape}")

    union = np.concatenate(reps_per_purpose, axis=1)
    print(f"  union shape: {union.shape}")

    np.savez(out_dir / "union_reps.npz", reps=union)
    np.savez(out_dir / "union_test_labels.npz", **sensitive_arrs, **task_arrs)
    print(f"  wrote {out_dir / 'union_reps.npz'}")
    print(f"  wrote {out_dir / 'union_test_labels.npz'}")


if __name__ == "__main__":
    main()
