"""MLP-probe evaluation for cached BIOS layer-12 representations.

Spec (RLACE-canonical 1-hidden-layer probe):
  - 128 ReLU hidden units
  - Adam, lr 1e-3 → 1e-4 cosine, 20 epochs, batch 512, weight_decay 1e-4
  - Predicts gender (the disallowed attribute) from the representations
  - Max accuracy over 3 random init seeds (the standard adversarial-probe report)

Per RLACE / Kernelized CE / TaCo / Obliviator empirics, an MLP probe in the
[85%, 97%] range under a linear-R²(gender) constraint is EXPECTED, not a
failure mode. The MLP can re-extract gender from non-linear interactions that
the linear-R² constraint does not (and cannot) bound.

Inputs (any subset; flags below):
  --vanilla-cache      results/bios_pcrl_layer12/cache/cls_layer12_embeddings.npz
                       + .../labels.npz  (no-erasure baseline)
  --leace-reps         results/bios_pcrl_layer12/eval_reps_LEACE.npz
  --rlace-reps         results/bios_pcrl_layer12/eval_reps_RLACE.npz
  --pcrl-reps-dir      results/bios_pcrl_layer12/  (auto-discovers
                       eval_reps_{P1,P2,P3}_{seed}.npz)
  --output             results/bios_pcrl_layer12/mlp_probe_results.json

Each input file is expected to contain ``train_reps``, ``dev_reps``,
``gender_train``, ``gender_dev`` arrays.

Usage:
    python scripts/mlp_probe_eval.py \\
        --vanilla-cache results/bios_pcrl_layer12/cache \\
        --leace-reps results/bios_pcrl_layer12/eval_reps_LEACE.npz \\
        --rlace-reps results/bios_pcrl_layer12/eval_reps_RLACE.npz \\
        --pcrl-reps-dir results/bios_pcrl_layer12 \\
        --output results/bios_pcrl_layer12/mlp_probe_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


HIDDEN = 128
EPOCHS = 20
BATCH = 512
WD = 1e-4
LR_INIT = 1e-3
LR_FINAL = 1e-4
SEEDS = (0, 1, 2)


class _MLP(nn.Module):
    def __init__(self, d: int, n_classes: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_probe(
    train_x: np.ndarray, train_y: np.ndarray,
    dev_x: np.ndarray, dev_y: np.ndarray,
    *, seed: int, device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_classes = int(max(train_y.max(), dev_y.max())) + 1

    Xt = torch.from_numpy(train_x).float()
    yt = torch.from_numpy(train_y).long()
    Xv = torch.from_numpy(dev_x).float().to(device)
    yv = torch.from_numpy(dev_y).long().to(device)

    ds = TensorDataset(Xt, yt)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=g, drop_last=False)

    model = _MLP(d=train_x.shape[1], n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=LR_FINAL,
    )

    best_dev = 0.0
    best_train = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            dev_pred = model(Xv).argmax(dim=-1)
            dev_acc = float((dev_pred == yv).float().mean().item())
            # Train acc on a subsample to avoid an extra full pass.
            n_eval = min(8192, train_x.shape[0])
            idx = torch.randperm(train_x.shape[0], generator=torch.Generator().manual_seed(seed + epoch))[:n_eval]
            tr_pred = model(Xt[idx].to(device)).argmax(dim=-1)
            tr_acc = float((tr_pred == yt[idx].to(device)).float().mean().item())
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_train = tr_acc
    return {"train_acc": best_train, "dev_acc": best_dev, "seed": seed}


def _probe_with_seeds(
    train_x: np.ndarray, train_y: np.ndarray,
    dev_x: np.ndarray, dev_y: np.ndarray, *, device: torch.device,
) -> dict:
    runs = [
        _train_probe(train_x, train_y, dev_x, dev_y, seed=s, device=device)
        for s in SEEDS
    ]
    devs = [r["dev_acc"] for r in runs]
    trains = [r["train_acc"] for r in runs]
    base_rate = float(np.bincount(dev_y).max() / len(dev_y))
    return {
        "per_seed": runs,
        "max_dev_acc": float(max(devs)),
        "mean_dev_acc": float(np.mean(devs)),
        "std_dev_acc": float(np.std(devs)),
        "max_train_acc": float(max(trains)),
        "majority_baseline": base_rate,
    }


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obj = np.load(path)
    return (
        obj["train_reps"].astype(np.float32),
        obj["gender_train"].astype(np.int64),
        obj["dev_reps"].astype(np.float32),
        obj["gender_dev"].astype(np.int64),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vanilla-cache", type=Path, default=None,
                    help="Cache dir produced by cache_bert_embeddings.py.")
    ap.add_argument("--leace-reps", type=Path, default=None)
    ap.add_argument("--rlace-reps", type=Path, default=None)
    ap.add_argument("--pcrl-reps-dir", type=Path, default=None,
                    help="Directory containing eval_reps_{P1,P2,P3}_{seed}.npz.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[probe] device={device}", flush=True)

    results: dict[str, dict] = {}
    t0 = time.time()

    if args.vanilla_cache is not None:
        emb = np.load(args.vanilla_cache / "cls_layer12_embeddings.npz")
        lbl = np.load(args.vanilla_cache / "labels.npz")
        Xtr = emb["train_standardized"].astype(np.float32)
        Xdv = emb["dev_standardized"].astype(np.float32)
        gtr = lbl["gender_train"].astype(np.int64)
        gdv = lbl["gender_dev"].astype(np.int64)
        print(f"[probe] vanilla: N_train={Xtr.shape[0]}  N_dev={Xdv.shape[0]}",
              flush=True)
        results["vanilla"] = _probe_with_seeds(Xtr, gtr, Xdv, gdv, device=device)
        print(f"[probe]   vanilla max_dev={results['vanilla']['max_dev_acc']:.4f}",
              flush=True)

    for tag, path in [("LEACE", args.leace_reps), ("RLACE", args.rlace_reps)]:
        if path is None or not path.exists():
            continue
        Xtr, gtr, Xdv, gdv = _load_npz(path)
        print(f"[probe] {tag}: N_train={Xtr.shape[0]}  N_dev={Xdv.shape[0]}",
              flush=True)
        results[tag] = _probe_with_seeds(Xtr, gtr, Xdv, gdv, device=device)
        print(f"[probe]   {tag} max_dev={results[tag]['max_dev_acc']:.4f}",
              flush=True)

    if args.pcrl_reps_dir is not None and args.pcrl_reps_dir.exists():
        for purpose in ("P1", "P2", "P3"):
            for seed in (0, 1, 2):
                p = args.pcrl_reps_dir / f"eval_reps_{purpose}_{seed}.npz"
                if not p.exists():
                    print(f"[probe] skipping {p.name} (not found)", flush=True)
                    continue
                Xtr, gtr, Xdv, gdv = _load_npz(p)
                tag = f"PCRL_{purpose}_seed{seed}"
                print(f"[probe] {tag}: N_train={Xtr.shape[0]}  N_dev={Xdv.shape[0]}",
                      flush=True)
                results[tag] = _probe_with_seeds(
                    Xtr, gtr, Xdv, gdv, device=device,
                )
                print(f"[probe]   {tag} max_dev={results[tag]['max_dev_acc']:.4f}",
                      flush=True)

    payload = {
        "spec": {
            "hidden": HIDDEN, "epochs": EPOCHS, "batch": BATCH,
            "lr_init": LR_INIT, "lr_final": LR_FINAL,
            "weight_decay": WD, "seeds": list(SEEDS),
            "lr_schedule": "cosine",
        },
        "elapsed_s": time.time() - t0,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[probe] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
