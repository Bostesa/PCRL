#!/usr/bin/env python3
"""Rebuttal evidence: the per_dim_std cleanly-compliant floor is architectural.

Computes per_dim_std at four points along the encoder forward path on the
test sets used by the published erase-layer pilot, so reviewers can
verify directly that the 0.5 cleanly-compliant threshold is not
recoverable on the published ``[128, 128] -> 64`` StandardEncoder
backbone by any erasure-side mechanism.

For each (dataset, seed) in {Adult, HMDA, Diabetes} x {0, 1, 2},
instantiate the same StandardEncoder backbone the pilot used (same seed,
same Kaiming init, BN running stats at the pilot's frozen values), re-fit
the joint LEACE eraser on the train data the same way the pilot did, and
report per_dim_std at:

  (a) backbone.network(x)                              -- 128-dim, pre-erase
  (b) eraser(backbone.network(x))                       -- 128-dim, post-LEACE
  (c) backbone.repr_proj(backbone.network(x))           -- 64-dim, no-LEACE counterfactual
  (d) backbone.repr_proj(eraser(backbone.network(x)))   -- 64-dim, post-LEACE (pre-LoRA)

The pilot's final h_p per_dim_std (post-LoRA, per-purpose) is also read
from results/rebuttal/erase_layer_pilot_aws/v2_<ds>_ERASE_PILOT/per_seed_results.json
for the (e) row.

DETERMINISM. The reconstructed backbone is bit-identical to the pilot's
because: (1) torch.manual_seed(seed) gives the same Kaiming/Xavier init;
(2) V2Trainer._freeze_backbone_bn() locks BN running stats at the default
(mean=0, var=1) from construction, so they never update during training;
(3) the backbone is frozen throughout pilot training (LoRA only adjusts
the post-erase representation, never network() or repr_proj()); (4) LEACE
is a closed-form OLS-style fit with no random init.

EXPECTED OUTPUT. On the published ``[128, 128] -> 64`` backbone:
  (a) backbone std  ~ 0.18 - 0.22  (architecture's natural variance floor)
  (b) +LEACE std    ~ 0.15 - 0.19  (~15% reduction)
  (c) +repr_proj    ~ 0.23 - 0.29  (Xavier projection spreads variance)
  (d) +LEACE +proj  ~ 0.20 - 0.25  (what the LoRA sees as input)
  (e) +LoRA (pilot) ~ 0.42         (LoRA + VICReg nearly doubles std but
                                    starts too low to reach 0.5)

INTERPRETATION. The 0.5 cleanly-compliant threshold is unreachable on
this architecture independent of the erasure mechanism. Removing the
LEACE projection entirely (column c) still yields std < 0.30 < 0.5.
Recovering the lost cells requires changing the backbone (wider hidden
dims, lower dropout) rather than changing the privacy mechanism.

USAGE:
    python scripts/diagnose_perdim_std_floor.py

Outputs JSON to results/rebuttal/erase_layer_vicreg_sweep_aws/diagnostic_perdim_std.json.

Originally written as a 30-minute decision diagnostic ("does a decoupled
diagnostic branch on raw backbone features help us recover per_dim_std
above 0.5?"). The result was a clear no across all 60 cells, which is the
rebuttal-useful finding: the floor is in the backbone, not in the erase
layer or the LoRA.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.models.encoder import StandardEncoder  # noqa: E402

DATASETS = ["adult", "hmda", "diabetes"]
SEEDS = [0, 1, 2]
HIDDEN_DIMS = [128, 128]
REPR_DIM = 64
DROPOUT = 0.3
HEALTH_PER_DIM_STD_MIN = 0.5

# Number of (purpose, attr) pairs per seed, per pilot's published grid.
CELLS_PER_SEED = {"adult": 8, "hmda": 6, "diabetes": 6}


def build_datasets(name: str):
    if name == "adult":
        from pcrl.data.adult import AdultDataset, get_adult_purposes
        purposes = get_adult_purposes()
        train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
        test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False,
                               norm_stats=train_ds.norm_stats)
    elif name == "hmda":
        from pcrl.data.hmda import HMDADataset, get_hmda_purposes
        purposes = get_hmda_purposes()
        train_ds = HMDADataset(purposes=purposes, root="data", split="train")
        test_ds = HMDADataset(purposes=purposes, root="data", split="test")
    elif name == "diabetes":
        from pcrl.data.diabetes import DiabetesDataset, get_diabetes_purposes
        purposes = get_diabetes_purposes()
        train_ds = DiabetesDataset(purposes=purposes, split="train")
        test_ds = DiabetesDataset(purposes=purposes, split="test")
    else:
        raise ValueError(name)
    return purposes, train_ds, test_ds


def collect(loader, fn):
    out = []
    for batch in loader:
        x = batch["features"]
        with torch.no_grad():
            h = fn(x)
        out.append(h.cpu().numpy())
    return np.concatenate(out, axis=0)


def collect_attrs(loader, attr_names):
    out = {a: [] for a in attr_names}
    for batch in loader:
        for a in attr_names:
            out[a].append(batch["sensitive_attrs"][a].long().numpy())
    return {a: np.concatenate(out[a], axis=0) for a in attr_names}


def fit_joint_leace(H_train_t: torch.Tensor, attrs: dict):
    """Replicates V2Trainer.fit_erase_layer: joint LEACE on union of disallowed attrs."""
    from concept_erasure import LeaceEraser

    oh_blocks = []
    for a_name, a_int in attrs.items():
        n_classes = int(a_int.max()) + 1
        oh = np.eye(n_classes)[a_int].astype(np.float32)
        oh_blocks.append(oh)
    A_oh = np.concatenate(oh_blocks, axis=1)
    A_oh_t = torch.from_numpy(A_oh).float()
    return LeaceEraser.fit(H_train_t.float(), A_oh_t)


def run_one(name: str, seed: int) -> dict:
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    purposes, train_ds, test_ds = build_datasets(name)
    input_dim = train_ds.info.num_features

    # Same construction the pilot used (use_erase_layer=True; the erase layer
    # is identity-init'd, we re-fit it below).
    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM,
        dropout=DROPOUT, use_erase_layer=True,
    )
    backbone.eval()  # frozen BN, disable dropout — matches pilot eval mode

    train_loader = DataLoader(
        train_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch,
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch,
    )

    # (a) Backbone features pre-erase, 128-dim
    H_train = collect(train_loader, backbone.network)
    H_test = collect(test_loader, backbone.network)
    backbone_std_128 = float(H_test.std(axis=0).mean())

    # Union of disallowed attrs (matches V2Trainer.fit_erase_layer)
    union_attrs = []
    for p in purposes:
        for a in p.disallowed_attrs:
            if a not in union_attrs:
                union_attrs.append(a)
    train_attrs = collect_attrs(train_loader, union_attrs)

    # Fit joint LEACE on train backbone features
    H_train_t = torch.from_numpy(H_train)
    eraser = fit_joint_leace(H_train_t, train_attrs)

    # (b) Post-LEACE 128-dim
    H_test_erased = eraser(torch.from_numpy(H_test).float()).numpy()
    post_leace_std_128 = float(H_test_erased.std(axis=0).mean())

    # (c) No-LEACE counterfactual: repr_proj(backbone.network(x)) → 64-dim
    with torch.no_grad():
        Z_no_leace = backbone.repr_proj(
            torch.from_numpy(H_test).float()
        ).numpy()
    no_leace_repr_proj_std_64 = float(Z_no_leace.std(axis=0).mean())

    # (d) Post-LEACE 64-dim, pre-LoRA: repr_proj(eraser(backbone.network(x)))
    with torch.no_grad():
        Z_post = backbone.repr_proj(
            torch.from_numpy(H_test_erased).float()
        ).numpy()
    post_leace_repr_proj_std_64 = float(Z_post.std(axis=0).mean())

    elapsed = time.time() - t0
    return {
        "dataset": name,
        "seed": seed,
        "n_test": int(H_test.shape[0]),
        "n_train": int(H_train.shape[0]),
        "input_dim": int(input_dim),
        "union_attrs": union_attrs,
        "backbone_std_128d": backbone_std_128,
        "post_leace_std_128d": post_leace_std_128,
        "no_leace_repr_proj_std_64d": no_leace_repr_proj_std_64,
        "post_leace_repr_proj_std_64d": post_leace_repr_proj_std_64,
        "elapsed_s": elapsed,
    }


def main():
    results = []
    print("=" * 110)
    print("per_dim_std diagnostic — fresh backbones at the same seeds as the erase-layer pilot")
    print("=" * 110)
    print(f"{'dataset':10s}  {'seed':>4s}  {'n_test':>7s}  "
          f"{'(a)backbone':>12s}  {'(b)+LEACE':>11s}  "
          f"{'(c)+proj':>11s}  {'(d)+LEACE+proj':>15s}  {'wall(s)':>7s}")
    print(f"{'':10s}  {'':>4s}  {'':>7s}  "
          f"{'128-dim':>12s}  {'128-dim':>11s}  {'64-dim':>11s}  {'64-dim':>15s}")
    print("-" * 110)
    for ds in DATASETS:
        for seed in SEEDS:
            r = run_one(ds, seed)
            results.append(r)
            print(f"{r['dataset']:10s}  {r['seed']:>4d}  {r['n_test']:>7d}  "
                  f"{r['backbone_std_128d']:>12.4f}  {r['post_leace_std_128d']:>11.4f}  "
                  f"{r['no_leace_repr_proj_std_64d']:>11.4f}  "
                  f"{r['post_leace_repr_proj_std_64d']:>15.4f}  {r['elapsed_s']:>7.1f}")
    print()

    # Cell-level fractions (each (dataset, seed) std applies to all its cells)
    metrics = [
        ("(a) backbone, 128-dim", "backbone_std_128d"),
        ("(b) +LEACE, 128-dim", "post_leace_std_128d"),
        ("(c) backbone +repr_proj (no LEACE), 64-dim", "no_leace_repr_proj_std_64d"),
        ("(d) +LEACE +repr_proj, 64-dim (pre-LoRA)", "post_leace_repr_proj_std_64d"),
    ]
    print(f"CELL-LEVEL FRACTIONS ≥ {HEALTH_PER_DIM_STD_MIN}:")
    for label, key in metrics:
        n_total = 0
        n_above = 0
        for r in results:
            c = CELLS_PER_SEED[r["dataset"]]
            n_total += c
            if r[key] >= HEALTH_PER_DIM_STD_MIN:
                n_above += c
        print(f"  {label:50s}  {n_above:2d}/{n_total}")

    # Pilot's final h_p per_dim_std (with LoRA, per-purpose) for reference
    print()
    print("EXISTING pilot final h_p per_dim_std (with LoRA, per-purpose):")
    pilot_paths = {
        "adult":    "results/rebuttal/erase_layer_pilot_aws/v2_adult_ERASE_PILOT/per_seed_results.json",
        "hmda":     "results/rebuttal/erase_layer_pilot_aws/v2_hmda_ERASE_PILOT/per_seed_results.json",
        "diabetes": "results/rebuttal/erase_layer_pilot_aws/v2_diabetes_ERASE_PILOT/per_seed_results.json",
    }
    pilot_total = 0
    pilot_above = 0
    for ds, path in pilot_paths.items():
        psr = json.loads(Path(path).read_text())
        for s in psr["per_seed"]:
            health = s["per_purpose_health"]
            for c in s["attribute_results"]:
                pilot_total += 1
                ph = health.get(c["purpose"], {})
                if ph.get("per_dim_std_mean", 0.0) >= HEALTH_PER_DIM_STD_MIN:
                    pilot_above += 1
    print(f"  (e) +LoRA, 64-dim (pilot h_p)                       {pilot_above:2d}/{pilot_total}")

    # Save
    out_dir = Path("results/rebuttal/erase_layer_vicreg_sweep_aws")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostic_perdim_std.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote: {out_dir / 'diagnostic_perdim_std.json'}")


if __name__ == "__main__":
    main()
