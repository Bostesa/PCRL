#!/usr/bin/env python3
"""Evaluate a 3-seed PCRL ensemble on CelebA.

Loads checkpoints/celeba_ensemble/seed_{0,1,2}/best.pt, runs the FIXED
generate_report() per seed, then runs it again on an averaged-rep
encoder that returns the elementwise mean of the three seeds' outputs.

Outputs:
  results/celeba/pcrl_ensemble_perpair.csv
    purpose, attribute, seed{0,1,2}_delta, seed{0,1,2}_r2,
    seed{0,1,2}_pass, majority_pass, averaged_rep_pass,
    averaged_rep_delta, averaged_rep_r2, delta_std

  results/celeba/pcrl_ensemble_vs_laftr.csv
    method, n_pairs, pass_count, notes
"""

from __future__ import annotations

import csv
import logging
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pcrl.training.trainer as _trainer_mod


class _QuietTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable) if self.iterable is not None else iter([])
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def set_postfix(self, *args, **kwargs):
        pass
    def update(self, *args):
        pass
    def close(self):
        pass


_trainer_mod.tqdm = _QuietTqdm

from pcrl.data.base import collate_pcrl_batch
from pcrl.data.celeba import CelebADataset, get_celeba_purposes
from pcrl.evaluation.certificates import generate_report
from pcrl.models.cnn_encoder import CNNPurposeProjectionEncoder
from pcrl.purposes.spec import PurposeRegistry

logging.basicConfig(level=logging.WARNING)

REPR_DIM = 128
CONV_CHANNELS = (32, 64, 128)
DROPOUT = 0.3
BATCH_SIZE = 256
MAX_TRAIN = 10000
MAX_TEST = 3000
SEEDS = (0, 1, 2)


class AveragedEnsembleEncoder(nn.Module):
    """Wraps multiple encoders, returns elementwise mean of their outputs."""

    def __init__(self, encoders):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x, purpose_idx):
        outs = [enc(x, purpose_idx) for enc in self.encoders]
        return torch.stack(outs, dim=0).mean(dim=0)


def load_encoder(seed, num_purposes, device):
    enc = CNNPurposeProjectionEncoder(
        repr_dim=REPR_DIM, num_purposes=num_purposes,
        conv_channels=CONV_CHANNELS, dropout=DROPOUT,
        backbone_grad_scale=1.0,
    )
    ckpt_path = project_root / "checkpoints" / "celeba_ensemble" / f"seed_{seed}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc.load_state_dict(ckpt["encoder"])
    enc.to(device).eval()
    return enc


def report_to_dict(reports):
    """Index by (purpose, attr) -> {delta, r2, adj_pass}."""
    out = {}
    for r in reports:
        delta = r.empirical_best_acc - r.majority_proportion
        out[(r.purpose_name, r.attr_name)] = {
            "delta": delta,
            "r2": r.linear_r2,
            "adj_pass": (delta < 0.02) and (r.linear_r2 < 0.05),
        }
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Ensemble eval, device={device} ===")

    purposes = get_celeba_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)
    num_purposes = len(purposes)

    train_ds = CelebADataset(purposes, root="data/celeba", split="train", max_samples=MAX_TRAIN)
    test_ds = CelebADataset(purposes, root="data/celeba", split="test", max_samples=MAX_TEST)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pcrl_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)

    # ── Per-seed reports ───────────────────────────────────────────────
    encoders = []
    seed_reports = {}
    for seed in SEEDS:
        print(f"  Loading seed {seed}...")
        enc = load_encoder(seed, num_purposes, device)
        encoders.append(enc)
        print(f"  Compliance audit on seed {seed}...")
        reports = generate_report(
            encoder=enc, train_loader=train_loader, test_loader=test_loader,
            purpose_registry=registry, device=device,
        )
        seed_reports[seed] = report_to_dict(reports)
        passed = sum(1 for v in seed_reports[seed].values() if v["adj_pass"])
        print(f"    seed {seed}: {passed}/{len(seed_reports[seed])} pairs pass")

    # ── Averaged-rep ensemble ──────────────────────────────────────────
    print("  Compliance audit on averaged-rep ensemble...")
    ens_enc = AveragedEnsembleEncoder(encoders).to(device).eval()
    ens_reports = generate_report(
        encoder=ens_enc, train_loader=train_loader, test_loader=test_loader,
        purpose_registry=registry, device=device,
    )
    ens_dict = report_to_dict(ens_reports)
    ens_passed = sum(1 for v in ens_dict.values() if v["adj_pass"])
    print(f"    averaged-rep: {ens_passed}/{len(ens_dict)} pairs pass")

    # ── Per-pair output ────────────────────────────────────────────────
    out_dir = project_root / "results" / "celeba"
    out_dir.mkdir(parents=True, exist_ok=True)
    perpair_path = out_dir / "pcrl_ensemble_perpair.csv"

    pair_keys = list(seed_reports[SEEDS[0]].keys())
    fieldnames = ["purpose", "attribute"]
    for s in SEEDS:
        fieldnames += [f"seed{s}_delta", f"seed{s}_r2", f"seed{s}_pass"]
    fieldnames += ["majority_pass", "averaged_rep_pass",
                   "averaged_rep_delta", "averaged_rep_r2", "delta_std"]

    majority_count = 0
    with open(perpair_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for purpose, attr in pair_keys:
            row = {"purpose": purpose, "attribute": attr}
            seed_passes = []
            seed_deltas = []
            for s in SEEDS:
                v = seed_reports[s][(purpose, attr)]
                row[f"seed{s}_delta"] = round(v["delta"], 6)
                row[f"seed{s}_r2"] = round(v["r2"], 6)
                row[f"seed{s}_pass"] = v["adj_pass"]
                seed_passes.append(v["adj_pass"])
                seed_deltas.append(v["delta"])
            majority_pass = sum(seed_passes) >= 2
            if majority_pass:
                majority_count += 1
            ev = ens_dict[(purpose, attr)]
            row["majority_pass"] = majority_pass
            row["averaged_rep_pass"] = ev["adj_pass"]
            row["averaged_rep_delta"] = round(ev["delta"], 6)
            row["averaged_rep_r2"] = round(ev["r2"], 6)
            row["delta_std"] = round(statistics.pstdev(seed_deltas), 6)
            writer.writerow(row)
    print(f"\nSaved {perpair_path}")
    print(f"  majority-vote: {majority_count}/{len(pair_keys)} pass")
    print(f"  averaged-rep:  {ens_passed}/{len(pair_keys)} pass")

    # ── Comparison summary ────────────────────────────────────────────
    summary_path = out_dir / "pcrl_ensemble_vs_laftr.csv"

    # Read prior baselines
    def count_pass(csv_path, pass_col):
        if not csv_path.exists():
            return None, None
        n = 0
        passed = 0
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                n += 1
                v = r.get(pass_col, "")
                if str(v).lower() in ("true", "1", "yes"):
                    passed += 1
        return passed, n

    pcrl1_pass, pcrl1_n = count_pass(out_dir / "baseline_v2_fixed.csv", "adj_pass")
    laftr_pass, laftr_n = count_pass(out_dir / "laftr_per_purpose.csv", "adj_pass")

    # LAFTR non-collapsed: drop attractiveness_prediction rows (collapsed model)
    laftr_nc_pass, laftr_nc_n = None, None
    laftr_csv = out_dir / "laftr_per_purpose.csv"
    if laftr_csv.exists():
        n = 0; passed = 0
        with open(laftr_csv) as f:
            for r in csv.DictReader(f):
                if r.get("purpose") == "attractiveness_prediction":
                    continue
                n += 1
                if str(r.get("adj_pass", "")).lower() in ("true", "1", "yes"):
                    passed += 1
        laftr_nc_pass, laftr_nc_n = passed, n

    rows = [
        {"method": "PCRL 1-model (baseline_v2_fixed)", "n_pairs": pcrl1_n, "pass_count": pcrl1_pass,
         "notes": "single PCRL run, fixed shuffle bug"},
        {"method": "PCRL 3-ensemble majority-vote", "n_pairs": len(pair_keys), "pass_count": majority_count,
         "notes": "pass if >=2/3 seeds pass"},
        {"method": "PCRL 3-ensemble averaged-rep", "n_pairs": len(pair_keys), "pass_count": ens_passed,
         "notes": "elementwise mean of 3 encoders, then compliance"},
        {"method": "LAFTR 5-model ensemble", "n_pairs": laftr_n, "pass_count": laftr_pass,
         "notes": "one model per purpose"},
        {"method": "LAFTR non-collapsed (4 models)", "n_pairs": laftr_nc_n, "pass_count": laftr_nc_pass,
         "notes": "drops attractiveness_prediction (task-acc collapsed)"},
    ]

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n_pairs", "pass_count", "notes"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved {summary_path}")
    print()
    for r in rows:
        print(f"  {r['method']:36s}  {r['pass_count']}/{r['n_pairs']}   {r['notes']}")


if __name__ == "__main__":
    main()
