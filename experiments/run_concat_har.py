#!/usr/bin/env python3
"""Concat-conditioning PCRL on real UCI HAR data.

Mirrors the FiLM PCRL run from run_seeds_fixed.py (har section) but with
conditioning="concat". HPs are chosen to match har_seeds_fixed.csv's
PCRL rows so the FiLM-vs-Concat comparison is clean:

    hidden_dims=[128,128], repr_dim=16, purpose_emb_dim=32, dropout=0.3
    batch_size=256, lr=1e-3, lambda_adv=2.0, lambda_verify=1.0
    auditor_steps=20, epochs=100, patience=None, confusion=entropy

NOTE: the calling instructions listed different HPs ([256,256]/repr=128/lam=50
/epochs=200/patience=25). Those do not match the HPs that produced the
FiLM baseline we are comparing to, so we use the matched HPs here.

3 seeds {0,1,2}. Saves to results/har_real/concat_har_seeds.csv with columns
including variant, seed, activity_accuracy, subject_delta, pass_count.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
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
from pcrl.data.har import HARDataset, get_har_purposes
from pcrl.evaluation.certificates import ComplianceReport, generate_report
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")


SEEDS = [0, 1, 2]
HIDDEN_DIMS = [128, 128]
REPR_DIM = 16
PURPOSE_EMB_DIM = 32
DROPOUT = 0.3
BATCH = 256
LR = 1e-3
LAMBDA_ADV = 2.0
LAMBDA_VERIFY = 1.0
AUDITOR_STEPS = 20
EPOCHS = 100
PATIENCE = None
CONDITIONING = "concat"


@dataclass
class SeedRun:
    seed: int
    task_accuracies: dict[str, float] = field(default_factory=dict)
    reports: list[ComplianceReport] = field(default_factory=list)
    train_time: float = 0.0


def run_seed(seed: int, device: str) -> SeedRun:
    print(f"\n{'='*60}\nHAR concat — seed {seed}\n{'='*60}")
    purposes = get_har_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = HARDataset(purposes=purposes, split="train")
    val_ds = HARDataset(purposes=purposes, split="val")
    test_ds = HARDataset(purposes=purposes, split="test")
    input_dim = train_ds.info.num_features

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)

    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = PurposeConditionedEncoder(
        input_dim=input_dim, hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM,
        num_purposes=len(purposes), purpose_emb_dim=PURPOSE_EMB_DIM,
        conditioning=CONDITIONING, dropout=DROPOUT,
    )
    task_heads: dict[str, torch.nn.Module] = {}
    auditors: dict[str, torch.nn.Module] = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    config = TrainerConfig(
        batch_size=BATCH, lr_encoder=LR, lr_auditor=LR,
        lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY,
        auditor_steps=AUDITOR_STEPS, epochs=EPOCHS, weight_decay=1e-4,
        early_stopping_patience=PATIENCE, confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"concat_har_{seed}"),
    )
    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    trainer.train(train_loader, val_loader=val_loader)
    elapsed = time.time() - t0
    ev = trainer.evaluate(test_loader)

    reports = generate_report(
        encoder=encoder, train_loader=train_loader, test_loader=test_loader,
        purpose_registry=registry, device=device,
    )

    subj_deltas = [r.empirical_best_acc - r.majority_proportion for r in reports if r.attr_name == "subject_id"]
    avg_subj = sum(subj_deltas) / len(subj_deltas) if subj_deltas else 0
    pass_count = sum(1 for r in reports if (r.empirical_best_acc - r.majority_proportion) < 0.02 and r.linear_r2 < 0.05)
    print(f"  activity_acc={ev.task_accuracy.get('activity', 0):.1%}  "
          f"mean_subject_Δ={avg_subj:+.1%}  pass={pass_count}/{len(reports)}  {elapsed:.0f}s")

    return SeedRun(seed=seed, task_accuracies=ev.task_accuracy, reports=reports, train_time=elapsed)


def write_csv(runs: list[SeedRun], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in runs:
        pass_count = sum(
            1 for r in run.reports
            if (r.empirical_best_acc - r.majority_proportion) < 0.02 and r.linear_r2 < 0.05
        )
        total = len(run.reports)
        subj_deltas = [r.empirical_best_acc - r.majority_proportion for r in run.reports if r.attr_name == "subject_id"]
        subj_delta_max = max(subj_deltas) if subj_deltas else float("nan")
        activity_acc = run.task_accuracies.get("activity", float("nan"))
        is_active_acc = run.task_accuracies.get("is_active", float("nan"))
        for r in run.reports:
            delta = r.empirical_best_acc - r.majority_proportion
            ok = delta < 0.02 and r.linear_r2 < 0.05
            rows.append({
                "variant": "Concat",
                "dataset": "har",
                "seed": run.seed,
                "purpose": r.purpose_name,
                "attribute": r.attr_name,
                "best_emp_acc": round(r.empirical_best_acc, 6),
                "majority_baseline": round(r.majority_proportion, 6),
                "delta": round(delta, 6),
                "linear_r2": round(r.linear_r2, 6),
                "adj_pass": ok,
                "pass_count": pass_count,
                "total_pairs": total,
                "subject_delta": round(subj_delta_max, 6),
                "activity_accuracy": round(activity_acc, 6) if not np.isnan(activity_acc) else "",
                "is_active_acc": round(is_active_acc, 6) if not np.isnan(is_active_acc) else "",
                "train_time_s": round(run.train_time, 1),
            })

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out_path}")


def main() -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")
    print(f"Conditioning: {CONDITIONING}")
    print(f"HPs: hidden={HIDDEN_DIMS} repr={REPR_DIM} emb={PURPOSE_EMB_DIM} "
          f"lam_adv={LAMBDA_ADV} lam_ver={LAMBDA_VERIFY} K={AUDITOR_STEPS} "
          f"epochs={EPOCHS} patience={PATIENCE} batch={BATCH}")

    out_path = project_root / "results" / "har_real" / "concat_har_seeds.csv"

    runs: list[SeedRun] = []
    for seed in SEEDS:
        run = run_seed(seed, device)
        runs.append(run)
        # Checkpoint intermediate results after each seed.
        write_csv(runs, out_path)

    print("\nSummary (Concat, HAR):")
    print(f"  {'seed':>4} {'activity':>10} {'subj Δ':>10} {'pass':>6} {'time':>7}")
    for run in runs:
        pc = sum(1 for r in run.reports if (r.empirical_best_acc - r.majority_proportion) < 0.02 and r.linear_r2 < 0.05)
        total = len(run.reports)
        subj_deltas = [r.empirical_best_acc - r.majority_proportion for r in run.reports if r.attr_name == "subject_id"]
        sd = max(subj_deltas) if subj_deltas else float("nan")
        acc = run.task_accuracies.get("activity", float("nan"))
        print(f"  {run.seed:>4} {acc:>10.1%} {sd:>+10.1%} {pc:>3}/{total} {run.train_time:>6.0f}s")


if __name__ == "__main__":
    main()
