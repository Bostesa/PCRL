#!/usr/bin/env python3
"""PCRL on Folktables ACSIncome: 3 seeds for PCRL plus single-seed Standard
and LAFTR baselines. Writes CSVs to results/folktables/.

Outputs:
  - results/folktables/pcrl_seeds.csv     per-(seed, purpose, attr) rows
  - results/folktables/baselines.csv      Standard + LAFTR, single seed
  - results/folktables/summary.csv        PCRL mean±std + baseline comparison
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Silence tqdm so the log file stays readable when tailed
import pcrl.training.trainer as _trainer_mod


class _QuietTqdm:
    def __init__(self, iterable=None, *a, **k): self.iterable = iterable
    def __iter__(self): return iter(self.iterable) if self.iterable is not None else iter([])
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def set_postfix(self, *a, **k): pass
    def update(self, *a): pass
    def close(self): pass


_trainer_mod.tqdm = _QuietTqdm

import warnings
warnings.filterwarnings("ignore")

from pcrl.data.base import collate_pcrl_batch
from pcrl.data.folktables import FolktablesIncomeDataset, get_folktables_purposes
from pcrl.evaluation.certificates import generate_report
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder, StandardEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

HIDDEN_DIMS = [128, 128]
REPR_DIM = 64
PURPOSE_EMB_DIM = 32
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 256
DELTA_THRESHOLD = 0.02
R2_THRESHOLD = 0.05


def save_csv(rows, path: Path) -> None:
    if not rows:
        logger.warning(f"No rows to save for {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    logger.info(f"Saved {path} ({len(rows)} rows)")


def load_folktables(states: list[str]) -> dict:
    purposes = get_folktables_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = FolktablesIncomeDataset(
        purposes=purposes, root="data", split="train",
        download=True, states=states,
    )
    val_ds = FolktablesIncomeDataset(
        purposes=purposes, root="data", split="val",
        download=False, states=states, norm_stats=train_ds.norm_stats,
    )
    test_ds = FolktablesIncomeDataset(
        purposes=purposes, root="data", split="test",
        download=False, states=states, norm_stats=train_ds.norm_stats,
    )
    logger.info(
        f"Folktables sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, "
        f"features={train_ds.info.num_features}"
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    train_extract = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                               collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)
    return {
        "purposes": purposes,
        "registry": registry,
        "input_dim": train_ds.info.num_features,
        "train_loader": train_loader,
        "train_extract_loader": train_extract,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def build_task_heads(purposes, repr_dim):
    heads = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
    return heads


def build_auditors(purposes, repr_dim):
    auditors = {}
    for p in purposes:
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )
    return auditors


def report_rows(reports, common_fields):
    """Convert ComplianceReport list to per-pair CSV rows."""
    rows = []
    for r in reports:
        delta = r.empirical_best_acc - r.majority_proportion
        adj_pass = bool(delta < DELTA_THRESHOLD and r.linear_r2 < R2_THRESHOLD)
        rows.append({
            **common_fields,
            "purpose": r.purpose_name,
            "attribute": r.attr_name,
            "best_emp_acc": round(r.empirical_best_acc, 6),
            "majority_baseline": round(r.majority_proportion, 6),
            "delta": round(delta, 6),
            "linear_r2": round(r.linear_r2, 6),
            "num_classes": r.num_classes,
            "adj_pass": adj_pass,
        })
    return rows


def train_pcrl(data, seed, device, tag):
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = PurposeConditionedEncoder(
        input_dim=data["input_dim"], hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM,
        num_purposes=len(data["purposes"]), purpose_emb_dim=PURPOSE_EMB_DIM,
        conditioning="film", dropout=0.3,
    )
    task_heads = build_task_heads(data["purposes"], REPR_DIM)
    auditors = build_auditors(data["purposes"], REPR_DIM)
    config = TrainerConfig(
        batch_size=BATCH_SIZE, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=50.0, lambda_verify=50.0, auditor_steps=20,
        epochs=EPOCHS, weight_decay=1e-4, early_stopping_patience=PATIENCE,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"folktables_pcrl_{tag}"),
    )
    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=data["registry"], device=device,
    )
    t0 = time.time()
    state = trainer.train(data["train_loader"], val_loader=data["val_loader"])
    train_time = time.time() - t0
    eval_metrics = trainer.evaluate(data["test_loader"])
    reports = generate_report(
        encoder=encoder, train_loader=data["train_loader"],
        test_loader=data["test_loader"], purpose_registry=data["registry"],
        device=device,
    )
    return reports, eval_metrics, train_time, state.epoch + 1


def train_standard(data, seed, device, tag):
    """Standard encoder (no privacy): lambda_adv=0, no purpose conditioning."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = StandardEncoder(
        input_dim=data["input_dim"], hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM, dropout=0.3,
    )
    task_heads = build_task_heads(data["purposes"], REPR_DIM)
    auditors = build_auditors(data["purposes"], REPR_DIM)
    config = TrainerConfig(
        batch_size=BATCH_SIZE, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=0.0, lambda_verify=0.0, auditor_steps=1,
        epochs=EPOCHS, weight_decay=1e-4, early_stopping_patience=PATIENCE,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"folktables_std_{tag}"),
    )
    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=data["registry"], device=device,
    )
    t0 = time.time()
    state = trainer.train(data["train_loader"], val_loader=data["val_loader"])
    train_time = time.time() - t0
    eval_metrics = trainer.evaluate(data["test_loader"])
    reports = generate_report(
        encoder=encoder, train_loader=data["train_loader"],
        test_loader=data["test_loader"], purpose_registry=data["registry"],
        device=device,
    )
    return reports, eval_metrics, train_time, state.epoch + 1


def train_laftr(data, seed, device, tag, lambda_adv=25.0):
    """LAFTR: StandardEncoder + adversarial loss (no purpose conditioning)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = StandardEncoder(
        input_dim=data["input_dim"], hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM, dropout=0.3,
    )
    task_heads = build_task_heads(data["purposes"], REPR_DIM)
    auditors = build_auditors(data["purposes"], REPR_DIM)
    config = TrainerConfig(
        batch_size=BATCH_SIZE, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=float(lambda_adv), lambda_verify=float(lambda_adv),
        auditor_steps=10,
        epochs=EPOCHS, weight_decay=1e-4, early_stopping_patience=PATIENCE,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"folktables_laftr_{tag}"),
    )
    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=data["registry"], device=device,
    )
    t0 = time.time()
    state = trainer.train(data["train_loader"], val_loader=data["val_loader"])
    train_time = time.time() - t0
    eval_metrics = trainer.evaluate(data["test_loader"])
    reports = generate_report(
        encoder=encoder, train_loader=data["train_loader"],
        test_loader=data["test_loader"], purpose_registry=data["registry"],
        device=device,
    )
    return reports, eval_metrics, train_time, state.epoch + 1


def summarize_pcrl(pcrl_rows):
    """Aggregate per-seed rows → mean/std per (purpose, attribute)."""
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for row in pcrl_rows:
        k = (row["purpose"], row["attribute"])
        by_pair.setdefault(k, []).append(row)
    summary = []
    for (purpose, attr), rows in by_pair.items():
        emp = np.array([r["best_emp_acc"] for r in rows])
        maj = np.array([r["majority_baseline"] for r in rows])
        delta = np.array([r["delta"] for r in rows])
        r2 = np.array([r["linear_r2"] for r in rows])
        passes = np.array([1 if r["adj_pass"] else 0 for r in rows])
        summary.append({
            "method": "PCRL",
            "purpose": purpose,
            "attribute": attr,
            "num_seeds": len(rows),
            "best_emp_acc_mean": round(float(emp.mean()), 6),
            "best_emp_acc_std": round(float(emp.std(ddof=0)), 6),
            "majority_baseline_mean": round(float(maj.mean()), 6),
            "delta_mean": round(float(delta.mean()), 6),
            "delta_std": round(float(delta.std(ddof=0)), 6),
            "linear_r2_mean": round(float(r2.mean()), 6),
            "linear_r2_std": round(float(r2.std(ddof=0)), 6),
            "pass_rate": round(float(passes.mean()), 4),
        })
    return summary


def summarize_baseline(baseline_rows, method):
    method_rows = [r for r in baseline_rows if r["method"] == method]
    summary = []
    for r in method_rows:
        summary.append({
            "method": method,
            "purpose": r["purpose"],
            "attribute": r["attribute"],
            "num_seeds": 1,
            "best_emp_acc_mean": r["best_emp_acc"],
            "best_emp_acc_std": 0.0,
            "majority_baseline_mean": r["majority_baseline"],
            "delta_mean": r["delta"],
            "delta_std": 0.0,
            "linear_r2_mean": r["linear_r2"],
            "linear_r2_std": 0.0,
            "pass_rate": 1.0 if r["adj_pass"] else 0.0,
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", default="CA",
                        help="Comma-separated state codes (default CA)")
    parser.add_argument("--seeds", default="0,1,2",
                        help="Comma-separated seeds for PCRL")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip Standard and LAFTR baselines")
    args = parser.parse_args()

    states = args.states.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}, states={states}, PCRL seeds={seeds}")

    results_dir = project_root / "results" / "folktables"
    results_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    data = load_folktables(states=states)

    # ── PCRL: 3 seeds ───────────────────────────────────────────────────
    pcrl_rows: list[dict] = []
    for seed in seeds:
        logger.info(f"═══ PCRL seed={seed} ═══")
        reports, eval_metrics, train_time, stopped_epoch = train_pcrl(
            data, seed, device, tag=f"s{seed}",
        )
        common = {
            "method": "PCRL",
            "seed": seed,
            "test_income_acc": round(eval_metrics.task_accuracy.get("income", 0.0), 6),
            "test_hours_band_acc": round(eval_metrics.task_accuracy.get("hours_band", 0.0), 6),
            "test_education_level_acc": round(eval_metrics.task_accuracy.get("education_level", 0.0), 6),
            "stopped_epoch": stopped_epoch,
            "train_time_s": round(train_time, 1),
        }
        pcrl_rows.extend(report_rows(reports, common))
        logger.info(
            f"  seed={seed}: income_acc={common['test_income_acc']:.4f} "
            f"ep={stopped_epoch} {train_time:.0f}s"
        )
        # Incremental save so we never lose partials
        save_csv(pcrl_rows, results_dir / "pcrl_seeds.csv")

    # ── Baselines ───────────────────────────────────────────────────────
    baseline_rows: list[dict] = []
    if not args.skip_baselines:
        logger.info("═══ Standard (no privacy) seed=0 ═══")
        reports, eval_metrics, train_time, stopped_epoch = train_standard(
            data, seed=0, device=device, tag="s0",
        )
        common = {
            "method": "Standard",
            "seed": 0,
            "test_income_acc": round(eval_metrics.task_accuracy.get("income", 0.0), 6),
            "test_hours_band_acc": round(eval_metrics.task_accuracy.get("hours_band", 0.0), 6),
            "test_education_level_acc": round(eval_metrics.task_accuracy.get("education_level", 0.0), 6),
            "stopped_epoch": stopped_epoch,
            "train_time_s": round(train_time, 1),
        }
        baseline_rows.extend(report_rows(reports, common))
        save_csv(baseline_rows, results_dir / "baselines.csv")

        logger.info("═══ LAFTR (λ=25) seed=0 ═══")
        reports, eval_metrics, train_time, stopped_epoch = train_laftr(
            data, seed=0, device=device, tag="s0_lam25", lambda_adv=25.0,
        )
        common = {
            "method": "LAFTR",
            "seed": 0,
            "lambda_adv": 25.0,
            "test_income_acc": round(eval_metrics.task_accuracy.get("income", 0.0), 6),
            "test_hours_band_acc": round(eval_metrics.task_accuracy.get("hours_band", 0.0), 6),
            "test_education_level_acc": round(eval_metrics.task_accuracy.get("education_level", 0.0), 6),
            "stopped_epoch": stopped_epoch,
            "train_time_s": round(train_time, 1),
        }
        baseline_rows.extend(report_rows(reports, common))
        save_csv(baseline_rows, results_dir / "baselines.csv")

    # ── Summary ─────────────────────────────────────────────────────────
    summary = summarize_pcrl(pcrl_rows)
    if baseline_rows:
        summary.extend(summarize_baseline(baseline_rows, "Standard"))
        summary.extend(summarize_baseline(baseline_rows, "LAFTR"))
    save_csv(summary, results_dir / "summary.csv")

    total_min = (time.time() - t_total) / 60.0
    logger.info(f"Done. Total wall time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
