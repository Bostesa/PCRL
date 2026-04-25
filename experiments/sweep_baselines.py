#!/usr/bin/env python3
"""Baseline hyperparameter sweeps on Adult: LAFTR, INLP, LEACE.

Validation-based selection. Winners evaluated on test with 3 seeds.

Outputs:
  - results/adult/laftr_sweep.csv           (5 configs, val)
  - results/adult/laftr_sweep_winner_seeds.csv   (3 seeds, test)
  - results/adult/inlp_sweep.csv            (5 configs, val)
  - results/adult/inlp_sweep_winner_seeds.csv    (3 seeds, test)
  - results/adult/leace_sweep.csv           (4 configs, val)
  - results/adult/leace_sweep_winner_seeds.csv   (3 seeds, test)
  - results/adult/baseline_sweep_summary.csv     (final comparison)
"""

from __future__ import annotations

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

import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")

from pcrl.data.adult import AdultDataset, get_adult_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.evaluation.certificates import (
    ComplianceReport,
    EmpiricalAudit,
    LinearAudit,
    generate_report,
)
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.baselines import INLPProjector, LEACEEraser
from pcrl.models.encoder import StandardEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)

SEEDS = [0, 1, 2]
SWEEP_SEED = 0
HIDDEN_DIMS = [128, 128]
REPR_DIM = 64
EPOCHS = 200
PATIENCE = 20
DELTA_THRESHOLD = 0.02
R2_THRESHOLD = 0.05


# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════


def extract_reprs(encoder, loader, device):
    """Run encoder over loader, collect (reprs, task_labels, sens_attrs)."""
    encoder.eval()
    all_r, tasks, sens = [], {}, {}
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder(x)
            all_r.append(h.cpu().numpy())
            for k, v in batch["task_labels"].items():
                tasks.setdefault(k, []).append(v.numpy())
            for k, v in batch["sensitive_attrs"].items():
                sens.setdefault(k, []).append(v.numpy())
    return (
        np.concatenate(all_r),
        {k: np.concatenate(v) for k, v in tasks.items()},
        {k: np.concatenate(v) for k, v in sens.items()},
    )


def compliance_on_reprs(train_reprs, eval_reprs, train_sens, eval_sens,
                        purpose_name, disallowed):
    """Run linear+empirical audit on a (train, eval) pair."""
    linear_audit = LinearAudit()
    empirical_audit = EmpiricalAudit()
    reports = []
    for attr in disallowed:
        linear_result, null_result = linear_audit.audit(eval_reprs, eval_sens[attr])
        all_labels = np.concatenate([train_sens[attr], eval_sens[attr]])
        unique, counts = np.unique(all_labels, return_counts=True)
        num_classes = len(unique)
        chance_acc = 1.0 / max(num_classes, 1)
        majority = float(counts.max() / len(all_labels))
        best_acc, emp_results = empirical_audit.audit(
            train_reprs, train_sens[attr], eval_reprs, eval_sens[attr],
        )
        empirical_ok = (best_acc - chance_acc) < 0.05
        certified = linear_result.certified and empirical_ok
        reports.append(ComplianceReport(
            purpose_name=purpose_name, attr_name=attr,
            linear_r2=linear_result.r_squared,
            linear_certified=linear_result.certified,
            null_space_r2=null_result.r_squared,
            variance_preserved=null_result.variance_preserved or 1.0,
            empirical_best_acc=best_acc,
            empirical_chance_acc=chance_acc,
            empirical_results=emp_results,
            certified=certified,
            majority_proportion=majority,
            num_classes=num_classes,
        ))
    return reports


def eval_task_logreg(train_reprs, train_y, eval_reprs, eval_y):
    """Fit LogReg on train reprs and return eval accuracy."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(train_reprs, train_y)
    return float(clf.score(eval_reprs, eval_y))


def count_pass(reports):
    return sum(1 for r in reports
               if (r.empirical_best_acc - r.majority_proportion) < DELTA_THRESHOLD
               and r.linear_r2 < R2_THRESHOLD)


def mean_delta(reports):
    if not reports:
        return 0.0
    return float(np.mean([r.empirical_best_acc - r.majority_proportion
                          for r in reports]))


def mean_r2(reports):
    if not reports:
        return 0.0
    return float(np.mean([r.linear_r2 for r in reports]))


def save_csv(rows, path):
    if not rows:
        print(f"  (no rows to save for {path})")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use the union of all keys across rows
    all_keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in all_keys})
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════


def load_adult():
    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)
    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False,
                          norm_stats=train_ds.norm_stats)
    test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False,
                           norm_stats=train_ds.norm_stats)
    input_dim = train_ds.info.num_features
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    # Train-for-extraction: no shuffle, for aligning reprs with labels
    train_extract_loader = DataLoader(train_ds, batch_size=256, shuffle=False,
                                      collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)
    return {
        "purposes": purposes, "registry": registry, "input_dim": input_dim,
        "train_loader": train_loader,
        "train_extract_loader": train_extract_loader,
        "val_loader": val_loader, "test_loader": test_loader,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LAFTR
# ═══════════════════════════════════════════════════════════════════════════


def train_laftr(lambda_adv, seed, data, device, eval_loader, tag):
    """Train LAFTR, evaluate on eval_loader. Returns (reports, income_acc, train_time, stopped_epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    purposes = data["purposes"]
    registry = data["registry"]
    input_dim = data["input_dim"]

    encoder = StandardEncoder(
        input_dim=input_dim, hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM, dropout=0.3,
    )
    task_heads, auditors = {}, {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=float(lambda_adv), lambda_verify=float(lambda_adv),
        auditor_steps=10,
        epochs=EPOCHS, weight_decay=1e-4, early_stopping_patience=PATIENCE,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"swp_laftr_{tag}"),
    )
    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    state = trainer.train(data["train_loader"], val_loader=data["val_loader"])
    train_time = time.time() - t0

    eval_metrics = trainer.evaluate(eval_loader)
    reports = generate_report(
        encoder=encoder, train_loader=data["train_loader"],
        test_loader=eval_loader, purpose_registry=registry, device=device,
    )
    return reports, eval_metrics.task_accuracy.get("income", 0.0), train_time, state.epoch + 1


def report_rows(reports, common_fields):
    rows = []
    pc = count_pass(reports)
    for r in reports:
        delta = r.empirical_best_acc - r.majority_proportion
        ok = delta < DELTA_THRESHOLD and r.linear_r2 < R2_THRESHOLD
        row = {
            **common_fields,
            "purpose": r.purpose_name, "attribute": r.attr_name,
            "best_emp_acc": round(r.empirical_best_acc, 6),
            "majority_baseline": round(r.majority_proportion, 6),
            "delta": round(delta, 6),
            "linear_r2": round(r.linear_r2, 6),
            "adj_pass": ok,
            "pass_count": pc,
            "total_pairs": len(reports),
        }
        rows.append(row)
    return rows


def laftr_sweep(data, device):
    print("\n" + "#" * 60)
    print("# LAFTR sweep on Adult")
    print("#" * 60)
    lambdas = [10, 25, 50, 100, 200]
    sweep_rows = []
    ranked = []  # (key, lambda_adv)

    for la in lambdas:
        print(f"  λ={la} seed={SWEEP_SEED} ...", flush=True)
        reports, income_val, ttime, sepoch = train_laftr(
            la, SWEEP_SEED, data, device, eval_loader=data["val_loader"],
            tag=f"lam{la}_s{SWEEP_SEED}",
        )
        pc = count_pass(reports)
        md = mean_delta(reports)
        mr = mean_r2(reports)
        print(f"    val_pass={pc}/{len(reports)} mean_Δ={md:+.4f} mean_R²={mr:.4f} "
              f"income_val={income_val:.4f} ep={sepoch} {ttime:.0f}s")
        sweep_rows.append({
            "lambda_adv": la, "seed": SWEEP_SEED,
            "val_pass_count": pc, "val_total_pairs": len(reports),
            "val_mean_delta": round(md, 6), "val_mean_r2": round(mr, 6),
            "val_income_acc": round(income_val, 6),
            "stopped_epoch": sepoch, "train_time_s": round(ttime, 1),
        })
        key = (pc, -md, income_val)
        ranked.append((key, la))

    save_csv(sweep_rows, project_root / "results/adult/laftr_sweep.csv")

    ranked.sort(reverse=True)
    winner_lambda = ranked[0][1]
    print(f"\n  WINNER: λ={winner_lambda} "
          f"(ranked keys: {[(r[1], r[0]) for r in ranked[:3]]})")

    print(f"\n  Running winner λ={winner_lambda} on TEST with 3 seeds ...")
    seed_rows = []
    for seed in SEEDS:
        print(f"    seed={seed} ...", flush=True)
        reports, income_test, ttime, sepoch = train_laftr(
            winner_lambda, seed, data, device, eval_loader=data["test_loader"],
            tag=f"lam{winner_lambda}_winner_s{seed}",
        )
        pc = count_pass(reports)
        md = mean_delta(reports)
        print(f"      test_pass={pc}/{len(reports)} mean_Δ={md:+.4f} "
              f"income_test={income_test:.4f} ep={sepoch} {ttime:.0f}s")
        seed_rows.extend(report_rows(reports, {
            "method": "LAFTR", "lambda_adv": winner_lambda, "seed": seed,
            "test_income_acc": round(income_test, 6),
            "stopped_epoch": sepoch, "train_time_s": round(ttime, 1),
        }))

    save_csv(seed_rows, project_root / "results/adult/laftr_sweep_winner_seeds.csv")
    return {"method": "LAFTR", "hp": winner_lambda, "seed_rows": seed_rows}


# ═══════════════════════════════════════════════════════════════════════════
# Standard encoder (shared by INLP/LEACE)
# ═══════════════════════════════════════════════════════════════════════════


def train_standard_encoder(seed, data, device):
    """Train a Standard encoder and extract reprs from all 3 splits."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    purposes = data["purposes"]
    registry = data["registry"]
    input_dim = data["input_dim"]

    encoder = StandardEncoder(
        input_dim=input_dim, hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM, dropout=0.3,
    )
    task_heads, auditors = {}, {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )
    config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=0.0, lambda_verify=0.0, auditor_steps=1,
        epochs=EPOCHS, weight_decay=1e-4, early_stopping_patience=PATIENCE,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"swp_std_s{seed}"),
    )
    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )
    t0 = time.time()
    trainer.train(data["train_loader"], val_loader=data["val_loader"])
    ttime = time.time() - t0
    train_reprs, train_tasks, train_sens = extract_reprs(
        encoder, data["train_extract_loader"], device)
    val_reprs, val_tasks, val_sens = extract_reprs(
        encoder, data["val_loader"], device)
    test_reprs, test_tasks, test_sens = extract_reprs(
        encoder, data["test_loader"], device)
    return {
        "encoder": encoder, "train_time": ttime,
        "train_reprs": train_reprs, "val_reprs": val_reprs, "test_reprs": test_reprs,
        "train_tasks": train_tasks, "val_tasks": val_tasks, "test_tasks": test_tasks,
        "train_sens": train_sens, "val_sens": val_sens, "test_sens": test_sens,
    }


# ═══════════════════════════════════════════════════════════════════════════
# INLP
# ═══════════════════════════════════════════════════════════════════════════


def apply_inlp(std_data, purposes, iters, eval_split):
    """Apply INLP per-purpose with max_iters=iters to std_data reprs.
    eval_split in {'val', 'test'}. Returns (reports, income_acc)."""
    eval_reprs_key = f"{eval_split}_reprs"
    eval_sens_key = f"{eval_split}_sens"
    eval_tasks_key = f"{eval_split}_tasks"

    all_reports = []
    income_acc = None
    iters_used_stats = []

    for p in purposes:
        proj_train = std_data["train_reprs"].copy()
        proj_eval = std_data[eval_reprs_key].copy()
        purpose_iters_used = []
        for attr in p.disallowed_attrs:
            proj = INLPProjector(max_iters=iters, min_accuracy=0.52, random_state=42)
            proj.fit(proj_train, std_data["train_sens"][attr])
            proj_train = proj.transform(proj_train)
            proj_eval = proj.transform(proj_eval)
            purpose_iters_used.append(proj.num_iters_used)
        iters_used_stats.append((p.name, purpose_iters_used))

        task_name = p.allowed_tasks[0]
        if task_name == "income":
            income_acc = eval_task_logreg(
                proj_train, std_data["train_tasks"][task_name],
                proj_eval, std_data[eval_tasks_key][task_name],
            )

        reports = compliance_on_reprs(
            proj_train, proj_eval,
            std_data["train_sens"], std_data[eval_sens_key],
            p.name, p.disallowed_attrs,
        )
        all_reports.extend(reports)

    return all_reports, income_acc if income_acc is not None else 0.0, iters_used_stats


def inlp_sweep(data, std_data_seed0, device):
    print("\n" + "#" * 60)
    print("# INLP sweep on Adult")
    print("#" * 60)

    iter_grid = [10, 25, 50, 75, 100]
    purpose1 = data["purposes"][0]  # income_prediction
    sweep_rows = []
    ranked = []
    iters_to_hit_pcrl_sex = None  # PCRL's sex_delta ~0.2%, target <2%

    for iters in iter_grid:
        t0 = time.time()
        reports, income_val, iters_used = apply_inlp(
            std_data_seed0, data["purposes"], iters, eval_split="val",
        )
        dt = time.time() - t0

        # Selection metric: pass count on purpose 1 (2 pairs)
        purpose1_reports = [r for r in reports if r.purpose_name == purpose1.name]
        p1_pass = count_pass(purpose1_reports)
        p1_md = mean_delta(purpose1_reports)
        full_pass = count_pass(reports)
        full_md = mean_delta(reports)

        # Sex delta check
        sex_report = next((r for r in purpose1_reports if r.attr_name == "sex"), None)
        sex_delta = (sex_report.empirical_best_acc - sex_report.majority_proportion) \
            if sex_report else None
        if sex_delta is not None and sex_delta < 0.02 and iters_to_hit_pcrl_sex is None:
            iters_to_hit_pcrl_sex = iters

        print(f"  iters={iters:>3}  p1_pass={p1_pass}/2  sex_Δ={sex_delta:+.4f}  "
              f"full_pass={full_pass}/{len(reports)}  income_val={income_val:.4f}  "
              f"iters_used={[u for _, u in iters_used]}  {dt:.0f}s")

        sweep_rows.append({
            "max_iters": iters, "seed": SWEEP_SEED,
            "purpose1_pass_count": p1_pass, "purpose1_total_pairs": 2,
            "purpose1_mean_delta": round(p1_md, 6),
            "sex_delta": round(sex_delta, 6) if sex_delta is not None else "",
            "val_pass_count": full_pass, "val_total_pairs": len(reports),
            "val_mean_delta": round(full_md, 6),
            "val_income_acc": round(income_val, 6),
            "iters_used_by_purpose": str([u for _, u in iters_used]),
            "time_s": round(dt, 1),
        })
        key = (p1_pass, -p1_md, income_val)
        ranked.append((key, iters))

    save_csv(sweep_rows, project_root / "results/adult/inlp_sweep.csv")

    ranked.sort(reverse=True)
    winner_iters = ranked[0][1]
    print(f"\n  WINNER: max_iters={winner_iters}  "
          f"(iters to match PCRL sex_Δ<2%: {iters_to_hit_pcrl_sex or 'never in grid'})")

    # 3 seeds on TEST. We need Standard encoders for seeds 1, 2
    print(f"\n  Running winner max_iters={winner_iters} on TEST with 3 seeds ...")
    seed_rows = []
    for seed in SEEDS:
        if seed == 0:
            sd = std_data_seed0
        else:
            print(f"    training Standard seed={seed} ...", flush=True)
            sd = train_standard_encoder(seed, data, device)
        t0 = time.time()
        reports, income_test, iters_used = apply_inlp(
            sd, data["purposes"], winner_iters, eval_split="test",
        )
        dt = time.time() - t0
        pc = count_pass(reports)
        md = mean_delta(reports)
        print(f"    seed={seed}  test_pass={pc}/{len(reports)}  mean_Δ={md:+.4f}  "
              f"income_test={income_test:.4f}  {dt:.0f}s")
        seed_rows.extend(report_rows(reports, {
            "method": "INLP", "max_iters": winner_iters, "seed": seed,
            "test_income_acc": round(income_test, 6),
            "iters_used_by_purpose": str([u for _, u in iters_used]),
            "time_s": round(dt, 1),
        }))

    save_csv(seed_rows, project_root / "results/adult/inlp_sweep_winner_seeds.csv")
    return {
        "method": "INLP", "hp": winner_iters, "seed_rows": seed_rows,
        "iters_to_hit_pcrl_sex": iters_to_hit_pcrl_sex,
    }


# ═══════════════════════════════════════════════════════════════════════════
# LEACE
# ═══════════════════════════════════════════════════════════════════════════


def apply_leace(std_data, purposes, regularization, eval_split):
    eval_reprs_key = f"{eval_split}_reprs"
    eval_sens_key = f"{eval_split}_sens"
    eval_tasks_key = f"{eval_split}_tasks"

    all_reports = []
    income_acc = None

    for p in purposes:
        proj_train = std_data["train_reprs"].copy()
        proj_eval = std_data[eval_reprs_key].copy()
        for attr in p.disallowed_attrs:
            eraser = LEACEEraser(regularization=regularization)
            eraser.fit(proj_train, std_data["train_sens"][attr])
            proj_train = eraser.transform(proj_train)
            proj_eval = eraser.transform(proj_eval)

        task_name = p.allowed_tasks[0]
        if task_name == "income":
            income_acc = eval_task_logreg(
                proj_train, std_data["train_tasks"][task_name],
                proj_eval, std_data[eval_tasks_key][task_name],
            )
        reports = compliance_on_reprs(
            proj_train, proj_eval,
            std_data["train_sens"], std_data[eval_sens_key],
            p.name, p.disallowed_attrs,
        )
        all_reports.extend(reports)

    return all_reports, income_acc if income_acc is not None else 0.0


def leace_sweep(data, std_data_seed0, device):
    print("\n" + "#" * 60)
    print("# LEACE sweep on Adult")
    print("#" * 60)

    reg_grid = [1e-5, 1e-4, 1e-3, 1e-2]
    purpose1 = data["purposes"][0]
    sweep_rows = []
    ranked = []

    for reg in reg_grid:
        t0 = time.time()
        reports, income_val = apply_leace(
            std_data_seed0, data["purposes"], reg, eval_split="val",
        )
        dt = time.time() - t0

        purpose1_reports = [r for r in reports if r.purpose_name == purpose1.name]
        p1_pass = count_pass(purpose1_reports)
        p1_md = mean_delta(purpose1_reports)
        full_pass = count_pass(reports)
        full_md = mean_delta(reports)

        print(f"  reg={reg:.0e}  p1_pass={p1_pass}/2  "
              f"full_pass={full_pass}/{len(reports)}  income_val={income_val:.4f}  "
              f"{dt:.0f}s")

        sweep_rows.append({
            "regularization": reg, "seed": SWEEP_SEED,
            "purpose1_pass_count": p1_pass, "purpose1_total_pairs": 2,
            "purpose1_mean_delta": round(p1_md, 6),
            "val_pass_count": full_pass, "val_total_pairs": len(reports),
            "val_mean_delta": round(full_md, 6),
            "val_income_acc": round(income_val, 6),
            "time_s": round(dt, 1),
        })
        ranked.append(((p1_pass, -p1_md, income_val), reg))

    save_csv(sweep_rows, project_root / "results/adult/leace_sweep.csv")

    ranked.sort(reverse=True)
    winner_reg = ranked[0][1]
    print(f"\n  WINNER: regularization={winner_reg:.0e}")

    print(f"\n  Running winner reg={winner_reg:.0e} on TEST with 3 seeds ...")
    seed_rows = []
    std_cache = {0: std_data_seed0}
    for seed in SEEDS:
        if seed == 0:
            sd = std_data_seed0
        elif seed in std_cache:
            sd = std_cache[seed]
        else:
            print(f"    training Standard seed={seed} ...", flush=True)
            sd = train_standard_encoder(seed, data, device)
            std_cache[seed] = sd
        t0 = time.time()
        reports, income_test = apply_leace(
            sd, data["purposes"], winner_reg, eval_split="test",
        )
        dt = time.time() - t0
        pc = count_pass(reports)
        md = mean_delta(reports)
        print(f"    seed={seed}  test_pass={pc}/{len(reports)}  mean_Δ={md:+.4f}  "
              f"income_test={income_test:.4f}  {dt:.0f}s")
        seed_rows.extend(report_rows(reports, {
            "method": "LEACE", "regularization": winner_reg, "seed": seed,
            "test_income_acc": round(income_test, 6),
            "time_s": round(dt, 1),
        }))

    save_csv(seed_rows, project_root / "results/adult/leace_sweep_winner_seeds.csv")
    return {"method": "LEACE", "hp": winner_reg, "seed_rows": seed_rows,
            "std_cache": std_cache}


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════


def summarize(results_list):
    """Given [{method, hp, seed_rows}], write a summary CSV."""
    summary_rows = []
    for r in results_list:
        method = r["method"]
        hp = r["hp"]
        # seed_rows are per-pair; gather per-seed pass counts
        per_seed = {}
        for row in r["seed_rows"]:
            seed = row["seed"]
            if seed not in per_seed:
                per_seed[seed] = {
                    "pass_count": row["pass_count"],
                    "income_acc": row.get("test_income_acc", 0.0),
                }
        pass_counts = [v["pass_count"] for v in per_seed.values()]
        income_accs = [v["income_acc"] for v in per_seed.values()]
        summary_rows.append({
            "method": method,
            "best_hyperparameter": hp,
            "test_pass_count_mean": round(float(np.mean(pass_counts)), 4),
            "test_pass_count_std": round(float(np.std(pass_counts, ddof=0)), 4),
            "test_income_accuracy_mean": round(float(np.mean(income_accs)), 6),
            "test_income_accuracy_std": round(float(np.std(income_accs, ddof=0)), 6),
            "num_seeds": len(per_seed),
        })
    save_csv(summary_rows, project_root / "results/adult/baseline_sweep_summary.csv")
    return summary_rows


# ═══════════════════════════════════════════════════════════════════════════


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t_total = time.time()

    data = load_adult()
    print(f"Loaded Adult: purposes={[p.name for p in data['purposes']]}")
    print(f"  (purpose 1 = {data['purposes'][0].name}, "
          f"disallowed={data['purposes'][0].disallowed_attrs})")

    # ── LAFTR sweep ─────────────────────────────────────────────────────
    laftr_result = laftr_sweep(data, device)

    # ── Standard encoder seed=0 for INLP/LEACE ─────────────────────────
    print("\n" + "#" * 60)
    print("# Training Standard encoder seed=0 for INLP/LEACE post-hoc sweeps")
    print("#" * 60)
    t0 = time.time()
    std_seed0 = train_standard_encoder(SWEEP_SEED, data, device)
    print(f"  Standard seed=0 trained in {std_seed0['train_time']:.0f}s")

    # ── INLP sweep ─────────────────────────────────────────────────────
    inlp_result = inlp_sweep(data, std_seed0, device)

    # ── LEACE sweep ────────────────────────────────────────────────────
    leace_result = leace_sweep(data, std_seed0, device)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "#" * 60)
    print("# Summary")
    print("#" * 60)
    summary = summarize([laftr_result, inlp_result, leace_result])
    for r in summary:
        print(f"  {r['method']:<6}  hp={r['best_hyperparameter']:<10}  "
              f"test_pass={r['test_pass_count_mean']:.2f}±{r['test_pass_count_std']:.2f}  "
              f"income={r['test_income_accuracy_mean']:.4f}±{r['test_income_accuracy_std']:.4f}")

    itpsex = inlp_result.get("iters_to_hit_pcrl_sex")
    print(f"\n  INLP iters to match PCRL sex_Δ<2%: "
          f"{itpsex if itpsex is not None else 'never in grid [10,25,50,75,100]'}")

    total_min = (time.time() - t_total) / 60
    print(f"\nTotal: {total_min:.1f} min ({total_min/60:.2f} h)")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
