#!/usr/bin/env python3
"""Multi-seed runs for NeurIPS Tables 2 (Adult) and 3 (HAR).

Runs Standard, LAFTR, INLP, LEACE, PCRL with seeds {0, 1, 2} on both
Adult Census and UCI HAR datasets. Outputs:
  - results/adult/adult_seeds.csv
  - results/har_real/har_seeds.csv
  - results/seeds_summary.csv (mean ± std for each table cell)

No hyperparameter changes — frozen config from existing experiments.
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

# Suppress tqdm
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
from pcrl.evaluation.certificates import (
    ComplianceReport,
    EmpiricalAudit,
    LinearAudit,
    generate_report,
)
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.baselines import INLPProjector, LEACEEraser
from pcrl.models.encoder import PurposeConditionedEncoder, StandardEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec
from pcrl.purposes.verification import certified_accuracy_bound
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")

SEEDS = [0, 1, 2]


# ── Shared utilities ─────────────────────────────────────────────────────


@dataclass
class MethodResult:
    name: str
    task_accuracies: dict[str, float] = field(default_factory=dict)
    reports: list[ComplianceReport] = field(default_factory=list)
    train_time: float = 0.0


def extract_representations(
    encoder: torch.nn.Module,
    loader: DataLoader,
    device: str,
    purpose_idx: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    encoder.eval()
    all_reprs, all_tasks, all_sens = [], {}, {}
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            if purpose_idx is not None:
                h = encoder(x, purpose_idx)
            else:
                h = encoder(x)
            all_reprs.append(h.cpu().numpy())
            for k, v in batch["task_labels"].items():
                all_tasks.setdefault(k, []).append(v.numpy())
            for k, v in batch["sensitive_attrs"].items():
                all_sens.setdefault(k, []).append(v.numpy())
    reprs = np.concatenate(all_reprs)
    tasks = {k: np.concatenate(v) for k, v in all_tasks.items()}
    sens = {k: np.concatenate(v) for k, v in all_sens.items()}
    return reprs, tasks, sens


def evaluate_task_accuracy(
    train_reprs: np.ndarray,
    train_labels: np.ndarray,
    test_reprs: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(train_reprs, train_labels)
    return float(clf.score(test_reprs, test_labels))


def run_compliance_on_reprs(
    train_reprs: np.ndarray,
    test_reprs: np.ndarray,
    train_sens: dict[str, np.ndarray],
    test_sens: dict[str, np.ndarray],
    purpose_name: str,
    disallowed_attrs: list[str],
) -> list[ComplianceReport]:
    linear_audit = LinearAudit()
    empirical_audit = EmpiricalAudit()
    reports = []
    for attr_name in disallowed_attrs:
        train_labels = train_sens[attr_name]
        test_labels = test_sens[attr_name]
        linear_result, null_result = linear_audit.audit(test_reprs, test_labels)
        all_labels = np.concatenate([train_labels, test_labels])
        unique_classes, class_counts = np.unique(all_labels, return_counts=True)
        num_classes = len(unique_classes)
        chance_acc = 1.0 / max(num_classes, 1)
        majority_proportion = float(class_counts.max() / len(all_labels))
        best_acc, emp_results = empirical_audit.audit(
            train_reprs, train_labels, test_reprs, test_labels,
        )
        empirical_ok = (best_acc - chance_acc) < 0.05
        certified = linear_result.certified and empirical_ok
        reports.append(ComplianceReport(
            purpose_name=purpose_name,
            attr_name=attr_name,
            linear_r2=linear_result.r_squared,
            linear_certified=linear_result.certified,
            null_space_r2=null_result.r_squared,
            variance_preserved=null_result.variance_preserved or 1.0,
            empirical_best_acc=best_acc,
            empirical_chance_acc=chance_acc,
            empirical_results=emp_results,
            certified=certified,
            majority_proportion=majority_proportion,
            num_classes=num_classes,
        ))
    return reports


def results_to_rows(
    result: MethodResult,
    seed: int,
    dataset: str,
) -> list[dict]:
    rows = []
    pass_count = 0
    total = 0
    for r in result.reports:
        delta = r.empirical_best_acc - r.majority_proportion
        ok = delta < 0.02 and r.linear_r2 < 0.05
        total += 1
        if ok:
            pass_count += 1

    for r in result.reports:
        delta = r.empirical_best_acc - r.majority_proportion
        ok = delta < 0.02 and r.linear_r2 < 0.05
        row = {
            "dataset": dataset,
            "method": result.name,
            "seed": seed,
            "purpose": r.purpose_name,
            "attribute": r.attr_name,
            "best_emp_acc": round(r.empirical_best_acc, 6),
            "majority_baseline": round(r.majority_proportion, 6),
            "delta": round(delta, 6),
            "linear_r2": round(r.linear_r2, 6),
            "adj_pass": ok,
            "pass_count": pass_count,
            "total_pairs": total,
            "train_time_s": round(result.train_time, 1),
        }
        # Add all task accuracies as columns
        for task, acc in sorted(result.task_accuracies.items()):
            row[f"{task}_acc"] = round(acc, 6)
        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# ADULT EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════


def run_adult_seed(seed: int, device: str) -> list[dict]:
    from pcrl.data.adult import AdultDataset, get_adult_purposes

    print(f"\n{'='*60}")
    print(f"ADULT — seed {seed}")
    print(f"{'='*60}")

    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    # Data loading is deterministic — no seed needed
    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False,
                          norm_stats=train_ds.norm_stats)
    test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False,
                           norm_stats=train_ds.norm_stats)

    input_dim = train_ds.info.num_features

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)

    hidden_dims = [128, 128]
    repr_dim = 64
    all_rows = []

    # ── 1. Standard (no privacy) ─────────────────────────────────────────
    print(f"  Standard (no privacy)...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    std_encoder = StandardEncoder(
        input_dim=input_dim, hidden_dims=hidden_dims, repr_dim=repr_dim, dropout=0.3,
    )
    std_task_heads = {}
    std_auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        std_task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        std_auditors[p.name] = MultiAttributeAuditor(
            repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    std_config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=0.0, lambda_verify=0.0, auditor_steps=1,
        epochs=200, weight_decay=1e-4, early_stopping_patience=15,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"seed_std_adult_{seed}"),
    )
    std_trainer = PCRLTrainer(
        encoder=std_encoder, task_heads=std_task_heads, auditors=std_auditors,
        config=std_config, purpose_registry=registry, device=device,
    )
    t0 = time.time()
    std_trainer.train(train_loader, val_loader=val_loader)
    std_time = time.time() - t0
    std_eval = std_trainer.evaluate(test_loader)

    # Extract representations for INLP/LEACE (they use the standard encoder)
    train_reprs_std, train_tasks, train_sens = extract_representations(
        std_encoder, train_loader, device)
    test_reprs_std, test_tasks, test_sens = extract_representations(
        std_encoder, test_loader, device)

    # Standard compliance — use generate_report for consistency
    std_reports = generate_report(
        encoder=std_encoder, train_loader=train_loader, test_loader=test_loader,
        purpose_registry=registry, device=device,
    )
    std_result = MethodResult(
        name="Standard", task_accuracies=std_eval.task_accuracy,
        reports=std_reports, train_time=std_time,
    )
    all_rows.extend(results_to_rows(std_result, seed, "adult"))
    print(f"    income_acc={std_eval.task_accuracy.get('income', 0):.1%}, {std_time:.0f}s")

    # ── 2. LAFTR ──────────────────────────────────────────────────────────
    print(f"  LAFTR...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    laftr_encoder = StandardEncoder(
        input_dim=input_dim, hidden_dims=hidden_dims, repr_dim=repr_dim, dropout=0.3,
    )
    laftr_task_heads = {}
    laftr_auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        laftr_task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        laftr_auditors[p.name] = MultiAttributeAuditor(
            repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    laftr_config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=50.0, lambda_verify=50.0, auditor_steps=10,
        epochs=200, weight_decay=1e-4, early_stopping_patience=15,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"seed_laftr_adult_{seed}"),
    )
    laftr_trainer = PCRLTrainer(
        encoder=laftr_encoder, task_heads=laftr_task_heads, auditors=laftr_auditors,
        config=laftr_config, purpose_registry=registry, device=device,
    )
    t0 = time.time()
    laftr_trainer.train(train_loader, val_loader=val_loader)
    laftr_time = time.time() - t0
    laftr_eval = laftr_trainer.evaluate(test_loader)

    laftr_reports = generate_report(
        encoder=laftr_encoder, train_loader=train_loader, test_loader=test_loader,
        purpose_registry=registry, device=device,
    )
    laftr_result = MethodResult(
        name="LAFTR", task_accuracies=laftr_eval.task_accuracy,
        reports=laftr_reports, train_time=laftr_time,
    )
    all_rows.extend(results_to_rows(laftr_result, seed, "adult"))
    print(f"    income_acc={laftr_eval.task_accuracy.get('income', 0):.1%}, {laftr_time:.0f}s")

    # ── 3. INLP ──────────────────────────────────────────────────────────
    print(f"  INLP...")
    t0 = time.time()
    inlp_reports = []
    inlp_task_accs = {}
    for p in purposes:
        proj_train = train_reprs_std.copy()
        proj_test = test_reprs_std.copy()
        for attr_name in p.disallowed_attrs:
            projector = INLPProjector(max_iters=35, min_accuracy=0.52)
            projector.fit(proj_train, train_sens[attr_name])
            proj_train = projector.transform(proj_train)
            proj_test = projector.transform(proj_test)

        task_name = p.allowed_tasks[0]
        task_acc = evaluate_task_accuracy(
            proj_train, train_tasks[task_name],
            proj_test, test_tasks[task_name],
        )
        inlp_task_accs[task_name] = task_acc

        reports = run_compliance_on_reprs(
            proj_train, proj_test, train_sens, test_sens,
            p.name, p.disallowed_attrs,
        )
        inlp_reports.extend(reports)
    inlp_time = time.time() - t0

    inlp_result = MethodResult(
        name="INLP", task_accuracies=inlp_task_accs,
        reports=inlp_reports, train_time=inlp_time,
    )
    all_rows.extend(results_to_rows(inlp_result, seed, "adult"))
    print(f"    income_acc={inlp_task_accs.get('income', 0):.1%}, {inlp_time:.0f}s")

    # ── 4. LEACE ──────────────────────────────────────────────────────────
    print(f"  LEACE...")
    t0 = time.time()
    leace_reports = []
    leace_task_accs = {}
    for p in purposes:
        proj_train = train_reprs_std.copy()
        proj_test = test_reprs_std.copy()
        for attr_name in p.disallowed_attrs:
            eraser = LEACEEraser()
            eraser.fit(proj_train, train_sens[attr_name])
            proj_train = eraser.transform(proj_train)
            proj_test = eraser.transform(proj_test)

        task_name = p.allowed_tasks[0]
        task_acc = evaluate_task_accuracy(
            proj_train, train_tasks[task_name],
            proj_test, test_tasks[task_name],
        )
        leace_task_accs[task_name] = task_acc

        reports = run_compliance_on_reprs(
            proj_train, proj_test, train_sens, test_sens,
            p.name, p.disallowed_attrs,
        )
        leace_reports.extend(reports)
    leace_time = time.time() - t0

    leace_result = MethodResult(
        name="LEACE", task_accuracies=leace_task_accs,
        reports=leace_reports, train_time=leace_time,
    )
    all_rows.extend(results_to_rows(leace_result, seed, "adult"))
    print(f"    income_acc={leace_task_accs.get('income', 0):.1%}, {leace_time:.0f}s")

    # ── 5. PCRL ───────────────────────────────────────────────────────────
    print(f"  PCRL...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    pcrl_encoder = PurposeConditionedEncoder(
        input_dim=input_dim, hidden_dims=hidden_dims, repr_dim=repr_dim,
        num_purposes=len(purposes), purpose_emb_dim=32,
        conditioning="film", dropout=0.3,
    )
    pcrl_task_heads = {}
    pcrl_auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        pcrl_task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        pcrl_auditors[p.name] = MultiAttributeAuditor(
            repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    pcrl_config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=50.0, lambda_verify=50.0, auditor_steps=10,
        epochs=200, weight_decay=1e-4, early_stopping_patience=15,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"seed_pcrl_adult_{seed}"),
    )
    pcrl_trainer = PCRLTrainer(
        encoder=pcrl_encoder, task_heads=pcrl_task_heads, auditors=pcrl_auditors,
        config=pcrl_config, purpose_registry=registry, device=device,
    )
    t0 = time.time()
    pcrl_trainer.train(train_loader, val_loader=val_loader)
    pcrl_time = time.time() - t0
    pcrl_eval = pcrl_trainer.evaluate(test_loader)

    pcrl_reports = generate_report(
        encoder=pcrl_encoder, train_loader=train_loader, test_loader=test_loader,
        purpose_registry=registry, device=device,
    )
    pcrl_result = MethodResult(
        name="PCRL", task_accuracies=pcrl_eval.task_accuracy,
        reports=pcrl_reports, train_time=pcrl_time,
    )
    all_rows.extend(results_to_rows(pcrl_result, seed, "adult"))
    print(f"    income_acc={pcrl_eval.task_accuracy.get('income', 0):.1%}, {pcrl_time:.0f}s")

    return all_rows


# ═══════════════════════════════════════════════════════════════════════════
# HAR EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════


def run_har_seed(seed: int, device: str) -> list[dict]:
    from pcrl.data.har import HARDataset, get_har_purposes, N_SUBJECTS, N_ACTIVITIES

    print(f"\n{'='*60}")
    print(f"HAR — seed {seed}")
    print(f"{'='*60}")

    purposes = get_har_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = HARDataset(purposes=purposes, split="train")
    val_ds = HARDataset(purposes=purposes, split="val")
    test_ds = HARDataset(purposes=purposes, split="test")

    input_dim = train_ds.info.num_features
    hidden_dims = [128, 128]
    repr_dim = 16

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)

    all_rows = []

    # ── 1. Standard ───────────────────────────────────────────────────────
    print(f"  Standard...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    std_encoder = StandardEncoder(
        input_dim=input_dim, hidden_dims=hidden_dims, repr_dim=repr_dim, dropout=0.3,
    )
    std_task_heads = {}
    std_auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        std_task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        std_auditors[p.name] = MultiAttributeAuditor(
            repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    std_config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=0.0, lambda_verify=0.0, auditor_steps=1,
        epochs=100, weight_decay=1e-4, early_stopping_patience=None,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"seed_std_har_{seed}"),
    )
    std_trainer = PCRLTrainer(
        encoder=std_encoder, task_heads=std_task_heads, auditors=std_auditors,
        config=std_config, purpose_registry=registry, device=device,
    )
    t0 = time.time()
    std_trainer.train(train_loader, val_loader=val_loader)
    std_time = time.time() - t0
    std_eval = std_trainer.evaluate(test_loader)

    # Extract for INLP/LEACE
    train_reprs_std, train_tasks, train_sens = extract_representations(
        std_encoder, train_loader, device)
    test_reprs_std, test_tasks, test_sens = extract_representations(
        std_encoder, test_loader, device)

    std_reports = generate_report(
        encoder=std_encoder, train_loader=train_loader, test_loader=test_loader,
        purpose_registry=registry, device=device,
    )
    std_result = MethodResult(
        name="Standard", task_accuracies=std_eval.task_accuracy,
        reports=std_reports, train_time=std_time,
    )
    all_rows.extend(results_to_rows(std_result, seed, "har"))
    print(f"    activity_acc={std_eval.task_accuracy.get('activity', 0):.1%}, {std_time:.0f}s")

    # ── 2. LAFTR ──────────────────────────────────────────────────────────
    print(f"  LAFTR...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    laftr_encoder = StandardEncoder(
        input_dim=input_dim, hidden_dims=hidden_dims, repr_dim=repr_dim, dropout=0.3,
    )
    laftr_task_heads = {}
    laftr_auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        laftr_task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        laftr_auditors[p.name] = MultiAttributeAuditor(
            repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    laftr_config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=2.0, lambda_verify=1.0, auditor_steps=20,
        epochs=100, weight_decay=1e-4, early_stopping_patience=None,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"seed_laftr_har_{seed}"),
    )
    laftr_trainer = PCRLTrainer(
        encoder=laftr_encoder, task_heads=laftr_task_heads, auditors=laftr_auditors,
        config=laftr_config, purpose_registry=registry, device=device,
    )
    t0 = time.time()
    laftr_trainer.train(train_loader, val_loader=val_loader)
    laftr_time = time.time() - t0
    laftr_eval = laftr_trainer.evaluate(test_loader)

    laftr_reports = generate_report(
        encoder=laftr_encoder, train_loader=train_loader, test_loader=test_loader,
        purpose_registry=registry, device=device,
    )
    laftr_result = MethodResult(
        name="LAFTR", task_accuracies=laftr_eval.task_accuracy,
        reports=laftr_reports, train_time=laftr_time,
    )
    all_rows.extend(results_to_rows(laftr_result, seed, "har"))
    print(f"    activity_acc={laftr_eval.task_accuracy.get('activity', 0):.1%}, {laftr_time:.0f}s")

    # ── 3. INLP ──────────────────────────────────────────────────────────
    print(f"  INLP...")
    t0 = time.time()
    inlp_reports = []
    inlp_task_accs = {}
    for p in purposes:
        proj_train = train_reprs_std.copy()
        proj_test = test_reprs_std.copy()
        for attr_name in p.disallowed_attrs:
            projector = INLPProjector(max_iters=35, min_accuracy=0.52)
            projector.fit(proj_train, train_sens[attr_name])
            proj_train = projector.transform(proj_train)
            proj_test = projector.transform(proj_test)

        task_name = p.allowed_tasks[0]
        task_acc = evaluate_task_accuracy(
            proj_train, train_tasks[task_name],
            proj_test, test_tasks[task_name],
        )
        inlp_task_accs[task_name] = task_acc

        reports = run_compliance_on_reprs(
            proj_train, proj_test, train_sens, test_sens,
            p.name, p.disallowed_attrs,
        )
        inlp_reports.extend(reports)
    inlp_time = time.time() - t0

    inlp_result = MethodResult(
        name="INLP", task_accuracies=inlp_task_accs,
        reports=inlp_reports, train_time=inlp_time,
    )
    all_rows.extend(results_to_rows(inlp_result, seed, "har"))
    print(f"    activity_acc={inlp_task_accs.get('activity', 0):.1%}, {inlp_time:.0f}s")

    # ── 4. LEACE ──────────────────────────────────────────────────────────
    print(f"  LEACE...")
    t0 = time.time()
    leace_reports = []
    leace_task_accs = {}
    for p in purposes:
        proj_train = train_reprs_std.copy()
        proj_test = test_reprs_std.copy()
        for attr_name in p.disallowed_attrs:
            eraser = LEACEEraser()
            eraser.fit(proj_train, train_sens[attr_name])
            proj_train = eraser.transform(proj_train)
            proj_test = eraser.transform(proj_test)

        task_name = p.allowed_tasks[0]
        task_acc = evaluate_task_accuracy(
            proj_train, train_tasks[task_name],
            proj_test, test_tasks[task_name],
        )
        leace_task_accs[task_name] = task_acc

        reports = run_compliance_on_reprs(
            proj_train, proj_test, train_sens, test_sens,
            p.name, p.disallowed_attrs,
        )
        leace_reports.extend(reports)
    leace_time = time.time() - t0

    leace_result = MethodResult(
        name="LEACE", task_accuracies=leace_task_accs,
        reports=leace_reports, train_time=leace_time,
    )
    all_rows.extend(results_to_rows(leace_result, seed, "har"))
    print(f"    activity_acc={leace_task_accs.get('activity', 0):.1%}, {leace_time:.0f}s")

    # ── 5. PCRL ───────────────────────────────────────────────────────────
    print(f"  PCRL...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    pcrl_encoder = PurposeConditionedEncoder(
        input_dim=input_dim, hidden_dims=hidden_dims, repr_dim=repr_dim,
        num_purposes=len(purposes), purpose_emb_dim=32,
        conditioning="film", dropout=0.3,
    )
    pcrl_task_heads = {}
    pcrl_auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        pcrl_task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        pcrl_auditors[p.name] = MultiAttributeAuditor(
            repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    pcrl_config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=2.0, lambda_verify=1.0, auditor_steps=20,
        epochs=100, weight_decay=1e-4, early_stopping_patience=None,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"seed_pcrl_har_{seed}"),
    )
    pcrl_trainer = PCRLTrainer(
        encoder=pcrl_encoder, task_heads=pcrl_task_heads, auditors=pcrl_auditors,
        config=pcrl_config, purpose_registry=registry, device=device,
    )
    t0 = time.time()
    pcrl_trainer.train(train_loader, val_loader=val_loader)
    pcrl_time = time.time() - t0
    pcrl_eval = pcrl_trainer.evaluate(test_loader)

    pcrl_reports = generate_report(
        encoder=pcrl_encoder, train_loader=train_loader, test_loader=test_loader,
        purpose_registry=registry, device=device,
    )
    pcrl_result = MethodResult(
        name="PCRL", task_accuracies=pcrl_eval.task_accuracy,
        reports=pcrl_reports, train_time=pcrl_time,
    )
    all_rows.extend(results_to_rows(pcrl_result, seed, "har"))
    print(f"    activity_acc={pcrl_eval.task_accuracy.get('activity', 0):.1%}, {pcrl_time:.0f}s")

    return all_rows


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════


def compute_summary(all_rows: list[dict]) -> list[dict]:
    """Compute mean ± std for each (dataset, method, metric) triple."""
    from collections import defaultdict

    # Group by (dataset, method)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in all_rows:
        key = (row["dataset"], row["method"])
        groups[key].append(row)

    summary_rows = []
    for (dataset, method), rows in sorted(groups.items()):
        # Group by seed to get per-seed aggregates
        seed_data: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            seed_data[row["seed"]].append(row)

        # Compute per-seed metrics
        seed_metrics = []
        for seed_val, seed_rows in sorted(seed_data.items()):
            # Task accuracies — get from any row (same for all pairs in a seed)
            task_acc_cols = [k for k in seed_rows[0].keys() if k.endswith("_acc")]
            task_accs = {col: seed_rows[0][col] for col in task_acc_cols}

            # Per-attribute worst-case delta (max across purposes)
            attr_deltas: dict[str, float] = {}
            for row in seed_rows:
                attr = row["attribute"]
                attr_deltas[attr] = max(attr_deltas.get(attr, -1.0), row["delta"])

            # Pass count (from any row — same for all in a seed)
            pass_count = seed_rows[0]["pass_count"]
            total_pairs = seed_rows[0]["total_pairs"]

            seed_metrics.append({
                **task_accs,
                **{f"delta_{attr}": d for attr, d in attr_deltas.items()},
                "pass_count": pass_count,
                "total_pairs": total_pairs,
            })

        # Compute mean ± std across seeds
        summary = {"dataset": dataset, "method": method, "n_seeds": len(seed_metrics)}
        all_keys = set()
        for sm in seed_metrics:
            all_keys.update(sm.keys())

        for key in sorted(all_keys):
            vals = [sm.get(key, float("nan")) for sm in seed_metrics]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals, ddof=0)  # population std for 3 seeds
                summary[f"{key}_mean"] = round(float(mean), 4)
                summary[f"{key}_std"] = round(float(std), 4)

        summary_rows.append(summary)

    return summary_rows


def save_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Collect all fieldnames across all rows
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def print_summary_table(summary_rows: list[dict]) -> None:
    """Print mean ± std tables for Adult and HAR."""
    for dataset in ["adult", "har"]:
        ds_rows = [r for r in summary_rows if r["dataset"] == dataset]
        if not ds_rows:
            continue

        print(f"\n{'='*80}")
        if dataset == "adult":
            print("TABLE 2: Adult Census (mean ± std over 3 seeds)")
            print(f"{'='*80}")
            print(f"{'Method':<12} {'Income Acc':>14} {'Race Δ':>14} "
                  f"{'Sex Δ':>14} {'Marital Δ':>14} {'Pass':>10}")
            print("-" * 80)
            for r in ds_rows:
                inc = f"{r.get('income_acc_mean', 0):.1%}±{r.get('income_acc_std', 0):.1%}"
                race = f"{r.get('delta_race_mean', 0):+.1%}±{r.get('delta_race_std', 0):.1%}"
                sex = f"{r.get('delta_sex_mean', 0):+.1%}±{r.get('delta_sex_std', 0):.1%}"
                mar = f"{r.get('delta_marital_status_mean', 0):+.1%}±{r.get('delta_marital_status_std', 0):.1%}"
                pc = f"{r.get('pass_count_mean', 0):.1f}±{r.get('pass_count_std', 0):.1f}"
                print(f"{r['method']:<12} {inc:>14} {race:>14} {sex:>14} {mar:>14} {pc:>10}")
        else:
            print("TABLE 3: UCI HAR (mean ± std over 3 seeds)")
            print(f"{'='*80}")
            print(f"{'Method':<12} {'Activity Acc':>16} {'Subject Δ':>16} {'Pass':>10}")
            print("-" * 80)
            for r in ds_rows:
                act = f"{r.get('activity_acc_mean', 0):.1%}±{r.get('activity_acc_std', 0):.1%}"
                subj = f"{r.get('delta_subject_id_mean', 0):+.1%}±{r.get('delta_subject_id_std', 0):.1%}"
                pc = f"{r.get('pass_count_mean', 0):.1f}±{r.get('pass_count_std', 0):.1f}"
                print(f"{r['method']:<12} {act:>16} {subj:>16} {pc:>10}")
        print(f"{'='*80}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")

    all_adult_rows = []
    all_har_rows = []

    for seed in SEEDS:
        all_adult_rows.extend(run_adult_seed(seed, device))
        all_har_rows.extend(run_har_seed(seed, device))

    all_rows = all_adult_rows + all_har_rows

    # Save per-dataset CSVs
    save_rows(all_adult_rows, project_root / "results" / "adult" / "adult_seeds.csv")
    save_rows(all_har_rows, project_root / "results" / "har_real" / "har_seeds.csv")

    # Summary
    summary = compute_summary(all_rows)
    save_rows(summary, project_root / "results" / "seeds_summary.csv")

    print_summary_table(summary)

    print("\nDone!")


if __name__ == "__main__":
    main()
