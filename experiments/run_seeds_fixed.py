#!/usr/bin/env python3
"""Re-run multi-seed experiments with the generate_report() fix.

Saves to *_seeds_fixed.csv alongside the buggy originals.
Runs: Adult (3 seeds), HAR (3 seeds), CelebA v2 (1 run).
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
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")

SEEDS = [0, 1, 2]


@dataclass
class MethodResult:
    name: str
    task_accuracies: dict[str, float] = field(default_factory=dict)
    reports: list[ComplianceReport] = field(default_factory=list)
    train_time: float = 0.0


def extract_representations(
    encoder: torch.nn.Module, loader: DataLoader, device: str, purpose_idx: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    encoder.eval()
    all_reprs, all_tasks, all_sens = [], {}, {}
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder(x, purpose_idx) if purpose_idx is not None else encoder(x)
            all_reprs.append(h.cpu().numpy())
            for k, v in batch["task_labels"].items():
                all_tasks.setdefault(k, []).append(v.numpy())
            for k, v in batch["sensitive_attrs"].items():
                all_sens.setdefault(k, []).append(v.numpy())
    return np.concatenate(all_reprs), {k: np.concatenate(v) for k, v in all_tasks.items()}, {k: np.concatenate(v) for k, v in all_sens.items()}


def evaluate_task_accuracy(train_reprs, train_labels, test_reprs, test_labels) -> float:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(train_reprs, train_labels)
    return float(clf.score(test_reprs, test_labels))


def run_compliance_on_reprs(
    train_reprs, test_reprs, train_sens, test_sens, purpose_name, disallowed_attrs,
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
        best_acc, emp_results = empirical_audit.audit(train_reprs, train_labels, test_reprs, test_labels)
        empirical_ok = (best_acc - chance_acc) < 0.05
        certified = linear_result.certified and empirical_ok
        reports.append(ComplianceReport(
            purpose_name=purpose_name, attr_name=attr_name,
            linear_r2=linear_result.r_squared, linear_certified=linear_result.certified,
            null_space_r2=null_result.r_squared, variance_preserved=null_result.variance_preserved or 1.0,
            empirical_best_acc=best_acc, empirical_chance_acc=chance_acc, empirical_results=emp_results,
            certified=certified, majority_proportion=majority_proportion, num_classes=num_classes,
        ))
    return reports


def results_to_rows(result: MethodResult, seed: int, dataset: str) -> list[dict]:
    rows = []
    pass_count = 0
    total = len(result.reports)
    for r in result.reports:
        delta = r.empirical_best_acc - r.majority_proportion
        if delta < 0.02 and r.linear_r2 < 0.05:
            pass_count += 1

    for r in result.reports:
        delta = r.empirical_best_acc - r.majority_proportion
        ok = delta < 0.02 and r.linear_r2 < 0.05
        row = {
            "dataset": dataset, "method": result.name, "seed": seed,
            "purpose": r.purpose_name, "attribute": r.attr_name,
            "best_emp_acc": round(r.empirical_best_acc, 6),
            "majority_baseline": round(r.majority_proportion, 6),
            "delta": round(delta, 6), "linear_r2": round(r.linear_r2, 6),
            "adj_pass": ok, "pass_count": pass_count, "total_pairs": total,
            "train_time_s": round(result.train_time, 1),
        }
        for task, acc in sorted(result.task_accuracies.items()):
            row[f"{task}_acc"] = round(acc, 6)
        rows.append(row)
    return rows


def save_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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


# ═══════════════════════════════════════════════════════════════════════════
# ADULT
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

    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False, norm_stats=train_ds.norm_stats)
    test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False, norm_stats=train_ds.norm_stats)
    input_dim = train_ds.info.num_features

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)

    hidden_dims = [128, 128]
    repr_dim = 64
    all_rows = []

    # ── 1. Standard ──────────────────────────────────────────────────────
    print(f"  Standard...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    std_encoder = StandardEncoder(input_dim=input_dim, hidden_dims=hidden_dims, repr_dim=repr_dim, dropout=0.3)
    std_task_heads, std_auditors = {}, {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        std_task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        std_auditors[p.name] = MultiAttributeAuditor(repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims, hidden_dim=256, num_layers=3)

    std_config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3, lambda_adv=0.0, lambda_verify=0.0,
        auditor_steps=1, epochs=200, weight_decay=1e-4, early_stopping_patience=15,
        confusion_type="entropy", checkpoint_dir=str(project_root / "checkpoints" / f"fix_std_adult_{seed}"),
    )
    std_trainer = PCRLTrainer(encoder=std_encoder, task_heads=std_task_heads, auditors=std_auditors,
                              config=std_config, purpose_registry=registry, device=device)
    t0 = time.time()
    std_trainer.train(train_loader, val_loader=val_loader)
    std_time = time.time() - t0
    std_eval = std_trainer.evaluate(test_loader)

    # Extract for INLP/LEACE
    train_reprs_std, train_tasks, train_sens = extract_representations(std_encoder, train_loader, device)
    test_reprs_std, test_tasks, test_sens = extract_representations(std_encoder, test_loader, device)

    # Standard compliance — uses fixed generate_report
    std_reports = generate_report(encoder=std_encoder, train_loader=train_loader, test_loader=test_loader,
                                  purpose_registry=registry, device=device)
    std_result = MethodResult(name="Standard", task_accuracies=std_eval.task_accuracy, reports=std_reports, train_time=std_time)
    all_rows.extend(results_to_rows(std_result, seed, "adult"))

    # Quick smoke check
    sex_deltas = [r.empirical_best_acc - r.majority_proportion for r in std_reports if r.attr_name == "sex"]
    avg_sex = sum(sex_deltas) / len(sex_deltas) if sex_deltas else 0
    print(f"    income_acc={std_eval.task_accuracy.get('income', 0):.1%}, Sex Δ={avg_sex:+.1%}, {std_time:.0f}s")

    # ── 2. LAFTR ─────────────────────────────────────────────────────────
    print(f"  LAFTR...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    laftr_encoder = StandardEncoder(input_dim=input_dim, hidden_dims=hidden_dims, repr_dim=repr_dim, dropout=0.3)
    laftr_task_heads, laftr_auditors = {}, {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        laftr_task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        laftr_auditors[p.name] = MultiAttributeAuditor(repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims, hidden_dim=256, num_layers=3)

    laftr_config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3, lambda_adv=50.0, lambda_verify=50.0,
        auditor_steps=10, epochs=200, weight_decay=1e-4, early_stopping_patience=15,
        confusion_type="entropy", checkpoint_dir=str(project_root / "checkpoints" / f"fix_laftr_adult_{seed}"),
    )
    laftr_trainer = PCRLTrainer(encoder=laftr_encoder, task_heads=laftr_task_heads, auditors=laftr_auditors,
                                config=laftr_config, purpose_registry=registry, device=device)
    t0 = time.time()
    laftr_trainer.train(train_loader, val_loader=val_loader)
    laftr_time = time.time() - t0
    laftr_eval = laftr_trainer.evaluate(test_loader)

    laftr_reports = generate_report(encoder=laftr_encoder, train_loader=train_loader, test_loader=test_loader,
                                    purpose_registry=registry, device=device)
    laftr_result = MethodResult(name="LAFTR", task_accuracies=laftr_eval.task_accuracy, reports=laftr_reports, train_time=laftr_time)
    all_rows.extend(results_to_rows(laftr_result, seed, "adult"))
    print(f"    income_acc={laftr_eval.task_accuracy.get('income', 0):.1%}, {laftr_time:.0f}s")

    # ── 3. INLP ─────────────────────────────────────────────────────────
    print(f"  INLP...")
    t0 = time.time()
    inlp_reports, inlp_task_accs = [], {}
    for p in purposes:
        proj_train, proj_test = train_reprs_std.copy(), test_reprs_std.copy()
        for attr_name in p.disallowed_attrs:
            projector = INLPProjector(max_iters=35, min_accuracy=0.52)
            projector.fit(proj_train, train_sens[attr_name])
            proj_train = projector.transform(proj_train)
            proj_test = projector.transform(proj_test)
        task_name = p.allowed_tasks[0]
        inlp_task_accs[task_name] = evaluate_task_accuracy(proj_train, train_tasks[task_name], proj_test, test_tasks[task_name])
        inlp_reports.extend(run_compliance_on_reprs(proj_train, proj_test, train_sens, test_sens, p.name, p.disallowed_attrs))
    inlp_time = time.time() - t0
    inlp_result = MethodResult(name="INLP", task_accuracies=inlp_task_accs, reports=inlp_reports, train_time=inlp_time)
    all_rows.extend(results_to_rows(inlp_result, seed, "adult"))
    print(f"    income_acc={inlp_task_accs.get('income', 0):.1%}, {inlp_time:.0f}s")

    # ── 4. LEACE ─────────────────────────────────────────────────────────
    print(f"  LEACE...")
    t0 = time.time()
    leace_reports, leace_task_accs = [], {}
    for p in purposes:
        proj_train, proj_test = train_reprs_std.copy(), test_reprs_std.copy()
        for attr_name in p.disallowed_attrs:
            eraser = LEACEEraser()
            eraser.fit(proj_train, train_sens[attr_name])
            proj_train = eraser.transform(proj_train)
            proj_test = eraser.transform(proj_test)
        task_name = p.allowed_tasks[0]
        leace_task_accs[task_name] = evaluate_task_accuracy(proj_train, train_tasks[task_name], proj_test, test_tasks[task_name])
        leace_reports.extend(run_compliance_on_reprs(proj_train, proj_test, train_sens, test_sens, p.name, p.disallowed_attrs))
    leace_time = time.time() - t0
    leace_result = MethodResult(name="LEACE", task_accuracies=leace_task_accs, reports=leace_reports, train_time=leace_time)
    all_rows.extend(results_to_rows(leace_result, seed, "adult"))
    print(f"    income_acc={leace_task_accs.get('income', 0):.1%}, {leace_time:.0f}s")

    # ── 5. PCRL ──────────────────────────────────────────────────────────
    print(f"  PCRL...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    pcrl_encoder = PurposeConditionedEncoder(
        input_dim=input_dim, hidden_dims=hidden_dims, repr_dim=repr_dim,
        num_purposes=len(purposes), purpose_emb_dim=32, conditioning="film", dropout=0.3,
    )
    pcrl_task_heads, pcrl_auditors = {}, {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        pcrl_task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        pcrl_auditors[p.name] = MultiAttributeAuditor(repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims, hidden_dim=256, num_layers=3)

    pcrl_config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3, lambda_adv=50.0, lambda_verify=50.0,
        auditor_steps=10, epochs=200, weight_decay=1e-4, early_stopping_patience=15,
        confusion_type="entropy", checkpoint_dir=str(project_root / "checkpoints" / f"fix_pcrl_adult_{seed}"),
    )
    pcrl_trainer = PCRLTrainer(encoder=pcrl_encoder, task_heads=pcrl_task_heads, auditors=pcrl_auditors,
                               config=pcrl_config, purpose_registry=registry, device=device)
    t0 = time.time()
    pcrl_trainer.train(train_loader, val_loader=val_loader)
    pcrl_time = time.time() - t0
    pcrl_eval = pcrl_trainer.evaluate(test_loader)

    pcrl_reports = generate_report(encoder=pcrl_encoder, train_loader=train_loader, test_loader=test_loader,
                                   purpose_registry=registry, device=device)
    pcrl_result = MethodResult(name="PCRL", task_accuracies=pcrl_eval.task_accuracy, reports=pcrl_reports, train_time=pcrl_time)
    all_rows.extend(results_to_rows(pcrl_result, seed, "adult"))
    print(f"    income_acc={pcrl_eval.task_accuracy.get('income', 0):.1%}, {pcrl_time:.0f}s")

    return all_rows


# ═══════════════════════════════════════════════════════════════════════════
# HAR
# ═══════════════════════════════════════════════════════════════════════════

def run_har_seed(seed: int, device: str) -> list[dict]:
    from pcrl.data.har import HARDataset, get_har_purposes

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

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch, num_workers=0)

    all_rows = []

    # ── 1. Standard ──────────────────────────────────────────────────────
    print(f"  Standard...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    std_encoder = StandardEncoder(input_dim=input_dim, hidden_dims=hidden_dims, repr_dim=repr_dim, dropout=0.3)
    std_task_heads, std_auditors = {}, {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        std_task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        std_auditors[p.name] = MultiAttributeAuditor(repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims, hidden_dim=256, num_layers=3)

    std_config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3, lambda_adv=0.0, lambda_verify=0.0,
        auditor_steps=1, epochs=100, weight_decay=1e-4, early_stopping_patience=None,
        confusion_type="entropy", checkpoint_dir=str(project_root / "checkpoints" / f"fix_std_har_{seed}"),
    )
    std_trainer = PCRLTrainer(encoder=std_encoder, task_heads=std_task_heads, auditors=std_auditors,
                              config=std_config, purpose_registry=registry, device=device)
    t0 = time.time()
    std_trainer.train(train_loader, val_loader=val_loader)
    std_time = time.time() - t0
    std_eval = std_trainer.evaluate(test_loader)

    train_reprs_std, train_tasks, train_sens = extract_representations(std_encoder, train_loader, device)
    test_reprs_std, test_tasks, test_sens = extract_representations(std_encoder, test_loader, device)

    std_reports = generate_report(encoder=std_encoder, train_loader=train_loader, test_loader=test_loader,
                                  purpose_registry=registry, device=device)
    std_result = MethodResult(name="Standard", task_accuracies=std_eval.task_accuracy, reports=std_reports, train_time=std_time)
    all_rows.extend(results_to_rows(std_result, seed, "har"))

    subj_deltas = [r.empirical_best_acc - r.majority_proportion for r in std_reports if r.attr_name == "subject_id"]
    avg_subj = sum(subj_deltas) / len(subj_deltas) if subj_deltas else 0
    print(f"    activity_acc={std_eval.task_accuracy.get('activity', 0):.1%}, Subject Δ={avg_subj:+.1%}, {std_time:.0f}s")

    # ── 2. LAFTR ─────────────────────────────────────────────────────────
    print(f"  LAFTR...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    laftr_encoder = StandardEncoder(input_dim=input_dim, hidden_dims=hidden_dims, repr_dim=repr_dim, dropout=0.3)
    laftr_task_heads, laftr_auditors = {}, {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        laftr_task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        laftr_auditors[p.name] = MultiAttributeAuditor(repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims, hidden_dim=256, num_layers=3)

    laftr_config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3, lambda_adv=2.0, lambda_verify=1.0,
        auditor_steps=20, epochs=100, weight_decay=1e-4, early_stopping_patience=None,
        confusion_type="entropy", checkpoint_dir=str(project_root / "checkpoints" / f"fix_laftr_har_{seed}"),
    )
    laftr_trainer = PCRLTrainer(encoder=laftr_encoder, task_heads=laftr_task_heads, auditors=laftr_auditors,
                                config=laftr_config, purpose_registry=registry, device=device)
    t0 = time.time()
    laftr_trainer.train(train_loader, val_loader=val_loader)
    laftr_time = time.time() - t0
    laftr_eval = laftr_trainer.evaluate(test_loader)

    laftr_reports = generate_report(encoder=laftr_encoder, train_loader=train_loader, test_loader=test_loader,
                                    purpose_registry=registry, device=device)
    laftr_result = MethodResult(name="LAFTR", task_accuracies=laftr_eval.task_accuracy, reports=laftr_reports, train_time=laftr_time)
    all_rows.extend(results_to_rows(laftr_result, seed, "har"))
    print(f"    activity_acc={laftr_eval.task_accuracy.get('activity', 0):.1%}, {laftr_time:.0f}s")

    # ── 3. INLP ─────────────────────────────────────────────────────────
    print(f"  INLP...")
    t0 = time.time()
    inlp_reports, inlp_task_accs = [], {}
    for p in purposes:
        proj_train, proj_test = train_reprs_std.copy(), test_reprs_std.copy()
        for attr_name in p.disallowed_attrs:
            projector = INLPProjector(max_iters=35, min_accuracy=0.52)
            projector.fit(proj_train, train_sens[attr_name])
            proj_train = projector.transform(proj_train)
            proj_test = projector.transform(proj_test)
        task_name = p.allowed_tasks[0]
        inlp_task_accs[task_name] = evaluate_task_accuracy(proj_train, train_tasks[task_name], proj_test, test_tasks[task_name])
        inlp_reports.extend(run_compliance_on_reprs(proj_train, proj_test, train_sens, test_sens, p.name, p.disallowed_attrs))
    inlp_time = time.time() - t0
    inlp_result = MethodResult(name="INLP", task_accuracies=inlp_task_accs, reports=inlp_reports, train_time=inlp_time)
    all_rows.extend(results_to_rows(inlp_result, seed, "har"))
    print(f"    activity_acc={inlp_task_accs.get('activity', 0):.1%}, {inlp_time:.0f}s")

    # ── 4. LEACE ─────────────────────────────────────────────────────────
    print(f"  LEACE...")
    t0 = time.time()
    leace_reports, leace_task_accs = [], {}
    for p in purposes:
        proj_train, proj_test = train_reprs_std.copy(), test_reprs_std.copy()
        for attr_name in p.disallowed_attrs:
            eraser = LEACEEraser()
            eraser.fit(proj_train, train_sens[attr_name])
            proj_train = eraser.transform(proj_train)
            proj_test = eraser.transform(proj_test)
        task_name = p.allowed_tasks[0]
        leace_task_accs[task_name] = evaluate_task_accuracy(proj_train, train_tasks[task_name], proj_test, test_tasks[task_name])
        leace_reports.extend(run_compliance_on_reprs(proj_train, proj_test, train_sens, test_sens, p.name, p.disallowed_attrs))
    leace_time = time.time() - t0
    leace_result = MethodResult(name="LEACE", task_accuracies=leace_task_accs, reports=leace_reports, train_time=leace_time)
    all_rows.extend(results_to_rows(leace_result, seed, "har"))
    print(f"    activity_acc={leace_task_accs.get('activity', 0):.1%}, {leace_time:.0f}s")

    # ── 5. PCRL ──────────────────────────────────────────────────────────
    print(f"  PCRL...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    pcrl_encoder = PurposeConditionedEncoder(
        input_dim=input_dim, hidden_dims=hidden_dims, repr_dim=repr_dim,
        num_purposes=len(purposes), purpose_emb_dim=32, conditioning="film", dropout=0.3,
    )
    pcrl_task_heads, pcrl_auditors = {}, {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        pcrl_task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        pcrl_auditors[p.name] = MultiAttributeAuditor(repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims, hidden_dim=256, num_layers=3)

    pcrl_config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3, lambda_adv=2.0, lambda_verify=1.0,
        auditor_steps=20, epochs=100, weight_decay=1e-4, early_stopping_patience=None,
        confusion_type="entropy", checkpoint_dir=str(project_root / "checkpoints" / f"fix_pcrl_har_{seed}"),
    )
    pcrl_trainer = PCRLTrainer(encoder=pcrl_encoder, task_heads=pcrl_task_heads, auditors=pcrl_auditors,
                               config=pcrl_config, purpose_registry=registry, device=device)
    t0 = time.time()
    pcrl_trainer.train(train_loader, val_loader=val_loader)
    pcrl_time = time.time() - t0
    pcrl_eval = pcrl_trainer.evaluate(test_loader)

    pcrl_reports = generate_report(encoder=pcrl_encoder, train_loader=train_loader, test_loader=test_loader,
                                   purpose_registry=registry, device=device)
    pcrl_result = MethodResult(name="PCRL", task_accuracies=pcrl_eval.task_accuracy, reports=pcrl_reports, train_time=pcrl_time)
    all_rows.extend(results_to_rows(pcrl_result, seed, "har"))
    print(f"    activity_acc={pcrl_eval.task_accuracy.get('activity', 0):.1%}, {pcrl_time:.0f}s")

    return all_rows


# ═══════════════════════════════════════════════════════════════════════════
# CelebA v2
# ═══════════════════════════════════════════════════════════════════════════

def run_celeba_v2(device: str) -> list[dict]:
    import torch.nn as nn
    from pcrl.data.celeba import CelebADataset, get_celeba_purposes
    from pcrl.models.cnn_encoder import CNNPurposeProjectionEncoder

    print(f"\n{'='*60}")
    print(f"CelebA v2")
    print(f"{'='*60}")

    torch.manual_seed(42)

    purposes = get_celeba_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = CelebADataset(purposes, root="data/celeba", split="train", max_samples=10000)
    val_ds = CelebADataset(purposes, root="data/celeba", split="val", max_samples=3000)
    test_ds = CelebADataset(purposes, root="data/celeba", split="test", max_samples=3000)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pcrl_batch)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_pcrl_batch)

    REPR_DIM = 128
    CONV_CHANNELS = (32, 64, 128)

    encoder = CNNPurposeProjectionEncoder(
        repr_dim=REPR_DIM, num_purposes=len(purposes), conv_channels=CONV_CHANNELS,
        dropout=0.3, backbone_grad_scale=1.0,
    )

    class MultiTaskHead(nn.Module):
        def __init__(self, heads):
            super().__init__()
            self.heads = nn.ModuleDict(heads)
        def forward(self, x):
            return {name: head(x) for name, head in self.heads.items()}

    task_heads = {}
    for p in purposes:
        if len(p.allowed_tasks) == 1:
            task_name = p.allowed_tasks[0]
            output_dim = p.allowed_task_dims.get(task_name, 2)
            task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
        else:
            sub = {}
            for task_name in p.allowed_tasks:
                output_dim = p.allowed_task_dims.get(task_name, 2)
                sub[task_name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
            task_heads[p.name] = MultiTaskHead(sub)

    auditors = {}
    for p in purposes:
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM, attr_output_dims=p.disallowed_attr_dims, hidden_dim=256, num_layers=3,
        )

    config = TrainerConfig(
        batch_size=256, lr_encoder=1e-3, lr_auditor=1e-3, lambda_adv=0.5, lambda_verify=0.3,
        auditor_steps=5, epochs=50, weight_decay=1e-4, early_stopping_patience=None,
        log_interval=100, confusion_type="entropy", warmup_epochs=5,
        gradient_reversal=False, sequential_purposes=False,
        checkpoint_dir=str(project_root / "checkpoints" / "celeba_v2_fixed"),
    )

    trainer = PCRLTrainer(encoder=encoder, task_heads=task_heads, auditors=auditors,
                          config=config, purpose_registry=registry, device=device)

    print("  Training...")
    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0
    print(f"  Training complete — epoch {state.epoch + 1}, {train_time:.0f}s")

    eval_metrics = trainer.evaluate(test_loader)
    print(f"  Task accuracies:")
    for task, acc in sorted(eval_metrics.task_accuracy.items()):
        print(f"    {task}: {acc:.1%}")

    reports = generate_report(encoder=encoder, train_loader=train_loader, test_loader=test_loader,
                              purpose_registry=registry, device=device)

    # Build rows
    all_rows = []
    pass_count = sum(1 for r in reports if (r.empirical_best_acc - r.majority_proportion) < 0.02 and r.linear_r2 < 0.05)
    total = len(reports)

    for r in reports:
        delta = r.empirical_best_acc - r.majority_proportion
        ok = delta < 0.02 and r.linear_r2 < 0.05
        row = {
            "purpose": r.purpose_name, "attribute": r.attr_name,
            "linear_r2": round(r.linear_r2, 6), "linear_pass": r.linear_certified,
            "empirical_best_acc": round(r.empirical_best_acc, 4),
            "majority_baseline": round(r.majority_proportion, 4),
            "delta": round(delta, 4), "adj_pass": ok,
            "nonlinear_bound": round(r.nonlinear_bound, 4) if r.nonlinear_bound else None,
            "train_time_s": round(train_time, 1),
        }
        all_rows.append(row)

    print(f"\n  Adjusted compliance: {pass_count}/{total}")
    for r in reports:
        delta = r.empirical_best_acc - r.majority_proportion
        ok = delta < 0.02 and r.linear_r2 < 0.05
        status = "PASS" if ok else "FAIL"
        print(f"    {r.purpose_name:30s} / {r.attr_name:12s}  delta={delta:+.1%}  R²={r.linear_r2:.4f}  {status}")

    return all_rows


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def compute_summary(all_rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in all_rows:
        key = (row["dataset"], row["method"])
        groups[key].append(row)

    summary_rows = []
    for (dataset, method), rows in sorted(groups.items()):
        seed_data: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            seed_data[row["seed"]].append(row)

        seed_metrics = []
        for seed_val, seed_rows in sorted(seed_data.items()):
            task_acc_cols = [k for k in seed_rows[0].keys() if k.endswith("_acc")]
            task_accs = {col: seed_rows[0][col] for col in task_acc_cols}
            attr_deltas: dict[str, float] = {}
            for row in seed_rows:
                attr = row["attribute"]
                attr_deltas[attr] = max(attr_deltas.get(attr, -1.0), row["delta"])
            pass_count = seed_rows[0]["pass_count"]
            total_pairs = seed_rows[0]["total_pairs"]
            seed_metrics.append({**task_accs, **{f"delta_{attr}": d for attr, d in attr_deltas.items()},
                                 "pass_count": pass_count, "total_pairs": total_pairs})

        summary = {"dataset": dataset, "method": method, "n_seeds": len(seed_metrics)}
        all_keys = set()
        for sm in seed_metrics:
            all_keys.update(sm.keys())
        for key in sorted(all_keys):
            vals = [sm.get(key, float("nan")) for sm in seed_metrics]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if vals:
                summary[f"{key}_mean"] = round(float(np.mean(vals)), 4)
                summary[f"{key}_std"] = round(float(np.std(vals, ddof=0)), 4)
        summary_rows.append(summary)
    return summary_rows


def print_summary_table(summary_rows: list[dict]) -> None:
    for dataset in ["adult", "har"]:
        ds_rows = [r for r in summary_rows if r["dataset"] == dataset]
        if not ds_rows:
            continue
        print(f"\n{'='*80}")
        if dataset == "adult":
            print("TABLE 2: Adult Census (mean +/- std over 3 seeds)")
            print(f"{'='*80}")
            print(f"{'Method':<12} {'Income Acc':>14} {'Race D':>14} {'Sex D':>14} {'Marital D':>14} {'Pass':>10}")
            print("-" * 80)
            for r in ds_rows:
                inc = f"{r.get('income_acc_mean', 0):.1%}+/-{r.get('income_acc_std', 0):.1%}"
                race = f"{r.get('delta_race_mean', 0):+.1%}+/-{r.get('delta_race_std', 0):.1%}"
                sex = f"{r.get('delta_sex_mean', 0):+.1%}+/-{r.get('delta_sex_std', 0):.1%}"
                mar = f"{r.get('delta_marital_status_mean', 0):+.1%}+/-{r.get('delta_marital_status_std', 0):.1%}"
                pc = f"{r.get('pass_count_mean', 0):.1f}+/-{r.get('pass_count_std', 0):.1f}"
                print(f"{r['method']:<12} {inc:>14} {race:>14} {sex:>14} {mar:>14} {pc:>10}")
        else:
            print("TABLE 3: UCI HAR (mean +/- std over 3 seeds)")
            print(f"{'='*80}")
            print(f"{'Method':<12} {'Activity Acc':>16} {'Subject D':>16} {'Pass':>10}")
            print("-" * 80)
            for r in ds_rows:
                act = f"{r.get('activity_acc_mean', 0):.1%}+/-{r.get('activity_acc_std', 0):.1%}"
                subj = f"{r.get('delta_subject_id_mean', 0):+.1%}+/-{r.get('delta_subject_id_std', 0):.1%}"
                pc = f"{r.get('pass_count_mean', 0):.1f}+/-{r.get('pass_count_std', 0):.1f}"
                print(f"{r['method']:<12} {act:>16} {subj:>16} {pc:>10}")
        print(f"{'='*80}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"NOTE: Using fixed generate_report() — single-pass extraction")

    # ── Re-run 1: Adult multi-seed ──────────────────────────────────────
    print("\n" + "#" * 60)
    print("# RE-RUN 1: ADULT MULTI-SEED (fixed)")
    print("#" * 60)

    all_adult_rows = []
    for seed in SEEDS:
        all_adult_rows.extend(run_adult_seed(seed, device))

    save_rows(all_adult_rows, project_root / "results" / "adult" / "adult_seeds_fixed.csv")

    # Quick acceptance check
    adult_summary = compute_summary(all_adult_rows)
    std_row = next((r for r in adult_summary if r["method"] == "Standard"), {})
    pcrl_row = next((r for r in adult_summary if r["method"] == "PCRL"), {})
    print(f"\n  ACCEPTANCE CHECK (Adult):")
    print(f"    Standard Sex D: {std_row.get('delta_sex_mean', 0):+.1%} (expect [20%, 30%])")
    print(f"    Standard Marital D: {std_row.get('delta_marital_status_mean', 0):+.1%} (expect [40%, 50%])")
    print(f"    PCRL Sex D: {pcrl_row.get('delta_sex_mean', 0):+.1%}")
    print(f"    PCRL Pass: {pcrl_row.get('pass_count_mean', 0):.1f}/8")

    # ── Re-run 2: HAR multi-seed ────────────────────────────────────────
    print("\n" + "#" * 60)
    print("# RE-RUN 2: HAR MULTI-SEED (fixed)")
    print("#" * 60)

    all_har_rows = []
    for seed in SEEDS:
        all_har_rows.extend(run_har_seed(seed, device))

    save_rows(all_har_rows, project_root / "results" / "har_real" / "har_seeds_fixed.csv")

    har_summary = compute_summary(all_har_rows)
    std_har = next((r for r in har_summary if r["method"] == "Standard"), {})
    print(f"\n  ACCEPTANCE CHECK (HAR):")
    print(f"    Standard Subject D: {std_har.get('delta_subject_id_mean', 0):+.1%} (expect > +5%)")

    # ── Re-run 3: CelebA v2 ────────────────────────────────────────────
    print("\n" + "#" * 60)
    print("# RE-RUN 3: CelebA v2 (fixed)")
    print("#" * 60)

    celeba_rows = run_celeba_v2(device)
    out_path = project_root / "results" / "celeba" / "baseline_v2_fixed.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if celeba_rows:
        fieldnames = list(celeba_rows[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(celeba_rows)
        print(f"  Saved {out_path}")

    # ── Combined summary ────────────────────────────────────────────────
    all_rows = all_adult_rows + all_har_rows
    summary = compute_summary(all_rows)
    save_rows(summary, project_root / "results" / "seeds_summary_fixed.csv")
    print_summary_table(summary)

    print("\nAll re-runs complete!")


if __name__ == "__main__":
    main()
