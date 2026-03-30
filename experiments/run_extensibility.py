#!/usr/bin/env python3
"""Extensibility experiment: add a new purpose without retraining.

Trains PCRL with 2 purposes (income_prediction, employment_analysis) on Adult.
Then freezes the encoder entirely and adds a 3rd purpose (education_assessment)
by training ONLY:
  - A new purpose embedding
  - A new task head
  - New auditors for the 3rd purpose

Reports whether the new purpose achieves reasonable task accuracy AND compliance.
Compares against training all 3 purposes from scratch.

Saves to results/adult/extensibility.csv.
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
import torch.nn as nn
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

from pcrl.data.adult import AdultDataset, get_adult_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.evaluation.certificates import (
    ComplianceReport,
    EmpiricalAudit,
    LinearAudit,
    _extract_representations_and_labels,
    generate_report,
)
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")

# Architecture matches run_adult.py
HIDDEN_DIMS = [128, 128]
REPR_DIM = 64
PURPOSE_EMB_DIM = 32
DROPOUT = 0.3
LR = 1e-3
BATCH_SIZE = 256
LAMBDA_ADV = 50.0
LAMBDA_VERIFY = 50.0
AUDITOR_STEPS = 10
EPOCHS = 200
PATIENCE = 15


@dataclass
class ExtensibilityResult:
    method: str
    purpose: str
    task_name: str
    task_accuracy: float
    pairs: list[dict] = field(default_factory=list)
    compliance_pass: int = 0
    compliance_total: int = 0
    train_time: float = 0.0


def compute_majority_baselines(dataset: AdultDataset) -> dict[str, float]:
    baselines = {}
    for attr_name, labels in dataset.sensitive_attrs.items():
        _, counts = labels.unique(return_counts=True)
        baselines[attr_name] = counts.max().item() / len(labels)
    return baselines


def train_base_model(
    purposes: list[PurposeSpec],
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    device: str,
) -> tuple[PurposeConditionedEncoder, PCRLTrainer, float]:
    """Train PCRL with 2 purposes."""
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    torch.manual_seed(42)
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        repr_dim=REPR_DIM,
        num_purposes=len(purposes),
        purpose_emb_dim=PURPOSE_EMB_DIM,
        conditioning="film",
        dropout=DROPOUT,
    )

    task_heads = {}
    auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM,
            attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    config = TrainerConfig(
        batch_size=BATCH_SIZE,
        lr_encoder=LR, lr_auditor=LR,
        lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY,
        auditor_steps=AUDITOR_STEPS,
        epochs=EPOCHS,
        weight_decay=1e-4,
        early_stopping_patience=PATIENCE,
        confusion_type="entropy",
    )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0

    print(f"  Base model: {state.epoch + 1} epochs, {train_time:.0f}s")
    return encoder, trainer, train_time


def extend_with_new_purpose(
    encoder: PurposeConditionedEncoder,
    new_purpose: PurposeSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
) -> tuple[nn.Embedding, TaskHead, MultiAttributeAuditor, float]:
    """Train a new purpose embedding + task head + auditor with frozen encoder.

    We add a new embedding slot, freeze all encoder parameters except the
    new embedding, and train only the new components.
    """
    # Create new purpose embedding (single slot)
    new_emb = nn.Embedding(1, PURPOSE_EMB_DIM).to(device)

    # Create new task head
    task_name = new_purpose.allowed_tasks[0]
    output_dim = new_purpose.allowed_task_dims.get(task_name, 2)
    new_task_head = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim).to(device)

    # Create new auditor
    new_auditor = MultiAttributeAuditor(
        repr_dim=REPR_DIM,
        attr_output_dims=new_purpose.disallowed_attr_dims,
        hidden_dim=256, num_layers=3,
    ).to(device)

    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    # Optimizers — only new components
    encoder_params = list(new_emb.parameters()) + list(new_task_head.parameters())
    encoder_opt = torch.optim.AdamW(encoder_params, lr=LR, weight_decay=1e-4)
    auditor_opt = torch.optim.AdamW(new_auditor.parameters(), lr=LR, weight_decay=1e-4)

    task_ce = nn.CrossEntropyLoss()
    confusion_fn = _entropy_confusion

    t0 = time.time()
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        # --- Training ---
        new_task_head.train()
        new_auditor.train()

        for batch in train_loader:
            x = batch["features"].to(device)
            task_labels = batch["task_labels"][task_name].to(device)
            sensitive = {k: v.to(device) for k, v in batch["sensitive_attrs"].items()
                         if k in new_purpose.disallowed_attrs}

            # Get embedding and forward through frozen encoder
            emb = new_emb(torch.zeros(1, dtype=torch.long, device=device)).squeeze(0)
            with torch.no_grad():
                h = encoder.forward_with_embedding(x, emb)
            h_detached = h.detach()

            # Auditor steps
            for _ in range(AUDITOR_STEPS):
                auditor_opt.zero_grad()
                aud_logits = new_auditor(h_detached)
                aud_loss = sum(
                    task_ce(aud_logits[attr], sensitive[attr])
                    for attr in aud_logits if attr in sensitive
                )
                aud_loss.backward()
                auditor_opt.step()

            # Encoder step (only new_emb + task_head update)
            encoder_opt.zero_grad()
            # Re-forward with gradient through embedding
            emb = new_emb(torch.zeros(1, dtype=torch.long, device=device)).squeeze(0)
            h = encoder.forward_with_embedding(x, emb)

            # Task loss
            logits = new_task_head(h)
            t_loss = task_ce(logits, task_labels)

            # Confusion loss (maximize auditor entropy)
            aud_logits = new_auditor(h)
            c_loss = sum(
                confusion_fn(aud_logits[attr])
                for attr in aud_logits if attr in sensitive
            )

            total_loss = t_loss + LAMBDA_ADV * c_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
            encoder_opt.step()

        # --- Validation ---
        new_task_head.eval()
        new_auditor.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            emb = new_emb(torch.zeros(1, dtype=torch.long, device=device)).squeeze(0)
            for batch in val_loader:
                x = batch["features"].to(device)
                labels = batch["task_labels"][task_name].to(device)
                h = encoder.forward_with_embedding(x, emb)
                logits = new_task_head(h)
                val_loss += task_ce(logits, labels).item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.shape[0]

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if PATIENCE is not None and patience_counter >= PATIENCE:
            print(f"    Extension: early stop at epoch {epoch + 1}, val_acc={val_acc:.1%}")
            break

    train_time = time.time() - t0

    # Unfreeze encoder for downstream use
    for param in encoder.parameters():
        param.requires_grad = True

    return new_emb, new_task_head, new_auditor, train_time


def _entropy_confusion(logits: torch.Tensor) -> torch.Tensor:
    """Entropy confusion loss: maximize entropy of auditor predictions."""
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    return -entropy.mean()  # Negative because we maximize entropy


def evaluate_extended_purpose(
    encoder: PurposeConditionedEncoder,
    new_emb: nn.Embedding,
    new_task_head: TaskHead,
    purpose: PurposeSpec,
    train_loader: DataLoader,
    test_loader: DataLoader,
    majority_baselines: dict[str, float],
    device: str,
) -> ExtensibilityResult:
    """Evaluate the extended purpose."""
    encoder.eval()
    new_task_head.eval()
    task_name = purpose.allowed_tasks[0]

    # Task accuracy
    correct = total = 0
    with torch.no_grad():
        emb = new_emb(torch.zeros(1, dtype=torch.long, device=device)).squeeze(0)
        for batch in test_loader:
            x = batch["features"].to(device)
            labels = batch["task_labels"][task_name].to(device)
            h = encoder.forward_with_embedding(x, emb)
            preds = new_task_head(h).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
    task_acc = correct / total if total > 0 else 0.0

    # Compliance audit for each disallowed attr
    linear_audit = LinearAudit()
    empirical_audit = EmpiricalAudit()
    pairs = []
    pass_count = 0

    for attr_name in purpose.disallowed_attrs:
        # Extract representations
        train_reprs, train_labels = [], []
        test_reprs, test_labels = [], []

        with torch.no_grad():
            emb = new_emb(torch.zeros(1, dtype=torch.long, device=device)).squeeze(0)
            for batch in train_loader:
                x = batch["features"].to(device)
                h = encoder.forward_with_embedding(x, emb)
                train_reprs.append(h.cpu().numpy())
                train_labels.append(batch["sensitive_attrs"][attr_name].numpy())
            for batch in test_loader:
                x = batch["features"].to(device)
                h = encoder.forward_with_embedding(x, emb)
                test_reprs.append(h.cpu().numpy())
                test_labels.append(batch["sensitive_attrs"][attr_name].numpy())

        train_r = np.concatenate(train_reprs)
        train_l = np.concatenate(train_labels)
        test_r = np.concatenate(test_reprs)
        test_l = np.concatenate(test_labels)

        linear_result, _ = linear_audit.audit(test_r, test_l)
        best_acc, _ = empirical_audit.audit(train_r, train_l, test_r, test_l)
        baseline = majority_baselines.get(attr_name, 0.5)
        delta = best_acc - baseline
        adj_pass = delta < 0.02 and linear_result.r_squared < 0.05
        if adj_pass:
            pass_count += 1

        pairs.append({
            "attribute": attr_name,
            "emp_acc": best_acc,
            "majority": baseline,
            "delta": delta,
            "r2": linear_result.r_squared,
            "pass": adj_pass,
        })

    return ExtensibilityResult(
        method="extended (frozen encoder)",
        purpose=purpose.name,
        task_name=task_name,
        task_accuracy=task_acc,
        pairs=pairs,
        compliance_pass=pass_count,
        compliance_total=len(pairs),
    )


def train_from_scratch(
    purposes: list[PurposeSpec],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    test_dataset: AdultDataset,
    input_dim: int,
    device: str,
) -> list[ExtensibilityResult]:
    """Train all 3 purposes from scratch."""
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    torch.manual_seed(42)
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        repr_dim=REPR_DIM,
        num_purposes=len(purposes),
        purpose_emb_dim=PURPOSE_EMB_DIM,
        conditioning="film",
        dropout=DROPOUT,
    )

    task_heads = {}
    auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM,
            attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    config = TrainerConfig(
        batch_size=BATCH_SIZE,
        lr_encoder=LR, lr_auditor=LR,
        lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY,
        auditor_steps=AUDITOR_STEPS,
        epochs=EPOCHS,
        weight_decay=1e-4,
        early_stopping_patience=PATIENCE,
        confusion_type="entropy",
    )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0

    eval_metrics = trainer.evaluate(test_loader)
    reports = generate_report(
        encoder=encoder, train_loader=train_loader,
        test_loader=test_loader, purpose_registry=registry, device=device,
    )

    majority_baselines = compute_majority_baselines(test_dataset)

    results = []
    for purpose in purposes:
        task_name = purpose.allowed_tasks[0]
        task_acc = eval_metrics.task_accuracy.get(task_name, 0.0)
        pairs = []
        pass_count = 0
        for r in reports:
            if r.purpose_name != purpose.name:
                continue
            baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            delta = r.empirical_best_acc - baseline
            adj_pass = delta < 0.02 and r.linear_r2 < 0.05
            if adj_pass:
                pass_count += 1
            pairs.append({
                "attribute": r.attr_name,
                "emp_acc": r.empirical_best_acc,
                "majority": baseline,
                "delta": delta,
                "r2": r.linear_r2,
                "pass": adj_pass,
            })

        results.append(ExtensibilityResult(
            method="from scratch (3 purposes)",
            purpose=purpose.name,
            task_name=task_name,
            task_accuracy=task_acc,
            pairs=pairs,
            compliance_pass=pass_count,
            compliance_total=len(pairs),
            train_time=train_time,
        ))

    print(f"  From scratch: {state.epoch + 1} epochs, {train_time:.0f}s")
    return results


def save_csv(results: list[ExtensibilityResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        for p in r.pairs:
            rows.append({
                "method": r.method,
                "purpose": r.purpose,
                "task_name": r.task_name,
                "task_accuracy": round(r.task_accuracy, 4),
                "attribute": p["attribute"],
                "emp_acc": round(p["emp_acc"], 4),
                "majority": round(p["majority"], 4),
                "delta": round(p["delta"], 4),
                "r2": round(p["r2"], 6),
                "adj_pass": p["pass"],
                "compliance": f"{r.compliance_pass}/{r.compliance_total}",
                "train_time_s": round(r.train_time, 1),
            })

    if not rows:
        print(f"No rows to save to {path}")
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    all_purposes = get_adult_purposes()
    base_purposes = all_purposes[:2]  # income_prediction, employment_analysis
    new_purpose = all_purposes[2]     # education_assessment

    print(f"Base purposes: {[p.name for p in base_purposes]}")
    print(f"New purpose:   {new_purpose.name}")

    # Load data with ALL purposes so education_level labels are available
    train_ds = AdultDataset(purposes=all_purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=all_purposes, root="data", split="val", download=False)
    test_ds = AdultDataset(purposes=all_purposes, root="data", split="test", download=False)

    input_dim = train_ds.info.num_features
    print(f"Adult: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}, "
          f"Features={input_dim}")

    majority_baselines = compute_majority_baselines(test_ds)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # A. Train base model with 2 purposes
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("A. Training base model (2 purposes)")
    print("=" * 60)
    encoder, base_trainer, base_time = train_base_model(
        base_purposes, train_loader, val_loader, input_dim, device,
    )

    # Evaluate base purposes
    base_eval = base_trainer.evaluate(test_loader)
    print("  Base model task accuracies:")
    for task, acc in base_eval.task_accuracy.items():
        print(f"    {task}: {acc:.1%}")

    # ═══════════════════════════════════════════════════════════════════════
    # B. Extend with 3rd purpose (frozen encoder)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("B. Extending with education_assessment (frozen encoder)")
    print("=" * 60)
    new_emb, new_task_head, new_auditor, ext_time = extend_with_new_purpose(
        encoder, new_purpose, train_loader, val_loader, device,
    )
    print(f"    Extension training: {ext_time:.0f}s")

    ext_result = evaluate_extended_purpose(
        encoder, new_emb, new_task_head, new_purpose,
        train_loader, test_loader, majority_baselines, device,
    )
    ext_result.train_time = ext_time

    # Also evaluate base purposes with original indices
    base_results = []
    base_reports = generate_report(
        encoder=encoder, train_loader=train_loader,
        test_loader=test_loader,
        purpose_registry=PurposeRegistry(),
        device=device,
    )
    # Re-register just base purposes for report
    base_reg = PurposeRegistry()
    for p in base_purposes:
        base_reg.register(p)
    base_reports = generate_report(
        encoder=encoder, train_loader=train_loader,
        test_loader=test_loader, purpose_registry=base_reg, device=device,
    )
    for purpose in base_purposes:
        task_name = purpose.allowed_tasks[0]
        task_acc = base_eval.task_accuracy.get(task_name, 0.0)
        pairs = []
        pc = 0
        for r in base_reports:
            if r.purpose_name != purpose.name:
                continue
            baseline = majority_baselines.get(r.attr_name, r.empirical_chance_acc)
            delta = r.empirical_best_acc - baseline
            adj_pass = delta < 0.02 and r.linear_r2 < 0.05
            if adj_pass:
                pc += 1
            pairs.append({
                "attribute": r.attr_name,
                "emp_acc": r.empirical_best_acc,
                "majority": baseline,
                "delta": delta,
                "r2": r.linear_r2,
                "pass": adj_pass,
            })
        base_results.append(ExtensibilityResult(
            method="extended (frozen encoder)",
            purpose=purpose.name,
            task_name=task_name,
            task_accuracy=task_acc,
            pairs=pairs,
            compliance_pass=pc,
            compliance_total=len(pairs),
            train_time=base_time,
        ))

    extended_results = base_results + [ext_result]

    # ═══════════════════════════════════════════════════════════════════════
    # C. Train all 3 from scratch
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("C. Training all 3 purposes from scratch")
    print("=" * 60)
    scratch_results = train_from_scratch(
        all_purposes, train_loader, val_loader, test_loader, test_ds, input_dim, device,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # D. Comparison
    # ═══════════════════════════════════════════════════════════════════════
    all_results = extended_results + scratch_results

    print("\n" + "=" * 100)
    print("EXTENSIBILITY RESULTS")
    print("=" * 100)
    print(f"{'Method':<30} {'Purpose':<22} {'Task':>10} {'Pass':>8}")
    print("-" * 100)
    for r in all_results:
        print(f"{r.method:<30} {r.purpose:<22} {r.task_accuracy:>9.1%} "
              f"{r.compliance_pass:>4}/{r.compliance_total}")

    # Detail
    print(f"\n{'Method':<30} {'Purpose':<22} {'Attribute':<16} "
          f"{'Emp':>8} {'Maj':>8} {'Delta':>8} {'R²':>8} {'Pass':>6}")
    print("-" * 100)
    for r in all_results:
        for p in r.pairs:
            status = "PASS" if p["pass"] else "FAIL"
            print(f"{r.method:<30} {r.purpose:<22} {p['attribute']:<16} "
                  f"{p['emp_acc']:>7.1%} {p['majority']:>7.1%} "
                  f"{p['delta']:>+7.1%} {p['r2']:>8.4f} {status:>6}")
    print("=" * 100)

    # Key comparison: education_assessment
    ext_edu = ext_result
    scratch_edu = next(
        (r for r in scratch_results if r.purpose == "education_assessment"), None
    )

    print(f"\n{'=' * 60}")
    print("KEY COMPARISON: education_assessment")
    print(f"{'=' * 60}")
    print(f"  Extended (frozen encoder):  acc={ext_edu.task_accuracy:.1%}, "
          f"compliance={ext_edu.compliance_pass}/{ext_edu.compliance_total}, "
          f"time={ext_edu.train_time:.0f}s")
    if scratch_edu:
        print(f"  From scratch (3 purposes): acc={scratch_edu.task_accuracy:.1%}, "
              f"compliance={scratch_edu.compliance_pass}/{scratch_edu.compliance_total}, "
              f"time={scratch_edu.train_time:.0f}s")
    print(f"{'=' * 60}")

    results_dir = project_root / "results" / "adult"
    save_csv(all_results, results_dir / "extensibility.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
