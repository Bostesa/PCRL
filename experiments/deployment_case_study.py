#!/usr/bin/env python3
"""Deployment case study: multi-tenant hospital embedding sharing.

Scenario: A hospital shares patient embeddings with three parties:
  1. Insurance company: allowed=[risk_score], disallowed=[diagnosis, demographics]
  2. Research lab:      allowed=[diagnosis],   disallowed=[identity, demographics]
  3. Hospital admin:    allowed=[length_of_stay], disallowed=[diagnosis, identity]

Using the Adult dataset as a proxy:
  - income         -> risk_score     (Insurance task)
  - occupation_group -> diagnosis    (Research task)
  - education_level  -> length_of_stay (Admin task)
  - race, sex       -> demographics  (sensitive for Insurance & Research)
  - income          -> identity proxy (sensitive for Research & Admin)

Key conflict: income is ALLOWED for Insurance but DISALLOWED for Research
and Admin.  A single-representation method cannot simultaneously support
income prediction and hide income — the impossibility theorem applies.

This script:
  1. Defines the three hospital-proxy purposes.
  2. Trains PCRL.
  3. Shows each party gets a representation supporting ONLY their allowed task.
  4. Demonstrates composition: Insurance AND Research produces a
     representation hiding the union of both parties' constraints.
  5. Prints compliance table for all parties.
  6. Saves to results/deployment_case_study.csv.
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

from pcrl.data.adult import AdultDataset
from pcrl.data.base import collate_pcrl_batch
from pcrl.evaluation.certificates import (
    EmpiricalAudit,
    LinearAudit,
)
from pcrl.evaluation.mine import MINEstimator
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.composition import compose_and, compose_embeddings_additive
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec
from pcrl.purposes.verification import (
    find_conflicting_attributes,
    impossibility_bound,
    verify_impossibility,
)
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")


# ── Hospital-proxy purpose definitions ────────────────────────────────────

def get_hospital_purposes() -> list[PurposeSpec]:
    """Define three hospital-proxy purposes using Adult dataset features.

    Maps Adult features to hospital concepts:
      income          -> risk_score     (Insurance)
      occupation_group -> diagnosis     (Research)
      education_level  -> length_of_stay (Admin)
      race, sex       -> demographics
      income          -> identity proxy
    """
    return [
        PurposeSpec(
            name="insurance_risk",
            allowed_tasks=["income"],
            disallowed_attrs=["race", "sex"],
            task_type="classification",
            allowed_task_dims={"income": 2},
            disallowed_attr_dims={"race": 5, "sex": 2},
        ),
        PurposeSpec(
            name="research_lab",
            allowed_tasks=["occupation_group"],
            disallowed_attrs=["race", "age_group", "income"],
            task_type="classification",
            allowed_task_dims={"occupation_group": 6},
            disallowed_attr_dims={"race": 5, "age_group": 4, "income": 2},
        ),
        PurposeSpec(
            name="hospital_admin",
            allowed_tasks=["education_level"],
            disallowed_attrs=["sex", "race", "income"],
            task_type="classification",
            allowed_task_dims={"education_level": 4},
            disallowed_attr_dims={"sex": 2, "race": 5, "income": 2},
        ),
    ]


HOSPITAL_NAMES = {
    "insurance_risk": "Insurance Co.",
    "research_lab": "Research Lab",
    "hospital_admin": "Hospital Admin",
}

FEATURE_NAMES = {
    "income": "risk_score",
    "occupation_group": "diagnosis",
    "education_level": "length_of_stay",
    "race": "demographics(race)",
    "sex": "demographics(sex)",
    "age_group": "demographics(age)",
}


def extract_all(
    encoder: torch.nn.Module,
    loader: DataLoader,
    purpose_idx: int,
    device: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract representations, task labels, and sensitive attrs."""
    encoder.eval()
    all_reprs, all_tasks, all_sens = [], {}, {}
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder(x, purpose_idx)
            all_reprs.append(h.cpu().numpy())
            for k, v in batch["task_labels"].items():
                all_tasks.setdefault(k, []).append(v.numpy())
            for k, v in batch["sensitive_attrs"].items():
                all_sens.setdefault(k, []).append(v.numpy())
    return (
        np.concatenate(all_reprs),
        {k: np.concatenate(v) for k, v in all_tasks.items()},
        {k: np.concatenate(v) for k, v in all_sens.items()},
    )


def extract_with_embedding(
    encoder: torch.nn.Module,
    loader: DataLoader,
    purpose_emb: torch.Tensor,
    device: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract representations using a precomputed purpose embedding."""
    encoder.eval()
    all_reprs, all_tasks, all_sens = [], {}, {}
    emb = purpose_emb.unsqueeze(0)  # (1, emb_dim)
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            batch_emb = emb.expand(x.shape[0], -1)
            h = encoder.forward_with_embedding(x, batch_emb)
            all_reprs.append(h.cpu().numpy())
            for k, v in batch["task_labels"].items():
                all_tasks.setdefault(k, []).append(v.numpy())
            for k, v in batch["sensitive_attrs"].items():
                all_sens.setdefault(k, []).append(v.numpy())
    return (
        np.concatenate(all_reprs),
        {k: np.concatenate(v) for k, v in all_tasks.items()},
        {k: np.concatenate(v) for k, v in all_sens.items()},
    )


def run_compliance_audit(
    train_reprs: np.ndarray,
    test_reprs: np.ndarray,
    train_labels: dict[str, np.ndarray],
    test_labels: dict[str, np.ndarray],
    purpose: PurposeSpec,
    task_train_labels: dict[str, np.ndarray],
    task_test_labels: dict[str, np.ndarray],
) -> list[dict]:
    """Run compliance audit for a single purpose."""
    linear_audit = LinearAudit()
    empirical_audit = EmpiricalAudit()
    rows = []

    # Task accuracy (allowed task)
    from sklearn.linear_model import LogisticRegression
    for task_name in purpose.allowed_tasks:
        if task_name in task_train_labels and task_name in task_test_labels:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(train_reprs, task_train_labels[task_name])
            task_acc = clf.score(test_reprs, task_test_labels[task_name])
        else:
            task_acc = 0.0
        rows.append({
            "purpose": purpose.name,
            "party": HOSPITAL_NAMES.get(purpose.name, purpose.name),
            "attribute": FEATURE_NAMES.get(task_name, task_name),
            "role": "ALLOWED (task)",
            "linear_r2": float("nan"),
            "accuracy_bound": None,
            "empirical_acc": task_acc,
            "majority_prop": float("nan"),
            "status": f"{task_acc:.1%}",
        })

    # Disallowed attribute audit
    for attr_name in purpose.disallowed_attrs:
        if attr_name not in test_labels:
            continue
        attr_test = test_labels[attr_name]
        attr_train = train_labels[attr_name]

        linear_result, _ = linear_audit.audit(test_reprs, attr_test)

        all_labels = np.concatenate([attr_train, attr_test])
        unique, counts = np.unique(all_labels, return_counts=True)
        num_classes = len(unique)
        majority_prop = float(counts.max() / len(all_labels))

        best_acc, _ = empirical_audit.audit(
            train_reprs, attr_train, test_reprs, attr_test,
        )

        bound = None  # Retired invalid classification-accuracy guarantee.

        certified = (
            linear_result.certified
            and (best_acc - 1.0 / num_classes) < 0.05
        )
        status = "PASS" if certified else "FAIL"

        rows.append({
            "purpose": purpose.name,
            "party": HOSPITAL_NAMES.get(purpose.name, purpose.name),
            "attribute": FEATURE_NAMES.get(attr_name, attr_name),
            "role": "DISALLOWED",
            "linear_r2": linear_result.r_squared,
            "accuracy_bound": bound,
            "empirical_acc": best_acc,
            "majority_prop": majority_prop,
            "status": status,
        })

    return rows


def print_deployment_table(all_rows: list[dict], title: str) -> None:
    """Print a formatted deployment compliance table."""
    print(f"\n{'=' * 110}")
    print(title)
    print(f"{'=' * 110}")
    header = (
        f"{'Party':<18} {'Attribute':<22} {'Role':<16} "
        f"{'Lin R²':>8} {'Bound':>8} {'Emp Acc':>8} {'Status':>8}"
    )
    print(header)
    print("-" * 110)

    current_party = None
    for r in all_rows:
        party = r["party"]
        if party != current_party:
            if current_party is not None:
                print("-" * 110)
            current_party = party

        r2_str = f"{r['linear_r2']:.4f}" if not np.isnan(r['linear_r2']) else "—"
        bound_str = "retired"
        emp_str = f"{r['empirical_acc']:.1%}"

        print(
            f"{party:<18} {r['attribute']:<22} {r['role']:<16} "
            f"{r2_str:>8} {bound_str:>8} {emp_str:>8} {r['status']:>8}"
        )

    print(f"{'=' * 110}")


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "purpose", "party", "attribute", "role",
        "linear_r2", "accuracy_bound", "empirical_acc",
        "majority_prop", "status",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                k: (round(v, 6) if isinstance(v, float) and not np.isnan(v) else v)
                for k, v in r.items()
            })
    print(f"Saved {path}")


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # ── Setup ─────────────────────────────────────────────────────────────
    purposes = get_hospital_purposes()

    print("\nScenario: Hospital shares patient embeddings with three parties")
    print("=" * 70)
    for p in purposes:
        party = HOSPITAL_NAMES[p.name]
        tasks = ", ".join(FEATURE_NAMES.get(t, t) for t in p.allowed_tasks)
        hidden = ", ".join(FEATURE_NAMES.get(a, a) for a in p.disallowed_attrs)
        print(f"  {party}:")
        print(f"    Allowed:    {tasks}")
        print(f"    Disallowed: {hidden}")

    # Check for conflicts
    conflicts = find_conflicting_attributes(purposes)
    if conflicts:
        print(f"\nConflicting attributes ({len(conflicts)} conflicts):")
        for attr, need, forbid in conflicts:
            print(
                f"  {FEATURE_NAMES.get(attr, attr)}: "
                f"needed by {HOSPITAL_NAMES.get(need, need)}, "
                f"forbidden by {HOSPITAL_NAMES.get(forbid, forbid)}"
            )

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading Adult dataset...")
    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(purposes=purposes, root="data", split="val")
    test_ds = AdultDataset(purposes=purposes, root="data", split="test")
    print(f"  Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    bs = 256
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )

    # ── Train PCRL ────────────────────────────────────────────────────────
    print("\n--- Training PCRL ---")
    input_dim = train_ds.info.num_features
    repr_dim = 64

    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    torch.manual_seed(42)
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=[128, 128],
        repr_dim=repr_dim,
        num_purposes=len(purposes),
        purpose_emb_dim=32,
        conditioning="film",
        dropout=0.3,
    )

    task_heads = {}
    auditors = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=repr_dim, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )

    config = TrainerConfig(
        batch_size=bs, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=50.0, lambda_verify=50.0,
        auditor_steps=10, epochs=80,
        weight_decay=1e-4, early_stopping_patience=15,
        confusion_type="entropy",
    )

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    elapsed = time.time() - t0
    print(f"  Trained {state.epoch + 1} epochs in {elapsed:.0f}s")

    eval_metrics = trainer.evaluate(test_loader)
    for task, acc in eval_metrics.task_accuracy.items():
        print(f"    {task}: {acc:.1%}")

    # ── Per-party compliance audit ────────────────────────────────────────
    print("\n--- Per-party compliance audit ---")
    all_rows = []

    for purpose_idx, purpose in enumerate(purposes):
        train_reprs, train_tasks, train_sens = extract_all(
            encoder, train_loader, purpose_idx, device,
        )
        test_reprs, test_tasks, test_sens = extract_all(
            encoder, test_loader, purpose_idx, device,
        )

        rows = run_compliance_audit(
            train_reprs, test_reprs,
            train_sens, test_sens,
            purpose,
            train_tasks, test_tasks,
        )
        all_rows.extend(rows)

    print_deployment_table(all_rows, "Hospital Multi-Tenant Compliance Report")

    # ── Composition: Insurance AND Research ───────────────────────────────
    print("\n--- Composition: Insurance AND Research ---")
    p_ins = purposes[0]  # insurance_risk
    p_res = purposes[1]  # research_lab
    composed = compose_and(p_ins, p_res)
    composed_spec = composed.to_purpose_spec()

    print(f"  Composed purpose: {composed.name}")
    print(f"    Allowed tasks:    {composed.allowed_tasks}")
    print(f"    Disallowed attrs: {composed.disallowed_attrs}")

    # Compose embeddings
    emb_ins = encoder.get_purpose_embedding(0)
    emb_res = encoder.get_purpose_embedding(1)
    composed_emb = compose_embeddings_additive(emb_ins, emb_res)

    train_reprs_c, train_tasks_c, train_sens_c = extract_with_embedding(
        encoder, train_loader, composed_emb, device,
    )
    test_reprs_c, test_tasks_c, test_sens_c = extract_with_embedding(
        encoder, test_loader, composed_emb, device,
    )

    comp_rows = run_compliance_audit(
        train_reprs_c, test_reprs_c,
        train_sens_c, test_sens_c,
        composed_spec,
        train_tasks_c, test_tasks_c,
    )

    print_deployment_table(
        comp_rows,
        "Composed Purpose (Insurance AND Research) — Compliance Report",
    )

    # ── Impossibility analysis ────────────────────────────────────────────
    print("\n--- Impossibility analysis ---")
    if conflicts:
        # Extract representations for all purposes
        purpose_reprs = {}
        all_labels: dict[str, np.ndarray] = {}

        for idx, p in enumerate(purposes):
            reprs, tasks, sens = extract_all(encoder, test_loader, idx, device)
            purpose_reprs[p.name] = reprs
            # Collect all labels
            for k, v in sens.items():
                if k not in all_labels:
                    all_labels[k] = v
            for k, v in tasks.items():
                if k not in all_labels:
                    all_labels[k] = v

        impossibility_results = verify_impossibility(
            purpose_reprs, all_labels, purposes, device=device,
        )

        print(f"\n  Summary:")
        for r in impossibility_results:
            print(
                f"    {FEATURE_NAMES.get(r.attribute, r.attribute)}: "
                f"Single-rep must leak >= {r.single_rep_min_accuracy:.1%}, "
                f"PCRL leaks {r.pcrl_accuracy_bound:.1%}"
            )

    # ── Save results ──────────────────────────────────────────────────────
    combined_rows = all_rows + comp_rows
    save_csv(
        combined_rows,
        project_root / "results" / "deployment_case_study.csv",
    )

    print(f"\n{'=' * 70}")
    print("DONE — Deployment case study complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
