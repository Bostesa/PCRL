#!/usr/bin/env python3
"""MINE (Mutual Information Neural Estimation) audit on Adult and HAR.

For each (purpose, disallowed_attr) pair, estimates I(h_p; z_p) using
MINE (Belghazi et al. 2018). This gives a single scalar measuring total
information leakage — not just linear, not just what specific auditors
find, but an estimate of ALL recoverable information.

Low MI + low R² + low empirical accuracy = very strong evidence of compliance.

Results saved to results/adult/mutual_information.csv and
results/har_real/mutual_information.csv.
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

from pcrl.data.adult import AdultDataset, get_adult_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.data.har import HARDataset, get_har_purposes
from pcrl.evaluation.certificates import (
    ComplianceReport,
    EmpiricalAudit,
    LinearAudit,
)
from pcrl.evaluation.mine import MINEstimator
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.purposes.verification import certified_accuracy_bound
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")


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


def run_dataset(
    name: str,
    purposes: list,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    encoder_config: dict,
    trainer_config: TrainerConfig,
    device: str,
) -> list[dict]:
    """Train PCRL and run MINE + linear + empirical audit."""

    print(f"\n{'=' * 70}")
    print(f"{name} — MINE Audit")
    print(f"{'=' * 70}")

    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    repr_dim = encoder_config["repr_dim"]

    torch.manual_seed(42)
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=encoder_config["hidden_dims"],
        repr_dim=repr_dim,
        num_purposes=len(purposes),
        purpose_emb_dim=encoder_config["purpose_emb_dim"],
        conditioning="film",
        dropout=encoder_config["dropout"],
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

    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=trainer_config, purpose_registry=registry, device=device,
    )

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0
    print(f"  Trained {state.epoch + 1} epochs in {train_time:.0f}s")

    eval_metrics = trainer.evaluate(test_loader)
    for task, acc in eval_metrics.task_accuracy.items():
        print(f"    {task}: {acc:.1%}")

    # Run audits
    linear_audit = LinearAudit()
    empirical_audit = EmpiricalAudit()
    mine_estimator = MINEstimator(
        hidden_dim=256, num_steps=500, batch_size=512, lr=1e-3,
    )

    rows = []
    print(f"\n  Running MINE + compliance audits...")

    for purpose_idx, purpose in enumerate(purposes):
        train_reprs, train_tasks, train_sens = extract_all(
            encoder, train_loader, purpose_idx, device,
        )
        test_reprs, test_tasks, test_sens = extract_all(
            encoder, test_loader, purpose_idx, device,
        )

        for attr_name in purpose.disallowed_attrs:
            train_labels = train_sens[attr_name]
            test_labels = test_sens[attr_name]

            # Linear certificate
            linear_result, _ = linear_audit.audit(test_reprs, test_labels)

            # Empirical audit
            all_labels = np.concatenate([train_labels, test_labels])
            unique, counts = np.unique(all_labels, return_counts=True)
            num_classes = len(unique)
            majority_prop = float(counts.max() / len(all_labels))

            best_acc, emp_results = empirical_audit.audit(
                train_reprs, train_labels, test_reprs, test_labels,
            )

            # MINE estimate (on test set)
            mine_result = mine_estimator.estimate(
                test_reprs, test_labels, device=device,
            )

            # Theoretical maximum MI for this attribute
            # H(Z) = -sum(p_i * log(p_i))
            test_unique, test_counts = np.unique(test_labels, return_counts=True)
            probs = test_counts / len(test_labels)
            entropy_z = -float(np.sum(probs * np.log(probs + 1e-12)))
            entropy_z_bits = entropy_z / np.log(2)

            lin_bound = certified_accuracy_bound(
                linear_result.r_squared, majority_prop, num_classes,
            )

            certified = (
                linear_result.certified
                and (best_acc - 1.0 / num_classes) < 0.05
            )

            row = {
                "purpose": purpose.name,
                "attribute": attr_name,
                "linear_r2": linear_result.r_squared,
                "linear_bound": lin_bound,
                "mi_nats": mine_result.mi_estimate,
                "mi_bits": mine_result.mi_bits,
                "entropy_z_bits": entropy_z_bits,
                "mi_fraction": mine_result.mi_bits / max(entropy_z_bits, 1e-12),
                "empirical_best_acc": best_acc,
                "majority_proportion": majority_prop,
                "num_classes": num_classes,
                "certified": certified,
            }
            rows.append(row)

            status = "PASS" if certified else "FAIL"
            print(
                f"    {purpose.name}/{attr_name}: "
                f"R²={linear_result.r_squared:.4f}, "
                f"MI={mine_result.mi_bits:.4f} bits "
                f"(of {entropy_z_bits:.2f} max), "
                f"EmpAcc={best_acc:.1%}, {status}"
            )

    return rows


def print_mine_table(rows: list[dict], title: str) -> None:
    """Print results table with MINE estimates."""
    print(f"\n{'=' * 120}")
    print(title)
    print(f"{'=' * 120}")
    header = (
        f"{'Purpose':<25} {'Attribute':<16} {'Lin R²':>8} "
        f"{'MI (bits)':>10} {'H(Z) bits':>10} {'MI/H(Z)':>8} "
        f"{'Emp Acc':>8} {'Bound':>8} {'Status':>8}"
    )
    print(header)
    print("-" * 120)

    for r in rows:
        status = "PASS" if r["certified"] else "FAIL"
        print(
            f"{r['purpose']:<25} {r['attribute']:<16} "
            f"{r['linear_r2']:>8.4f} "
            f"{r['mi_bits']:>10.4f} {r['entropy_z_bits']:>10.2f} "
            f"{r['mi_fraction']:>7.1%} "
            f"{r['empirical_best_acc']:>7.1%} "
            f"{r['linear_bound']:>7.1%} "
            f"{status:>8}"
        )

    print("-" * 120)
    certified_count = sum(1 for r in rows if r["certified"])
    avg_mi = sum(r["mi_bits"] for r in rows) / len(rows) if rows else 0
    avg_frac = sum(r["mi_fraction"] for r in rows) / len(rows) if rows else 0
    print(f"Certified: {certified_count}/{len(rows)} | "
          f"Avg MI: {avg_mi:.4f} bits | "
          f"Avg MI/H(Z): {avg_frac:.1%}")
    print(f"{'=' * 120}")


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "purpose", "attribute", "linear_r2", "linear_bound",
            "mi_nats", "mi_bits", "entropy_z_bits", "mi_fraction",
            "empirical_best_acc", "majority_proportion",
            "num_classes", "certified",
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                             for k, v in r.items()})
    print(f"Saved {path}")


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # ── Adult ─────────────────────────────────────────────────────────────
    adult_purposes = get_adult_purposes()
    adult_train = AdultDataset(purposes=adult_purposes, root="data", split="train", download=True)
    adult_val = AdultDataset(purposes=adult_purposes, root="data", split="val")
    adult_test = AdultDataset(purposes=adult_purposes, root="data", split="test")

    print(f"Adult: Train={len(adult_train)}, Val={len(adult_val)}, "
          f"Test={len(adult_test)}, Features={adult_train.info.num_features}")

    bs = 256
    adult_train_loader = DataLoader(
        adult_train, batch_size=bs, shuffle=True,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    adult_val_loader = DataLoader(
        adult_val, batch_size=bs, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    adult_test_loader = DataLoader(
        adult_test, batch_size=bs, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )

    adult_rows = run_dataset(
        name="ADULT",
        purposes=adult_purposes,
        train_loader=adult_train_loader,
        val_loader=adult_val_loader,
        test_loader=adult_test_loader,
        input_dim=adult_train.info.num_features,
        encoder_config={
            "hidden_dims": [128, 128], "repr_dim": 64,
            "purpose_emb_dim": 32, "dropout": 0.3,
        },
        trainer_config=TrainerConfig(
            batch_size=bs, lr_encoder=1e-3, lr_auditor=1e-3,
            lambda_adv=50.0, lambda_verify=50.0,
            auditor_steps=10, epochs=80,
            weight_decay=1e-4, early_stopping_patience=15,
            confusion_type="entropy",
        ),
        device=device,
    )

    print_mine_table(adult_rows, "ADULT — MINE Mutual Information Audit")
    save_csv(adult_rows, project_root / "results" / "adult" / "mutual_information.csv")

    # ── HAR ───────────────────────────────────────────────────────────────
    har_purposes = get_har_purposes()
    har_train = HARDataset(purposes=har_purposes, root="data", split="train")
    har_val = HARDataset(purposes=har_purposes, root="data", split="val")
    har_test = HARDataset(purposes=har_purposes, root="data", split="test")

    har_input_dim = har_train[0]["features"].shape[0]
    print(f"\nHAR: Train={len(har_train)}, Val={len(har_val)}, "
          f"Test={len(har_test)}, Features={har_input_dim}")

    har_train_loader = DataLoader(
        har_train, batch_size=bs, shuffle=True,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    har_val_loader = DataLoader(
        har_val, batch_size=bs, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )
    har_test_loader = DataLoader(
        har_test, batch_size=bs, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=0, pin_memory=False,
    )

    har_rows = run_dataset(
        name="HAR",
        purposes=har_purposes,
        train_loader=har_train_loader,
        val_loader=har_val_loader,
        test_loader=har_test_loader,
        input_dim=har_input_dim,
        encoder_config={
            "hidden_dims": [128, 128], "repr_dim": 16,
            "purpose_emb_dim": 32, "dropout": 0.3,
        },
        trainer_config=TrainerConfig(
            batch_size=bs, lr_encoder=1e-3, lr_auditor=1e-3,
            lambda_adv=2.0, lambda_verify=1.0,
            auditor_steps=20, epochs=100,
            weight_decay=1e-4, early_stopping_patience=None,
            confusion_type="entropy",
        ),
        device=device,
    )

    print_mine_table(har_rows, "HAR — MINE Mutual Information Audit")
    save_csv(har_rows, project_root / "results" / "har_real" / "mutual_information.csv")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("DONE — MINE audit complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
