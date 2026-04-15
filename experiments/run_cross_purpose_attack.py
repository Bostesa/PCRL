#!/usr/bin/env python3
"""Cross-purpose attack experiment on Adult dataset.

For each person, extracts representations from ALL purposes, concatenates
them into one vector, and trains post-hoc auditors to predict disallowed
attributes. Compares concatenated-representation auditor accuracy against
single-purpose auditor accuracy to test whether combining representations
across purposes leaks more information than any single purpose alone.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pcrl.data.adult import AdultDataset, get_adult_purposes
from pcrl.data.base import collate_pcrl_batch
from pcrl.models.auditor import MultiAttributeAuditor, PostHocAuditorSuite
from pcrl.models.encoder import PurposeConditionedEncoder
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
DROPOUT = 0.3
BATCH_SIZE = 256
EPOCHS = 200
PATIENCE = 15
LAMBDA_ADV = 50.0
LAMBDA_VERIFY = 50.0
AUDITOR_STEPS = 10


def extract_representations(
    encoder: torch.nn.Module,
    loader: DataLoader,
    purpose_idx: int,
    device: str,
) -> np.ndarray:
    """Extract representations for a single purpose."""
    encoder.eval()
    all_reps = []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder(x, purpose_idx)
            all_reps.append(h.cpu().numpy())
    return np.concatenate(all_reps, axis=0)


def extract_labels(loader: DataLoader, attr_name: str) -> np.ndarray:
    """Extract attribute labels."""
    all_labels = []
    for batch in loader:
        all_labels.append(batch["sensitive_attrs"][attr_name].numpy())
    return np.concatenate(all_labels, axis=0)


def run_auditors(
    train_reps: np.ndarray,
    train_labels: np.ndarray,
    test_reps: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, float]:
    """Train PostHocAuditorSuite and return per-classifier accuracy."""
    suite = PostHocAuditorSuite(random_state=42)
    suite.fit(train_reps, train_labels)
    results = suite.evaluate(test_reps, test_labels)
    return {name: metrics["accuracy"] for name, metrics in results.items()}


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Setup ────────────────────────────────────────────────────────────
    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    purpose_names = [p.name for p in purposes]

    # All sensitive attributes across all purposes
    all_attrs = sorted({a for p in purposes for a in p.disallowed_attrs})
    print(f"Purposes: {purpose_names}")
    print(f"Sensitive attributes: {all_attrs}")

    # ── Load data ────────────────────────────────────────────────────────
    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    val_ds = AdultDataset(
        purposes=purposes, root="data", split="val", download=False,
        norm_stats=train_ds.norm_stats,
    )
    test_ds = AdultDataset(
        purposes=purposes, root="data", split="test", download=False,
        norm_stats=train_ds.norm_stats,
    )

    input_dim = train_ds.info.num_features
    print(f"Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}, Features={input_dim}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pcrl_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)

    # Separate loaders with shuffle=False for consistent extraction
    train_extract_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pcrl_batch)

    # ── Build and train model ────────────────────────────────────────────
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        repr_dim=REPR_DIM,
        num_purposes=len(purposes),
        purpose_emb_dim=PURPOSE_EMB_DIM,
        conditioning="film",
        dropout=DROPOUT,
    )

    task_heads: dict[str, torch.nn.Module] = {}
    for p in purposes:
        task_name = p.allowed_tasks[0]
        output_dim = p.allowed_task_dims.get(task_name, 2)
        task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=output_dim)

    auditors: dict[str, torch.nn.Module] = {}
    for p in purposes:
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM,
            attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256,
            num_layers=3,
        )

    config = TrainerConfig(
        batch_size=BATCH_SIZE,
        lr_encoder=1e-3,
        lr_auditor=1e-3,
        lambda_adv=LAMBDA_ADV,
        lambda_verify=LAMBDA_VERIFY,
        auditor_steps=AUDITOR_STEPS,
        epochs=EPOCHS,
        weight_decay=1e-4,
        early_stopping_patience=PATIENCE,
        log_interval=100,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / "cross_purpose_attack"),
    )

    trainer = PCRLTrainer(
        encoder=encoder,
        task_heads=task_heads,
        auditors=auditors,
        config=config,
        purpose_registry=registry,
        device=device,
    )

    print("\n" + "=" * 60)
    print("TRAINING PCRL MODEL")
    print("=" * 60)
    state = trainer.train(train_loader, val_loader=val_loader)
    print(f"Training completed at epoch {state.epoch + 1}")

    # ── Extract representations ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXTRACTING REPRESENTATIONS")
    print("=" * 60)

    encoder.to(device)
    encoder.eval()

    # Per-purpose representations
    train_reps = {}
    test_reps = {}
    for idx, pname in enumerate(purpose_names):
        train_reps[pname] = extract_representations(encoder, train_extract_loader, idx, device)
        test_reps[pname] = extract_representations(encoder, test_loader, idx, device)
        print(f"  {pname}: train {train_reps[pname].shape}, test {test_reps[pname].shape}")

    # Concatenated representations
    train_concat = np.concatenate([train_reps[p] for p in purpose_names], axis=1)
    test_concat = np.concatenate([test_reps[p] for p in purpose_names], axis=1)
    print(f"  concatenated: train {train_concat.shape}, test {test_concat.shape}")

    # Extract labels
    train_labels = {}
    test_labels = {}
    for attr in all_attrs:
        train_labels[attr] = extract_labels(train_extract_loader, attr)
        test_labels[attr] = extract_labels(test_loader, attr)

    # Majority baselines
    majority = {}
    for attr in all_attrs:
        _, counts = np.unique(test_labels[attr], return_counts=True)
        majority[attr] = float(counts.max() / len(test_labels[attr]))

    # ── Run auditor attacks ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RUNNING CROSS-PURPOSE ATTACK")
    print("=" * 60)

    rows = []

    for attr in all_attrs:
        print(f"\n--- Attribute: {attr} (majority baseline: {majority[attr]:.1%}) ---")

        # Single-purpose attacks
        best_single_acc = 0.0
        best_single_purpose = ""
        best_single_clf = ""

        for pname in purpose_names:
            accs = run_auditors(
                train_reps[pname], train_labels[attr],
                test_reps[pname], test_labels[attr],
            )
            best_clf = max(accs, key=accs.get)
            best_acc = accs[best_clf]
            delta = best_acc - majority[attr]
            print(f"  {pname:30s} best={best_acc:.1%} ({best_clf}, delta={delta:+.1%})")

            for clf_name, acc in accs.items():
                rows.append({
                    "attribute": attr,
                    "representation": pname,
                    "classifier": clf_name,
                    "accuracy": round(acc, 4),
                    "majority_baseline": round(majority[attr], 4),
                    "delta": round(acc - majority[attr], 4),
                })

            if best_acc > best_single_acc:
                best_single_acc = best_acc
                best_single_purpose = pname
                best_single_clf = best_clf

        # Concatenated attack
        concat_accs = run_auditors(
            train_concat, train_labels[attr],
            test_concat, test_labels[attr],
        )
        concat_best_clf = max(concat_accs, key=concat_accs.get)
        concat_best_acc = concat_accs[concat_best_clf]
        concat_delta = concat_best_acc - majority[attr]
        print(f"  {'CONCATENATED':30s} best={concat_best_acc:.1%} ({concat_best_clf}, delta={concat_delta:+.1%})")

        for clf_name, acc in concat_accs.items():
            rows.append({
                "attribute": attr,
                "representation": "concatenated",
                "classifier": clf_name,
                "accuracy": round(acc, 4),
                "majority_baseline": round(majority[attr], 4),
                "delta": round(acc - majority[attr], 4),
            })

        # Summary for this attribute
        gain = concat_best_acc - best_single_acc
        print(f"  >> Best single: {best_single_acc:.1%} ({best_single_purpose}/{best_single_clf})")
        print(f"  >> Concatenated: {concat_best_acc:.1%} ({concat_best_clf})")
        print(f"  >> Concatenation gain: {gain:+.1%}")

    # ── Save results ─────────────────────────────────────────────────────
    out_dir = project_root / "results" / "adult"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_purpose_attack.csv"

    fieldnames = ["attribute", "representation", "classifier", "accuracy",
                  "majority_baseline", "delta"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {out_path}")

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CROSS-PURPOSE ATTACK SUMMARY")
    print("=" * 60)

    any_gain = False
    for attr in all_attrs:
        # Best single
        single_rows = [r for r in rows if r["attribute"] == attr and r["representation"] != "concatenated"]
        concat_rows = [r for r in rows if r["attribute"] == attr and r["representation"] == "concatenated"]
        best_single = max(r["accuracy"] for r in single_rows)
        best_concat = max(r["accuracy"] for r in concat_rows)
        gain = best_concat - best_single
        marker = " ** GAIN **" if gain > 0.005 else ""
        print(f"  {attr:20s}  single={best_single:.1%}  concat={best_concat:.1%}  gain={gain:+.1%}{marker}")
        if gain > 0.005:
            any_gain = True

    print()
    if any_gain:
        print("CONCLUSION: Concatenation DOES recover more information for some attributes.")
        print("This means purpose-specific representations are not fully independent —")
        print("an adversary with access to multiple purposes can extract more than")
        print("any single purpose reveals alone.")
    else:
        print("CONCLUSION: Concatenation does NOT meaningfully improve attack accuracy.")
        print("Purpose-conditioned representations successfully isolate information —")
        print("combining them does not help an adversary beyond what the best single")
        print("purpose already reveals.")


if __name__ == "__main__":
    main()
