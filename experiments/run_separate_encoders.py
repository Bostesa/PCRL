#!/usr/bin/env python3
"""Separate-encoders threat experiment.

Trains |P| independent encoders (one per purpose, no FiLM, no purpose
embedding) for a given dataset, then evaluates each encoder's compliance on
its own purpose's disallowed attributes via the paper's adjusted criterion
(linear R^2 < 0.05 AND post-hoc auditor delta < 2pp).

Compares against published PCRL paper numbers to determine whether multi-
purpose conditioning is load-bearing or whether |P| separate encoders match
or beat the single conditioned encoder.

Per-purpose training uses the existing PCRLTrainer with a single-purpose
PurposeRegistry (and a vanilla StandardEncoder that ignores purpose_idx).
Hyperparameters mirror PCRL's seed-fixed runs:

    hidden_dims=[128, 128], repr_dim=64, dropout=0.3,
    lambda_adv=50, lambda_verify=50, epochs=200, patience=20,
    auditor_steps=10 (Adult)  / 20 (Diabetes/HMDA),
    auditor MLP: 3-layer, hidden=256, no spectral norm
    early stopping on val_metrics.task_loss (NOT composite)

Outputs:
    results/<dataset>_SEPARATE/per_purpose_results.json
    results/<dataset>_SEPARATE/summary.json
    checkpoints/separate_<dataset>_<purpose>_s<seed>/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Suppress tqdm noise from trainer (matches run_seeds_fixed.py).
import pcrl.training.trainer as _trainer_mod  # noqa: E402


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

from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.evaluation.certificates import generate_report  # noqa: E402
from pcrl.models.auditor import MultiAttributeAuditor  # noqa: E402
from pcrl.models.encoder import StandardEncoder  # noqa: E402
from pcrl.models.task_head import TaskHead  # noqa: E402
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec  # noqa: E402
from pcrl.training.trainer import PCRLTrainer, TrainerConfig  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("separate_encoders")
log.setLevel(logging.INFO)

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")


# ───────────────────────────────────────────────────────────────────────────
# Health stats
# ───────────────────────────────────────────────────────────────────────────


def effective_rank(reprs: np.ndarray) -> float:
    """Entropy-based effective rank of centered representation matrix."""
    if reprs.shape[0] < 2:
        return float("nan")
    centered = reprs - reprs.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    p = (s ** 2) / max((s ** 2).sum(), 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def repr_health(reprs: np.ndarray) -> dict:
    per_dim_std = reprs.std(axis=0)
    l2 = np.linalg.norm(reprs, axis=1)
    return {
        "shape": list(reprs.shape),
        "per_dim_std_mean": float(per_dim_std.mean()),
        "per_dim_std_max": float(per_dim_std.max()),
        "per_dim_std_min": float(per_dim_std.min()),
        "l2_norm_mean": float(l2.mean()),
        "l2_norm_std": float(l2.std()),
        "effective_rank": effective_rank(reprs),
    }


@torch.no_grad()
def extract_reprs(encoder: torch.nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    encoder.eval()
    out = []
    for batch in loader:
        h = encoder(batch["features"].to(device))
        out.append(h.cpu().numpy())
    return np.concatenate(out)


# ───────────────────────────────────────────────────────────────────────────
# Dataset + purposes
# ───────────────────────────────────────────────────────────────────────────


@dataclass
class DatasetBundle:
    name: str
    train_ds: torch.utils.data.Dataset
    val_ds: torch.utils.data.Dataset
    test_ds: torch.utils.data.Dataset
    purposes: list[PurposeSpec]
    input_dim: int
    auditor_steps: int  # K


def build_dataset(name: str) -> DatasetBundle:
    if name == "adult":
        from pcrl.data.adult import AdultDataset, get_adult_purposes
        purposes = get_adult_purposes()
        train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
        val_ds = AdultDataset(purposes=purposes, root="data", split="val", download=False, norm_stats=train_ds.norm_stats)
        test_ds = AdultDataset(purposes=purposes, root="data", split="test", download=False, norm_stats=train_ds.norm_stats)
        K = 10
    elif name == "diabetes":
        from pcrl.data.diabetes import DiabetesDataset, get_diabetes_purposes
        purposes = get_diabetes_purposes()
        train_ds = DiabetesDataset(purposes=purposes, root="data", split="train")
        val_ds = DiabetesDataset(purposes=purposes, root="data", split="val")
        test_ds = DiabetesDataset(purposes=purposes, root="data", split="test")
        K = 20
    elif name == "hmda":
        from pcrl.data.hmda import HMDADataset, get_hmda_purposes
        purposes = get_hmda_purposes()
        train_ds = HMDADataset(purposes=purposes, root="data", split="train")
        val_ds = HMDADataset(purposes=purposes, root="data", split="val")
        test_ds = HMDADataset(purposes=purposes, root="data", split="test")
        K = 20
    else:
        raise ValueError(f"unknown dataset: {name}")

    return DatasetBundle(
        name=name,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        purposes=purposes,
        input_dim=train_ds.info.num_features,
        auditor_steps=K,
    )


# ───────────────────────────────────────────────────────────────────────────
# Train + evaluate single purpose
# ───────────────────────────────────────────────────────────────────────────


def train_purpose(
    bundle: DatasetBundle,
    purpose: PurposeSpec,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
) -> dict:
    """Train a separate encoder for a single purpose and evaluate compliance.

    Returns a dict with task accuracy, per-attribute compliance, train time,
    representation health stats, and pass count.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Single-purpose registry
    registry = PurposeRegistry()
    registry.register(purpose)

    train_loader = DataLoader(bundle.train_ds, batch_size=256, shuffle=True,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(bundle.val_ds, batch_size=256, shuffle=False,
                            collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(bundle.test_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)

    # Vanilla MLP encoder, no FiLM, no purpose embedding.
    encoder = StandardEncoder(
        input_dim=bundle.input_dim,
        hidden_dims=[128, 128],
        repr_dim=64,
        dropout=0.3,
    )

    task_name = purpose.allowed_tasks[0]
    output_dim = purpose.allowed_task_dims.get(task_name, 2)
    task_head = TaskHead(repr_dim=64, output_dim=output_dim)

    # 3-layer 256 MLP auditor, NO spectral norm (matches PCRL's standard).
    auditor = MultiAttributeAuditor(
        repr_dim=64,
        attr_output_dims=purpose.disallowed_attr_dims,
        hidden_dim=256,
        num_layers=3,
    )

    ckpt_dir = ROOT / "checkpoints" / f"separate_{bundle.name}_{purpose.name}_s{seed}"
    config = TrainerConfig(
        batch_size=256,
        lr_encoder=1e-3,
        lr_auditor=1e-3,
        lambda_adv=50.0,
        lambda_verify=50.0,
        auditor_steps=bundle.auditor_steps,
        epochs=epochs,
        weight_decay=1e-4,
        early_stopping_patience=patience,
        confusion_type="entropy",
        checkpoint_dir=str(ckpt_dir),
    )

    trainer = PCRLTrainer(
        encoder=encoder,
        task_heads={purpose.name: task_head},
        auditors={purpose.name: auditor},
        config=config,
        purpose_registry=registry,
        device=device,
    )

    log.info(f"  [{bundle.name}/{purpose.name}] training (K={bundle.auditor_steps}, epochs={epochs}, patience={patience})")
    t0 = time.time()
    trainer.train(train_loader, val_loader=val_loader)
    train_time = time.time() - t0
    log.info(f"  [{bundle.name}/{purpose.name}] trained in {train_time:.0f}s")

    # Reload best checkpoint (saved by Trainer when val task_loss improved).
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        encoder.load_state_dict(state["encoder"])
        task_head.load_state_dict({k.removeprefix(f"{purpose.name}."): v
                                   for k, v in state["task_heads"].items()
                                   if k.startswith(f"{purpose.name}.")})
        log.info(f"  [{bundle.name}/{purpose.name}] reloaded best.pt")
    else:
        log.warning(f"  [{bundle.name}/{purpose.name}] best.pt missing, using final state")

    # Compliance report (paper's adjusted criterion).
    reports = generate_report(
        encoder=encoder,
        train_loader=train_loader,
        test_loader=test_loader,
        purpose_registry=registry,
        device=device,
    )

    pass_count = 0
    attr_results = []
    for r in reports:
        delta = r.empirical_best_acc - r.majority_proportion
        ok = bool(delta < 0.02 and r.linear_r2 < 0.05)
        if ok:
            pass_count += 1
        attr_results.append({
            "attribute": r.attr_name,
            "linear_r2": round(r.linear_r2, 6),
            "empirical_best_acc": round(r.empirical_best_acc, 6),
            "majority_baseline": round(r.majority_proportion, 6),
            "delta": round(delta, 6),
            "adj_pass": ok,
        })

    # Task accuracy from the trained head.
    test_eval = trainer.evaluate(test_loader)
    task_accs = {k: round(float(v), 6) for k, v in test_eval.task_accuracy.items()}

    # Representation health on test set.
    test_reprs = extract_reprs(encoder, test_loader, device)
    health = repr_health(test_reprs)

    return {
        "purpose": purpose.name,
        "task": task_name,
        "task_accuracies": task_accs,
        "disallowed_attrs": list(purpose.disallowed_attrs),
        "attribute_results": attr_results,
        "pass_count": pass_count,
        "total_pairs": len(reports),
        "train_time_s": round(train_time, 1),
        "health": health,
    }


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────


PCRL_PAPER = {
    "adult":    {"pass": 6, "total": 8, "task_label": "income",          "task_acc": 0.763},
    "diabetes": {"pass": 5, "total": 6, "task_label": "primary_diagnosis_category", "task_acc": None},
    "hmda":     {"pass": 5.3, "total": 6, "task_label": "loan_decision", "task_acc": None},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["adult", "diabetes", "hmda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default=None,
                        help="cpu, cuda, mps; auto-detect if omitted")
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\n{'=' * 70}")
    print(f"SEPARATE ENCODERS — {args.dataset.upper()} (seed={args.seed}, device={device})")
    print(f"{'=' * 70}\n")

    bundle = build_dataset(args.dataset)
    print(f"  N_train={len(bundle.train_ds)}  N_val={len(bundle.val_ds)}  "
          f"N_test={len(bundle.test_ds)}  D={bundle.input_dim}")
    print(f"  Purposes: {[p.name for p in bundle.purposes]}")
    print(f"  K={bundle.auditor_steps}, lambda_adv=50, lambda_verify=50")
    print()

    overall_t0 = time.time()
    per_purpose: list[dict] = []
    for purpose in bundle.purposes:
        result = train_purpose(
            bundle=bundle,
            purpose=purpose,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
        )
        per_purpose.append(result)
        print(f"  → {purpose.name}: pass {result['pass_count']}/{result['total_pairs']}, "
              f"task_acc={list(result['task_accuracies'].values())}, "
              f"per_dim_std_mean={result['health']['per_dim_std_mean']:.3f}")

    total_train_time = time.time() - overall_t0

    total_pass = sum(r["pass_count"] for r in per_purpose)
    total_pairs = sum(r["total_pairs"] for r in per_purpose)

    # Pull out the headline task accuracy for the "task_label" purpose, if defined.
    paper = PCRL_PAPER[args.dataset]
    headline_task_acc = None
    for r in per_purpose:
        if paper["task_label"] in r["task_accuracies"]:
            headline_task_acc = r["task_accuracies"][paper["task_label"]]
            break

    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "device": device,
        "auditor_steps_K": bundle.auditor_steps,
        "epochs": args.epochs,
        "patience": args.patience,
        "n_purposes": len(bundle.purposes),
        "total_pass": int(total_pass),
        "total_pairs": int(total_pairs),
        "headline_task": paper["task_label"],
        "headline_task_acc": headline_task_acc,
        "pcrl_paper_pass": paper["pass"],
        "pcrl_paper_total": paper["total"],
        "pcrl_paper_task_acc": paper["task_acc"],
        "total_train_time_s": round(total_train_time, 1),
    }

    out_dir = ROOT / "results" / f"{args.dataset}_SEPARATE"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_purpose_results.json", "w") as fh:
        json.dump({"summary": summary, "per_purpose": per_purpose}, fh, indent=2)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print()
    print(f"{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  separate pass: {total_pass}/{total_pairs}")
    print(f"  PCRL paper pass: {paper['pass']}/{paper['total']}")
    if headline_task_acc is not None:
        print(f"  separate task ({paper['task_label']}): {headline_task_acc:.1%}")
        if paper["task_acc"] is not None:
            print(f"  PCRL paper task ({paper['task_label']}): {paper['task_acc']:.1%}")
    print(f"  total wall time: {total_train_time:.0f}s")
    print(f"  saved: {out_dir / 'per_purpose_results.json'}")
    print(f"  saved: {out_dir / 'summary.json'}")
    print()


if __name__ == "__main__":
    main()
