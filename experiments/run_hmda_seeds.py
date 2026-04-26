#!/usr/bin/env python3
"""HMDA 2023 California: 5 methods × 3 seeds.

Methods (matched to Adult/diabetes hyperparameters):
    Standard, LAFTR, INLP, LEACE, PCRL.
    MLP [128, 128], repr_dim=64, lambda_adv=50, lambda_verify=50,
    K=20, epochs=200, patience=20.

Outputs:
    results/hmda/baselines_seeds.csv   — Standard / LAFTR / INLP / LEACE
    results/hmda/pcrl_seeds.csv        — PCRL only
    results/hmda/summary.csv           — per-method mean ± std across seeds
    results/hmda/purposes.md           — purpose definitions for the paper

Run order: seed 0 → seed 1 → seed 2.  Within each seed, methods run
Standard → LAFTR → INLP → LEACE → PCRL so that the most informative pair
(Standard for the base rate, PCRL for the proposed method) is captured
even if a later method crashes.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
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

from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.data.hmda import HMDADataset, get_hmda_purposes  # noqa: E402
from pcrl.evaluation.certificates import (  # noqa: E402
    ComplianceReport,
    EmpiricalAudit,
    LinearAudit,
    generate_report,
)
from pcrl.models.auditor import MultiAttributeAuditor  # noqa: E402
from pcrl.models.baselines import INLPProjector, LEACEEraser  # noqa: E402
from pcrl.models.encoder import PurposeConditionedEncoder, StandardEncoder  # noqa: E402
from pcrl.models.task_head import TaskHead  # noqa: E402
from pcrl.purposes.spec import PurposeRegistry  # noqa: E402
from pcrl.training.trainer import PCRLTrainer, TrainerConfig  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")


# ── Hyperparameters (matched to Adult/diabetes) ─────────────────────────
HIDDEN_DIMS = [128, 128]
REPR_DIM = 64
LAMBDA_ADV = 50.0
LAMBDA_VERIFY = 50.0
AUDITOR_STEPS = 20
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 256
LR = 1e-3
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4

PURPOSE_EMB_DIM = 32
# Combined adversarial stabilization fix: 1-layer 64-unit auditor with
# spectral norm + dropout 0.5; lambda annealing handled by TrainerConfig.
AUDITOR_HIDDEN = 64
AUDITOR_LAYERS = 1
AUDITOR_DROPOUT = 0.5


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
    reps: list[np.ndarray] = []
    tasks: dict[str, list[np.ndarray]] = {}
    sens: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            h = encoder(x, purpose_idx) if purpose_idx is not None else encoder(x)
            reps.append(h.cpu().numpy())
            for k, v in batch["task_labels"].items():
                tasks.setdefault(k, []).append(v.numpy())
            for k, v in batch["sensitive_attrs"].items():
                sens.setdefault(k, []).append(v.numpy())
    return (
        np.concatenate(reps),
        {k: np.concatenate(v) for k, v in tasks.items()},
        {k: np.concatenate(v) for k, v in sens.items()},
    )


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
    out: list[ComplianceReport] = []
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
            train_reprs, train_labels, test_reprs, test_labels
        )
        empirical_ok = (best_acc - chance_acc) < 0.05
        certified = linear_result.certified and empirical_ok
        out.append(
            ComplianceReport(
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
            )
        )
    return out


def results_to_rows(result: MethodResult, seed: int) -> list[dict]:
    pass_count = 0
    for r in result.reports:
        delta = r.empirical_best_acc - r.majority_proportion
        if delta < 0.02 and r.linear_r2 < 0.05:
            pass_count += 1
    total = len(result.reports)
    rows: list[dict] = []
    for r in result.reports:
        delta = r.empirical_best_acc - r.majority_proportion
        ok = delta < 0.02 and r.linear_r2 < 0.05
        row = {
            "dataset": "hmda",
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
        for task, acc in sorted(result.task_accuracies.items()):
            row[f"{task}_acc"] = round(acc, 6)
        rows.append(row)
    return rows


def save_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def append_rows(rows: list[dict], path: Path) -> None:
    """Append rows to CSV, writing the header on first call only."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    write_header = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────
# Per-seed runner
# ─────────────────────────────────────────────────────────────────────────

def run_hmda_seed(
    seed: int,
    device: str,
    out_dir: Path,
    methods: list[str],
) -> None:
    print(f"\n{'='*60}\nHMDA — seed {seed}\n{'='*60}")

    purposes = get_hmda_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    train_ds = HMDADataset(purposes=purposes, root="data", split="train")
    val_ds = HMDADataset(purposes=purposes, root="data", split="val")
    test_ds = HMDADataset(purposes=purposes, root="data", split="test")
    input_dim = train_ds.info.num_features
    print(f"  N_train={len(train_ds):,}, N_val={len(val_ds):,}, N_test={len(test_ds):,}, D={input_dim}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)

    pcrl_path = out_dir / "pcrl_seeds.csv"
    base_path = out_dir / "baselines_seeds.csv"

    train_reprs_std: np.ndarray | None = None
    test_reprs_std: np.ndarray | None = None
    train_tasks: dict[str, np.ndarray] | None = None
    test_tasks: dict[str, np.ndarray] | None = None
    train_sens: dict[str, np.ndarray] | None = None
    test_sens: dict[str, np.ndarray] | None = None

    # ── 1. Standard ─────────────────────────────────────────────────────
    if "Standard" in methods:
        print("  Standard...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        std_encoder = StandardEncoder(
            input_dim=input_dim, hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM, dropout=DROPOUT,
        )
        task_heads, auditors = {}, {}
        for p in purposes:
            tn = p.allowed_tasks[0]
            od = p.allowed_task_dims.get(tn, 2)
            task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=od)
            auditors[p.name] = MultiAttributeAuditor(
                repr_dim=REPR_DIM, attr_output_dims=p.disallowed_attr_dims,
                hidden_dim=AUDITOR_HIDDEN, num_layers=AUDITOR_LAYERS,
                dropout=AUDITOR_DROPOUT, use_spectral_norm=True,
            )
        cfg = TrainerConfig(
            batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
            lambda_adv=0.0, lambda_verify=0.0, auditor_steps=1,
            epochs=EPOCHS, weight_decay=WEIGHT_DECAY,
            early_stopping_patience=PATIENCE,
            confusion_type="entropy", lambda_anneal=True,
            checkpoint_dir=str(project_root / "checkpoints" / f"hmda_std_{seed}"),
        )
        tr = PCRLTrainer(encoder=std_encoder, task_heads=task_heads, auditors=auditors,
                         config=cfg, purpose_registry=registry, device=device)
        t0 = time.time()
        tr.train(train_loader, val_loader=val_loader)
        std_time = time.time() - t0
        std_eval = tr.evaluate(test_loader)

        train_reprs_std, train_tasks, train_sens = extract_representations(std_encoder, train_loader, device)
        test_reprs_std, test_tasks, test_sens = extract_representations(std_encoder, test_loader, device)

        std_reports = generate_report(
            encoder=std_encoder, train_loader=train_loader, test_loader=test_loader,
            purpose_registry=registry, device=device,
        )
        result = MethodResult(
            name="Standard", task_accuracies=std_eval.task_accuracy,
            reports=std_reports, train_time=std_time,
        )
        append_rows(results_to_rows(result, seed), base_path)
        race_deltas = [r.empirical_best_acc - r.majority_proportion for r in std_reports if r.attr_name == "race"]
        avg_race = sum(race_deltas) / len(race_deltas) if race_deltas else 0.0
        print(f"    decision_acc={std_eval.task_accuracy.get('loan_decision', 0):.1%}, Race Δ={avg_race:+.1%}, {std_time:.0f}s")

    # ── 2. LAFTR ────────────────────────────────────────────────────────
    if "LAFTR" in methods:
        print("  LAFTR...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        enc = StandardEncoder(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM, dropout=DROPOUT)
        task_heads, auditors = {}, {}
        for p in purposes:
            tn = p.allowed_tasks[0]
            od = p.allowed_task_dims.get(tn, 2)
            task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=od)
            auditors[p.name] = MultiAttributeAuditor(
                repr_dim=REPR_DIM, attr_output_dims=p.disallowed_attr_dims,
                hidden_dim=AUDITOR_HIDDEN, num_layers=AUDITOR_LAYERS,
                dropout=AUDITOR_DROPOUT, use_spectral_norm=True,
            )
        cfg = TrainerConfig(
            batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
            lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY, auditor_steps=AUDITOR_STEPS,
            epochs=EPOCHS, weight_decay=WEIGHT_DECAY,
            early_stopping_patience=PATIENCE,
            confusion_type="entropy", lambda_anneal=True,
            checkpoint_dir=str(project_root / "checkpoints" / f"hmda_laftr_{seed}"),
        )
        tr = PCRLTrainer(encoder=enc, task_heads=task_heads, auditors=auditors,
                         config=cfg, purpose_registry=registry, device=device)
        t0 = time.time()
        tr.train(train_loader, val_loader=val_loader)
        ltime = time.time() - t0
        leval = tr.evaluate(test_loader)
        lreports = generate_report(
            encoder=enc, train_loader=train_loader, test_loader=test_loader,
            purpose_registry=registry, device=device,
        )
        result = MethodResult(
            name="LAFTR", task_accuracies=leval.task_accuracy,
            reports=lreports, train_time=ltime,
        )
        append_rows(results_to_rows(result, seed), base_path)
        print(f"    decision_acc={leval.task_accuracy.get('loan_decision', 0):.1%}, {ltime:.0f}s")

    # ── 3. INLP ─────────────────────────────────────────────────────────
    if "INLP" in methods and train_reprs_std is not None:
        print("  INLP...")
        t0 = time.time()
        ireports: list[ComplianceReport] = []
        itacc: dict[str, float] = {}
        for p in purposes:
            ptr, pte = train_reprs_std.copy(), test_reprs_std.copy()
            for an in p.disallowed_attrs:
                proj = INLPProjector(max_iters=35, min_accuracy=0.52)
                proj.fit(ptr, train_sens[an])
                ptr = proj.transform(ptr)
                pte = proj.transform(pte)
            tn = p.allowed_tasks[0]
            itacc[tn] = evaluate_task_accuracy(ptr, train_tasks[tn], pte, test_tasks[tn])
            ireports.extend(run_compliance_on_reprs(ptr, pte, train_sens, test_sens, p.name, p.disallowed_attrs))
        itime = time.time() - t0
        result = MethodResult(name="INLP", task_accuracies=itacc, reports=ireports, train_time=itime)
        append_rows(results_to_rows(result, seed), base_path)
        print(f"    decision_acc={itacc.get('loan_decision', 0):.1%}, {itime:.0f}s")

    # ── 4. LEACE ────────────────────────────────────────────────────────
    if "LEACE" in methods and train_reprs_std is not None:
        print("  LEACE...")
        t0 = time.time()
        lreports: list[ComplianceReport] = []
        ltacc: dict[str, float] = {}
        for p in purposes:
            ptr, pte = train_reprs_std.copy(), test_reprs_std.copy()
            for an in p.disallowed_attrs:
                eraser = LEACEEraser()
                eraser.fit(ptr, train_sens[an])
                ptr = eraser.transform(ptr)
                pte = eraser.transform(pte)
            tn = p.allowed_tasks[0]
            ltacc[tn] = evaluate_task_accuracy(ptr, train_tasks[tn], pte, test_tasks[tn])
            lreports.extend(run_compliance_on_reprs(ptr, pte, train_sens, test_sens, p.name, p.disallowed_attrs))
        ltime = time.time() - t0
        result = MethodResult(name="LEACE", task_accuracies=ltacc, reports=lreports, train_time=ltime)
        append_rows(results_to_rows(result, seed), base_path)
        print(f"    decision_acc={ltacc.get('loan_decision', 0):.1%}, {ltime:.0f}s")

    # ── 5. PCRL ────────────────────────────────────────────────────────
    if "PCRL" in methods:
        print("  PCRL...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        enc = PurposeConditionedEncoder(
            input_dim=input_dim, hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM,
            num_purposes=len(purposes), purpose_emb_dim=PURPOSE_EMB_DIM,
            conditioning="film", dropout=DROPOUT,
        )
        task_heads, auditors = {}, {}
        for p in purposes:
            tn = p.allowed_tasks[0]
            od = p.allowed_task_dims.get(tn, 2)
            task_heads[p.name] = TaskHead(repr_dim=REPR_DIM, output_dim=od)
            auditors[p.name] = MultiAttributeAuditor(
                repr_dim=REPR_DIM, attr_output_dims=p.disallowed_attr_dims,
                hidden_dim=AUDITOR_HIDDEN, num_layers=AUDITOR_LAYERS,
                dropout=AUDITOR_DROPOUT, use_spectral_norm=True,
            )
        cfg = TrainerConfig(
            batch_size=BATCH_SIZE, lr_encoder=LR, lr_auditor=LR,
            lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY, auditor_steps=AUDITOR_STEPS,
            epochs=EPOCHS, weight_decay=WEIGHT_DECAY,
            early_stopping_patience=PATIENCE,
            confusion_type="entropy", lambda_anneal=True,
            checkpoint_dir=str(project_root / "checkpoints" / f"hmda_pcrl_{seed}"),
        )
        tr = PCRLTrainer(encoder=enc, task_heads=task_heads, auditors=auditors,
                         config=cfg, purpose_registry=registry, device=device)
        t0 = time.time()
        tr.train(train_loader, val_loader=val_loader)
        ptime = time.time() - t0
        peval = tr.evaluate(test_loader)
        preports = generate_report(
            encoder=enc, train_loader=train_loader, test_loader=test_loader,
            purpose_registry=registry, device=device,
        )
        result = MethodResult(
            name="PCRL", task_accuracies=peval.task_accuracy,
            reports=preports, train_time=ptime,
        )
        append_rows(results_to_rows(result, seed), pcrl_path)
        print(f"    decision_acc={peval.task_accuracy.get('loan_decision', 0):.1%}, {ptime:.0f}s")


# ─────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────

def compute_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"])].append(row)

    out_rows: list[dict] = []
    for (dataset, method), grp in sorted(groups.items()):
        seed_data: dict[int, list[dict]] = defaultdict(list)
        for row in grp:
            seed_data[row["seed"]].append(row)

        seed_metrics: list[dict] = []
        for _seed, srows in sorted(seed_data.items()):
            task_acc_cols = [k for k in srows[0].keys() if k.endswith("_acc")]
            task_accs = {col: srows[0][col] for col in task_acc_cols}
            attr_deltas: dict[str, float] = {}
            for r in srows:
                attr = r["attribute"]
                attr_deltas[attr] = max(attr_deltas.get(attr, -1.0), r["delta"])
            sm = {
                **task_accs,
                **{f"delta_{a}": d for a, d in attr_deltas.items()},
                "pass_count": srows[0]["pass_count"],
                "total_pairs": srows[0]["total_pairs"],
            }
            seed_metrics.append(sm)

        summary = {"dataset": dataset, "method": method, "n_seeds": len(seed_metrics)}
        all_keys: set[str] = set()
        for sm in seed_metrics:
            all_keys.update(sm.keys())
        for key in sorted(all_keys):
            vals = [sm.get(key, float("nan")) for sm in seed_metrics]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if vals:
                summary[f"{key}_mean"] = round(float(np.mean(vals)), 4)
                summary[f"{key}_std"] = round(float(np.std(vals, ddof=0)), 4)
        out_rows.append(summary)
    return out_rows


def write_purposes_md(out_path: Path) -> None:
    md = """# HMDA 2023 California — Purpose Specifications

Three purposes, six (purpose, sensitive-attribute) compliance pairs.
All sensitive attributes are derived per the HMDA 2023 schema.

| Purpose              | Allowed task          | Disallowed attributes  | Why                                                          |
|----------------------|-----------------------|------------------------|--------------------------------------------------------------|
| underwriting         | loan_decision         | race, ethnicity        | ECOA prohibits race and national origin from credit decisions. |
| pricing_analysis     | loan_amount_band      | race, sex              | Disparate-pricing analysis must not depend on protected class. |
| fair_lending_audit   | tract_denial_high     | race, sex              | Tract-level audit needs aggregate signal, not individual demographics. |

Filters applied to the public LAR snapshot:

- `state_code = CA`, `activity_year = 2023`
- `loan_purpose = 1` (home purchase)
- `lien_status = 1` (first lien)
- `construction_method = 1` (site-built)
- `action_taken in {1, 3}` (originated or denied)
- known race / ethnicity / sex / age / debt-to-income (drops `Joint`,
  `Sex Not Available`, `Race Not Available`, `8888`, `9999`, `Exempt`, `NA`)

Tasks:

- **loan_decision**: 1 if `action_taken = 1`, 0 if `action_taken = 3`.
- **loan_amount_band**: 5-class quintile of `loan_amount` fitted on
  training data only (cutoffs persisted in `metadata.json`).
- **tract_denial_high**: 1 if the applicant's `census_tract` has a denial
  rate above the median per-tract denial rate (median computed over
  filtered records before splitting).

Sensitive attribute encodings:

- **race** (5): 0=White, 1=Black, 2=Asian, 3=AIAN/NHPI/2+, 4=Joint.
- **ethnicity** (2): 0=Not Hispanic, 1=Hispanic.
- **sex** (2): 0=Female, 1=Male.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(md)
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--methods", nargs="+",
        default=["Standard", "LAFTR", "INLP", "LEACE", "PCRL"],
        help="Subset/ordering of methods. PCRL+LAFTR are prioritised; "
             "see runner ordering inside run_hmda_seed.",
    )
    parser.add_argument("--out-dir", default=str(project_root / "results" / "hmda"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_purposes_md(out_dir / "purposes.md")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, seeds: {args.seeds}, methods: {args.methods}")

    for seed in args.seeds:
        run_hmda_seed(seed, device, out_dir, args.methods)

    # ── Combined summary ────────────────────────────────────────────────
    all_rows: list[dict] = []
    for path in [out_dir / "baselines_seeds.csv", out_dir / "pcrl_seeds.csv"]:
        if not path.exists():
            continue
        with open(path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # Coerce numeric columns
                for k, v in list(row.items()):
                    if k in {"seed", "pass_count", "total_pairs"}:
                        row[k] = int(v) if v else 0
                    elif k in {"adj_pass"}:
                        row[k] = v == "True"
                    elif k in {"dataset", "method", "purpose", "attribute"}:
                        pass
                    else:
                        try:
                            row[k] = float(v)
                        except (TypeError, ValueError):
                            pass
                all_rows.append(row)
    summary = compute_summary(all_rows)
    save_rows(summary, out_dir / "summary.csv")

    print("\nHMDA seed runs complete.")


if __name__ == "__main__":
    main()
