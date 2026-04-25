#!/usr/bin/env python3
"""Run all five methods on UCI Diabetes 130-US with 3 seeds each.

Methods: Standard, LAFTR, INLP, LEACE, PCRL.
For each seed s in {0, 1, 2}:
  - Train Standard encoder (seed s). Extract reprs for train/val/test.
    -> Standard method: evaluate compliance directly on those reprs.
    -> INLP method: per-purpose nullspace projection; evaluate compliance.
    -> LEACE method: per-purpose LEACE eraser; evaluate compliance.
  - Train LAFTR encoder (seed s) with lambda_adv=50 on all purposes' auditors.
    -> LAFTR method: evaluate compliance on that encoder.
  - Train PCRL (PurposeConditionedEncoder) with lambda_adv=50, lambda_verify=50.
    -> PCRL method: evaluate compliance via generate_report (purpose-conditioned).

Writes:
  results/diabetes/pcrl_seeds.csv       (per-pair rows for PCRL, 3 seeds)
  results/diabetes/baselines_seeds.csv  (per-pair rows for Standard/LAFTR/INLP/LEACE)
  results/diabetes/summary.csv          (mean/std pass_count across seeds per method)
  results/diabetes/purposes.md          (preprocessing + purpose documentation)
"""

from __future__ import annotations

import csv
import logging
import sys
import time
import warnings
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

warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")

from pcrl.data.base import collate_pcrl_batch
from pcrl.data.diabetes import DiabetesDataset, get_diabetes_purposes
from pcrl.evaluation.certificates import (
    ComplianceReport, EmpiricalAudit, LinearAudit, generate_report,
)
from pcrl.models.auditor import MultiAttributeAuditor
from pcrl.models.baselines import INLPProjector, LEACEEraser
from pcrl.models.encoder import PurposeConditionedEncoder, StandardEncoder
from pcrl.models.task_head import TaskHead
from pcrl.purposes.spec import PurposeRegistry
from pcrl.training.trainer import PCRLTrainer, TrainerConfig

logging.basicConfig(level=logging.WARNING)

# ── Config (mirrors Adult spec from the user's prompt) ────────────────────
SEEDS = [0, 1, 2]
HIDDEN_DIMS = [128, 128]
REPR_DIM = 64
PURPOSE_EMB_DIM = 32
EPOCHS = 200
PATIENCE = 20
LAMBDA_ADV = 50.0
LAMBDA_VERIFY = 50.0
K_AUDITOR = 20
BATCH_SIZE = 256
DELTA_THRESHOLD = 0.02
R2_THRESHOLD = 0.05


# ── Shared utilities ─────────────────────────────────────────────────────

def extract_reprs(encoder, loader, device):
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


def count_pass(reports):
    return sum(
        1 for r in reports
        if (r.empirical_best_acc - r.majority_proportion) < DELTA_THRESHOLD
        and r.linear_r2 < R2_THRESHOLD
    )


def report_rows(reports, common):
    pc = count_pass(reports)
    rows = []
    for r in reports:
        delta = r.empirical_best_acc - r.majority_proportion
        ok = delta < DELTA_THRESHOLD and r.linear_r2 < R2_THRESHOLD
        row = dict(common)
        row.update({
            "purpose": r.purpose_name,
            "attribute": r.attr_name,
            "best_emp_acc": round(r.empirical_best_acc, 6),
            "majority_baseline": round(r.majority_proportion, 6),
            "delta": round(delta, 6),
            "linear_r2": round(r.linear_r2, 6),
            "adj_pass": ok,
            "pass_count": pc,
            "total_pairs": len(reports),
        })
        rows.append(row)
    return rows


def save_csv(rows, path):
    if not rows:
        print(f"  (no rows for {path})"); return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []; seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"  Saved {path}")


# ── Data ─────────────────────────────────────────────────────────────────

def load_data():
    purposes = get_diabetes_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)
    train_ds = DiabetesDataset(purposes, split="train")
    val_ds = DiabetesDataset(purposes, split="val")
    test_ds = DiabetesDataset(purposes, split="test")
    return {
        "purposes": purposes,
        "registry": registry,
        "input_dim": train_ds.info.num_features,
        "train_loader": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   collate_fn=collate_pcrl_batch, num_workers=0),
        "train_extract_loader": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                                           collate_fn=collate_pcrl_batch, num_workers=0),
        "val_loader": DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_pcrl_batch, num_workers=0),
        "test_loader": DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  collate_fn=collate_pcrl_batch, num_workers=0),
    }


# ── Standard / LAFTR encoder (non-purpose-conditioned) ───────────────────

def _build_shared_heads(purposes):
    task_heads, auditors = {}, {}
    for p in purposes:
        tname = p.allowed_tasks[0]
        task_heads[p.name] = TaskHead(
            repr_dim=REPR_DIM, output_dim=p.allowed_task_dims.get(tname, 2),
        )
        auditors[p.name] = MultiAttributeAuditor(
            repr_dim=REPR_DIM, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )
    return task_heads, auditors


def train_standard(seed, data, device):
    torch.manual_seed(seed); np.random.seed(seed)
    encoder = StandardEncoder(
        input_dim=data["input_dim"], hidden_dims=HIDDEN_DIMS,
        repr_dim=REPR_DIM, dropout=0.3,
    )
    task_heads, auditors = _build_shared_heads(data["purposes"])
    config = TrainerConfig(
        batch_size=BATCH_SIZE, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=0.0, lambda_verify=0.0, auditor_steps=1,
        epochs=EPOCHS, weight_decay=1e-4, early_stopping_patience=PATIENCE,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"diab_std_s{seed}"),
    )
    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=data["registry"], device=device,
    )
    t0 = time.time()
    trainer.train(data["train_loader"], val_loader=data["val_loader"])
    ttime = time.time() - t0

    tr_r, tr_t, tr_s = extract_reprs(encoder, data["train_extract_loader"], device)
    va_r, va_t, va_s = extract_reprs(encoder, data["val_loader"], device)
    te_r, te_t, te_s = extract_reprs(encoder, data["test_loader"], device)
    return {
        "encoder": encoder, "train_time": ttime,
        "train_reprs": tr_r, "val_reprs": va_r, "test_reprs": te_r,
        "train_tasks": tr_t, "val_tasks": va_t, "test_tasks": te_t,
        "train_sens": tr_s, "val_sens": va_s, "test_sens": te_s,
    }


def train_laftr(seed, data, device):
    torch.manual_seed(seed); np.random.seed(seed)
    encoder = StandardEncoder(
        input_dim=data["input_dim"], hidden_dims=HIDDEN_DIMS,
        repr_dim=REPR_DIM, dropout=0.3,
    )
    task_heads, auditors = _build_shared_heads(data["purposes"])
    config = TrainerConfig(
        batch_size=BATCH_SIZE, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY, auditor_steps=K_AUDITOR,
        epochs=EPOCHS, weight_decay=1e-4, early_stopping_patience=PATIENCE,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"diab_laftr_s{seed}"),
    )
    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=data["registry"], device=device,
    )
    t0 = time.time()
    trainer.train(data["train_loader"], val_loader=data["val_loader"])
    ttime = time.time() - t0
    reports = generate_report(
        encoder=encoder, train_loader=data["train_loader"],
        test_loader=data["test_loader"], purpose_registry=data["registry"],
        device=device,
    )
    return reports, ttime


def train_pcrl(seed, data, device):
    torch.manual_seed(seed); np.random.seed(seed)
    purposes = data["purposes"]
    encoder = PurposeConditionedEncoder(
        input_dim=data["input_dim"], hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM,
        num_purposes=len(purposes), purpose_emb_dim=PURPOSE_EMB_DIM,
        conditioning="film", dropout=0.3,
    )
    task_heads, auditors = _build_shared_heads(purposes)
    config = TrainerConfig(
        batch_size=BATCH_SIZE, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=LAMBDA_ADV, lambda_verify=LAMBDA_VERIFY, auditor_steps=K_AUDITOR,
        epochs=EPOCHS, weight_decay=1e-4, early_stopping_patience=PATIENCE,
        confusion_type="entropy",
        checkpoint_dir=str(project_root / "checkpoints" / f"diab_pcrl_s{seed}"),
    )
    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=config, purpose_registry=data["registry"], device=device,
    )
    t0 = time.time()
    trainer.train(data["train_loader"], val_loader=data["val_loader"])
    ttime = time.time() - t0
    reports = generate_report(
        encoder=encoder, train_loader=data["train_loader"],
        test_loader=data["test_loader"], purpose_registry=data["registry"],
        device=device,
    )
    return reports, ttime


# ── INLP / LEACE (post-hoc on Standard reprs) ─────────────────────────────

def apply_inlp(std, purposes):
    all_reports = []
    for p in purposes:
        proj_tr = std["train_reprs"].copy()
        proj_te = std["test_reprs"].copy()
        for attr in p.disallowed_attrs:
            proj = INLPProjector(max_iters=50, min_accuracy=0.52, random_state=42)
            proj.fit(proj_tr, std["train_sens"][attr])
            proj_tr = proj.transform(proj_tr)
            proj_te = proj.transform(proj_te)
        all_reports.extend(compliance_on_reprs(
            proj_tr, proj_te, std["train_sens"], std["test_sens"],
            p.name, p.disallowed_attrs,
        ))
    return all_reports


def apply_leace(std, purposes):
    all_reports = []
    for p in purposes:
        proj_tr = std["train_reprs"].copy()
        proj_te = std["test_reprs"].copy()
        for attr in p.disallowed_attrs:
            eraser = LEACEEraser(regularization=1e-4)
            eraser.fit(proj_tr, std["train_sens"][attr])
            proj_tr = eraser.transform(proj_tr)
            proj_te = eraser.transform(proj_te)
        all_reports.extend(compliance_on_reprs(
            proj_tr, proj_te, std["train_sens"], std["test_sens"],
            p.name, p.disallowed_attrs,
        ))
    return all_reports


def standard_reports_on_test(std, purposes):
    """Compliance on raw Standard reprs (no projection)."""
    all_reports = []
    for p in purposes:
        all_reports.extend(compliance_on_reprs(
            std["train_reprs"], std["test_reprs"],
            std["train_sens"], std["test_sens"],
            p.name, p.disallowed_attrs,
        ))
    return all_reports


# ── Purposes doc ──────────────────────────────────────────────────────────

PURPOSES_MD = """# UCI Diabetes 130-US — PCRL setup

## Source

Strack et al. (2014). UCI ML Repository, Diabetes 130-US hospitals for
years 1999-2008 (101,766 encounters, 130 hospitals, deidentified, public).

https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

## Preprocessing (experiments/preprocess_diabetes.py)

1. Dropped high-missing columns: `weight`, `payer_code`, `medical_specialty`.
2. Dropped rows with missing `age` or `diag_1`.
3. Deduplicated by `patient_nbr` (first encounter only) to prevent patient-level
   leakage across splits.
4. Encoded sensitive attributes:
   - `race` (5 categories: Caucasian, AfricanAmerican, Hispanic, Asian, Other;
     "?" mapped to Other)
   - `gender` (binary: Female/Male)
   - `age_bucket` (10 decade buckets: [0-10), ..., [90-100))
5. Grouped primary diagnosis `diag_1` into 9 ICD-9 categories using the Strack
   protocol:
   - circulatory (390-459, 785)
   - diabetes (250.xx)
   - digestive (520-579, 787)
   - injury (800-999)
   - musculoskeletal (710-739)
   - respiratory (460-519, 786)
   - genitourinary (580-629, 788)
   - neoplasms (140-239)
   - other (V/E codes + everything else)
6. Targets:
   - `readmission_outcome`: 1 if `readmitted == "<30"` else 0.
   - `medication_change_outcome`: 1 if `change == "Ch"` else 0.
   - `primary_diagnosis_category`: 9-class (above grouping).
7. Feature matrix: z-normalized numericals + one-hot categoricals (admission/
   discharge/source IDs, 23 medication columns × 4 levels, A1Cresult,
   max_glu_serum, diabetesMed, age_bucket).
8. 70/15/15 stratified split on `readmission_outcome`, seed=42.

## Purposes (3 purposes, 6 disallowed-attr pairs)

| Purpose | Allowed task | Disallowed attributes |
|---|---|---|
| `billing_audit` | `primary_diagnosis_category` (9-class) | race, gender |
| `quality_research` | `readmission_outcome` (binary) | race, age_bucket |
| `clinical_decision_support` | `medication_change_outcome` (binary) | race, gender |

## Methods and hyperparameters

Shared: MLP [128, 128], repr_dim=64, batch=256, epochs=200, patience=20,
seed ∈ {0, 1, 2}, 3 disallowed-attr auditor hidden layers × 256 dim.

| Method | Notes |
|---|---|
| Standard | StandardEncoder, no adversarial loss. Reprs audited directly. |
| LAFTR | StandardEncoder, all purposes' auditors, λ_adv=λ_verify=50, K=20. |
| INLP | Applied post-hoc to Standard reprs, max_iters=50, min_acc=0.52. |
| LEACE | Applied post-hoc to Standard reprs, regularization=1e-4. |
| PCRL | PurposeConditionedEncoder (FiLM), λ_adv=λ_verify=50, K=20. |

Compliance gate: delta < 2% AND linear R² < 5%.
"""


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t_total = time.time()

    data = load_data()
    print(f"Diabetes: input_dim={data['input_dim']}, "
          f"purposes={[p.name for p in data['purposes']]}")

    out_dir = project_root / "results" / "diabetes"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "purposes.md").write_text(PURPOSES_MD)
    print(f"  Wrote {out_dir / 'purposes.md'}")

    pcrl_rows = []
    baseline_rows = []
    per_seed_method_pass = {}  # (method, seed) -> pass_count
    per_seed_method_time = {}  # (method, seed) -> train_time_s

    for seed in SEEDS:
        print(f"\n{'=' * 60}\n=== SEED {seed}\n{'=' * 60}")

        # ── Standard (also powers INLP/LEACE) ──────────────────────────
        print(f"\n[seed={seed}] Training Standard...", flush=True)
        std = train_standard(seed, data, device)
        print(f"  Standard trained in {std['train_time']:.0f}s")
        per_seed_method_time[("Standard", seed)] = std["train_time"]

        std_reports = standard_reports_on_test(std, data["purposes"])
        pc = count_pass(std_reports)
        per_seed_method_pass[("Standard", seed)] = pc
        print(f"  Standard: {pc}/{len(std_reports)} pairs pass")
        baseline_rows.extend(report_rows(std_reports, {
            "method": "Standard", "seed": seed,
            "train_time_s": round(std["train_time"], 1),
        }))

        # ── INLP ───────────────────────────────────────────────────────
        print(f"\n[seed={seed}] Applying INLP...", flush=True)
        t0 = time.time()
        inlp_reports = apply_inlp(std, data["purposes"])
        inlp_time = time.time() - t0
        pc = count_pass(inlp_reports)
        per_seed_method_pass[("INLP", seed)] = pc
        per_seed_method_time[("INLP", seed)] = inlp_time
        print(f"  INLP: {pc}/{len(inlp_reports)} pairs pass ({inlp_time:.0f}s)")
        baseline_rows.extend(report_rows(inlp_reports, {
            "method": "INLP", "seed": seed,
            "train_time_s": round(inlp_time, 1),
        }))

        # ── LEACE ──────────────────────────────────────────────────────
        print(f"\n[seed={seed}] Applying LEACE...", flush=True)
        t0 = time.time()
        leace_reports = apply_leace(std, data["purposes"])
        leace_time = time.time() - t0
        pc = count_pass(leace_reports)
        per_seed_method_pass[("LEACE", seed)] = pc
        per_seed_method_time[("LEACE", seed)] = leace_time
        print(f"  LEACE: {pc}/{len(leace_reports)} pairs pass ({leace_time:.0f}s)")
        baseline_rows.extend(report_rows(leace_reports, {
            "method": "LEACE", "seed": seed,
            "train_time_s": round(leace_time, 1),
        }))

        # ── LAFTR ──────────────────────────────────────────────────────
        print(f"\n[seed={seed}] Training LAFTR...", flush=True)
        laftr_reports, laftr_time = train_laftr(seed, data, device)
        pc = count_pass(laftr_reports)
        per_seed_method_pass[("LAFTR", seed)] = pc
        per_seed_method_time[("LAFTR", seed)] = laftr_time
        print(f"  LAFTR: {pc}/{len(laftr_reports)} pairs pass ({laftr_time:.0f}s)")
        baseline_rows.extend(report_rows(laftr_reports, {
            "method": "LAFTR", "seed": seed,
            "train_time_s": round(laftr_time, 1),
        }))

        # ── PCRL ───────────────────────────────────────────────────────
        print(f"\n[seed={seed}] Training PCRL...", flush=True)
        pcrl_reports, pcrl_time = train_pcrl(seed, data, device)
        pc = count_pass(pcrl_reports)
        per_seed_method_pass[("PCRL", seed)] = pc
        per_seed_method_time[("PCRL", seed)] = pcrl_time
        print(f"  PCRL: {pc}/{len(pcrl_reports)} pairs pass ({pcrl_time:.0f}s)")
        pcrl_rows.extend(report_rows(pcrl_reports, {
            "method": "PCRL", "seed": seed,
            "train_time_s": round(pcrl_time, 1),
        }))

        # ── Save incrementally ────────────────────────────────────────
        save_csv(pcrl_rows, out_dir / "pcrl_seeds.csv")
        save_csv(baseline_rows, out_dir / "baselines_seeds.csv")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}\n=== SUMMARY\n{'=' * 60}")
    method_order = ["Standard", "LAFTR", "INLP", "LEACE", "PCRL"]
    summary_rows = []
    for method in method_order:
        passes = [per_seed_method_pass[(method, s)] for s in SEEDS
                  if (method, s) in per_seed_method_pass]
        times = [per_seed_method_time[(method, s)] for s in SEEDS
                 if (method, s) in per_seed_method_time]
        if not passes:
            continue
        mean_p = float(np.mean(passes))
        std_p = float(np.std(passes, ddof=0))
        summary_rows.append({
            "method": method,
            "seeds": ",".join(str(s) for s in SEEDS if (method, s) in per_seed_method_pass),
            "pass_count_mean": round(mean_p, 4),
            "pass_count_std": round(std_p, 4),
            "pass_count_per_seed": ",".join(str(p) for p in passes),
            "train_time_s_mean": round(float(np.mean(times)), 1),
            "total_pairs": 6,
        })
        print(f"  {method:<10} pass={mean_p:.2f}±{std_p:.2f}  "
              f"per-seed={passes}  avg_train_s={float(np.mean(times)):.0f}")

    save_csv(summary_rows, out_dir / "summary.csv")

    total_min = (time.time() - t_total) / 60
    print(f"\nTotal: {total_min:.1f} min ({total_min/60:.2f} h)")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
