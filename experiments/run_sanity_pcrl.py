#!/usr/bin/env python3
"""Phase-1 sanity check: train PCRL seed-N on {diabetes, hmda} with the
combined adversarial stabilization fix, then verify the representations are
healthy and the required tasks beat majority. Exits 0 on PASS, 1 on FAIL."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.evaluation.certificates import generate_report  # noqa: E402
from pcrl.models.auditor import MultiAttributeAuditor  # noqa: E402
from pcrl.models.encoder import PurposeConditionedEncoder  # noqa: E402
from pcrl.models.task_head import TaskHead  # noqa: E402
from pcrl.purposes.spec import PurposeRegistry  # noqa: E402
from pcrl.training.trainer import PCRLTrainer, TrainerConfig  # noqa: E402

REPR_DIM = 64
HIDDEN_DIMS = [128, 128]
PURPOSE_EMB_DIM_DEFAULTS = {"diabetes": 16, "hmda": 32, "adult": 32}


def load_diabetes():
    from pcrl.data.diabetes import DiabetesDataset, get_diabetes_purposes
    purposes = get_diabetes_purposes()
    return (
        purposes,
        DiabetesDataset(purposes=purposes, split="train"),
        DiabetesDataset(purposes=purposes, split="val"),
        DiabetesDataset(purposes=purposes, split="test"),
    )


def load_hmda():
    from pcrl.data.hmda import HMDADataset, get_hmda_purposes
    purposes = get_hmda_purposes()
    return (
        purposes,
        HMDADataset(purposes=purposes, split="train"),
        HMDADataset(purposes=purposes, split="val"),
        HMDADataset(purposes=purposes, split="test"),
    )


def load_adult():
    from pcrl.data.adult import AdultDataset, get_adult_purposes
    purposes = get_adult_purposes()
    return (
        purposes,
        AdultDataset(purposes=purposes, root="data", split="train", download=True),
        AdultDataset(purposes=purposes, root="data", split="val", download=False),
        AdultDataset(purposes=purposes, root="data", split="test", download=False),
    )


LOADERS = {"diabetes": load_diabetes, "hmda": load_hmda, "adult": load_adult}


def effective_rank(reprs: np.ndarray) -> float:
    s = np.linalg.svd(reprs - reprs.mean(axis=0, keepdims=True), compute_uv=False)
    p = (s ** 2) / max((s ** 2).sum(), 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["diabetes", "hmda", "adult"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--lambda-adv", type=float, default=50.0)
    ap.add_argument("--lambda-verify", type=float, default=50.0)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.dataset}] device={device}, seed={args.seed}")

    purposes, train_ds, val_ds, test_ds = LOADERS[args.dataset]()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)
    feat_dim = train_ds.features.shape[1]
    print(f"[{args.dataset}] feat_dim={feat_dim}, train={len(train_ds)}, "
          f"val={len(val_ds)}, test={len(test_ds)}, purposes={len(purposes)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    encoder = PurposeConditionedEncoder(
        input_dim=feat_dim, hidden_dims=HIDDEN_DIMS, repr_dim=REPR_DIM,
        num_purposes=len(purposes),
        purpose_emb_dim=PURPOSE_EMB_DIM_DEFAULTS[args.dataset],
        conditioning="film", dropout=0.3,
    )
    task_heads = {
        p.name: TaskHead(
            repr_dim=REPR_DIM,
            output_dim=p.allowed_task_dims[p.allowed_tasks[0]],
        )
        for p in purposes
    }
    auditors = {
        p.name: MultiAttributeAuditor(
            repr_dim=REPR_DIM, attr_output_dims=p.disallowed_attr_dims,
            hidden_dim=256, num_layers=3,
        )
        for p in purposes
    }
    ckpt_dir = ROOT / "checkpoints" / f"sanity_{args.dataset}_s{args.seed}"
    cfg = TrainerConfig(
        batch_size=args.batch_size, lr_encoder=1e-3, lr_auditor=1e-3,
        lambda_adv=args.lambda_adv, lambda_verify=args.lambda_verify,
        auditor_steps=args.K, epochs=args.epochs, weight_decay=1e-4,
        early_stopping_patience=args.patience, confusion_type="entropy",
        checkpoint_dir=str(ckpt_dir),
    )
    trainer = PCRLTrainer(
        encoder=encoder, task_heads=task_heads, auditors=auditors,
        config=cfg, purpose_registry=registry, device=device,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin,
                              collate_fn=collate_pcrl_batch)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False,
                            num_workers=0, pin_memory=pin,
                            collate_fn=collate_pcrl_batch)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=0, pin_memory=pin,
                             collate_fn=collate_pcrl_batch)

    t0 = time.time()
    state = trainer.train(train_loader, val_loader=val_loader)
    train_time_s = time.time() - t0

    best_path = ckpt_dir / "best.pt"
    if not best_path.exists():
        print(f"FATAL: best checkpoint missing at {best_path}", file=sys.stderr)
        return 2
    ck = torch.load(best_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ck["encoder"])
    nn.ModuleDict(task_heads).load_state_dict(ck["task_heads"])
    best_epoch = ck["state"]["epoch"]

    encoder.eval()
    for h in task_heads.values():
        h.eval()

    per_purpose: dict[str, dict] = {}
    task_acc: dict[str, dict] = {}
    majority_baselines: dict[str, float] = {}

    with torch.no_grad():
        for idx, p in enumerate(purposes):
            reprs_chunks = []
            preds_chunks: dict[str, list] = {t: [] for t in p.allowed_tasks}
            target_chunks: dict[str, list] = {t: [] for t in p.allowed_tasks}
            head = task_heads[p.name].to(device)
            for batch in test_loader:
                x = batch["features"].to(device)
                h = encoder(x, idx)
                reprs_chunks.append(h.cpu().numpy())
                logits = head(h)
                pri = p.allowed_tasks[0]
                preds_chunks[pri].append(logits.argmax(dim=-1).cpu().numpy())
                target_chunks[pri].append(batch["task_labels"][pri].numpy())
            reprs = np.concatenate(reprs_chunks)
            per_dim_std = reprs.std(axis=0)
            l2 = np.linalg.norm(reprs, axis=1)
            per_purpose[p.name] = {
                "shape": list(reprs.shape),
                "per_dim_std_mean": float(per_dim_std.mean()),
                "per_dim_std_max": float(per_dim_std.max()),
                "per_dim_std_min": float(per_dim_std.min()),
                "l2_norm_mean": float(l2.mean()),
                "l2_norm_std": float(l2.std()),
                "effective_rank": effective_rank(reprs),
            }
            for tname in p.allowed_tasks:
                if not preds_chunks[tname]:
                    continue
                preds = np.concatenate(preds_chunks[tname])
                targets = np.concatenate(target_chunks[tname])
                acc = float((preds == targets).mean())
                _, cnts = np.unique(targets, return_counts=True)
                maj = float(cnts.max() / cnts.sum())
                task_acc[f"{p.name}/{tname}"] = {
                    "acc": acc, "majority": maj, "delta": acc - maj,
                }
                majority_baselines[tname] = maj

    # ── Per-pair audit eval ──────────────────────────────────────────────
    # Saved auditor (loose, fast): the auditor weights from best.pt scored
    # on the test set per (purpose, sensitive attr).
    # Empirical post-hoc + R² (paper Table 3 methodology): generate_report()
    # fits a fresh post-hoc auditor per pair and computes the linear-attack
    # R² of the representation. Pair passes if Δacc<2pp AND R²<0.05.
    audit_saved: dict[str, dict] = {}
    sens_majority: dict[str, float] = {}
    if "auditors" in ck:
        aud_modules = nn.ModuleDict(auditors).to(device)
        aud_modules.load_state_dict(ck["auditors"])
        aud_modules.eval()
        feats_chunks: list[torch.Tensor] = []
        sens_chunks: dict[str, list] = {a: [] for a in test_ds.sensitive_attrs}
        for batch in test_loader:
            feats_chunks.append(batch["features"])
            for a in sens_chunks:
                if a in batch.get("sensitive_attrs", {}):
                    sens_chunks[a].append(batch["sensitive_attrs"][a])
        feats_t = torch.cat(feats_chunks).to(device)
        sens_lab = {
            a: torch.cat(c).cpu().numpy() if c else None
            for a, c in sens_chunks.items()
        }
        for a, lbls in test_ds.sensitive_attrs.items():
            _, c = lbls.unique(return_counts=True)
            sens_majority[a] = float(c.max().item() / len(lbls))
        with torch.no_grad():
            for idx, p in enumerate(purposes):
                z = encoder(feats_t, idx)
                preds_dict = aud_modules[p.name].predict(z)
                for attr in p.disallowed_attrs:
                    if attr not in preds_dict or sens_lab.get(attr) is None:
                        continue
                    preds = preds_dict[attr].cpu().numpy()
                    acc = float((preds == sens_lab[attr]).mean())
                    base = sens_majority[attr]
                    audit_saved[f"{p.name}/{attr}"] = {
                        "acc": acc, "majority": base, "delta": acc - base,
                        "pass": (acc - base) < 0.02,
                    }

    audit_pair: dict[str, dict] = {}
    try:
        reports = generate_report(
            encoder=encoder, train_loader=train_loader,
            test_loader=test_loader, purpose_registry=registry,
            device=device,
        )
        for r in reports:
            delta = r.empirical_best_acc - r.majority_proportion
            audit_pair[f"{r.purpose_name}/{r.attr_name}"] = {
                "best_acc": float(r.empirical_best_acc),
                "majority": float(r.majority_proportion),
                "delta": float(delta),
                "linear_r2": float(r.linear_r2),
                "pass_adjusted": bool(delta < 0.02 and r.linear_r2 < 0.05),
            }
    except Exception as e:
        print(f"[{args.dataset}] generate_report failed: {e}", file=sys.stderr)

    audit_pass_count = sum(1 for v in audit_saved.values() if v["pass"])
    audit_pass_count_adjusted = sum(
        1 for v in audit_pair.values() if v["pass_adjusted"]
    )
    audit_total = len(audit_saved) or len(audit_pair)

    # Sanity verdict
    reasons: list[str] = []
    mean_per_dim_std = float(np.mean(
        [per_purpose[p.name]["per_dim_std_mean"] for p in purposes]
    ))
    max_l2_norm_std = float(max(
        per_purpose[p.name]["l2_norm_std"] for p in purposes
    ))
    if mean_per_dim_std <= 0.5:
        reasons.append(f"mean per-dim std {mean_per_dim_std:.3f} <= 0.5")
    if max_l2_norm_std <= 1.0:
        reasons.append(f"max l2 norm std {max_l2_norm_std:.3f} <= 1.0")
    if best_epoch <= 5:
        reasons.append(f"best epoch {best_epoch} <= 5")

    if args.dataset == "diabetes":
        primary_d = task_acc.get(
            "billing_audit/primary_diagnosis_category", {}
        ).get("delta", -1.0)
        med_d = task_acc.get(
            "clinical_decision_support/medication_change_outcome", {}
        ).get("delta", -1.0)
        if primary_d <= 0.05:
            reasons.append(
                f"primary_diagnosis delta {primary_d:+.3f} <= 0.05 (REQUIRED)"
            )
        if med_d <= 0.05:
            reasons.append(
                f"medication_change delta {med_d:+.3f} <= 0.05 (REQUIRED)"
            )
    elif args.dataset == "hmda":
        amount_acc = task_acc.get(
            "pricing_analysis/loan_amount_band", {}
        ).get("acc", 0.0)
        decision_acc = task_acc.get(
            "underwriting/loan_decision", {}
        ).get("acc", 0.0)
        if amount_acc <= 0.35:
            reasons.append(
                f"loan_amount_band acc {amount_acc:.3f} <= 0.35 (REQUIRED)"
            )
        if decision_acc <= 0.80:
            reasons.append(
                f"loan_decision acc {decision_acc:.3f} <= 0.80 (REQUIRED)"
            )
    elif args.dataset == "adult":
        # STRICT criterion: adjusted_pass_count is the only gate.
        # Saved-auditor count is informational only.
        for p in purposes:
            ppd = per_purpose[p.name]["per_dim_std_mean"]
            if not (0.5 <= ppd <= 1.5):
                reasons.append(
                    f"{p.name} per_dim_std_mean {ppd:.3f} not in [0.5, 1.5]"
                )
            l2s = per_purpose[p.name]["l2_norm_std"]
            if l2s <= 1.0:
                reasons.append(f"{p.name} l2_norm_std {l2s:.3f} <= 1.0")
        income_acc = task_acc.get(
            "income_prediction/income", {}
        ).get("acc", 0.0)
        if not (0.75 <= income_acc <= 0.78):
            reasons.append(
                f"income acc {income_acc:.3f} not in [0.75, 0.78] "
                f"(paper Table 3: 0.763)"
            )
        if audit_total > 0 and audit_pass_count_adjusted < 5:
            reasons.append(
                f"adjusted_pass_count {audit_pass_count_adjusted}/{audit_total} "
                f"< 5 — paper achieves 6/8; revert may not be sufficient"
            )

    sanity_pass = len(reasons) == 0

    out = {
        "dataset": args.dataset,
        "seed": args.seed,
        "sanity_pass": sanity_pass,
        "fail_reasons": reasons,
        "stopped_epoch": state.epoch,
        "best_epoch": best_epoch,
        "train_time_s": train_time_s,
        "config": {
            "lambda_adv": args.lambda_adv,
            "lambda_verify": args.lambda_verify,
            "K": args.K,
            "patience": args.patience,
            "epochs": args.epochs,
            "lambda_anneal": False,
            "use_spectral_norm": False,
            "auditor_hidden": 256,
            "auditor_layers": 3,
        },
        "majority_baselines": majority_baselines,
        "per_purpose": per_purpose,
        "task_acc": task_acc,
        "audit_saved": audit_saved,
        "audit_pair": audit_pair,
        "audit_pass_count": audit_pass_count,
        "audit_pass_count_adjusted": audit_pass_count_adjusted,
        "audit_total": audit_total,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{args.dataset}] sanity {'PASS' if sanity_pass else 'FAIL'}: "
          f"wrote {out_path}")
    if reasons:
        for r in reasons:
            print(f"  - {r}")
    return 0 if sanity_pass else 1


if __name__ == "__main__":
    sys.exit(main())
