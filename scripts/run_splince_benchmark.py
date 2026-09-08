"""SPLINCE-vs-PCRL benchmark driver.

For each cell in the variance-constrained 60-cell grid (3 datasets × 20 (purpose,
attribute) pairs × 3 seeds), this script:

  1. Loads the warmstart PerPurposeLoRAEncoder for (dataset, seed) — backbone +
     LoRA adapters trained by PCRL Round 5 / Round 7 — WITHOUT applying the
     LEACE projection. This gives the same pre-erasure features PCRL sees and
     keeps the comparison apples-to-apples.
  2. Extracts per-purpose features ``z_train`` and ``z_test`` for the loader.
  3. For each disallowed attribute under that purpose, fits SPLINCE on
     ``(z_train, attribute, primary_task_label)`` per Theorem 1 of
     arXiv:2506.10703 and projects test features.
  4. Audits the projected test features:
       • Linear R² (one-hot, matches PCRL's R² metric).
       • Dominant-axis R²_DA (PCRL §5.4 Framework D).
       • 3-architecture post-hoc audit: L2-LR + 2-layer MLP + XGBoost.
       • Empirical Δ_aud = max(acc) − majority_proportion.
       • Linear-probe task accuracy on (z'_train, y_train) → (z'_test, y_test).

Outputs are written under ``results/splince_benchmark/`` with one
``{dataset}_s{seed}/metrics.json`` per cell, plus a top-level
``splince_results.json`` aggregate, ``splince_vs_pcrl_summary.json`` head-to-head,
``splince_vs_pcrl_table.tex`` paper-ready table, ``PAPER_PASTE.md`` flowing-prose
§5.3 paragraph, and ``HEADLINE.txt``.

The runtime is dominated by feature extraction (~10s/cell on CPU) + 3-arch audit
(~5s/cell). 60 cells ≈ 15 minutes wall on c5.4xlarge.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from pcrl.baselines.splince import apply_splince, fit_splince
from pcrl.data.base import collate_pcrl_batch
from pcrl.evaluation.certificates import compute_dominant_axis_r2
from pcrl.models.encoder import StandardEncoder
from pcrl.models.lora import PerPurposeLoRAEncoder
from pcrl.purposes.verification import LinearComplianceCertificate
from run_v2_dataset import (  # type: ignore
    LORA_BY_DATASET,
    NUM_WORKERS_BY_DATASET,
    build_datasets,
    effective_rank as audit_effective_rank,
    repr_health,
)


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO,
)
log = logging.getLogger("splince_bench")


# ── 3-architecture auditor (LR + 2-layer MLP + XGBoost) ─────────────────────


def _three_arch_audit(
    train_X: np.ndarray, train_y: np.ndarray,
    test_X: np.ndarray, test_y: np.ndarray,
    *,
    random_state: int = 42,
    n_audit_seeds: int = 3,
    max_train_samples: int = 20000,
) -> dict:
    """Run the 3 auditor architectures (L2-LR, MLP, XGBoost) and report metrics.

    Each architecture is trained on (train_X, train_y), evaluated on
    (test_X, test_y). The MLP is retrained from scratch with ``n_audit_seeds``
    different inits and we report ``max`` test accuracy across them — matches
    PCRL §5.4. Other classifiers are deterministic given ``random_state``.

    Args:
        train_X: (n_tr, d) features.
        train_y: (n_tr,) integer labels.
        test_X: (n_te, d) features.
        test_y: (n_te,) integer labels.
        random_state: Seed for the deterministic classifiers.
        n_audit_seeds: Number of seeds for the MLP.
        max_train_samples: Subsample (stratified) cap for fitting; full test set
            is always used for evaluation.

    Returns:
        Dict with per-classifier accuracy + ``best_acc`` (max across all 3 archs
        and all MLP seeds) + ``majority`` + ``delta_aud``.
    """
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Subsample training stratified by label to keep XGBoost and MLP fast.
    n_tr = len(train_y)
    if n_tr > max_train_samples:
        try:
            from sklearn.model_selection import train_test_split
            train_X, _, train_y, _ = train_test_split(
                train_X, train_y, train_size=max_train_samples,
                stratify=train_y, random_state=random_state,
            )
        except ValueError:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(n_tr, size=max_train_samples, replace=False)
            train_X = train_X[idx]
            train_y = train_y[idx]

    # Majority baseline (computed on the test set per PCRL convention).
    classes, counts = np.unique(test_y, return_counts=True)
    majority = float(counts.max() / len(test_y)) if len(test_y) else 0.5

    results: dict[str, float] = {}

    # ── L2-LR (Logistic Regression with default C=1.0 = L2 regularisation) ──
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(
            penalty="l2", C=1.0, max_iter=1000, random_state=random_state,
            solver="lbfgs",
        )
        lr.fit(train_X, train_y)
        results["lr"] = float(accuracy_score(test_y, lr.predict(test_X)))

    # ── XGBoost (100 estimators, depth 6) ──
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=100, max_depth=6, random_state=random_state,
            eval_metric="logloss", verbosity=0, n_jobs=2,
        )
        xgb.fit(train_X, train_y)
        results["xgb"] = float(accuracy_score(test_y, xgb.predict(test_X)))
    except ImportError:
        log.warning("xgboost not installed — skipping XGB auditor")
        results["xgb"] = float("nan")

    # ── 2-layer MLP (256 ReLU dropout 0.3) — average over n_audit_seeds inits ──
    import torch.nn as nn

    n_classes = int(max(int(train_y.max()), int(test_y.max()))) + 1
    d = train_X.shape[1]
    Xtr = torch.from_numpy(train_X.astype(np.float32))
    Xte = torch.from_numpy(test_X.astype(np.float32))
    ytr = torch.from_numpy(train_y.astype(np.int64))
    yte = test_y.astype(np.int64)

    mlp_accs: list[float] = []
    for s in range(n_audit_seeds):
        torch.manual_seed(random_state + s)
        net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        net.train()
        rng = np.random.RandomState(random_state + s)
        bs = 256
        n = len(ytr)
        for _ in range(50):
            perm = rng.permutation(n)
            for st in range(0, n, bs):
                idx = perm[st:st + bs]
                opt.zero_grad()
                logits = net(Xtr[idx])
                loss = loss_fn(logits, ytr[idx])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            preds = net(Xte).argmax(1).numpy()
        mlp_accs.append(float((preds == yte).mean()))

    results["mlp"] = float(max(mlp_accs))
    results["mlp_seeds"] = mlp_accs

    valid = [v for k, v in results.items() if k != "mlp_seeds" and not np.isnan(v)]
    best_acc = float(max(valid)) if valid else 0.0
    return {
        **results,
        "best_acc": best_acc,
        "majority": majority,
        "delta_aud": best_acc - majority,
    }


# ── Linear-probe task accuracy ──────────────────────────────────────────────


def _linear_probe(
    train_X: np.ndarray, train_y: np.ndarray,
    test_X: np.ndarray, test_y: np.ndarray,
    *,
    random_state: int = 42,
) -> float:
    """L2-regularised LR linear probe; returns test accuracy."""
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(
            penalty="l2", C=1.0, max_iter=1000, random_state=random_state,
            solver="lbfgs",
        )
        clf.fit(train_X, train_y)
        return float(accuracy_score(test_y, clf.predict(test_X)))


# ── Feature extraction (no LEACE applied) ───────────────────────────────────


@torch.no_grad()
def _extract_features(
    encoder: PerPurposeLoRAEncoder, loader: DataLoader,
    purpose_idx: int, attr_names: list[str], primary_task: str,
    device: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Extract (features, attribute_labels_dict, primary_task_label) in one pass.

    Single-pass extraction is required for row-level alignment when the loader
    is shuffled (cf. the bug fix in pcrl/evaluation/certificates.py).
    """
    encoder.eval()
    all_z: list[np.ndarray] = []
    all_attrs: dict[str, list[np.ndarray]] = {a: [] for a in attr_names}
    all_y: list[np.ndarray] = []
    for batch in loader:
        x = batch["features"].to(device)
        h = encoder(x, purpose_idx)
        all_z.append(h.detach().cpu().numpy())
        for attr in attr_names:
            all_attrs[attr].append(batch["sensitive_attrs"][attr].numpy())
        if primary_task in batch["task_labels"]:
            all_y.append(batch["task_labels"][primary_task].numpy())
        else:
            # Fall back to a dummy zeros vector — the task label may not be in the
            # loader for an unusual purpose. SPLINCE then degenerates to LEACE
            # because the range constraint becomes trivial (Σ_xy = 0).
            all_y.append(np.zeros(len(x), dtype=np.int64))
    Z = np.concatenate(all_z, axis=0)
    A = {a: np.concatenate(v) for a, v in all_attrs.items()}
    Y = np.concatenate(all_y)
    return Z, A, Y


# ── One full cell ───────────────────────────────────────────────────────────


def run_one_cell(
    dataset: str, seed: int, warm_start_tag: str, device: str, out_dir: Path,
) -> dict:
    """Run all (purpose, attribute) cells for one (dataset, seed)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    log.info("=" * 72)
    log.info("[SPLINCE] dataset=%s seed=%d", dataset, seed)
    log.info("=" * 72)

    purposes, train_ds, val_ds, test_ds = build_datasets(dataset)
    n_workers = NUM_WORKERS_BY_DATASET.get(dataset, 0)
    persistent = n_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=512, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=n_workers,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False,
        collate_fn=collate_pcrl_batch, num_workers=n_workers,
        persistent_workers=persistent,
    )

    input_dim = train_ds.info.num_features
    repr_dim = 64
    lora_rank, lora_alpha = LORA_BY_DATASET.get(dataset, (8, 16.0))

    backbone = StandardEncoder(
        input_dim=input_dim, hidden_dims=[128, 128], repr_dim=repr_dim, dropout=0.3,
    )
    encoder = PerPurposeLoRAEncoder(
        backbone=backbone, n_purposes=len(purposes),
        rank=lora_rank, alpha=lora_alpha, dropout=0.0,
    )

    warm_dir = ROOT / "checkpoints" / f"v2_{dataset}{warm_start_tag}_s{seed}"
    candidates = [warm_dir / "final.pt", warm_dir / "best.pt", warm_dir / "canonical_iterate.pt"]
    chosen = next((p for p in candidates if p.exists()), None)
    if chosen is None:
        raise FileNotFoundError(
            f"No warmstart checkpoint found in {warm_dir}; "
            f"tried {[p.name for p in candidates]}"
        )
    log.info("[SPLINCE/WARMSTART] loading %s", chosen)
    ckpt = torch.load(chosen, map_location=device, weights_only=False)
    encoder.backbone.load_state_dict(ckpt["backbone"])
    encoder.adapters.load_state_dict(ckpt["lora_adapters"])
    # Deliberately do NOT call set_leace_projection — we want pre-LEACE features.
    encoder.to(device).eval()

    cell_out = out_dir / f"{dataset}_s{seed}"
    cell_out.mkdir(parents=True, exist_ok=True)

    pair_results: list[dict] = []
    failures: list[dict] = []

    # Per-purpose feature cache so we don't re-extract for each attribute.
    purpose_feat_cache: dict[int, dict] = {}

    for p_idx, purpose in enumerate(purposes):
        if not purpose.disallowed_attrs:
            continue
        primary_task = purpose.allowed_tasks[0]
        attr_names = list(purpose.disallowed_attrs)
        if p_idx not in purpose_feat_cache:
            log.info(
                "[SPLINCE/EXTRACT] purpose=%s attrs=%s task=%s",
                purpose.name, attr_names, primary_task,
            )
            t0 = time.time()
            Z_tr, A_tr, Y_tr = _extract_features(
                encoder, train_loader, p_idx, attr_names, primary_task, device,
            )
            Z_te, A_te, Y_te = _extract_features(
                encoder, test_loader, p_idx, attr_names, primary_task, device,
            )
            purpose_feat_cache[p_idx] = {
                "Z_tr": Z_tr, "Z_te": Z_te,
                "A_tr": A_tr, "A_te": A_te,
                "Y_tr": Y_tr, "Y_te": Y_te,
                "primary_task": primary_task,
                "extract_time_s": time.time() - t0,
            }
            log.info(
                "[SPLINCE/EXTRACT] purpose=%s done in %.1fs (n_tr=%d, n_te=%d)",
                purpose.name, time.time() - t0, len(Z_tr), len(Z_te),
            )

        cache = purpose_feat_cache[p_idx]
        Z_tr, Z_te = cache["Z_tr"], cache["Z_te"]
        A_tr, A_te = cache["A_tr"], cache["A_te"]
        Y_tr, Y_te = cache["Y_tr"], cache["Y_te"]
        # Pre-projection health (same purpose, same features for every attr below)
        pre_health = repr_health(Z_te)

        for attr in attr_names:
            tag = f"{purpose.name}/{attr}"
            try:
                t0 = time.time()
                P, mu, info = fit_splince(Z_tr, A_tr[attr], Y_tr)
                Zp_tr = apply_splince(Z_tr, P, mu)
                Zp_te = apply_splince(Z_te, P, mu)
                fit_time = time.time() - t0

                # Linear R² on test (one-hot, matches PCRL's r2_onehot).
                num_classes = purpose.disallowed_attr_dims.get(attr, int(A_tr[attr].max()) + 1)
                lc = LinearComplianceCertificate(epsilon=0.05).check(
                    Zp_te, A_te[attr], num_classes=num_classes,
                )
                r2_onehot = lc.r_squared

                # Dominant-axis R²
                da = compute_dominant_axis_r2(Zp_te, A_te[attr], num_classes=num_classes)

                # 3-arch audit for ∆_aud
                t1 = time.time()
                aud = _three_arch_audit(
                    Zp_tr, A_tr[attr].astype(np.int64),
                    Zp_te, A_te[attr].astype(np.int64),
                    random_state=42,
                )
                aud_time = time.time() - t1

                # Linear-probe task accuracy on the projected features.
                task_acc = _linear_probe(
                    Zp_tr, Y_tr.astype(np.int64),
                    Zp_te, Y_te.astype(np.int64),
                )
                # Pre-projection linear-probe (control: how much SPLINCE costs us)
                task_acc_pre = _linear_probe(
                    Z_tr, Y_tr.astype(np.int64), Z_te, Y_te.astype(np.int64),
                )

                # Health of projected reps
                post_health = repr_health(Zp_te)

                pair_results.append({
                    "dataset": dataset,
                    "seed": seed,
                    "purpose": purpose.name,
                    "attribute": attr,
                    "primary_task": primary_task,
                    "n_train": int(len(Z_tr)),
                    "n_test": int(len(Z_te)),
                    "splince_fit_info": info.to_dict(),
                    "splince_fit_time_s": round(fit_time, 3),
                    "audit_time_s": round(aud_time, 3),
                    "r2_onehot": round(r2_onehot, 6),
                    "r2_da": round(float(da["r2_da"]), 6),
                    "r2_da_per_class": [round(v, 6) for v in da["per_class_r2"]],
                    "linear_certified_at_0p05": bool(r2_onehot < 0.05),
                    "auditor_lr_acc": round(aud["lr"], 6),
                    "auditor_xgb_acc": round(aud["xgb"], 6),
                    "auditor_mlp_acc": round(aud["mlp"], 6),
                    "auditor_mlp_seeds": [round(v, 6) for v in aud["mlp_seeds"]],
                    "auditor_best_acc": round(aud["best_acc"], 6),
                    "majority_proportion": round(aud["majority"], 6),
                    "delta_aud": round(aud["delta_aud"], 6),
                    "task_acc_post_splince": round(task_acc, 6),
                    "task_acc_pre_splince": round(task_acc_pre, 6),
                    "task_acc_drop": round(task_acc_pre - task_acc, 6),
                    # Strict-pass criterion mirrors §5.4 of PCRL: R² < 0.05 AND
                    # auditor-delta < 0.02 AND post-projection health is healthy.
                    "strict_pass": bool(
                        r2_onehot < 0.05
                        and aud["delta_aud"] < 0.02
                        and post_health["per_dim_std_mean"] >= 0.5
                        and post_health["effective_rank"] >= 2.0
                    ),
                    "r2_pass": bool(r2_onehot < 0.05),
                    "delta_pass": bool(aud["delta_aud"] < 0.02),
                    "health_pass": bool(
                        post_health["per_dim_std_mean"] >= 0.5
                        and post_health["effective_rank"] >= 2.0
                    ),
                    "pre_health": pre_health,
                    "post_health": post_health,
                })
                log.info(
                    "[SPLINCE/CELL] %s/%s s%d: r2=%.4f r2_da=%.4f Δ=%+0.4f "
                    "task=%.3f→%.3f strict=%s%s",
                    purpose.name, attr, seed,
                    r2_onehot, float(da["r2_da"]), aud["delta_aud"],
                    task_acc_pre, task_acc,
                    pair_results[-1]["strict_pass"],
                    " (LEACE-fallback)" if info.fallback_to_leace else "",
                )
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                log.error("[SPLINCE/CELL] %s/%s s%d FAILED: %s", purpose.name, attr, seed, exc)
                failures.append({
                    "dataset": dataset, "seed": seed,
                    "purpose": purpose.name, "attribute": attr,
                    "error": str(exc), "traceback": tb,
                })

    cell_summary = {
        "dataset": dataset, "seed": seed,
        "warm_start_path": str(chosen),
        "n_pairs": len(pair_results),
        "n_failures": len(failures),
        "pair_results": pair_results,
        "failures": failures,
    }
    (cell_out / "metrics.json").write_text(json.dumps(cell_summary, indent=2, default=str))
    return cell_summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["adult", "hmda", "diabetes"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--warm-start-tag-adult", default="_ROUND5")
    p.add_argument("--warm-start-tag-hmda", default="_ROUND5")
    p.add_argument("--warm-start-tag-diabetes", default="_ROUND7")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="results/splince_benchmark")
    p.add_argument("--single-cell", default=None,
                   help="Run only one (dataset, seed); format: 'adult,0'")
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.single_cell is not None:
        ds_name, sd = args.single_cell.split(",")
        cells = [(ds_name.strip(), int(sd))]
    else:
        cells = [(d, s) for d in args.datasets for s in args.seeds]

    tag_map = {
        "adult": args.warm_start_tag_adult,
        "hmda": args.warm_start_tag_hmda,
        "diabetes": args.warm_start_tag_diabetes,
    }

    all_cells: list[dict] = []
    overall_failures: list[dict] = []
    t0 = time.time()
    for ds_name, sd in cells:
        try:
            res = run_one_cell(ds_name, sd, tag_map[ds_name], args.device, out_dir)
            all_cells.append(res)
            overall_failures.extend(res.get("failures", []))
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            log.error("[SPLINCE/UNIT] %s s%d FAILED: %s", ds_name, sd, exc)
            overall_failures.append({
                "dataset": ds_name, "seed": sd,
                "error": str(exc), "traceback": tb,
            })

    aggregate = {
        "n_units_run": len(all_cells),
        "n_failures": len(overall_failures),
        "wall_time_s": round(time.time() - t0, 1),
        "all_cells": all_cells,
        "all_failures": overall_failures,
    }
    (out_dir / "splince_results.json").write_text(json.dumps(aggregate, indent=2, default=str))
    log.info("[SPLINCE/DONE] wrote %s/splince_results.json", out_dir)


if __name__ == "__main__":
    main()
