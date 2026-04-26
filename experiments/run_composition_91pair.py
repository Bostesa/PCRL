#!/usr/bin/env python3
"""Composition rule benchmark: additive vs sequential FiLM on Adult.

Loads the trained PCRL Adult checkpoint (``checkpoints/adult/best.pt``)
and builds a single composed representation by combining the three
purpose embeddings (income_prediction, employment_analysis,
education_assessment) with each composition rule:

- Additive: e_composed = e_0 + e_1 + e_2; passed once through
  ``encoder.forward_with_embedding``.
- Sequential: each FiLM head applied in order at every hidden layer via
  ``encoder.forward_sequential_composition``.

For each of the 14 input columns of the Adult schema we derive an
integer-coded label (binning continuous columns; collapsing rare
categories) and run a per-attribute compliance audit on each composed
representation. The 91 = C(14, 2) attribute pairs are then evaluated as
"the composed purpose claims to suppress BOTH attribute i AND attribute
j": a pair passes iff both attribute audits pass.

Outputs (relative to the project root):
    results/adult/composition_91pair.csv
        attribute_i, attribute_j, method, delta, r_squared, adj_pass,
        allowed_task_acc

    results/adult/composition_summary.csv
        method, total_pairs, pass_count, mean_delta, mean_r_squared,
        sequential_strict_better
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcrl.data.adult import AdultDataset, get_adult_purposes  # noqa: E402
from pcrl.data.base import collate_pcrl_batch  # noqa: E402
from pcrl.evaluation.composition import (  # noqa: E402
    audit_all_attrs,
    extract_composed_reprs_additive,
    extract_composed_reprs_sequential,
)
from pcrl.models.encoder import PurposeConditionedEncoder  # noqa: E402
from pcrl.models.task_head import TaskHead  # noqa: E402
from pcrl.purposes.spec import PurposeRegistry  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("composition_91pair")
log.setLevel(logging.INFO)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
warnings.filterwarnings("ignore", message=".*pin_memory.*")


# ─────────────────────────────────────────────────────────────────────────
# Adult: 14-attribute extractor
# ─────────────────────────────────────────────────────────────────────────

# The 14 input columns in the Adult schema (income is the prediction
# target, not a candidate disallowed attribute).
ADULT_14_COLUMNS: list[str] = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]


def discretise_adult_attrs(raw_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return integer-coded labels for each of the 14 Adult columns.

    Continuous columns are binned (quartiles or domain-meaningful cuts);
    high-cardinality categoricals get a ``rare`` bucket. Each output
    array has shape ``(N,)`` and aligns with ``raw_df.index``.
    """
    out: dict[str, np.ndarray] = {}
    n = len(raw_df)

    # Numeric → quartile bins (or domain bins for capital-gain/loss which
    # are 95% zero). Use rank-based binning to avoid all-same-edges errors.
    def quartile_bin(values: np.ndarray) -> np.ndarray:
        ranks = pd.Series(values).rank(method="first").values
        # 4 buckets with equal counts
        return np.minimum((ranks - 1) // (n / 4 + 1e-9), 3).astype(np.int64)

    out["age"] = quartile_bin(raw_df["age"].astype(float).values)
    out["fnlwgt"] = quartile_bin(raw_df["fnlwgt"].astype(float).values)
    out["education-num"] = quartile_bin(raw_df["education-num"].astype(float).values)
    out["hours-per-week"] = quartile_bin(raw_df["hours-per-week"].astype(float).values)

    # capital-gain / capital-loss: 88% zero in Adult; bin as zero / non-zero.
    out["capital-gain"] = (raw_df["capital-gain"].astype(float).values > 0).astype(np.int64)
    out["capital-loss"] = (raw_df["capital-loss"].astype(float).values > 0).astype(np.int64)

    # Categoricals → integer codes; collapse very rare categories (<1%) into
    # a single "_rare_" bucket so the empirical audit doesn't see a degenerate
    # singleton class.
    def codes_with_rare(series: pd.Series, threshold: float = 0.01) -> np.ndarray:
        counts = series.value_counts(normalize=True)
        common = set(counts[counts >= threshold].index)
        s = series.where(series.isin(common), other="_rare_")
        cats = sorted(s.unique().tolist())
        mapping = {c: i for i, c in enumerate(cats)}
        return s.map(mapping).astype(np.int64).values

    out["workclass"] = codes_with_rare(raw_df["workclass"].astype(str))
    out["education"] = codes_with_rare(raw_df["education"].astype(str))
    out["marital-status"] = codes_with_rare(raw_df["marital-status"].astype(str))
    out["occupation"] = codes_with_rare(raw_df["occupation"].astype(str))
    out["relationship"] = codes_with_rare(raw_df["relationship"].astype(str))
    out["race"] = codes_with_rare(raw_df["race"].astype(str))
    out["sex"] = codes_with_rare(raw_df["sex"].astype(str))
    out["native-country"] = codes_with_rare(raw_df["native-country"].astype(str))

    # Sanity: every column has at least 2 classes; all aligned to N.
    for name in ADULT_14_COLUMNS:
        assert name in out, f"missing column {name}"
        assert len(out[name]) == n, f"{name} length {len(out[name])} != N={n}"
        assert len(np.unique(out[name])) >= 2, f"{name} collapsed to singleton class"

    return out


# ─────────────────────────────────────────────────────────────────────────
# Encoder loader
# ─────────────────────────────────────────────────────────────────────────


def load_adult_pcrl_encoder(
    purposes: list,
    input_dim: int,
    checkpoint_path: Path,
    device: str,
) -> PurposeConditionedEncoder:
    encoder = PurposeConditionedEncoder(
        input_dim=input_dim,
        hidden_dims=[128, 128],
        repr_dim=64,
        num_purposes=len(purposes),
        purpose_emb_dim=32,
        conditioning="film",
        dropout=0.3,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("encoder") or ckpt.get("state_dict") or ckpt
    encoder.load_state_dict(state)
    encoder.to(device).eval()
    log.info(f"loaded encoder from {checkpoint_path}")
    return encoder


def load_income_task_head(
    repr_dim: int,
    checkpoint_path: Path,
    purpose_name: str,
    output_dim: int,
    device: str,
) -> TaskHead:
    head = TaskHead(repr_dim=repr_dim, output_dim=output_dim)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    heads = ckpt.get("task_heads", {})
    if purpose_name in heads:
        head.load_state_dict(heads[purpose_name])
    else:
        log.warning(f"task head for purpose '{purpose_name}' not in checkpoint; using random init")
    head.to(device).eval()
    return head


# ─────────────────────────────────────────────────────────────────────────
# Pair-level result aggregation
# ─────────────────────────────────────────────────────────────────────────


def pair_pass(audit_i, audit_j) -> bool:
    return audit_i.adj_pass and audit_j.adj_pass


def pair_delta(audit_i, audit_j) -> float:
    return max(audit_i.delta, audit_j.delta)


def pair_r2(audit_i, audit_j) -> float:
    return max(audit_i.r_squared, audit_j.r_squared)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint .pt; default checkpoints/adult/best.pt")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended before .csv (e.g. _FIXED)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")

    # ── Data + purposes ──────────────────────────────────────────────────
    purposes = get_adult_purposes()
    registry = PurposeRegistry()
    for p in purposes:
        registry.register(p)

    log.info("loading Adult dataset...")
    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=True)
    test_ds = AdultDataset(
        purposes=purposes, root="data", split="test", download=False,
        norm_stats=train_ds.norm_stats,
    )
    input_dim = train_ds.info.num_features
    log.info(f"  N_train={len(train_ds)}, N_test={len(test_ds)}, D={input_dim}")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)

    # ── Discretise 14 attributes from raw df ─────────────────────────────
    log.info("discretising 14 Adult attributes from raw df...")
    train_attrs_full = discretise_adult_attrs(train_ds.raw_df)
    test_attrs_full = discretise_adult_attrs(test_ds.raw_df)
    log.info(f"  attrs: {list(train_attrs_full.keys())}")

    # ── Load checkpoint ──────────────────────────────────────────────────
    ckpt_path = Path(args.ckpt) if args.ckpt else ROOT / "checkpoints" / "adult" / "best.pt"
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Adult PCRL checkpoint not found at {ckpt_path}")

    encoder = load_adult_pcrl_encoder(purposes, input_dim, ckpt_path, device)

    # Income task head, used for allowed_task_acc on each composed rep.
    income_head = load_income_task_head(
        repr_dim=64, checkpoint_path=ckpt_path,
        purpose_name="income_prediction", output_dim=2, device=device,
    )

    # ── Build the K=3 purpose embedding list ─────────────────────────────
    with torch.no_grad():
        embs = [encoder.get_purpose_embedding(i).detach().clone() for i in range(len(purposes))]
    log.info(f"composing K={len(embs)} purpose embeddings (purposes: {[p.name for p in purposes]})")

    # ── Extract composed representations: additive vs sequential ─────────
    log.info("extracting additive-composed reps...")
    t0 = time.time()
    train_reprs_add = extract_composed_reprs_additive(encoder, train_loader, embs, device)
    test_reprs_add = extract_composed_reprs_additive(encoder, test_loader, embs, device)
    log.info(f"  additive done in {time.time() - t0:.1f}s  shape={train_reprs_add.shape}")

    log.info("extracting sequential-composed reps...")
    t0 = time.time()
    train_reprs_seq = extract_composed_reprs_sequential(encoder, train_loader, embs, device)
    test_reprs_seq = extract_composed_reprs_sequential(encoder, test_loader, embs, device)
    log.info(f"  sequential done in {time.time() - t0:.1f}s  shape={train_reprs_seq.shape}")

    # ── Allowed task accuracy on the income head ─────────────────────────
    log.info("evaluating income task accuracy on each composed rep...")
    income_train = train_ds.task_labels["income"].numpy()
    income_test = test_ds.task_labels["income"].numpy()
    with torch.no_grad():
        # Use the trained income head directly on the composed reps.
        logits_add = income_head(torch.from_numpy(test_reprs_add).to(device))
        logits_seq = income_head(torch.from_numpy(test_reprs_seq).to(device))
        income_acc_add = float((logits_add.argmax(dim=-1).cpu().numpy() == income_test).mean())
        income_acc_seq = float((logits_seq.argmax(dim=-1).cpu().numpy() == income_test).mean())
    log.info(f"  income_acc additive   = {income_acc_add:.4f}")
    log.info(f"  income_acc sequential = {income_acc_seq:.4f}")

    # ── Per-attribute audits (run once per method) ───────────────────────
    log.info("auditing 14 attributes on additive reps...")
    t0 = time.time()
    audits_add = audit_all_attrs(
        train_reprs=train_reprs_add, test_reprs=test_reprs_add,
        train_attrs=train_attrs_full, test_attrs=test_attrs_full,
        attr_names=ADULT_14_COLUMNS,
    )
    log.info(f"  additive audits done in {time.time() - t0:.0f}s")

    log.info("auditing 14 attributes on sequential reps...")
    t0 = time.time()
    audits_seq = audit_all_attrs(
        train_reprs=train_reprs_seq, test_reprs=test_reprs_seq,
        train_attrs=train_attrs_full, test_attrs=test_attrs_full,
        attr_names=ADULT_14_COLUMNS,
    )
    log.info(f"  sequential audits done in {time.time() - t0:.0f}s")

    # ── Sanity / 4-original-attribute snapshot ───────────────────────────
    # The original composition_experiment audited 4 attrs: race, sex,
    # age (here: age quartile bin since we don't store age_group), and
    # marital-status. Print them as a side-by-side sanity check.
    SANITY_ATTRS = ["race", "sex", "age", "marital-status"]
    print("\n--- 4-attribute sanity check (additive vs sequential) ---")
    print(f"{'attribute':<16} {'add Δ':>10} {'add R²':>10} {'add pass':>10} "
          f"{'seq Δ':>10} {'seq R²':>10} {'seq pass':>10}")
    for a in SANITY_ATTRS:
        ra, rs = audits_add[a], audits_seq[a]
        print(f"{a:<16} {ra.delta:>+9.1%} {ra.r_squared:>10.4f} {str(ra.adj_pass):>10} "
              f"{rs.delta:>+9.1%} {rs.r_squared:>10.4f} {str(rs.adj_pass):>10}")
    print()

    # ── Build 91-pair table ──────────────────────────────────────────────
    pair_rows: list[dict] = []
    for a_i, a_j in itertools.combinations(ADULT_14_COLUMNS, 2):
        for method, audits, allowed_acc in [
            ("additive", audits_add, income_acc_add),
            ("sequential", audits_seq, income_acc_seq),
        ]:
            r_i, r_j = audits[a_i], audits[a_j]
            pair_rows.append({
                "attribute_i": a_i,
                "attribute_j": a_j,
                "method": method,
                "delta": round(pair_delta(r_i, r_j), 6),
                "r_squared": round(pair_r2(r_i, r_j), 6),
                "adj_pass": pair_pass(r_i, r_j),
                "allowed_task_acc": round(allowed_acc, 6),
                "delta_i": round(r_i.delta, 6),
                "delta_j": round(r_j.delta, 6),
                "r2_i": round(r_i.r_squared, 6),
                "r2_j": round(r_j.r_squared, 6),
            })

    # ── Save 91-pair CSV ─────────────────────────────────────────────────
    out_dir = ROOT / "results" / "adult"
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_csv = out_dir / f"composition_91pair{args.suffix}.csv"
    fieldnames = list(pair_rows[0].keys())
    with open(pair_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)
    log.info(f"saved {pair_csv}  ({len(pair_rows)} rows = 91 pairs × 2 methods)")

    # ── Summary statistics ──────────────────────────────────────────────
    methods = ["additive", "sequential"]
    summary_rows: list[dict] = []
    pair_lookup: dict[tuple[str, str, str], dict] = {
        (r["attribute_i"], r["attribute_j"], r["method"]): r for r in pair_rows
    }

    for m in methods:
        rows_m = [r for r in pair_rows if r["method"] == m]
        n = len(rows_m)
        passes = sum(1 for r in rows_m if r["adj_pass"])
        mean_delta = float(np.mean([r["delta"] for r in rows_m]))
        mean_r2 = float(np.mean([r["r_squared"] for r in rows_m]))
        summary_rows.append({
            "method": m,
            "total_pairs": n,
            "pass_count": passes,
            "mean_delta": round(mean_delta, 6),
            "mean_r_squared": round(mean_r2, 6),
        })

    # Sequential strictly better: pair-by-pair, sequential_delta < additive_delta
    # (or sequential passes where additive does not).
    strict_better = 0
    for a_i, a_j in itertools.combinations(ADULT_14_COLUMNS, 2):
        add = pair_lookup[(a_i, a_j, "additive")]
        seq = pair_lookup[(a_i, a_j, "sequential")]
        if (seq["delta"] < add["delta"] - 1e-6) or (seq["adj_pass"] and not add["adj_pass"]):
            strict_better += 1

    for r in summary_rows:
        r["sequential_strict_better"] = strict_better if r["method"] == "sequential" else ""

    summary_csv = out_dir / f"composition_summary{args.suffix}.csv"
    fieldnames_s = ["method", "total_pairs", "pass_count", "mean_delta",
                    "mean_r_squared", "sequential_strict_better"]
    with open(summary_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames_s)
        writer.writeheader()
        writer.writerows(summary_rows)
    log.info(f"saved {summary_csv}")

    # ── Stdout report ───────────────────────────────────────────────────
    add_row = next(r for r in summary_rows if r["method"] == "additive")
    seq_row = next(r for r in summary_rows if r["method"] == "sequential")

    print("\n=== 91-Pair Composition Benchmark ===")
    print(f"Additive composition:   {add_row['pass_count']}/{add_row['total_pairs']} pass, "
          f"mean delta {add_row['mean_delta']:+.1%}, mean R² {add_row['mean_r_squared']:.4f}")
    print(f"Sequential composition: {seq_row['pass_count']}/{seq_row['total_pairs']} pass, "
          f"mean delta {seq_row['mean_delta']:+.1%}, mean R² {seq_row['mean_r_squared']:.4f}")
    print(f"Sequential strictly better on {strict_better}/{seq_row['total_pairs']} pairs.")
    print()


if __name__ == "__main__":
    main()
