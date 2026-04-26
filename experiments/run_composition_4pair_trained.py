#!/usr/bin/env python3
"""4-pair composition benchmark on the trained-disallowed attribute set.

Composes ONLY the two purposes that share the union {race, sex,
age_group, marital_status} as their disallowed attributes:

    p1 income_prediction:    {race, sex}
    p2 employment_analysis:  {race, age_group, marital_status}

This is the apples-to-apples version of the 91-pair sweep — every
attribute audited here was a training-time suppression target for at
least one of the two composed purposes, so any failure to suppress is
a real composition-rule problem rather than an out-of-distribution
generalisation gap.

Outputs:
    results/adult/composition_4pair_trained.csv
        attribute, method, delta, r_squared, adj_pass, allowed_task_acc
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
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

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("composition_4pair_trained")
log.setLevel(logging.INFO)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
warnings.filterwarnings("ignore", message=".*pin_memory.*")

# Union of disallowed attributes for p1 (income_prediction) ∧ p2 (employment_analysis).
UNION_ATTRS = ["race", "sex", "age_group", "marital_status"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint .pt; default checkpoints/adult/best.pt")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended before .csv (e.g. _FIXED)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")

    purposes = get_adult_purposes()
    train_ds = AdultDataset(purposes=purposes, root="data", split="train", download=False)
    test_ds = AdultDataset(
        purposes=purposes, root="data", split="test", download=False,
        norm_stats=train_ds.norm_stats,
    )
    log.info(f"  N_train={len(train_ds)}, N_test={len(test_ds)}, D={train_ds.info.num_features}")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False,
                              collate_fn=collate_pcrl_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             collate_fn=collate_pcrl_batch, num_workers=0)

    # Pull the four sensitive labels straight from the dataset (these are
    # the integer-coded attrs the model was trained to suppress).
    train_attrs = {a: train_ds.sensitive_attrs[a].numpy() for a in UNION_ATTRS}
    test_attrs = {a: test_ds.sensitive_attrs[a].numpy() for a in UNION_ATTRS}

    ckpt_path = Path(args.ckpt) if args.ckpt else ROOT / "checkpoints" / "adult" / "best.pt"
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    encoder = PurposeConditionedEncoder(
        input_dim=train_ds.info.num_features,
        hidden_dims=[128, 128], repr_dim=64,
        num_purposes=len(purposes), purpose_emb_dim=32,
        conditioning="film", dropout=0.3,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)["encoder"]
    encoder.load_state_dict(state)
    encoder.to(device).eval()
    log.info(f"loaded {ckpt_path}")

    # Compose ONLY p1 + p2 (NOT p3) — the union of {race,sex} and
    # {race, age_group, marital_status} is exactly the 4 audit attrs.
    with torch.no_grad():
        e1 = encoder.get_purpose_embedding(0).detach().clone()  # income_prediction
        e2 = encoder.get_purpose_embedding(1).detach().clone()  # employment_analysis
    embs = [e1, e2]
    log.info("composing p1 (income_prediction) and p2 (employment_analysis)")

    # ── Extract composed reps ──────────────────────────────────────────
    log.info("extracting additive-composed reps...")
    t0 = time.time()
    train_add = extract_composed_reprs_additive(encoder, train_loader, embs, device)
    test_add = extract_composed_reprs_additive(encoder, test_loader, embs, device)
    log.info(f"  done in {time.time() - t0:.1f}s  shape={train_add.shape}")

    log.info("extracting sequential-composed reps...")
    t0 = time.time()
    train_seq = extract_composed_reprs_sequential(encoder, train_loader, embs, device)
    test_seq = extract_composed_reprs_sequential(encoder, test_loader, embs, device)
    log.info(f"  done in {time.time() - t0:.1f}s  shape={train_seq.shape}")

    # ── Allowed-task accuracy via fresh-LR probe (income head was
    #    collapsed in this checkpoint; probe gives reliable rep quality).
    from sklearn.linear_model import LogisticRegression
    inc_train = train_ds.task_labels["income"].numpy()
    inc_test = test_ds.task_labels["income"].numpy()
    inc_acc_add = float(LogisticRegression(max_iter=2000, random_state=42)
                        .fit(train_add, inc_train).score(test_add, inc_test))
    inc_acc_seq = float(LogisticRegression(max_iter=2000, random_state=42)
                        .fit(train_seq, inc_train).score(test_seq, inc_test))
    log.info(f"income fresh-LR probe -- additive {inc_acc_add:.4f}, sequential {inc_acc_seq:.4f}")

    # ── Per-attr audits ───────────────────────────────────────────────
    log.info("auditing 4 attrs on additive reps...")
    t0 = time.time()
    audits_add = audit_all_attrs(
        train_reprs=train_add, test_reprs=test_add,
        train_attrs=train_attrs, test_attrs=test_attrs,
        attr_names=UNION_ATTRS,
    )
    log.info(f"  done in {time.time() - t0:.0f}s")

    log.info("auditing 4 attrs on sequential reps...")
    t0 = time.time()
    audits_seq = audit_all_attrs(
        train_reprs=train_seq, test_reprs=test_seq,
        train_attrs=train_attrs, test_attrs=test_attrs,
        attr_names=UNION_ATTRS,
    )
    log.info(f"  done in {time.time() - t0:.0f}s")

    # ── CSV out ────────────────────────────────────────────────────────
    rows: list[dict] = []
    for a in UNION_ATTRS:
        for method, audits, acc in [
            ("additive", audits_add, inc_acc_add),
            ("sequential", audits_seq, inc_acc_seq),
        ]:
            r = audits[a]
            rows.append({
                "attribute": a,
                "method": method,
                "delta": round(r.delta, 6),
                "r_squared": round(r.r_squared, 6),
                "adj_pass": r.adj_pass,
                "allowed_task_acc": round(acc, 6),
            })

    out_csv = ROOT / "results" / "adult" / f"composition_4pair_trained{args.suffix}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info(f"saved {out_csv}")

    # ── Stdout summary ────────────────────────────────────────────────
    delta_sym = "delta"
    print("\n=== 4-Pair Composition (Trained Disallowed Attributes) ===")
    print(f"{'Attribute':<16} {'Add '+delta_sym:>10} {'Seq '+delta_sym:>10} {'Add R^2':>10} {'Seq R^2':>10}  Seq better?")
    pass_add = 0
    pass_seq = 0
    for a in UNION_ATTRS:
        ra, rs = audits_add[a], audits_seq[a]
        better = "yes" if (rs.delta < ra.delta - 1e-6 or (rs.adj_pass and not ra.adj_pass)) else "no"
        print(f"{a:<16} {ra.delta:>+9.1%} {rs.delta:>+9.1%} {ra.r_squared:>10.4f} {rs.r_squared:>10.4f}  {better}")
        pass_add += int(ra.adj_pass)
        pass_seq += int(rs.adj_pass)
    print(f"\nPass count: additive {pass_add}/4, sequential {pass_seq}/4")
    print(f"Income (fresh-LR probe): additive {inc_acc_add:.1%}, sequential {inc_acc_seq:.1%}")
    print(f"\nSaved: {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
