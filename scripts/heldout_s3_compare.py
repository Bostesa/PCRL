"""Compare Adult seed-3 held-out validation to seed-2 headline.

Inputs:
  results/v2_adult_HELDOUT_S3/per_seed_results.json   (held-out fold)
  results/v2_adult_HELDOUT_S3/dominant_axis_audit.json (held-out, after audit)
  results/v2_adult_ROUND5/per_seed_results.json        (headline)
  results/v2_adult_ROUND5/dominant_axis_audit.json     (headline)

Outputs:
  results/reviewer_dropins/heldout_s3_compare.json
  results/reviewer_dropins/PAPER_PASTE_heldout.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
TAU = 0.05
PER_DIM_MIN = 0.5
EFF_RANK_MIN = 2.0
HELDOUT_DIR = ROOT / "results" / "v2_adult_HELDOUT_S3"
HEADLINE_DIR = ROOT / "results" / "v2_adult_ROUND5"
OUT_DIR = ROOT / "results" / "reviewer_dropins"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dominant_audit(path: Path, seed: int) -> dict:
    with open(path) as f:
        d = json.load(f)
    rows = d["per_seed"][str(seed)]["rows"]
    return {(r["purpose"], r["attribute"]): r for r in rows}


def load_per_seed_health(path: Path, seed: int) -> dict:
    with open(path) as f:
        d = json.load(f)
    s = next(x for x in d["per_seed"] if x["seed"] == seed)
    return s["per_purpose_health"]


def cell_status(r2_onehot: float, health: dict) -> tuple[bool, bool]:
    """Return (r2_pass, cleanly_compliant)."""
    r2_pass = r2_onehot < TAU
    health_pass = (
        health["per_dim_std_mean"] >= PER_DIM_MIN
        and health["effective_rank"] >= EFF_RANK_MIN
    )
    return r2_pass, (r2_pass and health_pass)


def main() -> None:
    held_audit = load_dominant_audit(HELDOUT_DIR / "dominant_axis_audit.json", 3)
    held_health = load_per_seed_health(HELDOUT_DIR / "per_seed_results.json", 3)
    head_audit = load_dominant_audit(HEADLINE_DIR / "dominant_axis_audit.json", 2)
    head_health = load_per_seed_health(HEADLINE_DIR / "per_seed_results.json", 2)

    pairs = sorted(held_audit.keys())
    rows = []
    held_r2_pass = held_clean = head_r2_pass = head_clean = 0
    deltas = []
    for purpose, attr in pairs:
        h_r2 = float(held_audit[(purpose, attr)]["r2_onehot"])
        h_da = float(held_audit[(purpose, attr)]["r2_da"])
        x_r2 = float(head_audit[(purpose, attr)]["r2_onehot"])
        x_da = float(head_audit[(purpose, attr)]["r2_da"])
        h_health = held_health[purpose]
        x_health = head_health[purpose]
        h_pass, h_clean = cell_status(h_r2, h_health)
        x_pass, x_clean = cell_status(x_r2, x_health)
        held_r2_pass += int(h_pass)
        held_clean += int(h_clean)
        head_r2_pass += int(x_pass)
        head_clean += int(x_clean)
        deltas.append(h_r2 - x_r2)
        rows.append(
            {
                "purpose": purpose, "attribute": attr,
                "held_r2_onehot": h_r2, "head_r2_onehot": x_r2, "delta_r2_onehot": h_r2 - x_r2,
                "held_r2_da": h_da, "head_r2_da": x_da,
                "held_per_dim_std_mean": h_health["per_dim_std_mean"],
                "head_per_dim_std_mean": x_health["per_dim_std_mean"],
                "held_effective_rank": h_health["effective_rank"],
                "head_effective_rank": x_health["effective_rank"],
                "held_r2_pass": h_pass, "held_cleanly_compliant": h_clean,
                "head_r2_pass": x_pass, "head_cleanly_compliant": x_clean,
            }
        )

    n = len(pairs)
    delta_mean = mean(deltas)
    delta_std = stdev(deltas) if len(deltas) > 1 else 0.0
    diff = held_r2_pass - head_r2_pass
    if abs(diff) <= 1:
        verdict = "STRONG"
        interp = ("the schedule generalises to a fold that was untouched during "
                  "development, defusing the post-hoc-tuning concern")
    elif -3 <= diff < -1:
        verdict = "ACCEPTABLE"
        interp = ("the schedule partially generalises, with a measurable but small "
                  "drop on the held-out fold")
    elif diff <= -4:
        verdict = "WEAK"
        interp = ("the schedule does not generalise cleanly to a fold untouched "
                  "during development, and the post-hoc-tuning concern is "
                  "confirmed at scale")
    else:  # held > head + 1, treat as STRONG
        verdict = "STRONG"
        interp = ("the schedule generalises and in fact slightly improves on the "
                  "headline fold")

    payload = {
        "tau": TAU,
        "per_dim_std_min": PER_DIM_MIN,
        "effective_rank_min": EFF_RANK_MIN,
        "per_cell": rows,
        "held_r2_pass_count": held_r2_pass,
        "head_r2_pass_count": head_r2_pass,
        "held_cleanly_compliant_count": held_clean,
        "head_cleanly_compliant_count": head_clean,
        "n_cells": n,
        "delta_r2_onehot_mean": delta_mean,
        "delta_r2_onehot_std": delta_std,
        "delta_r2_onehot_min": min(deltas),
        "delta_r2_onehot_max": max(deltas),
        "verdict": verdict,
    }
    (OUT_DIR / "heldout_s3_compare.json").write_text(json.dumps(payload, indent=2))

    paragraph = (
        "To partially address the post-hoc-tuning concern, we ran a held-out "
        "validation on Adult seed~3, which was not used during development of "
        "the $\\lambda_{\\min}$ floor or the warmup-skip rule (the R1+R2 probe "
        "touched seeds 0, 1, and 2). Retraining all 8 Adult pairs on seed 3 "
        "from a fresh LEACE warm-start under the Round-5 schedule, the "
        f"held-out $R^2$-pass rate is ${held_r2_pass}/{n}$ versus "
        f"${head_r2_pass}/{n}$ on the headline Adult seed-2 run, and the "
        f"cleanly-compliant rate is ${held_clean}/{n}$ versus ${head_clean}/{n}$. "
        "Mean per-cell $\\Delta R^2_{\\mathrm{onehot}}$ between held-out and "
        f"headline is $\\bar\\Delta = {delta_mean:+.4f}$ "
        f"(std $\\sigma_\\Delta = {delta_std:.4f}$, range "
        f"$[{min(deltas):+.4f}, {max(deltas):+.4f}]$). " + interp.capitalize() + ". "
        "This is a single-fold partial validation on one of three datasets, "
        "not a full $k$-fold cross-validation; we recommend the latter as the "
        "rigorous follow-up."
    )
    paper = (
        "## Paper paste — §6 held-out validation of the dual-update schedule\n\n"
        f"\\paragraph{{Held-out validation of the dual-update schedule.}}\n"
        f"{paragraph}\n"
    )
    (OUT_DIR / "PAPER_PASTE_heldout.md").write_text(paper)

    print("\nPER-CELL TABLE")
    print(f"  {'purpose':<25} {'attr':<14}  {'held r2':>8} {'head r2':>8} {'delta':>8}  "
          f"{'held pass':>9} {'head pass':>9} {'held clean':>10}")
    for r in rows:
        print(
            f"  {r['purpose']:<25} {r['attribute']:<14}  "
            f"{r['held_r2_onehot']:>8.4f} {r['head_r2_onehot']:>8.4f} "
            f"{r['delta_r2_onehot']:>+8.4f}  "
            f"{str(r['held_r2_pass']):>9} {str(r['head_r2_pass']):>9} "
            f"{str(r['held_cleanly_compliant']):>10}"
        )
    print(f"\nHELD-OUT R²-pass: {held_r2_pass}/{n}  cleanly-compliant: {held_clean}/{n}")
    print(f"HEADLINE R²-pass: {head_r2_pass}/{n}  cleanly-compliant: {head_clean}/{n}")
    print(f"Mean ΔR²_onehot = {delta_mean:+.4f}  std = {delta_std:.4f}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
