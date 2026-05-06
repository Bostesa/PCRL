"""Enumerate the 49 collapse-compliant pair-seeds from existing summary.json
health_notes and write the target list consumed by the variance-constrained
retraining launcher.

A pair-seed is "collapse-compliant" iff:
    R²_onehot at τ=0.05 PASSES, but the (purpose, seed) it belongs to was
    flagged as COLLAPSED via per_dim_std<0.5 OR eff_rank<2.

Output:
    results/v2_pcrl_variance_constrained/target_cells.json
        {
            "generated_at": "...",
            "n_collapse_compliant": 49,
            "n_cleanly_compliant": 7,
            "n_r2_failed": 4,
            "n_total": 60,
            "retraining_units": [{"dataset": "...", "seed": 0}, ...],
            "per_pair_seed": [
                {"dataset": "adult", "purpose": "income_prediction",
                 "attribute": "race", "seed": 2,
                 "r2_onehot": 0.0145, "r2_pass": true,
                 "purpose_collapsed": true,
                 "status": "collapse-compliant"},
                ...
            ]
        }
"""
from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASETS = [
    ("adult", ROOT / "results" / "v2_adult_ROUND5"),
    ("hmda", ROOT / "results" / "v2_hmda_ROUND5"),
    ("diabetes", ROOT / "results" / "v2_diabetes_ROUND7"),
]

OUT_DIR = ROOT / "results" / "v2_pcrl_variance_constrained"
OUT_DIR.mkdir(parents=True, exist_ok=True)

R2_TAU = 0.05
HEALTH_PER_DIM_STD_MIN = 0.5
HEALTH_EFF_RANK_MIN = 2.0

NOTE_RX = re.compile(
    r"seed=(\d+)\s+purpose=(\S+)\s+(per_dim_std_mean|eff_rank)=([\d.]+)"
)


def load_collapsed_purposes(summary_path: Path) -> dict[tuple[int, str], list[str]]:
    """Parse summary.json health_notes into {(seed, purpose): [reason, ...]}."""
    if not summary_path.exists():
        return {}
    s = json.load(open(summary_path))
    notes = s.get("health_notes", [])
    out: dict[tuple[int, str], list[str]] = {}
    for line in notes:
        m = NOTE_RX.match(line)
        if not m:
            continue
        seed = int(m.group(1))
        purpose = m.group(2)
        metric = m.group(3)
        value = float(m.group(4))
        key = (seed, purpose)
        out.setdefault(key, []).append(f"{metric}={value:.3f}")
    return out


def load_pair_r2(da_path: Path) -> list[dict]:
    """Pull (purpose, attribute, seed, r2_onehot) rows from dominant_axis_audit.json."""
    d = json.load(open(da_path))
    rows: list[dict] = []
    for seed_str, sd in d["per_seed"].items():
        seed = int(seed_str)
        for r in sd["rows"]:
            rows.append({
                "purpose": r["purpose"],
                "attribute": r["attribute"],
                "seed": seed,
                "r2_onehot": float(r["r2_onehot"]),
            })
    return rows


def main() -> None:
    per_pair_seed: list[dict] = []
    retraining_units_set: set[tuple[str, int]] = set()
    counts = {
        "cleanly_compliant": 0,
        "collapse_compliant": 0,
        "r2_failed": 0,
        "total": 0,
    }

    for dataset, results_dir in DATASETS:
        summary_path = results_dir / "summary.json"
        da_path = results_dir / "dominant_axis_audit.json"
        if not da_path.exists():
            print(f"[WARN] no dominant_axis_audit.json at {da_path}; skipping {dataset}",
                  file=sys.stderr)
            continue
        collapsed = load_collapsed_purposes(summary_path)
        rows = load_pair_r2(da_path)
        for r in rows:
            seed = r["seed"]
            purpose = r["purpose"]
            r2 = r["r2_onehot"]
            r2_pass = r2 <= R2_TAU
            purpose_collapsed = (seed, purpose) in collapsed
            if not r2_pass:
                status = "r2-failed"
                counts["r2_failed"] += 1
            elif purpose_collapsed:
                status = "collapse-compliant"
                counts["collapse_compliant"] += 1
                retraining_units_set.add((dataset, seed))
            else:
                status = "cleanly-compliant"
                counts["cleanly_compliant"] += 1
            counts["total"] += 1
            per_pair_seed.append({
                "dataset": dataset,
                "purpose": purpose,
                "attribute": r["attribute"],
                "seed": seed,
                "r2_onehot": r2,
                "r2_pass": r2_pass,
                "purpose_collapsed": purpose_collapsed,
                "collapse_reasons": collapsed.get((seed, purpose), []),
                "status": status,
            })

    retraining_units = sorted(
        [{"dataset": d, "seed": s} for d, s in retraining_units_set],
        key=lambda x: (x["dataset"], x["seed"]),
    )

    out = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "n_total": counts["total"],
        "n_cleanly_compliant": counts["cleanly_compliant"],
        "n_collapse_compliant": counts["collapse_compliant"],
        "n_r2_failed": counts["r2_failed"],
        "thresholds": {
            "r2_tau": R2_TAU,
            "per_dim_std_min": HEALTH_PER_DIM_STD_MIN,
            "eff_rank_min": HEALTH_EFF_RANK_MIN,
        },
        "retraining_units": retraining_units,
        "per_pair_seed": per_pair_seed,
    }
    out_path = OUT_DIR / "target_cells.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    print(f"  total pair-seeds: {counts['total']}")
    print(f"  cleanly compliant: {counts['cleanly_compliant']}")
    print(f"  collapse-compliant (retraining targets): {counts['collapse_compliant']}")
    print(f"  R² failed (not retraining; user spec only retrains R²-passing): "
          f"{counts['r2_failed']}")
    print(f"  retraining units (dataset, seed): {len(retraining_units)}")
    for u in retraining_units:
        print(f"    {u['dataset']} s{u['seed']}")


if __name__ == "__main__":
    main()
