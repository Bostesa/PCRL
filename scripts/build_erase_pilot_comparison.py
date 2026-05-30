"""Aggregate erase-layer pilot vs Round 5/7 baseline into a paper-ready
LaTeX comparison table + side-by-side JSON.

Reads:
    results/v2_adult_ROUND5/{summary,per_seed_results,dominant_axis_audit}.json
    results/v2_hmda_ROUND5/{...}
    results/v2_diabetes_ROUND7/{...}
    results/v2_adult_ERASE_PILOT/{...}
    results/v2_hmda_ERASE_PILOT/{...}
    results/v2_diabetes_ERASE_PILOT/{...}

Writes:
    results/rebuttal/erase_layer_pilot/comparison_table.tex
    results/rebuttal/erase_layer_pilot/comparison.json
    results/rebuttal/erase_layer_pilot/HEADLINE.txt

The "cleanly compliant" cell count is the cleanest signal for the paper's
49/56 vs 7/60 framing: a cell counts if linear R² ≤ 0.05 AND per_dim_std
mean ≥ 0.5 AND effective_rank ≥ 2.0. We compute this directly from the
per_seed_results.json + per_purpose_health blocks rather than relying on
the baseline's summary.json (which doesn't expose the cleanly-compliant
breakdown in a single field).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "rebuttal" / "erase_layer_pilot"

# Health-check thresholds — these are the published cleanly-compliant
# criteria from V2TrainerConfig.thresholds (per_dim_std_min=0.5,
# effective_rank_min=2.0).
PER_DIM_STD_MIN = 0.5
EFF_RANK_MIN = 2.0
R2_THRESHOLD = 0.05


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact: {path}")
    with open(path) as fh:
        return json.load(fh)


def _cleanly_compliant_count(per_seed_results: list[dict]) -> tuple[int, int]:
    """Return (clean_count, total_cells) across all seeds.

    A cell (purpose, attr, seed) is cleanly compliant iff:
      1. linear_r2 <= 0.05 (the paper's strict-pass criterion)
      2. per_dim_std_mean(purpose) >= 0.5
      3. effective_rank(purpose) >= 2.0

    The purpose-level health applies to ALL attributes of that purpose:
    if a purpose collapses, every (purpose, attr) cell under it is unclean.
    """
    clean = 0
    total = 0
    for seed in per_seed_results:
        health = seed.get("per_purpose_health", {})
        for cell in seed["attribute_results"]:
            total += 1
            r2 = cell["linear_r2"]
            purpose = cell["purpose"]
            p_health = health.get(purpose, {})
            std_mean = p_health.get("per_dim_std_mean", 0.0)
            eff_rank = p_health.get("effective_rank", 0.0)
            if (r2 <= R2_THRESHOLD
                    and std_mean >= PER_DIM_STD_MIN
                    and eff_rank >= EFF_RANK_MIN):
                clean += 1
    return clean, total


def _strict_pass_count(per_seed_results: list[dict]) -> tuple[int, int]:
    """Strict R² pass: count of cells with linear_r2 ≤ 0.05 across all seeds."""
    strict = 0
    total = 0
    for seed in per_seed_results:
        for cell in seed["attribute_results"]:
            total += 1
            if cell["linear_r2"] <= R2_THRESHOLD:
                strict += 1
    return strict, total


def _per_purpose_health_summary(per_seed_results: list[dict]) -> dict[str, float]:
    """Aggregate per_dim_std and eff_rank across all (purpose, seed) pairs."""
    std_vals: list[float] = []
    rank_vals: list[float] = []
    for seed in per_seed_results:
        for purpose, h in seed.get("per_purpose_health", {}).items():
            std_vals.append(h.get("per_dim_std_mean", 0.0))
            rank_vals.append(h.get("effective_rank", 0.0))
    if not std_vals:
        return {"per_dim_std_mean": float("nan"), "eff_rank_mean": float("nan")}
    return {
        "per_dim_std_mean": sum(std_vals) / len(std_vals),
        "eff_rank_mean": sum(rank_vals) / len(rank_vals),
    }


def _mean_r2(per_seed_results: list[dict]) -> float:
    rs = [c["linear_r2"] for s in per_seed_results for c in s["attribute_results"]]
    return sum(rs) / len(rs) if rs else float("nan")


def _task_acc(summary: dict) -> str:
    """Format task acc mean across tasks as 'task1: x%, task2: y%, ...'"""
    acc = summary.get("task_acc_mean", {})
    return ", ".join(f"{k}: {v * 100:.1f}\\%" for k, v in acc.items())


def _row(name: str, dataset: str, base_tag: str, pilot_tag: str) -> dict[str, Any]:
    base_dir = REPO_ROOT / "results" / f"v2_{dataset}_{base_tag}"
    pilot_dir = REPO_ROOT / "results" / f"v2_{dataset}_{pilot_tag}"

    base_sum = _load(base_dir / "summary.json")
    base_per = _load(base_dir / "per_seed_results.json")
    pilot_sum = _load(pilot_dir / "summary.json")
    pilot_per = _load(pilot_dir / "per_seed_results.json")

    base_per_seed = base_per["per_seed"] if "per_seed" in base_per else base_per
    pilot_per_seed = pilot_per["per_seed"] if "per_seed" in pilot_per else pilot_per

    base_strict, base_total = _strict_pass_count(base_per_seed)
    pilot_strict, pilot_total = _strict_pass_count(pilot_per_seed)
    base_clean, _ = _cleanly_compliant_count(base_per_seed)
    pilot_clean, _ = _cleanly_compliant_count(pilot_per_seed)
    base_health = _per_purpose_health_summary(base_per_seed)
    pilot_health = _per_purpose_health_summary(pilot_per_seed)

    return {
        "dataset": dataset,
        "n_cells": base_total,
        "base_tag": base_tag,
        "pilot_tag": pilot_tag,
        "base": {
            "strict_pass": f"{base_strict}/{base_total}",
            "cleanly_compliant": f"{base_clean}/{base_total}",
            "mean_r2": _mean_r2(base_per_seed),
            "per_dim_std_mean": base_health["per_dim_std_mean"],
            "eff_rank_mean": base_health["eff_rank_mean"],
            "task_acc": dict(base_sum.get("task_acc_mean", {})),
        },
        "pilot": {
            "strict_pass": f"{pilot_strict}/{pilot_total}",
            "cleanly_compliant": f"{pilot_clean}/{pilot_total}",
            "mean_r2": _mean_r2(pilot_per_seed),
            "per_dim_std_mean": pilot_health["per_dim_std_mean"],
            "eff_rank_mean": pilot_health["eff_rank_mean"],
            "task_acc": dict(pilot_sum.get("task_acc_mean", {})),
        },
        "deltas": {
            "strict_pass_delta": pilot_strict - base_strict,
            "cleanly_compliant_delta": pilot_clean - base_clean,
            "mean_r2_delta": _mean_r2(pilot_per_seed) - _mean_r2(base_per_seed),
            "per_dim_std_delta": pilot_health["per_dim_std_mean"] - base_health["per_dim_std_mean"],
            "eff_rank_delta": pilot_health["eff_rank_mean"] - base_health["eff_rank_mean"],
        },
    }


def _render_tex(rows: list[dict[str, Any]]) -> str:
    """Side-by-side comparison table."""
    lines: list[str] = []
    lines.append(r"% Auto-generated by scripts/build_erase_pilot_comparison.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Erase-layer pilot vs.\ Round 5/7 baseline. " +
                 r"The pilot inserts a frozen joint-LEACE projection between " +
                 r"the backbone and a LoRA-trainable task projection " +
                 r"(\S5.5 vision architecture, ported to tabular). " +
                 r"\textit{Strict pass}: linear $R^2(h_p, A) \leq 0.05$. " +
                 r"\textit{Cleanly compliant}: strict pass + per-dim std $\geq 0.5$ " +
                 r"+ effective rank $\geq 2.0$.}")
    lines.append(r"\label{tab:erase_pilot_comparison}")
    lines.append(r"\begin{tabular}{l l r r r r}")
    lines.append(r"\toprule")
    lines.append(r" & & Strict pass & Cleanly comp. & per-dim std & eff.\ rank \\")
    lines.append(r"\midrule")
    grand_strict_base = 0
    grand_strict_pilot = 0
    grand_clean_base = 0
    grand_clean_pilot = 0
    grand_total = 0
    for r in rows:
        ds = r["dataset"].title()
        lines.append(rf"\multirow{{2}}{{*}}{{{ds}}} & Baseline ({r['base_tag']}) & "
                     rf"{r['base']['strict_pass']} & {r['base']['cleanly_compliant']} & "
                     rf"{r['base']['per_dim_std_mean']:.3f} & {r['base']['eff_rank_mean']:.2f} \\")
        lines.append(rf" & Erase pilot & "
                     rf"{r['pilot']['strict_pass']} & {r['pilot']['cleanly_compliant']} & "
                     rf"{r['pilot']['per_dim_std_mean']:.3f} & {r['pilot']['eff_rank_mean']:.2f} \\")
        lines.append(r"\midrule")
        base_strict_n = int(r["base"]["strict_pass"].split("/")[0])
        pilot_strict_n = int(r["pilot"]["strict_pass"].split("/")[0])
        base_clean_n = int(r["base"]["cleanly_compliant"].split("/")[0])
        pilot_clean_n = int(r["pilot"]["cleanly_compliant"].split("/")[0])
        grand_strict_base += base_strict_n
        grand_strict_pilot += pilot_strict_n
        grand_clean_base += base_clean_n
        grand_clean_pilot += pilot_clean_n
        grand_total += r["n_cells"]
    lines.append(rf"\multirow{{2}}{{*}}{{\textbf{{Total}}}} & Baseline & "
                 rf"{grand_strict_base}/{grand_total} & {grand_clean_base}/{grand_total} & --- & --- \\")
    lines.append(rf" & Erase pilot & "
                 rf"{grand_strict_pilot}/{grand_total} & {grand_clean_pilot}/{grand_total} & --- & --- \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _render_headline(rows: list[dict[str, Any]]) -> str:
    parts = ["ERASE-LAYER PILOT vs ROUND 5/7 BASELINE", "=" * 60, ""]
    grand_strict_base = grand_strict_pilot = 0
    grand_clean_base = grand_clean_pilot = 0
    grand_total = 0
    for r in rows:
        parts.append(f"  {r['dataset']:10s} "
                     f"strict {r['base']['strict_pass']} -> {r['pilot']['strict_pass']}  "
                     f"clean {r['base']['cleanly_compliant']} -> {r['pilot']['cleanly_compliant']}  "
                     f"R² mean {r['base']['mean_r2']:.4f} -> {r['pilot']['mean_r2']:.4f}")
        grand_strict_base += int(r["base"]["strict_pass"].split("/")[0])
        grand_strict_pilot += int(r["pilot"]["strict_pass"].split("/")[0])
        grand_clean_base += int(r["base"]["cleanly_compliant"].split("/")[0])
        grand_clean_pilot += int(r["pilot"]["cleanly_compliant"].split("/")[0])
        grand_total += r["n_cells"]
    parts.append("")
    parts.append(f"  TOTAL      strict {grand_strict_base}/{grand_total} -> {grand_strict_pilot}/{grand_total}  "
                 f"clean {grand_clean_base}/{grand_total} -> {grand_clean_pilot}/{grand_total}")
    parts.append("")
    parts.append("Definitions:")
    parts.append("  Strict pass:        linear R²(h_p, A) <= 0.05")
    parts.append("  Cleanly compliant:  strict pass + per_dim_std_mean >= 0.5 + eff_rank >= 2.0")
    return "\n".join(parts) + "\n"


def main() -> None:
    rows = [
        _row("Adult", "adult", "ROUND5", "ERASE_PILOT"),
        _row("HMDA", "hmda", "ROUND5", "ERASE_PILOT"),
        _row("Diabetes", "diabetes", "ROUND7", "ERASE_PILOT"),
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tex = _render_tex(rows)
    (OUT_DIR / "comparison_table.tex").write_text(tex)

    headline = _render_headline(rows)
    (OUT_DIR / "HEADLINE.txt").write_text(headline)

    payload = {
        "thresholds": {
            "r2": R2_THRESHOLD,
            "per_dim_std_min": PER_DIM_STD_MIN,
            "eff_rank_min": EFF_RANK_MIN,
        },
        "rows": rows,
    }
    (OUT_DIR / "comparison.json").write_text(json.dumps(payload, indent=2))

    print(headline)
    print(f"\nwrote: {OUT_DIR / 'comparison_table.tex'}")
    print(f"wrote: {OUT_DIR / 'comparison.json'}")
    print(f"wrote: {OUT_DIR / 'HEADLINE.txt'}")


if __name__ == "__main__":
    main()
