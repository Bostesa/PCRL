#!/usr/bin/env python3
"""Aggregate the LAFTR hard-R² pilot into a camera-ready rebuttal artifact.

Produces a 3-row × per-dataset comparison:
  1. PCRL (Round 5/7, paper headline)         — frozen, from paper_baseline_numbers.json
  2. LAFTR (Appendix Q, lambda_adv=1.0)        — frozen, from paper_baseline_numbers.json
  3. LAFTR-hard-R² (this pilot)                — computed live from
                                                 results/laftr_hard_r2_<ds>_LAFTR_HARD_R2/

Three columns per method per dataset (+ aggregated):
  * strict-pass rate         — fraction of (purpose, attr, seed) cells with linear R² < 0.05
  * mean R²_onehot on passing — mean linear R² over only the passing cells (depth-of-compliance);
                                shown as "—" when no cells pass
  * task accuracy            — mean task acc across (purpose, seed)

Writes:
    results/laftr_hard_r2/comparison_table.tex   (paper-ready LaTeX)
    results/laftr_hard_r2/comparison.json        (machine-readable)
    results/laftr_hard_r2/HEADLINE.txt           (one-line summary)

The aggregator reads PCRL and LAFTR-Q rows from a frozen JSON
(``scripts/paper_baseline_numbers.json``) so reviewer-facing numbers
are immutable across re-runs. The LAFTR-hard-R² row is computed live
from the AWS pilot's per_seed_results.json so the artifact updates
the moment new results land.

Usage:
    python scripts/aggregate_laftr_hard_r2.py
    python scripts/aggregate_laftr_hard_r2.py --root /custom/path
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["adult", "hmda", "diabetes"]
TAU = 0.05


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────


def load_frozen_baselines(root: Path) -> dict[str, Any]:
    """Load the frozen PCRL + LAFTR-Q baseline numbers."""
    p = root / "scripts" / "paper_baseline_numbers.json"
    return json.loads(p.read_text())


def load_laftr_hard_r2_per_seed(root: Path, dataset: str) -> dict | None:
    """Load this pilot's per_seed_results.json for one dataset, or None if absent."""
    p = root / "results" / f"laftr_hard_r2_{dataset}_LAFTR_HARD_R2" / "per_seed_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────


def compute_laftr_hard_r2_metrics(per_seed_file: dict) -> dict[str, Any]:
    """Aggregate the live LAFTR-hard-R² per_seed_results.json into the 3 columns.

    Strict-pass: count of (purpose, attr, seed) cells with linear_r2 < TAU,
    over total cells.

    Mean R² on passing: mean linear_r2 over only the passing cells. None if
    no cells pass.

    Task acc: arithmetic mean of every task_accuracy value across all seeds
    (each task in each seed contributes one value; weighted equally).
    """
    seeds = per_seed_file["per_seed"]
    n_cells_per_seed = seeds[0]["total_pairs"]
    n_seeds = len(seeds)
    n_total = sum(len(s["attribute_results"]) for s in seeds)

    passing = [
        c for s in seeds for c in s["attribute_results"]
        if c["linear_r2"] < TAU
    ]
    strict_pass_rate = len(passing) / n_total if n_total > 0 else float("nan")
    if passing:
        mean_r2_on_passing = sum(c["linear_r2"] for c in passing) / len(passing)
    else:
        mean_r2_on_passing = None

    task_accs = [
        v for s in seeds for v in s["task_accuracies"].values()
    ]
    task_acc = sum(task_accs) / len(task_accs) if task_accs else float("nan")

    return {
        "strict_pass_rate": strict_pass_rate,
        "mean_r2_on_passing": mean_r2_on_passing,
        "task_acc": task_acc,
        "n_cells_per_seed": n_cells_per_seed,
        "n_seeds": n_seeds,
        "n_passing": len(passing),
    }


def aggregate_across_datasets(
    per_dataset_blocks: dict[str, dict | None],
) -> dict[str, Any]:
    """Compute the "Aggregated (N cells)" column.

    Weighting: each cell contributes equally to strict-pass and mean-R²-on-passing.
    Task accuracy is averaged across datasets weighted by ``n_cells_per_seed *
    n_seeds`` (= total cells), which is equivalent to weighting equally per cell
    if each (purpose, seed) contributes one task acc per cell-block.
    """
    total_cells = 0
    total_passing = 0
    sum_r2_passing = 0.0
    n_summed_r2 = 0
    weighted_acc_sum = 0.0
    weighted_acc_n = 0

    for ds in DATASETS:
        block = per_dataset_blocks.get(ds)
        if block is None:
            continue
        n_cells = block["n_cells_per_seed"] * block["n_seeds"]
        total_cells += n_cells
        # n_passing handles the LAFTR-Q all-zero case
        n_pass = block.get("n_passing")
        if n_pass is None and block.get("strict_pass_rate") is not None:
            n_pass = int(round(block["strict_pass_rate"] * n_cells))
        n_pass = n_pass or 0
        total_passing += n_pass
        if block.get("mean_r2_on_passing") is not None and n_pass > 0:
            sum_r2_passing += block["mean_r2_on_passing"] * n_pass
            n_summed_r2 += n_pass
        if block.get("task_acc") is not None and not _is_nan(block["task_acc"]):
            weighted_acc_sum += block["task_acc"] * n_cells
            weighted_acc_n += n_cells

    if total_cells == 0:
        return {
            "strict_pass_rate": float("nan"),
            "mean_r2_on_passing": None,
            "task_acc": float("nan"),
            "n_cells_total": 0,
            "n_passing": 0,
        }
    return {
        "strict_pass_rate": total_passing / total_cells,
        "mean_r2_on_passing": (sum_r2_passing / n_summed_r2) if n_summed_r2 > 0 else None,
        "task_acc": weighted_acc_sum / weighted_acc_n if weighted_acc_n > 0 else float("nan"),
        "n_cells_total": total_cells,
        "n_passing": total_passing,
    }


def _is_nan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────────


def _fmt_pass_rate(b: dict | None) -> str:
    if b is None:
        return "—"
    rate = b.get("strict_pass_rate")
    if rate is None or _is_nan(rate):
        return "—"
    n_pass = b.get("n_passing")
    n_cells = b.get("n_cells_per_seed", 0) * b.get("n_seeds", 0) or b.get("n_cells_total", 0)
    if n_pass is not None and n_cells > 0:
        return f"{rate * 100:.1f}\\% ({n_pass}/{n_cells})"
    return f"{rate * 100:.1f}\\%"


def _fmt_pass_rate_plain(b: dict | None) -> str:
    if b is None:
        return "—"
    rate = b.get("strict_pass_rate")
    if rate is None or _is_nan(rate):
        return "—"
    n_pass = b.get("n_passing")
    n_cells = b.get("n_cells_per_seed", 0) * b.get("n_seeds", 0) or b.get("n_cells_total", 0)
    if n_pass is not None and n_cells > 0:
        return f"{rate * 100:.1f}% ({n_pass}/{n_cells})"
    return f"{rate * 100:.1f}%"


def _fmt_r2(b: dict | None) -> str:
    if b is None:
        return "—"
    v = b.get("mean_r2_on_passing")
    if v is None:
        return "—"
    return f"{v:.4f}"


def _fmt_acc(b: dict | None, latex: bool = False) -> str:
    if b is None:
        return "—"
    v = b.get("task_acc")
    if v is None or _is_nan(v):
        return "—"
    return f"{v * 100:.1f}\\%" if latex else f"{v * 100:.1f}%"


def render_tex(
    rows: list[tuple[str, str, dict[str, dict | None], dict]],
) -> str:
    """3-row LaTeX table. Each tuple: (method_label, method_short, per_dataset, aggregated)."""
    lines = []
    lines.append(r"% Auto-generated by scripts/aggregate_laftr_hard_r2.py — do not edit by hand.")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{LAFTR hard-R² rebuttal pilot. " +
                 r"Three methods on the same 60-cell grid (Adult 24 + HMDA 18 + Diabetes 18, " +
                 r"3 seeds each). " +
                 r"\textit{Strict pass}: fraction of (purpose, attribute, seed) cells with " +
                 r"linear $R^2(h_p, A) < 0.05$. " +
                 r"\textit{Mean $R^2$ on passing}: mean linear $R^2$ over only the passing cells " +
                 r"(depth-of-compliance; ``—'' when no cells pass). " +
                 r"\textit{Task acc}: arithmetic mean over (purpose, seed).}")
    lines.append(r"\label{tab:laftr_hard_r2}")
    lines.append(r"\begin{tabular}{l l c c c c}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{Method} & \multirow{2}{*}{Metric} & "
                 r"Adult & HMDA & Diabetes & Aggregated \\")
    lines.append(r" & & (24 cells) & (18 cells) & (18 cells) & (60 cells) \\")
    lines.append(r"\midrule")

    for method_label, method_short, per_ds, aggr in rows:
        # Strict-pass row
        lines.append(
            rf"\multirow{{3}}{{*}}{{{method_label}}} & "
            r"Strict pass ($R^2<0.05$) & "
            rf"{_fmt_pass_rate(per_ds.get('adult'))} & "
            rf"{_fmt_pass_rate(per_ds.get('hmda'))} & "
            rf"{_fmt_pass_rate(per_ds.get('diabetes'))} & "
            rf"{_fmt_pass_rate(aggr)} \\"
        )
        # Mean R^2 on passing
        lines.append(
            r" & Mean $R^2$ on passing & "
            rf"{_fmt_r2(per_ds.get('adult'))} & "
            rf"{_fmt_r2(per_ds.get('hmda'))} & "
            rf"{_fmt_r2(per_ds.get('diabetes'))} & "
            rf"{_fmt_r2(aggr)} \\"
        )
        # Task acc
        lines.append(
            r" & Task acc & "
            rf"{_fmt_acc(per_ds.get('adult'), latex=True)} & "
            rf"{_fmt_acc(per_ds.get('hmda'), latex=True)} & "
            rf"{_fmt_acc(per_ds.get('diabetes'), latex=True)} & "
            rf"{_fmt_acc(aggr, latex=True)} \\"
        )
        lines.append(r"\midrule")
    # Drop the trailing midrule, replace with bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def render_headline(
    rows: list[tuple[str, str, dict[str, dict | None], dict]],
) -> str:
    """One-line summary across all three methods."""
    parts: list[str] = []
    parts.append("LAFTR HARD-R² REBUTTAL PILOT — strict pass (R²<0.05) / depth / task acc")
    parts.append("=" * 78)
    parts.append("")
    parts.append(f"  {'Method':38s} {'Adult':18s} {'HMDA':18s} {'Diab':18s} {'Aggr':18s}")
    parts.append(f"  {'-' * 38} {'-' * 18} {'-' * 18} {'-' * 18} {'-' * 18}")
    for method_label, method_short, per_ds, aggr in rows:
        parts.append(
            f"  {method_short:38s} "
            f"{_fmt_pass_rate_plain(per_ds.get('adult')):18s} "
            f"{_fmt_pass_rate_plain(per_ds.get('hmda')):18s} "
            f"{_fmt_pass_rate_plain(per_ds.get('diabetes')):18s} "
            f"{_fmt_pass_rate_plain(aggr):18s}"
        )
    parts.append("")
    parts.append("Strict pass: linear R²(h_p, A) < 0.05; format \"rate (n_pass/n_cells)\"")
    parts.append("Depth-of-compliance + task acc are in comparison.json / comparison_table.tex")
    return "\n".join(parts) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def build_rows(
    root: Path,
) -> tuple[list[tuple[str, str, dict[str, dict | None], dict]], dict[str, Any]]:
    """Return (rows_for_renderers, payload_for_json)."""
    baselines = load_frozen_baselines(root)
    methods = baselines["methods"]

    rows: list[tuple[str, str, dict[str, dict | None], dict]] = []

    # Row 1: PCRL (frozen)
    pcrl_per = methods["PCRL_paper"]["per_dataset"]
    pcrl_aggr = aggregate_across_datasets(pcrl_per)
    rows.append((methods["PCRL_paper"]["label"], methods["PCRL_paper"]["short_label"],
                 pcrl_per, pcrl_aggr))

    # Row 2: LAFTR-Q (frozen; HMDA + Diabetes likely null on disk)
    laftr_q_per = methods["LAFTR_appendixQ"]["per_dataset"]
    laftr_q_aggr = aggregate_across_datasets(laftr_q_per)
    rows.append((methods["LAFTR_appendixQ"]["label"], methods["LAFTR_appendixQ"]["short_label"],
                 laftr_q_per, laftr_q_aggr))

    # Row 3: LAFTR-hard-R² (live; None per dataset until pilot lands)
    laftr_hr2_per: dict[str, dict | None] = {}
    for ds in DATASETS:
        f = load_laftr_hard_r2_per_seed(root, ds)
        laftr_hr2_per[ds] = compute_laftr_hard_r2_metrics(f) if f is not None else None
    laftr_hr2_aggr = aggregate_across_datasets(laftr_hr2_per)
    rows.append(("LAFTR-hard-R² (this pilot, lambda_adv=1.0 + proxy-Lagrangian)",
                 "LAFTR-hard-R²", laftr_hr2_per, laftr_hr2_aggr))

    payload = {
        "_source": baselines["_source"],
        "_strict_pass_threshold": TAU,
        "rows": [
            {
                "method": label,
                "short_label": short,
                "per_dataset": per,
                "aggregated": aggr,
            }
            for label, short, per, aggr in rows
        ],
    }
    return rows, payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(DEFAULT_REPO_ROOT),
                   help="Repo root containing scripts/ and results/")
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <root>/results/laftr_hard_r2/)")
    args = p.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root / "results" / "laftr_hard_r2")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, payload = build_rows(root)

    tex = render_tex(rows)
    headline = render_headline(rows)

    (out_dir / "comparison_table.tex").write_text(tex)
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2, default=str))
    (out_dir / "HEADLINE.txt").write_text(headline)

    print(headline)
    print(f"wrote: {out_dir / 'comparison_table.tex'}")
    print(f"wrote: {out_dir / 'comparison.json'}")
    print(f"wrote: {out_dir / 'HEADLINE.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
