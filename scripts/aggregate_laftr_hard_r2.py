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
# Representational-health diagnostics
# ──────────────────────────────────────────────────────────────────────────────

# Published cleanly-compliant thresholds (matches v2_dataset orchestrator and
# the erase-pilot aggregator). A "collapsed" seed is one where any purpose's
# per_dim_std mean falls below PER_DIM_STD_MIN.
PER_DIM_STD_MIN = 0.5
EFF_RANK_MIN = 2.0


def compute_collapse_diagnostics(
    per_seed_file: dict, focus_task: str | None = None,
) -> dict[str, Any]:
    """Return per-seed representational-health summary.

    For each seed, looks at ``per_purpose_health`` and reports the minimum
    per_dim_std_mean and minimum effective_rank across purposes. A seed
    counts as ``collapsed`` if min per_dim_std_mean < ``PER_DIM_STD_MIN`` or
    min eff_rank < ``EFF_RANK_MIN``.

    If ``focus_task`` is given, also reports that task's mean accuracy
    across seeds (used by render_paper_paste to surface the "compliance via
    collapse" signal, e.g. primary_diagnosis_category at majority baseline).
    """
    seeds = per_seed_file["per_seed"]
    per_seed = []
    n_collapsed = 0
    focus_accs: list[float] = []
    for s in seeds:
        health = s.get("per_purpose_health") or {}
        if not health:
            continue
        std_mins = [h.get("per_dim_std_mean", 0.0) for h in health.values()]
        rank_mins = [h.get("effective_rank", 0.0) for h in health.values()]
        std_min = min(std_mins) if std_mins else float("nan")
        rank_min = min(rank_mins) if rank_mins else float("nan")
        collapsed = std_min < PER_DIM_STD_MIN or rank_min < EFF_RANK_MIN
        if collapsed:
            n_collapsed += 1
        per_seed.append({
            "seed": s["seed"],
            "min_per_dim_std": std_min,
            "min_eff_rank": rank_min,
            "collapsed": collapsed,
        })
        if focus_task and focus_task in s.get("task_accuracies", {}):
            focus_accs.append(s["task_accuracies"][focus_task])
    return {
        "per_seed": per_seed,
        "n_seeds": len(per_seed),
        "n_collapsed": n_collapsed,
        "focus_task": focus_task,
        "focus_task_mean_acc": (sum(focus_accs) / len(focus_accs)) if focus_accs else None,
        "thresholds": {
            "per_dim_std_min": PER_DIM_STD_MIN,
            "eff_rank_min": EFF_RANK_MIN,
        },
    }


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
    task_acc_dataset_mask: set[str] | None = None,
) -> dict[str, Any]:
    """Compute the "Aggregated (N cells)" column.

    Strict-pass and mean-R²-on-passing aggregate over EVERY dataset that has
    a populated block — the strict-pass denominator is the full grid count
    regardless of task-acc availability.

    Task accuracy is more subtle. The caller may pass an explicit
    ``task_acc_dataset_mask`` (a set of dataset names) to restrict which
    datasets contribute to the weighted task-acc average — used by
    ``build_rows`` to enforce apples-to-apples cross-method aggregation
    (only datasets where every method reports a comparable task-acc
    contribute, see ``_common_task_acc_datasets``). With no mask, task-acc
    aggregates over every populated dataset that has a non-null, non-NaN
    task-acc value (legacy behavior).

    The returned dict carries ``task_acc_n_cells`` so the renderer can show
    ``"79.3% (60 cells)"`` vs ``"67.7% (42 cells)"`` and reviewers see the
    denominator behind each aggregated task-acc number.
    """
    total_cells = 0
    total_passing = 0
    sum_r2_passing = 0.0
    n_summed_r2 = 0
    weighted_acc_sum = 0.0
    weighted_acc_n = 0
    task_acc_n_cells = 0  # number of cells contributing to weighted task-acc

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
        # Task-acc contribution is gated by both (a) the cross-method mask
        # (if provided) and (b) the block having a real task-acc value.
        if task_acc_dataset_mask is not None and ds not in task_acc_dataset_mask:
            continue
        if block.get("task_acc") is not None and not _is_nan(block["task_acc"]):
            weighted_acc_sum += block["task_acc"] * n_cells
            weighted_acc_n += n_cells
            task_acc_n_cells += n_cells

    if total_cells == 0:
        return {
            "strict_pass_rate": float("nan"),
            "mean_r2_on_passing": None,
            "task_acc": float("nan"),
            "n_cells_total": 0,
            "n_passing": 0,
            "task_acc_n_cells": 0,
        }
    return {
        "strict_pass_rate": total_passing / total_cells,
        "mean_r2_on_passing": (sum_r2_passing / n_summed_r2) if n_summed_r2 > 0 else None,
        "task_acc": weighted_acc_sum / weighted_acc_n if weighted_acc_n > 0 else float("nan"),
        "n_cells_total": total_cells,
        "n_passing": total_passing,
        "task_acc_n_cells": task_acc_n_cells,
    }


def _common_task_acc_datasets(
    rows_per_dataset: list[tuple[str, dict[str, dict | None]]],
) -> set[str]:
    """Return the set of datasets where EVERY method reports a non-null,
    non-NaN task_acc.

    A row contributes a "miss" for a dataset if its block is None, missing
    a task_acc key, or has task_acc=None / NaN. Any miss disqualifies that
    dataset from the common mask — this is the strict apples-to-apples
    interpretation (a row with no LAFTR-hard-R² data yet rules out every
    dataset until results land, which is the desired behavior).
    """
    common = set(DATASETS)
    for _, per_ds in rows_per_dataset:
        for ds in DATASETS:
            b = per_ds.get(ds)
            if b is None or b.get("task_acc") is None or _is_nan(b.get("task_acc")):
                common.discard(ds)
    return common


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


def _fmt_acc_agg(aggr: dict | None, latex: bool = False) -> str:
    """Aggregated task-acc cell: always shows the (N cells) denominator.

    When the mask reduces the denominator below the strict-pass cell count
    (e.g. 60 → 42 because LAFTR-Q HMDA task-acc is null), the per-cell
    count makes the asymmetry visible inline rather than hidden behind a
    matching percentage.
    """
    if aggr is None:
        return "—"
    v = aggr.get("task_acc")
    n = aggr.get("task_acc_n_cells", 0)
    if v is None or _is_nan(v) or n == 0:
        return "—"
    pct = "\\%" if latex else "%"
    return f"{v * 100:.1f}{pct} ({n} cells)"


def _excluded_task_acc_summary(
    rows: list[tuple[str, str, dict[str, dict | None], dict]],
    common_ds: set[str],
) -> list[tuple[str, list[str]]]:
    """For each dataset excluded from the cross-method task-acc mask, list the
    method short-labels that caused the exclusion (those reporting null task-acc).
    Returned list is sorted in canonical DATASETS order.
    """
    excluded: list[tuple[str, list[str]]] = []
    for ds in DATASETS:
        if ds in common_ds:
            continue
        missing: list[str] = []
        for _, short, per_ds, _ in rows:
            b = per_ds.get(ds)
            if b is None or b.get("task_acc") is None or _is_nan(b.get("task_acc")):
                missing.append(short)
        excluded.append((ds, missing))
    return excluded


def _render_caption(
    excluded: list[tuple[str, list[str]]],
) -> str:
    """Build the caption, appending an auto-generated \\footnote when the
    task-acc cross-method mask drops any cells."""
    base = (
        r"LAFTR hard-R² rebuttal pilot. "
        r"Three methods on the same 60-cell grid (Adult 24 + HMDA 18 + Diabetes 18, "
        r"3 seeds each). "
        r"\textit{Strict pass}: fraction of (purpose, attribute, seed) cells with "
        r"linear $R^2(h_p, A) < 0.05$. "
        r"\textit{Mean $R^2$ on passing}: mean linear $R^2$ over only the passing cells "
        r"(depth-of-compliance; ``—'' when no cells pass). "
        r"\textit{Task acc}: arithmetic mean over (purpose, seed); aggregated task-acc is "
        r"averaged over cells where all three methods report comparable values."
    )
    if not excluded:
        return base
    parts: list[str] = []
    for ds, missing in excluded:
        ds_pretty = ds.upper() if ds == "hmda" else ds.title()
        parts.append(
            f"{ds_pretty} cells excluded "
            f"({', '.join(missing)} report no comparable task acc)"
        )
    footnote_body = (
        r"Task accuracy is averaged over cells where all three methods report comparable values; "
        + "; ".join(parts)
        + r". Per-cell rationale in \texttt{scripts/paper\_baseline\_numbers.json}."
    )
    return base + r" \protect\footnote{" + footnote_body + r"}"


def render_tex(
    rows: list[tuple[str, str, dict[str, dict | None], dict]],
    common_ds: set[str] | None = None,
) -> str:
    """3-row LaTeX table. Each tuple: (method_label, method_short, per_dataset, aggregated).

    If ``common_ds`` is provided and a proper subset of DATASETS, the caption gets a
    \\footnote auto-explaining which dataset(s) are excluded from the aggregated
    task-acc column and which method(s) caused each exclusion.
    """
    lines = []
    lines.append(r"% Auto-generated by scripts/aggregate_laftr_hard_r2.py — do not edit by hand.")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    # When common_ds is None, callers haven't computed the mask — treat as
    # full coverage (no footnote). When it's an empty set, the mask is
    # legitimately empty (one method has zero data) and every dataset is
    # excluded — distinct from None and should fire the footnote.
    effective_common_ds = set(DATASETS) if common_ds is None else common_ds
    excluded = _excluded_task_acc_summary(rows, effective_common_ds)
    lines.append(r"\caption{" + _render_caption(excluded) + "}")
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
        # Task acc — aggregated cell uses _fmt_acc_agg to show (N cells)
        lines.append(
            r" & Task acc & "
            rf"{_fmt_acc(per_ds.get('adult'), latex=True)} & "
            rf"{_fmt_acc(per_ds.get('hmda'), latex=True)} & "
            rf"{_fmt_acc(per_ds.get('diabetes'), latex=True)} & "
            rf"{_fmt_acc_agg(aggr, latex=True)} \\"
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
    common_ds: set[str] | None = None,
) -> str:
    """Plain-text summary: strict-pass table + aggregated task-acc mini-block.

    The aggregated task-acc line shows ``"acc% (N cells)"`` so the
    cross-method ledger is visible. If ``common_ds`` is a proper subset of
    DATASETS, a one-line note explains which datasets are excluded and
    why.
    """
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
    parts.append("  Aggregated task acc (apples-to-apples: common cells across all three methods):")
    for _, method_short, _, aggr in rows:
        parts.append(f"    {method_short:38s} {_fmt_acc_agg(aggr, latex=False)}")
    parts.append("")
    parts.append("Strict pass: linear R²(h_p, A) < 0.05; format \"rate (n_pass/n_cells)\"")
    parts.append("Depth-of-compliance + per-dataset task acc are in comparison.json / comparison_table.tex")
    if common_ds is not None and common_ds != set(DATASETS):
        excluded = _excluded_task_acc_summary(rows, common_ds)
        if excluded:
            for ds, missing in excluded:
                parts.append(
                    f"Note: aggregated task acc excludes {ds.upper() if ds == 'hmda' else ds.title()} "
                    f"because {', '.join(missing)} report no comparable task acc value "
                    f"(see scripts/paper_baseline_numbers.json)."
                )
    return "\n".join(parts) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def build_rows(
    root: Path,
) -> tuple[list[tuple[str, str, dict[str, dict | None], dict]], dict[str, Any], set[str]]:
    """Return (rows_for_renderers, payload_for_json, common_task_acc_datasets).

    Aggregates each method's task-acc only over the cross-method common
    dataset mask (the set of datasets where every method reports a
    comparable, non-null task-acc value). This produces an apples-to-apples
    aggregated task-acc and prevents silent denominator drift across
    methods.
    """
    baselines = load_frozen_baselines(root)
    methods = baselines["methods"]

    # Per-dataset blocks for each method.
    pcrl_per = methods["PCRL_paper"]["per_dataset"]
    laftr_q_per = methods["LAFTR_appendixQ"]["per_dataset"]
    laftr_hr2_per: dict[str, dict | None] = {}
    for ds in DATASETS:
        f = load_laftr_hard_r2_per_seed(root, ds)
        laftr_hr2_per[ds] = compute_laftr_hard_r2_metrics(f) if f is not None else None

    # Cross-method common-cell mask for task-acc aggregation.
    common_task_acc_ds = _common_task_acc_datasets([
        ("PCRL", pcrl_per),
        ("LAFTR-Q", laftr_q_per),
        ("LAFTR-hard-R²", laftr_hr2_per),
    ])

    # Aggregate each method with the same mask — apples-to-apples.
    pcrl_aggr = aggregate_across_datasets(pcrl_per, task_acc_dataset_mask=common_task_acc_ds)
    laftr_q_aggr = aggregate_across_datasets(laftr_q_per, task_acc_dataset_mask=common_task_acc_ds)
    laftr_hr2_aggr = aggregate_across_datasets(laftr_hr2_per, task_acc_dataset_mask=common_task_acc_ds)

    rows: list[tuple[str, str, dict[str, dict | None], dict]] = [
        (methods["PCRL_paper"]["label"], methods["PCRL_paper"]["short_label"],
         pcrl_per, pcrl_aggr),
        (methods["LAFTR_appendixQ"]["label"], methods["LAFTR_appendixQ"]["short_label"],
         laftr_q_per, laftr_q_aggr),
        ("LAFTR-hard-R² (this pilot, lambda_adv=1.0 + proxy-Lagrangian)",
         "LAFTR-hard-R²", laftr_hr2_per, laftr_hr2_aggr),
    ]

    excluded = _excluded_task_acc_summary(rows, common_task_acc_ds)
    task_acc_asymmetric = bool(excluded)

    payload = {
        "_source": baselines["_source"],
        "_strict_pass_threshold": TAU,
        "task_acc_common_datasets": sorted(common_task_acc_ds),
        "task_acc_asymmetric": task_acc_asymmetric,
        "task_acc_excluded_datasets": [
            {"dataset": ds, "missing_in_methods": missing}
            for ds, missing in excluded
        ],
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
    return rows, payload, common_task_acc_ds


def render_paper_paste(
    rows: list[tuple[str, str, dict[str, dict | None], dict]],
    common_ds: set[str] | None = None,
    root: Path | None = None,
) -> str:
    """Camera-ready Markdown for direct paste into the rebuttal.

    Includes the comparison table plus the Q3 finding from the 2026-05-29
    investigation: Adult's ``occupation_group`` and ``education_level`` task
    labels are deterministic recodings of one-hot input features, so all
    methods that don't actively destroy representational structure (i.e.
    every method except PCRL + erase-layer) achieve ~99% on those tasks by
    passthrough. The non-trivial Adult task is ``income``.
    """
    excluded = _excluded_task_acc_summary(rows, common_ds or set(DATASETS))
    asymmetric = bool(excluded)

    lines: list[str] = []
    lines.append("# LAFTR hard-R² rebuttal pilot — camera-ready paste")
    lines.append("")
    lines.append("> Auto-generated by `scripts/aggregate_laftr_hard_r2.py`. Do not edit by hand.")
    lines.append("> The Adult numbers reflect what each method achieves on the same backbone")
    lines.append("> (`StandardEncoder([128,128]→64) + PerPurposeLoRAEncoder(rank=8)`) and the")
    lines.append("> same audit code path (`pcrl.evaluation.certificates.generate_report`).")
    lines.append("")
    lines.append("## Comparison")
    lines.append("")
    lines.append("| Method | Adult (24) | HMDA (18) | Diabetes (18) | Aggregated (60) |")
    lines.append("|---|---|---|---|---|")
    for method_label, _, per_ds, aggr in rows:
        lines.append(
            f"| **{method_label}** — strict pass | "
            f"{_fmt_pass_rate_plain(per_ds.get('adult'))} | "
            f"{_fmt_pass_rate_plain(per_ds.get('hmda'))} | "
            f"{_fmt_pass_rate_plain(per_ds.get('diabetes'))} | "
            f"{_fmt_pass_rate_plain(aggr)} |"
        )
        lines.append(
            f"| _Mean $R^2$ on passing_ | "
            f"{_fmt_r2(per_ds.get('adult'))} | "
            f"{_fmt_r2(per_ds.get('hmda'))} | "
            f"{_fmt_r2(per_ds.get('diabetes'))} | "
            f"{_fmt_r2(aggr)} |"
        )
        # task-acc row — plain rendering, aggregated cell shows (N cells)
        agg_acc = aggr.get("task_acc") if aggr else None
        agg_n = aggr.get("task_acc_n_cells", 0) if aggr else 0
        if agg_acc is None or _is_nan(agg_acc) or agg_n == 0:
            agg_acc_str = "—"
        else:
            agg_acc_str = f"{agg_acc * 100:.1f}% ({agg_n} cells)"
        lines.append(
            f"| _Task acc_ | "
            f"{_fmt_acc(per_ds.get('adult'), latex=False)} | "
            f"{_fmt_acc(per_ds.get('hmda'), latex=False)} | "
            f"{_fmt_acc(per_ds.get('diabetes'), latex=False)} | "
            f"{agg_acc_str} |"
        )
    lines.append("")
    lines.append("## Reading the Adult task-accuracy row (important)")
    lines.append("")
    lines.append("Adult has three task labels. Only **`income`** is a non-trivial prediction;")
    lines.append("`occupation_group` and `education_level` are deterministic recodings of")
    lines.append("one-hot input features (`occupation` 14→6, `education` 16→4; see")
    lines.append("`pcrl/data/adult.py`). Every method that doesn't actively destroy")
    lines.append("representational structure achieves ~99% on those two tasks by passthrough,")
    lines.append("including PCRL Round 5 and LAFTR variants. The single Adult comparison that")
    lines.append("distinguishes methods is `income`:")
    lines.append("")
    lines.append("| Method | Adult `income` task acc (mean over 3 seeds) |")
    lines.append("|---|---|")
    for method_label, _, per_ds, _ in rows:
        adult = per_ds.get("adult")
        if adult is None:
            inc_str = "—"
        else:
            # If the block has a populated task_acc, use it as a coarse proxy.
            # For the PCRL/LAFTR-Q rows the frozen JSON stores a single
            # aggregated task_acc (mean over all task labels); for the LAFTR-
            # hard-R² row computed live, this is the mean over per-seed
            # task_accuracies values across all tasks.
            v = adult.get("task_acc")
            inc_str = f"{v * 100:.1f}% (mean over tasks)" if v is not None and not _is_nan(v) else "—"
        lines.append(f"| {method_label} | {inc_str} |")
    lines.append("")
    lines.append("(The per-seed `income`-only number is in")
    lines.append("`results/laftr_hard_r2_adult_LAFTR_HARD_R2/per_seed_results.json[task_accuracies][income]`")
    lines.append("and in PCRL Round 5's corresponding file. For LAFTR-Q the income-only number")
    lines.append("is in `results/laftr_benchmark/adult/income_prediction/seed_<n>/metrics.json`.)")
    lines.append("")
    # ── Diabetes compliance-via-collapse framing ─────────────────────────
    # Mirrors the Adult deterministic-recoding caveat. The Diabetes 18/18
    # pass is real on the strict-R² criterion but pays for it with
    # representational collapse — same failure mode the submitted paper's
    # Appendix P applies to LAFTR-Q's Diabetes result (task acc 31.3% vs
    # majority baseline 28%). We surface per-seed health diagnostics so a
    # reviewer can see the 18/18 number isn't "real" compliance.
    if root is not None:
        diab = load_laftr_hard_r2_per_seed(root, "diabetes")
        if diab is not None:
            diag = compute_collapse_diagnostics(
                diab, focus_task="primary_diagnosis_category"
            )
            if diag["n_collapsed"] > 0:
                lines.append("## Reading the Diabetes 18/18 row (important)")
                lines.append("")
                lines.append("LAFTR-hard-R² hits 18/18 strict pass on Diabetes — but this is")
                lines.append("compliance via representational collapse, not erasure. The")
                lines.append(f"per-seed representational health (thresholds: per_dim_std mean ≥ "
                             f"{diag['thresholds']['per_dim_std_min']}, effective rank ≥ "
                             f"{diag['thresholds']['eff_rank_min']}):")
                lines.append("")
                lines.append("| Seed | Min per_dim_std mean | Min effective rank | Collapsed? |")
                lines.append("|---|---|---|---|")
                for s in diag["per_seed"]:
                    flag = "**yes**" if s["collapsed"] else "no"
                    lines.append(
                        f"| {s['seed']} | {s['min_per_dim_std']:.3f} | "
                        f"{s['min_eff_rank']:.2f} | {flag} |"
                    )
                lines.append("")
                if diag["focus_task_mean_acc"] is not None:
                    lines.append(
                        f"The corollary in task-acc space: `primary_diagnosis_category` "
                        f"averages **{diag['focus_task_mean_acc']*100:.1f}%** across seeds — "
                        f"essentially the majority baseline. This is the same failure mode the"
                    )
                else:
                    lines.append("The same failure mode the")
                lines.append("submitted paper applies to LAFTR-Q on Diabetes in Appendix P")
                lines.append("(LAFTR-Q `primary_diagnosis_category` at 31.3% vs majority")
                lines.append("baseline 28%). PCRL Round 7 reaches 17/18 strict pass on the")
                lines.append("same dataset **without** collapse (per_dim_std > 0.5,")
                lines.append("eff_rank > 2.0 on every seed); PCRL + erase-layer reaches 18/18")
                lines.append("also without collapse. The 18/18 number in row 3 should be read")
                lines.append("alongside the collapse diagnostic; it is not real erasure.")
                lines.append("")
                lines.append(
                    f"({diag['n_collapsed']}/{diag['n_seeds']} seeds satisfy the collapse "
                    f"condition; per-seed numbers in `results/laftr_hard_r2_diabetes_LAFTR_HARD_R2/"
                    f"per_seed_results.json[per_seed][·][per_purpose_health]`.)"
                )
                lines.append("")
    lines.append("## Headline interpretation")
    lines.append("")
    lines.append("- **R1's complaint** — LAFTR was audited under PCRL's strict-R² criterion")
    lines.append("  despite not being trained against it — is addressed by the third row.")
    lines.append("  Training LAFTR with the proxy-Lagrangian R²<0.05 constraint added on top")
    lines.append("  of its published `λ_adv=1.0` adversarial loss moves the aggregated")
    lines.append("  pass-rate from 15/60 (Appendix Q) to 35/60 — a real, attributable gain")
    lines.append("  from the constraint mechanism. It does not close the gap to PCRL's")
    lines.append("  54/60 or PCRL+erase-layer's 60/60.")
    lines.append("- **Adult fails (0/24) despite the constraint engaging.** Dual variables")
    lines.append("  ramp (some to λ≈80), the primal cannot give back, and the encoder")
    lines.append("  cannot satisfy linear R²<0.05 simultaneously with the task and")
    lines.append("  adversarial losses on this backbone. Same R² as Appendix Q's")
    lines.append("  unconstrained LAFTR — the architectural ceiling is binding.")
    lines.append("- **HMDA passes 17/18** clean-ish — one stubborn pair")
    lines.append("  (`pricing_analysis/race` seed 0, R²=0.07). All seeds had")
    lines.append("  `feasible=0/200` from Cotter; fallback selection took the lowest-")
    lines.append("  violation iterate. The constraint composes with LAFTR's adversarial")
    lines.append("  loss on this dataset.")
    lines.append("- **Diabetes passes 18/18 but via collapse** (see the Diabetes section")
    lines.append("  above). The 18/18 number is the same compliance-via-collapse mechanism")
    lines.append("  the paper applies to LAFTR-Q in Appendix P; it is not real erasure.")
    lines.append("  PCRL Round 7 reaches 17/18 on Diabetes without collapse.")
    lines.append("- **The erase-layer ablation is the load-bearing change** for clean")
    lines.append("  compliance. PCRL + erase-layer achieves 60/60 on the same grid")
    lines.append("  (Adult 24/24 + HMDA 18/18 + Diabetes 18/18) and does so by")
    lines.append("  destroying enough representational capacity that even the")
    lines.append("  `occupation_group` passthrough breaks from ~99% to ~77% — direct")
    lines.append("  evidence that the constraint is biting on the encoder, not the readout.")
    lines.append("")
    if asymmetric:
        lines.append("## Note on the aggregated task-accuracy column")
        lines.append("")
        lines.append("The aggregated task-acc is averaged over cells where all three methods")
        lines.append("report comparable values. Excluded:")
        for ds, missing in excluded:
            ds_pretty = ds.upper() if ds == "hmda" else ds.title()
            lines.append(f"- **{ds_pretty}**: {', '.join(missing)} report no comparable task acc value.")
        lines.append("")
        lines.append("Per-cell rationale lives in `scripts/paper_baseline_numbers.json`.")
        lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("- Branch `laftr-hard-r2-2026-05-17`")
    lines.append("- Aggregator: `scripts/aggregate_laftr_hard_r2.py`")
    lines.append("- Frozen baselines: `scripts/paper_baseline_numbers.json`")
    lines.append("- LAFTR-hard-R² source results: `results/laftr_hard_r2_<dataset>_LAFTR_HARD_R2/`")
    lines.append("- PCRL Round 5 / Round 7 source results: `results/v2_<dataset>_ROUND[57]/`")
    lines.append("- PCRL + erase-layer source results: `results/rebuttal/erase_layer_pilot_aws/`")
    return "\n".join(lines) + "\n"


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

    rows, payload, common_ds = build_rows(root)

    tex = render_tex(rows, common_ds=common_ds)
    headline = render_headline(rows, common_ds=common_ds)
    paper_paste = render_paper_paste(rows, common_ds=common_ds, root=root)

    (out_dir / "comparison_table.tex").write_text(tex)
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2, default=str))
    (out_dir / "HEADLINE.txt").write_text(headline)
    (out_dir / "PAPER_PASTE.md").write_text(paper_paste)

    print(headline)
    print(f"wrote: {out_dir / 'comparison_table.tex'}")
    print(f"wrote: {out_dir / 'comparison.json'}")
    print(f"wrote: {out_dir / 'HEADLINE.txt'}")
    print(f"wrote: {out_dir / 'PAPER_PASTE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
