#!/usr/bin/env python3
"""Aggregate separate-encoders results across datasets and emit
results/SEPARATE_SUMMARY.md plus a stdout comparison table.

Reads results/<dataset>_SEPARATE/summary.json for whichever datasets are
present. Missing datasets are reported as "not run yet" rather than
failing the aggregation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["adult", "diabetes", "hmda"]


def load_summary(dataset: str) -> dict | None:
    path = ROOT / "results" / f"{dataset}_SEPARATE" / "summary.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def verdict(separate_pass: float, paper_pass: float,
            separate_task: float | None, paper_task: float | None) -> str:
    pass_diff = separate_pass - paper_pass
    if pass_diff >= -0.5:
        if separate_task is not None and paper_task is not None:
            task_diff = separate_task - paper_task
            if abs(task_diff) <= 0.02:
                return "Conditioning not load-bearing on this dataset (pass and task within tolerance)."
            elif task_diff < -0.02:
                return f"Pass parity, but separate loses {abs(task_diff):.1%} task accuracy."
            else:
                return f"Pass parity AND separate gains {task_diff:.1%} task accuracy — conditioning may be a parameter-efficiency choice, not a compliance one."
        return "Pass parity (task accuracy not directly comparable)."
    if pass_diff <= -1.5:
        return "Conditioning has real compliance value — separate encoders fail more disallowed-attribute audits."
    return "Borderline — separate close to PCRL but not parity. Conditioning is parameter-efficiency win at most."


def render_summary(summaries: dict[str, dict | None]) -> tuple[str, str]:
    """Returns (markdown, stdout_text)."""
    md_lines: list[str] = []
    md_lines.append("# Separate-Encoders Threat Experiment — Summary\n")
    md_lines.append("Trains |P| separate encoders (no FiLM, no purpose embedding) and "
                    "compares to PCRL paper compliance + task accuracy.\n")
    md_lines.append("| dataset | separate pass | PCRL paper pass | separate task | PCRL task | wall (s) |")
    md_lines.append("|---------|---------------|-----------------|---------------|-----------|----------|")

    stdout_lines: list[str] = []
    stdout_lines.append("=" * 90)
    stdout_lines.append("SEPARATE-ENCODERS THREAT EXPERIMENT SUMMARY")
    stdout_lines.append("=" * 90)
    stdout_lines.append(f"{'dataset':<10} {'sep pass':>10} {'paper pass':>12} "
                        f"{'sep task':>12} {'paper task':>12} {'wall (s)':>10}")
    stdout_lines.append("-" * 90)

    verdicts: list[tuple[str, str]] = []

    for ds in DATASETS:
        s = summaries.get(ds)
        if s is None:
            md_lines.append(f"| {ds} | — | — | — | — | (not yet run) |")
            stdout_lines.append(f"{ds:<10} {'—':>10} {'—':>12} {'—':>12} {'—':>12} {'(missing)':>10}")
            continue
        sep_pass = f"{s['total_pass']}/{s['total_pairs']}"
        paper_pass = f"{s['pcrl_paper_pass']}/{s['pcrl_paper_total']}"
        sep_task = f"{s['headline_task_acc']:.1%}" if s.get("headline_task_acc") is not None else "—"
        paper_task = f"{s['pcrl_paper_task_acc']:.1%}" if s.get("pcrl_paper_task_acc") is not None else "—"
        wall = f"{s.get('total_train_time_s', 0):.0f}"
        md_lines.append(f"| {ds} | {sep_pass} | {paper_pass} | {sep_task} | {paper_task} | {wall} |")
        stdout_lines.append(f"{ds:<10} {sep_pass:>10} {paper_pass:>12} {sep_task:>12} {paper_task:>12} {wall:>10}")

        v = verdict(
            float(s["total_pass"]),
            float(s["pcrl_paper_pass"]),
            s.get("headline_task_acc"),
            s.get("pcrl_paper_task_acc"),
        )
        verdicts.append((ds, v))

    stdout_lines.append("=" * 90)
    md_lines.append("")
    md_lines.append("## Per-dataset verdict\n")
    for ds, v in verdicts:
        md_lines.append(f"- **{ds}** — {v}")
        stdout_lines.append(f"  {ds}: {v}")
    md_lines.append("")

    if all(summaries.get(d) is not None for d in DATASETS):
        # Overall verdict
        total_sep = sum(summaries[d]["total_pass"] for d in DATASETS)
        total_paper = sum(summaries[d]["pcrl_paper_pass"] for d in DATASETS)
        total_pairs = sum(summaries[d]["total_pairs"] for d in DATASETS)
        md_lines.append("## Overall\n")
        md_lines.append(f"- Total pass: separate={total_sep}/{total_pairs} vs PCRL paper={total_paper}/{total_pairs}")
        if total_sep >= total_paper:
            md_lines.append("- **Direction:** separate encoders match or exceed conditioning at compliance.\n")
        else:
            md_lines.append("- **Direction:** PCRL conditioning has measurably better compliance than separate encoders.\n")
        stdout_lines.append("")
        stdout_lines.append(f"OVERALL: separate {total_sep}/{total_pairs} vs PCRL paper {total_paper}/{total_pairs}")

    return "\n".join(md_lines), "\n".join(stdout_lines)


def main() -> None:
    summaries: dict[str, dict | None] = {}
    for ds in DATASETS:
        summaries[ds] = load_summary(ds)

    md, stdout = render_summary(summaries)
    out_path = ROOT / "results" / "SEPARATE_SUMMARY.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(stdout)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
