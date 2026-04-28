#!/usr/bin/env python3
"""Aggregate v2 validation results across datasets and emit
results/V2_VALIDATION_SUMMARY.md plus a stdout comparison.

Compares v2 trainer (LoRA + HSIC + vCLUB + VICReg + proxy-Lagrangian)
to v1 paper, LEACE-on-raw, and the separate-encoders threat run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["adult", "diabetes", "hmda"]

# Reference numbers from prior experiments / paper
V1_PAPER = {
    "adult":    {"pass": 6.0, "total": 8, "task_label": "income",                     "task_acc": 0.763},
    "diabetes": {"pass": 5.0, "total": 6, "task_label": "primary_diagnosis_category", "task_acc": None},
    "hmda":     {"pass": 5.3, "total": 6, "task_label": "loan_decision",              "task_acc": None},
}

LEACE_ON_RAW = {  # 0/20 across all datasets per threat experiment 1
    "adult":    {"pass": 0, "total": 8},
    "diabetes": {"pass": 0, "total": 6},
    "hmda":     {"pass": 0, "total": 6},
}

SEPARATE = {  # threat experiment 2 (already on origin)
    "adult":    {"pass": 0, "total": 8, "status": "OK"},
    "diabetes": {"pass": 5, "total": 6, "status": "COLLAPSED"},
    "hmda":     {"pass": 0, "total": 6, "status": "OK"},
}


def load_v2(dataset: str) -> tuple[dict, str] | None:
    """Return (summary, source_tag) for the most recent v2 run we have for ``dataset``.

    Prefers Round 4 (BN-freeze + LEACE warm-start + final.pt selection) when
    present, falling back to the earlier ``v2_<dataset>/`` directory.
    """
    candidates = [
        (ROOT / "results" / f"v2_{dataset}_ROUND4" / "summary.json", "ROUND4 (final.pt)"),
        (ROOT / "results" / f"v2_{dataset}" / "summary.json", "initial"),
    ]
    for path, tag in candidates:
        if path.exists():
            with open(path) as fh:
                return json.load(fh), tag
    return None


def render() -> tuple[str, str]:
    md: list[str] = []
    md.append("# V2 Validation Summary\n")
    md.append("V2 = frozen StandardEncoder backbone + per-purpose LoRA adapters,\n"
              "HSIC + vCLUB independence penalty, VICReg anti-collapse,\n"
              "proxy-Lagrangian dual variables on HSIC ≤ 0.05.\n")
    md.append("Round 4 adds BN-freeze + LEACE warm-start of LoRA adapters and "
              "reports compliance from `final.pt` (the Cotter best-iterate "
              "selector currently picks epoch ~12; see "
              "`results/v2_adult_ROUND4/cotter_selection_bug.md`).\n")
    md.append("Compared against three reference points:\n")
    md.append("- **v1 paper** — adversarial PCRL with FiLM (mean over 3 seeds).")
    md.append("- **LEACE-on-raw** — concept-erasure baseline (threat exp 1).")
    md.append("- **Separate encoders** — one frozen StandardEncoder per purpose, "
              "no conditioning (threat exp 2).\n")

    md.append("| dataset | v2 pass (3 seeds) | v2 status | v1 paper | LEACE | separate | v2 task | v1 task |")
    md.append("|---|---|---|---|---|---|---|---|")

    stdout: list[str] = []
    stdout.append("=" * 110)
    stdout.append("V2 VALIDATION — comparison vs v1 paper, LEACE, separate encoders")
    stdout.append("=" * 110)
    stdout.append(f"{'dataset':<10} {'v2 pass':>14} {'v2 status':>11} {'v1 paper':>10} "
                  f"{'LEACE':>8} {'separate':>14} {'v2 task':>10} {'v1 task':>10}")
    stdout.append("-" * 110)

    overall_v2_pass = 0.0
    overall_v2_pairs = 0
    overall_v1_pass = 0.0
    have_all = True

    for ds in DATASETS:
        loaded = load_v2(ds)
        if loaded is None:
            md.append(f"| {ds} | (not yet run) | — | "
                      f"{V1_PAPER[ds]['pass']}/{V1_PAPER[ds]['total']} | "
                      f"{LEACE_ON_RAW[ds]['pass']}/{LEACE_ON_RAW[ds]['total']} | "
                      f"{SEPARATE[ds]['pass']}/{SEPARATE[ds]['total']} ({SEPARATE[ds]['status']}) | "
                      f"— | — |")
            stdout.append(f"{ds:<10} {'(missing)':>14} {'—':>11} "
                          f"{V1_PAPER[ds]['pass']}/{V1_PAPER[ds]['total']:>3} "
                          f"{LEACE_ON_RAW[ds]['pass']}/{LEACE_ON_RAW[ds]['total']:>3} "
                          f"{SEPARATE[ds]['pass']}/{SEPARATE[ds]['total']:>3} ({SEPARATE[ds]['status']:<10}) "
                          f"{'—':>10} {'—':>10}")
            have_all = False
            continue
        s, source_tag = loaded

        v2_pass = s["pass_count_mean"]
        v2_std = s["pass_count_std"]
        v2_pairs = s["total_pairs_per_seed"]
        v2_status = s["STATUS"]
        v1_pass = V1_PAPER[ds]["pass"]
        v1_pairs = V1_PAPER[ds]["total"]
        leace = LEACE_ON_RAW[ds]
        sep = SEPARATE[ds]

        v2_pass_str = f"{v2_pass:.2f}±{v2_std:.2f}/{v2_pairs}"
        v1_pass_str = f"{v1_pass}/{v1_pairs}"
        leace_str = f"{leace['pass']}/{leace['total']}"
        sep_str = f"{sep['pass']}/{sep['total']} ({sep['status']})"

        task_label = V1_PAPER[ds]["task_label"]
        v2_task_acc = s["task_acc_mean"].get(task_label)
        v2_task_std = s["task_acc_std"].get(task_label, 0.0)
        v1_task_acc = V1_PAPER[ds]["task_acc"]
        v2_task_str = f"{v2_task_acc:.1%}±{v2_task_std:.1%}" if v2_task_acc is not None else "—"
        v1_task_str = f"{v1_task_acc:.1%}" if v1_task_acc is not None else "—"

        md.append(f"| {ds} | {v2_pass_str} | {v2_status} | {v1_pass_str} | {leace_str} | {sep_str} | {v2_task_str} | {v1_task_str} |")
        stdout.append(f"{ds:<10} {v2_pass_str:>14} {v2_status:>11} {v1_pass_str:>10} {leace_str:>8} "
                      f"{sep_str:>14} {v2_task_str:>10} {v1_task_str:>10}")

        overall_v2_pass += v2_pass
        overall_v2_pairs += v2_pairs
        overall_v1_pass += v1_pass

    stdout.append("=" * 110)
    md.append("")

    # Per-dataset verdict
    md.append("## Per-dataset verdict\n")
    for ds in DATASETS:
        loaded = load_v2(ds)
        if loaded is None:
            md.append(f"- **{ds}** — not yet run.")
            continue
        s, _source_tag = loaded
        v2_pass = s["pass_count_mean"]
        v1_pass = V1_PAPER[ds]["pass"]
        sep = SEPARATE[ds]

        if s["STATUS"] == "COLLAPSED":
            verdict = (
                "**COLLAPSED** — representation health failed at least one "
                "threshold (per_dim_std < 0.5 or effective_rank < 2.0). Pass "
                "count is degenerate, same failure mode as separate encoders "
                "on diabetes."
            )
        elif v2_pass >= v1_pass - 0.5:
            verdict = (
                f"v2 matches or beats v1 paper on compliance ({v2_pass:.2f} vs "
                f"{v1_pass}/{V1_PAPER[ds]['total']}) **without** the adversarial "
                "minimax loop. Healthy representations confirm this is real "
                f"compliance, not collapse (vs separate encoders {sep['pass']}/"
                f"{sep['total']} {sep['status']})."
            )
        elif v2_pass >= v1_pass - 1.5:
            verdict = (
                f"v2 close to v1 paper ({v2_pass:.2f} vs {v1_pass}/"
                f"{V1_PAPER[ds]['total']}) — within 1 pair. Direction looks "
                "right; would benefit from a tuning sweep on HSIC threshold or "
                "vCLUB/VICReg weights."
            )
        else:
            verdict = (
                f"v2 **regresses** vs v1 ({v2_pass:.2f} vs {v1_pass}/"
                f"{V1_PAPER[ds]['total']}). Constraint formulation may not be "
                "tight enough on this dataset, or vCLUB q-net is under-trained."
            )
        md.append(f"- **{ds}** — {verdict}")
    md.append("")

    if have_all:
        md.append("## Overall\n")
        md.append(f"- v2 total pass (mean over 3 seeds): {overall_v2_pass:.2f}/{overall_v2_pairs}")
        md.append(f"- v1 paper total pass: {overall_v1_pass}/{overall_v2_pairs}")
        md.append(f"- LEACE total pass: 0/20")
        md.append(
            f"- Separate encoders: 5/20 (all 5 from collapsed diabetes encoder; "
            f"healthy purposes pass 0/14)"
        )

        if overall_v2_pass >= overall_v1_pass - 1.0:
            md.append("- **Direction:** v2 matches or beats adversarial v1 PCRL on "
                      "compliance, with a non-adversarial constrained-optimisation "
                      "objective and frozen-backbone + LoRA architecture.")
        else:
            md.append("- **Direction:** v2 underperforms v1 paper on at least one "
                      "dataset; revisit per-dataset hyperparameters before declaring "
                      "the redesign successful.")
        md.append("")

    return "\n".join(md), "\n".join(stdout)


def main() -> None:
    md, stdout = render()
    out_path = ROOT / "results" / "V2_VALIDATION_SUMMARY.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(stdout)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
