#!/usr/bin/env python3
"""Emit a compact plain-text STATUS file for a cross-purpose AWS run.

Forked from ``scripts/emit_run_status.py`` (LAFTR pilot). The metrics for
the cross-purpose rebuttal are different: per-pair strict-R² compliance,
h_concat R² pass count (the cross-purpose constraint metric), and the
cross-purpose attack flag count by auditor architecture (LR/MLP/XGB) —
the metric §5.5 reports as 22-26/33 in the submitted paper.

This file is the durable headline. Even if every result blob is later wiped
by an S3 lifecycle rule, the rebuttal-relevant numbers survive as a small
text file that's cheap to archive and trivial to read months later.

Reads, for each dataset listed in ``--datasets``:
    results/v2_<dataset>_<tag>/per_seed_results.json   (training side)
    results/v2_<dataset>_<tag>/results.json            (eval side, optional)

The training side is required for the per-pair + health numbers. The eval
side is optional — if eval has not yet completed (e.g. instance hit hard
cap mid-eval), the h_concat + attack lines report "(eval pending)".

Usage:
    python scripts/emit_run_status_cross_purpose.py \\
        --instance-id i-0abc --tag CROSS_PURPOSE_AB \\
        --datasets adult hmda --out STATUS.txt
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
TAU_PER_PAIR = 0.05
TAU_CROSS = 0.10
ATTACK_DELTA_PP = 1.0


def _read_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _per_pair_line(per_seed_json: dict | None) -> str:
    if not per_seed_json:
        return "per-pair MISSING"
    per_seed = per_seed_json.get("per_seed", [])
    n_total = sum(len(s.get("attribute_results", [])) for s in per_seed)
    n_pass = sum(1 for s in per_seed for c in s.get("attribute_results", [])
                 if float(c.get("linear_r2", 1.0)) < TAU_PER_PAIR)
    return f"per-pair {n_pass}/{n_total}"


def _health_summary(per_seed_json: dict | None) -> str:
    if not per_seed_json:
        return "health=?"
    status = (per_seed_json.get("summary") or {}).get("STATUS", "?")
    return f"health={status}"


def _task_acc(per_seed_json: dict | None) -> str:
    if not per_seed_json:
        return "task_acc[?]"
    summ = per_seed_json.get("summary") or {}
    accs = summ.get("task_acc_mean") or {}
    if not accs:
        return "task_acc[?]"
    return "task_acc[" + ", ".join(f"{k}={v * 100:.1f}%" for k, v in accs.items()) + "]"


def _concat_line(eval_json: dict | None) -> str:
    if not eval_json:
        return "h_concat (eval pending)"
    summary = eval_json.get("summary") or {}
    concat = summary.get("concat_r2_mean") or {}
    if not concat:
        return "h_concat ?"
    n_pass = sum(v["n_pass"] for v in concat.values())
    n_total = sum(v["n"] for v in concat.values())
    detail = " ".join(f"{a}({v['n_pass']}/{v['n']})" for a, v in concat.items())
    return f"h_concat {n_pass}/{n_total} [{detail}]"


def _attack_line(eval_json: dict | None) -> str:
    if not eval_json:
        return "attack (eval pending)"
    summary = eval_json.get("summary") or {}
    matrix = summary.get("attack_per_arch_per_attr") or {}
    if not matrix:
        return "attack ?"
    by_arch = {}
    for arch, attrs in matrix.items():
        flagged = sum(1 for a in attrs if attrs[a].get("flag_1pp_mean"))
        total = len(attrs)
        by_arch[arch] = (flagged, total)
    overall = summary.get("attack_flagged_above_1pp", "?")
    detail = " ".join(f"{arch}({f}/{t})" for arch, (f, t) in by_arch.items())
    return f"attack {overall} [{detail}]"


def _dataset_line(root: Path, ds: str, tag: str) -> str:
    d = root / "results" / f"v2_{ds}_{tag}"
    per_seed = _read_json(d / "per_seed_results.json")
    eval_json = _read_json(d / "results.json")
    if per_seed is None and eval_json is None:
        return f"  {ds:10s} MISSING (no results dir)"
    parts = [
        _per_pair_line(per_seed),
        _concat_line(eval_json),
        _attack_line(eval_json),
        _health_summary(per_seed),
        _task_acc(per_seed),
    ]
    return f"  {ds:10s} " + "  ".join(parts)


def build_status(root: Path, instance_id: str, tag: str, datasets: list[str]) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "CROSS-PURPOSE REBUTTAL — RUN STATUS",
        "=" * 70,
        f"instance_id:     {instance_id}",
        f"completed_utc:   {now}",
        f"tag:             {tag}",
        f"thresholds:      per-pair R² < {TAU_PER_PAIR}, "
        f"h_concat R² ≤ {TAU_CROSS}, attack flag mean Δ > +{ATTACK_DELTA_PP:.1f}pp",
        "",
    ]
    for ds in datasets:
        lines.append(_dataset_line(root, ds, tag))
    lines += [
        "",
        "Totals are per-dataset cells across 3 seeds; attack totals are 3 archs × |attrs|.",
        "This STATUS is the durable headline. Full per-cell data (if retained) is in",
        f"results/v2_<dataset>_{tag}/results.json on the instance.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instance-id", default="unknown")
    p.add_argument("--tag", required=True,
                   help="Run tag, e.g. CROSS_PURPOSE_AB or CROSS_PURPOSE_DIABETES")
    p.add_argument("--datasets", nargs="+", required=True,
                   choices=["adult", "hmda", "diabetes"])
    p.add_argument("--root", default=str(DEFAULT_REPO_ROOT))
    p.add_argument("--out", default=None, help="Output path; prints to stdout if omitted")
    args = p.parse_args()

    text = build_status(Path(args.root).resolve(), args.instance_id,
                        args.tag, list(args.datasets))
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
