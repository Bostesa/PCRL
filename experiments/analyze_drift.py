"""Analyze Round 4 optimizer drift on Adult + HMDA checkpoints.

Pulls per-epoch trajectories from checkpoints/v2_{adult,hmda}_s{0,1,2}/final.pt
and dumps:
  - Trajectory tables for failing & passing pairs
  - Drift classification (LATE / EARLY / OSCILLATION / SATURATION / STARVATION)
  - Final lambdas
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

# Failing pairs to investigate (from Round 4 final_vs_best.md)
FAILING = {
    "adult": [
        ("income_prediction__race", [0, 1, 2]),
    ],
    "hmda": [
        ("underwriting__race", [0, 1, 2]),
        ("pricing_analysis__race", [0, 1, 2]),
    ],
}

# Passing pairs for comparison
PASSING = {
    "adult": [
        ("income_prediction__sex", [0, 1, 2]),
        ("employment_analysis__age_group", [0, 1, 2]),
    ],
    "hmda": [
        ("pricing_analysis__sex", [0, 1, 2]),
        ("fair_lending_audit__sex", [0, 1, 2]),
    ],
}

LANDMARK_EPOCHS = [0, 4, 5, 10, 25, 50, 100, 150, 199, 204]
THRESHOLD = 0.05


def load_history(dataset: str, seed: int):
    p = ROOT / "checkpoints" / f"v2_{dataset}_s{seed}" / "final.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return ck.get("history", {}), ck.get("lambdas", {})


def trajectory(history: dict, pair_key: str) -> list[float]:
    rs = history.get("r2_per_pair_per_epoch", [])
    return [d.get(pair_key, float("nan")) for d in rs]


def classify(traj: list[float], lam_final: float, threshold: float = 0.05,
             cap: float = 1000.0) -> str:
    """Return one of LATE / EARLY / OSCILL / SATURATE / STARVE / FEASIBLE."""
    n = len(traj)
    if n < 50:
        return "INSUFFICIENT_HISTORY"

    # Get post-warmup trajectory only (warmup is first 5 epochs)
    constr = traj[5:]
    final_r2 = traj[-1] if traj else float("nan")
    final_in = final_r2 < threshold

    if final_in:
        return "FEASIBLE"

    # Lambda saturation?
    if lam_final >= 0.95 * cap:
        return "SATURATE"
    # Lambda starvation: lambda small and r2 high
    if lam_final < 1.0 and final_r2 > threshold * 1.5:
        return "STARVE"

    # Oscillation: count threshold crossings in second half
    second = constr[len(constr) // 2:]
    crossings = sum(
        1 for a, b in zip(second[:-1], second[1:])
        if (a < threshold) != (b < threshold)
    )
    if crossings >= 5:
        return "OSCILL"

    # Late vs early drift
    early = traj[5:50]   # epochs 5..49
    late = traj[150:]    # epochs 150..end
    early_min = min(early) if early else float("nan")
    late_max = max(late) if late else float("nan")

    # If R² became feasible by epoch 50 then drifted up later → LATE
    if not math.isnan(early_min) and early_min < threshold and late_max > threshold:
        return "LATE"
    # If R² rose quickly post-warmup and never came down → EARLY
    if not math.isnan(early[-1] if early else float("nan")) and early[-1] > threshold * 2:
        return "EARLY"
    return "OTHER"


def fmt_row(seed: int, traj: list[float], lam: float, klass: str) -> list[str]:
    cells = [f"s{seed}"]
    for e in LANDMARK_EPOCHS:
        if e < len(traj):
            v = traj[e]
            mark = "✓" if v < THRESHOLD else "✗"
            cells.append(f"{v:.3f}{mark}")
        else:
            cells.append("--")
    cells.append(f"{lam:.2f}")
    cells.append(klass)
    return cells


def render_table(title: str, rows: list[list[str]]) -> list[str]:
    header = ["seed"] + [f"e{e}" for e in LANDMARK_EPOCHS] + ["λ_final", "class"]
    lines = [f"### {title}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")
    return lines


def analyze_dataset(dataset: str, pairs_def: dict) -> tuple[list[str], dict]:
    out_lines = [f"## {dataset.upper()}"]
    classifications = {}
    for pair_key, seeds in pairs_def[dataset]:
        rows = []
        for s in seeds:
            hist, lam_dict = load_history(dataset, s)
            traj = trajectory(hist, pair_key)
            lam = lam_dict.get(pair_key, float("nan"))
            klass = classify(traj, lam)
            classifications[(dataset, pair_key, s)] = (klass, traj, lam)
            rows.append(fmt_row(s, traj, lam, klass))
        out_lines.extend(render_table(pair_key, rows))
    return out_lines, classifications


def main():
    out = ["# V2 Round 4 Optimizer Drift Audit", ""]
    out.append("Trajectory R²(z, attr) per landmark epoch from "
               "`history.r2_per_pair_per_epoch`. Threshold 0.05 (✓ pass / ✗ fail). "
               "λ_final from `ckpt['lambdas']`.")
    out.append("")
    out.append(f"Landmark epochs: {LANDMARK_EPOCHS} (warmup ends at epoch 4).")
    out.append("")
    out.append("Drift classes: **LATE** = feasible by ep50 then drifts up; "
               "**EARLY** = rises quickly after warmup, never comes down; "
               "**OSCILL** = ≥5 threshold crossings in second half; "
               "**SATURATE** = λ ≥ 0.95·cap and R² > threshold; "
               "**STARVE** = λ < 1.0 and R² > 1.5·threshold; "
               "**FEASIBLE** = final R² < threshold.")
    out.append("")

    cls_all = {}
    out.append("# FAILING PAIRS")
    for ds in ("adult", "hmda"):
        lines, cls = analyze_dataset(ds, FAILING)
        out.extend(lines)
        cls_all.update(cls)

    out.append("# PASSING PAIRS (comparison)")
    for ds in ("adult", "hmda"):
        lines, cls = analyze_dataset(ds, PASSING)
        out.extend(lines)
        cls_all.update(cls)

    # Aggregate classification
    out.append("# Classification summary")
    out.append("")
    failing_classes = [
        v[0] for k, v in cls_all.items()
        if (k[0], k[1]) in [(d, p) for d, lst in FAILING.items() for p, _ in lst]
    ]
    from collections import Counter
    out.append(f"Failing pairs (n={len(failing_classes)}): "
               f"{dict(Counter(failing_classes))}")
    out.append("")

    # Save
    path = ROOT / "results" / "v2_optimizer_drift_audit.md"
    path.write_text("\n".join(out))
    print("\n".join(out))
    print(f"\nWrote {path}")

    # Also dump raw classifications + final lambdas
    raw = {
        f"{k[0]}/{k[1]}/s{k[2]}": {
            "class": v[0],
            "lambda_final": v[2],
            "trajectory_landmarks": {
                f"e{e}": (v[1][e] if e < len(v[1]) else None)
                for e in LANDMARK_EPOCHS
            },
        }
        for k, v in cls_all.items()
    }
    raw_path = ROOT / "results" / "v2_optimizer_drift_audit.json"
    raw_path.write_text(json.dumps(raw, indent=2))
    print(f"Wrote {raw_path}")


if __name__ == "__main__":
    main()
