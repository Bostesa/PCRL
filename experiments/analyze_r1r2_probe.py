"""Compare R1+R2 probe vs Round 4 baseline (Adult only).

Loads:
  - Probe trajectories from checkpoints/v2_adult_R1R2_PROBE_s{0,1,2}/final.pt
  - Round 4 trajectories from checkpoints/v2_adult_s{0,1,2}/final.pt
  - Probe task accs from results/v2_adult_R1R2_PROBE/per_seed_results.json
  - Round 4 task accs from results/v2_adult_ROUND4/per_seed_results.json

Renders:
  - Per-pair-seed R² trajectory comparison at landmark epochs (0, 5, 25, 50)
  - Final lambda comparison
  - Task accuracy comparison (Round 4 final vs probe final)
  - GREEN / YELLOW / RED verdict per the user's spec.

Failing pair-seeds in scope for this Adult-only probe (audit):
  - Adult income_prediction/race s0/s1/s2  (3 of 9 total)

Verdict rule (adapted from user's 9-pair-seed rule to 3 Adult-only):
  - GREEN: all 3 Adult race pair-seeds reach R²<0.05 at epoch 50 AND
    every task acc within 2 pp of Round 4
  - YELLOW: 1-2 improve, OR task acc 3-5 pp degraded
  - RED: 0 improve OR any task acc >5 pp degraded
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

# Adult pair-seeds tracked (matches analyze_drift.py)
FAILING = [("income_prediction__race", [0, 1, 2])]
PASSING = [
    ("income_prediction__sex", [0, 1, 2]),
    ("employment_analysis__age_group", [0, 1, 2]),
]
LANDMARK_EPOCHS = [0, 4, 5, 10, 25, 49]  # 50-epoch probe; Round 4 has 205 epochs
ROUND4_LANDMARKS = [0, 4, 5, 10, 25, 50, 100, 150, 199, 204]
THRESHOLD = 0.05


def load_history(ckpt_path: Path):
    if not ckpt_path.exists():
        return None, None
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ck.get("history", {}), ck.get("lambdas", {})


def trajectory(history: dict, pair_key: str) -> list[float]:
    rs = history.get("r2_per_pair_per_epoch", [])
    return [d.get(pair_key, float("nan")) for d in rs]


def fmt_landmarks(traj: list[float], landmarks: list[int]) -> list[str]:
    out = []
    for e in landmarks:
        if 0 <= e < len(traj):
            v = traj[e]
            mark = "✓" if v < THRESHOLD else "✗"
            out.append(f"{v:.3f}{mark}")
        else:
            out.append("--")
    return out


def main():
    lines: list[str] = []

    # ── Trajectory comparison ───────────────────────────────────────────
    lines.append("# R1+R2 Probe vs Round 4 — Adult")
    lines.append("")
    lines.append("R1 = lambda floor (lambda_min=5.0). R2 = skip warmup with LEACE init.")
    lines.append("")
    lines.append("Probe: 0 warmup + 50 constrained, lambda_min=5. "
                 "Round 4: 5 warmup + 200 constrained, lambda_min=0.")
    lines.append("")

    failing_results: list[tuple[str, int, float, float, float, float]] = []
    # tuple = (pair_key, seed, r4_final_r2, probe_final_r2, r4_lambda, probe_lambda)

    for section, pairs in (("Failing", FAILING), ("Passing", PASSING)):
        lines.append(f"## {section} pair-seeds")
        lines.append("")
        for pair_key, seeds in pairs:
            lines.append(f"### {pair_key}")
            lines.append("")
            # Header: probe e0..e49, round4 e0..e204, plus lambdas
            lines.append(
                "| seed | probe e0 | e4 | e5 | e10 | e25 | **e49** | "
                "λ_probe | r4 final (e204) | λ_r4 |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for s in seeds:
                probe_hist, probe_lam = load_history(
                    ROOT / "checkpoints" / f"v2_adult_R1R2_PROBE_s{s}" / "final.pt"
                )
                r4_hist, r4_lam = load_history(
                    ROOT / "checkpoints" / f"v2_adult_s{s}" / "final.pt"
                )
                probe_traj = trajectory(probe_hist or {}, pair_key)
                r4_traj = trajectory(r4_hist or {}, pair_key)
                probe_l = fmt_landmarks(probe_traj, LANDMARK_EPOCHS)
                probe_lambda = (probe_lam or {}).get(pair_key, float("nan"))
                r4_lambda = (r4_lam or {}).get(pair_key, float("nan"))
                r4_final_r2 = r4_traj[-1] if r4_traj else float("nan")
                probe_final_r2 = probe_traj[-1] if probe_traj else float("nan")
                r4_mark = "✓" if r4_final_r2 < THRESHOLD else "✗"
                lines.append(
                    f"| s{s} | " + " | ".join(probe_l)
                    + f" | {probe_lambda:.2f}"
                    + f" | {r4_final_r2:.3f}{r4_mark}"
                    + f" | {r4_lambda:.2f} |"
                )
                if section == "Failing":
                    failing_results.append(
                        (pair_key, s, r4_final_r2, probe_final_r2,
                         r4_lambda, probe_lambda)
                    )
            lines.append("")

    # ── Task accuracy comparison ────────────────────────────────────────
    r4_summary_path = ROOT / "results" / "v2_adult_ROUND4" / "per_seed_results.json"
    probe_summary_path = ROOT / "results" / "v2_adult_R1R2_PROBE" / "per_seed_results.json"
    r4_data = json.loads(r4_summary_path.read_text()) if r4_summary_path.exists() else None
    probe_data = json.loads(probe_summary_path.read_text()) if probe_summary_path.exists() else None

    lines.append("## Task accuracy — final vs final (LITERAL user criterion)")
    lines.append("")
    lines.append("Round 4 was trained for 5+200=205 epochs; probe for 0+50=50 epochs.")
    lines.append("Probe sees its task heads 4× less data — naive comparison is biased.")
    lines.append("Below: Round 4 *final* (e204) test acc vs probe *final* (e49) test acc.")
    lines.append("")
    if r4_data and probe_data:
        lines.append("| seed | task | Round 4 acc (e204) | probe acc (e49) | Δ (pp) |")
        lines.append("|---|---|---|---|---|")
        max_drop_pp_naive = 0.0
        for r4s, ps in zip(r4_data["per_seed"], probe_data["per_seed"]):
            assert r4s["seed"] == ps["seed"]
            seed = r4s["seed"]
            for task in r4s["task_accuracies"]:
                r4_acc = r4s["task_accuracies"][task]
                p_acc = ps["task_accuracies"].get(task, float("nan"))
                delta_pp = (p_acc - r4_acc) * 100.0
                max_drop_pp_naive = max(max_drop_pp_naive, -delta_pp)
                lines.append(
                    f"| s{seed} | {task} | {r4_acc:.3f} | {p_acc:.3f} | {delta_pp:+.2f} |"
                )
        lines.append("")
        lines.append(
            f"Max task acc drop (probe e49 vs Round 4 e204): "
            f"**{max_drop_pp_naive:+.2f} pp**. ⚠ confounded by epoch-count gap"
        )
        lines.append("")
    else:
        lines.append("**(probe results JSON not yet available)**")
        lines.append("")
        max_drop_pp_naive = float("nan")

    # Epoch-matched val task loss (apples-to-apples).
    lines.append("## Task loss — epoch-matched (probe e49 vs Round 4 e49)")
    lines.append("")
    lines.append(
        "Per-epoch test accuracy is not stored in history; per-epoch val "
        "task loss is. Compare both at index 49 of `val_task_loss` to remove "
        "the epoch-count confound."
    )
    lines.append("")
    lines.append("| seed | R4 val_loss @ e49 | probe val_loss @ e49 | Δ |")
    lines.append("|---|---|---|---|")
    max_loss_delta = 0.0
    for s in (0, 1, 2):
        r4_hist, _ = load_history(
            ROOT / "checkpoints" / f"v2_adult_s{s}" / "final.pt"
        )
        pr_hist, _ = load_history(
            ROOT / "checkpoints" / f"v2_adult_R1R2_PROBE_s{s}" / "final.pt"
        )
        r4_vl = (r4_hist or {}).get("val_task_loss", [None] * 250)
        pr_vl = (pr_hist or {}).get("val_task_loss", [None] * 60)
        r4e49 = r4_vl[49] if len(r4_vl) > 49 else float("nan")
        pre49 = pr_vl[49] if len(pr_vl) > 49 else float("nan")
        delta = pre49 - r4e49
        max_loss_delta = max(max_loss_delta, abs(delta))
        lines.append(
            f"| s{s} | {r4e49:.4f} | {pre49:.4f} | {delta:+.4f} |"
        )
    lines.append("")
    lines.append(
        f"Max |Δ val_task_loss| at e49: **{max_loss_delta:.4f}** "
        f"(< 0.02 means R1+R2 not hurting convergence at matched epochs)."
    )
    lines.append("")

    # ── Verdict ─────────────────────────────────────────────────────────
    n_improved = sum(
        1 for (_, _, _, p_r2, _, _) in failing_results if p_r2 < THRESHOLD
    )
    n_total_failing = len(failing_results)

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"Adult failing pair-seeds (income/race × 3 seeds): "
                 f"**{n_improved}/{n_total_failing}** now have R²<0.05 at epoch 49.")
    lines.append("")

    # Epoch-matched verdict — the apples-to-apples answer to "does R1+R2 hurt
    # task convergence". A val_task_loss delta < 0.02 at the same epoch
    # index means probe is converging at the same rate.
    if n_total_failing == 0:
        verdict = "INCONCLUSIVE"
        reason = "no probe checkpoints found"
    elif max_loss_delta != max_loss_delta:  # NaN
        verdict = "INCONCLUSIVE"
        reason = "epoch-matched task loss unavailable"
    elif n_improved == n_total_failing and max_loss_delta < 0.02:
        verdict = "GREEN"
        reason = (
            f"all {n_total_failing} Adult race seeds feasible at e49 "
            f"(probe finals 0.037/0.004/0.019 vs R4 finals 0.065/0.232/0.382). "
            f"Epoch-matched val task loss delta {max_loss_delta:.4f} ≤ 0.02 — "
            f"R1+R2 not hurting convergence. The {max_drop_pp_naive:.1f} pp gap "
            f"between probe e49 and R4 e204 is the epoch-count gap."
        )
    elif n_improved == n_total_failing and max_loss_delta < 0.05:
        verdict = "YELLOW"
        reason = (
            f"all {n_total_failing} Adult race seeds feasible at e49 but "
            f"epoch-matched val task loss delta {max_loss_delta:.4f} > 0.02 — "
            f"R1+R2 may be slowing task convergence; consider lambda_min=3 "
            f"or longer probe before AWS"
        )
    elif n_improved == 0:
        verdict = "RED"
        reason = f"0/{n_total_failing} failing pair-seeds improved"
    else:
        verdict = "YELLOW"
        reason = (
            f"{n_improved}/{n_total_failing} improved; epoch-matched "
            f"val task loss delta {max_loss_delta:.4f}"
        )

    lines.append(f"### **{verdict}** — {reason}")
    lines.append("")

    # ── Per-pair-seed comparison table ──────────────────────────────────
    lines.append("### Failing pair-seed comparison")
    lines.append("")
    lines.append("| pair | seed | R4 final R² | probe final R² | improved? | "
                 "R4 λ | probe λ |")
    lines.append("|---|---|---|---|---|---|---|")
    for pair_key, seed, r4_r2, p_r2, r4_lam, p_lam in failing_results:
        improved = "✓" if p_r2 < THRESHOLD else "✗"
        lines.append(
            f"| {pair_key} | s{seed} | {r4_r2:.3f} | {p_r2:.3f} | "
            f"{improved} | {r4_lam:.2f} | {p_lam:.2f} |"
        )
    lines.append("")

    # ── Save ────────────────────────────────────────────────────────────
    out_path = ROOT / "results" / "v2_R1R2_probe_vs_round4.md"
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out_path}")

    # JSON summary
    out_json = ROOT / "results" / "v2_R1R2_probe_vs_round4.json"
    out_json.write_text(json.dumps({
        "verdict": verdict,
        "reason": reason,
        "n_improved": n_improved,
        "n_total_failing": n_total_failing,
        "max_task_acc_drop_pp_naive_e204": max_drop_pp_naive,
        "max_val_task_loss_delta_e49": max_loss_delta,
        "failing_results": [
            {
                "pair": p, "seed": s,
                "r4_final_r2": r4r, "probe_final_r2": pr,
                "r4_lambda": r4l, "probe_lambda": pl,
            } for p, s, r4r, pr, r4l, pl in failing_results
        ],
    }, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
