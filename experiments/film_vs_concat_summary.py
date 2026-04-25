#!/usr/bin/env python3
"""Produce the FiLM-vs-Concat summary table (results/film_vs_concat_summary.csv)
and print the interpretation to stdout.

Inputs:
  - HAR FiLM (existing):     results/har_real/har_seeds_fixed.csv  (method=PCRL rows)
  - HAR Concat (new):        results/har_real/concat_har_seeds.csv
  - CelebA FiLM reference:   results/celeba/film_v2matched_perpair.csv
                             (trained at the same v2 HPs as Concat; generated
                             alongside the Concat run by run_concat_celeba.py)
                             Fallback: results/celeba/celeba_perpair.csv
                             (the original v1 FiLM per-pair table).
  - CelebA Concat (new):     results/celeba/concat_perpair.csv

  - Adult (existing) reference numbers (from results/adult/conditioning_ablation_fixed.csv
    and sweep_report.md §4.2): FiLM 6/8, Concat 7/8.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import mean, stdev

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

RESULTS = project_root / "results"


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def f(x: str | float | None) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def har_stats(rows: list[dict], method_filter: str | None) -> dict:
    """Reduce a HAR per-pair CSV down to per-seed summary, then average."""
    if method_filter is not None:
        rows = [r for r in rows if r.get("method") == method_filter]
    per_seed: dict[int, dict] = {}
    for r in rows:
        seed = int(r["seed"])
        d = per_seed.setdefault(seed, {"activity": None, "subj_delta_max": -1, "pass_count": 0, "total_pairs": 0, "train_time": None})
        if d["activity"] is None:
            d["activity"] = f(r.get("activity_accuracy") or r.get("activity_acc"))
        if d["train_time"] is None:
            d["train_time"] = f(r.get("train_time_s"))
        d["pass_count"] = int(r.get("pass_count", 0))
        d["total_pairs"] = int(r.get("total_pairs", 0))
        if r.get("attribute") == "subject_id":
            delta = f(r.get("delta"))
            d["subj_delta_max"] = max(d["subj_delta_max"], delta)

    if not per_seed:
        return {}

    activity_vals = [d["activity"] for d in per_seed.values() if d["activity"] is not None]
    subj_vals = [d["subj_delta_max"] for d in per_seed.values() if d["subj_delta_max"] != -1]
    pass_vals = [d["pass_count"] for d in per_seed.values()]
    total = list(per_seed.values())[0]["total_pairs"]
    time_vals = [d["train_time"] for d in per_seed.values() if d["train_time"] is not None]
    n = len(per_seed)

    return {
        "n_seeds": n,
        "total_pairs": total,
        "activity_mean": mean(activity_vals) if activity_vals else float("nan"),
        "activity_std": stdev(activity_vals) if len(activity_vals) > 1 else 0.0,
        "subj_delta_mean": mean(subj_vals) if subj_vals else float("nan"),
        "subj_delta_std": stdev(subj_vals) if len(subj_vals) > 1 else 0.0,
        "pass_count_mean": mean(pass_vals) if pass_vals else 0,
        "pass_count_std": stdev(pass_vals) if len(pass_vals) > 1 else 0.0,
        "train_time_mean": mean(time_vals) if time_vals else float("nan"),
    }


def celeba_stats(rows: list[dict]) -> dict:
    pass_count = 0
    deltas = []
    r2s = []
    for r in rows:
        delta = f(r.get("delta"))
        r2 = f(r.get("linear_r2"))
        if delta < 0.02 and r2 < 0.05:
            pass_count += 1
        deltas.append(delta)
        r2s.append(r2)
    total = len(rows)
    return {
        "total_pairs": total,
        "pass_count": pass_count,
        "mean_delta": mean(deltas) if deltas else float("nan"),
        "max_delta": max(deltas) if deltas else float("nan"),
        "mean_r2": mean(r2s) if r2s else float("nan"),
    }


def main() -> None:
    # ── HAR ──────────────────────────────────────────────────────────────
    har_film_rows = load(RESULTS / "har_real" / "har_seeds_fixed.csv")
    har_concat_rows = load(RESULTS / "har_real" / "concat_har_seeds.csv")

    har_film = har_stats(har_film_rows, method_filter="PCRL")
    # Concat CSV is already filtered to PCRL-concat, no method column
    har_concat = har_stats(har_concat_rows, method_filter=None)

    # ── CelebA ───────────────────────────────────────────────────────────
    film_path = RESULTS / "celeba" / "film_v2matched_perpair.csv"
    if film_path.exists():
        film_src = "v2-matched (film_v2matched_perpair.csv — trained in same script as Concat)"
    elif (RESULTS / "celeba" / "baseline_v2_fixed.csv").exists():
        film_path = RESULTS / "celeba" / "baseline_v2_fixed.csv"
        film_src = "baseline_v2_fixed.csv (PCRL v2, shuffle-bug-fixed)"
    else:
        film_path = RESULTS / "celeba" / "celeba_perpair.csv"
        film_src = "v1 (celeba_perpair.csv — different HPs from v2, used as fallback)"

    celeba_film_rows = load(film_path)
    celeba_concat_rows = load(RESULTS / "celeba" / "concat_perpair.csv")

    # Normalize column names between the two celeba sources
    def norm(rows):
        out = []
        for r in rows:
            dd = dict(r)
            if "delta" not in dd and "delta" in r:
                pass
            # celeba_perpair.csv (v1) has columns: purpose, disallowed_attribute, delta, r_squared, pass
            if "disallowed_attribute" in r and "attribute" not in r:
                dd["attribute"] = r["disallowed_attribute"]
            if "r_squared" in r and "linear_r2" not in r:
                dd["linear_r2"] = r["r_squared"]
            out.append(dd)
        return out

    celeba_film_rows = norm(celeba_film_rows)
    celeba_concat_rows = norm(celeba_concat_rows)

    celeba_film = celeba_stats(celeba_film_rows)
    celeba_concat = celeba_stats(celeba_concat_rows)

    # ── Adult reference (from prior runs / paper) ────────────────────────
    adult_path = RESULTS / "adult" / "conditioning_ablation_fixed.csv"
    adult_film_pass = adult_concat_pass = adult_total = None
    adult_film_income = adult_concat_income = None
    if adult_path.exists():
        with open(adult_path) as f_:
            rows = list(csv.DictReader(f_))
        for cond in ("film", "concat"):
            sub = [r for r in rows if r.get("conditioning") == cond]
            if not sub:
                continue
            total = len(sub)
            pc = sum(1 for r in sub if r.get("adj_pass") in ("True", True, "true"))
            income = f(sub[0].get("income_acc"))
            if cond == "film":
                adult_film_pass, adult_film_income = pc, income
            else:
                adult_concat_pass, adult_concat_income = pc, income
            adult_total = total

    # ── Print summary ────────────────────────────────────────────────────
    def pct(x: float) -> str:
        try:
            return f"{x*100:.1f}%"
        except Exception:
            return "NA"

    print("=" * 78)
    print("FiLM vs Concat — cross-dataset comparison")
    print("=" * 78)
    print()
    print("ADULT  (from conditioning_ablation_fixed.csv, 1 seed, 8 pairs)")
    if adult_film_pass is not None:
        print(f"  FiLM:    pass {adult_film_pass}/{adult_total}, income={pct(adult_film_income)}")
        print(f"  Concat:  pass {adult_concat_pass}/{adult_total}, income={pct(adult_concat_income)}")
    else:
        print("  (adult data not available)")

    print()
    print("HAR    (3 seeds, 3 pairs per seed)")
    if har_film:
        print(f"  FiLM:    pass {har_film['pass_count_mean']:.1f}/{har_film['total_pairs']}  "
              f"activity_acc={pct(har_film['activity_mean'])} ± {pct(har_film['activity_std'])}  "
              f"subj_Δ={pct(har_film['subj_delta_mean'])} ± {pct(har_film['subj_delta_std'])}  "
              f"({har_film['train_time_mean']:.0f}s)")
    if har_concat:
        print(f"  Concat:  pass {har_concat['pass_count_mean']:.1f}/{har_concat['total_pairs']}  "
              f"activity_acc={pct(har_concat['activity_mean'])} ± {pct(har_concat['activity_std'])}  "
              f"subj_Δ={pct(har_concat['subj_delta_mean'])} ± {pct(har_concat['subj_delta_std'])}  "
              f"({har_concat['train_time_mean']:.0f}s)")

    print()
    print(f"CelebA (1 seed, 13 pairs)  [FiLM source: {film_src}]")
    print(f"  FiLM:    pass {celeba_film['pass_count']}/{celeba_film['total_pairs']}  "
          f"mean_Δ={pct(celeba_film['mean_delta'])} max_Δ={pct(celeba_film['max_delta'])} "
          f"mean_R²={celeba_film['mean_r2']:.3f}")
    print(f"  Concat:  pass {celeba_concat['pass_count']}/{celeba_concat['total_pairs']}  "
          f"mean_Δ={pct(celeba_concat['mean_delta'])} max_Δ={pct(celeba_concat['max_delta'])} "
          f"mean_R²={celeba_concat['mean_r2']:.3f}")

    # ── Write CSV ────────────────────────────────────────────────────────
    out = RESULTS / "film_vs_concat_summary.csv"
    rows_out = []
    if adult_film_pass is not None:
        rows_out.append({
            "dataset": "adult", "variant": "FiLM", "n_seeds": 1,
            "pass_count": adult_film_pass, "total_pairs": adult_total,
            "task_accuracy": adult_film_income, "task_metric": "income_acc",
            "subject_delta_mean": "",
        })
        rows_out.append({
            "dataset": "adult", "variant": "Concat", "n_seeds": 1,
            "pass_count": adult_concat_pass, "total_pairs": adult_total,
            "task_accuracy": adult_concat_income, "task_metric": "income_acc",
            "subject_delta_mean": "",
        })
    if har_film:
        rows_out.append({
            "dataset": "har", "variant": "FiLM", "n_seeds": har_film["n_seeds"],
            "pass_count": har_film["pass_count_mean"], "total_pairs": har_film["total_pairs"],
            "task_accuracy": har_film["activity_mean"], "task_metric": "activity_acc",
            "subject_delta_mean": har_film["subj_delta_mean"],
        })
    if har_concat:
        rows_out.append({
            "dataset": "har", "variant": "Concat", "n_seeds": har_concat["n_seeds"],
            "pass_count": har_concat["pass_count_mean"], "total_pairs": har_concat["total_pairs"],
            "task_accuracy": har_concat["activity_mean"], "task_metric": "activity_acc",
            "subject_delta_mean": har_concat["subj_delta_mean"],
        })
    rows_out.append({
        "dataset": "celeba", "variant": "FiLM", "n_seeds": 1,
        "pass_count": celeba_film["pass_count"], "total_pairs": celeba_film["total_pairs"],
        "task_accuracy": "", "task_metric": "per-pair",
        "subject_delta_mean": celeba_film["mean_delta"],
    })
    rows_out.append({
        "dataset": "celeba", "variant": "Concat", "n_seeds": 1,
        "pass_count": celeba_concat["pass_count"], "total_pairs": celeba_concat["total_pairs"],
        "task_accuracy": "", "task_metric": "per-pair",
        "subject_delta_mean": celeba_concat["mean_delta"],
    })
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nSaved {out}")

    # ── Interpretation ───────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("INTERPRETATION")
    print("=" * 78)

    adult_concat_wins = (adult_concat_pass or 0) > (adult_film_pass or 0)
    har_film_wins = (har_film or {}).get("pass_count_mean", 0) > (har_concat or {}).get("pass_count_mean", 0)
    celeba_film_wins = celeba_film["pass_count"] > celeba_concat["pass_count"]

    har_concat_activity = (har_concat or {}).get("activity_mean", 0)
    har_film_activity = (har_film or {}).get("activity_mean", 0)
    task_collapse = (har_concat or {}) and har_concat_activity < 0.30  # ~chance ≈ 16.7%

    if task_collapse:
        print("✗ HAR Concat task-collapses (activity_acc << FiLM's).")
        print("  STRONG evidence FiLM is needed for utility preservation under severe conflicts.")
    elif har_film_wins and celeba_film_wins and adult_concat_wins:
        print("✓ Conjecture supported: Concat helps under mild conflict (Adult),")
        print("  FiLM required under severe conflict (HAR, CelebA).")
        print("  Paper's §4.2 story: \"mild conflict → Concat OK; severe conflict → use FiLM\"")
    elif not har_film_wins and not celeba_film_wins:
        print("✗ Concat still competitive/winning everywhere.")
        print("  Paper should treat Concat as an equally valid conditioning option.")
    else:
        print("~ Mixed evidence. Summary table above shows the numbers.")

    if har_film:
        print(f"\n  HAR: FiLM pass={har_film['pass_count_mean']:.1f}, Concat pass={har_concat['pass_count_mean']:.1f}  "
              f"(FiLM {'wins' if har_film_wins else 'loses/ties'})")
    print(f"  CelebA: FiLM pass={celeba_film['pass_count']}, Concat pass={celeba_concat['pass_count']}  "
          f"(FiLM {'wins' if celeba_film_wins else 'loses/ties'})")
    if adult_film_pass is not None:
        print(f"  Adult: FiLM pass={adult_film_pass}, Concat pass={adult_concat_pass}  "
              f"(Concat {'wins' if adult_concat_wins else 'loses/ties'})")

    if har_film and har_concat:
        print(f"\n  Activity accuracy:  FiLM={pct(har_film_activity)}, Concat={pct(har_concat_activity)}  "
              f"(gap={pct(har_film_activity - har_concat_activity)})")


if __name__ == "__main__":
    main()
