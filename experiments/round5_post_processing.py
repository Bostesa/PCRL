"""Post-processing for V2 Round 5.

Polls /tmp/round5/ALL_PULLED. When the orchestrator marks all 3 tarballs
pulled, runs:
  1. eval_round4_final_vs_best_v2.py per dataset → final_vs_best.{md,json}
  2. Framework D dominant-axis eval per dataset (scripts/eval_round4_dominant_axis.py
     if it supports a tag arg, else copy/inline the per-checkpoint pattern).
  3. Aggregates into results/V2_ROUND5_SUMMARY.md
  4. Classifies into ALL-GREEN / PARTIAL / NO-CHANGE / WORSE
  5. Writes results/V2_ROUND5_VERDICT.md
  6. Commits results to origin/main if not WORSE

Hard cap: 16 hours from start.

Designed to be launched detached after the orchestrator. It owns post-
training cleanup and writes the verdict the user reads in the morning.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path("/tmp/round5")
LOG_FILE = STATE_DIR / "post.log"

DATASETS = ("adult", "hmda", "diabetes")
HARD_CAP_SEC = 16 * 3600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("r5post")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log.info(f"$ {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), **kw)
    if p.stdout:
        for line in p.stdout.splitlines()[-30:]:
            log.info(f"  out| {line}")
    if p.returncode != 0:
        for line in p.stderr.splitlines()[-30:]:
            log.error(f"  err| {line}")
    return p


def wait_for_all_pulled() -> bool:
    start = time.time()
    while True:
        if (STATE_DIR / "ALL_PULLED").exists():
            return True
        if time.time() - start > HARD_CAP_SEC:
            log.error("HARD CAP reached waiting for ALL_PULLED")
            return False
        time.sleep(120)


def parse_round4_metrics(dataset: str) -> dict:
    """Pull Round 4 metrics for the comparison table."""
    p = ROOT / "results" / f"v2_{dataset}_ROUND4" / "final_vs_best.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def parse_round5_metrics(dataset: str) -> dict:
    p = ROOT / "results" / f"v2_{dataset}_ROUND5" / "final_vs_best.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def run_evals() -> dict:
    """Run final_vs_best v2 evaluator on each Round 5 dataset.

    Returns per-dataset eval result dicts."""
    results: dict[str, dict] = {}
    for ds in DATASETS:
        log.info(f"=== {ds.upper()} eval (final vs best) ===")
        out_dir = ROOT / "results" / f"v2_{ds}_ROUND5"
        out_dir.mkdir(parents=True, exist_ok=True)
        p = _run([
            sys.executable, "experiments/eval_round4_final_vs_best_v2.py",
            "--dataset", ds, "--out-tag", "_ROUND5",
        ])
        results[ds] = {"eval_rc": p.returncode}
    return results


def run_dominant_axis() -> dict:
    """Try the dominant-axis script with --tag ROUND5; if it doesn't accept
    that flag, log and skip (Framework D in-progress per user constraint)."""
    out: dict[str, dict] = {}
    da_script = ROOT / "scripts" / "eval_round4_dominant_axis.py"
    if not da_script.exists():
        log.warning("dominant-axis script missing; skipping Framework D eval")
        return {}
    # Probe support for --tag
    p = _run([sys.executable, str(da_script), "--help"])
    accepts_tag = "--tag" in p.stdout or "--tag" in p.stderr
    if not accepts_tag:
        log.warning("dominant-axis script does not accept --tag; "
                    "skipping Framework D eval to avoid clobbering Round 4")
        return {"skipped": True, "reason": "script does not accept --tag"}
    for ds in DATASETS:
        log.info(f"=== {ds.upper()} Framework D dominant-axis ===")
        p = _run([sys.executable, str(da_script), "--tag", "ROUND5"])
        out[ds] = {"da_rc": p.returncode}
    return out


def write_summary(eval_results: dict, da_results: dict) -> None:
    lines = ["# V2 Round 5 Summary", ""]
    lines.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    lines.append("")
    lines.append("R1 (lambda floor=5) + R2 (skip warmup with LEACE init) "
                 "+ Diabetes rank-16 LoRA. 3 seeds × 200 epochs × 3 datasets "
                 "on AWS g4dn.xlarge. See `results/v2_optimizer_drift_audit.md` "
                 "and `results/v2_R1R2_probe_vs_round4.md` for the audit and "
                 "local probe that motivated this run.")
    lines.append("")

    lines.append("## Headline table")
    lines.append("")
    lines.append("| Dataset | R4 mean R²_onehot (final.pt) | R5 mean R²_onehot (final.pt) | "
                 "R4 strict (τ=0.05) | R5 strict | R4 R²_DA | R5 R²_DA |")
    lines.append("|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        r4 = parse_round4_metrics(ds)
        r5 = parse_round5_metrics(ds)
        r4_mean = _mean_r2_final(r4)
        r5_mean = _mean_r2_final(r5)
        r4_strict = _strict_pass_final(r4)
        r5_strict = _strict_pass_final(r5)
        lines.append(
            f"| {ds} | {r4_mean} | {r5_mean} | {r4_strict} | {r5_strict} "
            f"| (R4 DA — see Round 4 dominant-axis) | "
            f"{'(skipped)' if da_results.get('skipped') else 'pending'} |"
        )
    lines.append("")
    lines.append("## Failing pair-seed comparison (R4 → R5)")
    lines.append("")
    failing_pairs = [
        ("adult", "income_prediction__race"),
        ("hmda", "underwriting__race"),
        ("hmda", "pricing_analysis__race"),
        ("diabetes", "quality_research__age_bucket"),
        ("diabetes", "quality_research__race"),
    ]
    lines.append("| Dataset | Pair | R4 s0/s1/s2 | R5 s0/s1/s2 |")
    lines.append("|---|---|---|---|")
    for ds, pair in failing_pairs:
        r4_vals = _per_seed_r2_final(parse_round4_metrics(ds), pair)
        r5_vals = _per_seed_r2_final(parse_round5_metrics(ds), pair)
        lines.append(f"| {ds} | {pair} | {r4_vals} | {r5_vals} |")
    lines.append("")

    (ROOT / "results" / "V2_ROUND5_SUMMARY.md").write_text("\n".join(lines))
    log.info(f"wrote results/V2_ROUND5_SUMMARY.md")


def _mean_r2_final(d: dict) -> str:
    if not d:
        return "—"
    rs = [seed.get("final", {}).get("mean_linear_r2") for seed in d.get("per_seed", {}).values()]
    rs = [r for r in rs if r is not None]
    if not rs:
        return "—"
    return f"{sum(rs)/len(rs):.3f}"


def _strict_pass_final(d: dict) -> str:
    if not d:
        return "—"
    pairs_per_seed = d.get("pairs_per_seed", 0)
    seeds = d.get("per_seed", {})
    total = pairs_per_seed * len(seeds)
    passed = sum(seed.get("final", {}).get("pass_count", 0) for seed in seeds.values())
    return f"{passed}/{total}"


def _per_seed_r2_final(d: dict, pair: str) -> str:
    if not d:
        return "— / — / —"
    seeds = d.get("per_seed", {})
    parts = []
    for s in (0, 1, 2):
        rows = (seeds.get(str(s)) or seeds.get(s, {})).get("final", {}).get("rows", [])
        match = next((r for r in rows if f"{r.get('purpose','')}__{r.get('attribute','')}" == pair), None)
        if match is None:
            parts.append("—")
        else:
            tick = "✓" if match.get("linear_r2", 1.0) < 0.05 else "✗"
            parts.append(f"{match['linear_r2']:.3f}{tick}")
    return " / ".join(parts)


def classify_verdict() -> tuple[str, str]:
    """Apply the user's classification rules over Round 5 strict counts."""
    rules = {
        "adult": (22, 24),    # >= 22/24
        "hmda": (14, 18),     # >= 14/18
        "diabetes": (17, 18), # >= 17/18
    }
    statuses = []
    for ds, (need, total) in rules.items():
        r5 = parse_round5_metrics(ds)
        r4 = parse_round4_metrics(ds)
        seeds_r5 = r5.get("per_seed", {})
        seeds_r4 = r4.get("per_seed", {})
        passed_r5 = sum(s.get("final", {}).get("pass_count", 0) for s in seeds_r5.values())
        passed_r4 = sum(s.get("final", {}).get("pass_count", 0) for s in seeds_r4.values())
        if passed_r5 >= need:
            statuses.append("GREEN")
        elif passed_r5 + 2 < passed_r4:
            statuses.append("WORSE")
        elif passed_r5 > passed_r4:
            statuses.append("PARTIAL")
        else:
            statuses.append("NO_CHANGE")
    if "WORSE" in statuses:
        return "WORSE", f"per-dataset: {dict(zip(rules, statuses))}"
    if all(s == "GREEN" for s in statuses):
        return "ALL-GREEN", f"per-dataset: {dict(zip(rules, statuses))}"
    if any(s == "GREEN" or s == "PARTIAL" for s in statuses):
        return "PARTIAL", f"per-dataset: {dict(zip(rules, statuses))}"
    return "NO-CHANGE", f"per-dataset: {dict(zip(rules, statuses))}"


def write_verdict(verdict: str, reason: str, eval_rc: dict, da_rc: dict) -> None:
    lines = ["# V2 Round 5 Verdict", ""]
    lines.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    lines.append("")
    lines.append(f"## **{verdict}**")
    lines.append("")
    lines.append(reason)
    lines.append("")
    lines.append("## AWS instance status (post-stop)")
    lines.append("")
    lines.append("All 3 instances should be `stopped` by their on-instance watchdog.")
    lines.append("")
    lines.append("## Eval status")
    lines.append("")
    for ds, r in eval_rc.items():
        lines.append(f"- final_vs_best ({ds}): rc={r.get('eval_rc')}")
    lines.append("")
    if da_rc.get("skipped"):
        lines.append(f"- Framework D dominant-axis: skipped — {da_rc.get('reason')}")
    else:
        for ds, r in (da_rc or {}).items():
            lines.append(f"- dominant-axis ({ds}): rc={r.get('da_rc')}")
    lines.append("")
    lines.append("## Cost estimate")
    lines.append("")
    lines.append("3 × g4dn.xlarge × ~3h ≈ \\$10")
    lines.append("")
    lines.append("## Next action")
    lines.append("")
    if verdict == "ALL-GREEN":
        lines.append("Round 5 met all dataset thresholds. Suggested next: update paper "
                     "main results table with Round 5 numbers; commit `V2_ROUND5_SUMMARY.md` "
                     "+ `V2_ROUND5_VERDICT.md`.")
    elif verdict == "PARTIAL":
        lines.append("Some datasets improved, others stayed at Round 4 levels. Inspect "
                     "per-dataset breakdown in `V2_ROUND5_SUMMARY.md` to decide whether "
                     "to ship as-is or iterate on the laggards.")
    elif verdict == "NO-CHANGE":
        lines.append("R1+R2 didn't move the needle. Consider R3 (periodic LEACE re-anchor) "
                     "or accepting Round 4 numbers in the paper.")
    else:
        lines.append("**WORSE on at least one dataset — DO NOT push further commits.** "
                     "Inspect `V2_ROUND5_SUMMARY.md` and revert if needed.")
    lines.append("")
    lines.append("## Pointers")
    lines.append("")
    lines.append("- `results/V2_ROUND5_SUMMARY.md` — full table + per-pair-seed comparison")
    lines.append("- `results/v2_{adult,hmda,diabetes}_ROUND5/per_seed_results.json` — raw")
    lines.append("- `checkpoints/v2_{adult,hmda,diabetes}_ROUND5_s{0,1,2}/{final,best}.pt`")
    lines.append("- `/tmp/round5/orchestrator.log` — orchestrator history")
    lines.append("- `/tmp/round5/post.log` — post-processing history")
    lines.append("")
    (ROOT / "results" / "V2_ROUND5_VERDICT.md").write_text("\n".join(lines))
    log.info("wrote results/V2_ROUND5_VERDICT.md")


def maybe_commit(verdict: str) -> None:
    if verdict == "WORSE":
        log.warning("verdict=WORSE — NOT committing further per user spec")
        return
    log.info(f"verdict={verdict} — committing Round 5 results")
    p = _run([
        "git", "add",
        "results/V2_ROUND5_SUMMARY.md",
        "results/V2_ROUND5_VERDICT.md",
        "results/v2_adult_ROUND5",
        "results/v2_hmda_ROUND5",
        "results/v2_diabetes_ROUND5",
    ])
    if p.returncode != 0:
        log.error(f"git add failed; aborting commit")
        return
    msg = f"V2 Round 5 results: {verdict}\n\nSee results/V2_ROUND5_VERDICT.md."
    p = _run(["git", "commit", "-m", msg])
    if p.returncode != 0:
        log.error("git commit failed")
        return
    p = _run(["git", "push", "origin", "main"])
    if p.returncode != 0:
        log.error("git push failed")


def main() -> None:
    log.info("post-processor up; waiting for orchestrator's ALL_PULLED sentinel")
    if not wait_for_all_pulled():
        log.error("timed out waiting for ALL_PULLED")
        return

    log.info("ALL_PULLED detected — running evals")
    eval_results = run_evals()
    da_results = run_dominant_axis()
    write_summary(eval_results, da_results)
    verdict, reason = classify_verdict()
    log.info(f"verdict: {verdict} — {reason}")
    write_verdict(verdict, reason, eval_results, da_results)
    maybe_commit(verdict)
    log.info("post-processing complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception(f"post-processor crashed: {e}")
        raise
