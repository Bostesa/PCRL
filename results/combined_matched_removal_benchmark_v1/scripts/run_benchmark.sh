#!/bin/sh
# Matched removal benchmark runner (evaluation owner). Every stage goes through `python -m stored_model_eval bench`.
#
# The worktree is derived from this script's own location; the branch must be
# research/combined-matched-removal-benchmark-v1 (BENCH_EXTRA_ARGS may carry the test-only override).
#
#   STAGE=lock       build LOCK.json (+ SUPPORT_FROZEN.json, EFFECTIVE_PROTOCOL.json); after code freeze, before fits
#   STAGE=plan       expected vs runnable unit IDs (TIER, UNITS)                                        [no fits]
#   STAGE=dry-run    admission, roles, support, contracts, lock check (TIER, UNITS)                     [no fits]
#   STAGE=sanity     fit/validation-only shuffled-label sanity (UNITS required)
#   STAGE=tier1      lock-verified Tier-1 execution (UNITS optional; RESUME=1 skips hash-verified units)
#   STAGE=sigma-star sigma* from attacker_val only -> <private>/infer/SIGMA_STAR.json (after all Tier-1 noise units)
#   STAGE=tier2      Tier-2 E1 -> E2 -> E3, technical trigger + frozen budget stop rule (RESUME=1)
#   STAGE=infer      inference from saved predictions only -> <private>/infer/BENCH_INFER.json (TIER=1|2|all)
#   STAGE=report     tables into $PKG
#   STAGE=plots      recovery-vs-utility figures into $PKG/plots
#   STAGE=all        plan, dry-run, tier1, sigma-star, tier2, infer, report, plots
# Environment: PRIVATE (default ~/PCRL_eval_cache_private/bench_v1), LOCK, PKG, PY, TIER, UNITS, RESUME,
#              BENCH_EXTRA_ARGS (tests only: --synthetic --effective-override F --allow-other-branch-for-tests).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd -P)
WT=$(cd "$HERE/../../.." && pwd -P)
PY=${PY:-$HOME/PCRL/.venv/bin/python}
BRANCH=research/combined-matched-removal-benchmark-v1
EXTRA=${BENCH_EXTRA_ARGS:-}
actual=$(git -C "$WT" rev-parse --abbrev-ref HEAD)
if [ "$actual" != "$BRANCH" ]; then
  case " $EXTRA " in
    *" --allow-other-branch-for-tests "*) echo "WARNING: branch $actual (test override)" >&2 ;;
    *) echo "REFUSED: $WT is on branch '$actual', expected '$BRANCH'" >&2; exit 4 ;;
  esac
fi
PRIVATE=${PRIVATE:-$HOME/PCRL_eval_cache_private/bench_v1}
PKG=${PKG:-$WT/results/combined_matched_removal_benchmark_v1}
LOCK=${LOCK:-$PKG/LOCK.json}
STAGE=${STAGE:-plan}
TIER=${TIER:-1}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0
cd "$WT"
mkdir -p "$PRIVATE/logs"
common="--private-root $PRIVATE --lock $LOCK --package-dir $PKG --worktree $WT $EXTRA"
un=""; [ -n "${UNITS:-}" ] && un="--units $UNITS"
res=""; [ "${RESUME:-0}" = 1 ] && res="--resume"
TS=$(date -u +%Y%m%dT%H%M%SZ)
bench() { out=$1; shift; $PY -m stored_model_eval --out "$PRIVATE/logs/$out" bench "$@" $common >/dev/null; }

do_lock()   { bench "LOCK_BUILD_$TS.json" --lock-build; echo "lock written: $LOCK"; }
do_plan()   { bench "PLAN_T${TIER}.json" --plan --tier "$TIER" $un; echo "plan: $PRIVATE/logs/PLAN_T${TIER}.json"; }
do_dry()    { bench "DRY_RUN_T${TIER}.json" --dry-run --tier "$TIER" $un; echo "dry-run (no fits): $PRIVATE/logs/DRY_RUN_T${TIER}.json"; }
do_sanity() { [ -n "$un" ] || { echo "sanity needs UNITS" >&2; exit 2; }
              bench "SANITY_$TS.json" --shuffled-label-sanity $un; echo "sanity: $PRIVATE/sanity/SANITY.json"; }
do_tier1()  { bench "EXECUTE_T1_$TS.json" --execute-scientific-fits --tier 1 $res $un; echo "tier 1 finished"; }
do_sigma()  { bench "SIGMA_STAR_$TS.json" --sigma-star; echo "sigma*: $PRIVATE/infer/SIGMA_STAR.json"; }
do_tier2()  { bench "EXECUTE_T2_$TS.json" --execute-scientific-fits --tier 2 $res $un; echo "tier 2 finished (see BUDGET_LEDGER.json)"; }
do_infer()  { bench "INFER_$TS.json" --infer --tier "$TIER"; echo "inference: $PRIVATE/infer/BENCH_INFER.json"; }
do_report() { bench "REPORT_$TS.json" --report; echo "tables written to $PKG"; }
do_plots()  { $PY "$HERE/plots.py" --package-dir "$PKG" >/dev/null; echo "plots written to $PKG/plots"; }
case "$STAGE" in
  lock) do_lock ;;
  plan) do_plan ;;
  dry-run) do_dry ;;
  sanity) do_sanity ;;
  tier1) do_plan; do_tier1 ;;
  sigma-star) do_sigma ;;
  tier2) TIER=2; do_plan; do_tier2 ;;
  infer) do_infer ;;
  report) do_report ;;
  plots) do_plots ;;
  all) do_plan; do_dry; do_tier1; do_sigma; do_tier2; TIER=all; do_infer; do_report; do_plots ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
