#!/bin/sh
# CELL-A stored-model pilot runner (repaired). Replaces the preparation's run_pilot_cell_a.sh.
#
# The worktree is derived from this script's own location; the branch must be
# research/combined-stored-model-pilot-v1. Every stage goes through `python -m stored_model_eval`.
#
#   STAGE=lock    sh run_pilot.sh   build PILOT_LOCK_v2.json + EFFECTIVE_PROTOCOL.json (after inputs exist)
#   STAGE=run     sh run_pilot.sh   plan (expected vs actual unit IDs) + dry-run         [default; no fits]
#   EXECUTE=1     sh run_pilot.sh   plan + lock-verified scientific fits (--execute-scientific-fits)
#   EXECUTE=1 RESUME=1 sh ...       skip units whose outputs are complete and hash-verified; move partials aside
#   UNITS=a,b                       restrict to registered unit IDs
#   STAGE=infer   sh run_pilot.sh   inference from saved predictions only -> $RUN_DIR/infer/PILOT_INFER.json
#   STAGE=report  sh run_pilot.sh   tables -> $OUT (PRIMARY_ENDPOINTS.csv, DECOMPOSITION.csv, UTILITY.csv, ...)
#   STAGE=all                       run, then infer, then report
# Environment: RUN_DIR (default ~/PCRL_eval_cache_private/pilot_adult_s0/run_v1), LOCK, OUT, PY, HIST (historical
# dominant_axis_audit.json for the N0 comparison), FEATURES, PILOT_EXTRA_ARGS (tests only, e.g. --synthetic).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd -P)
WT=$(cd "$HERE/../../.." && pwd -P)
PY=${PY:-/Users/nathansamson/PCRL/.venv/bin/python}
BRANCH=research/combined-stored-model-pilot-v1
actual=$(git -C "$WT" rev-parse --abbrev-ref HEAD)
EXTRA=${PILOT_EXTRA_ARGS:-}
if [ "$actual" != "$BRANCH" ]; then
  case " $EXTRA " in
    *" --allow-other-branch-for-tests "*) echo "WARNING: branch $actual (test override)" >&2 ;;
    *) echo "REFUSED: $WT is on branch '$actual', expected '$BRANCH'" >&2; exit 4 ;;
  esac
fi
RUN_DIR=${RUN_DIR:-$HOME/PCRL_eval_cache_private/pilot_adult_s0/run_v1}
LOCK=${LOCK:-$WT/results/combined_stored_model_pilot_v1/PILOT_LOCK_v2.json}
OUT=${OUT:-$WT/results/combined_stored_model_pilot_v1}
FEATURES=${FEATURES:-$RUN_DIR/../features.npz}
STAGE=${STAGE:-run}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0
cd "$WT"
SME="$PY -m stored_model_eval"
common="--worktree $WT --run-dir $RUN_DIR --lock $LOCK"

do_lock() {
  fe=""; [ -f "$FEATURES" ] && fe="--features $FEATURES"
  lockextra=""; case " $EXTRA " in *" --allow-other-branch-for-tests "*) lockextra="--allow-other-branch-for-tests";; esac
  $SME --out "$RUN_DIR/LOCK_BUILD.json" lock build --worktree "$WT" --run-dir "$RUN_DIR" --lock "$LOCK" \
       --effective-out "$OUT/EFFECTIVE_PROTOCOL.json" $fe $lockextra >/dev/null
  echo "lock written: $LOCK"
}
do_run() {
  $SME --out "$RUN_DIR/PLAN.json" pilot --plan $common $EXTRA >/dev/null
  echo "plan OK (expected == actual registered unit IDs): $RUN_DIR/PLAN.json"
  if [ "${EXECUTE:-0}" = 1 ]; then
    res=""; [ "${RESUME:-0}" = 1 ] && res="--resume"
    un=""; [ -n "${UNITS:-}" ] && un="--units $UNITS"
    $SME --out "$RUN_DIR/EXECUTE_$(date -u +%Y%m%dT%H%M%SZ).json" pilot --execute-scientific-fits $res $un $common $EXTRA >/dev/null
    echo "execution finished"
  else
    un=""; [ -n "${UNITS:-}" ] && un="--units $UNITS"
    $SME --out "$RUN_DIR/DRY_RUN.json" pilot $un $common $EXTRA >/dev/null
    echo "dry-run OK (no fits): $RUN_DIR/DRY_RUN.json"
  fi
}
do_infer() {
  mkdir -p "$RUN_DIR/infer"
  $SME --out "$RUN_DIR/infer/PILOT_INFER.json" infer --units-dir "$RUN_DIR/units" >/dev/null
  echo "inference written: $RUN_DIR/infer/PILOT_INFER.json"
}
do_report() {
  h=""; [ -n "${HIST:-}" ] && h="--historical-native $HIST"
  $SME --out "$RUN_DIR/infer/REPORT_SUMMARY.json" report --pilot-infer "$RUN_DIR/infer/PILOT_INFER.json" \
       --tables-dir "$OUT" $h >/dev/null
  echo "tables written to $OUT"
}
case "$STAGE" in
  lock) do_lock ;;
  run) do_run ;;
  infer) do_infer ;;
  report) do_report ;;
  all) do_run; do_infer; do_report ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
