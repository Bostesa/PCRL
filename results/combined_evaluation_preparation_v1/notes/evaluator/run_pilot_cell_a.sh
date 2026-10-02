#!/bin/sh
# Pilot CELL-A (Adult, PCRL Round-4 seed 0): 8 untreated pairs + noise arms for income_prediction/sex.
# Default: admission + fit-attackers --dry-run for every manifest (no scientific fits).
# Scientific execution (attacker + closed-form fits, then infer and report): EXECUTE=1 sh run_pilot_cell_a.sh
# Inputs must exist (prepare_pilot_adult_s0.py --write-inputs; forward; --pair-manifests; --noise-manifests).
set -eu
WT=/Users/nathansamson/PCRL/.worktrees/combined-evaluation-preparation-v1
PY=/Users/nathansamson/PCRL/.venv/bin/python
M=$WT/results/combined_evaluation_preparation_v1/notes/methodology/protocol_config.json
C=$HOME/PCRL_eval_cache_private/pilot_adult_s0
cd "$WT"
case "${EXECUTE:-0}" in 1) MODE="--execute-scientific-fits";; *) MODE="--dry-run";; esac
if [ "$MODE" != "--dry-run" ]; then
  # refuse scientific execution unless protocol config and every manifest match the committed lock
  python3 - "$WT" "$C" <<'PYEOF' || { echo "LOCK CHECK FAILED: refusing to execute"; exit 3; }
import hashlib, json, os, sys
wt, c = sys.argv[1], sys.argv[2]
pk = os.path.join(wt, "results/combined_evaluation_preparation_v1")
lock = json.load(open(os.path.join(pk, "PILOT_LOCK.json")))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
assert sha(os.path.join(pk, "notes/methodology/protocol_config.json")) == lock["protocol_config_sha256"], "protocol config changed"
for name, h in lock["manifests_sha256"].items():
    assert sha(os.path.join(c, name)) == h, "manifest changed: " + name
print("lock check passed")
PYEOF
fi
n=0
for m in $C/manifest_*.json; do
  t=$(basename "$m" .json)
  case "$t" in
    *__p[0-9]_sigma*) case "$t" in manifest_income_prediction__sex__*) ;; *) continue;; esac ;;
  esac
  python3 -m stored_model_eval admit --manifest "$m" >/dev/null || { echo "ADMISSION FAILED: $t"; exit 1; }
  out=$C/scores/$t
  if [ "$MODE" = "--dry-run" ]; then
    OMP_NUM_THREADS=1 $PY -m stored_model_eval --protocol "$M" --dry-run fit-attackers --manifest "$m" --out-dir "$out" >/dev/null
  else
    OMP_NUM_THREADS=1 $PY -m stored_model_eval --protocol "$M" fit-attackers --manifest "$m" $MODE --out-dir "$out"
    $PY -m stored_model_eval --protocol "$M" --out "$out/infer.json" infer --scores "$out/scores.npz"
    $PY -m stored_model_eval --protocol "$M" --out "$out/report.json" report --scores "$out/scores.npz" --infer "$out/infer.json"
  fi
  n=$((n+1))
done
echo "units processed: $n (mode: $MODE)"
