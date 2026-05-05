#!/bin/bash
# Stage 2/3 worker queue runner.
#
# Args (positional):
#   $1: WORKER_IDX  — 0 or 1
#   $2: STAGE       — "adult" (Stage 2) or "hmda+diabetes" (Stage 3)
#
# Reads its assigned (dataset, purpose_idx, seed) tuples from a static
# queue baked into this script, runs scripts/run_laftr_benchmark.py for
# each (200 epochs, full data), writes per-run metrics, and syncs each
# completed run to S3 immediately. After all runs, drops a DONE marker.
#
# Pre-conditions (must be set up by user_data or SSH bootstrap):
#   /home/ec2-user/PCRL is a clone of branch laftr-benchmark-2026-05-05
#   /home/ec2-user/PCRL/.venv has the required deps
#   /home/ec2-user/PCRL/data/{adult,hmda_processed,diabetes_processed}
#   AWS credentials available (instance profile pcrl-bios-s3-writer)

set -uo pipefail

WORKER_IDX="${1:?worker_idx required}"
STAGE="${2:?stage required}"
S3_PREFIX="s3://pcrl-bios-overnight-20260504/laftr_benchmark"
WORKER_TAG="worker_${WORKER_IDX}_${STAGE}"
HOST_TAG="$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo unknown)"
LOG="/tmp/${WORKER_TAG}.log"

# ─────────────────────────── queue definitions ─────────────────────────────
# Format: "dataset purpose_idx seed"
ADULT_W0=(
  "adult 0 0"
  "adult 0 1"
  "adult 0 2"
  "adult 1 0"
  "adult 1 1"
)
ADULT_W1=(
  "adult 1 2"
  "adult 2 0"
  "adult 2 1"
  "adult 2 2"
)
HMDA_DIABETES_W0=(
  "hmda 0 0"
  "hmda 0 1"
  "hmda 0 2"
  "hmda 1 0"
  "hmda 1 1"
  "hmda 1 2"
  "hmda 2 0"
  "hmda 2 1"
  "hmda 2 2"
)
HMDA_DIABETES_W1=(
  "diabetes 0 0"
  "diabetes 0 1"
  "diabetes 0 2"
  "diabetes 1 0"
  "diabetes 1 1"
  "diabetes 1 2"
  "diabetes 2 0"
  "diabetes 2 1"
  "diabetes 2 2"
)

case "${STAGE}_${WORKER_IDX}" in
  adult_0)            JOBS=("${ADULT_W0[@]}") ;;
  adult_1)            JOBS=("${ADULT_W1[@]}") ;;
  hmda+diabetes_0)    JOBS=("${HMDA_DIABETES_W0[@]}") ;;
  hmda+diabetes_1)    JOBS=("${HMDA_DIABETES_W1[@]}") ;;
  *) echo "FATAL: unknown stage/worker combo ${STAGE}_${WORKER_IDX}"; exit 2 ;;
esac

PURPOSE_NAME() {
  local ds="$1" idx="$2"
  case "$ds:$idx" in
    adult:0)    echo "income_prediction" ;;
    adult:1)    echo "employment_analysis" ;;
    adult:2)    echo "education_assessment" ;;
    hmda:0)     echo "underwriting" ;;
    hmda:1)     echo "pricing_analysis" ;;
    hmda:2)     echo "fair_lending_audit" ;;
    diabetes:0) echo "billing_audit" ;;
    diabetes:1) echo "quality_research" ;;
    diabetes:2) echo "clinical_decision_support" ;;
    *) echo "UNKNOWN"; return 1 ;;
  esac
}

stage_marker() {
  local name="$1"
  echo "[$(date -u +%FT%TZ)] STAGE: $name (${HOST_TAG}/${WORKER_TAG})"
  echo "$name" | aws s3 cp - "${S3_PREFIX}/${STAGE}_${WORKER_IDX}_${HOST_TAG}_${name}.txt" || true
}

cd /home/ec2-user/PCRL
exec >>"$LOG" 2>&1

stage_marker "00_queue_start"
echo "JOBS: ${JOBS[@]}"
echo "HEAD: $(git rev-parse HEAD 2>/dev/null || echo no-git)"

# Pull latest branch
git pull --rebase origin laftr-benchmark-2026-05-05 || true
echo "Post-pull HEAD: $(git rev-parse HEAD 2>/dev/null || echo no-git)"

stage_marker "10_repo_synced"

# Iterate jobs
for job in "${JOBS[@]}"; do
  read ds pidx seed <<<"$job"
  pname="$(PURPOSE_NAME "$ds" "$pidx")"
  out_dir="results/laftr_benchmark/${ds}/${pname}/seed_${seed}"
  mkdir -p "$out_dir"

  if [ -f "$out_dir/metrics.json" ] && [ -f "$out_dir/encoder.pt" ] \
       && [ -f "$out_dir/eval_reps.npz" ] && [ -f "$out_dir/test_labels.npz" ]; then
    echo "[$(date -u +%FT%TZ)] SKIP $job — already complete"
    continue
  fi

  echo
  echo "=================================================================="
  echo "[$(date -u +%FT%TZ)] START $job  out=${out_dir}"
  echo "=================================================================="
  T0=$(date -u +%s)

  .venv/bin/python scripts/run_laftr_benchmark.py \
      --dataset "$ds" --purpose-idx "$pidx" --seed "$seed" \
      --output-dir "$out_dir" \
      --epochs 200 --log-every 25 \
      --lambda-adv 1.0 --lambda-warmup-epochs 100 \
      --patience 30 --batch-size 256 --lr 1e-3
  RC=$?
  T1=$(date -u +%s)
  WALL=$((T1 - T0))
  echo "[$(date -u +%FT%TZ)] END   $job  rc=${RC}  wall=${WALL}s"

  # Sync to S3 immediately (protects against worker loss mid-queue)
  aws s3 sync "$out_dir" "${S3_PREFIX}/${ds}/${pname}/seed_${seed}/" --quiet || true

  if [ "$RC" -ne 0 ]; then
    echo "[$(date -u +%FT%TZ)] FAIL on $job — continuing to next job"
    echo "$job rc=$RC wall=${WALL}s" | aws s3 cp - "${S3_PREFIX}/FAIL_${ds}_${pname}_seed${seed}.txt" || true
  fi
done

stage_marker "99_queue_done"

# Final sync of the entire bench root for this worker
aws s3 sync /home/ec2-user/PCRL/results/laftr_benchmark/ "${S3_PREFIX}/" --quiet || true

# DONE marker
echo "$(date -u +%FT%TZ) ${HOST_TAG} ${WORKER_TAG}" \
    | aws s3 cp - "${S3_PREFIX}/DONE_${STAGE}_${WORKER_IDX}.txt" || true

echo "[$(date -u +%FT%TZ)] queue complete"
