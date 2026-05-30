#!/bin/bash
# user-data for cross-purpose Phase D on g4dn.xlarge (1× T4, GPU).
# Two instance variants by INSTANCE_TAG:
#   CROSS_PURPOSE_AB        → trains Adult + HMDA
#   CROSS_PURPOSE_DIABETES  → trains Diabetes
#
# Per dataset: 3 seeds × 200 epochs, --use-erase-layer --lora-target=repr_proj_only,
# cross-purpose attrs hardcoded below, --cross-purpose-threshold 0.10.
# Hard wall cap: 20h (matches LAFTR). Expected wall: 4-5h.
#
# Stages:
#   1 env: clone + deps                                    → STAGE_01_env_ok.txt
#   2 data: dataset prep (S3 sync + fallback)              → STAGE_02_data_ok.txt
#   3 watchdog: 60s live-log sync                          → STAGE_03_watchdog_ok.txt
#   4 smoke: first dataset, 5 epochs, GPU sanity           → STAGE_04_smoke_ok.txt
#   5 train+eval: full 200-ep training, then run_eval_multi → STAGE_05_pilot_ok.txt
#   6 status: emit STATUS.txt, dual-upload to ARCHIVE_DEST → STAGE_06_status_ok.txt
#   7 sync: results_final + log dual-sync, shutdown        → STAGE_07_archive_ok.txt

set -u
exec > /var/log/cross_purpose.log 2>&1
echo "=== CROSS_PURPOSE bootstrap @ $(date -u) ==="

S3_PREFIX="__S3_PREFIX__"
S3_BUCKET="__S3_BUCKET__"
GIT_REF="__GIT_REF__"
ARCHIVE_DEST="__ARCHIVE_DEST__"
INSTANCE_TAG="__INSTANCE_TAG__"
DATASETS="__DATASETS__"
HARD_CAP_SEC=$(( 20 * 3600 ))
PER_DATASET_TIMEOUT=21600   # 6h × 3600s
INSTANCE_ID="$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo unknown)"
echo "instance-id=${INSTANCE_ID}"
echo "INSTANCE_TAG=${INSTANCE_TAG}"
echo "DATASETS=${DATASETS}"
echo "S3_PREFIX=${S3_PREFIX}"
echo "ARCHIVE_DEST=${ARCHIVE_DEST}"
echo "GIT_REF=${GIT_REF}"

# Cross-purpose attribute lists per dataset (hardcoded for safety; matches
# the §3 Phase C decision table in results/rebuttal/cross_purpose/PLAN.md).
adult_attrs="race sex age_group"
hmda_attrs="ethnicity race sex"
diabetes_attrs="race gender age_bucket"
get_attrs() { case "$1" in adult) echo "${adult_attrs}";; hmda) echo "${hmda_attrs}";; diabetes) echo "${diabetes_attrs}";; esac; }

if [ -x /opt/pytorch/bin/python ]; then
    PY=/opt/pytorch/bin/python
    PIP=/opt/pytorch/bin/pip
else
    PY=python3
    PIP="python3 -m pip"
fi
export PATH=/opt/pytorch/bin:$PATH
export HOME=/home/ubuntu
cd /home/ubuntu

# ── Backup auto-shutdown ────────────────────────────────────────────────
setsid nohup bash -c "sleep ${HARD_CAP_SEC}; echo BACKUP_SLEEPER_FIRING >> /home/ubuntu/watchdog.log; sudo shutdown -h now" \
  </dev/null >/dev/null 2>&1 &
disown
echo "backup sleeper armed (${HARD_CAP_SEC}s)"

# ── Stage 1: clone repo + install ───────────────────────────────────────
if [ ! -d /home/ubuntu/PCRL ]; then
  sudo -u ubuntu git clone https://github.com/Bostesa/PCRL.git PCRL
fi
cd /home/ubuntu/PCRL
sudo -u ubuntu git fetch --all
sudo -u ubuntu git checkout "${GIT_REF}"
HEAD_SHA="$(sudo -u ubuntu git rev-parse HEAD)"
echo "PCRL HEAD: ${HEAD_SHA}"

aws s3 cp /var/log/cross_purpose.log "${S3_PREFIX}/bootstrap.log" --no-progress >/dev/null 2>&1 || true
aws s3 cp <(echo "${HEAD_SHA}") "${S3_PREFIX}/git_head_${INSTANCE_ID}.txt" --no-progress >/dev/null 2>&1 || true

${PIP} install -q -r requirements.txt
${PIP} install -q "concept-erasure>=0.2.0"
echo "deps installed"
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_01_env_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 2: data acquisition ───────────────────────────────────────────
mkdir -p /home/ubuntu/PCRL/data /home/ubuntu/PCRL/checkpoints /home/ubuntu/PCRL/results
echo "attempting to sync preprocessed data from s3://${S3_BUCKET}/data/ ..."
aws s3 sync "s3://${S3_BUCKET}/data/" /home/ubuntu/PCRL/data/ --no-progress >/dev/null 2>&1 || true

for DS in ${DATASETS}; do
  case "${DS}" in
    hmda)
      if [ ! -f /home/ubuntu/PCRL/data/hmda_processed/train.parquet ]; then
        if [ ! -f /home/ubuntu/PCRL/data/hmda_raw/hmda_2023_ca.csv ]; then
          mkdir -p /home/ubuntu/PCRL/data/hmda_raw
          aws s3 cp "s3://${S3_BUCKET}/raw_data/hmda_2023_ca.csv" \
            /home/ubuntu/PCRL/data/hmda_raw/hmda_2023_ca.csv --no-progress 2>&1 || \
            echo "WARN: HMDA raw not in S3; HMDA run will fail"
        fi
        if [ -f /home/ubuntu/PCRL/data/hmda_raw/hmda_2023_ca.csv ]; then
          sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && ${PY} experiments/prepare_hmda.py" \
            > /home/ubuntu/PCRL/prepare_hmda.log 2>&1 || echo "WARN: prepare_hmda failed"
          aws s3 cp /home/ubuntu/PCRL/prepare_hmda.log "${S3_PREFIX}/prepare_hmda.log" --no-progress >/dev/null 2>&1 || true
        fi
      fi ;;
    diabetes)
      if [ ! -f /home/ubuntu/PCRL/data/diabetes_processed/train.parquet ]; then
        if [ ! -d /home/ubuntu/PCRL/data/diabetes/dataset_diabetes ]; then
          mkdir -p /home/ubuntu/PCRL/data/diabetes
          aws s3 sync "s3://${S3_BUCKET}/raw_data/diabetes/" \
            /home/ubuntu/PCRL/data/diabetes/ --no-progress 2>&1 || \
            echo "WARN: Diabetes raw not in S3; Diabetes run will fail"
        fi
        if [ -d /home/ubuntu/PCRL/data/diabetes/dataset_diabetes ]; then
          sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && ${PY} experiments/preprocess_diabetes.py" \
            > /home/ubuntu/PCRL/preprocess_diabetes.log 2>&1 || echo "WARN: preprocess_diabetes failed"
          aws s3 cp /home/ubuntu/PCRL/preprocess_diabetes.log "${S3_PREFIX}/preprocess_diabetes.log" --no-progress >/dev/null 2>&1 || true
        fi
      fi ;;
    adult) : ;;
  esac
done

chown -R ubuntu:ubuntu /home/ubuntu/PCRL/data /home/ubuntu/PCRL/checkpoints /home/ubuntu/PCRL/results
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_02_data_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 3: live-log watchdog ──────────────────────────────────────────
cat > /home/ubuntu/sync_watchdog.sh <<WATCHDOG_EOF
#!/bin/bash
S3_PREFIX="${S3_PREFIX}"
INSTANCE_TAG="${INSTANCE_TAG}"
while true; do
  for LOG in /home/ubuntu/PCRL/cross_purpose_*.log /var/log/cross_purpose.log; do
    [ -f "\${LOG}" ] && aws s3 cp "\${LOG}" "\${S3_PREFIX}/live_logs/\$(basename \${LOG})" --no-progress >/dev/null 2>&1 || true
  done
  for DS in adult hmda diabetes; do
    if [ -d /home/ubuntu/PCRL/results/v2_\${DS}_\${INSTANCE_TAG} ]; then
      aws s3 sync /home/ubuntu/PCRL/results/v2_\${DS}_\${INSTANCE_TAG} \
        "\${S3_PREFIX}/results_partial/v2_\${DS}_\${INSTANCE_TAG}/" --no-progress >/dev/null 2>&1 || true
    fi
  done
  echo "watchdog @ \$(date -u)" > /home/ubuntu/watchdog.log
  aws s3 cp /home/ubuntu/watchdog.log "\${S3_PREFIX}/live_logs/watchdog.log" --no-progress >/dev/null 2>&1 || true
  sleep 60
done
WATCHDOG_EOF
chmod +x /home/ubuntu/sync_watchdog.sh
setsid nohup /home/ubuntu/sync_watchdog.sh </dev/null >/dev/null 2>&1 &
disown
echo "sync watchdog armed (60s cadence)"
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_03_watchdog_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 4: smoke (first listed dataset, seed 0, 5 epochs) ─────────────
SMOKE_DS="$(echo ${DATASETS} | awk '{print $1}')"
SMOKE_ATTRS="$(get_attrs ${SMOKE_DS})"
SMOKE_LOG=/home/ubuntu/PCRL/cross_purpose_smoke.log
sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && ${PY} -u experiments/run_v2_dataset.py \
    --dataset ${SMOKE_DS} --seeds 0 --epochs 5 --device cuda \
    --use-erase-layer --lora-target repr_proj_only \
    --cross-purpose-attrs ${SMOKE_ATTRS} --cross-purpose-threshold 0.10 \
    --out-tag _CROSS_PURPOSE_SMOKE" > "${SMOKE_LOG}" 2>&1
SMOKE_RC=$?
echo "smoke rc=${SMOKE_RC}"
aws s3 cp "${SMOKE_LOG}" "${S3_PREFIX}/smoke/smoke.log" --no-progress >/dev/null 2>&1 || true
if [ -f /home/ubuntu/PCRL/results/v2_${SMOKE_DS}_CROSS_PURPOSE_SMOKE/summary.json ]; then
  aws s3 cp /home/ubuntu/PCRL/results/v2_${SMOKE_DS}_CROSS_PURPOSE_SMOKE/summary.json \
    "${S3_PREFIX}/smoke/summary.json" --no-progress >/dev/null 2>&1 || true
fi
if [ "${SMOKE_RC}" != "0" ]; then
  echo "SMOKE FAILED — aborting full pilot" > /home/ubuntu/PCRL/results/SMOKE_FAILED.txt
  aws s3 cp /home/ubuntu/PCRL/results/SMOKE_FAILED.txt "${S3_PREFIX}/SMOKE_FAILED.txt" --no-progress >/dev/null 2>&1 || true
  touch /home/ubuntu/done.flag
  sudo shutdown -h now
  exit 0
fi
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_04_smoke_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 5: full training + cross-purpose eval per dataset ─────────────
for DS in ${DATASETS}; do
  ATTRS="$(get_attrs ${DS})"
  echo "=== train ${DS} (attrs=${ATTRS}) @ $(date -u) ==="
  sudo -u ubuntu timeout ${PER_DATASET_TIMEOUT} bash -c "cd /home/ubuntu/PCRL && ${PY} -u experiments/run_v2_dataset.py \
      --dataset ${DS} --seeds 0 1 2 --epochs 200 --device cuda \
      --use-erase-layer --lora-target repr_proj_only \
      --cross-purpose-attrs ${ATTRS} --cross-purpose-threshold 0.10 \
      --out-tag _${INSTANCE_TAG}" \
      > /home/ubuntu/PCRL/cross_purpose_${DS}_train.log 2>&1
  TRAIN_RC=$?
  echo "${DS} train rc=${TRAIN_RC}"
  aws s3 cp /home/ubuntu/PCRL/cross_purpose_${DS}_train.log "${S3_PREFIX}/logs/${DS}_train.log" --no-progress >/dev/null 2>&1 || true
  if [ -f /home/ubuntu/PCRL/results/v2_${DS}_${INSTANCE_TAG}/summary.json ]; then
    aws s3 cp /home/ubuntu/PCRL/results/v2_${DS}_${INSTANCE_TAG}/summary.json \
      "${S3_PREFIX}/per_dataset_summary/${DS}_summary.json" --no-progress >/dev/null 2>&1 || true
  fi

  echo "=== eval ${DS} @ $(date -u) ==="
  sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && ${PY} -u scripts/crosspurp/run_eval_multi.py \
      --dataset ${DS} --tag ${INSTANCE_TAG}" \
      > /home/ubuntu/PCRL/cross_purpose_${DS}_eval.log 2>&1
  EVAL_RC=$?
  echo "${DS} eval rc=${EVAL_RC}"
  aws s3 cp /home/ubuntu/PCRL/cross_purpose_${DS}_eval.log "${S3_PREFIX}/logs/${DS}_eval.log" --no-progress >/dev/null 2>&1 || true
  if [ -f /home/ubuntu/PCRL/results/v2_${DS}_${INSTANCE_TAG}/results.json ]; then
    aws s3 cp /home/ubuntu/PCRL/results/v2_${DS}_${INSTANCE_TAG}/results.json \
      "${S3_PREFIX}/per_dataset_summary/${DS}_eval_results.json" --no-progress >/dev/null 2>&1 || true
  fi
done
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_05_pilot_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 6: durable status file ────────────────────────────────────────
STATUS_FILE=/home/ubuntu/PCRL/STATUS.txt
sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && ${PY} -u scripts/emit_run_status_cross_purpose.py \
    --instance-id ${INSTANCE_ID} --tag ${INSTANCE_TAG} --datasets ${DATASETS} --out ${STATUS_FILE}" \
    > /home/ubuntu/PCRL/emit_status.log 2>&1 || echo "WARN: emit_run_status_cross_purpose failed"
aws s3 cp "${STATUS_FILE}" "${S3_PREFIX}/STATUS.txt" --no-progress >/dev/null 2>&1 || true
aws s3 cp "${STATUS_FILE}" "${ARCHIVE_DEST}/STATUS.txt" --no-progress >/dev/null 2>&1 || true
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_06_status_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 7: final S3 sync (run prefix + durable archive) + shutdown ─────
for DS in ${DATASETS}; do
  if [ -d /home/ubuntu/PCRL/results/v2_${DS}_${INSTANCE_TAG} ]; then
    aws s3 sync /home/ubuntu/PCRL/results/v2_${DS}_${INSTANCE_TAG} \
      "${S3_PREFIX}/results_final/v2_${DS}_${INSTANCE_TAG}/" --no-progress >/dev/null 2>&1 || true
    aws s3 sync /home/ubuntu/PCRL/results/v2_${DS}_${INSTANCE_TAG} \
      "${ARCHIVE_DEST}/results_final/v2_${DS}_${INSTANCE_TAG}/" --no-progress >/dev/null 2>&1 || true
  fi
done
aws s3 cp /var/log/cross_purpose.log "${S3_PREFIX}/cross_purpose.log" --no-progress >/dev/null 2>&1 || true
aws s3 cp /var/log/cross_purpose.log "${ARCHIVE_DEST}/cross_purpose.log" --no-progress >/dev/null 2>&1 || true
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_07_archive_ok.txt" --no-progress >/dev/null 2>&1 || true

touch /home/ubuntu/done.flag
echo "all done @ $(date -u); archived to ${ARCHIVE_DEST}; shutting down"
sudo shutdown -h now
