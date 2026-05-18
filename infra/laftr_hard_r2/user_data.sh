#!/bin/bash
# user-data for LAFTR_HARD_R2 on g4dn.xlarge (1× T4, GPU).
# Tag: LAFTR_HARD_R2
# Bucket prefix: s3://__S3_BUCKET__/laftr_hard_r2/
# Hard wall cap: 20 hours (3 datasets × 6h timeout each + smoke + sync margin).
#
# Pilot: LAFTR baseline retrained with proxy-Lagrangian R²<=0.05 constraint
# (Appendix Q apples-to-apples + the new mechanism). 60-cell grid: Adult 24
# pairs/seed×3 + HMDA 18×3 + Diabetes 18×3 × 200 epochs.
#
# Stages:
#   1. clone repo at __GIT_REF__, install deps
#   2. acquire data (S3 mirror first, fallback to in-repo prep scripts)
#   3. live-log watchdog (60s S3 sync of partial results + logs)
#   4. smoke (Adult seed 0, 5 epochs) — sanity-check the GPU pipeline; bail on failure
#   5. full pilot: 3 datasets × 3 seeds × 200 epochs, 6h timeout per dataset
#   6. final S3 sync + shutdown

set -u
exec > /var/log/laftr_hard_r2.log 2>&1
echo "=== LAFTR_HARD_R2 bootstrap @ $(date -u) ==="

S3_PREFIX="__S3_PREFIX__"
S3_BUCKET="__S3_BUCKET__"
GIT_REF="__GIT_REF__"
HARD_CAP_SEC=$(( 20 * 3600 ))
PER_DATASET_TIMEOUT=21600   # 6h × 3600s
INSTANCE_ID="$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo unknown)"
echo "instance-id=${INSTANCE_ID}"
echo "S3_PREFIX=${S3_PREFIX}"
echo "GIT_REF=${GIT_REF}"

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

aws s3 cp /var/log/laftr_hard_r2.log "${S3_PREFIX}/bootstrap.log" --no-progress >/dev/null 2>&1 || true
aws s3 cp <(echo "${HEAD_SHA}") "${S3_PREFIX}/git_head_${INSTANCE_ID}.txt" --no-progress >/dev/null 2>&1 || true

${PIP} install -q -r requirements.txt
${PIP} install -q "concept-erasure>=0.2.0"
echo "deps installed"
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_01_deps_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 2: data acquisition ───────────────────────────────────────────
mkdir -p /home/ubuntu/PCRL/data /home/ubuntu/PCRL/checkpoints /home/ubuntu/PCRL/results
echo "attempting to sync preprocessed data from s3://${S3_BUCKET}/data/ ..."
aws s3 sync "s3://${S3_BUCKET}/data/" /home/ubuntu/PCRL/data/ --no-progress >/dev/null 2>&1 || true
ls /home/ubuntu/PCRL/data/ 2>/dev/null | sed 's/^/  data\//'

mkdir -p /home/ubuntu/PCRL/data/adult

if [ ! -f /home/ubuntu/PCRL/data/hmda_processed/train.parquet ]; then
  if [ ! -f /home/ubuntu/PCRL/data/hmda_raw/hmda_2023_ca.csv ]; then
    echo "HMDA raw missing — attempting s3 fallback"
    aws s3 cp "s3://${S3_BUCKET}/raw_data/hmda_2023_ca.csv" \
      /home/ubuntu/PCRL/data/hmda_raw/hmda_2023_ca.csv --no-progress 2>&1 || \
      echo "WARN: HMDA raw not in S3; HMDA run will fail"
  fi
  if [ -f /home/ubuntu/PCRL/data/hmda_raw/hmda_2023_ca.csv ]; then
    echo "running prepare_hmda.py"
    sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && ${PY} experiments/prepare_hmda.py" \
      > /home/ubuntu/PCRL/prepare_hmda.log 2>&1 || echo "WARN: prepare_hmda failed"
    aws s3 cp /home/ubuntu/PCRL/prepare_hmda.log "${S3_PREFIX}/prepare_hmda.log" --no-progress >/dev/null 2>&1 || true
  fi
fi

if [ ! -f /home/ubuntu/PCRL/data/diabetes_processed/train.parquet ]; then
  if [ ! -d /home/ubuntu/PCRL/data/diabetes/dataset_diabetes ]; then
    echo "Diabetes raw missing — attempting s3 fallback"
    mkdir -p /home/ubuntu/PCRL/data/diabetes
    aws s3 sync "s3://${S3_BUCKET}/raw_data/diabetes/" \
      /home/ubuntu/PCRL/data/diabetes/ --no-progress 2>&1 || \
      echo "WARN: Diabetes raw not in S3; Diabetes run will fail"
  fi
  if [ -d /home/ubuntu/PCRL/data/diabetes/dataset_diabetes ]; then
    echo "running preprocess_diabetes.py"
    sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && ${PY} experiments/preprocess_diabetes.py" \
      > /home/ubuntu/PCRL/preprocess_diabetes.log 2>&1 || echo "WARN: preprocess_diabetes failed"
    aws s3 cp /home/ubuntu/PCRL/preprocess_diabetes.log "${S3_PREFIX}/preprocess_diabetes.log" --no-progress >/dev/null 2>&1 || true
  fi
fi

chown -R ubuntu:ubuntu /home/ubuntu/PCRL/data /home/ubuntu/PCRL/checkpoints /home/ubuntu/PCRL/results
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_02_data_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 3: live-log watchdog ──────────────────────────────────────────
cat > /home/ubuntu/sync_watchdog.sh <<WATCHDOG_EOF
#!/bin/bash
S3_PREFIX="${S3_PREFIX}"
while true; do
  for LOG in /home/ubuntu/PCRL/laftr_hard_r2_*.log /var/log/laftr_hard_r2.log; do
    [ -f "\${LOG}" ] && aws s3 cp "\${LOG}" "\${S3_PREFIX}/live_logs/\$(basename \${LOG})" --no-progress >/dev/null 2>&1 || true
  done
  for DS in adult hmda diabetes; do
    if [ -d /home/ubuntu/PCRL/results/laftr_hard_r2_\${DS}_LAFTR_HARD_R2 ]; then
      aws s3 sync /home/ubuntu/PCRL/results/laftr_hard_r2_\${DS}_LAFTR_HARD_R2 \
        "\${S3_PREFIX}/results_partial/laftr_hard_r2_\${DS}/" --no-progress >/dev/null 2>&1 || true
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

# ── Stage 4: smoke (Adult seed 0, 5 epochs) ─────────────────────────────
SMOKE_LOG=/home/ubuntu/PCRL/laftr_hard_r2_smoke.log
sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && ${PY} -u experiments/run_laftr_hard_r2.py \
    --dataset adult --out-tag _LAFTR_SMOKE --seeds 0 --epochs 5 --device cuda" \
    > "${SMOKE_LOG}" 2>&1
SMOKE_RC=$?
echo "smoke rc=${SMOKE_RC}"
aws s3 cp "${SMOKE_LOG}" "${S3_PREFIX}/smoke/smoke.log" --no-progress >/dev/null 2>&1 || true
if [ -f /home/ubuntu/PCRL/results/laftr_hard_r2_adult_LAFTR_SMOKE/summary.json ]; then
  aws s3 cp /home/ubuntu/PCRL/results/laftr_hard_r2_adult_LAFTR_SMOKE/summary.json \
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

# ── Stage 5: full pilot (3 datasets × 3 seeds × 200 epochs) ─────────────
for DS in adult hmda diabetes; do
  echo "=== full pilot ${DS} @ $(date -u) ==="
  sudo -u ubuntu timeout ${PER_DATASET_TIMEOUT} bash -c "cd /home/ubuntu/PCRL && ${PY} -u experiments/run_laftr_hard_r2.py \
      --dataset ${DS} --out-tag _LAFTR_HARD_R2 --seeds 0 1 2 --device cuda" \
      > /home/ubuntu/PCRL/laftr_hard_r2_${DS}.log 2>&1
  RC=$?
  echo "${DS} rc=${RC}"
  aws s3 cp /home/ubuntu/PCRL/laftr_hard_r2_${DS}.log "${S3_PREFIX}/logs/${DS}.log" --no-progress >/dev/null 2>&1 || true
  if [ -f /home/ubuntu/PCRL/results/laftr_hard_r2_${DS}_LAFTR_HARD_R2/summary.json ]; then
    aws s3 cp /home/ubuntu/PCRL/results/laftr_hard_r2_${DS}_LAFTR_HARD_R2/summary.json \
      "${S3_PREFIX}/per_dataset_summary/${DS}_summary.json" --no-progress >/dev/null 2>&1 || true
  fi
done
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_05_pilot_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 6: final S3 sync + shutdown ───────────────────────────────────
for DS in adult hmda diabetes; do
  if [ -d /home/ubuntu/PCRL/results/laftr_hard_r2_${DS}_LAFTR_HARD_R2 ]; then
    aws s3 sync /home/ubuntu/PCRL/results/laftr_hard_r2_${DS}_LAFTR_HARD_R2 \
      "${S3_PREFIX}/results_final/laftr_hard_r2_${DS}/" --no-progress >/dev/null 2>&1 || true
  fi
done
aws s3 cp /var/log/laftr_hard_r2.log "${S3_PREFIX}/laftr_hard_r2.log" --no-progress >/dev/null 2>&1 || true

touch /home/ubuntu/done.flag
echo "all done @ $(date -u); shutting down"
sudo shutdown -h now
