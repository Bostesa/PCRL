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
# Durable, lifecycle-exempt destination for results_final + STATUS + checkpoints.
ARCHIVE_DEST="__ARCHIVE_DEST__"
# Canonical raw-data location (durable, lifecycle-exempt). HMDA + Diabetes raw
# files live here so the instance has a reliable source to bootstrap from.
ARCHIVE_RAW_DATA="s3://${S3_BUCKET}/archive/raw_data"
# Datasets to TRAIN this run (smoke always runs Adult regardless). Can be
# narrowed when relaunching a subset, e.g. "hmda diabetes". Existing
# results_final/ for omitted datasets are pre-synced from the archive at
# Stage 2.5 so the final STATUS.txt remains comprehensive.
DATASETS="__DATASETS__"
HARD_CAP_SEC=$(( 20 * 3600 ))
PER_DATASET_TIMEOUT=21600   # 6h × 3600s
INSTANCE_ID="$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo unknown)"
echo "instance-id=${INSTANCE_ID}"
echo "S3_PREFIX=${S3_PREFIX}"
echo "ARCHIVE_DEST=${ARCHIVE_DEST}"
echo "ARCHIVE_RAW_DATA=${ARCHIVE_RAW_DATA}"
echo "DATASETS=${DATASETS}"
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

# ── Stage 2: data acquisition (canonical source: ARCHIVE_RAW_DATA) ──────
mkdir -p /home/ubuntu/PCRL/data /home/ubuntu/PCRL/checkpoints /home/ubuntu/PCRL/results
echo "attempting to sync preprocessed data from s3://${S3_BUCKET}/data/ ..."
aws s3 sync "s3://${S3_BUCKET}/data/" /home/ubuntu/PCRL/data/ --no-progress >/dev/null 2>&1 || true
ls /home/ubuntu/PCRL/data/ 2>/dev/null | sed 's/^/  data\//'

mkdir -p /home/ubuntu/PCRL/data/adult

# HMDA: copy raw csv from the durable archive, then run prepare_hmda.py.
if [ ! -f /home/ubuntu/PCRL/data/hmda_processed/train.parquet ]; then
  if [ ! -f /home/ubuntu/PCRL/data/hmda_raw/hmda_2023_ca.csv ]; then
    echo "HMDA raw missing — fetching from ${ARCHIVE_RAW_DATA}/hmda_2023_ca.csv"
    mkdir -p /home/ubuntu/PCRL/data/hmda_raw
    aws s3 cp "${ARCHIVE_RAW_DATA}/hmda_2023_ca.csv" \
      /home/ubuntu/PCRL/data/hmda_raw/hmda_2023_ca.csv --no-progress 2>&1 || \
      echo "WARN: HMDA raw not in archive; HMDA run will fail"
  fi
  if [ -f /home/ubuntu/PCRL/data/hmda_raw/hmda_2023_ca.csv ]; then
    echo "running prepare_hmda.py"
    sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && ${PY} experiments/prepare_hmda.py" \
      > /home/ubuntu/PCRL/prepare_hmda.log 2>&1 || echo "WARN: prepare_hmda failed"
    aws s3 cp /home/ubuntu/PCRL/prepare_hmda.log "${S3_PREFIX}/prepare_hmda.log" --no-progress >/dev/null 2>&1 || true
  fi
fi

# Diabetes: copy raw csv from the durable archive. preprocess_diabetes.py
# accepts ``data/diabetes/diabetic_data.csv`` directly (it only falls back to
# the zip-extract path if that file is missing), so we sync the file itself
# rather than reconstituting the unzipped ``dataset_diabetes/`` directory.
if [ ! -f /home/ubuntu/PCRL/data/diabetes_processed/train.parquet ]; then
  if [ ! -f /home/ubuntu/PCRL/data/diabetes/diabetic_data.csv ]; then
    echo "Diabetes raw missing — fetching from ${ARCHIVE_RAW_DATA}/diabetes/diabetic_data.csv"
    mkdir -p /home/ubuntu/PCRL/data/diabetes
    aws s3 cp "${ARCHIVE_RAW_DATA}/diabetes/diabetic_data.csv" \
      /home/ubuntu/PCRL/data/diabetes/diabetic_data.csv --no-progress 2>&1 || \
      echo "WARN: Diabetes raw not in archive; Diabetes run will fail"
  fi
  if [ -f /home/ubuntu/PCRL/data/diabetes/diabetic_data.csv ]; then
    echo "running preprocess_diabetes.py"
    sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && ${PY} experiments/preprocess_diabetes.py" \
      > /home/ubuntu/PCRL/preprocess_diabetes.log 2>&1 || echo "WARN: preprocess_diabetes failed"
    aws s3 cp /home/ubuntu/PCRL/preprocess_diabetes.log "${S3_PREFIX}/preprocess_diabetes.log" --no-progress >/dev/null 2>&1 || true
  fi
fi

# Stage 2.5: pre-sync ANY existing per-dataset results from the durable
# archive into the local results/ tree. This lets the final STATUS at
# Stage 6 reflect the full 3-dataset picture even when DATASETS is narrowed
# to a subset (e.g. the relaunch-only-HMDA+Diabetes case). Idempotent: any
# dataset re-trained this run overwrites the synced copy.
#
# Tries both naming conventions for backward compat with the 2026-05-29
# Adult result that landed at ``…/laftr_hard_r2_<ds>/`` (no suffix). The
# new naming (suffix-preserving) is written by Stage 7 below.
echo "syncing prior results from ${ARCHIVE_DEST}/results_final/ ..."
mkdir -p /home/ubuntu/PCRL/results
for DS in adult hmda diabetes; do
  LOCAL_DIR=/home/ubuntu/PCRL/results/laftr_hard_r2_${DS}_LAFTR_HARD_R2
  aws s3 sync "${ARCHIVE_DEST}/results_final/laftr_hard_r2_${DS}_LAFTR_HARD_R2/" \
    "${LOCAL_DIR}/" --no-progress >/dev/null 2>&1 || true
  # Backward compat: older Adult archive entry has no suffix.
  aws s3 sync "${ARCHIVE_DEST}/results_final/laftr_hard_r2_${DS}/" \
    "${LOCAL_DIR}/" --no-progress >/dev/null 2>&1 || true
done

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

# ── Stage 5: full pilot (DATASETS × 3 seeds × 200 epochs) ──────────────
# Iterates over the run's DATASETS list (defaults to "adult hmda diabetes";
# narrow via launch.sh DATASETS_OVERRIDE for relaunches). Existing
# results for omitted datasets were pre-synced from the archive at
# Stage 2.5 so the final STATUS still reflects the full picture.
for DS in ${DATASETS}; do
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

# ── Stage 6: durable status file (survives even if result blobs are lost) ─
# Headline numbers in plain text: instance id, completion time, per-dataset
# strict-pass count + mean R² + task acc. Written to BOTH the run prefix and
# the no-lifecycle archival destination so the numbers survive an S3
# lifecycle expiration of the result blobs.
STATUS_FILE=/home/ubuntu/PCRL/STATUS.txt
sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && ${PY} -u scripts/emit_run_status.py \
    --instance-id ${INSTANCE_ID} --out ${STATUS_FILE}" \
    > /home/ubuntu/PCRL/emit_status.log 2>&1 || echo "WARN: emit_run_status failed"
aws s3 cp "${STATUS_FILE}" "${S3_PREFIX}/STATUS.txt" --no-progress >/dev/null 2>&1 || true
aws s3 cp "${STATUS_FILE}" "${ARCHIVE_DEST}/STATUS.txt" --no-progress >/dev/null 2>&1 || true
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_06_status_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 7: final S3 sync (run prefix + durable archive) + shutdown ─────
# Results JSON+summaries: sync to BOTH the run prefix (convenient browse path)
# and the durable archive. Checkpoints (.pt): sync to the durable archive
# ONLY — they're small per-run (~250KB per file × ~18 files = ~5 MB total
# on the LoRA-only setup), but the principle is "checkpoints to archive,
# not the convenient-but-ephemeral run prefix" (user guidance 2026-05-29).
for DS in adult hmda diabetes; do
  if [ -d /home/ubuntu/PCRL/results/laftr_hard_r2_${DS}_LAFTR_HARD_R2 ]; then
    # Suffix-preserving naming so the local aggregator (which reads
    # ``results/laftr_hard_r2_<ds>_LAFTR_HARD_R2/``) finds the synced copy
    # without further renaming.
    aws s3 sync /home/ubuntu/PCRL/results/laftr_hard_r2_${DS}_LAFTR_HARD_R2 \
      "${S3_PREFIX}/results_final/laftr_hard_r2_${DS}_LAFTR_HARD_R2/" --no-progress >/dev/null 2>&1 || true
    aws s3 sync /home/ubuntu/PCRL/results/laftr_hard_r2_${DS}_LAFTR_HARD_R2 \
      "${ARCHIVE_DEST}/results_final/laftr_hard_r2_${DS}_LAFTR_HARD_R2/" --no-progress >/dev/null 2>&1 || true
  fi
done
# Checkpoints: archive-only, per-seed, only for datasets trained this run.
# Glob over the actual ckpt dirs that exist; orchestrator pattern is
# ``checkpoints/laftr_hard_r2_<ds>_LAFTR_HARD_R2_s<seed>/{best,final,...}.pt``.
for CKPT_DIR in /home/ubuntu/PCRL/checkpoints/laftr_hard_r2_*_LAFTR_HARD_R2_s*; do
  [ -d "${CKPT_DIR}" ] || continue
  BASENAME="$(basename ${CKPT_DIR})"
  aws s3 sync "${CKPT_DIR}" "${ARCHIVE_DEST}/checkpoints/${BASENAME}/" \
    --no-progress --exclude '*.npz' >/dev/null 2>&1 || true
done
aws s3 cp /var/log/laftr_hard_r2.log "${S3_PREFIX}/laftr_hard_r2.log" --no-progress >/dev/null 2>&1 || true
aws s3 cp /var/log/laftr_hard_r2.log "${ARCHIVE_DEST}/laftr_hard_r2.log" --no-progress >/dev/null 2>&1 || true
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_07_archive_ok.txt" --no-progress >/dev/null 2>&1 || true

touch /home/ubuntu/done.flag
echo "all done @ $(date -u); archived to ${ARCHIVE_DEST}; shutting down"
sudo shutdown -h now
