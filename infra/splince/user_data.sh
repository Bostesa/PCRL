#!/bin/bash
# user-data for SPLINCE-vs-PCRL benchmark on c5.4xlarge (CPU, 16 vCPU Standard family).
# Tag: SPLICE_BENCHMARK
# Bucket prefix: s3://pcrl-bios-overnight-20260504/splice_benchmark/
# Hard wall cap: 4 hours.
set -u
exec > /var/log/splince_benchmark.log 2>&1
echo "=== SPLICE_BENCHMARK bootstrap @ $(date -u) ==="

# ── REPLACE BEFORE LAUNCH ────────────────────────────────────────────────
S3_PREFIX="__S3_PREFIX__"
S3_PCRL_PREFIX="__S3_PCRL_PREFIX__"   # source bucket for warmstart + data
GIT_REF="__GIT_REF__"
HARD_CAP_SEC=$(( 4 * 3600 ))
INSTANCE_ID="$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo unknown)"
echo "instance-id=${INSTANCE_ID}"

# This is a CPU-only AMI (no /opt/pytorch). We'll use the Ubuntu DLAMI's
# pytorch venv if present, else the system python plus pip-installed deps.
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

# ── Backup auto-shutdown (defense in depth) ─────────────────────────────
setsid nohup bash -c "sleep ${HARD_CAP_SEC}; echo BACKUP_SLEEPER_FIRING >> /home/ubuntu/watchdog.log; sudo shutdown -h now" \
  </dev/null >/dev/null 2>&1 &
disown
echo "backup sleeper armed (${HARD_CAP_SEC}s)"

# ── Stage 0: clone repo at the pinned commit ────────────────────────────
if [ ! -d /home/ubuntu/PCRL ]; then
  sudo -u ubuntu git clone https://github.com/Bostesa/PCRL.git PCRL
fi
cd /home/ubuntu/PCRL
sudo -u ubuntu git fetch --all
sudo -u ubuntu git checkout "${GIT_REF}"
echo "PCRL HEAD: $(sudo -u ubuntu git rev-parse HEAD)"

aws s3 cp /var/log/splince_benchmark.log "${S3_PREFIX}/bootstrap.log" --no-progress >/dev/null 2>&1 || true
aws s3 cp <(echo "$(sudo -u ubuntu git rev-parse HEAD)") "${S3_PREFIX}/git_head_${INSTANCE_ID}.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 1: install deps ───────────────────────────────────────────────
${PIP} install -q -r requirements.txt
${PIP} install -q "concept-erasure>=0.2.0" "peft>=0.11.0"
echo "deps installed"
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_${INSTANCE_ID}_01_deps_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 2: pull warmstart checkpoints + datasets from PCRL bucket ─────
mkdir -p /home/ubuntu/PCRL/checkpoints /home/ubuntu/PCRL/data
echo "fetching warm-start checkpoints from ${S3_PCRL_PREFIX}/warmstart/ ..."
aws s3 sync "${S3_PCRL_PREFIX}/warmstart/" /home/ubuntu/PCRL/ --no-progress
echo "warm-start checkpoints synced; tree:"
ls /home/ubuntu/PCRL/checkpoints/ | head -20

echo "fetching preprocessed datasets from ${S3_PCRL_PREFIX}/data/ ..."
aws s3 sync "${S3_PCRL_PREFIX}/data/" /home/ubuntu/PCRL/data/ --no-progress
echo "data synced; tree:"
ls /home/ubuntu/PCRL/data/ | head -20

# Mirror the variance-constrained baseline target_cells + (if present) results
# so the aggregator has the PCRL-side metrics to compare against.
mkdir -p /home/ubuntu/PCRL/results/v2_pcrl_variance_constrained
aws s3 cp "${S3_PCRL_PREFIX}/results_partial/target_cells.json" \
  /home/ubuntu/PCRL/results/v2_pcrl_variance_constrained/target_cells.json \
  --no-progress >/dev/null 2>&1 || true
aws s3 cp "${S3_PCRL_PREFIX}/results_final/results.json" \
  /home/ubuntu/PCRL/results/v2_pcrl_variance_constrained/results.json \
  --no-progress >/dev/null 2>&1 || true

chown -R ubuntu:ubuntu /home/ubuntu/PCRL/data /home/ubuntu/PCRL/checkpoints /home/ubuntu/PCRL/results
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_${INSTANCE_ID}_02_data_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 3: watchdog ───────────────────────────────────────────────────
cat > /home/ubuntu/sync_watchdog.sh <<'WATCHDOG_EOF'
#!/bin/bash
S3_PREFIX="__S3_PREFIX__"
LOG=/home/ubuntu/PCRL/splince_train.log
while true; do
  if [ -f "$LOG" ]; then
    aws s3 cp "$LOG" "${S3_PREFIX}/live_logs/splince_train.log" --no-progress >/dev/null 2>&1 || true
  fi
  if [ -d /home/ubuntu/PCRL/results/splince_benchmark ]; then
    aws s3 sync /home/ubuntu/PCRL/results/splince_benchmark \
      "${S3_PREFIX}/results_partial/" --no-progress >/dev/null 2>&1 || true
  fi
  echo "watchdog @ $(date -u)" > /home/ubuntu/watchdog.log
  aws s3 cp /home/ubuntu/watchdog.log "${S3_PREFIX}/live_logs/watchdog.log" --no-progress >/dev/null 2>&1 || true
  sleep 60
done
WATCHDOG_EOF
sed -i "s|__S3_PREFIX__|${S3_PREFIX}|g" /home/ubuntu/sync_watchdog.sh
chmod +x /home/ubuntu/sync_watchdog.sh
setsid nohup /home/ubuntu/sync_watchdog.sh </dev/null >/dev/null 2>&1 &
disown
echo "sync watchdog armed (60s cadence)"
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_${INSTANCE_ID}_03_watchdog_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 4: smoke test (Adult seed 0) ──────────────────────────────────
SMOKE_LOG=/home/ubuntu/PCRL/splince_smoke.log
echo "=== Smoke start: $(date -u)" > /home/ubuntu/PCRL/splince_train.log
sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && ${PY} scripts/run_splince_benchmark.py \
    --single-cell adult,0 --device cpu \
    --out-dir results/splince_benchmark/_smoke" \
    > "${SMOKE_LOG}" 2>&1
SMOKE_RC=$?
echo "=== Smoke exit: rc=${SMOKE_RC} @ $(date -u)" >> /home/ubuntu/PCRL/splince_train.log
aws s3 cp "${SMOKE_LOG}" "${S3_PREFIX}/smoke/smoke.log" --no-progress >/dev/null 2>&1 || true
aws s3 sync /home/ubuntu/PCRL/results/splince_benchmark/_smoke \
  "${S3_PREFIX}/smoke/" --no-progress >/dev/null 2>&1 || true

if [ "${SMOKE_RC}" != "0" ]; then
  cat > /home/ubuntu/PCRL/results/splince_benchmark/FAILURE.txt <<'FAIL_EOF'
SPLICE_BENCHMARK — SMOKE FAILED before full run.

The Adult seed-0 smoke run did not complete successfully. See
${S3_PREFIX}/smoke/smoke.log for the traceback. Likely causes are
data missing in S3 or a Python dependency mismatch. The instance
is shutting itself down to avoid burning compute on a broken stack.
FAIL_EOF
  aws s3 sync /home/ubuntu/PCRL/results/splince_benchmark \
    "${S3_PREFIX}/results_final/" --no-progress >/dev/null 2>&1 || true
  echo "smoke failed; shutting down" >> /home/ubuntu/PCRL/splince_train.log
  touch /home/ubuntu/done.flag
  sudo shutdown -h now
  exit 0
fi
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_${INSTANCE_ID}_04_smoke_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 5: full 60-cell run ───────────────────────────────────────────
echo "=== Stage 5 start: $(date -u)" >> /home/ubuntu/PCRL/splince_train.log
sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && ${PY} scripts/run_splince_benchmark.py \
    --datasets adult hmda diabetes --seeds 0 1 2 \
    --device cpu --out-dir results/splince_benchmark" \
    >> /home/ubuntu/PCRL/splince_train.log 2>&1
echo "=== Stage 5 done: $(date -u)" >> /home/ubuntu/PCRL/splince_train.log
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_${INSTANCE_ID}_05_full_run_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 6: aggregation + paper artefacts ──────────────────────────────
sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && ${PY} scripts/build_splince_vs_pcrl.py \
    --out-dir results/splince_benchmark" \
    >> /home/ubuntu/PCRL/splince_train.log 2>&1
aws s3 cp /dev/null "${S3_PREFIX}/STAGE_${INSTANCE_ID}_06_aggregate_ok.txt" --no-progress >/dev/null 2>&1 || true

# ── Stage 7: final S3 sync + shutdown ───────────────────────────────────
aws s3 sync /home/ubuntu/PCRL/results/splince_benchmark \
  "${S3_PREFIX}/results_final/" --no-progress >/dev/null 2>&1 || true
aws s3 cp /home/ubuntu/PCRL/splince_train.log \
  "${S3_PREFIX}/splince_train.log" --no-progress >/dev/null 2>&1 || true
aws s3 cp /home/ubuntu/PCRL/results/splince_benchmark/HEADLINE.txt \
  "${S3_PREFIX}/HEADLINE.txt" --no-progress >/dev/null 2>&1 || true

touch /home/ubuntu/done.flag
echo "all done @ $(date -u); shutting down"
sudo shutdown -h now
