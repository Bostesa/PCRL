#!/bin/bash
# user-data for INLP_BENCHMARK (27 cells = 3 datasets x 3 purposes x 3 seeds).
# Tag: INLP_BENCHMARK
# Bucket prefix: s3://pcrl-bios-overnight-20260504/inlp_benchmark/
# Hard wall cap: 3 hours (expected ~1-2h on c5.4xlarge with 4-way concurrency).
set -u
exec > /var/log/inlp_benchmark.log 2>&1
echo "=== INLP_BENCHMARK bootstrap @ $(date -u) ==="

# ── REPLACED BY launch.sh via sed ────────────────────────────────────────
S3_PREFIX="__S3_PREFIX__"
GIT_REF="__GIT_REF__"
HARD_CAP_SEC=$(( 3 * 3600 ))
INSTANCE_ID="$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo unknown)"
echo "instance-id=${INSTANCE_ID}"

export PATH=/opt/pytorch/bin:$PATH
export HOME=/home/ubuntu
cd /home/ubuntu

# ── Backup auto-shutdown (defense in depth) ─────────────────────────────
setsid nohup bash -c "sleep ${HARD_CAP_SEC}; echo BACKUP_SLEEPER_FIRING >> /home/ubuntu/watchdog.log; sudo shutdown -h now" \
  </dev/null >/dev/null 2>&1 &
disown
echo "backup sleeper armed (${HARD_CAP_SEC}s)"

# ── Clone repo at the pinned commit ─────────────────────────────────────
if [ ! -d /home/ubuntu/PCRL ]; then
  sudo -u ubuntu git clone https://github.com/Bostesa/PCRL.git PCRL
fi
cd /home/ubuntu/PCRL
sudo -u ubuntu git fetch --all
sudo -u ubuntu git checkout "${GIT_REF}"
echo "PCRL HEAD: $(sudo -u ubuntu git rev-parse HEAD)"

# ── Install deps ────────────────────────────────────────────────────────
/opt/pytorch/bin/pip install -q -r requirements.txt
/opt/pytorch/bin/pip install -q "scikit-learn>=1.3.0"
echo "deps installed"

# ── Pull preprocessed datasets from S3 (reuse varconstraint cache) ──────
echo "fetching preprocessed datasets from s3://pcrl-bios-overnight-20260504/pcrl_varconstraint/data/ ..."
mkdir -p /home/ubuntu/PCRL/data
aws s3 sync "s3://pcrl-bios-overnight-20260504/pcrl_varconstraint/data/" /home/ubuntu/PCRL/data/ --no-progress
echo "data synced; tree:"
ls /home/ubuntu/PCRL/data/ | head -20
chown -R ubuntu:ubuntu /home/ubuntu/PCRL/data

# ── Watchdog: stream training log + partial results to S3 every 60s ─────
cat > /home/ubuntu/sync_watchdog.sh <<'WATCHDOG_EOF'
#!/bin/bash
S3_PREFIX="__S3_PREFIX__"
LOG=/home/ubuntu/PCRL/inlp_train.log
while true; do
  if [ -f "$LOG" ]; then
    aws s3 cp "$LOG" "${S3_PREFIX}/live_logs/inlp_train.log" --no-progress >/dev/null 2>&1 || true
  fi
  if [ -d /home/ubuntu/PCRL/results/inlp_benchmark ]; then
    aws s3 sync /home/ubuntu/PCRL/results/inlp_benchmark \
      "${S3_PREFIX}/results_partial/" --no-progress >/dev/null 2>&1 || true
  fi
  sleep 60
done
WATCHDOG_EOF
sed -i "s|__S3_PREFIX__|${S3_PREFIX}|g" /home/ubuntu/sync_watchdog.sh
chmod +x /home/ubuntu/sync_watchdog.sh
setsid nohup /home/ubuntu/sync_watchdog.sh </dev/null >/dev/null 2>&1 &
disown
echo "sync watchdog armed (60s cadence)"

# ── Run all 27 cells (3 datasets × 3 purposes × 3 seeds) ────────────────
# c5.4xlarge has 16 vCPU; INLP is CPU-bound (no GPU). 4 concurrent leaves
# headroom for OS + torch threads. Per-cell wall ~3-5 min on Adult, ~5-10 min
# on HMDA (91K rows). Total expected ~30-60 min.
TRAIN_LOG=/home/ubuntu/PCRL/inlp_train.log
echo "=== Stage 1 start: $(date -u)" >> "${TRAIN_LOG}"

DATASETS=(adult hmda diabetes)
MAX_CONCURRENT=4

run_one() {
  local ds="$1"; local pidx="$2"; local seed="$3"
  local cell="${ds}_p${pidx}_s${seed}"
  local outd="results/inlp_benchmark/${ds}/purpose_${pidx}/seed_${seed}"
  local logf="/home/ubuntu/PCRL/inlp_${cell}.log"
  echo "  start ${cell} @ $(date -u)" >> "${TRAIN_LOG}"
  sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      /opt/pytorch/bin/python scripts/run_inlp_benchmark.py \
        --dataset ${ds} --purpose-idx ${pidx} --seed ${seed} \
        --output-dir ${outd}" \
      > "${logf}" 2>&1
  local rc=$?
  echo "  exit ${cell} rc=${rc} @ $(date -u)" >> "${TRAIN_LOG}"
  aws s3 cp "${logf}" "${S3_PREFIX}/full_logs/$(basename ${logf})" --no-progress >/dev/null 2>&1 || true
}

i=0
pids=()
for ds in "${DATASETS[@]}"; do
  for pidx in 0 1 2; do
    for seed in 0 1 2; do
      run_one "${ds}" "${pidx}" "${seed}" &
      pids+=($!)
      i=$((i+1))
      if [ "$((i % MAX_CONCURRENT))" -eq 0 ]; then
        wait "${pids[@]}"
        pids=()
      fi
    done
  done
done
wait "${pids[@]}"
echo "=== Stage 1 done: $(date -u)" >> "${TRAIN_LOG}"

# Sync partial results before aggregation, in case aggregation fails.
aws s3 sync /home/ubuntu/PCRL/results/inlp_benchmark \
  "${S3_PREFIX}/results_cells/" --no-progress >/dev/null 2>&1 || true

# ── Stage 2: aggregate + three-way comparison ───────────────────────────
echo "=== Stage 2 (aggregate) start: $(date -u)" >> "${TRAIN_LOG}"
sudo -u ubuntu bash -c "cd /home/ubuntu/PCRL && /opt/pytorch/bin/python infra/inlp/aggregate.py \
    --inlp-dir results/inlp_benchmark \
    --laftr-s3-prefix s3://pcrl-bios-overnight-20260504/laftr_benchmark \
    --output-dir results/inlp_benchmark" \
    >> "${TRAIN_LOG}" 2>&1
AGG_RC=$?
echo "=== Stage 2 (aggregate) exit rc=${AGG_RC} @ $(date -u)" >> "${TRAIN_LOG}"

# ── Final S3 sync + shutdown ────────────────────────────────────────────
aws s3 sync /home/ubuntu/PCRL/results/inlp_benchmark \
  "${S3_PREFIX}/results_final/" --no-progress >/dev/null 2>&1 || true
aws s3 cp "${TRAIN_LOG}" \
  "${S3_PREFIX}/inlp_train.log" --no-progress >/dev/null 2>&1 || true

touch /home/ubuntu/done.flag
echo "all done @ $(date -u); shutting down"
sudo shutdown -h now
