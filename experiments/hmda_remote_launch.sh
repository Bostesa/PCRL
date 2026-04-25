#!/usr/bin/env bash
# Runs ON the EC2 instance after deploy. Sets up env, kicks off
# preprocessing + training, and starts both watchdogs detached.
# Idempotent: skips redundant steps if state already matches.

set -euo pipefail

PCRL_DIR=${PCRL_DIR:-/home/ubuntu/PCRL}
LOG_DIR=${LOG_DIR:-/home/ubuntu/logs}
mkdir -p "$LOG_DIR"

cd "$PCRL_DIR"

echo "[remote-launch] $(date -u +%FT%TZ) starting setup"

# ── Choose Python env (Deep Learning AMI ships /opt/pytorch as a venv) ───
if [[ -f /opt/pytorch/bin/activate ]]; then
  source /opt/pytorch/bin/activate
  echo "[remote-launch] activated venv /opt/pytorch  (python=$(which python))"
fi

# ── Ensure deps ──────────────────────────────────────────────────────────
python -m pip install --quiet --upgrade pip
python -m pip install --quiet \
  "numpy<2" pandas scikit-learn pyyaml tqdm matplotlib

# Verify torch + GPU.
python - <<'PY'
import torch
print(f"[remote-launch] torch={torch.__version__}, cuda={torch.cuda.is_available()}, "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

# ── Step 1: preprocess HMDA (download + filter + encode) ─────────────────
if [[ ! -f data/hmda_processed/train.npz ]]; then
  echo "[remote-launch] running prepare_hmda.py"
  python experiments/prepare_hmda.py 2>&1 | tee "$LOG_DIR/prepare_hmda.log"
else
  echo "[remote-launch] data/hmda_processed/train.npz already present, skipping prepare"
fi

# ── Step 2: launch training detached, then start watchdogs ───────────────
TRAIN_LOG="$LOG_DIR/run_hmda_seeds.log"
echo "[remote-launch] launching run_hmda_seeds.py detached"
setsid nohup python -u experiments/run_hmda_seeds.py \
    >"$TRAIN_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
disown $TRAIN_PID
echo $TRAIN_PID > /home/ubuntu/run_hmda.pid
echo "[remote-launch] train pid=$TRAIN_PID  log=$TRAIN_LOG"

# ── Step 3: primary watchdog ─────────────────────────────────────────────
echo "[remote-launch] starting primary watchdog"
setsid nohup bash "$PCRL_DIR/experiments/hmda_watchdog.sh" \
    </dev/null >/dev/null 2>&1 &
WATCH_PID=$!
disown $WATCH_PID
echo "[remote-launch] watchdog pid=$WATCH_PID"

# ── Step 4: backup sleeper ───────────────────────────────────────────────
echo "[remote-launch] starting backup sleeper"
setsid nohup bash "$PCRL_DIR/experiments/hmda_backup_sleeper.sh" \
    </dev/null >/dev/null 2>&1 &
BACKUP_PID=$!
disown $BACKUP_PID
echo "[remote-launch] backup pid=$BACKUP_PID"

# ── Print final status ───────────────────────────────────────────────────
sleep 3
echo
echo "================ FINAL STATUS ================"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "?")
SHUTDOWN_TS_UTC=$(date -u -d "+10 hours" +%FT%TZ 2>/dev/null || python3 -c "
import datetime; print((datetime.datetime.utcnow()+datetime.timedelta(hours=10)).strftime('%Y-%m-%dT%H:%M:%SZ'))
")
echo "Instance ID:           $INSTANCE_ID"
echo "Watchdog PID:          $WATCH_PID  (PPID=$(ps -o ppid= -p $WATCH_PID 2>/dev/null | tr -d ' '))"
echo "Backup PID:            $BACKUP_PID  (PPID=$(ps -o ppid= -p $BACKUP_PID 2>/dev/null | tr -d ' '))"
echo "Training PID:          $TRAIN_PID  (PPID=$(ps -o ppid= -p $TRAIN_PID 2>/dev/null | tr -d ' '))"
echo "Hard-cap shutdown UTC: $SHUTDOWN_TS_UTC"
echo "Logs:"
echo "  $LOG_DIR/prepare_hmda.log"
echo "  $LOG_DIR/run_hmda_seeds.log"
echo "  /home/ubuntu/watchdog_hmda.log"
echo "  /home/ubuntu/watchdog_hmda_backup.log"
echo "=============================================="
