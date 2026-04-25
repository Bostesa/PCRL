#!/bin/bash
# Launch folktables training + watchdog + backup kill, all detached from SSH.
# Prints PIDs and start timestamps, then exits so the SSH session can close.

set -eu

cd /home/ubuntu/PCRL
mkdir -p /home/ubuntu/PCRL/results/folktables

# ── Activate pytorch venv and start training (detached) ────────────────
# setsid -> new session, new process group (PGID=PID)
# nohup  -> ignore SIGHUP when SSH closes
# <&-    -> close stdin
# >log 2>&1  -> redirect output to file
# &      -> background
# Parent is the ssh session; double-fork via setsid reparents to init.
START_EPOCH=$(date +%s)

setsid nohup bash -c '
source /opt/pytorch/bin/activate
cd /home/ubuntu/PCRL
exec python experiments/run_folktables.py --states CA --seeds 0,1,2 \
  > /home/ubuntu/folktables_train.log 2>&1
' <&- > /dev/null 2>&1 &
TRAIN_PID=$!
disown

# Give training a moment to actually fork
sleep 2

# ── Start watchdog (detached), polls TRAIN_PID, 10 h cap ──────────────
setsid nohup bash /home/ubuntu/watchdog_folktables.sh "$TRAIN_PID" "$START_EPOCH" 36000 \
  <&- > /home/ubuntu/watchdog_folktables.out 2>&1 &
WATCHDOG_PID=$!
disown

# ── Start independent backup sleep (detached) ─────────────────────────
# Absolute 10 h hard kill regardless of watchdog state.
setsid nohup bash -c '
sleep 36000
echo "[$(date -Iseconds)] backup-sleep fired, issuing shutdown" \
  >> /home/ubuntu/backup_shutdown.log
sudo /sbin/shutdown -h now "backup-10h" >> /home/ubuntu/backup_shutdown.log 2>&1
' <&- > /home/ubuntu/backup_shutdown.out 2>&1 &
BACKUP_PID=$!
disown

sleep 3

# ── Record launch state ───────────────────────────────────────────────
SHUTDOWN_ABS_UTC=$(date -u -d "@$((START_EPOCH + 36000))" +"%Y-%m-%dT%H:%M:%SZ")

{
  echo "=== launch_all.sh status ==="
  echo "instance_id=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo unknown)"
  echo "start_epoch=$START_EPOCH ($(date -u -d "@$START_EPOCH" +"%Y-%m-%dT%H:%M:%SZ"))"
  echo "backup_shutdown_abs_utc=$SHUTDOWN_ABS_UTC"
  echo ""
  echo "TRAIN_PID=$TRAIN_PID  PPID=$(ps -o ppid= -p $TRAIN_PID 2>/dev/null | tr -d ' ')  PGID=$(ps -o pgid= -p $TRAIN_PID 2>/dev/null | tr -d ' ')  STATE=$(ps -o stat= -p $TRAIN_PID 2>/dev/null | tr -d ' ')"
  echo "WATCHDOG_PID=$WATCHDOG_PID  PPID=$(ps -o ppid= -p $WATCHDOG_PID 2>/dev/null | tr -d ' ')  PGID=$(ps -o pgid= -p $WATCHDOG_PID 2>/dev/null | tr -d ' ')  STATE=$(ps -o stat= -p $WATCHDOG_PID 2>/dev/null | tr -d ' ')"
  echo "BACKUP_PID=$BACKUP_PID  PPID=$(ps -o ppid= -p $BACKUP_PID 2>/dev/null | tr -d ' ')  PGID=$(ps -o pgid= -p $BACKUP_PID 2>/dev/null | tr -d ' ')  STATE=$(ps -o stat= -p $BACKUP_PID 2>/dev/null | tr -d ' ')"
  echo ""
  echo "train_log=/home/ubuntu/folktables_train.log"
  echo "watchdog_log=/home/ubuntu/watchdog_folktables.log"
  echo "backup_log=/home/ubuntu/backup_shutdown.log"
} | tee /home/ubuntu/launch_status.txt
