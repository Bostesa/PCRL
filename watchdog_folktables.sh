#!/bin/bash
# Watchdog for run_folktables.py on this instance.
#
# Triggers sudo shutdown -h now (InstanceInitiatedShutdownBehavior=stop)
# on either:
#   (a) training PID has exited (success or crash path)
#   (b) wall time exceeds HARD_CAP (default 10h = 36000s)
#
# Usage: watchdog_folktables.sh <train_pid> <start_epoch_seconds> [hard_cap_seconds]

set -u

TRAIN_PID="${1:?missing train_pid}"
START="${2:?missing start_epoch}"
HARD_CAP="${3:-36000}"   # 10 h

LOGFILE=/home/ubuntu/watchdog_folktables.log
RESULTS_DIR=/home/ubuntu/PCRL/results/folktables
POLL_S=60

log() { echo "[$(date -Iseconds)] $*" >> "$LOGFILE"; }

log "watchdog_start pid=$TRAIN_PID start=$START hard_cap_s=$HARD_CAP poll_s=$POLL_S"
log "watchdog_pid=$$ ppid=$(ps -o ppid= -p $$ | tr -d ' ') pgid=$(ps -o pgid= -p $$ | tr -d ' ')"

list_partial() {
    log "partial_csvs:"
    for f in \
        "$RESULTS_DIR/pcrl_seeds.csv" \
        "$RESULTS_DIR/baselines.csv" \
        "$RESULTS_DIR/summary.csv" \
        "$RESULTS_DIR/adaptation_notes.md"
    do
        if [ -f "$f" ]; then
            lines=$(wc -l < "$f" 2>/dev/null || echo 0)
            mtime=$(stat -c '%y' "$f" 2>/dev/null)
            log "  HAVE $f  lines=$lines  mtime=$mtime"
        else
            log "  MISS $f"
        fi
    done
    log "train_log_tail:"
    tail -30 /home/ubuntu/folktables_train.log 2>&1 | sed 's/^/    /' >> "$LOGFILE"
}

kill_tree() {
    local pid=$1
    # Training launched with setsid, so PGID == PID. Kill the whole group.
    if kill -0 "$pid" 2>/dev/null; then
        log "kill_tree TERM -$pid"
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        sleep 15
        if kill -0 "$pid" 2>/dev/null; then
            log "kill_tree KILL -$pid"
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
    pkill -KILL -f 'experiments/run_folktables.py' 2>/dev/null || true
}

do_shutdown() {
    local reason="$1"
    log "shutdown_begin reason=$reason"
    sync
    list_partial
    sync
    log "calling: sudo /sbin/shutdown -h now"
    sudo /sbin/shutdown -h now "$reason" >> "$LOGFILE" 2>&1 || \
        sudo /sbin/poweroff >> "$LOGFILE" 2>&1 || \
        sudo /sbin/halt -p >> "$LOGFILE" 2>&1 || true
    for _ in 1 2 3 4 5 6; do
        sleep 30
        log "still_up post_shutdown; retrying"
        sudo /sbin/shutdown -h now "$reason" >> "$LOGFILE" 2>&1 || true
    done
}

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))

    if [ "$ELAPSED" -gt "$HARD_CAP" ]; then
        log "HARD_CAP hit elapsed=${ELAPSED}s"
        kill_tree "$TRAIN_PID"
        sleep 5
        do_shutdown "hard-cap-10h"
        exit 0
    fi

    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        log "train_exited pid=$TRAIN_PID elapsed=${ELAPSED}s"
        sleep 10
        do_shutdown "training-complete"
        exit 0
    fi

    # Periodic heartbeat every 30 min
    if [ $((ELAPSED % 1800)) -lt "$POLL_S" ]; then
        log "heartbeat elapsed=${ELAPSED}s pid_alive=yes"
    fi

    sleep "$POLL_S"
done
