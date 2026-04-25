#!/bin/bash
# Watchdog for sweep_baselines.py on this instance.
#
# Triggers shutdown (which stops the instance — InstanceInitiatedShutdownBehavior=stop)
# on either:
#   (a) sweep PID has exited (success or crash path)
#   (b) wall time exceeds HARD_CAP (9 h by default)
#
# Usage: watchdog.sh <sweep_pid> <start_epoch_seconds> [hard_cap_seconds]

set -u

SWEEP_PID="${1:?missing sweep_pid}"
START="${2:?missing start_epoch}"
HARD_CAP="${3:-32400}"   # 9 h = 32400 s

LOGFILE=/home/ubuntu/PCRL/watchdog.log
RESULTS_DIR=/home/ubuntu/PCRL/results/adult
POLL_S=60

log() { echo "[$(date -Iseconds)] $*" >> "$LOGFILE"; }

log "watchdog_start pid=$SWEEP_PID start=$START hard_cap_s=$HARD_CAP poll_s=$POLL_S"
log "watchdog_pid=$$ pgid=$(ps -o pgid= -p $$ | tr -d ' ')"

list_partial() {
    log "partial_csvs:"
    for f in \
        "$RESULTS_DIR/laftr_sweep.csv" \
        "$RESULTS_DIR/laftr_sweep_winner_seeds.csv" \
        "$RESULTS_DIR/inlp_sweep.csv" \
        "$RESULTS_DIR/inlp_sweep_winner_seeds.csv" \
        "$RESULTS_DIR/leace_sweep.csv" \
        "$RESULTS_DIR/leace_sweep_winner_seeds.csv" \
        "$RESULTS_DIR/baseline_sweep_summary.csv"
    do
        if [ -f "$f" ]; then
            lines=$(wc -l < "$f" 2>/dev/null || echo 0)
            mtime=$(stat -c '%y' "$f" 2>/dev/null)
            log "  HAVE $f  lines=$lines  mtime=$mtime"
        else
            log "  MISS $f"
        fi
    done
    log "sweep_log_tail:"
    tail -20 /home/ubuntu/PCRL/sweep_baselines.log 2>&1 | sed 's/^/    /' >> "$LOGFILE"
}

kill_tree() {
    local pid=$1
    # Sweep was launched with setsid, so PGID == PID. Kill the whole group.
    if kill -0 "$pid" 2>/dev/null; then
        log "kill_tree TERM -$pid"
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        sleep 15
        if kill -0 "$pid" 2>/dev/null; then
            log "kill_tree KILL -$pid"
            kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
    # Belt-and-braces: also kill any stray python experiments/sweep_baselines.py
    pkill -KILL -f 'experiments/sweep_baselines.py' 2>/dev/null || true
}

do_shutdown() {
    local reason="$1"
    log "shutdown_begin reason=$reason"
    sync
    list_partial
    sync
    log "calling: sudo /sbin/shutdown -h now"
    # Fire and forget; even if we block here, shutdown brings the system down.
    sudo /sbin/shutdown -h now "$reason" >> "$LOGFILE" 2>&1 || \
        sudo /sbin/poweroff >> "$LOGFILE" 2>&1 || \
        sudo /sbin/halt -p >> "$LOGFILE" 2>&1 || true
    # Spin for a while; if shutdown didn't fire for some reason, try again.
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
        kill_tree "$SWEEP_PID"
        sleep 5
        do_shutdown "hard-cap-9h"
        exit 0
    fi

    if ! kill -0 "$SWEEP_PID" 2>/dev/null; then
        log "sweep_exited pid=$SWEEP_PID elapsed=${ELAPSED}s"
        # Give any last flushes a few seconds
        sleep 10
        do_shutdown "sweep-complete"
        exit 0
    fi

    # Periodic heartbeat every 30 min
    if [ $((ELAPSED % 1800)) -lt "$POLL_S" ]; then
        log "heartbeat elapsed=${ELAPSED}s pid_alive=yes"
    fi

    sleep "$POLL_S"
done
