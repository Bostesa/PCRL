#!/usr/bin/env bash
# PCRL HMDA primary watchdog — runs on the EC2 instance.
#
# Policy:
#   - Hard ceiling: HARD_LIMIT_SEC seconds from launch → SIGTERM/SIGKILL
#     experiment processes, archive results, shutdown.
#   - Idle success: IDLE_GRACE_SEC with no python experiment processes →
#     archive results, shutdown.
#
# Stop method: ``sudo shutdown -h now``
#   The instance's InstanceInitiatedShutdownBehavior is "stop", so a
#   shutdown halts the OS, then EC2 transitions the instance to the
#   STOPPED state (no terminate). Compute stops being billed; EBS
#   continues to be billed at standard rates so results persist.
#
# Result archiving:
#   On either trigger we tar up ${RESULTS_DIR} and ${CHECKPOINT_DIR} into
#   /home/ubuntu/hmda_results_final.tar.gz so the data is on the EBS
#   volume even if the user wants to scp later. If S3_BUCKET is set and
#   the instance can write to S3, we also upload the tarball.

set -u

LOG=${LOG:-/home/ubuntu/watchdog_hmda.log}
HARD_LIMIT_SEC=${HARD_LIMIT_SEC:-36000}     # 10h
IDLE_GRACE_SEC=${IDLE_GRACE_SEC:-300}       # 5 min
WATCH_PATTERN=${WATCH_PATTERN:-python.*experiments/(run_hmda|prepare_hmda)}
CHECK_INTERVAL=${CHECK_INTERVAL:-60}
PCRL_DIR=${PCRL_DIR:-/home/ubuntu/PCRL}
RESULTS_DIR=${RESULTS_DIR:-${PCRL_DIR}/results/hmda}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${PCRL_DIR}/checkpoints}
ARCHIVE_PATH=${ARCHIVE_PATH:-/home/ubuntu/hmda_results_final.tar.gz}
S3_BUCKET=${S3_BUCKET:-}

START=$(date +%s)
PIDFILE=/home/ubuntu/watchdog_hmda.pid
echo $$ > "$PIDFILE"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

count_exp() {
  pgrep -f "$WATCH_PATTERN" 2>/dev/null | wc -l | tr -d ' '
}

archive_results() {
  log "archiving ${RESULTS_DIR} and ${CHECKPOINT_DIR} → ${ARCHIVE_PATH}"
  local extra=()
  [[ -d "$RESULTS_DIR" ]] && extra+=("results/hmda")
  [[ -d "$CHECKPOINT_DIR" ]] && extra+=("checkpoints")
  if (( ${#extra[@]} == 0 )); then
    log "  nothing to archive yet"
    return
  fi
  ( cd "$PCRL_DIR" && tar -czf "$ARCHIVE_PATH" "${extra[@]}" 2>>"$LOG" ) || \
    log "  tar failed (non-fatal)"
  log "  archive size: $(du -h "$ARCHIVE_PATH" 2>/dev/null | cut -f1)"

  if [[ -n "$S3_BUCKET" ]]; then
    log "  attempting S3 upload to s3://${S3_BUCKET}/..."
    aws s3 cp "$ARCHIVE_PATH" "s3://${S3_BUCKET}/" >>"$LOG" 2>&1 \
      && log "  S3 upload OK" || log "  S3 upload failed (creds probably missing)"
  fi
}

stop_instance() {
  local reason="$1"
  log "STOP requested: $reason"
  archive_results
  sudo shutdown -h now "PCRL HMDA watchdog: $reason" >>"$LOG" 2>&1 || {
    log "shutdown -h now failed, falling back to halt"
    sudo halt -p >>"$LOG" 2>&1 || true
  }
}

kill_experiments() {
  log "terminating experiment processes (SIGTERM)..."
  pkill -TERM -f "$WATCH_PATTERN" 2>/dev/null || true
  sleep 30
  local still
  still=$(count_exp)
  if [[ "$still" -gt 0 ]]; then
    log "$still experiment procs survived SIGTERM, escalating to SIGKILL"
    pkill -KILL -f "$WATCH_PATTERN" 2>/dev/null || true
  fi
}

log "==== HMDA watchdog started (pid=$$, hard=${HARD_LIMIT_SEC}s, idle=${IDLE_GRACE_SEC}s) ===="
log "watch_pattern='$WATCH_PATTERN'"
log "current experiment procs:"
pgrep -af "$WATCH_PATTERN" >>"$LOG" 2>&1 || log "  (none yet)"

idle_since=0
saw_running=0
while true; do
  now=$(date +%s)
  elapsed=$((now - START))
  nprocs=$(count_exp)

  # Track that we've seen experiments running at least once before
  # trusting the idle timer. This avoids shutting down before training
  # actually starts.
  if (( nprocs > 0 )); then
    saw_running=1
  fi

  # Hard ceiling has top priority.
  if (( elapsed >= HARD_LIMIT_SEC )); then
    log "HARD LIMIT reached (${elapsed}s >= ${HARD_LIMIT_SEC}s)"
    kill_experiments
    stop_instance "hard 10h limit"
    exit 1
  fi

  if (( saw_running == 1 && nprocs == 0 )); then
    if (( idle_since == 0 )); then
      idle_since=$now
      log "no experiment processes; idle grace timer started"
    else
      idle_for=$((now - idle_since))
      if (( idle_for >= IDLE_GRACE_SEC )); then
        log "idle for ${idle_for}s — successful completion"
        stop_instance "successful completion (idle ${idle_for}s)"
        exit 0
      fi
    fi
  else
    if (( idle_since != 0 )); then
      log "experiments resumed (${nprocs} procs), clearing idle timer"
      idle_since=0
    fi
  fi

  if (( elapsed % 900 < CHECK_INTERVAL )); then
    log "heartbeat: elapsed=${elapsed}s procs=${nprocs} idle_since=${idle_since}"
  fi

  sleep "$CHECK_INTERVAL"
done
