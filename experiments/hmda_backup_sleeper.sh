#!/usr/bin/env bash
# PCRL HMDA backup watchdog — independent dumb sleeper.
# Sleeps for SLEEP_SEC then issues sudo shutdown -h now, no matter what.
# Runs alongside hmda_watchdog.sh as a defence-in-depth shutdown.

set -u

LOG=${LOG:-/home/ubuntu/watchdog_hmda_backup.log}
SLEEP_SEC=${SLEEP_SEC:-36000}    # 10h
PIDFILE=/home/ubuntu/watchdog_hmda_backup.pid

echo $$ > "$PIDFILE"
printf '[%s] backup sleeper started (pid=%s, sleep=%ss)\n' \
  "$(date -u +%FT%TZ)" "$$" "$SLEEP_SEC" >> "$LOG"

sleep "$SLEEP_SEC"

printf '[%s] backup sleeper firing — sudo shutdown -h now\n' \
  "$(date -u +%FT%TZ)" >> "$LOG"
sudo shutdown -h now "PCRL HMDA backup sleeper" >> "$LOG" 2>&1 || \
  sudo halt -p >> "$LOG" 2>&1 || true
