#!/bin/bash
set -euo pipefail
mkdir -p /opt/pcrl/work /opt/pcrl/logs
cat >/usr/local/sbin/pcrl-adaptive-watchdog <<'EOF'
#!/bin/bash
set -u
prefix=s3://pcrl-ux-archive-ed9d21fd/pcrl_adaptive_release_v1/emergency
date -u +%FT%TZ >/opt/pcrl/logs/watchdog-started-utc.txt
if [ -d /opt/pcrl/work/results/pcrl_adaptive_release_v1 ]; then
  aws s3 sync /opt/pcrl/work/results/pcrl_adaptive_release_v1 "$prefix/results" --sse AES256 --only-show-errors >/opt/pcrl/logs/watchdog-sync.log 2>&1
  echo $? >/opt/pcrl/logs/watchdog-sync-exit-code.txt
else
  echo missing-results-directory >/opt/pcrl/logs/watchdog-sync.log
  echo 2 >/opt/pcrl/logs/watchdog-sync-exit-code.txt
fi
aws s3 cp /opt/pcrl/logs/watchdog-sync.log "$prefix/watchdog-sync.log" --sse AES256 --only-show-errors || true
aws s3 cp /opt/pcrl/logs/watchdog-sync-exit-code.txt "$prefix/watchdog-sync-exit-code.txt" --sse AES256 --only-show-errors || true
aws s3 cp /opt/pcrl/logs/watchdog-started-utc.txt "$prefix/watchdog-started-utc.txt" --sse AES256 --only-show-errors || true
sync
shutdown -h now
EOF
chmod 0700 /usr/local/sbin/pcrl-adaptive-watchdog
cat >/etc/systemd/system/pcrl-adaptive-watchdog.service <<'EOF'
[Unit]
Description=Archive and shut down PCRL adaptive study instance
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/pcrl-adaptive-watchdog
EOF
cat >/etc/systemd/system/pcrl-adaptive-watchdog.timer <<'EOF'
[Unit]
Description=PCRL adaptive study fixed cloud backstop
[Timer]
OnCalendar=2026-09-24 21:33:31 UTC
AccuracySec=1min
Persistent=true
Unit=pcrl-adaptive-watchdog.service
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now pcrl-adaptive-watchdog.timer
EOF
