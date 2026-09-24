#!/bin/bash
# Privacy-first selector v1 host bootstrap (adapted from SC 537e74a) (Amazon Linux 2023, x86_64, SSM only).
# Rendered by cloud.py: every double-underscore placeholder is replaced and checked.
# Order matters: the watchdog is armed before anything that can fail.
set -euo pipefail
mkdir -p /opt/pcrl/work /opt/pcrl/logs /opt/pcrl/locks
exec > >(tee -a /opt/pcrl/logs/bootstrap.log) 2>&1

STUDY=pcrl_privacy_first_selector_v1
SC_STUDY=pcrl_shared_context_release_v1
BUCKET=pcrl-ux-archive-ed9d21fd
DEST="s3://$BUCKET/$STUDY"
COMMIT=__COMMIT__
BRANCH=__BRANCH__
REPO=__REPO__
MANIFEST_VERSION=__MANIFEST_VERSION__
MANIFEST_SHA=__MANIFEST_SHA__
export AWS_DEFAULT_REGION=us-east-1
export HOME=${HOME:-/root}

fail() {
  echo "BOOTSTRAP_FAILED at line $1" | tee /opt/pcrl/logs/BOOTSTRAP_FAILED
  aws s3 cp /opt/pcrl/logs/bootstrap.log "$DEST/control/bootstrap.log" --sse AES256 --only-show-errors || true
  aws s3 cp /opt/pcrl/logs/BOOTSTRAP_FAILED "$DEST/control/BOOTSTRAP_FAILED" --sse AES256 --only-show-errors || true
}
trap 'fail $LINENO' ERR

# 1. Watchdog: archive results, then shut down (instance terminates on shutdown).
cat >/usr/local/sbin/pcrl-pfs-watchdog <<'EOF'
#!/bin/bash
set -u
dest=s3://pcrl-ux-archive-ed9d21fd/pcrl_privacy_first_selector_v1/emergency
logs=/opt/pcrl/logs
date -u +%FT%TZ >"$logs/watchdog-started-utc.txt"
systemctl stop 'pcrl-pfs-step-*' >/dev/null 2>&1 || true
if [ -d /opt/pcrl/work/results/pcrl_privacy_first_selector_v1 ]; then
  timeout 1500 aws s3 sync /opt/pcrl/work/results/pcrl_privacy_first_selector_v1 "$dest/results" \
    --sse AES256 --only-show-errors >"$logs/watchdog-sync.log" 2>&1
  echo $? >"$logs/watchdog-sync-exit-code.txt"
else
  echo missing-results-directory >"$logs/watchdog-sync.log"
  echo 2 >"$logs/watchdog-sync-exit-code.txt"
fi
timeout 300 aws s3 cp "$logs" "$dest/logs" --recursive --sse AES256 --only-show-errors || true
sync
shutdown -h now
EOF
chmod 0700 /usr/local/sbin/pcrl-pfs-watchdog
cat >/etc/systemd/system/pcrl-pfs-watchdog.service <<'EOF'
[Unit]
Description=Archive and shut down the PCRL shared-context study host
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/pcrl-pfs-watchdog
EOF
cat >/etc/systemd/system/pcrl-pfs-watchdog.timer <<'EOF'
[Unit]
Description=PCRL shared-context fixed watchdog (sync then shutdown)
[Timer]
OnCalendar=__WATCHDOG_UTC__ UTC
AccuracySec=1min
Persistent=true
Unit=pcrl-pfs-watchdog.service
[Install]
WantedBy=timers.target
EOF
# Hard stop 30 minutes later even if the sync hangs.
cat >/etc/systemd/system/pcrl-pfs-hardstop.service <<'EOF'
[Unit]
Description=PCRL shared-context hard stop
[Service]
Type=oneshot
ExecStart=/usr/sbin/shutdown -h now
EOF
cat >/etc/systemd/system/pcrl-pfs-hardstop.timer <<'EOF'
[Unit]
Description=PCRL shared-context hard stop backstop
[Timer]
OnCalendar=__HARD_STOP_UTC__ UTC
AccuracySec=1min
Persistent=true
Unit=pcrl-pfs-hardstop.service
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now pcrl-pfs-watchdog.timer pcrl-pfs-hardstop.timer
systemctl list-timers pcrl-pfs-watchdog.timer pcrl-pfs-hardstop.timer --no-pager
printf '%s\n' "__WATCHDOG_UTC__" >/opt/pcrl/logs/WATCHDOG_ARMED
aws s3 cp /opt/pcrl/logs/WATCHDOG_ARMED "$DEST/control/WATCHDOG_ARMED" --sse AES256 --only-show-errors

# 2. Runtime: the predecessor's exact Linux stack (Python 3.11 venv, pinned wheels).
dnf install -y -q python3.11 python3.11-pip gcc gcc-c++ git zstd tar
python3.11 -m venv /opt/pcrl/venv
/opt/pcrl/venv/bin/python -m pip install -q --upgrade pip
/opt/pcrl/venv/bin/python -m pip install -q --no-cache-dir numpy==2.4.2 scipy==1.17.1 \
  scikit-learn==1.8.0 pandas==3.0.1 joblib==1.5.3 threadpoolctl==3.6.0 psutil pytest
/opt/pcrl/venv/bin/python -m pip install -q --no-cache-dir torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cpu
/opt/pcrl/venv/bin/python -m pip freeze >/opt/pcrl/logs/environment.txt

# 3. Code: sparse, blobless clone of the pushed branch, pinned to COMMIT.
git clone -q --filter=blob:none --no-checkout --sparse --depth 200 --single-branch \
  --branch "$BRANCH" "$REPO" /opt/pcrl/work
cd /opt/pcrl/work
git sparse-checkout set --no-cone \
  /experiments/__init__.py /experiments/pcrl_adaptive_release_v1/ \
  /experiments/pcrl_task_aligned_cuts_v1/ /experiments/pcrl_task_directed_release_v1/ \
  /experiments/pcrl_final_prospective_v1/ /experiments/pcrl_shared_context_release_v1/ \
  /tests/__init__.py /tests/conftest.py /tests/pcrl_adaptive_release_v1/ \
  /tests/pcrl_shared_context_release_v1/ /results/pcrl_adaptive_release_v1/ \
  /results/pcrl_task_aligned_cuts_v1/ /results/pcrl_shared_context_release_v1/ \
  /experiments/pcrl_privacy_first_selector_v1/ /tests/pcrl_privacy_first_selector_v1/ \
  /results/pcrl_privacy_first_selector_v1/
git merge-base --is-ancestor "$COMMIT" "origin/$BRANCH"
git checkout -q --detach "$COMMIT"
test "$(git rev-parse HEAD)" = "$COMMIT"
printf '%s\n' "$COMMIT" >/opt/pcrl/work/SOURCE_COMMIT.txt

# 4. Inputs: the predecessor's 17 label-stripped objects, by version and SHA-256.
aws s3api get-object --bucket "$BUCKET" --key "$SC_STUDY/control/INPUT_STAGE.json" \
  --version-id "$MANIFEST_VERSION" /opt/pcrl/INPUT_STAGE.json >/dev/null
printf '%s  %s\n' "$MANIFEST_SHA" /opt/pcrl/INPUT_STAGE.json | sha256sum -c -
PYTHONPATH=/opt/pcrl/work /opt/pcrl/venv/bin/python -m \
  experiments.pcrl_shared_context_release_v1.stage restore \
  --manifest /opt/pcrl/INPUT_STAGE.json --receipt /opt/pcrl/logs/INPUT_RESTORE.json

# 5. (No SC archive sweep: this study archives explicitly via host.py archive.)

# 6. Ready marker.
/opt/pcrl/venv/bin/python - <<'PY'
import hashlib, json, pathlib, platform, sys
logs = pathlib.Path("/opt/pcrl/logs")
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
record = {"schema": "pcrl-pfs-host-ready-v1", "system": platform.system(),
          "machine": platform.machine(), "platform": platform.platform(),
          "python": sys.version.split()[0],
          "source_commit": pathlib.Path("/opt/pcrl/work/SOURCE_COMMIT.txt").read_text().strip(),
          "environment_sha256": sha(logs / "environment.txt"),
          "input_restore_sha256": sha(logs / "INPUT_RESTORE.json"),
          "input_restore_all_verified": json.loads((logs / "INPUT_RESTORE.json").read_text())["all_verified"]}
(logs / "READY.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
aws s3 cp /opt/pcrl/logs/READY.json "$DEST/control/READY.json" --sse AES256 --only-show-errors
aws s3 cp /opt/pcrl/logs/environment.txt "$DEST/control/environment.txt" --sse AES256 --only-show-errors
aws s3 cp /opt/pcrl/logs/bootstrap.log "$DEST/control/bootstrap.log" --sse AES256 --only-show-errors
echo BOOTSTRAP_READY
