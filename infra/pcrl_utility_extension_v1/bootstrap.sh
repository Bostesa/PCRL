#!/bin/bash
# Cloud host bootstrap (EC2 user-data, Ubuntu 24.04 x86_64). Values in __DOUBLE_UNDERSCORE__ are
# substituted by launch.py; nothing secret is embedded (S3 access comes from the instance role).
set -euxo pipefail
exec > >(tee -a /var/log/pcrl-ux-bootstrap.log) 2>&1

BUCKET=__BUCKET__
PREFIX=__PREFIX__
BRANCH=__BRANCH__
U=ubuntu
H=__HOME__

apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y git zstd unzip curl build-essential
curl -sSfL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip
unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install
mkdir -p /data
aws s3 cp s3://$BUCKET/$PREFIX/run/runcfg.json /data/runcfg.json

# 0. In-host maximum-runtime stop, independent of the scheduler process: a PERSISTENT timer at an
#    absolute UTC deadline (survives stop/start; fires at boot if the deadline has passed).
#    Instance-initiated shutdown behaviour is 'stop' (set at launch), so EBS evidence survives.
DEADLINE=$(python3 -c "import json;print(json.load(open('/data/runcfg.json'))['deadline_systemd'])")
cat > /etc/systemd/system/pcrl-ux-deadline.service <<EOF
[Unit]
Description=PCRL hard runtime deadline
[Service]
Type=oneshot
ExecStart=/sbin/shutdown -h now
EOF
cat > /etc/systemd/system/pcrl-ux-deadline.timer <<EOF
[Unit]
Description=PCRL hard runtime deadline timer
[Timer]
OnCalendar=$DEADLINE
Persistent=true
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload && systemctl enable --now pcrl-ux-deadline.timer
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

mkdir -p /data $H/.config/superpowers/worktrees/PCRL
chown -R $U:$U /data $H

sudo -u $U -H bash -euxo pipefail <<EOS
cd /data
git clone -q --no-checkout __REPO_URL__ pcrl-git
cd pcrl-git
git fetch -q origin $BRANCH
git worktree add -q --detach $H/PCRL ad2c08872815185e4b63ae1d160e44f3aea1d5f4
git worktree add -q --detach $H/PCRL-terminal-1-invariant 73903b7f28df68284285f0610a4036beb32b208f
git worktree add -q --detach $H/.config/superpowers/worktrees/PCRL/residual-spectral-20260910 349efa454afd907389760fd1f59fd8806a215efd
git worktree add -q -B $BRANCH $H/PCRL-terminal-1-utility origin/$BRANCH
git -C $H/PCRL-terminal-1-utility branch --set-upstream-to=origin/$BRANCH
uv python install 3.13.7
uv venv -q --python 3.13.7 /data/venv
VIRTUAL_ENV=/data/venv uv pip install -q numpy==2.4.2 scipy==1.17.1 scikit-learn==1.8.0 pandas==3.0.1 \
  joblib==1.5.3 threadpoolctl==3.6.0 concept-erasure==0.2.4 PyYAML==6.0.3 pytest==9.0.2 tqdm==4.67.3 matplotlib==3.10.8
VIRTUAL_ENV=/data/venv uv pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
EOS

# 1. Run configuration and the execution bundle (restored in place, every file sha256-verified).
mkdir -p /data/verify /data/scratch && chown -R $U:$U /data
aws s3 cp s3://$BUCKET/$PREFIX/run/roots.json /data/roots.json
chown $U:$U /data/runcfg.json /data/roots.json
EXEC_CHUNKS=$(python3 -c "import json;print(' '.join(json.load(open('/data/runcfg.json'))['exec_chunk_ids']))")
sudo -u $U -H /data/venv/bin/python $H/PCRL-terminal-1-utility/infra/pcrl_utility_extension_v1/restore.py \
  --bucket $BUCKET --prefix $PREFIX --chunks $EXEC_CHUNKS --roots /data/roots.json --verify-dir /data/verify || true

# 2. Services: scheduler and background archive verifier. Environment is explicit.
ENVLINES="Environment=PCRL_UX_RUNCFG=/data/runcfg.json
Environment=PCRL_UX_OUT=$H/PCRL-terminal-1-utility/results/pcrl_utility_extension_v1
Environment=PCRL_DAX_INVARIANT_ROOT=$H/PCRL-terminal-1-invariant
Environment=PYTHONPATH=$H/PCRL-terminal-1-utility
Environment=PATH=/data/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1"
cat > /etc/systemd/system/pcrl-ux.service <<EOF
[Unit]
Description=PCRL utility-extension tier scheduler
After=network-online.target
[Service]
User=$U
WorkingDirectory=$H/PCRL-terminal-1-utility
$ENVLINES
ExecStart=/data/venv/bin/python -m experiments.pcrl_utility_extension_v1.scheduler
Restart=on-failure
RestartSec=60
StandardOutput=append:/data/scheduler.stdout
StandardError=append:/data/scheduler.stdout
[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/pcrl-ux-verify.service <<EOF
[Unit]
Description=PCRL archive read-back verifier
After=network-online.target
[Service]
User=$U
Nice=10
$ENVLINES
ExecStart=/data/venv/bin/python $H/PCRL-terminal-1-utility/infra/pcrl_utility_extension_v1/verifier_loop.py
Restart=on-failure
RestartSec=120
StandardOutput=append:/data/verifier.stdout
StandardError=append:/data/verifier.stdout
[Install]
WantedBy=multi-user.target
EOF
echo "$U ALL=(root) NOPASSWD: /sbin/shutdown, /usr/sbin/shutdown" > /etc/sudoers.d/pcrl-ux
systemctl daemon-reload
# Both services are enabled; the scheduler waits for the START flag that launch.py writes only
# after the independent-stop rehearsal has passed.
systemctl enable pcrl-ux.service pcrl-ux-verify.service
systemctl start pcrl-ux-verify.service pcrl-ux.service
aws s3 cp /var/log/pcrl-ux-bootstrap.log s3://$BUCKET/$PREFIX/run/bootstrap.log --sse AES256 || true
touch /data/BOOTSTRAP_DONE
aws s3 cp /data/BOOTSTRAP_DONE s3://$BUCKET/$PREFIX/run/BOOTSTRAP_DONE --sse AES256
