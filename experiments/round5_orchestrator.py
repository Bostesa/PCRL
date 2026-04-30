"""Round 5 orchestrator/backstop — runs on local Mac.

Manages 3 EC2 instances (Adult, HMDA, Diabetes) for the V2 Round 5 retrains.
Diabetes is queued because the AWS G-family vCPU quota (8) is fully consumed
by Adult+HMDA (4+4 vCPUs each); we launch Diabetes when one of the first
two stops.

Polls every 60 s. For each instance:
  1. While running: nothing.
  2. When stopped: scp the result tarball to local, untar into the repo,
     mark done. If Diabetes hasn't launched yet, launch it.
  3. After all 3 are stopped AND tarballs pulled: write a "done" sentinel
     and exit. Post-processing (eval, summary, verdict) is a separate
     manual step the user runs in the morning.

State persists in /tmp/round5/state.json so the orchestrator can crash and
restart without re-doing work.

Hard cap: 16 hours from start. After that, force-stops everything and exits.

Usage:
    python experiments/round5_orchestrator.py
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path("/tmp/round5")
STATE_FILE = STATE_DIR / "state.json"
LOG_FILE = STATE_DIR / "orchestrator.log"
KEY = Path.home() / ".ssh" / "pcrl-gpu-key.pem"

POLL_SEC = 60
HARD_CAP_SEC = 16 * 3600
USERDATA = {
    "diabetes": "/tmp/round5/userdata_diabetes.sh",
}

# Keys: tag -> (instance_id, dataset_name, status_cur)
TAG_TO_DS = {
    "v2-adult-r5": "adult",
    "v2-hmda-r5": "hmda",
    "v2-diabetes-r5": "diabetes",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("r5")


def _run(cmd: list[str], check: bool = False, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check, **kw)


def aws_describe(iid: str) -> dict:
    p = _run([
        "aws", "ec2", "describe-instances", "--instance-ids", iid,
        "--query", "Reservations[].Instances[0].{State:State.Name,IP:PublicIpAddress,DNS:PublicDnsName}",
        "--output", "json",
    ])
    if p.returncode != 0:
        return {"State": "unknown", "IP": None, "DNS": None, "_err": p.stderr.strip()}
    out = json.loads(p.stdout or "[]")
    return out[0] if out else {"State": "unknown", "IP": None, "DNS": None}


def aws_run_instance(name: str, userdata_path: str) -> str:
    """Launch a g4dn.xlarge with the given user-data, return instance ID."""
    p = _run([
        "aws", "ec2", "run-instances",
        "--image-id", "ami-05603a42e5254c4bb",
        "--instance-type", "g4dn.xlarge",
        "--key-name", "pcrl-gpu-key",
        "--security-group-ids", "sg-0f93a621e295c9ab9",
        "--subnet-id", "subnet-0f7574a980c746112",
        "--instance-initiated-shutdown-behavior", "stop",
        "--user-data", f"file://{userdata_path}",
        "--tag-specifications",
        f"ResourceType=instance,Tags=[{{Key=Name,Value={name}}}]",
        "--query", "Instances[0].InstanceId",
        "--output", "text",
    ])
    if p.returncode != 0:
        log.error(f"run-instances failed: {p.stderr.strip()}")
        return ""
    return p.stdout.strip()


def scp_tarball(ip: str, dataset: str) -> bool:
    """Pull /home/ubuntu/results_v2_<dataset>_ROUND5.tar.gz back to local
    and untar into the repo. Returns True on success."""
    if not ip:
        log.error(f"[{dataset}] cannot scp: no IP (instance is stopped, no public IP)")
        return False
    remote_tar = f"/home/ubuntu/results_v2_{dataset}_ROUND5.tar.gz"
    local_tar = STATE_DIR / f"results_v2_{dataset}_ROUND5.tar.gz"
    log.info(f"[{dataset}] scp {ip}:{remote_tar} → {local_tar}")
    p = _run([
        "scp", "-i", str(KEY),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=20",
        f"ubuntu@{ip}:{remote_tar}", str(local_tar),
    ])
    if p.returncode != 0:
        log.error(f"[{dataset}] scp failed: {p.stderr.strip()}")
        return False
    if not local_tar.exists() or local_tar.stat().st_size < 100:
        log.error(f"[{dataset}] tarball missing or empty after scp")
        return False
    log.info(f"[{dataset}] tarball pulled: {local_tar.stat().st_size} bytes; untarring")
    p = _run(["tar", "-xzf", str(local_tar), "-C", str(ROOT)])
    if p.returncode != 0:
        log.error(f"[{dataset}] untar failed: {p.stderr.strip()}")
        return False
    return True


def start_instance(iid: str) -> bool:
    """If a stopped instance was started (e.g. for scp), kick it."""
    p = _run(["aws", "ec2", "start-instances", "--instance-ids", iid])
    if p.returncode != 0:
        log.error(f"start-instances failed for {iid}: {p.stderr.strip()}")
        return False
    return True


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(s: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(s, indent=2))


def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    s = load_state()
    s.setdefault("start_epoch", int(time.time()))
    s.setdefault("instances", {
        "adult": {"iid": Path("/tmp/round5/iid_adult.txt").read_text().strip(), "tarball_pulled": False},
        "hmda": {"iid": Path("/tmp/round5/iid_hmda.txt").read_text().strip(), "tarball_pulled": False},
        "diabetes": {"iid": "", "tarball_pulled": False, "queued": True},
    })
    save_state(s)

    log.info(f"orchestrator up; start_epoch={s['start_epoch']} state={STATE_FILE}")
    log.info(f"Adult={s['instances']['adult']['iid']} HMDA={s['instances']['hmda']['iid']}")

    while True:
        elapsed = int(time.time()) - s["start_epoch"]
        if elapsed > HARD_CAP_SEC:
            log.error(f"HARD CAP {HARD_CAP_SEC}s reached; exiting")
            break

        # ── Refresh state for active instances ──────────────────────────
        any_running_or_pending = False
        any_in_capacity = 0  # count of running/pending big instances
        for ds in ("adult", "hmda", "diabetes"):
            entry = s["instances"][ds]
            iid = entry.get("iid", "")
            if not iid:
                continue
            d = aws_describe(iid)
            entry["state"] = d.get("State")
            entry["ip"] = d.get("IP")
            entry["dns"] = d.get("DNS")
            if entry["state"] in ("pending", "running"):
                any_running_or_pending = True
                any_in_capacity += 1
            log.info(f"[{ds}] iid={iid} state={entry['state']} ip={entry['ip']}")

        # ── Launch Diabetes when capacity frees ─────────────────────────
        diab = s["instances"]["diabetes"]
        if not diab.get("iid") and diab.get("queued") and any_in_capacity < 2:
            log.info("capacity available — launching Diabetes")
            iid = aws_run_instance("v2-diabetes-r5", USERDATA["diabetes"])
            if iid:
                diab["iid"] = iid
                diab["queued"] = False
                Path("/tmp/round5/iid_diabetes.txt").write_text(iid + "\n")
                log.info(f"[diabetes] launched iid={iid}")

        # ── Pull tarballs from stopped instances ────────────────────────
        for ds in ("adult", "hmda", "diabetes"):
            entry = s["instances"][ds]
            iid = entry.get("iid", "")
            if not iid:
                continue
            if entry.get("tarball_pulled"):
                continue
            # Instance must be running (still has IP) to scp from. Once it
            # stops, IP disappears, so we need to scp BEFORE it stops or
            # restart it. The on-instance script tarballs to /home/ubuntu/
            # then touches done.flag, then watchdog waits 5min and shuts
            # down. Catch the 5-minute window.
            if entry.get("state") == "running" and entry.get("ip"):
                # Probe done.flag to see if training is done.
                p = _run([
                    "ssh", "-i", str(KEY),
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "ConnectTimeout=10",
                    f"ubuntu@{entry['ip']}", "test -f /home/ubuntu/done.flag && echo DONE",
                ])
                if "DONE" in p.stdout:
                    log.info(f"[{ds}] done.flag detected — pulling tarball")
                    if scp_tarball(entry["ip"], ds):
                        entry["tarball_pulled"] = True
            elif entry.get("state") == "stopped":
                # Instance stopped. Restart, scp, stop again.
                log.info(f"[{ds}] stopped without tarball pull — restarting to scp")
                if start_instance(iid):
                    # Wait up to 3min for it to become reachable
                    for _ in range(18):
                        time.sleep(10)
                        d = aws_describe(iid)
                        if d.get("State") == "running" and d.get("IP"):
                            entry["ip"] = d["IP"]
                            log.info(f"[{ds}] restarted; IP={entry['ip']}; sleeping 30s for ssh")
                            time.sleep(30)
                            break
                    if scp_tarball(entry.get("ip", ""), ds):
                        entry["tarball_pulled"] = True
                    # Stop it again to save cost
                    _run(["aws", "ec2", "stop-instances", "--instance-ids", iid])

        save_state(s)

        # ── All done? ───────────────────────────────────────────────────
        all_pulled = all(s["instances"][ds].get("tarball_pulled") for ds in ("adult", "hmda", "diabetes"))
        all_have_iid = all(s["instances"][ds].get("iid") for ds in ("adult", "hmda", "diabetes"))
        if all_pulled and all_have_iid:
            log.info("all 3 tarballs pulled — orchestrator exiting")
            (STATE_DIR / "ALL_PULLED").write_text(f"{int(time.time())}\n")
            break

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception(f"orchestrator crashed: {e}")
        raise
