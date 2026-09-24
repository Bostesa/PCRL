"""One task-tagged, SSM-only c7i.8xlarge for the shared-context study.

Coordinator-facing CLI (profile `vein`, us-east-1; credentials are never
printed):

    preflight                      identity, AMI, instance profile, subnet, vCPU room,
                                   bucket versioning/SSE, no live study instance
    render --commit SHA            print the rendered user data (no AWS calls)
    launch --commit SHA [--watchdog-utc T] [--dry-run]
                                   dry run = EC2 DryRun on every call, nothing created
    send/get/wait                  SSM AWS-RunShellScript on the verified study instance
    stage-code --commit SHA        re-pin the host checkout to another pushed commit
    start-runner --queue Q --workers N
                                   systemd-run the queue runner as pcrl-sc-runner
    status                         instance, READY marker, timers, runner status
    terminate --confirm ID         tag-checked termination
    cost [--record]                cost ledger (compute, gp3, S3 prefix bytes)

Launch facts verified 2026-09-24 (see agents/infra/INFRA_NOTES.md): the
adaptive-release predecessor ran ami-0b2c9d1f3edcfd709 (Amazon Linux 2023,
CloudTrail RunInstances 2026-09-24T02:54:24Z) with instance profile
pcrl-ux-ec2 in subnet-092b1557c0ffb806f; its security group was deleted at
closeout, so a new zero-ingress group is created here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STUDY = "pcrl_shared_context_release_v1"
PRIVATE = ROOT / "results" / STUDY / "agents" / "infra" / "private"  # git-ignored (infra/)
RESOURCE = PRIVATE / "AWS_RESOURCES.json"
COMMANDS = PRIVATE / "SSM_COMMANDS.jsonl"
COST_LEDGER = PRIVATE / "COST_LEDGER.json"
USER_DATA = Path(__file__).resolve().parent / "cloud_user_data.sh"
BUCKET = "pcrl-ux-archive-ed9d21fd"
REGION = "us-east-1"
PROFILE = "vein"
INSTANCE_TYPE = "c7i.8xlarge"
INSTANCE_VCPUS = 32
AMI = "ami-0b2c9d1f3edcfd709"          # AL2023 2023.12.20260918.0, used by the predecessor
ALT_AMI = "ami-025d99823a4caad37"      # Ubuntu 24.04 (task-directed study); not used here
INSTANCE_PROFILE = "pcrl-ux-ec2"
SUBNET = "subnet-092b1557c0ffb806f"    # default VPC, us-east-1a (predecessor)
VOLUME_GIB = 120
REPO = "https://github.com/Bostesa/PCRL.git"
BRANCH = "research/pcrl-shared-context-release-v1"
TAGS = {"Project": "pcrl", "Study": STUDY}
PRICE = {"c7i.8xlarge_usd_per_hour": 1.428,   # AWS price list 2026-09-21 (predecessor ledger)
         "gp3_usd_per_gib_month": 0.08, "s3_standard_usd_per_gb_month": 0.023}
STANDARD_FAMILIES = tuple("acdhimrtz")
DEFAULT_HOURS = 10
HARD_STOP_MINUTES = 30
# Protocol ceiling (RUN_STATUS: 20 h elapsed). Amendment M5.6: clamped in code.
CEILING_UTC = datetime(2026, 9, 25, 13, 26, 0, tzinfo=timezone.utc)


def clamp_schedule(watchdog: datetime) -> tuple[datetime, datetime]:
    """Watchdog (sync + shutdown) and hard-stop backstop, both no later than the ceiling."""
    watchdog = min(watchdog, CEILING_UTC - timedelta(minutes=HARD_STOP_MINUTES))
    return watchdog, watchdog + timedelta(minutes=HARD_STOP_MINUTES)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def aws(*args: str, check: bool = True) -> dict:
    command = ["aws", "--profile", PROFILE, "--region", REGION, *args, "--output", "json"]
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if process.returncode != 0:
        if not check:
            return {"_error": process.stderr.strip()}
        raise RuntimeError(f"aws {' '.join(args[:2])} failed: {process.stderr.strip()[:500]}")
    return json.loads(process.stdout) if process.stdout.strip() else {}


def dry(*args: str) -> str:
    """EC2 DryRun: 'DryRunOperation' means the real call would be authorized."""
    result = aws(*args, "--dry-run", check=False)
    error = result.get("_error", "")
    if "DryRunOperation" in error:
        return "authorized"
    raise RuntimeError(f"dry run of {' '.join(args[:2])} failed: {error[:500]}")


def _tag_spec(resource: str, name: str, extra: dict | None = None) -> str:
    tags = [{"Key": k, "Value": v} for k, v in {**TAGS, "Name": name, **(extra or {})}.items()]
    return json.dumps([{"ResourceType": resource, "Tags": tags}])


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise FileExistsError(f"{path.name} is write-once; a study instance is already recorded")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def live_study_instances() -> list[dict]:
    found = aws("ec2", "describe-instances", "--filters", f"Name=tag:Study,Values={STUDY}",
                "Name=instance-state-name,Values=pending,running,stopping,stopped")
    return [{"id": i["InstanceId"], "state": i["State"]["Name"]}
            for r in found["Reservations"] for i in r["Instances"]]


def standard_vcpus_in_use() -> int:
    found = aws("ec2", "describe-instances",
                "--filters", "Name=instance-state-name,Values=pending,running")
    total = 0
    for reservation in found["Reservations"]:
        for item in reservation["Instances"]:
            if item["InstanceType"][0] in STANDARD_FAMILIES:
                cpu = item.get("CpuOptions", {})
                total += cpu.get("CoreCount", 0) * cpu.get("ThreadsPerCore", 1)
    return total


def preflight() -> dict:
    identity = aws("sts", "get-caller-identity")
    image = aws("ec2", "describe-images", "--image-ids", AMI)["Images"][0]
    profile = aws("iam", "get-instance-profile", "--instance-profile-name", INSTANCE_PROFILE)
    subnet = aws("ec2", "describe-subnets", "--subnet-ids", SUBNET)["Subnets"][0]
    quota = aws("service-quotas", "get-service-quota", "--service-code", "ec2",
                "--quota-code", "L-1216C47A")["Quota"]["Value"]
    in_use = standard_vcpus_in_use()
    versioning = aws("s3api", "get-bucket-versioning", "--bucket", BUCKET).get("Status")
    encryption = aws("s3api", "get-bucket-encryption", "--bucket", BUCKET)[
        "ServerSideEncryptionConfiguration"]["Rules"][0]["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"]
    ready_manifest = aws("s3api", "head-object", "--bucket", BUCKET,
                         "--key", f"{STUDY}/control/INPUT_STAGE.json", check=False)
    live = live_study_instances()
    record = {
        "caller_role": identity["Arn"].split("/")[-2] if "assumed-role" in identity["Arn"] else "user",
        "ami": {"id": AMI, "name": image["Name"], "state": image["State"],
                "architecture": image["Architecture"]},
        "instance_profile": {"name": INSTANCE_PROFILE,
                             "roles": [r["RoleName"] for r in profile["InstanceProfile"]["Roles"]]},
        "subnet": {"id": SUBNET, "az": subnet["AvailabilityZone"], "vpc": subnet["VpcId"],
                   "public_ip_on_launch": subnet["MapPublicIpOnLaunch"]},
        "standard_vcpu": {"quota": quota, "in_use": in_use, "needed": INSTANCE_VCPUS,
                          "fits": in_use + INSTANCE_VCPUS <= quota},
        "bucket": {"versioning": versioning, "default_sse": encryption},
        "input_manifest_staged": "_error" not in ready_manifest,
        "live_study_instances": live,
    }
    record["ok"] = (image["State"] == "available" and record["standard_vcpu"]["fits"] and
                    versioning == "Enabled" and not live and record["input_manifest_staged"])
    return record


# ---------------------------------------------------------------------------
# user data and launch
# ---------------------------------------------------------------------------

def verify_pushed(commit: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full 40-hex commit required")
    subprocess.run(["git", "fetch", "-q", "origin", f"refs/heads/{BRANCH}"], cwd=ROOT, check=True)
    tip = subprocess.check_output(["git", "rev-parse", "FETCH_HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, tip], cwd=ROOT).returncode:
        raise ValueError(f"commit {commit[:12]} is not on pushed origin/{BRANCH} (tip {tip[:12]})")


def manifest_pin() -> dict:
    head = aws("s3api", "head-object", "--bucket", BUCKET,
               "--key", f"{STUDY}/control/INPUT_STAGE.json")
    local = PRIVATE / "INPUT_STAGE.json"
    sha = hashlib.sha256(local.read_bytes()).hexdigest()
    return {"version_id": head["VersionId"], "sha256": sha}


def render_user_data(commit: str, watchdog: datetime, *, manifest_version: str,
                     manifest_sha: str) -> str:
    watchdog, hard_stop = clamp_schedule(watchdog)
    fmt = "%Y-%m-%d %H:%M:%S"
    values = {"__COMMIT__": commit, "__BRANCH__": BRANCH, "__REPO__": REPO,
              "__MANIFEST_VERSION__": manifest_version, "__MANIFEST_SHA__": manifest_sha,
              "__WATCHDOG_UTC__": watchdog.strftime(fmt), "__HARD_STOP_UTC__": hard_stop.strftime(fmt)}
    text = USER_DATA.read_text()
    for key, value in values.items():
        if not re.fullmatch(r"[A-Za-z0-9_.:/ -]+", value):
            raise ValueError(f"unsafe user-data value for {key}")
        text = text.replace(key, value)
    leftover = re.findall(r"__[A-Z_]+__", text)
    if leftover:
        raise ValueError(f"unrendered user-data placeholders {sorted(set(leftover))}")
    if len(text.encode()) > 16000:
        raise ValueError("user data exceeds the EC2 16 KB limit")
    return text


def launch(commit: str, *, watchdog_utc: str | None = None, dry_run: bool = True,
           skip_push_check: bool = False) -> dict:
    if not skip_push_check:
        verify_pushed(commit)
    check = preflight()
    if not check["ok"] and not dry_run:
        raise RuntimeError(f"preflight failed: {json.dumps(check, sort_keys=True)}")
    now = utcnow()
    watchdog = (datetime.strptime(watchdog_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                if watchdog_utc else now + timedelta(hours=DEFAULT_HOURS))
    watchdog, hard_stop = clamp_schedule(watchdog)
    if not now + timedelta(minutes=30) <= watchdog:
        raise ValueError("watchdog must be at least 30 minutes from now (and is clamped to the ceiling)")
    pin = manifest_pin()
    user_data = render_user_data(commit, watchdog, manifest_version=pin["version_id"],
                                 manifest_sha=pin["sha256"])
    PRIVATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    user_data_path = PRIVATE / ("user_data.dryrun.sh" if dry_run else "user_data.sh")
    user_data_path.write_text(user_data)
    vpc = aws("ec2", "describe-subnets", "--subnet-ids", SUBNET)["Subnets"][0]["VpcId"]
    deadline = {"DeadlineUTC": iso(watchdog)}
    mappings = json.dumps([{"DeviceName": "/dev/xvda", "Ebs": {
        "VolumeSize": VOLUME_GIB, "VolumeType": "gp3", "Encrypted": True,
        "DeleteOnTermination": True}}])
    base = ["ec2", "run-instances", "--image-id", AMI, "--instance-type", INSTANCE_TYPE,
            "--count", "1", "--subnet-id", SUBNET,
            "--iam-instance-profile", f"Name={INSTANCE_PROFILE}",
            "--instance-initiated-shutdown-behavior", "terminate",
            "--metadata-options", "HttpTokens=required,HttpEndpoint=enabled",
            "--block-device-mappings", mappings,
            "--user-data", f"file://{user_data_path}",
            "--tag-specifications",
            json.dumps(json.loads(_tag_spec("instance", "pcrl-sc-v1-worker", deadline)) +
                       json.loads(_tag_spec("volume", "pcrl-sc-v1-root")))]
    if dry_run:
        default_sg = aws("ec2", "describe-security-groups", "--filters", f"Name=vpc-id,Values={vpc}",
                         "Name=group-name,Values=default")["SecurityGroups"][0]["GroupId"]
        return {"dry_run": True, "preflight": check,
                "create_security_group": dry("ec2", "create-security-group", "--group-name",
                                             "pcrl-sc-v1-ssm-only", "--description",
                                             "PCRL shared-context SSM only, no ingress",
                                             "--vpc-id", vpc, "--tag-specifications",
                                             _tag_spec("security-group", "pcrl-sc-v1-ssm-only")),
                "run_instances": dry(*base, "--security-group-ids", default_sg),
                "watchdog_utc": iso(watchdog), "user_data_path": str(user_data_path),
                "user_data_sha256": hashlib.sha256(user_data.encode()).hexdigest()}
    if RESOURCE.exists():
        raise FileExistsError("a study instance is already recorded; refusing a second launch")
    group = aws("ec2", "create-security-group", "--group-name",
                f"pcrl-sc-v1-ssm-only-{now.strftime('%Y%m%dT%H%M%S')}",
                "--description", "PCRL shared-context SSM only, no ingress", "--vpc-id", vpc,
                "--tag-specifications", _tag_spec("security-group", "pcrl-sc-v1-ssm-only"))["GroupId"]
    ingress = aws("ec2", "describe-security-groups", "--group-ids", group)[
        "SecurityGroups"][0]["IpPermissions"]
    if ingress:
        raise RuntimeError("new security group unexpectedly has ingress rules")
    result = aws(*base, "--security-group-ids", group)
    item = result["Instances"][0]
    record = {"schema": "pcrl-sc-aws-v1", "study_tag": STUDY, "tags": TAGS, "region": REGION,
              "profile": PROFILE, "instance_id": item["InstanceId"], "type": INSTANCE_TYPE,
              "ami": AMI, "instance_profile": INSTANCE_PROFILE, "subnet": SUBNET,
              "security_group_id": group, "security_group_ingress_rules": 0,
              "volume_gib": VOLUME_GIB, "launch_utc": iso(now), "watchdog_utc": iso(watchdog),
              "hard_stop_utc": iso(hard_stop), "ceiling_utc": iso(CEILING_UTC),
              "shutdown_behavior": "terminate", "source_commit": commit, "branch": BRANCH,
              "input_manifest": pin,
              "user_data_sha256": hashlib.sha256(user_data.encode()).hexdigest(),
              "price_usd_per_hour": PRICE["c7i.8xlarge_usd_per_hour"]}
    _write_once(RESOURCE, record)
    return record


# ---------------------------------------------------------------------------
# SSM
# ---------------------------------------------------------------------------

def instance() -> str:
    record = json.loads(RESOURCE.read_text())
    if record.get("study_tag") != STUDY or record.get("region") != REGION:
        raise PermissionError("resource manifest is not for this study")
    ident = record["instance_id"]
    found = [i for r in aws("ec2", "describe-instances", "--instance-ids", ident)["Reservations"]
             for i in r["Instances"]]
    if len(found) != 1:
        raise ValueError("study instance could not be uniquely verified")
    tags = {t["Key"]: t["Value"] for t in found[0].get("Tags", [])}
    if tags.get("Study") != STUDY or tags.get("Project") != "pcrl":
        raise PermissionError("refusing untagged or unrelated instance")
    return ident


def send(commands: list[str], description: str, timeout_seconds: int = 3600) -> str:
    if not commands or not all(isinstance(c, str) for c in commands):
        raise ValueError("nonempty shell command list required")
    ident = instance()
    params = json.dumps({"commands": commands, "executionTimeout": [str(timeout_seconds)]})
    result = aws("ssm", "send-command", "--instance-ids", ident,
                 "--document-name", "AWS-RunShellScript", "--parameters", params,
                 "--timeout-seconds", str(min(timeout_seconds, 2592000)),
                 "--comment", description[:100])
    command_id = result["Command"]["CommandId"]
    COMMANDS.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with COMMANDS.open("a") as stream:
        stream.write(json.dumps({"utc": iso(utcnow()), "id": command_id,
                                 "description": description, "commands": commands},
                                sort_keys=True) + "\n")
    return command_id


def get(command_id: str) -> dict:
    result = aws("ssm", "get-command-invocation", "--command-id", command_id,
                 "--instance-id", instance())
    return {key: result.get(key) for key in
            ("Status", "ResponseCode", "StandardOutputContent", "StandardErrorContent")}


def wait(command_id: str, deadline_seconds: int = 3600) -> dict:
    end = time.monotonic() + deadline_seconds
    while time.monotonic() < end:
        result = get(command_id)
        if result["Status"] not in ("Pending", "InProgress", "Delayed"):
            return result
        time.sleep(5)
    raise TimeoutError("SSM command did not finish before local wait deadline")


def stage_code(commit: str) -> str:
    verify_pushed(commit)
    return send(["set -eu", "cd /opt/pcrl/work",
                 f"git fetch -q --depth 200 origin {BRANCH}",
                 f"git merge-base --is-ancestor {commit} FETCH_HEAD",
                 f"git checkout -q --detach {commit}",
                 f"test \"$(git rev-parse HEAD)\" = {commit}",
                 "git status --porcelain --untracked-files=no | (! grep .)",
                 f"printf '%s\\n' {commit} > /opt/pcrl/work/SOURCE_COMMIT.txt"],
                f"Pin PCRL source {commit[:12]}")


def start_runner(queue: str, workers: int, *, extra: list[str] | None = None) -> str:
    if not 1 <= workers <= INSTANCE_VCPUS:
        raise ValueError("workers must be between 1 and 32")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", queue):
        raise ValueError("unsafe queue path")
    units = f"/opt/pcrl/work/results/{STUDY}/private/units"
    if any(not re.fullmatch(r"[A-Za-z0-9_./=-]+", token) for token in (extra or [])):
        raise ValueError("unsafe extra runner argument")
    args = " ".join(extra or [])
    return send(["set -eu", "systemctl is-active pcrl-sc-watchdog.timer",
                 "test -f /opt/pcrl/logs/READY.json",
                 "systemctl reset-failed pcrl-sc-runner.service 2>/dev/null || true",
                 "systemd-run --unit=pcrl-sc-runner --working-directory=/opt/pcrl/work "
                 "--setenv=PYTHONPATH=/opt/pcrl/work --setenv=OMP_NUM_THREADS=1 "
                 "--setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 "
                 "--setenv=AWS_DEFAULT_REGION=us-east-1 "
                 f"/opt/pcrl/venv/bin/python -m experiments.{STUDY}.runner run --queue {queue} "
                 f"--units-root {units} --workers {workers} {args}",
                 "sleep 2", "systemctl status pcrl-sc-runner.service --no-pager | head -5"],
                f"Start SC queue runner x{workers}")


LOCKCHECK = "/opt/pcrl/lockcheck"
SPARSE = ("/experiments/__init__.py /experiments/pcrl_adaptive_release_v1/ "
          "/experiments/pcrl_task_aligned_cuts_v1/ /experiments/pcrl_task_directed_release_v1/ "
          "/experiments/pcrl_final_prospective_v1/ /experiments/pcrl_shared_context_release_v1/ "
          "/results/pcrl_adaptive_release_v1/ /results/pcrl_task_aligned_cuts_v1/ "
          "/results/pcrl_shared_context_release_v1/")


def lockcheck(commit: str, directory: str = LOCKCHECK) -> str:
    """Separate checkout at a pushed commit; /opt/pcrl/work is never touched."""
    verify_pushed(commit)
    if directory in ("/opt/pcrl/work", "/opt/pcrl") or not re.fullmatch(r"/opt/pcrl/[a-z0-9_-]+", directory):
        raise ValueError("separate checkout must be a new directory directly under /opt/pcrl")
    LOCKCHECK_ = directory
    return send(["set -eu",
                 f"if [ ! -d {LOCKCHECK_}/.git ]; then git clone -q --filter=blob:none --no-checkout "
                 f"--sparse --depth 200 --single-branch --branch {BRANCH} {REPO} {LOCKCHECK_}; fi",
                 f"cd {LOCKCHECK_}", f"git sparse-checkout set --no-cone {SPARSE}",
                 f"git fetch -q --depth 200 origin {BRANCH}",
                 f"git merge-base --is-ancestor {commit} FETCH_HEAD",
                 f"git checkout -q --detach {commit}",
                 f"test \"$(git rev-parse HEAD)\" = {commit}",
                 "git status --porcelain --untracked-files=no | (! grep .)",
                 f"printf '%s\\n' {commit} > {LOCKCHECK_}/SOURCE_COMMIT.txt",
                 "git -C /opt/pcrl/work rev-parse HEAD"],
                f"Separate checkout {directory} at {commit[:12]}")


def pull(remote_path: str, local_path: str, *, timeout_seconds: int = 600) -> dict:
    """Copy one host file to the Mac through the private bucket, SHA-verified end to end."""
    if not re.fullmatch(r"/opt/pcrl/[A-Za-z0-9_./-]+", remote_path) or ".." in remote_path:
        raise ValueError("remote path must be a plain file under /opt/pcrl")
    target = Path(local_path)
    key = f"{STUDY}/control/transfer/{utcnow().strftime('%Y%m%dT%H%M%SZ')}-{Path(remote_path).name}"
    command_id = send(["set -eu", f"test -f {remote_path}",
                       f"aws s3 cp {remote_path} s3://{BUCKET}/{key} --sse AES256 --only-show-errors",
                       f"sha256sum {remote_path} | cut -d' ' -f1"],
                      f"Pull {Path(remote_path).name}", timeout_seconds)
    result = wait(command_id, timeout_seconds + 60)
    if result["Status"] != "Success":
        raise RuntimeError(f"host upload failed: {result['StandardErrorContent'][-500:]}")
    digest = result["StandardOutputContent"].strip().splitlines()[-1]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError("host did not report a SHA-256")
    temporary = target.with_name(target.name + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    aws("s3", "cp", f"s3://{BUCKET}/{key}", str(temporary), "--only-show-errors")
    local = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if local != digest:
        temporary.unlink()
        raise ValueError("downloaded bytes differ from the host SHA-256")
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        temporary.unlink()
        raise FileExistsError(f"{target} exists with different bytes; not overwritten")
    os.replace(temporary, target)
    return {"remote": remote_path, "local": str(target), "sha256": digest, "s3_key": key}


def status() -> dict:
    ident = instance()
    item = aws("ec2", "describe-instances", "--instance-ids", ident)["Reservations"][0]["Instances"][0]
    out = {"instance_id": ident, "state": item["State"]["Name"],
           "launch_time": item.get("LaunchTime")}
    for name in ("READY.json", "BOOTSTRAP_FAILED", "WATCHDOG_ARMED", "STATUS.json"):
        head = aws("s3api", "head-object", "--bucket", BUCKET,
                   "--key", f"{STUDY}/control/{name}", check=False)
        out[name] = head.get("LastModified") if "_error" not in head else None
    return out


def terminate(confirm: str) -> dict:
    ident = instance()
    if confirm != ident:
        raise PermissionError("--confirm must repeat the recorded study instance id")
    return aws("ec2", "terminate-instances", "--instance-ids", ident)


# ---------------------------------------------------------------------------
# cost ledger
# ---------------------------------------------------------------------------

def _prefix_bytes(prefix: str) -> int:
    total, token = 0, None
    while True:
        args = ["s3api", "list-object-versions", "--bucket", BUCKET, "--prefix", prefix]
        if token:
            args += ["--key-marker", token[0], "--version-id-marker", token[1]]
        page = aws(*args)
        total += sum(int(v["Size"]) for v in page.get("Versions", []))
        if not page.get("IsTruncated"):
            return total
        token = (page["NextKeyMarker"], page["NextVersionIdMarker"])


def cost(record_path: Path = RESOURCE, *, now: datetime | None = None,
         state: str | None = None, end_utc: str | None = None,
         s3_bytes: int | None = None) -> dict:
    """Upper-bound spend: billed seconds x on-demand price, gp3 prorated, S3 month."""
    record = json.loads(Path(record_path).read_text())
    launch = datetime.strptime(record["launch_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    now = now or utcnow()
    if state is None:
        item = aws("ec2", "describe-instances", "--instance-ids", record["instance_id"],
                   check=False)
        found = [i for r in item.get("Reservations", []) for i in r["Instances"]]
        state = found[0]["State"]["Name"] if found else "unknown"
        reason = found[0].get("StateTransitionReason", "") if found else ""
        match = re.search(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) GMT\)", reason)
        if match and state in ("terminated", "shutting-down", "stopped"):
            end_utc = match.group(1).replace(" ", "T") + "Z"
    end = (datetime.strptime(end_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
           if end_utc else now)
    hours = max(0.0, (end - launch).total_seconds()) / 3600
    if s3_bytes is None:
        s3_bytes = _prefix_bytes(f"{STUDY}/")
    compute = hours * record["price_usd_per_hour"]
    ebs = record.get("volume_gib", VOLUME_GIB) * PRICE["gp3_usd_per_gib_month"] * hours / 730
    s3 = s3_bytes / 1e9 * PRICE["s3_standard_usd_per_gb_month"]
    entry = {"utc": iso(now), "instance_id": record["instance_id"], "state": state,
             "billed_hours_upper": math.ceil(hours * 3600) / 3600,
             "compute_usd_upper": round(compute, 4), "gp3_usd_upper": round(ebs, 4),
             "s3_prefix_all_version_bytes": s3_bytes, "s3_usd_per_month": round(s3, 4),
             "total_usd_upper_to_date": round(compute + ebs + s3, 4),
             "prices": PRICE, "end_utc": iso(end) if end_utc else None}
    return entry


def record_cost(entry: dict) -> None:
    ledger = json.loads(COST_LEDGER.read_text()) if COST_LEDGER.exists() else {
        "schema": "pcrl-sc-cost-ledger-v1", "ceiling_usd": 50, "entries": []}
    ledger["entries"].append(entry)
    COST_LEDGER.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    COST_LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("preflight")
    item = sub.add_parser("render"); item.add_argument("--commit", required=True)
    item.add_argument("--watchdog-utc")
    item = sub.add_parser("launch"); item.add_argument("--commit", required=True)
    item.add_argument("--watchdog-utc", help="YYYY-MM-DDTHH:MM:SSZ; default launch + 10 h")
    item.add_argument("--dry-run", action="store_true")
    item = sub.add_parser("send"); item.add_argument("description"); item.add_argument("command")
    item.add_argument("--timeout", type=int, default=3600)
    item = sub.add_parser("get"); item.add_argument("command_id")
    item = sub.add_parser("wait"); item.add_argument("command_id")
    item = sub.add_parser("stage-code"); item.add_argument("--commit", required=True)
    item = sub.add_parser("start-runner"); item.add_argument("--queue", required=True)
    item.add_argument("--workers", type=int, default=16)
    item.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    item = sub.add_parser("lockcheck"); item.add_argument("--commit", required=True)
    item.add_argument("--dir", default=LOCKCHECK, help="e.g. /opt/pcrl/posthoc")
    item = sub.add_parser("pull"); item.add_argument("--remote", required=True)
    item.add_argument("--local", required=True)
    sub.add_parser("status")
    item = sub.add_parser("terminate"); item.add_argument("--confirm", required=True)
    item = sub.add_parser("cost"); item.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "preflight":
        value = preflight()
    elif args.action == "render":
        watchdog = (datetime.strptime(args.watchdog_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    if args.watchdog_utc else utcnow() + timedelta(hours=DEFAULT_HOURS))
        pin = manifest_pin()
        sys.stdout.write(render_user_data(args.commit, watchdog, manifest_version=pin["version_id"],
                                          manifest_sha=pin["sha256"]))
        return
    elif args.action == "launch":
        value = launch(args.commit, watchdog_utc=args.watchdog_utc, dry_run=args.dry_run)
    elif args.action == "send":
        value = {"command_id": send(["set -eu", args.command], args.description, args.timeout)}
    elif args.action == "get":
        value = get(args.command_id)
    elif args.action == "wait":
        value = wait(args.command_id)
    elif args.action == "stage-code":
        value = {"command_id": stage_code(args.commit)}
    elif args.action == "start-runner":
        value = {"command_id": start_runner(args.queue, args.workers, extra=args.extra)}
    elif args.action == "lockcheck":
        value = {"command_id": lockcheck(args.commit, args.dir)}
    elif args.action == "pull":
        value = pull(args.remote, args.local)
    elif args.action == "status":
        value = status()
    elif args.action == "terminate":
        value = terminate(args.confirm)
    else:
        value = cost()
        if args.record:
            record_cost(value)
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
