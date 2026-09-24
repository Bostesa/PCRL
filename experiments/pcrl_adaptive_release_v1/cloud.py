"""Task-tagged, SSM-only execution and source transport for the owned worker."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "results/pcrl_adaptive_release_v1/private"
RESOURCE = PRIVATE / "AWS_RESOURCES.json"
STUDY = "pcrl_adaptive_release_v1"
BUCKET = "pcrl-ux-archive-ed9d21fd"
REGION = "us-east-1"
PROFILE = "vein"


def aws(*args: str) -> dict:
    command = ["aws", "--profile", PROFILE, "--region", REGION,
               *args, "--output", "json"]
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(process.stdout) if process.stdout.strip() else {}


def instance() -> str:
    record = json.loads(RESOURCE.read_text())
    if record.get("study_tag") != STUDY or record.get("region") != REGION:
        raise PermissionError("task resource manifest is not for this study")
    ident = record["instance_id"]
    result = aws("ec2", "describe-instances", "--instance-ids", ident)
    found = [value for reservation in result["Reservations"]
             for value in reservation["Instances"]]
    if len(found) != 1:
        raise ValueError("task instance could not be uniquely verified")
    current = found[0]
    if not any(tag["Key"] == "Study" and tag["Value"] == STUDY
               for tag in current.get("Tags", [])):
        raise PermissionError("refusing untagged or unrelated instance")
    return ident


def send(commands: list[str], description: str, timeout_seconds: int = 3600) -> str:
    if not commands or any(not isinstance(command, str) for command in commands):
        raise ValueError("nonempty shell command list required")
    ident = instance()
    params = json.dumps({"commands": commands,
                         "executionTimeout": [str(timeout_seconds)]})
    result = aws("ssm", "send-command", "--instance-ids", ident,
                 "--document-name", "AWS-RunShellScript", "--parameters", params,
                 "--timeout-seconds", str(timeout_seconds),
                 "--comment", description[:100])
    command_id = result["Command"]["CommandId"]
    PRIVATE.mkdir(parents=True, exist_ok=True)
    with (PRIVATE / "SSM_COMMANDS.jsonl").open("a") as stream:
        stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                                 "id": command_id, "description": description,
                                 "commands": commands}, sort_keys=True) + "\n")
    return command_id


def get(command_id: str) -> dict:
    result = aws("ssm", "get-command-invocation", "--command-id", command_id,
                 "--instance-id", instance())
    return {key: result.get(key) for key in
            ("Status", "ResponseCode", "StandardOutputContent",
             "StandardErrorContent")}


def wait(command_id: str, deadline_seconds: int = 3600) -> dict:
    end = time.monotonic() + deadline_seconds
    while time.monotonic() < end:
        result = get(command_id)
        if result["Status"] not in ("Pending", "InProgress", "Delayed"):
            return result
        time.sleep(5)
    raise TimeoutError("SSM command did not finish before local wait deadline")


def stage_source(commit: str, paths: list[str]) -> dict:
    """Upload only committed source at a pinned Git SHA, then extract on host."""
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    if commit != actual or len(commit) != 40:
        raise ValueError("source stage requires exact committed HEAD")
    staging = PRIVATE / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    archive = staging / f"source-{commit}.tar.gz"
    if not archive.exists():
        with archive.open("wb") as stream:
            subprocess.run(["git", "archive", "--format=tar.gz", commit, *paths],
                           cwd=ROOT, stdout=stream, check=True)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    key = f"{STUDY}/source/{archive.name}"
    aws("s3", "cp", str(archive), f"s3://{BUCKET}/{key}",
        "--sse", "AES256", "--only-show-errors")
    head = aws("s3api", "head-object", "--bucket", BUCKET, "--key", key)
    if head.get("ContentLength") != archive.stat().st_size or head.get("ServerSideEncryption") != "AES256":
        raise ValueError("source archive size/encryption verification failed")
    remote_archive = f"/opt/pcrl/source-{commit}.tar.gz"
    commands = ["set -eu", "mkdir -p /opt/pcrl/work",
                f"aws s3api get-object --bucket {BUCKET} --key {key} "
                f"--version-id {head['VersionId']} {remote_archive} >/dev/null",
                f"printf '%s  %s\\n' '{digest}' '{remote_archive}' | sha256sum -c -",
                f"tar -xzf {remote_archive} -C /opt/pcrl/work",
                f"printf '%s\\n' '{commit}' > /opt/pcrl/work/SOURCE_COMMIT.txt"]
    return {"command_id": send(commands, f"Stage committed PCRL source {commit[:12]}"),
            "git_commit": commit, "archive_sha256": digest,
            "s3_key": key, "s3_version_id": head["VersionId"],
            "bytes": archive.stat().st_size}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    item = sub.add_parser("get"); item.add_argument("command_id")
    item = sub.add_parser("wait"); item.add_argument("command_id")
    item = sub.add_parser("send"); item.add_argument("description"); item.add_argument("command")
    args = parser.parse_args()
    if args.action == "get":
        print(json.dumps(get(args.command_id), indent=2))
    elif args.action == "wait":
        print(json.dumps(wait(args.command_id), indent=2))
    else:
        print(send(["set -eu", args.command], args.description))
