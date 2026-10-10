"""Receipt-only builder for the independent snapshot numerical replay.

It hashes restored bytes and writes an immutable queue config/spec. It does not
deserialize any role, policy, model, prediction, task, or SEX archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


UNIT = "R5_INDEPENDENT_SNAPSHOT_REPLAY_V1"
ARMS = ("P0", "V_L", "V_J", "V_S21")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("snapshot replay source missing or symlinked")
    return {"path": str(path.absolute()), "sha256": sha(path)}


def write_new(path: Path, value: dict, mode: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sha(path)


def required_aliases(plan: dict) -> list[str]:
    """Exact label-free aliases consumed by the independent checker."""
    aliases = [item["alias"] for item in plan["entries"] if item["alias"].startswith("code/")]
    aliases += ["locks/campaign.json", "locks/strongest_control.json", "private/stream_roots.json"]
    for seed in (0, 1):
        aliases += [f"banks/s{seed}/P0_PUBLIC_BANK.json", f"role/s{seed}/raw32.npz",
                    f"role/s{seed}/FIT_P0_RELEASE.npz",
                    f"frozen/s{seed}/teacher/model.pt",
                    f"frozen/s{seed}/teacher/head_0.joblib",
                    f"frozen/s{seed}/teacher/head_1.joblib",
                    f"frozen/s{seed}/teacher/COMPLETE.json",
                    f"frozen/s{seed}/policy/policy.json",
                    f"campaign/r1/R1_P0_S{seed}_ADMISSION/INNER_SELECTION/P0_RELEASE.npz"]
        for arm in ARMS:
            if arm != "P0":
                base = f"policies/{arm}/s{seed}/"
                aliases += [base + "policy.npz", base + "POLICY_RECEIPT.json",
                            f"campaign/r2/units/R2_{arm}_S{seed}_FIT/INNER_SELECTION_R0/release_codes.npz"]
                for draw in (0, 1, 2):
                    aliases.append(base + f"draw{draw}/release_codes.npz")
            unit = f"R4_FINAL_{arm}_S{seed}_R0"
            readers = f"campaign/r4/attacks/{unit}/pair/readers/"
            aliases += [readers + "READERS_RECEIPT.json", readers + "view_inner.npz",
                        readers + "model__PROB_HGB_15x200.bin",
                        readers + "predictions_inner.npz"]
            for draw in ((0,) if arm == "P0" else (0, 1, 2)):
                deployed = (f"R1_DEPLOY_P0_S{seed}_M" if arm == "P0" else
                            f"R1_DEPLOY_{arm}_S{seed}_R{draw}_{'O' if arm == 'V_S21' else 'M'}")
                aliases += [f"campaign/r1/replay/{deployed}/recipient1.jsonl",
                            f"campaign/r1/replay/{deployed}/recipient2.jsonl"]
    if len(aliases) != len(set(aliases)):
        raise ValueError("independent restore source alias repeated")
    return aliases


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("repo", "restore-root", "restore-receipt-sha256", "python",
                 "amendment-sha256", "queue-seal-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve(strict=True)
    restore = Path(args.restore_root).resolve(strict=True)
    code_root = restore / "files/code"
    results = repo / "results/pcrl_incumbent_view_risk_v1"
    receipt_path = restore / "RESTORE_RECEIPT.json"
    plan_path = restore / "metadata/CUSTODY_PLAN.json"
    if sha(receipt_path) != args.restore_receipt_sha256:
        raise PermissionError("restored receipt bytes differ")
    receipt = json.loads(receipt_path.read_text())
    plan = json.loads(plan_path.read_text())
    if (receipt.get("status") != "RESTORED_FROM_SNAPSHOT_ONLY" or
            receipt.get("plan_sha256") != sha(plan_path) or
            plan.get("nominee") != "V_L" or plan.get("joint_diagnostic") != "V_J" or
            plan.get("strongest_control") != "V_S21"):
        raise PermissionError("restored plan/selection differs")
    table = {item["alias"]: item for item in plan["entries"]}
    if len(table) != len(plan["entries"]):
        raise PermissionError("restored source alias duplicated")
    sources = []
    for alias in required_aliases(plan):
        if alias not in table or alias.startswith("/") or ".." in Path(alias).parts:
            raise PermissionError("required restored source absent or unsafe")
        path = restore / "files" / alias
        item = table[alias]
        actual = pin(path)
        if actual["sha256"] != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise PermissionError("restored source differs from plan")
        sources.append(actual)
    sources += [pin(receipt_path), pin(plan_path)]
    state_path = results / "r0/.private/compute/state.json"
    state = json.loads(state_path.read_text())
    seal = results / "r0/ADMISSION_LOCK.json"
    if (sha(seal) != args.queue_seal_sha256 or
            state["seal"]["sha256"] != args.queue_seal_sha256):
        raise PermissionError("queue seal differs")
    latest = [a for a in state["amendments"] if "verification" in a["stages"]][-1]
    if (latest["sha256"] != args.amendment_sha256 or
            sha(Path(latest["path"])) != args.amendment_sha256):
        raise PermissionError("latest verification amendment differs")
    sources += [pin(seal), pin(Path(latest["path"]))]
    python = Path(os.path.abspath(args.python))  # Preserve the venv entrypoint.
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError("venv Python entrypoint absent")
    smoke = subprocess.run(
        [str(python), "-c", "import numpy, torch, joblib, jcv.train, incumbent_view.r5_restore_independent"],
        cwd=code_root, env={**os.environ, "PYTHONPATH": str(code_root), "PYTHONNOUSERSITE": "1"},
        capture_output=True, text=True, timeout=30, check=False)
    if smoke.returncode:
        raise RuntimeError("isolated restored R5 checker import smoke failed")
    config_path = results / "r5/.private/R5_INDEPENDENT_SNAPSHOT_REPLAY_V1_CONFIG.json"
    spec_path = results / "r5/.private/R5_INDEPENDENT_SNAPSHOT_REPLAY_V1_QUEUE_SPEC.json"
    output = results / "r5/R5_INDEPENDENT_SNAPSHOT_REPLAY_V1.json"
    if spec_path.exists() or output.exists():
        raise FileExistsError("write-once R5 restore replay already exists")
    config = {"schema": "IVR1-r5-independent-restore-config-v1", "unit_id": UNIT,
              "repo": str(repo), "restore_root": str(restore),
              "restore_receipt_sha256": args.restore_receipt_sha256,
              "plan_sha256": sha(plan_path),
              "campaign_lock_sha256": sha(restore / "files/locks/campaign.json"),
              "control_lock_sha256": sha(restore / "files/locks/strongest_control.json"),
              "admission_lock_sha256": args.queue_seal_sha256,
              "amendment_sha256": args.amendment_sha256,
              "output": str(output)}
    write_new(config_path, config, 0o600)
    sources.append(pin(config_path))
    spec = {"unit_id": UNIT, "stage": "verification", "cwd": str(code_root),
            "argv": [str(python), "-m", "incumbent_view.r5_restore_independent",
                     "--config", str(config_path)],
            "max_wall_s": 600, "cpu_reserve_s": 600, "max_rss_gib": 3.0,
            "projected_output_bytes": 2_000_000, "outputs": [str(output)], "deps": [],
            "sources": sorted(sources, key=lambda item: item["path"]),
            "env": {"PYTHONPATH": str(code_root), "PYTHONNOUSERSITE": "1"},
            "amendment_sha256": args.amendment_sha256}
    write_new(spec_path, spec, 0o600)
    index = {"schema": "IVR1-r5-independent-restored-replay-spec-index-v1",
             "unit_id": UNIT, "spec_sha256": sha(spec_path), "config_sha256": sha(config_path),
             "source_count": len(sources), "restore_receipt_sha256": args.restore_receipt_sha256,
             "amendment_sha256": args.amendment_sha256}
    write_new(results / "r5/R5_INDEPENDENT_SNAPSHOT_REPLAY_V1_SPEC_INDEX.json", index, 0o644)
    print(json.dumps(index, sort_keys=True))


if __name__ == "__main__":
    main()
