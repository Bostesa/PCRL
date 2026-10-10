"""Prepare hash-pinned queue specs for snapshot-only numerical replay.

The builder reads manifests and hashes bytes; it does not open private arrays.
R0 enqueues only after the prospective verification amendment and R5 audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pin(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("restored queue source missing or symlinked")
    return {"path": str(path.absolute()), "sha256": sha(path)}


def write_new(path: Path, value: dict, mode: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sha(path)


def selected_aliases(plan: dict, mode: str, arm: str, seed: int, draw: int) -> list[str]:
    aliases = [entry["alias"] for entry in plan["entries"]]
    chosen = [alias for alias in aliases if alias.startswith("code/")]
    common = ("locks/campaign.json", "locks/strongest_control.json",
              f"banks/s{seed}/P0_PUBLIC_BANK.json")
    chosen.extend(common)
    if mode == "release":
        prefixes = (f"frozen/s{seed}/", f"role/s{seed}/")
        chosen.extend(alias for alias in aliases if alias.startswith(prefixes))
        if arm != "P0":
            chosen.extend(("private/stream_roots.json",
                           f"policies/{arm}/s{seed}/policy.npz",
                           f"policies/{arm}/s{seed}/POLICY_RECEIPT.json",
                           f"policies/{arm}/s{seed}/draw{draw}/release_codes.npz",
                           f"policies/{arm}/s{seed}/draw{draw}/RELEASE_RECEIPT.json"))
    elif mode == "reader":
        chosen.extend(alias for alias in aliases if alias.startswith(f"reader/s{seed}/"))
    elif mode == "fresh_reader":
        unit = f"R4_FINAL_{arm}_S{seed}_R{draw}"
        chosen.extend((f"public/r4/{unit}_RECEIPT.json",
                       f"campaign/r4/attacks/{unit}/pair/readers/READERS_RECEIPT.json",
                       f"campaign/r4/attacks/{unit}/pair/readers/model__PROB_HGB_15x200.bin",
                       f"campaign/r4/attacks/{unit}/pair/readers/predictions_inner.npz",
                       f"campaign/r4/attacks/{unit}/pair/readers/view_inner.npz"))
        if arm == "P0":
            chosen.append(f"campaign/r1/R1_P0_S{seed}_ADMISSION/INNER_SELECTION/P0_RELEASE.npz")
        else:
            chosen.append(f"campaign/r2/units/R2_{arm}_S{seed}_FIT/"
                          f"INNER_SELECTION_R{draw}/release_codes.npz")
    else:
        raise ValueError("unsupported restored numerical replay mode")
    if len(set(chosen)) != len(chosen):
        raise ValueError("restored source aliases repeated")
    if not set(chosen) <= set(aliases):
        raise ValueError("restored source closure lacks required alias")
    return chosen


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("repo", "restore-root", "python", "unit-id", "mode", "arm-id",
                 "seed", "realization", "restore-receipt-sha256",
                 "amendment-sha256", "queue-seal"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve(strict=True)
    root = repo / "results/pcrl_incumbent_view_risk_v1"
    restore = Path(args.restore_root).resolve(strict=True)
    code_root = restore / "files/code"
    receipt_path = restore / "RESTORE_RECEIPT.json"
    if sha(receipt_path) != args.restore_receipt_sha256:
        raise ValueError("restored custody receipt differs")
    receipt = json.loads(receipt_path.read_text())
    plan_path = restore / "metadata/CUSTODY_PLAN.json"
    plan = json.loads(plan_path.read_text())
    if (sha(plan_path) != receipt["plan_sha256"] or
            plan["schema"] != "IVR1-r1-custody-plan-v1"):
        raise ValueError("restored plan identity differs")
    if args.mode not in ("release", "reader", "fresh_reader"):
        raise ValueError("invalid restored replay mode")
    if args.arm_id not in {"P0", plan["nominee"], plan["joint_diagnostic"],
                           plan["strongest_control"]}:
        raise ValueError("restored replay arm differs from locked selection")
    seed, draw = int(args.seed), int(args.realization)
    if seed not in (0, 1) or draw not in (0, 1, 2) or (
            args.mode == "fresh_reader" and draw != 0):
        raise ValueError("invalid representative seed/draw")
    if args.mode == "release" and args.arm_id == "P0" and draw != 0:
        raise ValueError("P0 release draws are deterministic aliases")
    if args.mode == "reader" and (args.arm_id != "P0" or draw != 0):
        raise ValueError("historical frozen reader replay is P0 draw 0")
    python = Path(os.path.abspath(args.python))  # Keep venv symlink entrypoint.
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("venv Python entrypoint missing")
    smoke = subprocess.run(
        [str(python), "-c", "import scipy, torch, joblib, incumbent_view.r1_restore_replay"],
        cwd=code_root, env={**os.environ, "PYTHONPATH": str(code_root), "PYTHONNOUSERSITE": "1"},
        capture_output=True, text=True, timeout=30, check=False)
    if smoke.returncode:
        raise RuntimeError("restored-only code import smoke failed")
    source_entries = {item["alias"]: item for item in plan["entries"]}
    sources = []
    for alias in selected_aliases(plan, args.mode, args.arm_id, seed, draw):
        path = restore / "files" / alias
        item = source_entries[alias]
        pinned = pin(path)
        if pinned["sha256"] != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise ValueError("restored source bytes differ from plan")
        sources.append(pinned)
    for path in (receipt_path, plan_path, restore / "runtime/P0_SOURCE_PATHS.json",
                 restore / "research.bundle"):
        sources.append(pin(path))
    private = root / "r1/.private/custody/verify" / args.unit_id
    public = root / "r1"
    output = private / "output"
    receipt_output = public / f"{args.unit_id}_RECEIPT.json"
    if private.exists() or receipt_output.exists():
        raise FileExistsError("write-once restored numerical unit exists")
    config = {"schema": "IVR1-r1-restored-replay-config-v1", "unit_id": args.unit_id,
              "restore_root": str(restore), "restore_receipt_sha256": args.restore_receipt_sha256,
              "python": str(python), "mode": args.mode, "seed": seed,
              "arm_id": args.arm_id, "realization": draw,
              "policy_lock_sha256": sha(restore / "files/locks/campaign.json"),
              "output_dir": str(output), "receipt_output": str(receipt_output)}
    config_path = private / "CONFIG.json"
    config_sha = write_new(config_path, config, 0o600)
    sources.append(pin(config_path))
    queue_seal = Path(args.queue_seal).resolve(strict=True)
    if sha(queue_seal) != sha(root / "r0/ADMISSION_LOCK.json"):
        raise ValueError("original queue seal differs")
    sources.append(pin(queue_seal))
    state = json.loads((root / "r0/.private/compute/state.json").read_text())
    latest = next((item for item in reversed(state["amendments"])
                   if "verification" in item["stages"]), None)
    if (latest is None or latest["sha256"] != args.amendment_sha256 or
            sha(Path(latest["path"])) != args.amendment_sha256):
        raise ValueError("latest verification amendment hash differs")
    sources.append(pin(Path(latest["path"])))
    outputs = [str(receipt_output)]
    if args.mode == "release":
        outputs.extend(str(output / f"recipient{i}.jsonl") for i in (1, 2))
        if args.arm_id not in ("P0", "V_D"):
            outputs.append(str(output / "uniforms.npz"))
    else:
        outputs.append(str(output / ("reader_predictions.npy" if args.mode == "reader"
                                    else "fresh_reader_predictions.npy")))
    spec = {"unit_id": args.unit_id, "stage": "verification",
            "cwd": str(code_root), "argv": [str(python), "-m",
            "incumbent_view.r1_restore_replay", "--unit-id", args.unit_id,
            "--config", str(config_path), "--config-sha256", config_sha],
            "max_wall_s": 120, "cpu_reserve_s": 120, "max_rss_gib": 2.0,
            "projected_output_bytes": 2_000_000, "outputs": outputs,
            "deps": [], "sources": sorted(sources, key=lambda x: x["path"]),
            "env": {"PYTHONPATH": str(code_root), "PYTHONNOUSERSITE": "1"},
            "source_lock_sha256": sha(queue_seal),
            "amendment_sha256": args.amendment_sha256}
    spec_path = private / "QUEUE_SPEC.json"
    spec_sha = write_new(spec_path, spec, 0o600)
    index = {"schema": "IVR1-r1-restored-numerical-spec-index-v1",
             "unit_id": args.unit_id, "mode": args.mode, "arm_id": args.arm_id,
             "seed": seed, "realization": draw, "config_sha256": config_sha,
             "spec_sha256": spec_sha, "source_count": len(sources),
             "restore_receipt_sha256": args.restore_receipt_sha256,
             "amendment_sha256": args.amendment_sha256}
    write_new(public / f"{args.unit_id}_SPEC_INDEX.json", index, 0o644)
    print(json.dumps({"spec": str(spec_path), "spec_sha256": spec_sha,
                      "source_count": len(sources)}, sort_keys=True))


if __name__ == "__main__":
    main()
