"""Construct immutable, source-pinned R1 selected raw replay queue specs.

This builder hashes files but never deserializes person-level archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sha(path)


def pin(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": sha(path)}


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--selected-unit-id", required=True)
    parser.add_argument("--policy-lock-sha256", required=True)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--queue-seal", required=True)
    parser.add_argument("--stream-roots", required=True)
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve(strict=True)
    root = repo / "results/pcrl_incumbent_view_risk_v1"
    python = Path(os.path.abspath(args.python))  # Preserve virtualenv symlink entrypoint.
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("virtualenv Python entrypoint missing")
    smoke = subprocess.run(
        [str(python), "-c", "import scipy, torch, hcal.calib, lra.deploy, incumbent_view.r1_selected_deploy"],
        cwd=repo, capture_output=True, text=True, timeout=30, check=False)
    if smoke.returncode:
        raise RuntimeError("exact virtualenv argv cannot import deployment dependencies")
    selected_dir = root / "r2/.private/units" / args.selected_unit_id
    policy_dir = selected_dir / "policy"
    selected_fit_dir = selected_dir / "OSF_DEFENSE_FIT_R0"
    policy_receipt = json.loads((policy_dir / "POLICY_RECEIPT.json").read_text())
    selected_receipt = json.loads((selected_fit_dir / "RELEASE_RECEIPT.json").read_text())
    if (policy_receipt["arm_id"] != args.arm_id or
            policy_receipt["source_seed"] != args.seed or
            policy_receipt["source_lock_sha256"] != args.policy_lock_sha256 or
            selected_receipt["policy_receipt_sha256"] != sha(policy_dir / "POLICY_RECEIPT.json")):
        raise ValueError("R2 selected policy/release identity differs")
    source_manifest = root / "r1/.private/P0_SOURCE_PATHS.json"
    source = json.loads(source_manifest.read_text())["seeds"][str(args.seed)]
    bank_record = json.loads(Path(source["public_bank"]["path"]).read_text())
    if (policy_receipt["public_bank_sha256"] != source["public_bank"]["sha256"] or
            policy_receipt["source_binding_sha256"] != bank_record["source_binding_sha256"]):
        raise ValueError("selected policy does not use exact P0 source bank")
    if (selected_receipt["arm_id"] != args.arm_id or
            selected_receipt["source_seed"] != args.seed or
            selected_receipt["role"] != "OSF_DEFENSE_FIT" or
            selected_receipt["realization"] != 0):
        raise ValueError("R2 selected release is not realization-0 FIT for this arm/seed")
    raw32 = root / "r1/.private" / f"R1_P0_S{args.seed}_ADMISSION/OSF_DEFENSE_FIT/sample_input.npz"
    essential = {
        "source_manifest": pin(source_manifest),
        "public_bank": pin(source["public_bank"]["path"]),
        "p0_fit": pin(source["roles"]["OSF_DEFENSE_FIT"]["path"]),
        "selected_fit": pin(selected_fit_dir / "release_codes.npz"),
        "selected_receipt": pin(selected_fit_dir / "RELEASE_RECEIPT.json"),
        "raw32": pin(raw32),
        "stream_roots": pin(args.stream_roots),
    }
    for field in ("public_bank", "p0_fit"):
        if essential[field]["sha256"] != (source["public_bank"]["sha256"] if field == "public_bank" else
                                           source["roles"]["OSF_DEFENSE_FIT"]["sha256"]):
            raise ValueError("R1 previously admitted source pin differs")
    if essential["selected_fit"]["sha256"] != selected_receipt["codes_file_sha256"]:
        raise ValueError("R2 cached FIT release bytes differ")

    code = [repo / "incumbent_view" / name for name in (
        "r1_bank.py", "r1_deploy.py", "r1_channel.py", "r1_selected_deploy.py", "r1_replay.py",
        "r1_build_replay_specs.py", "r2_space.py", "r2_policy.py", "r2_package.py")]
    for directory in ("qpc", "hcal", "lra",
                      "results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/impl",
                      "results/pcrl_latent_and_sampled_methods_v1/latent_repair_v1"):
        code.extend(sorted((repo / directory).rglob("*.py")))
    code.extend((repo / "results/pcrl_bank_anchored_release_v1/bank_anchored/r2_laws.py",
                 root / "r0/compute_queue.py", Path(args.queue_seal)))
    source_paths = [*(item["path"] for item in essential.values()),
                    *(item["path"] for item in source["pins"].values()),
                    str(policy_dir / "policy.npz"), str(policy_dir / "POLICY_RECEIPT.json"),
                    *map(str, code)]
    sources = [pin(p) for p in sorted(set(source_paths))]
    output_dir = root / "r1/.private/replay" / args.unit_id
    receipt_output = root / "r1" / f"{args.unit_id}_RECEIPT.json"
    if output_dir.exists() or receipt_output.exists():
        raise FileExistsError("replay unit output already exists")
    config = {"schema": "IVR1-r1-selected-replay-config-v1", "unit_id": args.unit_id,
              "repo": str(repo), "python": str(python), "source_seed": args.seed,
              "arm_id": args.arm_id, "policy_dir": str(policy_dir),
              "policy_receipt_sha256": sha(policy_dir / "POLICY_RECEIPT.json"),
              "policy_lock_sha256": args.policy_lock_sha256,
              "output_dir": str(output_dir), "receipt_output": str(receipt_output),
              "sources": sources, **essential}
    config_path = root / "r1/.private/specs" / f"{args.unit_id}_CONFIG.json"
    config_sha = write_new(config_path, config)
    queue_sources = sorted([*sources, pin(config_path)], key=lambda x: x["path"])
    seal_sha = sha(args.queue_seal)
    outputs = [str(receipt_output), str(output_dir / "recipient1.jsonl"),
               str(output_dir / "recipient2.jsonl")]
    if policy_receipt["law_kind"] not in ("incumbent_identity_v1", "incumbent_deterministic_v1"):
        outputs.append(str(output_dir / "uniforms.npz"))
    spec = {"unit_id": args.unit_id, "stage": "verification", "cwd": str(repo),
            "argv": [str(python), "-m", "incumbent_view.r1_replay", "--unit-id", args.unit_id,
                     "--config", str(config_path), "--config-sha256", config_sha],
            "max_wall_s": 120, "cpu_reserve_s": 120, "max_rss_gib": 2.,
            "projected_output_bytes": 2_000_000, "outputs": outputs,
            "deps": [args.selected_unit_id], "sources": queue_sources,
            "env": {}, "source_lock_sha256": seal_sha,
            "amendment_sha256": args.amendment_sha256}
    spec_path = root / "r1/.private/specs" / f"{args.unit_id}_QUEUE_SPEC.json"
    spec_sha = write_new(spec_path, spec)
    index = {"schema": "IVR1-r1-selected-raw32-spec-index-v1", "unit_id": args.unit_id,
             "selected_unit_id": args.selected_unit_id, "source_seed": args.seed,
             "arm_id": args.arm_id, "config_sha256": config_sha, "spec_sha256": spec_sha,
             "source_count": len(queue_sources), "source_lock_sha256": seal_sha,
             "amendment_sha256": args.amendment_sha256, "role_arrays_read": False,
             "queue_stage": "verification", "max_wall_s": 120, "max_rss_gib": 2.}
    public_path = root / "r1" / f"{args.unit_id}_SPEC_INDEX.json"
    write_new(public_path, index, 0o644)
    print(json.dumps({"unit_id": args.unit_id, "spec": str(spec_path),
                      "spec_sha256": spec_sha, "source_count": len(queue_sources)}, sort_keys=True))


if __name__ == "__main__":
    main()
