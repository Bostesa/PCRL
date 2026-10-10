"""Build source-pinned queue spec for exact P0 raw83 CLI replay."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from .r1_build_replay_specs import pin, sha, write_new


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--amendment-sha256", required=True)
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve(strict=True)
    root = repo / "results/pcrl_incumbent_view_risk_v1"
    queue_lock = root / "r0/ADMISSION_LOCK.json"
    science_lock = root / "r0/CAMPAIGN_LOCK.json"
    python = Path(os.path.abspath(args.python))
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("virtualenv Python entrypoint missing")
    smoke = subprocess.run(
        [str(python), "-c", "import scipy, torch, hcal.calib, lra.deploy, incumbent_view.r1_deploy"],
        cwd=repo, capture_output=True, text=True, timeout=30, check=False)
    if smoke.returncode:
        raise RuntimeError("exact P0 deployment virtualenv imports fail")
    source_manifest = root / "r1/.private/P0_SOURCE_PATHS.json"
    source = json.loads(source_manifest.read_text())["seeds"][str(args.seed)]
    raw32 = root / "r1/.private" / f"R1_P0_S{args.seed}_ADMISSION/OSF_DEFENSE_FIT/sample_input.npz"
    essential = {"source_manifest": pin(source_manifest),
                 "public_bank": pin(source["public_bank"]["path"]),
                 "p0_fit": pin(source["roles"]["OSF_DEFENSE_FIT"]["path"]),
                 "raw32": pin(raw32)}
    if (essential["public_bank"]["sha256"] != source["public_bank"]["sha256"] or
            essential["p0_fit"]["sha256"] != source["roles"]["OSF_DEFENSE_FIT"]["sha256"]):
        raise ValueError("frozen P0 bank or FIT release bytes changed")
    code = [repo / "incumbent_view" / name for name in (
        "r1_bank.py", "r1_deploy.py", "r1_replay.py", "r1_p0_replay.py",
        "r1_build_replay_specs.py", "r1_build_p0_spec.py", "r2_package.py", "r2_policy.py", "r2_space.py")]
    for directory in ("qpc", "hcal", "lra", "dpc", "cbp", "jcv", "rgj",
                      "results/pcrl_latent_and_sampled_methods_v1/r01",
                      "results/pcrl_latent_and_sampled_methods_v1/r02",
                      "results/pcrl_latent_and_sampled_methods_v1/r03",
                      "results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/impl",
                      "results/pcrl_latent_and_sampled_methods_v1/sampled_channel_v1/controls",
                      "results/pcrl_latent_and_sampled_methods_v1/latent_repair_v1"):
        code.extend(sorted((repo / directory).rglob("*.py")))
    code.extend((repo / "results/pcrl_bank_anchored_release_v1/bank_anchored/r2_laws.py",
                 root / "r0/compute_queue.py", queue_lock, science_lock))
    paths = [*(item["path"] for item in essential.values()),
             *(item["path"] for item in source["pins"].values()), *map(str, code)]
    sources = [pin(path) for path in sorted(set(paths))]
    output_dir = root / "r1/.private/replay" / args.unit_id
    receipt_output = root / "r1" / f"{args.unit_id}_RECEIPT.json"
    if output_dir.exists() or receipt_output.exists():
        raise FileExistsError("write-once P0 replay output already exists")
    config = {"schema": "IVR1-r1-p0-replay-config-v1", "unit_id": args.unit_id,
              "repo": str(repo), "python": str(python), "source_seed": args.seed,
              "science_lock_sha256": sha(science_lock),
              "output_dir": str(output_dir), "receipt_output": str(receipt_output),
              "sources": sources, **essential}
    config_path = root / "r1/.private/specs" / f"{args.unit_id}_CONFIG.json"
    config_sha = write_new(config_path, config)
    queue_sources = sorted([*sources, pin(config_path)], key=lambda item: item["path"])
    outputs = [str(receipt_output), str(output_dir / "recipient1.jsonl"),
               str(output_dir / "recipient2.jsonl")]
    spec = {"unit_id": args.unit_id, "stage": "verification", "cwd": str(repo),
            "argv": [str(python), "-m", "incumbent_view.r1_p0_replay",
                     "--unit-id", args.unit_id, "--config", str(config_path),
                     "--config-sha256", config_sha],
            "max_wall_s": 120, "cpu_reserve_s": 120, "max_rss_gib": 2.,
            "projected_output_bytes": 2_000_000, "outputs": outputs,
            "deps": [f"R1_P0_S{args.seed}_ADMISSION"],
            "sources": queue_sources, "env": {},
            "source_lock_sha256": sha(queue_lock),
            "amendment_sha256": args.amendment_sha256}
    spec_path = root / "r1/.private/specs" / f"{args.unit_id}_QUEUE_SPEC.json"
    spec_sha = write_new(spec_path, spec)
    index = {"schema": "IVR1-r1-p0-raw32-spec-index-v1", "unit_id": args.unit_id,
             "source_seed": args.seed, "config_sha256": config_sha,
             "spec_sha256": spec_sha, "source_count": len(queue_sources),
             "queue_seal_sha256": sha(queue_lock), "science_lock_sha256": sha(science_lock),
             "amendment_sha256": args.amendment_sha256,
             "role_arrays_read": False, "queue_stage": "verification",
             "max_wall_s": 120, "max_rss_gib": 2.}
    write_new(root / "r1" / f"{args.unit_id}_SPEC_INDEX.json", index, 0o644)
    print(json.dumps({"unit_id": args.unit_id, "spec": str(spec_path),
                      "spec_sha256": spec_sha, "source_count": len(queue_sources)}, sort_keys=True))


if __name__ == "__main__":
    main()
