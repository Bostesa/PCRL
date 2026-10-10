"""Freeze one owner-only R03 spec for the independent fixed A endpoint replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .independent_a_source import sha_file


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
WORKTREE = HERE.parents[2]
UNIT = "R04-A-FIXED-ENDPOINT-001"
MODULES = ("audit_a_family_s1.py", "independent_a_intervals.py",
           "independent_a_source.py", "independent_d_intervals.py",
           "independent_replay.py", "replay_a_endpoint.py",
           "replay_a_prefit.py", "replay_a_fixed_endpoint.py")


def make(args: argparse.Namespace) -> dict:
    producer_path = Path(args.producer).resolve(strict=True)
    analysis_path = Path(args.analysis_amendment).resolve(strict=True)
    verification_path = Path(args.verification_amendment).resolve(strict=True)
    role_path = Path(args.role_manifest).resolve(strict=True)
    receipt_path = Path(args.endpoint_receipt).resolve(strict=True)
    endpoint_json = Path(args.endpoint_json).resolve(strict=True)
    endpoint_csv = Path(args.endpoint_csv).resolve(strict=True)
    private_root = Path(args.endpoint_private_root).resolve(strict=True)
    producer_sha = sha_file(producer_path)
    analysis_sha = sha_file(analysis_path)
    verification_sha = sha_file(verification_path)
    role_sha = sha_file(role_path)
    receipt_sha = sha_file(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    verification = json.loads(verification_path.read_text())
    binding_path = private_root / "A_SOURCE_BINDING.json"
    binding = json.loads(binding_path.read_text())
    if (verification.get("status") != "LOCALLY_SEALED_R04_A_FIXED_ENDPOINT_REPLAY" or
            verification.get("authorized_unit_ids") != [UNIT] or
            verification.get("producer_amendment_sha256") != producer_sha or
            verification.get("endpoint_analysis_amendment_sha256") != analysis_sha or
            verification.get("endpoint_receipt_sha256") != receipt_sha or
            receipt.get("status") != "INCOMPLETE_FITTED_ARM_SOURCES" or
            receipt.get("private_binding_sha256") != sha_file(binding_path) or
            len(binding.get("fixed_cells", [])) != 18 or
            binding.get("analysis_amendment_sha256") != analysis_sha):
        raise ValueError("fixed A endpoint verification admission differs")
    files = [CAMPAIGN / "CAMPAIGN_LOCK.json", producer_path, analysis_path,
             verification_path, role_path, receipt_path, endpoint_json,
             endpoint_csv, binding_path,
             CAMPAIGN / "r02/A_FAMILY_ORDER_S1.json",
             CAMPAIGN / "r03/A_FIXED_ATTACK_COMPLETION_RECEIPT.json",
             *(HERE / f"A_FIXED_ATTACKS_S{seed}_INDEPENDENT_REPLAY_V4.json"
               for seed in (0, 1)), *(HERE / name for name in MODULES)]
    manifest = json.loads(role_path.read_text())
    for field in ("income", "occupation", "sex"):
        files.append(role_path.parent.parent / manifest["role_file_relative_path"][
            f"INNER_SELECTION:{field}"])
    for cell in binding["fixed_cells"]:
        normalized = Path(cell["normalized_private_cell"]["private_file"])
        files.append(normalized)
        attack_root = Path(cell["private_attack_root"])
        files += [attack_root / "ATTACK_ATTEMPT_INDEX.json",
                  attack_root / "common/FROZEN_COMPLETE.json"]
        for view in ("r1", "r2", "pair"):
            files += [attack_root / "common" / view / "readers/READERS_RECEIPT.json",
                      attack_root / "common" / view / "controls/CUSTODY_RECEIPT.json"]
    unique = {str(p.resolve(strict=True)): p.resolve(strict=True) for p in files}
    sources = [{"path": key, "sha256": sha_file(path)}
               for key, path in sorted(unique.items())]
    output = HERE / "A_FIXED_ENDPOINT_INDEPENDENT_REPLAY.json"
    if output.exists():
        raise FileExistsError("fixed A endpoint replay output exists")
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_a_fixed_endpoint",
            "--campaign", str(CAMPAIGN.resolve()),
            "--original-lock-sha256", sha_file(CAMPAIGN / "CAMPAIGN_LOCK.json"),
            "--producer", str(producer_path), "--producer-sha256", producer_sha,
            "--analysis-amendment", str(analysis_path),
            "--analysis-amendment-sha256", analysis_sha,
            "--verification-amendment", str(verification_path),
            "--verification-amendment-sha256", verification_sha,
            "--endpoint-private-root", str(private_root),
            "--endpoint-receipt", str(receipt_path),
            "--endpoint-receipt-sha256", receipt_sha,
            "--endpoint-json", str(endpoint_json), "--endpoint-csv", str(endpoint_csv),
            "--role-manifest", str(role_path), "--role-manifest-sha256", role_sha,
            "--completion-sha256", sha_file(CAMPAIGN / "r03/A_FIXED_ATTACK_COMPLETION_RECEIPT.json"),
            "--independent-attack-s0-sha256",
            sha_file(HERE / "A_FIXED_ATTACKS_S0_INDEPENDENT_REPLAY_V4.json"),
            "--independent-attack-s1-sha256",
            sha_file(HERE / "A_FIXED_ATTACKS_S1_INDEPENDENT_REPLAY_V4.json"),
            "--unit-id", UNIT, "--output", str(output)]
    spec = {"unit_id": UNIT, "stage": "verification", "argv": argv,
            "cwd": str(WORKTREE), "deps": ["R02-A-K6885-FIXED-ENDPOINT-001-R2",
                                           "R04-A-FIXED-ATTACK-S0-004",
                                           "R04-A-FIXED-ATTACK-S1-004"],
            "max_wall_s": 900, "cpu_reserve_s": 900, "max_rss_gib": 3,
            "projected_output_bytes": 1000000, "outputs": [str(output)],
            "sources": sources, "amendment_sha256": verification_sha}
    spec_path = HERE / ".private" / f"{UNIT}_SPEC.json"
    if spec_path.exists():
        raise FileExistsError("fixed A endpoint replay spec exists")
    spec_path.parent.mkdir(mode=0o700, exist_ok=True)
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    spec_path.chmod(0o600)
    return {"unit_id": UNIT, "spec_path": str(spec_path),
            "spec_sha256": sha_file(spec_path), "source_count": len(sources),
            "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("producer", "analysis-amendment", "verification-amendment",
                 "role-manifest", "endpoint-receipt", "endpoint-json",
                 "endpoint-csv", "endpoint-private-root"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(make(parser.parse_args()), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
