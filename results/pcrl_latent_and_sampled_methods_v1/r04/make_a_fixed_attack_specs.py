"""Freeze two owner-only R03 verification specs for completed fixed A attacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .independent_a_source import sha_file


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
WORKTREE = HERE.parents[2]
MODULES = ("independent_a_source.py", "independent_a_fixed_views.py",
           "independent_a_q.py", "independent_a_views.py",
           "independent_reader_custody.py", "independent_replay.py",
           "replay_a_attack.py", "replay_a_prefit.py",
           "replay_a_fixed_attacks.py")


def make(args: argparse.Namespace) -> list[dict]:
    producer_path = Path(args.producer).resolve(strict=True)
    completion_path = CAMPAIGN / "r03/A_FIXED_ATTACK_COMPLETION_RECEIPT.json"
    amendment_path = Path(args.verification_amendment).resolve(strict=True)
    manifest_path = Path(args.role_manifest).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    producer_sha, completion_sha = sha_file(producer_path), sha_file(completion_path)
    amendment_sha, manifest_sha = sha_file(amendment_path), sha_file(manifest_path)
    completion = json.loads(completion_path.read_text())
    if (completion.get("scientific_cell_count") != 18 or
            completion.get("producer_amendment_sha256") != producer_sha or
            len(completion.get("cells", [])) != 18):
        raise ValueError("completed 18-cell fixed attack inventory differs")
    out = []
    for seed in (0, 1):
        unit_id = f"R04-A-FIXED-ATTACK-S{seed}-001"
        cells = [c for c in completion["cells"] if c["scientific_cell_id"].endswith(f"-S{seed}")]
        if len(cells) != 9:
            raise ValueError("fixed attack seed inventory differs")
        dependencies = sorted(c["physical_unit_id"] for c in cells)
        files = [CAMPAIGN / "CAMPAIGN_LOCK.json", producer_path, completion_path,
                 amendment_path, manifest_path, *(HERE / name for name in MODULES)]
        role_root = manifest_path.parent.parent
        for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
            key = f"{role}:sex"
            files.append(role_root / manifest["role_file_relative_path"][key])
        for cell in cells:
            physical = cell["physical_unit_id"]
            arm = cell["scientific_cell_id"].split("-S")[0].split("ATTACK-")[-1]
            files += [p for p in (CAMPAIGN / "r03/.private/a_fixed_units" / physical).rglob("*")
                      if p.is_file()]
            prefix = "A-DIAG-RELEASE" if arm.startswith("A_") else "A-REFERENCE-RELEASE"
            for short in ("ATT", "INNER"):
                root = CAMPAIGN / "r03/.private/a_fixed_units" / f"{prefix}-{arm}-S{seed}-{short}"
                files += [p for p in root.rglob("*") if p.is_file()]
            if arm not in ("U0", "Ucal"):
                mapped = "T_STAR" if arm.startswith("A_") else arm
                files += [CAMPAIGN / "r01" / f"A_FIXED_MAP_{mapped}_S{seed}.json",
                          CAMPAIGN / "r01" / f"A_FIXED_MAP_{mapped}_S{seed}.npz"]
                qroot = CAMPAIGN / "r03/.private/a_q_units" / f"A-Q-PREPROCESS-{arm}-S{seed}"
                files += [p for p in qroot.rglob("*") if p.is_file()]
        unique = {str(p.resolve(strict=True)): p.resolve(strict=True) for p in files}
        sources = [{"path": key, "sha256": sha_file(value)} for key, value in sorted(unique.items())]
        output = HERE / f"A_FIXED_ATTACKS_S{seed}_INDEPENDENT_REPLAY.json"
        if output.exists():
            raise FileExistsError("fixed attack replay output exists")
        argv = [sys.executable, "-m",
                "results.pcrl_latent_and_sampled_methods_v1.r04.replay_a_fixed_attacks",
                "--campaign", str(CAMPAIGN.resolve()),
                "--original-lock-sha256", sha_file(CAMPAIGN / "CAMPAIGN_LOCK.json"),
                "--producer", str(producer_path), "--producer-sha256", producer_sha,
                "--completion", str(completion_path.resolve()),
                "--completion-sha256", completion_sha,
                "--verification-amendment", str(amendment_path),
                "--verification-amendment-sha256", amendment_sha,
                "--role-manifest", str(manifest_path),
                "--role-manifest-sha256", manifest_sha,
                "--unit-id", unit_id, "--seed", str(seed), "--output", str(output)]
        spec = {"unit_id": unit_id, "stage": "verification", "argv": argv,
                "cwd": str(WORKTREE), "deps": dependencies,
                "max_wall_s": 600, "cpu_reserve_s": 600, "max_rss_gib": 3,
                "projected_output_bytes": 1000000, "outputs": [str(output)],
                "sources": sources, "amendment_sha256": amendment_sha}
        spec_path = HERE / ".private" / f"{unit_id}_SPEC.json"
        if spec_path.exists():
            raise FileExistsError("fixed attack verification spec exists")
        spec_path.parent.mkdir(mode=0o700, exist_ok=True)
        spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
        spec_path.chmod(0o600)
        out.append({"unit_id": unit_id, "spec_path": str(spec_path),
                    "spec_sha256": sha_file(spec_path), "source_count": len(sources),
                    "dependency_count": len(dependencies), "output": str(output)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", required=True)
    parser.add_argument("--verification-amendment", required=True)
    parser.add_argument("--role-manifest", required=True)
    print(json.dumps(make(parser.parse_args()), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
