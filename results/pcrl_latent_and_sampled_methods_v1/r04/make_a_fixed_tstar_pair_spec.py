"""Build one immutable owner-only R03 verification spec after dated amendment seal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .independent_a_source import sha_file


HERE = Path(__file__).resolve().parent
CAMP = HERE.parent
WORKTREE = CAMP.parents[1]
PORTFOLIO = WORKTREE.parent / "pcrl-method-portfolio-v1/results/pcrl_method_portfolio_v1"
REPAIR = WORKTREE.parent / "pcrl-utility-reference-repair-v1/results/pcrl_utility_reference_repair_v1/m1/DATA_ACCESS_MANIFEST.json"
UNIT = "R04-A-TSTAR-S0-FIXED-PAIR-001"


def source_paths(amendment: Path) -> list[Path]:
    role_manifest = json.loads(REPAIR.read_text())
    bank_root = PORTFOLIO / "shared/.private/tstar_score_bank_s0_v1"
    paths = [HERE / "replay_a_fixed_tstar_pair.py", HERE / "independent_a_source.py",
             CAMP / "CAMPAIGN_LOCK.json", CAMP / "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_RETRY_001.json",
             CAMP / "A_DOWNSTREAM_SOURCE_MANIFEST_20261010_001.json",
             CAMP / "A_DOWNSTREAM_SOURCE_REGISTRATION_20261010_001.json",
             CAMP / "r01/SOURCE_AUDIT.json", REPAIR,
             bank_root / "BANK_ADMISSION_RECEIPT.json",
             bank_root / "bank_r1.npz", bank_root / "bank_r2.npz", amendment]
    for short, role in (("ATT", "ATTACK_FIT_NEW"), ("INNER", "INNER_SELECTION")):
        release = CAMP / "r03/.private/a_fixed_units" / f"A-REFERENCE-RELEASE-T_STAR-S0-{short}"
        queue = CAMP / "r03/.private/compute/units" / f"A-REFERENCE-RELEASE-T_STAR-S0-{short}"
        paths += [bank_root / f"tokens_{role}.npz",
                  REPAIR.parent.parent / role_manifest["role_file_relative_path"][f"{role}:references_s0"],
                  release / "reference_release.npz", release / "REFERENCE_RELEASE_RECEIPT.json",
                  queue / "RECEIPT.json"]
    return [p.resolve(strict=True) for p in paths]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amendment", type=Path, required=True)
    ap.add_argument("--unit-id", default=UNIT)
    args = ap.parse_args()
    amendment = args.amendment.resolve(strict=True)
    paths = source_paths(amendment)
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate source paths")
    private = HERE / ".private"
    private.mkdir(mode=0o700, exist_ok=True)
    spec_path = private / f"{args.unit_id}_SPEC.json"
    output = HERE / "A_TSTAR_S0_FIXED_PAIR_INDEPENDENT_REPLAY.json"
    if spec_path.exists() or output.exists():
        raise FileExistsError("verification spec/output already exists")
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_a_fixed_tstar_pair",
            "--campaign", str(CAMP.resolve(strict=True)),
            "--portfolio", str(PORTFOLIO.resolve(strict=True)),
            "--repair-manifest", str(REPAIR.resolve(strict=True)),
            "--amendment-sha256", sha_file(amendment), "--output", str(output)]
    spec = {"unit_id": args.unit_id, "stage": "verification",
            "amendment_sha256": sha_file(amendment),
            "deps": ["A-REFERENCE-RELEASE-T_STAR-S0-ATT",
                     "A-REFERENCE-RELEASE-T_STAR-S0-INNER"],
            "argv": argv, "cwd": str(WORKTREE.resolve(strict=True)),
            "max_wall_s": 120, "cpu_reserve_s": 120, "max_rss_gib": 2,
            "projected_output_bytes": 16000,
            "sources": [{"path": str(p), "sha256": sha_file(p)} for p in paths],
            "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"spec_path": str(spec_path), "spec_sha256": sha_file(spec_path),
                      "sources": len(paths), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
