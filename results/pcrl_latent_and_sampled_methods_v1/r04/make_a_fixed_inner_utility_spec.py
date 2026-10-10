"""Freeze a private R03 spec for saved fixed A INNER task utility replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .independent_a_source import sha_file
from .make_a_fixed_tstar_pair_spec import CAMP, HERE, PORTFOLIO, REPAIR, WORKTREE


UNIT = "R04-A-FIXED-INNER-UTILITY-001"
ARMS = ("T_STAR", "A_IDENTITY", "A_CLASS_ONLY")


def sources(amendment: Path) -> list[Path]:
    manifest = json.loads(REPAIR.read_text())
    paths = [HERE / name for name in (
        "replay_a_fixed_inner_utility.py", "audit_diagnostic_q_parity.py",
        "replay_a_fixed_tstar_pair.py", "independent_a_source.py",
        "independent_replay.py", "A_PREFIT_INDEPENDENT_REPLAY.json",
        "A_DIAGNOSTIC_Q_SOURCE_PARITY.json")]
    paths += [CAMP / "CAMPAIGN_LOCK.json",
              CAMP / "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_RETRY_001.json",
              CAMP / "r01/SOURCE_AUDIT.json", REPAIR, amendment]
    for kind in ("income", "occupation", "references_s0", "references_s1"):
        key = f"INNER_SELECTION:{kind}"
        paths.append(REPAIR.parent.parent / manifest["role_file_relative_path"][key])
    for seed in (0, 1):
        bank = PORTFOLIO / f"shared/.private/tstar_score_bank_s{seed}_v1"
        paths += [bank / "BANK_ADMISSION_RECEIPT.json", bank / "bank_r1.npz",
                  bank / "bank_r2.npz", bank / "tokens_INNER_SELECTION.npz"]
        for arm in ARMS:
            kind = "REFERENCE" if arm == "T_STAR" else "DIAG"
            unit = f"A-{kind}-RELEASE-{arm}-S{seed}-INNER"
            root = CAMP / "r03/.private/a_fixed_units" / unit
            queue = CAMP / "r03/.private/compute/units" / unit
            paths += [root / "reference_release.npz",
                      root / "REFERENCE_RELEASE_RECEIPT.json",
                      queue / "RECEIPT.json"]
    return [path.resolve(strict=True) for path in paths]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amendment", type=Path, required=True)
    args = ap.parse_args()
    amendment = args.amendment.resolve(strict=True)
    paths = sources(amendment)
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate source path")
    private = HERE / ".private"
    private.mkdir(mode=0o700, exist_ok=True)
    spec_path = private / f"{UNIT}_SPEC.json"
    output = HERE / "A_FIXED_INNER_UTILITY_INDEPENDENT_REPLAY.json"
    if spec_path.exists() or output.exists():
        raise FileExistsError("fixed INNER utility spec/output exists")
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_a_fixed_inner_utility",
            "--campaign", str(CAMP.resolve(strict=True)),
            "--portfolio", str(PORTFOLIO.resolve(strict=True)),
            "--repair-manifest", str(REPAIR.resolve(strict=True)),
            "--amendment-sha256", sha_file(amendment), "--output", str(output)]
    deps = [f"A-{'REFERENCE' if arm == 'T_STAR' else 'DIAG'}-RELEASE-{arm}-S{seed}-INNER"
            for seed in (0, 1) for arm in ARMS]
    spec = {"unit_id": UNIT, "stage": "verification",
            "amendment_sha256": sha_file(amendment), "deps": deps,
            "argv": argv, "cwd": str(WORKTREE.resolve(strict=True)),
            "max_wall_s": 120, "cpu_reserve_s": 120, "max_rss_gib": 2,
            "projected_output_bytes": 100000,
            "sources": [{"path": str(path), "sha256": sha_file(path)} for path in paths],
            "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"spec": str(spec_path), "sha256": sha_file(spec_path),
                      "sources": len(paths)}, sort_keys=True))


if __name__ == "__main__":
    main()
