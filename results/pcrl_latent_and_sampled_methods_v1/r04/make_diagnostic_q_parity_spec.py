"""Prepare a source-pinned R03 public-only diagnostic/Q parity unit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .independent_a_source import sha_file


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", required=True, type=Path)
    parser.add_argument("--unit-id", default="R04-A-DIAGNOSTIC-Q-PARITY-001")
    args = parser.parse_args()
    portfolio = args.portfolio.resolve(strict=True)
    campaign = HERE.parent
    sources = [HERE / "audit_diagnostic_q_parity.py", HERE / "independent_a_source.py",
               HERE / "make_diagnostic_q_parity_spec.py",
               campaign / "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json",
               campaign / "r01/SOURCE_AUDIT.json"]
    for seed in (0, 1):
        sources.extend([campaign / f"r01/A_FIXED_MAP_T_STAR_S{seed}.json",
                        campaign / f"r01/A_FIXED_MAP_T_STAR_S{seed}.npz",
                        campaign / f"r01/A_PUBLIC_FINE_CENTROIDS_S{seed}.npz"])
        bank_root = portfolio / f"shared/.private/tstar_score_bank_s{seed}_v1"
        sources.extend([bank_root / "BANK_ADMISSION_RECEIPT.json",
                        bank_root / "bank_r1.npz", bank_root / "bank_r2.npz"])
    sources = list(dict.fromkeys(path.resolve(strict=True) for path in sources))
    private = HERE / ".private"
    private.mkdir(mode=0o700, exist_ok=True)
    spec_path = private / f"{args.unit_id}_SPEC.json"
    output = HERE / "A_DIAGNOSTIC_Q_SOURCE_PARITY.json"
    if spec_path.exists() or output.exists():
        raise FileExistsError("diagnostic Q parity unit already prepared")
    spec = {"unit_id": args.unit_id, "stage": "verification",
            "argv": [sys.executable, "-m",
                     "results.pcrl_latent_and_sampled_methods_v1.r04.audit_diagnostic_q_parity",
                     "--campaign", str(campaign), "--portfolio", str(portfolio),
                     "--public-receipt", str(output)],
            "cwd": str(WORKTREE), "max_wall_s": 180, "cpu_reserve_s": 180,
            "max_rss_gib": 1, "projected_output_bytes": 12000,
            "sources": [{"path": str(path), "sha256": sha_file(path)} for path in sources],
            "deps": ["R04-HISTORICAL-SUCCESSOR-001"], "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    spec_path.chmod(0o600)
    print(spec_path)


if __name__ == "__main__":
    main()
