"""Prepare owner-only R03 verification spec for the A S1 fixed family."""

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
    for name in ("manifest", "old-manifest", "d0-replay", "pstar-replay"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--unit-id", default="R04-A-S1-FAMILY-001")
    args = parser.parse_args()
    if not args.unit_id.startswith("R04-A-S1-FAMILY-"):
        raise ValueError("A family audit must use R04 namespace")
    paths = {name: getattr(args, name.replace("-", "_")).resolve(strict=True)
             for name in ("manifest", "old-manifest", "d0-replay", "pstar-replay")}
    private = HERE / ".private"
    private.mkdir(mode=0o700, exist_ok=True)
    spec_path = private / f"{args.unit_id}_SPEC.json"
    output = HERE / "A_S1_FAMILY_INDEPENDENT_AUDIT.json"
    if spec_path.exists() or output.exists():
        raise FileExistsError("A S1 family audit already prepared")
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.audit_a_family_s1",
            "--manifest", str(paths["manifest"]),
            "--manifest-sha256", sha_file(paths["manifest"])]
    for name in ("old-manifest", "d0-replay", "pstar-replay"):
        argv.extend(("--" + name, str(paths[name])))
    argv.extend(("--public-receipt", str(output)))
    source = [HERE / "make_a_family_s1_spec.py", HERE / "audit_a_family_s1.py",
              HERE / "independent_a_source.py", *paths.values()]
    spec = {"unit_id": args.unit_id, "stage": "verification", "argv": argv,
            "cwd": str(WORKTREE), "max_wall_s": 60, "cpu_reserve_s": 60,
            "max_rss_gib": 1, "projected_output_bytes": 20000,
            "sources": [{"path": str(path), "sha256": sha_file(path)} for path in source],
            "deps": ["R04-D0-UTILITY-001", "R04-PSTAR-SOURCE-001"],
            "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    spec_path.chmod(0o600)
    print(spec_path)


if __name__ == "__main__":
    main()
