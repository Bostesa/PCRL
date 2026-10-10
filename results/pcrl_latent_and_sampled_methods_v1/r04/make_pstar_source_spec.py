"""Prepare an owner-only R03 spec for independent P* ATT/INNER source parity."""

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
    for name in ("seal", "role-manifest", "source-audit", "historical-audit",
                 "portfolio-package"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--unit-id", default="R04-PSTAR-SOURCE-001")
    args = parser.parse_args()
    if not args.unit_id.startswith("R04-PSTAR-SOURCE-"):
        raise ValueError("P* replay must use R04 namespace")
    paths = {name: getattr(args, name.replace("-", "_")).resolve(strict=True)
             for name in ("seal", "role-manifest", "source-audit",
                          "historical-audit", "portfolio-package")}
    historical = json.loads(paths["historical-audit"].read_text())
    role_manifest = json.loads(paths["role-manifest"].read_text())
    if historical.get("status") != "ROLE_ONLY_HISTORICAL_SCORE_SOURCES_VERIFIED":
        raise ValueError("R01 historical source audit not final")
    private = HERE / ".private"
    private.mkdir(mode=0o700, exist_ok=True)
    spec_path = private / f"{args.unit_id}_SPEC.json"
    output = HERE / "PSTAR_SOURCE_INDEPENDENT_REPLAY.json"
    if spec_path.exists() or output.exists():
        raise FileExistsError("P* source replay already prepared")
    source = [HERE / name for name in (
        "make_pstar_source_spec.py", "replay_pstar_source.py", "independent_pstar.py",
        "independent_a_source.py", "replay_a_prefit.py")]
    source.extend(path for name, path in paths.items() if name != "portfolio-package")
    source.append(paths["portfolio-package"] / "DATA_ACCESS_MANIFEST.json")
    for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        for seed in (0, 1):
            key = f"{role}:references_s{seed}"
            source.append(paths["role-manifest"].parent.parent /
                          role_manifest["role_file_relative_path"][key])
    for seed in (0, 1):
        entry = historical["pstar"][f"s{seed}"]
        source.append(Path(entry["receipt_path"]))
        source.extend(Path(record["path"]) for record in entry["source_paths_sha256"].values())
        source.extend(Path(record["path"]) for record in entry["banks"].values())
        source.extend(Path(entry["roles"][role]["path"])
                      for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"))
    source = [path.resolve(strict=True) for path in source]
    if len(source) != len(set(source)):
        raise ValueError("duplicate P* source inventory path")
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_pstar_source"]
    for name, path in paths.items():
        argv.extend(("--" + name, str(path)))
        if name not in ("portfolio-package",):
            argv.extend(("--" + name + "-sha256", sha_file(path)))
    argv.extend(("--public-receipt", str(output)))
    spec = {"unit_id": args.unit_id, "stage": "verification", "argv": argv,
            "cwd": str(WORKTREE), "max_wall_s": 180, "cpu_reserve_s": 180,
            "max_rss_gib": 1, "projected_output_bytes": 100000,
            "sources": [{"path": str(path), "sha256": sha_file(path)} for path in source],
            "deps": ["R04-D-INTERVALS-001"], "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    spec_path.chmod(0o600)
    print(spec_path)


if __name__ == "__main__":
    main()
