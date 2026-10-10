"""Prepare owner-only R03 spec for independent six-cell historical D0 utility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .independent_a_source import sha_file


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
ARMS = {"LOCAL": "hcal_local_d0", "SEQ12": "hcal_seq12_d0", "SEQ21": "lra_seq21"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "role-manifest", "source-audit", "historical-audit",
                 "portfolio-package", "cache-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--unit-id", default="R04-D0-UTILITY-001")
    args = parser.parse_args()
    if not args.unit_id.startswith("R04-D0-UTILITY-"):
        raise ValueError("D0 independent replay must use R04 namespace")
    paths = {name: getattr(args, name.replace("-", "_")).resolve(strict=True)
             for name in ("seal", "role-manifest", "source-audit",
                          "historical-audit", "portfolio-package", "cache-root")}
    historical = json.loads(paths["historical-audit"].read_text())
    role_manifest = json.loads(paths["role-manifest"].read_text())
    if historical.get("status") != "ROLE_ONLY_HISTORICAL_SCORE_SOURCES_VERIFIED":
        raise ValueError("historical D0 audit not frozen")
    private = HERE / ".private"
    private.mkdir(mode=0o700, exist_ok=True)
    spec_path = private / f"{args.unit_id}_SPEC.json"
    map_path = private / f"{args.unit_id}_R02_RECEIPT_SHA256.json"
    output = HERE / "D0_UTILITY_INDEPENDENT_REPLAY.json"
    if spec_path.exists() or map_path.exists() or output.exists():
        raise FileExistsError("D0 utility replay already prepared")
    source = [HERE / name for name in (
        "make_d0_utility_spec.py", "replay_d0_utility.py", "independent_a_source.py",
        "independent_replay.py", "replay_a_prefit.py")]
    source.extend(path for name, path in paths.items()
                  if name not in ("portfolio-package", "cache-root"))
    source.append(paths["portfolio-package"] / "DATA_ACCESS_MANIFEST.json")
    for field in ("references_s0", "references_s1", "income", "occupation"):
        key = f"INNER_SELECTION:{field}"
        source.append(paths["role-manifest"].parent.parent /
                      role_manifest["role_file_relative_path"][key])
    receipt_hashes = {}
    deps = []
    for arm, key in ARMS.items():
        for seed in (0, 1):
            entry = historical["d0"][key][f"s{seed}"]
            unit = f"R02-D0-UTILITY-{arm}-S{seed}"
            receipt = paths["seal"].parent / "r02" / f"{unit}_RECEIPT.json"
            receipt_hashes[unit] = sha_file(receipt)
            deps.append(unit)
            source.extend((receipt, paths["portfolio-package"] / entry["receipt_path"],
                           paths["portfolio-package"] / entry["roles"]["INNER_SELECTION"]["path"]))
            source.extend(paths["portfolio-package"] / record["path"]
                          for record in entry["banks"].values())
            source.extend(paths["cache-root"] / record["path"]
                          for record in entry["source_paths_sha256"].values())
    map_path.write_text(json.dumps(receipt_hashes, sort_keys=True, indent=2) + "\n")
    map_path.chmod(0o600)
    source.append(map_path)
    source = list(dict.fromkeys(path.resolve(strict=True) for path in source))
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_d0_utility"]
    for name, path in paths.items():
        argv.extend(("--" + name, str(path)))
        if name not in ("portfolio-package", "cache-root"):
            argv.extend(("--" + name + "-sha256", sha_file(path)))
    argv.extend(("--receipt-map", str(map_path), "--receipt-map-sha256", sha_file(map_path),
                 "--public-receipt", str(output)))
    spec = {"unit_id": args.unit_id, "stage": "verification", "argv": argv,
            "cwd": str(WORKTREE), "max_wall_s": 180, "cpu_reserve_s": 180,
            "max_rss_gib": 1, "projected_output_bytes": 100000,
            "sources": [{"path": str(path), "sha256": sha_file(path)} for path in source],
            "deps": deps, "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    spec_path.chmod(0o600)
    print(spec_path)


if __name__ == "__main__":
    main()
