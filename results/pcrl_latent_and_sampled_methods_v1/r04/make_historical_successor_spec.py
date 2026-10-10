"""Prepare owner-only R03 spec for the R01 path-only successor audit."""

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
    for name in ("old-audit", "new-audit", "custody-amendment",
                 "portfolio-package", "cache-root", "campaign"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--unit-id", default="R04-HISTORICAL-SUCCESSOR-001")
    args = parser.parse_args()
    paths = {name: getattr(args, name.replace("-", "_")).resolve(strict=True)
             for name in ("old-audit", "new-audit", "custody-amendment",
                          "portfolio-package", "cache-root", "campaign")}
    new = json.loads(paths["new-audit"].read_text())
    amendment = json.loads(paths["custody-amendment"].read_text())
    private = HERE / ".private"
    private.mkdir(mode=0o700, exist_ok=True)
    spec_path = private / f"{args.unit_id}_SPEC.json"
    output = HERE / "HISTORICAL_AUDIT_SUCCESSOR_REPLAY.json"
    if spec_path.exists() or output.exists():
        raise FileExistsError("historical successor unit already prepared")
    source = [HERE / "make_historical_successor_spec.py",
              HERE / "audit_historical_successor.py", HERE / "independent_a_source.py",
              paths["old-audit"], paths["new-audit"], paths["custody-amendment"]]
    for section in ("pstar", "d0"):
        entries = ([new[section][f"s{seed}"] for seed in (0, 1)] if section == "pstar"
                   else [new[section][key][f"s{seed}"] for key in new[section]
                         for seed in (0, 1)])
        for entry in entries:
            source.append(paths["portfolio-package"] / entry["receipt_path"])
            source.extend(paths["portfolio-package"] / v["path"]
                          for field in ("banks", "roles") for v in entry[field].values())
            source.extend(paths["cache-root"] / v["path"]
                          for v in entry["source_paths_sha256"].values())
    for seed in (0, 1):
        entry = new["public_centroids"][f"s{seed}"]
        source.extend((paths["campaign"] / entry["clean_relative_path"],
                       paths["cache-root"] / entry["source_fine_cache_relative_path"]))
    for name in amendment["fixed_public_map_receipt_sha256_changes"]:
        receipt = paths["campaign"] / "r01" / name
        source.extend((receipt, receipt.with_suffix(".npz")))
    source = list(dict.fromkeys(path.resolve(strict=True) for path in source))
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.audit_historical_successor"]
    for name, path in paths.items():
        argv.extend(("--" + name, str(path)))
    argv.extend(("--public-receipt", str(output)))
    spec = {"unit_id": args.unit_id, "stage": "verification", "argv": argv,
            "cwd": str(WORKTREE), "max_wall_s": 180, "cpu_reserve_s": 180,
            "max_rss_gib": 1, "projected_output_bytes": 10000,
            "sources": [{"path": str(path), "sha256": sha_file(path)} for path in source],
            "deps": ["R04-D0-UTILITY-001", "R04-PSTAR-SOURCE-001"],
            "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    spec_path.chmod(0o600)
    print(spec_path)


if __name__ == "__main__":
    main()
