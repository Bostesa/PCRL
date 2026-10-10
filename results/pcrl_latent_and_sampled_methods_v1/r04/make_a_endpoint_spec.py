"""Build a private R03 spec for independent A point and K=6,885 replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .independent_a_source import sha_file


HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
WORKTREE = CAMPAIGN.parents[1]
ENDPOINT_UNIT = "R02-A-K6885-ENDPOINT-001"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-lock", "a-amendment", "analysis-amendment",
                 "role-manifest", "endpoint-private-root", "endpoint-receipt",
                 "endpoint-json", "endpoint-csv"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--unit-id", required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name.replace("-", "_")).resolve(strict=True)
             for name in ("original-lock", "a-amendment", "analysis-amendment",
                          "role-manifest", "endpoint-private-root", "endpoint-receipt",
                          "endpoint-json", "endpoint-csv")}
    receipt = json.loads(paths["endpoint-receipt"].read_text())
    root = paths["endpoint-private-root"]
    binding_path = root / "A_SOURCE_BINDING.json"
    if (".private" not in root.parts or receipt.get("status") != "COMPLETE"
            or receipt.get("attack_instances") != 58
            or receipt.get("clauses") != 6885
            or sha_file(binding_path) != receipt.get("private_binding_sha256")
            or sha_file(paths["endpoint-json"]) != receipt.get("interval_json_sha256")
            or sha_file(paths["endpoint-csv"]) != receipt.get("point_csv_sha256")
            or args.unit_id != "R04-A-ENDPOINT-6885-001"):
        raise ValueError("completed A endpoint is not ready for independent replay")
    binding = json.loads(binding_path.read_text())
    if (binding.get("status") != "ALL_58_COMPLETE_VIEWS_VERIFIED"
            or len(binding.get("cells", [])) != 58):
        raise ValueError("A endpoint normalized private source inventory differs")
    manifest = json.loads(paths["role-manifest"].read_text())
    role_package = paths["role-manifest"].parent.parent
    source = [path for name, path in paths.items() if name != "endpoint-private-root"]
    source += [binding_path, CAMPAIGN / "r02/A_FAMILY_ORDER_S1.json",
              HERE / "audit_a_family_s1.py", HERE / "independent_a_intervals.py",
              HERE / "independent_d_intervals.py", HERE / "independent_a_source.py",
              HERE / "replay_a_prefit.py", HERE / "replay_a_endpoint.py"]
    for field in ("income", "occupation", "sex"):
        source.append(role_package / manifest["role_file_relative_path"][
            f"INNER_SELECTION:{field}"])
    for cell in binding["cells"]:
        item = cell["normalized_private_cell"]
        source.append(Path(item["private_file"]))
        attack = Path(cell["private_attack_root"])
        source.extend([attack / "ATTACK_ATTEMPT_INDEX.json",
                       attack / "common/FROZEN_COMPLETE.json"])
        for view in ("r1", "r2", "pair"):
            source.extend([attack / "common" / view / "readers/READERS_RECEIPT.json",
                           attack / "common" / view / "controls/CUSTODY_RECEIPT.json"])
    source = sorted({path.resolve(strict=True) for path in source})
    output = HERE / "A_ENDPOINT_6885_INDEPENDENT_REPLAY.json"
    spec_dir = HERE / ".private"
    spec_dir.mkdir(mode=0o700, exist_ok=True)
    spec_path = spec_dir / f"{args.unit_id}_SPEC.json"
    if output.exists() or spec_path.exists():
        raise FileExistsError("A independent endpoint spec or receipt already exists")
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_a_endpoint"]
    for name, path in {**paths, "campaign": CAMPAIGN,
                       "public-receipt": output}.items():
        argv.extend(("--" + name, str(path)))
    for name, path in (("original-lock", paths["original-lock"]),
                       ("a-amendment", paths["a-amendment"]),
                       ("analysis-amendment", paths["analysis-amendment"]),
                       ("endpoint-receipt", paths["endpoint-receipt"]),
                       ("role-manifest", paths["role-manifest"])):
        argv.extend(("--" + name + "-sha256", sha_file(path)))
    spec = {"unit_id": args.unit_id, "stage": "verification",
            "mode": "real_data", "deps": [ENDPOINT_UNIT], "argv": argv,
            "cwd": str(WORKTREE), "max_wall_s": 1800,
            "cpu_reserve_s": 1800, "max_rss_gib": 3,
            "projected_output_bytes": 1000000,
            "sources": [{"path": str(path), "sha256": sha_file(path)}
                        for path in source],
            "outputs": [str(output)],
            "env": {name: "1" for name in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}}
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    spec_path.chmod(0o600)
    print(f"{spec_path} {sha_file(spec_path)}")


if __name__ == "__main__":
    main()
