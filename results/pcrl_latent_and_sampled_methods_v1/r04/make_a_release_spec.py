"""Prepare a private R03 spec for one independent A sampled release replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .independent_a_source import sha_file


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-lock", "a-amendment", "fit-dir", "release-dir",
                 "portfolio-package", "role-manifest", "source-audit",
                 "private-stream-roots", "public-stream-manifest"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--unit-id", required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name.replace("-", "_")).resolve(strict=True)
             for name in ("original-lock", "a-amendment", "fit-dir", "release-dir",
                          "portfolio-package", "role-manifest", "source-audit",
                          "private-stream-roots", "public-stream-manifest")}
    fit_dir, release_dir = paths["fit-dir"], paths["release-dir"]
    receipt = json.loads((release_dir / "RELEASE_RECEIPT.json").read_text())
    fit = json.loads((fit_dir / "FIT_RECEIPT.json").read_text())
    role, seed, arm, draw = (receipt["role"], receipt["source_seed"],
                             fit["arm_id"], receipt["realization"])
    short = {"ATTACK_FIT_NEW": "ATT", "INNER_SELECTION": "INNER"}.get(role)
    if short is None or receipt["status"] != "FROZEN_ONE_DRAW":
        raise ValueError("A sampled role release not complete")
    producer = f"A-RELEASE-{arm}-S{seed}-{short}-R{draw}"
    manifest = json.loads(paths["role-manifest"].read_text())
    source = [paths["original-lock"], paths["a-amendment"],
              paths["role-manifest"], paths["source-audit"],
              paths["private-stream-roots"], paths["public-stream-manifest"],
              paths["portfolio-package"] / "DATA_ACCESS_MANIFEST.json",
              fit_dir / "FIT_RECEIPT.json", fit_dir / "POLICY_RECEIPT.json",
              fit_dir / "policy.npz", release_dir / "RELEASE_RECEIPT.json",
              release_dir / "release_codes.npz", release_dir / "cache.sqlite3",
              HERE / "independent_a_source.py", HERE / "independent_a_inputs.py",
              HERE / "independent_a_law.py", HERE / "independent_a_stream.py",
              HERE / "independent_replay.py", HERE / "replay_a_prefit.py",
              HERE / "replay_a_release.py"]
    bank_root = paths["portfolio-package"] / f"shared/.private/tstar_score_bank_s{seed}_v1"
    source += [bank_root / "BANK_ADMISSION_RECEIPT.json",
               bank_root / "bank_r1.npz", bank_root / "bank_r2.npz",
               bank_root / f"tokens_{role}.npz"]
    kinds = [f"references_s{seed}"]
    if short == "INNER":
        kinds += ["income", "occupation"]
    for kind in kinds:
        key = f"{role}:{kind}"
        source.append(paths["role-manifest"].parent.parent /
                      manifest["role_file_relative_path"][key])
    source = [path.resolve(strict=True) for path in source]
    if len(set(source)) != len(source):
        raise ValueError("A sampled release verification source duplicated a path")
    output = HERE / f"A_RELEASE_{arm}_S{seed}_{short}_R{draw}_INDEPENDENT_REPLAY.json"
    spec_root = HERE / ".private"
    spec_root.mkdir(mode=0o700, exist_ok=True)
    spec_path = spec_root / f"{args.unit_id}_SPEC.json"
    if output.exists() or spec_path.exists():
        raise FileExistsError("A sampled release replay spec or receipt exists")
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_a_release"]
    flags = {**paths, "public-receipt": output}
    for name, path in flags.items():
        argv.extend(["--" + name, str(path)])
    hashes = {"original-lock-sha256": sha_file(paths["original-lock"]),
              "a-amendment-sha256": sha_file(paths["a-amendment"]),
              "fit-receipt-sha256": sha_file(fit_dir / "FIT_RECEIPT.json"),
              "policy-receipt-sha256": sha_file(fit_dir / "POLICY_RECEIPT.json"),
              "release-receipt-sha256": sha_file(release_dir / "RELEASE_RECEIPT.json"),
              "public-stream-manifest-sha256": sha_file(paths["public-stream-manifest"]),
              "role-manifest-sha256": sha_file(paths["role-manifest"]),
              "source-audit-sha256": sha_file(paths["source-audit"])}
    for name, value in hashes.items():
        argv.extend(["--" + name, value])
    spec = {"unit_id": args.unit_id, "stage": "verification",
            "deps": [producer], "argv": argv, "cwd": str(WORKTREE),
            "max_wall_s": 1200, "cpu_reserve_s": 1200, "max_rss_gib": 3,
            "projected_output_bytes": 250000,
            "sources": [{"path": str(path), "sha256": sha_file(path)} for path in source],
            "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    print(f"{spec_path} {sha_file(spec_path)}")


if __name__ == "__main__":
    main()
