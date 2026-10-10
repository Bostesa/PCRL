"""Prepare owner-only R03 verification of one sampled A complete-view attack."""

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
    for name in ("original-lock", "a-amendment", "attack-root", "fit-dir",
                 "att-release-dir", "inner-release-dir", "att-release-replay",
                 "inner-release-replay", "q-table-dir", "portfolio-package",
                 "role-manifest", "source-audit"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("unit-id", "att-release-replay-unit", "inner-release-replay-unit"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    path_names = ("original-lock", "a-amendment", "attack-root", "fit-dir",
                  "att-release-dir", "inner-release-dir", "att-release-replay",
                  "inner-release-replay", "q-table-dir", "portfolio-package",
                  "role-manifest", "source-audit")
    paths = {name: getattr(args, name.replace("-", "_")).resolve(strict=True)
             for name in path_names}
    attack = json.loads((paths["attack-root"] / "ATTACK_ATTEMPT_INDEX.json").read_text())
    arm, seed, draw = attack["arm_id"], attack["source_seed"], attack["realization"]
    if (attack.get("schema") != "SCV1-sampled-attack-attempt-v1"
            or attack.get("status") != "COMPLETE"
            or attack.get("a_amendment_sha256") != sha_file(paths["a-amendment"])
            or not args.unit_id.startswith("R04-A-ATTACK-")):
        raise ValueError("completed sampled A attack not ready for independent replay")
    expected_attack = f"A-ATTACK-{arm}-S{seed}-R{draw}"
    if attack.get("queue_unit_id") != expected_attack:
        raise ValueError("A attack producer ID differs from sealed Cartesian plan")
    manifest = json.loads(paths["role-manifest"].read_text())
    role_package = paths["role-manifest"].parent.parent
    source = [paths["original-lock"], paths["a-amendment"],
              paths["role-manifest"], paths["source-audit"],
              paths["portfolio-package"] / "DATA_ACCESS_MANIFEST.json",
              paths["fit-dir"] / "POLICY_RECEIPT.json",
              paths["fit-dir"] / "policy.npz",
              paths["att-release-replay"], paths["inner-release-replay"],
              paths["q-table-dir"] / "Q_TABLE_RECEIPT.json",
              paths["q-table-dir"] / "q_table.npz",
              HERE / "independent_a_source.py", HERE / "independent_a_inputs.py",
              HERE / "independent_a_law.py", HERE / "independent_a_q.py",
              HERE / "independent_a_views.py", HERE / "independent_reader_custody.py",
              HERE / "independent_replay.py", HERE / "replay_a_prefit.py",
              HERE / "replay_a_attack.py"]
    bank_root = paths["portfolio-package"] / f"shared/.private/tstar_score_bank_s{seed}_v1"
    source += [bank_root / "BANK_ADMISSION_RECEIPT.json",
               bank_root / "bank_r1.npz", bank_root / "bank_r2.npz"]
    for short, role in (("att", "ATTACK_FIT_NEW"), ("inner", "INNER_SELECTION")):
        release = paths[f"{short}-release-dir"]
        source.extend(release / name for name in
                      ("RELEASE_RECEIPT.json", "release_codes.npz", "cache.sqlite3"))
        source.append(bank_root / f"tokens_{role}.npz")
        for kind in (f"references_s{seed}", "sex"):
            key = f"{role}:{kind}"
            source.append(role_package / manifest["role_file_relative_path"][key])
    source += [file for file in paths["attack-root"].rglob("*") if file.is_file()]
    source = [path.resolve(strict=True) for path in source]
    if len(set(source)) != len(source):
        raise ValueError("sampled A attack verification source list duplicates a file")
    output = HERE / f"A_ATTACK_{arm}_S{seed}_R{draw}_INDEPENDENT_REPLAY.json"
    spec_dir = HERE / ".private"
    spec_dir.mkdir(mode=0o700, exist_ok=True)
    spec_path = spec_dir / f"{args.unit_id}_SPEC.json"
    if output.exists() or spec_path.exists():
        raise FileExistsError("A attack verification spec or receipt already exists")
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_a_attack"]
    flag_paths = {**paths, "public-receipt": output}
    for name, path in flag_paths.items():
        argv.extend(("--" + name, str(path)))
    hashes = {"original-lock-sha256": sha_file(paths["original-lock"]),
              "a-amendment-sha256": sha_file(paths["a-amendment"]),
              "attack-index-sha256": sha_file(paths["attack-root"] / "ATTACK_ATTEMPT_INDEX.json"),
              "policy-receipt-sha256": sha_file(paths["fit-dir"] / "POLICY_RECEIPT.json"),
              "att-release-replay-sha256": sha_file(paths["att-release-replay"]),
              "inner-release-replay-sha256": sha_file(paths["inner-release-replay"]),
              "q-table-receipt-sha256": sha_file(paths["q-table-dir"] / "Q_TABLE_RECEIPT.json"),
              "role-manifest-sha256": sha_file(paths["role-manifest"]),
              "source-audit-sha256": sha_file(paths["source-audit"])}
    for name, value in hashes.items():
        argv.extend(("--" + name, value))
    spec = {"unit_id": args.unit_id, "stage": "verification",
            "deps": [expected_attack, args.att_release_replay_unit,
                     args.inner_release_replay_unit],
            "argv": argv, "cwd": str(WORKTREE), "max_wall_s": 1200,
            "cpu_reserve_s": 1200, "max_rss_gib": 3,
            "projected_output_bytes": 400000,
            "sources": [{"path": str(path), "sha256": sha_file(path)} for path in source],
            "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    spec_path.chmod(0o600)
    print(f"{spec_path} {sha_file(spec_path)}")


if __name__ == "__main__":
    main()
