"""Prepare owner-only R03 spec for independent five-arm complete-view replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .independent_a_source import sha_file


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
ARMS = ("L", "S12", "S21", "C", "F")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "attack-amendment", "role-manifest", "source-private",
                 "source-receipt", "source-verification", "first-verification"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1))
    parser.add_argument("--source-verification-unit-id", required=True)
    parser.add_argument("--unit-id", default=None)
    args = parser.parse_args()
    paths = {name: getattr(args, name.replace("-", "_")).resolve(strict=True)
             for name in ("seal", "attack-amendment", "role-manifest",
                          "source-private", "source-receipt",
                          "source-verification", "first-verification")}
    seal = json.loads(paths["seal"].read_text())
    amendment = json.loads(paths["attack-amendment"].read_text())
    receipt = json.loads(paths["source-receipt"].read_text())
    if (seal.get("status") != "LOCALLY_SEALED_D_STAGE"
            or amendment.get("status") != "LOCALLY_SEALED_D_FOLLOWUP"
            or amendment.get("effective_seal_sha256") != sha_file(paths["seal"])
            or amendment.get("control_receipt_sha256_by_seed", {}).get(str(args.seed))
               != sha_file(paths["source-receipt"])
            or receipt.get("status") != "D_FOLLOWUP_RELEASES_READY"
            or receipt.get("seed") != args.seed):
        raise ValueError("effective followup attack source/amendment differs")
    if json.loads(paths["source-verification"].read_text()).get("status") != "FOLLOWUP_RELEASE_REPLAY_PASS":
        raise ValueError("independent five-arm release replay not complete")
    if json.loads(paths["first-verification"].read_text()).get("status") != "TECHNICAL_REPLAY_PASS":
        raise ValueError("independent paired first-arm replay not complete")
    private_dir = HERE / ".private"
    private_dir.mkdir(mode=0o700, exist_ok=True)
    unit_id = args.unit_id or f"R04-D-FOLLOWUP-S{args.seed}-ATTACKS-001"
    if not unit_id.startswith(f"R04-D-FOLLOWUP-S{args.seed}-ATTACKS-"):
        raise ValueError("followup attack verifier ID outside R04 namespace")
    serial = unit_id.rsplit("-", 1)[-1]
    if not serial.isdigit():
        raise ValueError("followup verifier ID requires numeric serial")
    attack_map_path = private_dir / f"R04_FOLLOWUP_S{args.seed}_ATTACK_MAP_{serial}.json"
    spec_path = private_dir / f"R04_FOLLOWUP_S{args.seed}_ATTACK_SPEC_{serial}.json"
    output = HERE / f"FOLLOWUP_S{args.seed}_ATTACK_INDEPENDENT_REPLAY.json"
    if attack_map_path.exists() or spec_path.exists() or output.exists():
        raise FileExistsError("five-arm attack replay already prepared")
    campaign = paths["seal"].parent
    source = [HERE / name for name in (
        "make_d_followup_attacks_spec.py", "replay_d_followup_attacks.py",
        "independent_a_source.py", "independent_d_view.py", "independent_f_view.py",
        "independent_reader_custody.py", "independent_replay.py",
        "replay_first_pair.py", "independent_d_restore.py",
        "independent_release.py")]
    source += [paths[name] for name in ("seal", "attack-amendment", "role-manifest",
                                        "source-receipt", "source-verification",
                                        "first-verification")]
    role_manifest = json.loads(paths["role-manifest"].read_text())
    for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        key = f"{role}:sex"
        source.append(paths["role-manifest"].parent.parent /
                      role_manifest["role_file_relative_path"][key])
    attack_map = {}
    deps = [args.source_verification_unit_id]
    for arm in ARMS:
        attack_unit = f"D_ATTACK_{arm}_S{args.seed}"
        receipt_path = campaign / "latent_repair_v1" / "attacks" / f"{attack_unit}_RECEIPT.json"
        private = campaign / ".private" / f"d_attack_{arm.lower()}_s{args.seed}"
        receipt_path = receipt_path.resolve(strict=True)
        private = private.resolve(strict=True)
        attack_receipt = json.loads(receipt_path.read_text())
        if (attack_receipt.get("unit_id") != attack_unit
                or attack_receipt.get("arm_id") != arm
                or attack_receipt.get("source_seed") != args.seed
                or attack_receipt.get("source_receipt_sha256") != sha_file(paths["source-receipt"])):
            raise ValueError("followup attack receipt/source not complete")
        attack_map[arm] = {"receipt": str(receipt_path),
                           "receipt_sha256": sha_file(receipt_path),
                           "private_root": str(private)}
        deps.append(attack_unit)
        source.append(receipt_path)
        source.extend(path for path in private.rglob("*") if path.is_file())
        package = paths["source-private"] / arm / "manifest.json"
        source.append(package)
        for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
            source.append(paths["source-private"] / f"release__{arm}__{role}.npz")
    attack_map_path.write_text(json.dumps(attack_map, sort_keys=True, indent=2) + "\n")
    attack_map_path.chmod(0o600)
    source.append(attack_map_path)
    source = [path.resolve(strict=True) for path in source]
    if len(source) != len(set(source)):
        raise ValueError("duplicate pinned reader/source path")
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_d_followup_attacks",
            "--seed", str(args.seed)]
    for name, path in paths.items():
        argv += ["--" + name, str(path)]
        if name in ("seal", "attack-amendment", "role-manifest", "source-receipt",
                    "source-verification", "first-verification"):
            argv += ["--" + name + "-sha256", sha_file(path)]
    argv += ["--attack-map", str(attack_map_path),
             "--attack-map-sha256", sha_file(attack_map_path),
             "--public-receipt", str(output)]
    spec = {"unit_id": unit_id, "stage": "verification", "argv": argv,
            "cwd": str(WORKTREE), "max_wall_s": 1800, "cpu_reserve_s": 1800,
            "max_rss_gib": 3, "projected_output_bytes": 1000000,
            "sources": [{"path": str(path), "sha256": sha_file(path)} for path in source],
            "deps": deps, "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    spec_path.chmod(0o600)
    print(spec_path)


if __name__ == "__main__":
    main()
