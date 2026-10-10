"""Prepare hash-bound private R03 input and spec for all 567 D intervals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .independent_a_source import sha_file
from .independent_d_intervals import ARMS, ANCHORS, VIEWS


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
LABELS = {"D_T": "T", "D_K": "D-K", "D_L": "L", "D_S12": "S12",
          "D_S21": "S21", "D_C": "C", "D_F": "F"}


def _item(path: Path, expected: str | None = None) -> dict:
    path = path.resolve(strict=True)
    actual = sha_file(path)
    if expected is not None and actual != expected:
        raise ValueError("D interval source bytes differ from public custody")
    return {"path": str(path), "sha256": actual}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "role-manifest", "family-manifest", "anchor-verification"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--anchor-verification-unit-id", default="R04-D-ANCHORS-001")
    parser.add_argument("--unit-id", default="R04-D-INTERVALS-001")
    args = parser.parse_args()
    if not args.unit_id.startswith("R04-D-INTERVALS-"):
        raise ValueError("D interval unit must use R04 namespace")
    paths = {name: getattr(args, name.replace("-", "_")).resolve(strict=True)
             for name in ("seal", "role-manifest", "family-manifest",
                          "anchor-verification")}
    campaign = paths["seal"].parent
    lock = json.loads(paths["seal"].read_text())
    if (lock.get("status") != "LOCALLY_SEALED_D_STAGE"
            or sha_file(paths["role-manifest"])
               != lock["source_files_sha256"]["source_role_manifest"]):
        raise ValueError("D interval effective source seal differs")
    anchor_verify = json.loads(paths["anchor-verification"].read_text())
    if anchor_verify.get("status") != "FOUR_SCORE_ANCHOR_REPLAY_PASS":
        raise ValueError("independent U anchor replay not complete")
    role = json.loads(paths["role-manifest"].read_text())
    source_map = json.loads((campaign / "r01/U_ANCHOR_SOURCE_MAP.json").read_text())
    source_paths = [HERE / name for name in (
        "make_d_intervals_spec.py", "replay_d_intervals.py",
        "independent_d_intervals.py", "independent_replay.py",
        "independent_a_source.py", "replay_first_pair.py")]
    source_paths += list(paths.values())
    source_paths.append(campaign / "r01/U_ANCHOR_SOURCE_MAP.json")
    sex_key = "INNER_SELECTION:sex"
    source_paths.append(paths["role-manifest"].parent.parent /
                        role["role_file_relative_path"][sex_key])
    private_dir = HERE / ".private"
    private_dir.mkdir(mode=0o700, exist_ok=True)
    map_path = private_dir / f"{args.unit_id}_INPUT_MAP.json"
    spec_path = private_dir / f"{args.unit_id}_SPEC.json"
    output = HERE / "D_567_INDEPENDENT_INTERVALS.json"
    if map_path.exists() or spec_path.exists() or output.exists():
        raise FileExistsError("D interval input/spec/output already exists")
    mapped = {"schema": "pcrl-r04-d-interval-input-v1", "seeds": {}}
    deps = [args.anchor_verification_unit_id]
    for seed in (0, 1):
        first_verify_path = HERE / ("FIRST_PAIR_INDEPENDENT_REPLAY.json" if seed == 0
                                    else "FIRST_PAIR_S1_INDEPENDENT_REPLAY.json")
        allarm_path = HERE / f"FOLLOWUP_S{seed}_ATTACK_INDEPENDENT_REPLAY.json"
        paths[f"first-s{seed}"] = first_verify_path.resolve(strict=True)
        paths[f"allarm-s{seed}"] = allarm_path.resolve(strict=True)
        first_verify = json.loads(first_verify_path.read_text())
        allarm = json.loads(allarm_path.read_text())
        if (first_verify.get("status") != "TECHNICAL_REPLAY_PASS"
                or allarm.get("status") != "FOLLOWUP_COMPLETE_VIEW_REPLAY_PASS"):
            raise ValueError("D seven-arm independent source/attack replay missing")
        deps.append(f"R04-D-FOLLOWUP-S{seed}-ATTACKS-001")
        first_source = campaign / ("D_FIRST_S0_R1_RECEIPT.json" if seed == 0
                                   else "D_FIRST_S1_RECEIPT.json")
        followup_source = campaign / f"D_FOLLOWUP_CONTROLS_S{seed}_RECEIPT.json"
        first_receipt = json.loads(first_source.read_text())
        followup_receipt = json.loads(followup_source.read_text())
        if (sha_file(first_source) != first_verify["source_receipt_sha256"]
                or sha_file(followup_source) != allarm["source_receipt_sha256"]):
            raise ValueError("D interval source receipts differ from independent replays")
        source_paths += [first_source, followup_source]
        first_private = campaign / ".private" / f"d_first_s{seed}"
        followup_private = campaign / ".private" / f"d_followup_s{seed}"
        endpoint = _item(first_private / "endpoint__INNER_SELECTION.npz",
                         first_verify["endpoint_sha256"])
        source_paths.append(Path(endpoint["path"]))
        releases = {}
        for name in (*ARMS, *ANCHORS):
            if name in ANCHORS:
                archive_key = f"INNER_SELECTION:references_s{seed}"
                archive = (paths["role-manifest"].parent.parent /
                           role["role_file_relative_path"][archive_key])
                archive_sha = source_map["rows"]["INNER_SELECTION"][f"references_s{seed}_sha256"]
                source = _item(archive, archive_sha)
                unit = f"D_U_ANCHOR_{name.upper()}_S{seed}"
                receipt_path = campaign / "r02" / f"{unit}_RECEIPT.json"
                private = campaign / ".private" / f"d_u_anchor_{name.lower()}_s{seed}"
                kind = "anchor"
                expected_attack = anchor_verify["anchors"][f"{name}_s{seed}"]["attack_receipt_sha256"]
            else:
                label = LABELS[name]
                if name in ("D_T", "D_K"):
                    source = _item(first_private / f"release__{label}__INNER_SELECTION.npz",
                                   first_receipt["roles"]["INNER_SELECTION"][label]
                                   ["private_release_sha256"])
                    attack_label = "DK" if name == "D_K" else "T"
                    expected_attack = first_verify["attack_receipt_sha256"][label]
                else:
                    source = _item(followup_private /
                                   f"release__{label}__INNER_SELECTION.npz",
                                   followup_receipt["roles"]["INNER_SELECTION"][label]
                                   ["private_release_sha256"])
                    attack_label = label
                    expected_attack = allarm["arms"][label]["attack_receipt_sha256"]
                unit = f"D_ATTACK_{attack_label}_S{seed}"
                receipt_path = campaign / "latent_repair_v1" / "attacks" / f"{unit}_RECEIPT.json"
                private = campaign / ".private" / f"d_attack_{attack_label.lower()}_s{seed}"
                kind = "continuous" if name == "D_F" else "hard"
            attack = _item(receipt_path, expected_attack)
            private = private.resolve(strict=True)
            attack_record = json.loads(Path(attack["path"]).read_text())
            if (attack_record.get("status") != "COMPLETE_PRIVATE_DEVELOPMENT_ONLY"
                    or attack_record.get("source_seed") != seed
                    or attack_record.get("unit_id") != unit):
                raise ValueError("D interval attack receipt identity differs")
            source_paths += [Path(source["path"]), Path(attack["path"])]
            for view in VIEWS:
                directory = private / view / "readers"
                receipt = _item(directory / "READERS_RECEIPT.json",
                                attack_record["attack"]["reader_custody"][view]
                                ["private_receipt_sha256"])
                reader_doc = json.loads(Path(receipt["path"]).read_text())
                source_paths.append(Path(receipt["path"]))
                for leaf in ("predictions_inner.npz", "view_inner.npz"):
                    member = _item(directory / leaf, reader_doc["files"][leaf])
                    source_paths.append(Path(member["path"]))
            releases[name] = {"kind": kind, **source,
                              "attack_receipt": attack["path"],
                              "attack_receipt_sha256": attack["sha256"],
                              "private_root": str(private)}
        mapped["seeds"][str(seed)] = {"endpoint": endpoint, "releases": releases}
    map_path.write_text(json.dumps(mapped, sort_keys=True, indent=2) + "\n")
    map_path.chmod(0o600)
    source_paths.append(map_path)
    unique = {str(path.resolve(strict=True)): path.resolve(strict=True) for path in source_paths}
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_d_intervals"]
    for name, path in paths.items():
        argv += ["--" + name, str(path), "--" + name + "-sha256", sha_file(path)]
    argv += ["--input-map", str(map_path), "--input-map-sha256", sha_file(map_path),
             "--public-receipt", str(output)]
    spec = {"unit_id": args.unit_id, "stage": "verification", "argv": argv,
            "cwd": str(WORKTREE), "max_wall_s": 1800, "cpu_reserve_s": 1800,
            "max_rss_gib": 3, "projected_output_bytes": 2000000,
            "sources": [_item(path) for path in unique.values()],
            "deps": deps, "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    spec_path.chmod(0o600)
    print(spec_path)


if __name__ == "__main__":
    main()
