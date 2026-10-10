"""Prepare the owner-only R03 spec for four independent D anchor replays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .independent_a_source import sha_file


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
ANCHORS = ("U0", "Ucal")
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "anchor-amendment", "role-manifest", "source-map"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--unit-id", default="R04-D-ANCHORS-001")
    args = parser.parse_args()
    if not args.unit_id.startswith("R04-D-ANCHORS-"):
        raise ValueError("anchor replay must use the R04 namespace")
    paths = {name: getattr(args, name.replace("-", "_")).resolve(strict=True)
             for name in ("seal", "anchor-amendment", "role-manifest", "source-map")}
    lock = json.loads(paths["seal"].read_text())
    amendment = json.loads(paths["anchor-amendment"].read_text())
    source_map = json.loads(paths["source-map"].read_text())
    if (lock.get("status") != "LOCALLY_SEALED_D_STAGE"
            or amendment.get("status") != "LOCALLY_SEALED_D_FOLLOWUP"
            or amendment.get("effective_seal_sha256") != sha_file(paths["seal"])
            or amendment.get("r01_source_map_sha256") != sha_file(paths["source-map"])
            or amendment.get("role_manifest_sha256") != sha_file(paths["role-manifest"])
            or Path(source_map["repair_manifest"]).resolve() != paths["role-manifest"]):
        raise ValueError("D anchor amendment/source does not match effective seal")
    campaign = paths["seal"].parent
    private_dir = HERE / ".private"
    private_dir.mkdir(mode=0o700, exist_ok=True)
    anchor_map_path = private_dir / f"{args.unit_id}_MAP.json"
    spec_path = private_dir / f"{args.unit_id}_SPEC.json"
    output = HERE / "D_ANCHORS_INDEPENDENT_REPLAY.json"
    if anchor_map_path.exists() or spec_path.exists() or output.exists():
        raise FileExistsError("anchor replay artifact already prepared")
    source = [HERE / name for name in (
        "make_d_anchor_spec.py", "replay_d_anchors.py", "independent_anchor_view.py",
        "independent_a_source.py", "independent_d_view.py",
        "independent_reader_custody.py", "independent_replay.py",
        "replay_first_pair.py")]
    source += list(paths.values())
    role = json.loads(paths["role-manifest"].read_text())
    for role_name in ROLES:
        for field in ("sex", "references_s0", "references_s1"):
            key = f"{role_name}:{field}"
            source.append(paths["role-manifest"].parent.parent /
                          role["role_file_relative_path"][key])
    mapping = {}
    deps = []
    for seed in (0, 1):
        for anchor in ANCHORS:
            unit = f"D_U_ANCHOR_{anchor.upper()}_S{seed}"
            receipt = (campaign / "r02" / f"{unit}_RECEIPT.json").resolve(strict=True)
            private = (campaign / ".private" /
                       f"d_u_anchor_{anchor.lower()}_s{seed}").resolve(strict=True)
            record = json.loads(receipt.read_text())
            if (record.get("status") != "COMPLETE_PRIVATE_DEVELOPMENT_ONLY"
                    or record.get("unit_id") != unit
                    or record.get("anchor") != anchor
                    or record.get("source_seed") != seed):
                raise ValueError("four anchor attacks have not all completed")
            mapping[f"{anchor}_s{seed}"] = {
                "receipt": str(receipt), "receipt_sha256": sha_file(receipt),
                "private_root": str(private)}
            deps.append(unit)
            source.append(receipt)
            source.extend(path for path in private.rglob("*") if path.is_file())
    anchor_map_path.write_text(json.dumps(mapping, sort_keys=True, indent=2) + "\n")
    anchor_map_path.chmod(0o600)
    source.append(anchor_map_path)
    source = [path.resolve(strict=True) for path in source]
    if len(source) != len(set(source)):
        raise ValueError("duplicate pinned anchor source path")
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_d_anchors"]
    for name, path in paths.items():
        argv += ["--" + name, str(path), "--" + name + "-sha256", sha_file(path)]
    argv += ["--anchor-map", str(anchor_map_path),
             "--anchor-map-sha256", sha_file(anchor_map_path),
             "--public-receipt", str(output)]
    spec = {"unit_id": args.unit_id, "stage": "verification", "argv": argv,
            "cwd": str(WORKTREE), "max_wall_s": 1800, "cpu_reserve_s": 1800,
            "max_rss_gib": 3, "projected_output_bytes": 1000000,
            "sources": [{"path": str(path), "sha256": sha_file(path)} for path in source],
            "deps": deps, "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    spec_path.chmod(0o600)
    print(spec_path)


if __name__ == "__main__":
    main()
