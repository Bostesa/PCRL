"""Prepare owner-only R03 specification for five-arm D release replay."""

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
    for name in ("seal", "amendment", "role-manifest", "source-checkpoint",
                 "first-private", "first-receipt", "followup-private",
                 "followup-receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--source-unit-id", required=True)
    parser.add_argument("--unit-id", default=None)
    parser.add_argument("--max-rss-gib", type=float, default=3.0)
    args = parser.parse_args()
    paths = {name: getattr(args, name.replace("-", "_")).resolve(strict=True)
             for name in ("seal", "amendment", "role-manifest", "source-checkpoint",
                          "first-private", "first-receipt", "followup-private",
                          "followup-receipt")}
    seal = json.loads(paths["seal"].read_text())
    amendment = json.loads(paths["amendment"].read_text())
    if (seal.get("status") != "LOCALLY_SEALED_D_STAGE"
            or amendment.get("status") != "LOCALLY_SEALED_D_FOLLOWUP"
            or amendment.get("effective_seal_sha256") != sha_file(paths["seal"])):
        raise ValueError("effective followup seal/amendment differs")
    followup = json.loads(paths["followup-receipt"].read_text())
    first = json.loads(paths["first-receipt"].read_text())
    if (followup.get("status") != "D_FOLLOWUP_RELEASES_READY"
            or followup.get("seed") != args.seed
            or followup.get("unit_id") != args.source_unit_id
            or followup.get("first_receipt_sha256") != sha_file(paths["first-receipt"])
            or first.get("seed") != args.seed
            or set(followup.get("packages", {})) != set(ARMS)):
        raise ValueError("completed paired followup source not frozen")
    output = HERE / f"FOLLOWUP_S{args.seed}_INDEPENDENT_REPLAY.json"
    private_dir = HERE / ".private"
    private_dir.mkdir(mode=0o700, exist_ok=True)
    unit_id = args.unit_id or f"R04-D-FOLLOWUP-S{args.seed}-SOURCE-001"
    if (not unit_id.startswith(f"R04-D-FOLLOWUP-S{args.seed}-SOURCE-")
            or not unit_id.rsplit("-", 1)[-1].isdigit()
            or not 0 < args.max_rss_gib <= 4):
        raise ValueError("unit ID or memory cap outside R04 followup contract")
    spec_path = private_dir / f"R04_FOLLOWUP_S{args.seed}_QUEUE_SPEC_{unit_id.rsplit('-', 1)[-1]}.json"
    if output.exists() or spec_path.exists():
        raise FileExistsError("followup source replay already prepared")
    manifest = json.loads(paths["role-manifest"].read_text())
    source = [HERE / name for name in (
        "make_d_followup_source_spec.py", "replay_d_followup_source.py", "independent_a_source.py",
        "independent_d_restore.py", "independent_d_view.py",
        "independent_f_restore.py", "independent_reader_custody.py",
        "independent_release.py", "independent_replay.py",
        "replay_first_pair.py")]
    source += [paths[name] for name in ("seal", "amendment", "role-manifest",
                                        "source-checkpoint", "first-receipt",
                                        "followup-receipt")]
    source.append(paths["first-private"] / "endpoint__INNER_SELECTION.npz")
    for arm in ARMS:
        source += [paths["followup-private"] / arm / "manifest.json",
                   paths["followup-private"] / arm / "weights.npz"]
        for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
            source.append(paths["followup-private"] / f"release__{arm}__{role}.npz")
    for role, kinds in (("OSF_DEFENSE_FIT", ("features", "income", "occupation")),
                        ("ATTACK_FIT_NEW", ("features",)),
                        ("INNER_SELECTION", ("features",))):
        for kind in kinds:
            key = f"{role}:{kind}"
            source.append(paths["role-manifest"].parent.parent /
                          manifest["role_file_relative_path"][key])
    source = [p.resolve(strict=True) for p in source]
    if len(source) != len(set(source)):
        raise ValueError("duplicate verification source")
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_d_followup_source",
            "--seed", str(args.seed)]
    for name, path in paths.items():
        argv += ["--" + name, str(path)]
        if name in ("seal", "amendment", "role-manifest", "first-receipt",
                    "followup-receipt"):
            argv += ["--" + name + "-sha256", sha_file(path)]
    argv += ["--public-receipt", str(output)]
    spec = {"unit_id": unit_id,
            "stage": "verification", "argv": argv, "cwd": str(WORKTREE),
            "max_wall_s": 1800, "cpu_reserve_s": 1800,
            "max_rss_gib": args.max_rss_gib, "projected_output_bytes": 400000,
            "sources": [{"path": str(p), "sha256": sha_file(p)} for p in source],
            "deps": [args.source_unit_id], "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    print(spec_path)


if __name__ == "__main__":
    main()
