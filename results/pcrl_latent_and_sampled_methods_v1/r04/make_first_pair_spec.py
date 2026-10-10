"""Assemble R03's immutable first real D-pair independent-verification unit.

Run only after the source T/K and both complete-view attack receipts exist.
This metadata-only command neither opens numerical arrays nor starts a fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
VERIFY_CODE = (
    "replay_first_pair.py", "independent_d_restore.py", "independent_d_view.py",
    "independent_release.py", "independent_reader_custody.py", "independent_replay.py",
)


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "role-manifest", "source-checkpoint", "source-private",
                 "source-receipt", "attack-t-private", "attack-t-receipt",
                 "attack-k-private", "attack-k-receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--deps", required=True, help="comma-separated R03 source and attack unit IDs")
    parser.add_argument("--unit-id", default="R04-D-FIRST-PAIR-001")
    args = parser.parse_args()
    seal = args.seal.resolve(strict=True)
    if json.loads(seal.read_text()).get("status") != "LOCALLY_SEALED_D_STAGE":
        raise ValueError("effective D stage seal required")
    for private in (args.source_private, args.attack_t_private, args.attack_k_private):
        if ".private" not in private.resolve(strict=True).parts:
            raise ValueError("per-person source must be in private custody")
    public_receipt = HERE / "FIRST_PAIR_INDEPENDENT_REPLAY.json"
    private_specs = HERE / ".private"
    private_specs.mkdir(mode=0o700, exist_ok=True)
    spec_path = private_specs / "R04_FIRST_PAIR_QUEUE_SPEC.json"
    if public_receipt.exists() or spec_path.exists():
        raise FileExistsError("first pair R04 verification already prepared or run")
    argv = [str(Path("/Users/nathansamson/PCRL/.venv/bin/python")), "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_first_pair"]
    paths = {
        "seal": seal, "role-manifest": args.role_manifest.resolve(strict=True),
        "source-checkpoint": args.source_checkpoint.resolve(strict=True),
        "source-private": args.source_private.resolve(strict=True),
        "source-receipt": args.source_receipt.resolve(strict=True),
        "attack-t-private": args.attack_t_private.resolve(strict=True),
        "attack-t-receipt": args.attack_t_receipt.resolve(strict=True),
        "attack-k-private": args.attack_k_private.resolve(strict=True),
        "attack-k-receipt": args.attack_k_receipt.resolve(strict=True),
        "public-receipt": public_receipt,
    }
    for flag, path in paths.items():
        argv += ["--" + flag, str(path)]
        if flag in ("seal", "source-receipt", "attack-t-receipt", "attack-k-receipt"):
            argv += ["--" + flag + "-sha256", _hash(path)]
    source = [HERE / name for name in VERIFY_CODE]
    source += [paths[name] for name in ("seal", "role-manifest", "source-checkpoint",
                                        "source-receipt", "attack-t-receipt", "attack-k-receipt")]
    first_root = paths["source-private"]
    source += [first_root / "endpoint__INNER_SELECTION.npz"]
    for arm in ("T", "D-K"):
        source += [first_root / arm / "manifest.json", first_root / arm / "weights.npz"]
        for role in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
            source.append(first_root / f"release__{arm}__{role}.npz")
    for root in (paths["attack-t-private"], paths["attack-k-private"]):
        source.append(root / "FROZEN_COMPLETE.json")
        for view in ("r1", "r2", "pair"):
            readers = root / view / "readers"
            source += [readers / name for name in ("READERS_RECEIPT.json", "view_att.npz",
                                                   "view_inner.npz", "predictions_inner.npz")]
            controls = root / view / "controls"
            source += [controls / "CUSTODY_RECEIPT.json", controls / "probe_replay.npz"]
    if len(set(source)) != len(source):
        raise ValueError("duplicate immutable input path")
    sources = [{"path": str(p), "sha256": _hash(p)} for p in source]
    deps = [value for value in args.deps.split(",") if value]
    if len(deps) != 3 or len(set(deps)) != 3:
        raise ValueError("one source and two attack queue dependency IDs required")
    spec = {"unit_id": args.unit_id, "stage": "verification", "argv": argv,
            "cwd": str(WORKTREE), "max_wall_s": 1800, "cpu_reserve_s": 1800,
            "max_rss_gib": 3, "projected_output_bytes": 250000,
            "sources": sources, "deps": deps, "outputs": [str(public_receipt)]}
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    print(spec_path)


if __name__ == "__main__":
    main()
