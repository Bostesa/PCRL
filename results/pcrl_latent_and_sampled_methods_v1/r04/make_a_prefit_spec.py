"""Prepare hash-bound private R03 spec for independent A source admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .independent_a_source import sha_file


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "role-manifest", "source-audit", "portfolio-package", "source-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--unit-id", default="R04-A-PREFIT-SOURCE-001")
    args = parser.parse_args()
    seal = args.seal.resolve(strict=True)
    lock = json.loads(seal.read_text())
    if lock.get("status") != "LOCALLY_SEALED_D_STAGE":
        raise ValueError("effective local seal required for verification unit")
    manifest_path = args.role_manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    audit_path = args.source_audit.resolve(strict=True)
    portfolio = args.portfolio_package.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output = HERE / "A_PREFIT_INDEPENDENT_REPLAY.json"
    private_dir = HERE / ".private"
    private_dir.mkdir(mode=0o700, exist_ok=True)
    spec_path = private_dir / "R04_A_PREFIT_QUEUE_SPEC.json"
    if output.exists() or spec_path.exists():
        raise FileExistsError("A prefit replay already prepared or complete")
    source = [HERE / "independent_a_source.py", HERE / "replay_a_prefit.py",
              HERE / "independent_replay.py", seal, manifest_path, audit_path,
              portfolio / "DATA_ACCESS_MANIFEST.json"]
    for seed in (0, 1):
        root = portfolio / f"shared/.private/tstar_score_bank_s{seed}_v1"
        source += [root / "BANK_ADMISSION_RECEIPT.json"]
        source += [root / f"bank_r{recipient}.npz" for recipient in (1, 2)]
        source += [root / f"tokens_{role}.npz"
                   for role in ("OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION")]
        source += [source_root / f"qpc_v1/run/units/fine__s{seed}/fine.json",
                   source_root / f"qpc_v1/run/units/pol__s{seed}__U_FINE-TASK_i8o64/policy.json",
                   source_root / f"hcal_v1/package/decoder__s{seed}__U_FINE-TASK_i8o64_H-GLOBAL-TEMP.json",
                   source_root / f"hcal_v1/admitted/rel__s{seed}__U/model.pt"]
        for role in ("OSF_DEFENSE_FIT", "ATTACK_FIT_NEW", "INNER_SELECTION"):
            key = f"{role}:references_s{seed}"
            source.append(manifest_path.parent.parent / manifest["role_file_relative_path"][key])
    for role in ("OSF_DEFENSE_FIT", "INNER_SELECTION"):
        for task in ("income", "occupation"):
            key = f"{role}:{task}"
            source.append(manifest_path.parent.parent / manifest["role_file_relative_path"][key])
    source = [path.resolve(strict=True) for path in source]
    if len(source) != len(set(source)):
        raise ValueError("duplicate queued source")
    argv = ["/Users/nathansamson/PCRL/.venv/bin/python", "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_a_prefit"]
    flags = {"seal": seal, "role-manifest": manifest_path, "source-audit": audit_path,
             "portfolio-package": portfolio, "source-root": source_root,
             "public-receipt": output}
    for name, path in flags.items():
        argv.extend(["--" + name, str(path)])
        if name in ("seal", "role-manifest", "source-audit"):
            argv.extend(["--" + name + "-sha256", sha_file(path)])
    spec = {"unit_id": args.unit_id, "stage": "verification", "argv": argv,
            "cwd": str(WORKTREE), "max_wall_s": 900, "cpu_reserve_s": 900,
            "max_rss_gib": 3, "projected_output_bytes": 200000,
            "sources": [{"path": str(path), "sha256": sha_file(path)} for path in source],
            "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    print(spec_path)


if __name__ == "__main__":
    main()
