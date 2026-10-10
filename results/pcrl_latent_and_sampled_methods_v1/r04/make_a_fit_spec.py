"""Prepare one owner-only R03 spec for independent selected A FIT replay."""

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
    for name in ("original-lock", "a-amendment", "fit-dir", "portfolio-package",
                 "role-manifest", "source-audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--unit-id", required=True)
    args = parser.parse_args()
    lock, amendment, fit_dir, portfolio, manifest_path, audit = (
        getattr(args, name.replace("-", "_")).resolve(strict=True)
        for name in ("original-lock", "a-amendment", "fit-dir", "portfolio-package",
                     "role-manifest", "source-audit"))
    fit_path = fit_dir / "FIT_RECEIPT.json"
    policy_path = fit_dir / "POLICY_RECEIPT.json"
    policy_npz = fit_dir / "policy.npz"
    fit = json.loads(fit_path.read_text())
    arm, seed = fit["arm_id"], fit["source_seed"]
    if (fit["status"] != "SELECTED_FIT_POLICY"
            or fit["a_amendment_sha256"] != sha_file(amendment)
            or not args.unit_id.startswith("R04-A-FIT-")):
        raise ValueError("selected A FIT not ready for independent replay")
    plan = json.loads(amendment.read_text())["unit_plan"][fit["unit_id"]]
    manifest = json.loads(manifest_path.read_text())
    source = [lock, amendment, fit_path, policy_path, policy_npz, manifest_path, audit,
              portfolio / "DATA_ACCESS_MANIFEST.json",
              HERE / "independent_a_source.py", HERE / "independent_a_inputs.py",
              HERE / "independent_a_law.py", HERE / "independent_a_fit.py",
              HERE / "replay_a_prefit.py", HERE / "replay_a_fit.py"]
    bank_root = portfolio / f"shared/.private/tstar_score_bank_s{seed}_v1"
    source += [bank_root / "BANK_ADMISSION_RECEIPT.json",
               bank_root / "bank_r1.npz", bank_root / "bank_r2.npz",
               bank_root / "tokens_OSF_DEFENSE_FIT.npz"]
    for kind in (f"references_s{seed}", "income", "occupation"):
        key = f"OSF_DEFENSE_FIT:{kind}"
        source.append(manifest_path.parent.parent / manifest["role_file_relative_path"][key])
    if plan["objective_kind"] == "protected_bayes":
        key = "OSF_DEFENSE_FIT:sex"
        source.append(manifest_path.parent.parent / manifest["role_file_relative_path"][key])
    source = [path.resolve(strict=True) for path in source]
    if len(set(source)) != len(source):
        raise ValueError("A FIT verification source list duplicated a path")
    output = HERE / f"A_FIT_{arm}_S{seed}_INDEPENDENT_REPLAY.json"
    spec_dir = HERE / ".private"
    spec_dir.mkdir(mode=0o700, exist_ok=True)
    spec_path = spec_dir / f"{args.unit_id}_SPEC.json"
    if output.exists() or spec_path.exists():
        raise FileExistsError("A FIT verification spec or receipt already exists")
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_a_fit"]
    flags = {"original-lock": lock, "a-amendment": amendment, "fit-dir": fit_dir,
             "portfolio-package": portfolio, "role-manifest": manifest_path,
             "source-audit": audit, "public-receipt": output}
    for name, path in flags.items():
        argv.extend(["--" + name, str(path)])
    hashes = {"original-lock-sha256": sha_file(lock),
              "a-amendment-sha256": sha_file(amendment),
              "fit-receipt-sha256": sha_file(fit_path),
              "policy-receipt-sha256": sha_file(policy_path),
              "role-manifest-sha256": sha_file(manifest_path),
              "source-audit-sha256": sha_file(audit)}
    for name, value in hashes.items():
        argv.extend(["--" + name, value])
    spec = {"unit_id": args.unit_id, "stage": "verification",
            "deps": [fit["unit_id"]], "argv": argv, "cwd": str(WORKTREE),
            "max_wall_s": 900, "cpu_reserve_s": 900, "max_rss_gib": 3,
            "projected_output_bytes": 150000,
            "sources": [{"path": str(path), "sha256": sha_file(path)} for path in source],
            "outputs": [str(output)]}
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    print(f"{spec_path} {sha_file(spec_path)}")


if __name__ == "__main__":
    main()
