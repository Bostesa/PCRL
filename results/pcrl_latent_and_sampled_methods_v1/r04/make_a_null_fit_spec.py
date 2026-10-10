"""Prepare private R03 verification of every saved null-selected A FIT checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .independent_a_source import sha_file


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
UNIT = "R04-A-FIT-NULL-JOINT-S0-001"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-lock", "a-amendment", "fit-dir", "portfolio-package",
                 "role-manifest", "source-audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name.replace("-", "_")).resolve(strict=True)
             for name in ("original-lock", "a-amendment", "fit-dir",
                          "portfolio-package", "role-manifest", "source-audit")}
    fit_dir = paths["fit-dir"]
    fit_path = fit_dir / "FIT_RECEIPT.json"
    fit = json.loads(fit_path.read_text())
    if (fit.get("status") != "COMPLETE_NO_FEASIBLE_POLICY" or
            fit.get("selected") is not None or
            fit.get("unit_id") != "A-FIT-A_JOINT_STOCH-S0-R1" or
            fit.get("a_amendment_sha256") != sha_file(paths["a-amendment"])):
        raise ValueError("A null-selected retry FIT receipt differs")
    portfolio = paths["portfolio-package"]
    bank_root = portfolio / "shared/.private/tstar_score_bank_s0_v1"
    role_manifest = json.loads(paths["role-manifest"].read_text())
    role_root = paths["role-manifest"].parent.parent
    source = [paths[name] for name in paths
              if name not in ("fit-dir", "portfolio-package")]
    source += [fit_path, portfolio / "DATA_ACCESS_MANIFEST.json",
               bank_root / "BANK_ADMISSION_RECEIPT.json",
               bank_root / "bank_r1.npz", bank_root / "bank_r2.npz",
               bank_root / "tokens_OSF_DEFENSE_FIT.npz",
               HERE / "audit_diagnostic_q_parity.py",
               HERE / "independent_a_source.py", HERE / "independent_a_inputs.py",
               HERE / "independent_a_law.py", HERE / "independent_a_fit.py",
               HERE / "replay_a_prefit.py", HERE / "replay_a_null_fit.py"]
    for field in ("references_s0", "income", "occupation", "sex"):
        key = f"OSF_DEFENSE_FIT:{field}"
        source.append(role_root / role_manifest["role_file_relative_path"][key])
    for start in ("identity_biased", "uniform_logit"):
        for checkpoint in fit["starts"][start]["checkpoints"]:
            path = fit_dir / checkpoint["filename"]
            if sha_file(path) != checkpoint["file_sha256"]:
                raise ValueError("A null FIT checkpoint changed before source pin")
            source.append(path)
    source = [path.resolve(strict=True) for path in source]
    if len(source) != len(set(source)):
        raise ValueError("A null FIT source list repeats a path")
    output = HERE / "A_FIT_A_JOINT_STOCH_S0_NULL_CHECKPOINT_REPLAY.json"
    spec_dir = HERE / ".private"
    spec_dir.mkdir(mode=0o700, exist_ok=True)
    spec_path = spec_dir / f"{UNIT}_SPEC.json"
    if output.exists() or spec_path.exists():
        raise FileExistsError("A null FIT replay spec or receipt exists")
    argv = [sys.executable, "-m",
            "results.pcrl_latent_and_sampled_methods_v1.r04.replay_a_null_fit"]
    for name, path in {**paths, "public-receipt": output}.items():
        argv.extend(("--" + name, str(path)))
    for name, path in (("original-lock", paths["original-lock"]),
                       ("a-amendment", paths["a-amendment"]),
                       ("fit-receipt", fit_path),
                       ("role-manifest", paths["role-manifest"]),
                       ("source-audit", paths["source-audit"])):
        argv.extend(("--" + name + "-sha256", sha_file(path)))
    spec = {"unit_id": UNIT, "stage": "verification", "mode": "real_data",
            "deps": [fit["unit_id"]], "argv": argv, "cwd": str(WORKTREE),
            "max_wall_s": 900, "cpu_reserve_s": 900, "max_rss_gib": 3,
            "projected_output_bytes": 150000,
            "sources": [{"path": str(path), "sha256": sha_file(path)} for path in source],
            "outputs": [str(output)],
            "env": {name: "1" for name in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}}
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    spec_path.chmod(0o600)
    print(f"{spec_path} {sha_file(spec_path)}")


if __name__ == "__main__":
    main()
