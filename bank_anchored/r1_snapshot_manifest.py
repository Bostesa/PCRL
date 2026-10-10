"""Build an owner-only, hash-bound restore inventory for R1 deployment.

The inventory reads JSON metadata and streams file hashes only. It never
deserializes role arrays, labels, fitted policies, or sampled releases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat


SOURCE_ROLES = ("complete", "model", "head0", "head1", "record",
                "policy", "decoder", "public_table")
OPTIONAL_ARMS = frozenset({"B0", "T", "D", "P", "L", "J", "S12", "S21"})


def _sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _entry(path: Path, *, role: str, archive_name: str, private: bool) -> dict:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"dependency is not a regular file: {role}")
    return {"role": role, "source_path": str(path),
            "archive_name": archive_name, "sha256": _sha(path),
            "bytes": path.stat().st_size, "owner_only": private}


def build(root: Path, output: Path, *, selected_arms: tuple[str, ...] = (),
          fit_unit_ids: dict[tuple[str, int], str] | None = None) -> dict:
    root = root.resolve(strict=True)
    package = root / "results/pcrl_bank_anchored_release_v1"
    private = package / "r1/.private"
    if (output.parent.resolve(strict=True) != private or output.exists() or
            stat.S_IMODE(private.stat().st_mode) & 0o077 or
            len(set(selected_arms)) != len(selected_arms) or
            not set(selected_arms) <= OPTIONAL_ARMS):
        raise ValueError("private output or selected arms differ")
    fit_unit_ids = {} if fit_unit_ids is None else fit_unit_ids
    if set(fit_unit_ids) - {(arm, seed) for arm in selected_arms for seed in (0, 1)}:
        raise ValueError("unexpected selected FIT unit override")
    entries: list[dict] = []
    for seed in (0, 1):
        cfg_path = private / f"R1_BANK_S{seed}_FIT_SAMPLE32_CONFIG.json"
        cfg = json.loads(cfg_path.read_text())
        receipt_path = package / "r1" / f"R1_BANK_S{seed}_FIT_RAW_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text())
        sample_dir = private / f"R1_BANK_S{seed}_FIT_SAMPLE32"
        if (cfg.get("seed") != seed or receipt.get("seed") != seed or
                receipt.get("status") != "PASS" or
                receipt.get("sample_input_sha256") !=
                _sha(sample_dir / "sample_input.npz")):
            raise ValueError(f"seed {seed} raw witness is not admitted")
        source_root = Path(cfg["source_root"]).resolve(strict=True)
        qpc_root = Path(cfg["qpc_root"]).resolve(strict=True)
        for alias in SOURCE_ROLES:
            path = Path(cfg["pins"][alias]).resolve(strict=True)
            if alias == "policy":
                archive_name = f"seed{seed}/qpc_root/{path.relative_to(qpc_root)}"
            elif alias == "public_table":
                archive_name = f"seed{seed}/public_table/{path.name}"
            else:
                archive_name = f"seed{seed}/source_root/{path.relative_to(source_root)}"
            entries.append(_entry(path, role=f"seed{seed}.{alias}",
                                  archive_name=archive_name,
                                  private=alias != "public_table"))
        for alias, path, name in (
                ("raw_witness_config", cfg_path, "witness_config.json"),
                ("raw_witness_receipt", receipt_path, "witness_receipt.json"),
                ("raw_sample", sample_dir / "sample_input.npz", "input.npz"),
                ("raw_schema", sample_dir / "schema.json", "schema.json")):
            entries.append(_entry(path, role=f"seed{seed}.{alias}",
                                  archive_name=f"seed{seed}/sample/{name}",
                                  private=alias != "raw_witness_receipt"))
        for arm in selected_arms:
            fit_id = fit_unit_ids.get((arm, seed), f"R2_{arm}_S{seed}_FIT")
            if re.fullmatch(rf"R2_{arm}_S{seed}_FIT(?:_[A-Z][1-9][0-9]*)?",
                            fit_id) is None:
                raise ValueError("selected FIT successor identity differs")
            unit = package / "r2/.private/units" / fit_id
            policy = unit / "policy"
            policy_receipt = policy / "POLICY_RECEIPT.json"
            policy_bytes = policy / "policy.npz"
            public_unit_receipt = package / "r2" / f"{fit_id}_UNIT_RECEIPT.json"
            public_fit_receipt = package / "r2" / f"{fit_id}_FIT_RECEIPT.json"
            p = json.loads(policy_receipt.read_text())
            u = json.loads(public_unit_receipt.read_text())
            f = json.loads(public_fit_receipt.read_text())
            if (p.get("arm_id") != arm or p.get("source_seed") != seed or
                    p.get("policy_file_sha256") != _sha(policy_bytes) or
                    u.get("unit_id") != fit_id or
                    u.get("policy_receipt_sha256") != _sha(policy_receipt) or
                    u.get("fit_receipt_sha256") != _sha(public_fit_receipt) or
                    p.get("fit_receipt_sha256") != _sha(public_fit_receipt) or
                    not str(u.get("status", "")).startswith("COMPLETE_") or
                    f.get("status") != u.get("status") or
                    f.get("unit_id") != fit_id):
                raise ValueError(f"selected policy {fit_id} is not complete")
            for alias, path in (("policy_receipt", policy_receipt),
                                ("policy_npz", policy_bytes),
                                ("unit_receipt", public_unit_receipt),
                                ("fit_receipt", public_fit_receipt)):
                entries.append(_entry(path, role=f"seed{seed}.{arm}.{alias}",
                                      archive_name=f"seed{seed}/selected/{arm}/policy/{path.name}"
                                      if alias not in ("unit_receipt", "fit_receipt") else
                                      f"seed{seed}/selected/{arm}/{path.name}",
                                      private=alias not in ("unit_receipt", "fit_receipt")))
            # These two files bind the same 32 people and realization-0 codes
            # for independent replay; neither contains task or SEX labels.
            cached = unit / "OSF_DEFENSE_FIT_R0"
            cached_receipt = cached / "RELEASE_RECEIPT.json"
            cached_codes = cached / "release_codes.npz"
            c = json.loads(cached_receipt.read_text())
            if (c.get("policy_receipt_sha256") != _sha(policy_receipt) or
                    c.get("release_codes_sha256") != _sha(cached_codes)):
                raise ValueError(f"selected FIT release {fit_id} differs")
            for alias, path in (("cached_receipt", cached_receipt),
                                ("cached_codes", cached_codes)):
                entries.append(_entry(path, role=f"seed{seed}.{arm}.{alias}",
                                      archive_name=f"seed{seed}/selected/{arm}/OSF_DEFENSE_FIT_R0/{path.name}",
                                      private=True))
            replay_dir = private / f"R1_DEPLOY_S{seed}_{arm}"
            draws = replay_dir / "cached_fit_uniforms.npz"
            if arm not in ("B0", "D"):
                if draws.is_file():
                    entries.append(_entry(draws, role=f"seed{seed}.{arm}.draws",
                                          archive_name=f"seed{seed}/selected/{arm}/draws.npz",
                                          private=True))
                else:
                    # The root and FIT unit IDs are sufficient to reproduce the
                    # sealed 32 draws inside a queued, label-free restore.
                    for alias, path in (("sampling_root", package / "r2/.private/stream_roots_v1.json"),
                                        ("fit_features", Path(cfg["pins"]["role_features"]))):
                        entries.append(_entry(path, role=f"seed{seed}.{arm}.{alias}",
                                              archive_name=f"seed{seed}/selected/{arm}/{alias}{path.suffix}",
                                              private=True))
    archive_names = [item["archive_name"] for item in entries]
    if len(archive_names) != len(set(archive_names)):
        raise ValueError("restore archive names collide")
    result = {"schema": "pcrl-bank-r1-deployment-snapshot-dependencies-v1",
              "status": "SELECTED_POLICY_BOUND" if selected_arms else "AWAITING_POLICY",
              "selected_arms": list(selected_arms),
              "seeds": [0, 1], "disclosure": "private_restore_only",
              "content_rule": "No task or SEX label arrays are included; FIT features are predictor-only",
              "entries": entries}
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as dest:
        json.dump(result, dest, indent=2, sort_keys=True)
        dest.write("\n")
        dest.flush(); os.fsync(dest.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-arm", action="append", default=[])
    parser.add_argument("--fit-unit-id", action="append", default=[],
                        help="explicit R2_<ARM>_S<seed>_FIT[_<revision>] successor ID")
    args = parser.parse_args()
    overrides = {}
    for value in args.fit_unit_id:
        match = re.fullmatch(r"R2_(B0|T|D|P|L|J|S12|S21)_S([01])_FIT(?:_[A-Z][1-9][0-9]*)?",
                             value)
        if match is None:
            raise ValueError("selected FIT unit ID differs")
        key = (match.group(1), int(match.group(2)))
        if key in overrides:
            raise ValueError("duplicate selected FIT unit ID")
        overrides[key] = value
    result = build(args.root, args.output,
                   selected_arms=tuple(args.selected_arm), fit_unit_ids=overrides)
    print(json.dumps({"status": result["status"],
                      "entries": len(result["entries"]),
                      "manifest_sha256": _sha(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
