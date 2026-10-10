"""Build one pinned R3 FIT/INNER scoring unit from a completed R2 law.

Only file bytes and public/private receipts are inspected. Role archives and
the law NPZ are hashed as opaque files; task/SEX labels are never opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from .r3_make_witness_configs import ROOT, OUT, record, sha, write_new


PROTECTED_ARMS = ("T", "D", "P", "L", "J", "S12", "S21")


def _bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def derive_documents(*, base: dict, source_spec: dict, release: dict,
                     arm: str, unit_id: str, law_record: dict,
                     release_record: dict, policy: dict, policy_record: dict,
                     policy_package_record: dict, identity: dict,
                     identity_record: dict, amendment_record: dict,
                     config_path: str, outputs: dict, deps: list[str],
                     max_wall_s: int = 60, cpu_reserve_s: int = 60) -> tuple[dict, dict]:
    """Derive immutable successor documents without reading any role array."""
    if (base.get("schema") != "pcrl-bank-r3-utility-job-v1"
            or base.get("stage") != "witness" or base.get("arm") != "B0"
            or base.get("law") is not None
            or base.get("role") not in ("OSF_DEFENSE_FIT", "INNER_SELECTION")
            or base.get("seed") not in (0, 1)
            or source_spec.get("unit_id") != base.get("unit_id")
            or source_spec.get("stage") != "witness"
            or arm not in PROTECTED_ARMS or not unit_id or unit_id == base["unit_id"]
            or release.get("schema") != "BAR1-role-release-v1"
            or release.get("arm_id") != arm
            or release.get("source_seed") != base["seed"]
            or release.get("role") != base["role"]
            or release.get("realization") != 0
            or release.get("law_sha256") != law_record.get("sha256")
            or release.get("policy_receipt_sha256") != policy_record.get("sha256")
            or policy.get("schema") != "BAR1-selected-policy-v1"
            or policy.get("arm_id") != arm
            or policy.get("source_seed") != base["seed"]
            or policy.get("policy_file_sha256") != policy_package_record.get("sha256")
            or identity.get("schema") != "BAR1-r2-full-fit-identity-v1"
            or identity.get("status") != "PASS"
            or identity.get("source_seed") != base["seed"]
            or identity.get("role") != "OSF_DEFENSE_FIT"
            or identity.get("source_lock_sha256") != base["seal"]["sha256"]
            or release.get("source_binding_sha256") != policy.get("source_binding_sha256")
            or policy.get("source_binding_sha256") != identity.get("source_binding_sha256")
            or set(outputs) != {"csv", "private_vectors", "receipt"}
            or set(outputs.values()) & set(base["outputs"].values())
            or not deps or len(deps) != len(set(deps))
            or max_wall_s <= 0 or cpu_reserve_s <= 0):
        raise PermissionError("R3 protected law/source identity differs")
    cfg = {**base, "unit_id": unit_id, "stage": "protection", "arm": arm,
           "law": law_record, "outputs": outputs}
    config_rec = {"path": str(config_path),
                  "sha256": hashlib.sha256(_bytes(cfg)).hexdigest()}
    old_sources = source_spec.get("sources", [])
    if not old_sources or not isinstance(old_sources, list):
        raise PermissionError("B0 source pins missing")
    # The first source is the old B0 config. Retain all inherited immutable
    # bank, role, worker and archive sources; replace only that config.
    sources = [config_rec, *old_sources[1:]]
    for item in (law_record, release_record, policy_record,
                 policy_package_record, identity_record, amendment_record):
        if item not in sources:
            sources.append(item)
    spec = {"unit_id": unit_id, "stage": "protection",
            "argv": [sys.executable, "-m", "bank_anchored.r3_worker",
                     "--config", str(config_path)],
            "cwd": str(ROOT), "max_wall_s": max_wall_s,
            "cpu_reserve_s": cpu_reserve_s, "max_rss_gib": 1.0,
            "projected_output_bytes": 8_000_000, "deps": deps,
            "outputs": list(outputs.values()), "env": {}, "sources": sources,
            "amendment_sha256": amendment_record["sha256"]}
    return cfg, spec


def build(*, base_config: Path, base_spec: Path, law: Path,
          release_receipt: Path, policy_receipt: Path,
          policy_package: Path, source_identity: Path,
          amendment: Path, arm: str,
          unit_id: str, deps: list[str]) -> dict:
    base_config_rec = record(base_config)
    base_spec_rec = record(base_spec)
    base = json.loads(base_config.read_text())
    old_spec = json.loads(base_spec.read_text())
    if (old_spec.get("sources", [{}])[0] != base_config_rec
            or old_spec.get("unit_id") != base["unit_id"]):
        raise PermissionError("B0 config/source spec pair differs")
    for source in old_spec["sources"]:
        record(Path(source["path"]), source["sha256"])
    law_record = record(law)
    release_record = record(release_receipt)
    policy_record = record(policy_receipt)
    policy_package_record = record(policy_package)
    identity_record = record(source_identity)
    amendment_record = record(amendment)
    release = json.loads(release_receipt.read_text())
    policy = json.loads(policy_receipt.read_text())
    identity = json.loads(source_identity.read_text())
    if (".private" not in Path(law_record["path"]).parts
            or ".private" not in Path(release_record["path"]).parts
            or ".private" not in Path(policy_record["path"]).parts
            or ".private" not in Path(policy_package_record["path"]).parts):
        raise PermissionError("R2 law, release and selected policy must be private")
    private = OUT / ".private"
    private.mkdir(mode=0o700, exist_ok=True)
    os.chmod(private, 0o700)
    config_path = private / f"{unit_id}_config.json"
    csv_path = OUT / f"{unit_id}.csv"
    vectors_path = private / f"{unit_id}_vectors.npz"
    receipt_path = OUT / f"{unit_id}_RECEIPT.json"
    outputs = {"csv": str(csv_path), "private_vectors": str(vectors_path),
               "receipt": str(receipt_path)}
    cfg, spec = derive_documents(base=base, source_spec=old_spec, release=release,
                                 arm=arm, unit_id=unit_id, law_record=law_record,
                                 release_record=release_record,
                                 policy=policy, policy_record=policy_record,
                                 policy_package_record=policy_package_record,
                                 identity=identity, identity_record=identity_record,
                                 amendment_record=amendment_record,
                                 config_path=str(config_path), outputs=outputs,
                                 deps=deps)
    cfg_sha = write_new(config_path, _bytes(cfg), mode=0o600)
    if cfg_sha != spec["sources"][0]["sha256"]:
        raise AssertionError("private config byte digest differs")
    spec_path = private / f"{unit_id}_queue_spec.json"
    spec_sha = write_new(spec_path, _bytes(spec), mode=0o600)
    index = {"schema": "pcrl-bank-r3-protection-unit-index-v1",
             "unit_id": unit_id, "arm": arm, "seed": base["seed"],
             "role": base["role"], "config_sha256": cfg_sha,
             "queue_spec_sha256": spec_sha,
             "source_b0_config_sha256": base_config_rec["sha256"],
             "source_b0_spec_sha256": base_spec_rec["sha256"],
             "r2_law_sha256": law_record["sha256"],
             "r2_release_receipt_sha256": release_record["sha256"],
             "r2_policy_receipt_sha256": policy_record["sha256"],
             "r2_policy_package_sha256": policy_package_record["sha256"],
             "r2_full_fit_identity_sha256": identity_record["sha256"],
             "source_binding_sha256": identity["source_binding_sha256"],
             "amendment_sha256": amendment_record["sha256"],
             "dependency_unit_ids": deps}
    index_path = OUT / f"{unit_id}_UNIT_INDEX.json"
    index_sha = write_new(index_path, _bytes(index), mode=0o644)
    return {**index, "index_sha256": index_sha}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("base-config", "base-spec", "law", "release-receipt",
                 "policy-receipt", "policy-package", "source-identity",
                 "amendment"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=PROTECTED_ARMS, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--dep", action="append", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(base_config=args.base_config, base_spec=args.base_spec,
                           law=args.law, release_receipt=args.release_receipt,
                           policy_receipt=args.policy_receipt,
                           policy_package=args.policy_package,
                           source_identity=args.source_identity,
                           amendment=args.amendment, arm=args.arm,
                           unit_id=args.unit_id, deps=args.dep), sort_keys=True))


if __name__ == "__main__":
    main()
