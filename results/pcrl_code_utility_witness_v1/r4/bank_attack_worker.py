"""Queue-only complete-view attack on an eligible inherited public bank.

The fixed W_Q/W_TSTAR public decoder table, rather than a learned latent
package, replays the code-only output. The private score router is not an
attacker feature. No real fit is admitted until R0 locks bank compatibility,
utility eligibility and this unit under the shared attacks queue stage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_file,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.protocol import run_three_view_attack
from results.pcrl_code_utility_witness_v1.r0.compute_queue import assert_active_from_env

from .attack_worker import (READER_PLAN, ROLES, _load_release, _pinned,
                            _private_output, _write_once)
from .bank_service import make_bank_service
from .code_views import prepare_complete_views


LOCK_SCHEMA = "pcrl-code-witness-bank-attack-lock-v1"
SOURCE_SCHEMA = "code-utility-witness-worker-v1"


def validate_bank_attack_lock(lock: Mapping, *, campaign_seal_sha256: str,
                              role_manifest_sha256: str, worker_sha256: str,
                              unit_id: str, arm_id: str, seed: int, bits: int,
                              source_model_sha256: str, source_receipt_sha256: str,
                              public_table_sha256: str,
                              policy_sha256: str, decoder_sha256: str | None,
                              release_sha256: Mapping[str, str]) -> None:
    if (not isinstance(lock, Mapping) or lock.get("schema") != LOCK_SCHEMA
            or lock.get("status") != "LOCALLY_SEALED_ATTACKS"
            or lock.get("campaign_seal_sha256") != campaign_seal_sha256
            or lock.get("role_manifest_sha256") != role_manifest_sha256
            or lock.get("worker_sha256") != worker_sha256
            or lock.get("reader_plan") != READER_PLAN):
        raise PermissionError("bank attack lock worker/source/slate differs")
    expected = {"arm_id": arm_id, "seed": seed, "bits": bits,
                "source_model_sha256": source_model_sha256,
                "source_receipt_sha256": source_receipt_sha256,
                "public_table_sha256": public_table_sha256,
                "policy_sha256": policy_sha256,
                "decoder_sha256": decoder_sha256,
                "release_sha256": dict(release_sha256)}
    if (arm_id not in ("W_Q", "W_TSTAR") or seed not in (0, 1) or bits != 10
            or set(release_sha256) != set(ROLES)
            or lock.get("units", {}).get(unit_id) != expected):
        raise PermissionError("bank attack lock unit/release registry differs")


def run(args: argparse.Namespace) -> dict:
    def guard() -> None:
        active = assert_active_from_env(unit_id=args.unit_id, stage="attacks")
        if active["seal_sha256"] != args.seal_sha256:
            raise PermissionError("bank attack campaign seal differs from active queue")

    guard()
    seal = _pinned(args.seal, args.seal_sha256)
    attack_lock = _pinned(args.attack_lock, args.attack_lock_sha256)
    role_path = _pinned(args.role_manifest, args.role_manifest_sha256, private=True)
    table_path = _pinned(args.public_table, args.public_table_sha256)
    source_receipt_path = _pinned(args.source_receipt, args.source_receipt_sha256)
    release = {role: _pinned(getattr(args, f"release_{role.lower()}"),
                             getattr(args, f"release_{role.lower()}_sha256"),
                             private=True) for role in ROLES}
    private_out = _private_output(args.private_output)
    public_out = Path(args.public_receipt).resolve(strict=False)
    if public_out.exists() or not public_out.parent.is_dir():
        raise FileExistsError("new public bank attack receipt path required")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    if not any(item["sha256"] == args.attack_lock_sha256 and "attacks" in item["stages"]
               for item in state.get("amendments", [])):
        raise PermissionError("R0 did not register bank attack stage lock")
    lock = json.loads(attack_lock.read_text())
    hashes = {role: getattr(args, f"release_{role.lower()}_sha256") for role in ROLES}
    validate_bank_attack_lock(
        lock, campaign_seal_sha256=args.seal_sha256,
        role_manifest_sha256=args.role_manifest_sha256,
        worker_sha256=sha_file(Path(__file__)), unit_id=args.unit_id,
        arm_id=args.arm_id, seed=args.seed, bits=args.bits,
        source_model_sha256=args.source_model_sha256,
        source_receipt_sha256=args.source_receipt_sha256,
        public_table_sha256=args.public_table_sha256,
        policy_sha256=args.policy_sha256, decoder_sha256=args.decoder_sha256,
        release_sha256=hashes)
    source = json.loads(source_receipt_path.read_text())
    if (source.get("schema") != SOURCE_SCHEMA or source.get("stage") != "bank"
            or source.get("status") != "COMPLETE" or source.get("seed") != args.seed
            or source.get("role_manifest_sha256") != args.role_manifest_sha256
            or source.get("source_model_sha256") != args.source_model_sha256
            or args.arm_id not in source.get("result", {}).get("arm_ids", ())):
        raise PermissionError("pinned W_BANK compilation receipt differs")
    for role in ROLES:
        item = source["result"]["roles"][role]["arms"][args.arm_id]
        if (item["private_release_sha256"] != hashes[role]
                or item["private_release_basename"] != release[role].name
                or item["policy_sha256"] != args.policy_sha256
                or item["decoder_sha256"] != args.decoder_sha256
                or item["public_table_sha256"] != args.public_table_sha256):
            raise PermissionError("bank receipt/release/policy bytes differ")
    store = RoleOnlyStore(role_path)
    table = json.loads(table_path.read_text())
    service = make_bank_service(
        table, arm_id=args.arm_id, seed=args.seed,
        source_model_sha256=args.source_model_sha256,
        feature_schema_sha256=store.manifest["source_schema_sha256"],
        public_table_sha256=args.public_table_sha256,
        policy_sha256=args.policy_sha256, decoder_sha256=args.decoder_sha256)
    att = _load_release(release["ATTACK_FIT_NEW"], hashes["ATTACK_FIT_NEW"],
                        store.manifest["roles"]["ATTACK_FIT_NEW"], service, bits=10)
    inner = _load_release(release["INNER_SELECTION"], hashes["INNER_SELECTION"],
                          store.manifest["roles"]["INNER_SELECTION"], service, bits=10)
    prepared = prepare_complete_views(att, inner, bits=10)
    guard()
    sex_att = store.read("ATTACK_FIT_NEW", "sex", procedure="attack")
    assert_same_people(sex_att, att)

    def load_inner_sex() -> dict:
        guard()
        if not (private_out / "FROZEN_COMPLETE.json").is_file():
            raise PermissionError("INNER SEX cannot open before bank readers freeze")
        record = store.read("INNER_SELECTION", "sex", procedure="screen")
        assert_same_people(record, inner)
        return record

    attack = run_three_view_attack(
        prepared, sex_att, load_inner_sex=load_inner_sex, family="D",
        seed=args.attacker_seed, private_root=private_out,
        lock_path=seal, lock_sha256=args.seal_sha256, queue_guard=guard)
    receipt = {"schema": "pcrl-code-witness-bank-attack-receipt-v1",
               "status": "COMPLETE_PRIVATE_DEVELOPMENT_ONLY",
               "unit_id": args.unit_id, "arm_id": args.arm_id,
               "source_seed": args.seed, "attacker_seed": args.attacker_seed,
               "bits": 10, "campaign_seal_sha256": args.seal_sha256,
               "attack_lock_sha256": args.attack_lock_sha256,
               "role_manifest_sha256": args.role_manifest_sha256,
               "source_model_sha256": args.source_model_sha256,
               "public_table_sha256": args.public_table_sha256,
               "policy_sha256": args.policy_sha256,
               "decoder_sha256": args.decoder_sha256,
               "source_receipt_sha256": args.source_receipt_sha256,
               "release_sha256": hashes, "attack": attack["public"]}
    _write_once(public_out, receipt)
    return {"status": receipt["status"], "unit_id": args.unit_id,
            "receipt_sha256": sha_file(public_out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for flag in ("seal", "seal-sha256", "attack-lock", "attack-lock-sha256",
                 "unit-id", "role-manifest", "role-manifest-sha256",
                 "public-table", "public-table-sha256", "source-receipt",
                 "source-receipt-sha256", "source-model-sha256", "policy-sha256",
                 "release-attack-fit-new", "release-attack-fit-new-sha256",
                 "release-inner-selection", "release-inner-selection-sha256",
                 "arm-id", "private-output", "public-receipt"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--decoder-sha256")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--attacker-seed", type=int, required=True)
    parser.add_argument("--bits", type=int, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
