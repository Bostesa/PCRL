"""Queue-only bounded utility intervals for T0 and eligible inherited banks.

Only admitted INNER role labels/references are opened. Candidate releases are
first byte-pinned, aligned, and replayed from their public code-only service.
All privacy, absent capacity, and unavailable F-reference clauses stay null.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
from typing import Mapping

import numpy as np

from code_witness.service import load_service
from results.pcrl_code_utility_witness_v1.r0.compute_queue import assert_active_from_env
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_array, sha_file,
)

from .attack_worker import (_load_release, _role_manifest, _write_once,
                            validate_service_manifest_binding)
from .bank_service import make_bank_service
from . import finite_family, inference_rule, utility_intervals
from .finite_family import ordered_clause_ids
from .utility_intervals import BOOTSTRAP_SEED, evaluate_utility_family


CONFIG_SCHEMA = "pcrl-r4-utility-interval-config-v1"
ROLE = "INNER_SELECTION"


def validate_interval_config(config: Mapping, *, unit_id: str,
                             worker_sha256: str, utility_source_sha256: str,
                             inference_rule_source_sha256: str,
                             finite_family_source_sha256: str,
                             seal_sha256: str,
                             inference_lock_sha256: str) -> None:
    if (not isinstance(config, Mapping) or config.get("schema") != CONFIG_SCHEMA
            or config.get("unit_id") != unit_id
            or config.get("worker_sha256") != worker_sha256
            or config.get("utility_source_sha256") != utility_source_sha256
            or config.get("inference_rule_source_sha256")
               != inference_rule_source_sha256
            or config.get("finite_family_source_sha256")
               != finite_family_source_sha256
            or config.get("seal", {}).get("sha256") != seal_sha256
            or config.get("inference_lock", {}).get("sha256") != inference_lock_sha256
            or Path(config.get("role_manifest", {}).get("path", "")).name
               != "DATA_ACCESS_MANIFEST.json"
            or Path(config.get("inference_proposal", {}).get("path", "")).name
               != "INFERENCE_RULE_PROPOSAL.json"
            or Path(config.get("finite_family_proposal", {}).get("path", "")).name
               != "D_FINITE_FAMILY_PROPOSAL.json"
            or not isinstance(config.get("candidates"), list)
            or not isinstance(config.get("output"), str)):
        raise PermissionError("utility interval config source or schema differs")
    seen = set()
    for item in config["candidates"]:
        if not isinstance(item, Mapping):
            raise PermissionError("utility interval config candidate differs")
        arm, kind, seed = item.get("arm_id"), item.get("kind"), item.get("seed")
        if (seed not in (0, 1) or (arm, seed) in seen
                or (kind == "learned" and arm not in ("T0_B10", "T0_B16"))
                or (kind == "bank" and arm not in ("W_Q", "W_TSTAR"))
                or kind not in ("learned", "bank")):
            raise PermissionError("utility interval config arm/seed differs")
        seen.add((arm, seed))
        keys = {"arm_id", "kind", "seed", "source_model_sha256",
                "receipt", "release", "service_manifest" if kind == "learned"
                else "public_table"}
        if kind == "bank":
            keys |= {"policy_sha256", "decoder_sha256"}
        if set(item) != keys:
            raise PermissionError("utility interval config candidate inventory differs")


def _registered(config_path: Path, unit_id: str) -> dict[Path, str]:
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    sources = {Path(rec["path"]).resolve(strict=True): rec["sha256"]
               for rec in state["units"][unit_id]["spec"]["sources"]}
    if sources.get(config_path) != sha_file(config_path):
        raise PermissionError("interval config is not registered with queue")
    return sources


def _record(record: Mapping, sources: Mapping[Path, str]) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise PermissionError("interval source record differs")
    raw = Path(record["path"])
    if raw.is_symlink():
        raise PermissionError("interval source is a symlink")
    path = raw.resolve(strict=True)
    if (not path.is_file() or sources.get(path) != record["sha256"]
            or sha_file(path) != record["sha256"]):
        raise PermissionError("interval source bytes are not queue pinned")
    return path


def _role_sources(store: RoleOnlyStore, sources: Mapping[Path, str]) -> None:
    for field in ("income", "occupation", "references_s0", "references_s1"):
        key = f"{ROLE}:{field}"
        path = (store.package_root /
                store.manifest["role_file_relative_path"][key]).resolve(strict=True)
        expected = store.manifest["role_file_sha256"][key]
        if sources.get(path) != expected or sha_file(path) != expected:
            raise PermissionError("INNER label/reference archive is not queue pinned")


def _candidate(entry: Mapping, *, sources: Mapping[Path, str],
               seal: Mapping, store: RoleOnlyStore, role_meta: Mapping) -> dict:
    arm, seed = entry["arm_id"], entry["seed"]
    receipt_path = _record(entry["receipt"], sources)
    release_path = _record(entry["release"], sources)
    receipt = json.loads(receipt_path.read_text())
    expected_source = seal["source_model_files_sha256_by_seed"][str(seed)]["model_pt_sha256"]
    if (entry["source_model_sha256"] != expected_source
            or receipt.get("schema") != "code-utility-witness-worker-v1"
            or receipt.get("status") != "COMPLETE"
            or receipt.get("seed") != seed
            or receipt.get("source_model_sha256") != expected_source
            or receipt.get("role_manifest_sha256") != seal["role_manifest_sha256"]
            or receipt.get("queue_seal_sha256") != seal["_seal_sha256"]):
        raise PermissionError("utility candidate source receipt differs")
    result = receipt["result"]
    if entry["kind"] == "learned":
        if (receipt.get("stage") != "fit" or result.get("arm_id") != arm
                or result.get("bit_count") != int(arm.rsplit("B", 1)[1])):
            raise PermissionError("learned utility candidate stage/capacity differs")
        item = result["roles"][ROLE]
        manifest_path = _record(entry["service_manifest"], sources)
        if (".private" not in manifest_path.parts
                or stat.S_IMODE(manifest_path.stat().st_mode) & 0o077):
            raise PermissionError("learned utility package must stay owner-only")
        if (item["private_release_sha256"] != entry["release"]["sha256"]
                or item["private_release_basename"] != release_path.name
                or result["package"]["manifest_sha256"]
                   != entry["service_manifest"]["sha256"]):
            raise PermissionError("learned utility release/package differs")
        manifest = json.loads(manifest_path.read_text())
        validate_service_manifest_binding(
            manifest, arm_id=arm, seed=seed,
            source_model_sha256=expected_source,
            role_manifest_sha256=seal["role_manifest_sha256"],
            source_schema_sha256=store.manifest["source_schema_sha256"],
            bits=result["bit_count"])
        weights = manifest_path.with_name("weights.npz").resolve(strict=True)
        if (sources.get(weights) != manifest["weights_sha256"]
                or sha_file(weights) != manifest["weights_sha256"]
                or stat.S_IMODE(weights.stat().st_mode) & 0o077):
            raise PermissionError("learned utility service weights are not queue pinned")
        service = load_service(manifest_path,
                               expected_manifest_sha256=entry["service_manifest"]["sha256"],
                               expected_source_sha256=expected_source,
                               expected_bits=result["bit_count"])
        bits = result["bit_count"]
    else:
        if receipt.get("stage") != "bank" or arm not in result.get("arm_ids", []):
            raise PermissionError("bank utility candidate stage/arm differs")
        item = result["roles"][ROLE]["arms"][arm]
        table_path = _record(entry["public_table"], sources)
        if (item["private_release_sha256"] != entry["release"]["sha256"]
                or item["private_release_basename"] != release_path.name
                or item["public_table_sha256"] != entry["public_table"]["sha256"]
                or item["policy_sha256"] != entry["policy_sha256"]
                or item["decoder_sha256"] != entry["decoder_sha256"]):
            raise PermissionError("bank utility release/table differs")
        service = make_bank_service(
            json.loads(table_path.read_text()), arm_id=arm, seed=seed,
            source_model_sha256=expected_source,
            feature_schema_sha256=store.manifest["source_schema_sha256"],
            public_table_sha256=entry["public_table"]["sha256"],
            policy_sha256=entry["policy_sha256"],
            decoder_sha256=entry["decoder_sha256"])
        bits = 10
    return _load_release(release_path, entry["release"]["sha256"],
                         role_meta, service, bits=bits)


def run(config_path: Path) -> dict:
    active = assert_active_from_env(stage="verification")
    path = config_path.resolve(strict=True)
    if (".private" not in path.parts or not path.is_file()
            or stat.S_IMODE(path.stat().st_mode) & 0o077
            or stat.S_IMODE(path.parent.stat().st_mode) & 0o077):
        raise PermissionError("interval config must be owner-only")
    sources = _registered(path, active["unit_id"])
    config = json.loads(path.read_text())
    validate_interval_config(
        config, unit_id=active["unit_id"], worker_sha256=sha_file(Path(__file__)),
        utility_source_sha256=sha_file(Path(utility_intervals.__file__)),
        inference_rule_source_sha256=sha_file(Path(inference_rule.__file__)),
        finite_family_source_sha256=sha_file(Path(finite_family.__file__)),
        seal_sha256=active["seal_sha256"],
        inference_lock_sha256=config["inference_lock"]["sha256"])
    for source in (Path(__file__), Path(utility_intervals.__file__),
                   Path(inference_rule.__file__), Path(finite_family.__file__)):
        if sources.get(source) != sha_file(source):
            raise PermissionError("utility interval implementation is not queue pinned")
    seal_path = _record(config["seal"], sources)
    lock_path = _record(config["inference_lock"], sources)
    _record(config["inference_proposal"], sources)
    _record(config["finite_family_proposal"], sources)
    seal = json.loads(seal_path.read_text())
    seal["_seal_sha256"] = active["seal_sha256"]
    lock = json.loads(lock_path.read_text())
    if (lock.get("schema") != "pcrl-code-utility-inference-randomness-lock-v1"
            or lock.get("status") != "LOCALLY_SEALED_BEFORE_T0_INNER_GATE"
            or lock.get("campaign_lock_sha256") != active["seal_sha256"]
            or lock.get("randomness", {}).get("seed_integer") != BOOTSTRAP_SEED
            or lock.get("paired_draws") != 2048
            or lock.get("fixed_family_K") != 2376
            or lock.get("inference_rule_sha256")
               != config["inference_proposal"]["sha256"]
            or lock.get("finite_family_sha256")
               != config["finite_family_proposal"]["sha256"]):
        raise PermissionError("sealed utility interval randomness differs")
    role_path = _record(config["role_manifest"], sources)
    _role_manifest(str(role_path), config["role_manifest"]["sha256"])
    if config["role_manifest"]["sha256"] != seal["role_manifest_sha256"]:
        raise PermissionError("utility role manifest differs from campaign seal")
    store = RoleOnlyStore(role_path)
    if store.manifest["source_schema_sha256"] != seal["source_schema_sha256"]:
        raise PermissionError("utility role schema differs from seal")
    _role_sources(store, sources)
    role_meta = store.manifest["roles"][ROLE]
    wires = {}
    for entry in config["candidates"]:
        active = assert_active_from_env(unit_id=config["unit_id"], stage="verification")
        wire = _candidate(entry, sources=sources, seal=seal, store=store,
                          role_meta=role_meta)
        wires[(entry["seed"], entry["arm_id"])] = wire
    income = store.read(ROLE, "income", procedure="screen")
    occupation = store.read(ROLE, "occupation", procedure="screen")
    assert_same_people(income, occupation)
    if (sha_array(np.column_stack((income["row_id"], income["unit"])))
            != role_meta["row_unit_pair_sha256"]):
        raise PermissionError("utility INNER people differ from role identity")
    references, candidates = {}, {}
    for seed in (0, 1):
        reference = store.read(ROLE, f"references_s{seed}")
        assert_same_people(income, reference)
        for task, purpose in (("income", 0), ("occupation", 1)):
            for ref in ("U0", "Ucal"):
                references[(seed, task, ref)] = reference[f"{ref}_{purpose+1}"]
    for (seed, arm), wire in wires.items():
        assert_same_people(income, wire)
        for task, name in (("income", "first"), ("occupation", "second")):
            candidates[(seed, arm, task)] = wire[name]["probabilities"]
    result = evaluate_utility_family(
        candidates, references,
        {"income": income["income"], "occupation": occupation["occupation"]},
        income["unit"])
    if result["ordered_ids_sha256"] != lock["ordered_ids_sha256"]:
        raise PermissionError("utility family order differs from sealed registry")
    result.update({"status": "COMPLETE_PRIVATE_DEVELOPMENT_AGGREGATES",
                   "unit_id": config["unit_id"],
                   "campaign_seal_sha256": config["seal"]["sha256"],
                   "inference_lock_sha256": config["inference_lock"]["sha256"],
                   "role_manifest_sha256": config["role_manifest"]["sha256"],
                   "utility_source_sha256": config["utility_source_sha256"],
                   "candidate_ids": sorted(set(arm for _, arm in wires))})
    output = Path(config["output"]).resolve(strict=False)
    if output.exists() or output.parent != Path(__file__).parent:
        raise PermissionError("new public R4 aggregate output required")
    _write_once(output, result)
    return {"status": result["status"], "unit_id": config["unit_id"],
            "estimable_clauses": result["estimable_clauses"],
            "receipt_sha256": sha_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True, type=Path)
    print(json.dumps(run(parser.parse_args().config), sort_keys=True))


if __name__ == "__main__":
    main()
