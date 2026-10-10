"""Receipt-only builder for two P* H2 technical successor attack units.

The two H1 failures, all other H1 results and every historical artifact stay
immutable. Preparation requires R0's first dated correction amendment;
finalization requires its subsequent exact H2 attack-stage lock.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Mapping

from . import reference_worker_h2, spec_builder


ROOT = Path(__file__).resolve().parent
SCHEMA = "pcrl-bank-r4-pstar-h2-build-config-v1"
INDEX_SCHEMA = "pcrl-bank-r4-pstar-h2-unit-index-v1"
QUEUE_SCHEMA = "pcrl-bank-r4-pstar-h2-queue-index-v1"
CONFIG_KEYS = frozenset({
    "schema", "h1_config_pin", "h1_attack_index_pin",
    "h1_queue_index_pin", "h1_stage_lock_pin", "h2_source_plan_pin",
    "h2_correction_amendment_pin",
})


def _source_pins() -> list[dict[str, str]]:
    names = spec_builder._modules("pstar")
    names.pop("reference_worker.py")
    names["reference_worker_h2.py"] = reference_worker_h2.__name__
    names["pstar_views_h2.py"] = (
        "results.pcrl_bank_anchored_release_v1.r4.pstar_views_h2")
    if set(names) != reference_worker_h2.REQUIRED_SOURCES:
        raise AssertionError("H2 worker/source inventory differs")
    return [spec_builder._pin(Path(importlib.import_module(names[name]).__file__))
            for name in sorted(names)]


def _load(config_path: str, *, expected_latest: str | None = None) -> tuple[dict, dict, dict]:
    path = Path(config_path).resolve(strict=True)
    if ".private" not in path.parts or path.stat().st_mode & 0o077:
        raise PermissionError("H2 build config must be owner-only")
    config = json.loads(path.read_text())
    if set(config) != CONFIG_KEYS or config.get("schema") != SCHEMA:
        raise ValueError("H2 build config schema differs")
    h1 = spec_builder._read_pin(config["h1_config_pin"], private=True)
    index = spec_builder._read_pin(config["h1_attack_index_pin"])
    queue = spec_builder._read_pin(config["h1_queue_index_pin"])
    lock = spec_builder._read_pin(config["h1_stage_lock_pin"])
    plan = spec_builder._read_pin(config["h2_source_plan_pin"])
    correction = spec_builder._read_pin(config["h2_correction_amendment_pin"])
    if (h1.get("schema") != spec_builder.SCHEMA or
            index.get("schema") != spec_builder.INDEX_SCHEMA or
            queue.get("schema") != spec_builder.QUEUE_INDEX_SCHEMA or
            lock.get("schema") != reference_worker_h2.LOCK_SCHEMA or
            index.get("stage_lock_units") != lock.get("units") or
            queue.get("attack_unit_index_sha256") !=
            config["h1_attack_index_pin"]["sha256"] or
            queue.get("stage_lock_sha256") !=
            config["h1_stage_lock_pin"]["sha256"] or
            plan.get("schema") != "pcrl-bank-r4-pstar-h2-source-plan-v1" or
            plan.get("status") != "PLAN_ONLY_NO_H2_WORKER_SPEC" or
            plan.get("unit_count") != 2 or plan.get("planned_fits") != 60 or
            plan.get("h1_attack_unit_index_sha256") !=
            config["h1_attack_index_pin"]["sha256"] or
            plan.get("h1_attack_queue_index_sha256") !=
            config["h1_queue_index_pin"]["sha256"] or
            plan.get("h1_stage_lock_sha256") !=
            config["h1_stage_lock_pin"]["sha256"] or
            plan.get("h2_worker_sha256") !=
            spec_builder._sha(Path(reference_worker_h2.__file__)) or
            plan.get("h2_view_adapter_sha256") !=
            spec_builder._sha(Path(importlib.import_module(
                "results.pcrl_bank_anchored_release_v1.r4.pstar_views_h2").__file__)) or
            correction.get("h2_source_plan_sha256") !=
            config["h2_source_plan_pin"]["sha256"] or
            correction.get("successor_suffix") != "H2" or
            correction.get("unit_count") != 2):
        raise PermissionError("H2 correction, source or predecessor science differs")
    state = json.loads((Path(h1["queue_root"]) / "state.json").read_text())
    amendments = [a for a in state.get("amendments", [])
                  if "attacks" in a.get("stages", [])]
    if (not any(a["sha256"] == config["h2_correction_amendment_pin"]["sha256"]
                for a in amendments) or
            (expected_latest and amendments[-1]["sha256"] != expected_latest)):
        raise PermissionError("H2 correction is not the required attacks amendment")
    entries = {row["unit_id"]: row for row in index["units"]}
    old_queue = {row["unit_id"]: row for row in queue["units"]}
    if (len(plan["units"]) != 2 or
            {row["source_seed"] for row in plan["units"]} != {0, 1}):
        raise PermissionError("H2 must cover both P* seeds exactly")
    for row in plan["units"]:
        seed = row["source_seed"]
        uid = f"R4_ATTACK_P_STAR_S{seed}_R0_H1"
        if (row.get("predecessor_unit_id") != uid or
                row.get("successor_unit_id") != uid[:-2] + "H2" or
                row.get("arm_id") != "P_STAR" or
                row.get("realization") != 0 or
                row.get("attacker_seed") != 100000 + 1000 * seed or
                row.get("predecessor_worker_spec_sha256") !=
                entries[uid]["spec_sha256"] or
                row.get("predecessor_queue_spec_sha256") !=
                old_queue[uid]["queue_spec_sha256"] or
                row.get("predecessor_failure_status") != "failed" or
                row.get("predecessor_returncode") != 1 or
                state["units"][uid]["status"] != "failed" or
                state["units"][uid]["returncode"] != 1):
            raise PermissionError("H2 P* failed predecessor identity differs")
    return config, h1, plan


def _new_spec(old: Mapping, row: Mapping) -> dict:
    uid = row["successor_unit_id"]
    if (old["unit_id"] != row["predecessor_unit_id"] or
            old["arm_id"] != "P_STAR" or
            old["source_seed"] != row["source_seed"] or
            old["realization"] != 0 or
            old["attacker_seed"] != row["attacker_seed"]):
        raise PermissionError("H2 predecessor attack contract differs")
    spec = dict(old)
    spec.update({"schema": reference_worker_h2.SPEC_SCHEMA,
                 "unit_id": uid,
                 "private_output_path": str(Path(old["private_output_path"]).with_name(uid)),
                 "public_receipt_path": str(Path(old["public_receipt_path"]).with_name(
                     uid + "_RECEIPT.json")),
                 "source_pins": _source_pins()})
    allowed = {"schema", "unit_id", "private_output_path", "public_receipt_path",
               "source_pins"}
    if {k: v for k, v in old.items() if k not in allowed} != \
            {k: v for k, v in spec.items() if k not in allowed}:
        raise AssertionError("H2 changed another scientific attack field")
    reference_worker_h2.validate_spec(spec)
    return spec


def prepare(config_path: str) -> dict:
    config, h1, plan = _load(config_path, expected_latest=None)
    state = json.loads((Path(h1["queue_root"]) / "state.json").read_text())
    latest = next(a["sha256"] for a in reversed(state["amendments"])
                  if "attacks" in a.get("stages", []))
    if latest != config["h2_correction_amendment_pin"]["sha256"]:
        raise PermissionError("H2 preparation requires latest correction amendment")
    private = Path(h1["private_root"])
    public = Path(h1["public_root"])
    spec_builder._private_directory(private / "units")
    spec_builder._private_directory(private / "queue_units")
    spec_builder._private_directory(private / "attacks")
    entries, lock_units = [], {}
    for row in plan["units"]:
        old_path = private / "units" / (row["predecessor_unit_id"] + ".json")
        if spec_builder._sha(old_path) != row["predecessor_worker_spec_sha256"]:
            raise PermissionError("failed H1 P* worker spec bytes differ")
        spec = _new_spec(json.loads(old_path.read_text()), row)
        uid = row["successor_unit_id"]
        spec_sha = spec_builder._write_once(private / "units" / (uid + ".json"),
                                             spec, 0o600)
        entries.append({"unit_id": uid, "arm_id": "P_STAR",
                        "source_seed": row["source_seed"], "realization": 0,
                        "worker_kind": "pstar_h2", "spec_sha256": spec_sha,
                        "planned_fits": 30})
        lock_units[uid] = {"spec_sha256": spec_sha, "arm_id": "P_STAR",
                           "source_seed": row["source_seed"],
                           "realization": 0, "point_eligible": True}
    index = {"schema": INDEX_SCHEMA, "status": "SPECS_READY_NO_ATTACK_FIT",
             "campaign_seal_sha256": h1["campaign_seal_sha256"],
             "builder_sha256": spec_builder._sha(Path(__file__)),
             "h2_source_plan_sha256": config["h2_source_plan_pin"]["sha256"],
             "h2_correction_amendment_sha256":
                 config["h2_correction_amendment_pin"]["sha256"],
             "predecessor_attack_unit_index_sha256":
                 config["h1_attack_index_pin"]["sha256"],
             "unit_count": 2, "planned_fits": 60,
             "units": entries, "stage_lock_units": lock_units}
    digest = spec_builder._write_once(public / "PSTAR_H2_UNIT_INDEX.json", index, 0o644)
    return {"status": index["status"], "index_sha256": digest, "unit_count": 2}


def finalize(config_path: str, stage_lock_path: str,
             stage_lock_sha256: str) -> dict:
    config, h1, plan = _load(config_path)
    public = Path(h1["public_root"])
    private = Path(h1["private_root"])
    index_path = public / "PSTAR_H2_UNIT_INDEX.json"
    index = json.loads(index_path.read_text())
    if (index.get("schema") != INDEX_SCHEMA or
            index.get("builder_sha256") != spec_builder._sha(Path(__file__)) or
            index.get("h2_source_plan_sha256") !=
            config["h2_source_plan_pin"]["sha256"]):
        raise PermissionError("H2 prepared index differs from active builder")
    lock_path = Path(stage_lock_path).resolve(strict=True)
    lock = spec_builder._read_pin({"path": str(lock_path),
                                   "sha256": stage_lock_sha256})
    if (lock.get("schema") != reference_worker_h2.LOCK_SCHEMA or
            lock.get("status") != "LOCALLY_SEALED_ATTACKS" or
            lock.get("campaign_seal_sha256") != h1["campaign_seal_sha256"] or
            lock.get("units") != index["stage_lock_units"]):
        raise PermissionError("H2 exact attack-stage lock differs")
    state = json.loads((Path(h1["queue_root"]) / "state.json").read_text())
    latest = next(a["sha256"] for a in reversed(state["amendments"])
                  if "attacks" in a.get("stages", []))
    if latest != stage_lock_sha256:
        raise PermissionError("H2 lock is not the latest registered attacks amendment")
    python = spec_builder._python_entrypoint(h1["python_executable"])
    entries = []
    for row in index["units"]:
        uid = row["unit_id"]
        spec_path = private / "units" / (uid + ".json")
        if spec_builder._sha(spec_path) != row["spec_sha256"]:
            raise PermissionError("H2 worker spec differs from exact stage lock")
        spec = json.loads(spec_path.read_text())
        sources = spec_builder._queue_sources(spec, spec_path, lock_path) + [
            h1["bounds_pin"], config["h1_config_pin"],
            config["h1_attack_index_pin"], config["h1_queue_index_pin"],
            config["h1_stage_lock_pin"], config["h2_source_plan_pin"],
            config["h2_correction_amendment_pin"]]
        queue_spec = spec_builder.build_queue_spec(
            unit_id=uid, kind="pstar", python=python,
            spec_path=str(spec_path), spec_sha=row["spec_sha256"],
            lock_path=str(lock_path), lock_sha=stage_lock_sha256,
            public_receipt_path=spec["public_receipt_path"],
            cwd=str(Path(__file__).parents[3]),
            max_wall_s=h1["max_wall_s"],
            cpu_reserve_s=h1["cpu_reserve_s"],
            max_rss_gib=h1["max_rss_gib"],
            projected_output_bytes=h1["projected_output_bytes"],
            source_pins=sources)
        queue_spec["argv"][2] = reference_worker_h2.__name__
        spec_sha = spec_builder._write_once(private / "queue_units" / (uid + ".json"),
                                             queue_spec, 0o600)
        entries.append({"unit_id": uid, "attack_spec_sha256": row["spec_sha256"],
                        "queue_spec_sha256": spec_sha})
    queue_index = {"schema": QUEUE_SCHEMA,
                   "status": "READY_TO_ENQUEUE_NO_ATTACK_FIT",
                   "attack_unit_index_sha256": spec_builder._sha(index_path),
                   "stage_lock_sha256": stage_lock_sha256,
                   "amendment_sha256": stage_lock_sha256,
                   "h2_source_plan_sha256": config["h2_source_plan_pin"]["sha256"],
                   "unit_count": 2, "units": entries}
    digest = spec_builder._write_once(public / "PSTAR_H2_QUEUE_INDEX.json",
                                       queue_index, 0o644)
    return {"status": queue_index["status"], "queue_index_sha256": digest,
            "unit_count": 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("prepare")
    first.add_argument("--config", required=True)
    second = sub.add_parser("finalize")
    second.add_argument("--config", required=True)
    second.add_argument("--stage-lock", required=True)
    second.add_argument("--stage-lock-sha256", required=True)
    args = parser.parse_args()
    result = prepare(args.config) if args.command == "prepare" else finalize(
        args.config, args.stage_lock, args.stage_lock_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
