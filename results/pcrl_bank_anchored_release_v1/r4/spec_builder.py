"""Prefit R4 attack spec builder; receipts only, no person arrays or labels.

`prepare` writes private immutable worker specs and a public hash-only index.
R0 then seals the index's `stage_lock_units` in a dated attacks amendment.
`finalize` checks that registered stage lock and writes queue specs. Neither
operation launches a fit or imports a protected label.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Mapping

from . import policy_worker, reference_worker, witness_worker


SCHEMA = "pcrl-bank-r4-spec-build-config-v1"
INDEX_SCHEMA = "pcrl-bank-r4-attack-unit-index-v1"
QUEUE_INDEX_SCHEMA = "pcrl-bank-r4-queue-unit-index-v1"
ARMS = ("B0", "T", "D", "P", "L", "J", "S12", "S21", "W_Q", "P_STAR")
CHANNEL_ARMS = ("T", "D", "P", "L", "J", "S12", "S21")
ROLES = ("ATTACK_FIT_NEW", "INNER_SELECTION")
CONFIG_KEYS = frozenset({
    "schema", "campaign_seal_path", "campaign_seal_sha256",
    "role_manifest_path", "role_manifest_sha256",
    "table_by_seed", "gate_by_arm", "pstar_gate",
    "pstar_historical_gate", "queue_root",
    "selected_r2_root",
    "witness_source_root", "pstar_source_root",
    "private_root", "public_root", "python_executable",
    "max_wall_s", "cpu_reserve_s", "max_rss_gib",
    "projected_output_bytes",
})
PIN_KEYS = frozenset({"path", "sha256"})


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pin(path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise PermissionError("source pin cannot be a symlink")
    p = path.resolve(strict=True)
    if not p.is_file():
        raise PermissionError("source pin must be a regular file")
    return {"path": str(p), "sha256": _sha(p)}


def _read_pin(item: Mapping, *, private: bool = False) -> dict:
    if not isinstance(item, Mapping) or set(item) != PIN_KEYS:
        raise ValueError("source path/SHA pin differs")
    p = Path(item["path"])
    if (p.is_symlink() or len(item["sha256"]) != 64 or
            p.resolve(strict=True) != p or _sha(p) != item["sha256"]):
        raise PermissionError("source bytes/path differ from supplied pin")
    if private and (".private" not in p.parts or p.stat().st_mode & 0o077):
        raise PermissionError("private receipt custody differs")
    return json.loads(p.read_text())


def _write_once(path: Path, payload: Mapping, mode: int) -> str:
    value = (json.dumps(payload, sort_keys=True, indent=2,
                        allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != value:
            raise FileExistsError("immutable attack spec/index differs")
        return _sha(path)
    if path.parent.is_symlink():
        raise PermissionError("attack spec parent cannot be a symlink")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(value).hexdigest()


def _modules(kind: str) -> dict[str, str]:
    common = {
        "role_store.py": "results.pcrl_latent_and_sampled_methods_v1.r01.role_store",
        "protocol.py": "results.pcrl_latent_and_sampled_methods_v1.r02.protocol",
        "complete_view.py": "results.pcrl_latent_and_sampled_methods_v1.r02.complete_view",
        "attack_eval.py": "attack_eval",
        "attack_controls_v2.py": "attack_controls_v2",
        "common_eval.py": "common_eval",
        "safe_data_guard.py": "safe_data_guard",
    }
    if kind == "policy":
        return {**common,
                "policy_worker.py": policy_worker.__name__,
                "views.py": "results.pcrl_bank_anchored_release_v1.r4.views",
                "reference_worker.py": reference_worker.__name__,
                "r2_package.py": "results.pcrl_bank_anchored_release_v1.bank_anchored.r2_package",
                "r2_materialize.py": "results.pcrl_bank_anchored_release_v1.bank_anchored.r2_materialize",
                "r2_laws.py": "results.pcrl_bank_anchored_release_v1.bank_anchored.r2_laws",
                "r1_release.py": "bank_anchored.r1_release"}
    if kind == "witness":
        return {**common,
                "witness_worker.py": witness_worker.__name__,
                "views.py": "results.pcrl_bank_anchored_release_v1.r4.views",
                "reference_worker.py": reference_worker.__name__,
                "bank_service.py": "results.pcrl_code_utility_witness_v1.r4.bank_service"}
    if kind == "pstar":
        return {**common,
                "reference_worker.py": reference_worker.__name__,
                "reference_wire.py":
                "results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.reference_wire"}
    raise ValueError("unknown R4 attack worker kind")


def source_pins(kind: str) -> list[dict[str, str]]:
    names = _modules(kind)
    required = {"policy": policy_worker.SOURCES,
                "witness": witness_worker.SOURCES,
                "pstar": reference_worker.REQUIRED_SOURCES}[kind]
    if set(names) != required:
        raise AssertionError("builder source list differs from worker allowlist")
    return [_pin(Path(importlib.import_module(names[name]).__file__))
            for name in sorted(names)]


def planned_units(eligible_arms: set[str]) -> list[tuple[str, int, int]]:
    if not eligible_arms <= set(ARMS) or "B0" not in eligible_arms:
        raise ValueError("B0 and only registered attack arms are required")
    return [(arm, seed, realization)
            for arm in ARMS if arm in eligible_arms
            for seed in (0, 1)
            for realization in (range(3) if arm in CHANNEL_ARMS and arm != "D"
                                else range(1))]


def _unit_id(arm: str, seed: int, realization: int) -> str:
    return f"R4_ATTACK_{arm}_S{seed}_R{realization}"


def _validate_config(config: Mapping) -> tuple[dict, dict]:
    if not isinstance(config, Mapping) or set(config) != CONFIG_KEYS or config["schema"] != SCHEMA:
        raise ValueError("attack builder config schema differs")
    if set(config["table_by_seed"]) != {"0", "1"} or \
            set(config["gate_by_arm"]) != set(ARMS) - {"P_STAR"}:
        raise ValueError("all two-seed tables and non-P* arm gates required")
    if (type(config["max_wall_s"]) not in (int, float) or
            type(config["cpu_reserve_s"]) not in (int, float) or
            not 0 < config["max_wall_s"] <= config["cpu_reserve_s"] or
            config["max_rss_gib"] <= 0 or config["max_rss_gib"] > 8 or
            type(config["projected_output_bytes"]) is not int or
            config["projected_output_bytes"] < 0):
        raise ValueError("attack queue resource bounds differ")
    seal = _read_pin({"path": config["campaign_seal_path"],
                      "sha256": config["campaign_seal_sha256"]})
    if seal.get("schema") != "pcrl-bank-anchored-campaign-lock-v1" or \
            set(seal.get("arms", [])) != set(ARMS):
        raise PermissionError("sealed registered arm family differs")
    role = _read_pin({"path": config["role_manifest_path"],
                      "sha256": config["role_manifest_sha256"]})
    if not all(role.get("roles", {}).get(key, {}).get("rows") == rows
               for key, rows in (("ATTACK_FIT_NEW", 4064),
                                 ("INNER_SELECTION", 2235))):
        raise PermissionError("attack role manifest row counts differ")
    for seed in (0, 1):
        table = _read_pin(config["table_by_seed"][str(seed)])
        if table.get("schema") != "code-utility-public-bank-tables-v1" or \
                table.get("seed") != seed or table.get("bit_count") != 10:
            raise PermissionError("public bank source/seed differs")
    return seal, role


def eligible_arms(config: Mapping) -> set[str]:
    out = set()
    for arm in ARMS:
        if arm == "P_STAR":
            historical = _read_pin(config["pstar_historical_gate"])
            if (historical.get("schema") !=
                    "pcrl-bank-r3-historical-original-inner-eligibility-v1" or
                    historical.get("arm_id") != "P_STAR" or
                    historical.get("all_two_seed_task_reference_point_caps_pass") is not True or
                    historical.get("original_inner_point_clauses_passed") != 24 or
                    historical.get("original_inner_point_clauses_total") != 24 or
                    historical.get("fit_bank_proxy_applied") is not False or
                    historical.get("fit_bank_proxy_pass") is not None):
                raise PermissionError("separate P* historical gate differs")
            gate = _read_pin(config["pstar_gate"])
            if (gate.get("schema") != "pcrl-bank-r5-pstar-common-inner-replay-v1" or
                    gate.get("status") != "PASS_UTILITY_POINT_ONLY" or
                    gate.get("all_24_point_clauses_pass") is not True):
                raise PermissionError("P* reference point gate differs")
            out.add(arm)
            continue
        gate = _read_pin(config["gate_by_arm"][arm])
        if gate.get("schema") == "pcrl-bank-r4-incomplete-arm-gate-v1":
            _validate_incomplete_gate(config, arm, gate)
            continue
        if gate.get("arm_id") != arm or \
                gate.get("all_two_seed_task_reference_point_caps_pass") not in (True, False):
            raise PermissionError("R3/old reference arm gate identity differs")
        if arm == "W_Q":
            if (gate.get("schema") != "pcrl-bank-r3-historical-original-inner-eligibility-v1" or
                    gate.get("original_inner_point_clauses_passed") != 24 or
                    gate.get("original_inner_point_clauses_total") != 24 or
                    gate.get("fit_bank_proxy_applied") is not False or
                    gate.get("fit_bank_proxy_pass") is not None):
                raise PermissionError("separate W_Q historical gate differs")
        elif (gate.get("schema") != "pcrl-bank-r3-arm-point-eligibility-v1" or
              gate.get("source_seal_sha256") != config["campaign_seal_sha256"] or
              gate.get("fit_bank_proxy_pass") not in (True, False) or
              gate.get("inner_original_caps_pass") not in (True, False) or
              gate["all_two_seed_task_reference_point_caps_pass"] !=
              (gate["fit_bank_proxy_pass"] and gate["inner_original_caps_pass"])):
            raise PermissionError("two-seed original common gate differs")
        if gate["all_two_seed_task_reference_point_caps_pass"]:
            out.add(arm)
    if "B0" not in out:
        raise PermissionError("selected bank B0 gate must pass before attacks")
    return out


def _validate_incomplete_gate(config: Mapping, arm: str,
                              gate: Mapping) -> None:
    if arm not in CHANNEL_ARMS:
        raise PermissionError("reference requires its own completed point gate")
    if (set(gate) != {"schema", "arm_id", "source_seal_sha256", "status",
                      "all_two_seed_task_reference_point_caps_pass", "reason",
                      "source_receipts_by_seed", "missing_r3_unit_ids"} or
            gate["arm_id"] != arm or
            gate["source_seal_sha256"] != config["campaign_seal_sha256"] or
            gate["all_two_seed_task_reference_point_caps_pass"] is not False or
            gate["status"] not in ("INCOMPLETE_SEARCH", "MISSING_UTILITY") or
            type(gate["reason"]) is not str or not gate["reason"] or
            type(gate["source_receipts_by_seed"]) is not dict or
            set(gate["source_receipts_by_seed"]) != {"0", "1"} or
            type(gate["missing_r3_unit_ids"]) is not list or
            any(type(x) is not str or not x for x in gate["missing_r3_unit_ids"])):
        raise PermissionError("incomplete search marker structure differs")
    nonempty = 0
    incomplete = 0
    for seed in (0, 1):
        pair = gate["source_receipts_by_seed"][str(seed)]
        if pair is None:
            continue
        if type(pair) is not dict or set(pair) != {"unit", "fit"}:
            raise PermissionError("incomplete marker R2 pin pair differs")
        unit = _read_pin(pair["unit"])
        fit = _read_pin(pair["fit"])
        if (unit.get("schema") != "BAR1-r2-unit-receipt-v1" or
                fit.get("schema") != "BAR1-r2-fit-receipt-v1" or
                fit.get("arm_id") != arm or fit.get("source_seed") != seed or
                fit.get("source_lock_sha256") != config["campaign_seal_sha256"] or
                unit.get("unit_id") != fit.get("unit_id") or
                (unit.get("arm_id") not in (None, arm)) or
                (unit.get("source_seed") not in (None, seed)) or
                unit.get("source_lock_sha256") not in
                (None, config["campaign_seal_sha256"]) or
                unit.get("fit_receipt_sha256") != pair["fit"]["sha256"]):
            raise PermissionError("incomplete marker R2 unit/FIT source differs")
        nonempty += 1
        incomplete += unit.get("status") == "INCOMPLETE_SEARCH"
    if nonempty == 0 or (gate["status"] == "INCOMPLETE_SEARCH" and incomplete == 0):
        raise PermissionError("incomplete marker has no pinned incomplete R2 attempt")
    if gate["status"] == "MISSING_UTILITY":
        if not gate["missing_r3_unit_ids"]:
            raise PermissionError("missing utility marker names no missing R3 unit")
        if any(not any(name.startswith(f"R3_{arm}_S{seed}_")
                       for seed in (0, 1)) or
               not ("_FIT" in name or "_INNER" in name)
               for name in gate["missing_r3_unit_ids"]):
            raise PermissionError("missing R3 unit does not belong to this arm")
        r3_dir = Path(config["gate_by_arm"][arm]["path"]).parent
        if any((r3_dir / f"{name}_RECEIPT.json").exists()
               for name in gate["missing_r3_unit_ids"]):
            raise PermissionError("claimed missing R3 receipt already exists")


def _common(config: Mapping, arm: str, seed: int, realization: int,
            kind: str) -> dict:
    unit_id = _unit_id(arm, seed, realization)
    private = Path(config["private_root"])
    public = Path(config["public_root"])
    return {
        "unit_id": unit_id, "arm_id": arm, "source_seed": seed,
        "realization": realization, "attacker_seed": 100000 + 1000 * seed + realization,
        "campaign_seal_path": config["campaign_seal_path"],
        "campaign_seal_sha256": config["campaign_seal_sha256"],
        "role_manifest_path": config["role_manifest_path"],
        "role_manifest_sha256": config["role_manifest_sha256"],
        "source_pins": source_pins(kind),
        "private_output_path": str(private / "attacks" / unit_id),
        "public_receipt_path": str(public / f"{unit_id}_RECEIPT.json"),
    }


def _selected_spec(config: Mapping, arm: str, seed: int,
                   realization: int, role: Mapping) -> dict:
    r2_root = Path(config["selected_r2_root"])
    candidates = []
    for path in r2_root.glob("*_UNIT_RECEIPT.json"):
        value = _read_pin(_pin(path))
        if value.get("schema") == "BAR1-r2-unit-receipt-v1" and \
                value.get("arm_id") == arm and value.get("source_seed") == seed:
            candidates.append((path, value))
    if len(candidates) != 1:
        raise PermissionError("exactly one completed R2 selected source unit required")
    unit_path, source_unit = candidates[0]
    if (source_unit.get("status") == "INCOMPLETE_SEARCH" or
            source_unit.get("source_lock_sha256") != config["campaign_seal_sha256"]):
        raise PermissionError("R2 source unit incomplete or from another campaign")
    source_unit_pin = _pin(unit_path)
    unit_root = r2_root / ".private" / "units" / source_unit["unit_id"]
    policy_dir = unit_root / "policy"
    policy_pin = _pin(policy_dir / "POLICY_RECEIPT.json")
    receipt = _read_pin(policy_pin, private=True)
    if (receipt.get("schema") != "BAR1-selected-policy-v1" or
            receipt.get("arm_id") != arm or receipt.get("source_seed") != seed or
            _sha(policy_dir / "policy.npz") != receipt.get("policy_file_sha256")):
        raise PermissionError("selected policy receipt/package identity differs")
    if (source_unit.get("policy_receipt_sha256") != policy_pin["sha256"] or
            source_unit.get("fit_receipt_sha256") != receipt.get("fit_receipt_sha256") or
            source_unit.get("source_binding_sha256") != receipt.get("source_binding_sha256")):
        raise PermissionError("R2 selected policy differs from its public source unit")
    role_index = {(item["role"], item["realization"]): item
                  for item in source_unit["role_releases"]}
    if len(role_index) != len(source_unit["role_releases"]):
        raise PermissionError("R2 unit repeated a role/realization materialization")
    release_dirs, release_hashes, role_receipts = {}, {}, {}
    for role_name in ROLES:
        entry = role_index[(role_name, realization)]
        directory = unit_root / entry["private_relative_dir"]
        pin = _pin(directory / "RELEASE_RECEIPT.json")
        release = _read_pin(pin, private=True)
        if (release.get("schema") != "BAR1-role-release-v1" or
                release.get("arm_id") != arm or
                release.get("source_seed") != seed or
                release.get("role") != role_name or
                release.get("realization") != realization or
                release.get("policy_receipt_sha256") != policy_pin["sha256"] or
                release.get("source_binding_sha256") !=
                receipt.get("source_binding_sha256") or
                release.get("law_sha256") != entry.get("law_sha256") or
                pin["sha256"] != entry.get("receipt_sha256") or
                release.get("rows") != role["roles"][role_name]["rows"]):
            raise PermissionError("selected cached role release differs")
        for filename, field in (("law.npz", "law_sha256"),
                                ("release_codes.npz", "release_codes_sha256")):
            if _sha(directory / filename) != release.get(field):
                raise PermissionError("selected role release bytes differ")
        release_dirs[role_name] = str(directory.resolve(strict=True))
        release_hashes[role_name] = pin["sha256"]
        role_receipts[role_name] = release
    fit_entry = role_index[("OSF_DEFENSE_FIT", 0)]
    fit_directory = unit_root / fit_entry["private_relative_dir"]
    fit_pin = _pin(fit_directory / "RELEASE_RECEIPT.json")
    if fit_pin["sha256"] != fit_entry["receipt_sha256"]:
        raise PermissionError("R2 source FIT release receipt differs")
    gate = _read_pin(config["gate_by_arm"][arm])
    links = {}
    for utility_role, short in (("OSF_DEFENSE_FIT", "fit"),
                                ("INNER_SELECTION", "inner")):
        source = gate["source_receipts"][f"s{seed}_{short}"]
        name = source["unit_id"]
        r3_root = Path(config["gate_by_arm"][arm]["path"]).parent
        receipt_path = r3_root / f"{name}_RECEIPT.json"
        csv_path = r3_root / f"{name}.csv"
        if (_sha(receipt_path) != source["receipt_sha256"] or
                _sha(csv_path) != source["csv_sha256"]):
            raise PermissionError("R3 gate source receipt/CSV bytes differ")
        links[utility_role] = {
            "receipt_path": str(receipt_path.resolve(strict=True)),
            "receipt_sha256": source["receipt_sha256"],
            "csv_path": str(csv_path.resolve(strict=True)),
            "csv_sha256": source["csv_sha256"]}
    spec = {
        **_common(config, arm, seed, realization, "policy"),
        "schema": policy_worker.SCHEMA,
        "public_table_path": config["table_by_seed"][str(seed)]["path"],
        "public_table_sha256": config["table_by_seed"][str(seed)]["sha256"],
        "policy_directory": str(policy_dir.resolve(strict=True)),
        "policy_receipt_sha256": policy_pin["sha256"],
        "release_directory_by_role": release_dirs,
        "release_receipt_sha256_by_role": release_hashes,
        "source_unit_receipt_path": source_unit_pin["path"],
        "source_unit_receipt_sha256": source_unit_pin["sha256"],
        "fit_release_directory": str(fit_directory.resolve(strict=True)),
        "fit_release_receipt_sha256": fit_pin["sha256"],
        "utility_receipts_by_role": links,
        "eligibility_receipt_path": config["gate_by_arm"][arm]["path"],
        "eligibility_receipt_sha256": config["gate_by_arm"][arm]["sha256"],
    }
    policy_worker.validate_spec(spec)
    policy_worker.validate_utility_linkage(spec, gate, receipt, role_receipts)
    return spec


def _witness_spec(config: Mapping, arm: str, seed: int) -> dict:
    family = "W_TSTAR" if arm == "B0" else "W_Q"
    root = Path(config["witness_source_root"])
    receipt_pin = _pin(root / "r1" / f"W_BANK_S{seed}_R1_RECEIPT.json")
    source = _read_pin(receipt_pin)
    table_pin = config["table_by_seed"][str(seed)]
    table = _read_pin(table_pin)
    family_doc = table["families"][family]
    if (source.get("schema") != "code-utility-witness-worker-v1" or
            source.get("status") != "COMPLETE" or source.get("stage") != "bank" or
            source.get("seed") != seed or
            source.get("role_manifest_sha256") != config["role_manifest_sha256"] or
            source.get("source_model_sha256") != table["source_model_sha256"] or
            family not in source.get("result", {}).get("arm_ids", [])):
        raise PermissionError("inherited bank witness receipt differs")
    releases, hashes = {}, {}
    for role_name in ROLES:
        item = source["result"]["roles"][role_name]["arms"][family]
        release = root / "r1" / ".private" / f"W_BANK_S{seed}_R1" / \
            item["private_release_basename"]
        if (release.stat().st_mode & 0o077 or
                _sha(release) != item["private_release_sha256"] or
                item["policy_sha256"] != family_doc["policy_sha256"] or
                item["decoder_sha256"] != family_doc["decoder_sha256"] or
                item["public_table_sha256"] != table_pin["sha256"]):
            raise PermissionError("inherited role wire/source table differs")
        releases[role_name] = str(release.resolve(strict=True))
        hashes[role_name] = item["private_release_sha256"]
    spec = {
        **_common(config, arm, seed, 0, "witness"),
        "schema": witness_worker.SCHEMA,
        "public_table_path": table_pin["path"],
        "public_table_sha256": table_pin["sha256"],
        "source_receipt_path": receipt_pin["path"],
        "source_receipt_sha256": receipt_pin["sha256"],
        "source_model_sha256": table["source_model_sha256"],
        "feature_schema_sha256": table["feature_schema_sha256"],
        "policy_sha256": family_doc["policy_sha256"],
        "decoder_sha256": family_doc["decoder_sha256"],
        "release_path_by_role": releases,
        "release_sha256_by_role": hashes,
        "eligibility_receipt_path": config["gate_by_arm"][arm]["path"],
        "eligibility_receipt_sha256": config["gate_by_arm"][arm]["sha256"],
    }
    witness_worker.validate_spec(spec)
    return spec


def _pstar_spec(config: Mapping, seed: int) -> dict:
    root = Path(config["pstar_source_root"])
    map_dir = root / "r01"
    map_pin = _pin(map_dir / f"A_FIXED_MAP_P_STAR_S{seed}.json")
    archive_pin = _pin(map_dir / f"A_FIXED_MAP_P_STAR_S{seed}.npz")
    map_receipt = _read_pin(map_pin)
    if (map_receipt.get("arm_id") != "P_STAR" or
            map_receipt.get("seed") != seed or
            map_receipt.get("npz_sha256") != archive_pin["sha256"]):
        raise PermissionError("P* fixed map receipt/archive differs")
    directories, hashes = {}, {}
    for role, short in (("ATTACK_FIT_NEW", "ATT"),
                        ("INNER_SELECTION", "INNER")):
        directory = (root / "r03" / ".private" / "a_fixed_units" /
                     f"A-REFERENCE-RELEASE-P_STAR-S{seed}-{short}")
        pin = _pin(directory / "REFERENCE_RELEASE_RECEIPT.json")
        receipt = _read_pin(pin, private=True)
        if (receipt.get("arm_id") != "P_STAR" or
                receipt.get("source_seed") != seed or
                receipt.get("role") != role or
                receipt.get("source_binding_sha256") !=
                map_receipt.get("source_binding_sha256") or
                _sha(directory / "reference_release.npz") !=
                receipt.get("codes_file_sha256")):
            raise PermissionError("P* reference role package differs")
        directories[role] = str(directory.resolve(strict=True))
        hashes[role] = pin["sha256"]
    spec = {
        **_common(config, "P_STAR", seed, 0, "pstar"),
        "schema": reference_worker.SPEC_SCHEMA,
        "fixed_map_receipt_path": map_pin["path"],
        "fixed_map_receipt_sha256": map_pin["sha256"],
        "fixed_map_archive_path": archive_pin["path"],
        "fixed_map_archive_sha256": archive_pin["sha256"],
        "att_package_dir": directories["ATTACK_FIT_NEW"],
        "att_receipt_sha256": hashes["ATTACK_FIT_NEW"],
        "inner_package_dir": directories["INNER_SELECTION"],
        "inner_receipt_sha256": hashes["INNER_SELECTION"],
        "eligibility_receipt_path": config["pstar_gate"]["path"],
        "eligibility_receipt_sha256": config["pstar_gate"]["sha256"],
    }
    reference_worker.validate_spec(spec)
    return spec


def _private_directory(path: Path) -> None:
    if ".private" not in path.parts or path.is_symlink():
        raise PermissionError("attack builder private root differs")
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.stat().st_mode & 0o077:
        raise PermissionError("attack builder private directory is not owner-only")


def prepare(config_path: str) -> dict:
    """Write all eligible immutable R4 worker specs, then public hash index."""
    config_file = Path(config_path).resolve(strict=True)
    if (".private" not in config_file.parts or
            config_file.stat().st_mode & 0o077):
        raise PermissionError("attack builder input must be private")
    config = json.loads(config_file.read_text())
    _, role = _validate_config(config)
    eligible = eligible_arms(config)
    units = planned_units(eligible)
    private = Path(config["private_root"])
    public = Path(config["public_root"])
    _private_directory(private)
    _private_directory(private / "units")
    _private_directory(private / "attacks")
    _private_directory(private / "queue_units")
    if not public.is_dir() or public.is_symlink() or ".private" in public.parts:
        raise PermissionError("attack public receipt root differs")
    lock_units, entries = {}, []
    for arm, seed, realization in units:
        if arm in CHANNEL_ARMS:
            spec, kind = _selected_spec(config, arm, seed, realization, role), "policy"
        elif arm in ("B0", "W_Q"):
            spec, kind = _witness_spec(config, arm, seed), "witness"
        else:
            spec, kind = _pstar_spec(config, seed), "pstar"
        uid = spec["unit_id"]
        spec_path = private / "units" / f"{uid}.json"
        spec_sha = _write_once(spec_path, spec, 0o600)
        lock_units[uid] = {"spec_sha256": spec_sha, "arm_id": arm,
                           "source_seed": seed, "realization": realization,
                           "point_eligible": True}
        entries.append({"unit_id": uid, "arm_id": arm,
                        "source_seed": seed, "realization": realization,
                        "worker_kind": kind, "spec_sha256": spec_sha,
                        "planned_fits": 30})
    index = {"schema": INDEX_SCHEMA,
             "campaign_seal_sha256": config["campaign_seal_sha256"],
             "builder_sha256": _sha(Path(__file__)),
             "eligible_arms": [arm for arm in ARMS if arm in eligible],
             "ineligible_arms": [arm for arm in ARMS if arm not in eligible],
             "gate_sha256_by_arm": {
                 **{arm: config["gate_by_arm"][arm]["sha256"]
                    for arm in ARMS if arm != "P_STAR"},
                 "P_STAR": config["pstar_historical_gate"]["sha256"]},
             "pstar_independent_replay_sha256": config["pstar_gate"]["sha256"],
             "unit_count": len(entries), "planned_fits": 30 * len(entries),
             "units": entries, "stage_lock_units": lock_units}
    index_sha = _write_once(public / "ATTACK_UNIT_INDEX.json", index, 0o644)
    return {"status": "SPECS_READY_NO_ATTACK_FIT", "index_sha256": index_sha,
            "unit_count": len(entries), "planned_fits": 30 * len(entries),
            "eligible_arms": index["eligible_arms"]}


def _queue_sources(spec: Mapping, spec_path: Path,
                   lock_path: Path) -> list[dict[str, str]]:
    paths = [spec_path, lock_path, Path(spec["campaign_seal_path"]),
             Path(spec["role_manifest_path"]),
             Path(spec["eligibility_receipt_path"])]
    paths.extend(Path(pin["path"]) for pin in spec["source_pins"])
    if spec["arm_id"] in CHANNEL_ARMS:
        policy = Path(spec["policy_directory"])
        paths.extend((Path(spec["public_table_path"]),
                      Path(spec["source_unit_receipt_path"]),
                      policy / "POLICY_RECEIPT.json", policy / "policy.npz"))
        fit_root = Path(spec["fit_release_directory"])
        paths.extend((fit_root / "RELEASE_RECEIPT.json", fit_root / "law.npz"))
        for link in spec["utility_receipts_by_role"].values():
            paths.extend((Path(link["receipt_path"]), Path(link["csv_path"])))
        for directory in spec["release_directory_by_role"].values():
            root = Path(directory)
            paths.extend((root / "RELEASE_RECEIPT.json", root / "law.npz",
                          root / "release_codes.npz"))
    elif spec["arm_id"] in ("B0", "W_Q"):
        paths.extend((Path(spec["public_table_path"]),
                      Path(spec["source_receipt_path"])))
        paths.extend(Path(value) for value in spec["release_path_by_role"].values())
    else:
        paths.extend((Path(spec["fixed_map_receipt_path"]),
                      Path(spec["fixed_map_archive_path"])))
        for key in ("att_package_dir", "inner_package_dir"):
            root = Path(spec[key])
            paths.extend((root / "REFERENCE_RELEASE_RECEIPT.json",
                          root / "reference_release.npz"))
    return [_pin(path) for path in sorted(set(paths))]


def finalize(config_path: str, stage_lock_path: str,
             stage_lock_sha256: str) -> dict:
    """After R0 registers its exact lock, write queue specs but never enqueue."""
    config_file = Path(config_path).resolve(strict=True)
    if ".private" not in config_file.parts or config_file.stat().st_mode & 0o077:
        raise PermissionError("attack builder input must be private")
    config = json.loads(config_file.read_text())
    _validate_config(config)
    public = Path(config["public_root"])
    index = json.loads((public / "ATTACK_UNIT_INDEX.json").read_text())
    if (index.get("schema") != INDEX_SCHEMA or
            index.get("builder_sha256") != _sha(Path(__file__)) or
            index.get("campaign_seal_sha256") != config["campaign_seal_sha256"]):
        raise PermissionError("attack spec index differs from active builder/source")
    lock_path = Path(stage_lock_path).resolve(strict=True)
    lock = _read_pin({"path": str(lock_path), "sha256": stage_lock_sha256})
    if (lock.get("schema") != "pcrl-bank-attack-stage-lock-v1" or
            lock.get("status") != "LOCALLY_SEALED_ATTACKS" or
            lock.get("campaign_seal_sha256") != config["campaign_seal_sha256"] or
            lock.get("units") != index["stage_lock_units"]):
        raise PermissionError("R0 attack-stage lock omits or alters a planned unit")
    queue_root = Path(config["queue_root"]).resolve(strict=True)
    state = json.loads((queue_root / "state.json").read_text())
    if not any(a["sha256"] == stage_lock_sha256 and "attacks" in a["stages"]
               for a in state.get("amendments", [])):
        raise PermissionError("R0 did not register attack-stage lock in global queue")
    python = Path(config["python_executable"]).resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise PermissionError("numerical Python executable missing")
    private = Path(config["private_root"])
    queue_entries = []
    modules = {"policy": "results.pcrl_bank_anchored_release_v1.r4.policy_worker",
               "witness": "results.pcrl_bank_anchored_release_v1.r4.witness_worker",
               "pstar": "results.pcrl_bank_anchored_release_v1.r4.reference_worker"}
    for entry in index["units"]:
        uid = entry["unit_id"]
        spec_path = private / "units" / f"{uid}.json"
        if _sha(spec_path) != entry["spec_sha256"]:
            raise PermissionError("private attack unit spec differs from sealed index")
        spec = json.loads(spec_path.read_text())
        kind = entry["worker_kind"]
        queue_spec = {
            "unit_id": uid, "stage": "attacks",
            "argv": [str(python), "-m", modules[kind],
                     "--unit-spec", str(spec_path),
                     "--unit-spec-sha256", entry["spec_sha256"],
                     "--stage-lock", str(lock_path),
                     "--stage-lock-sha256", stage_lock_sha256],
            "cwd": str(Path(__file__).parents[3]),
            "max_wall_s": config["max_wall_s"],
            "cpu_reserve_s": config["cpu_reserve_s"],
            "max_rss_gib": config["max_rss_gib"],
            "projected_output_bytes": config["projected_output_bytes"],
            "deps": [], "env": {},
            "outputs": [spec["public_receipt_path"]],
            "sources": _queue_sources(spec, spec_path, lock_path),
        }
        queue_path = private / "queue_units" / f"{uid}.json"
        queue_sha = _write_once(queue_path, queue_spec, 0o600)
        queue_entries.append({"unit_id": uid, "queue_spec_sha256": queue_sha,
                              "attack_spec_sha256": entry["spec_sha256"]})
    queue_index = {"schema": QUEUE_INDEX_SCHEMA,
                   "attack_unit_index_sha256": _sha(public / "ATTACK_UNIT_INDEX.json"),
                   "stage_lock_sha256": stage_lock_sha256,
                   "unit_count": len(queue_entries), "units": queue_entries,
                   "status": "READY_TO_ENQUEUE_NO_ATTACK_FIT"}
    queue_index_sha = _write_once(public / "ATTACK_QUEUE_INDEX.json", queue_index, 0o644)
    return {"status": queue_index["status"],
            "queue_index_sha256": queue_index_sha,
            "unit_count": len(queue_entries)}


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
    out = (prepare(args.config) if args.command == "prepare" else
           finalize(args.config, args.stage_lock, args.stage_lock_sha256))
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
