"""Queue-gated, label-free historical score release on ATT or INNER.

The R03 unit and effective A amendment are checked before any person archive.
All predecessor role reads go through R01's role-only source stores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import (
    RoleOnlyStore, assert_same_people, sha_array, sha_file,
)
from results.pcrl_latent_and_sampled_methods_v1.r01.tstar_source import TStarSourceStore
from results.pcrl_latent_and_sampled_methods_v1.r01.pstar_source import PStarSourceStore
from results.pcrl_latent_and_sampled_methods_v1.r01.d0_source import D0SourceStore
from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import assert_active_from_env

from .admitted_fit import REQUIRED as A_REQUIRED, _sha
from .reference_materialize import materialize_reference_role


REFERENCES = frozenset({"U0", "Ucal", "T_STAR", "P_STAR", "D0_LOCAL",
                        "D0_SEQ12", "D0_SEQ21"})
D0_KEYS = {"D0_LOCAL": "hcal_local_d0", "D0_SEQ12": "hcal_seq12_d0",
           "D0_SEQ21": "lra_seq21"}
SOURCE_FILES = frozenset({
    "r01/role_store.py", "r01/tstar_source.py", "r01/pstar_source.py",
    "r01/d0_source.py", "r03/compute_queue.py",
    "sampled_channel_v1/impl/admitted_fit.py",
    "sampled_channel_v1/impl/reference_materialize.py",
    "sampled_channel_v1/impl/run_reference.py",
})
SOURCE_AUDITS = {
    "T_STAR": ("r04/A_PREFIT_INDEPENDENT_REPLAY.json", "PREFIT_SOURCE_DECODER_IDENTITY_PASS"),
    "P_STAR": ("r04/PSTAR_SOURCE_INDEPENDENT_REPLAY.json", "PSTAR_ATT_INNER_SOURCE_REPLAY_PASS"),
    "D0_LOCAL": ("r04/D0_UTILITY_INDEPENDENT_REPLAY.json", "SIX_D0_INNER_UTILITY_REPLAY_PASS"),
    "D0_SEQ12": ("r04/D0_UTILITY_INDEPENDENT_REPLAY.json", "SIX_D0_INNER_UTILITY_REPLAY_PASS"),
    "D0_SEQ21": ("r04/D0_UTILITY_INDEPENDENT_REPLAY.json", "SIX_D0_INNER_UTILITY_REPLAY_PASS"),
}


def _listed_source(spec: dict, path: Path, expected: str) -> None:
    rows = spec.get("sources", [])
    if (type(expected) is not str or len(expected) != 64 or
            not all(c in "0123456789abcdef" for c in expected) or
            {Path(item["path"]).resolve(): item["sha256"] for item in rows}
            .get(path.resolve()) != expected or sha_file(path) != expected):
        raise PermissionError("reference source is not pinned by R03 unit")


def _private_output(path) -> Path:
    supplied = Path(path)
    result = supplied.resolve(strict=False)
    if (any(part.is_symlink() for part in (supplied, *supplied.parents)) or
            ".private" not in result.parts or result.exists() or
            not result.parent.is_dir() or result.parent.is_symlink() or
            result.parent.stat().st_uid != os.getuid() or
            stat.S_IMODE(result.parent.stat().st_mode) & 0o077):
        raise PermissionError("reference output must be a fresh private directory")
    return result


def _runtime_origin_guard(campaign: Path) -> None:
    names = {
        "r01.role_store": "r01/role_store.py",
        "r01.tstar_source": "r01/tstar_source.py",
        "r01.pstar_source": "r01/pstar_source.py",
        "r01.d0_source": "r01/d0_source.py",
        "r03.compute_queue": "r03/compute_queue.py",
        "sampled_channel_v1.impl.admitted_fit": "sampled_channel_v1/impl/admitted_fit.py",
        "sampled_channel_v1.impl.reference_materialize":
            "sampled_channel_v1/impl/reference_materialize.py",
        "sampled_channel_v1.impl.run_reference":
            "sampled_channel_v1/impl/run_reference.py",
    }
    prefix = "results.pcrl_latent_and_sampled_methods_v1."
    for suffix, relative in names.items():
        module = sys.modules.get(prefix + suffix)
        if module is None and suffix.endswith(".run_reference"):
            module = sys.modules.get("__main__")
        if (module is None or Path(getattr(module, "__file__", "")).resolve() !=
                (campaign / relative).resolve(strict=True)):
            raise PermissionError("reference imported implementation differs from pinned source")


def admit_reference_unit(*, unit_id: str, lock_path, amendment_path,
                         amendment_sha256: str, output_dir,
                         reference_id: str, seed: int, role: str) -> tuple[dict, dict, Path]:
    """Fail closed before opening even a label-free role archive."""
    active = assert_active_from_env(unit_id=unit_id, stage="a")
    lock = Path(lock_path).resolve(strict=True)
    amendment_path = Path(amendment_path).resolve(strict=True)
    output = _private_output(output_dir)
    if (_sha(lock) != active["seal_sha256"] or
            _sha(amendment_path) != amendment_sha256):
        raise PermissionError("reference stage seal bytes differ")
    original = json.loads(lock.read_text())
    amendment = json.loads(amendment_path.read_text())
    if (original.get("schema") != "pcrl-latent-and-sampled-campaign-lock-v1" or
            original.get("status") != "LOCALLY_SEALED_D_STAGE" or
            amendment.keys() != A_REQUIRED or
            amendment.get("schema") != "pcrl-latent-sampled-A-stage-amendment-v1" or
            amendment.get("status") != "LOCALLY_SEALED_A_STAGE" or
            amendment.get("original_lock_sha256") != active["seal_sha256"] or
            reference_id not in REFERENCES or seed not in (0, 1) or
            role not in ("ATTACK_FIT_NEW", "INNER_SELECTION")):
        raise PermissionError("reference arm/role has no effective A seal")
    pins = original.get("source_files_sha256", {})
    binding = amendment.get("role_bindings", {})
    if (binding.get("repair_manifest_sha256") != pins.get("source_role_manifest") or
            binding.get("source_audit_sha256") != pins.get("r01/SOURCE_AUDIT.json")):
        raise PermissionError("reference role manifest differs from original lock")
    queue = Path(os.environ["PCRL_QUEUE_ROOT"]).resolve(strict=True)
    state = json.loads((queue / "state.json").read_text())
    amendments = [item for item in state.get("amendments", []) if "a" in item.get("stages", [])]
    spec = state.get("units", {}).get(unit_id, {}).get("spec", {})
    plan = amendment.get("unit_plan", {}).get(unit_id)
    expected_unit = (f"A-REFERENCE-RELEASE-{reference_id}-S{seed}-"
                     f"{'ATT' if role == 'ATTACK_FIT_NEW' else 'INNER'}")
    if (not amendments or amendments[-1].get("sha256") != amendment_sha256 or
            Path(amendments[-1].get("path", "")).resolve() != amendment_path or
            type(plan) is not dict or plan.get("kind") != "reference_release" or
            plan.get("reference_id") != reference_id or plan.get("seed") != seed or
            plan.get("role") != role or plan.get("realization") != 0 or
            unit_id != expected_unit or
            plan.get("output_id") != unit_id or
            plan.get("expected_outputs") !=
            ["reference_release.npz", "REFERENCE_RELEASE_RECEIPT.json"] or
            spec.get("stage") != "a" or spec.get("mode") != "real_data" or
            spec.get("amendment_sha256") != amendment_sha256 or
            spec.get("unit_id") != unit_id or
            spec.get("reference_id") != reference_id or
            spec.get("seed") != seed or spec.get("role") != role or
            spec.get("output_id") != unit_id or
            spec.get("outputs") != [str(output / "REFERENCE_RELEASE_RECEIPT.json")] or
            type(plan.get("cpu_seconds")) not in (int, float) or
            type(plan.get("wall_seconds")) not in (int, float) or
            plan["cpu_seconds"] <= 0 or plan["wall_seconds"] <= 0 or
            spec.get("cpu_reserve_s", 0) < plan["cpu_seconds"] or
            spec.get("max_wall_s", 0) < plan["wall_seconds"]):
        raise PermissionError("reference R03 unit/resource/output plan differs")
    implementation = amendment.get("implementation_source_sha256", {})
    if not SOURCE_FILES <= set(implementation):
        raise PermissionError("reference mandatory source pins missing")
    for relative in SOURCE_FILES:
        path = (lock.parent / relative).resolve(strict=True)
        if not path.is_relative_to(lock.parent):
            raise PermissionError("reference source escapes campaign")
        _listed_source(spec, path, implementation[relative])
    _runtime_origin_guard(lock.parent)
    _listed_source(spec, lock, active["seal_sha256"])
    _listed_source(spec, amendment_path, amendment_sha256)
    source_pins = amendment.get("source_sha256", {})
    audits = {"r01/SOURCE_AUDIT.json", "r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"}
    if reference_id in SOURCE_AUDITS:
        audits.add(SOURCE_AUDITS[reference_id][0])
    for relative in audits:
        path = (lock.parent / relative).resolve(strict=True)
        _listed_source(spec, path, source_pins.get(relative))
        if relative in SOURCE_AUDITS.get(reference_id, ())[:1]:
            if json.loads(path.read_text()).get("status") != SOURCE_AUDITS[reference_id][1]:
                raise PermissionError("reference independent source replay did not pass")
    return amendment, spec, output


def _source_binding(reference_id: str, seed: int, manifest_sha: str,
                    audit_sha: str, reference_sha: str, source_audit: dict) -> str:
    """Role-independent continuous source identity; role file stays in receipt."""
    model = source_audit["u0_models"][str(seed)]["components_sha256"]
    value = {"schema": "SCV1-continuous-reference-source-v1", "reference_id": reference_id,
             "seed": seed, "repair_manifest_sha256": manifest_sha,
             "source_audit_sha256": audit_sha, "u0_components_sha256": model}
    if reference_id == "Ucal":
        value["calibration_record_sha256"] = source_audit["ucal_calibration"][str(seed)]["record_sha256"]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def collect_reference_role(*, reference_id: str, seed: int, role: str,
                           repair_manifest_path, source_audit_path,
                           historical_audit_path, portfolio_package,
                           historical_cache_root) -> dict:
    """Use R01 only after admission; never open task, SEX or assessment fields."""
    store = RoleOnlyStore(repair_manifest_path)
    references = store.read(role, f"references_s{seed}")
    manifest = store.manifest
    key = f"{role}:references_s{seed}"
    ref_sha = manifest["role_file_sha256"][key]
    row_sha = manifest["roles"][role]["row_unit_pair_sha256"]
    audit_sha = sha_file(Path(source_audit_path))
    if reference_id in ("U0", "Ucal"):
        binding = _source_binding(reference_id, seed, sha_file(Path(repair_manifest_path)),
                                  audit_sha, ref_sha,
                                  json.loads(Path(source_audit_path).read_text()))
        kind = reference_id
        return dict(row_id=references["row_id"], unit=references["unit"],
                    prob1=references[f"{kind}_1"], prob2=references[f"{kind}_2"],
                    decision1=references["d_1"], decision2=references["d_2"],
                    code1=np.full(len(references["row_id"]), -1, np.int64),
                    code2=np.full(len(references["row_id"]), -1, np.int64),
                    source_binding_sha256=binding, interface_kind="continuous",
                    bank1=None, bank2=None,
                    bank_sha256_by_recipient={"r1": None, "r2": None},
                    expected_row_unit_sha256=row_sha,
                    source_identity={"role_source_file_sha256": ref_sha,
                                     "reference_file_sha256": ref_sha,
                                     "bank_receipt_sha256": None,
                                     "source_audit_sha256": audit_sha})
    if reference_id == "T_STAR":
        source = TStarSourceStore(portfolio_package, repair_manifest_path,
                                  source_audit_path, seed)
        role_audit_sha = audit_sha
    else:
        role_audit_sha = sha_file(Path(historical_audit_path))
        if reference_id == "P_STAR":
            source = PStarSourceStore(portfolio_package, repair_manifest_path,
                                      historical_audit_path, historical_cache_root, seed)
        else:
            source = D0SourceStore(portfolio_package, repair_manifest_path,
                                   historical_audit_path, historical_cache_root,
                                   D0_KEYS[reference_id], seed)
    cells = source.read_role_cells(role)
    assert_same_people(cells, references)
    bank1, bank2 = source.read_bank(1), source.read_bank(2)
    if reference_id == "T_STAR":
        wire1 = {"row_id": cells["row_id"], "unit": cells["unit"],
                 "code": cells["token_1"],
                 "probabilities": bank1["probability"][cells["token_1"]],
                 "decision": bank1["token_class"][cells["token_1"]]}
        wire2 = {"row_id": cells["row_id"], "unit": cells["unit"],
                 "code": cells["token_2"],
                 "probabilities": bank2["probability"][cells["token_2"]],
                 "decision": bank2["token_class"][cells["token_2"]]}
    else:
        wire1, wire2 = source.read_role_wire(role, 1), source.read_role_wire(role, 2)
    assert_same_people(wire1, wire2, references)
    bank_receipt = sha_file(source.root / "BANK_ADMISSION_RECEIPT.json")
    role_file = sha_file(source.root / f"tokens_{role}.npz")
    return dict(row_id=references["row_id"], unit=references["unit"],
                prob1=wire1["probabilities"], prob2=wire2["probabilities"],
                decision1=wire1["decision"], decision2=wire2["decision"],
                code1=wire1["code"], code2=wire2["code"],
                source_binding_sha256=source.source_binding_sha256,
                interface_kind="finite", bank1=bank1["probability"],
                bank2=bank2["probability"],
                bank_sha256_by_recipient={"r1": sha_array(bank1["probability"]),
                                          "r2": sha_array(bank2["probability"])},
                expected_row_unit_sha256=row_sha,
                source_identity={"role_source_file_sha256": role_file,
                                 "reference_file_sha256": ref_sha,
                                 "bank_receipt_sha256": bank_receipt,
                                 "source_audit_sha256": role_audit_sha})


def run(*, unit_id: str, lock_path, amendment_path, amendment_sha256: str,
        reference_id: str, seed: int, role: str, repair_manifest_path,
        source_audit_path, historical_audit_path, portfolio_package,
        historical_cache_root, output_dir) -> dict:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    amendment, spec, output = admit_reference_unit(
        unit_id=unit_id, lock_path=lock_path, amendment_path=amendment_path,
        amendment_sha256=amendment_sha256, output_dir=output_dir,
        reference_id=reference_id, seed=seed, role=role)
    campaign = Path(lock_path).resolve(strict=True).parent
    binding = amendment["role_bindings"]
    if (sha_file(Path(repair_manifest_path)) != binding["repair_manifest_sha256"] or
            Path(source_audit_path).resolve(strict=True) !=
            (campaign / "r01/SOURCE_AUDIT.json").resolve(strict=True) or
            sha_file(Path(source_audit_path)) != binding["source_audit_sha256"] or
            sha_file(Path(historical_audit_path)) !=
            amendment["source_sha256"]["r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"]):
        raise PermissionError("reference R01 source location/hash differs")
    _listed_source(spec, Path(repair_manifest_path), binding["repair_manifest_sha256"])
    _listed_source(spec, Path(source_audit_path), binding["source_audit_sha256"])
    _listed_source(spec, Path(historical_audit_path),
                   amendment["source_sha256"]["r01/A_HISTORICAL_SCORE_SOURCE_AUDIT.json"])
    if (reference_id not in ("U0", "Ucal") and
            (not Path(portfolio_package).resolve(strict=True).is_dir() or
             not Path(historical_cache_root).resolve(strict=True).is_dir())):
        raise PermissionError("reference predecessor source paths missing")
    values = collect_reference_role(
        reference_id=reference_id, seed=seed, role=role,
        repair_manifest_path=repair_manifest_path,
        source_audit_path=source_audit_path,
        historical_audit_path=historical_audit_path,
        portfolio_package=portfolio_package,
        historical_cache_root=historical_cache_root)
    plan = amendment["unit_plan"][unit_id]
    if (time.process_time() - started_cpu >= plan["cpu_seconds"] or
            time.monotonic() - started_wall >= plan["wall_seconds"]):
        raise TimeoutError("reference unit expired before release write")
    result = materialize_reference_role(
        **values, arm_id=reference_id, seed=seed, role=role,
        release_id=f"{reference_id}-s{seed}-v1", output_dir=output)
    if (time.process_time() - started_cpu > plan["cpu_seconds"] or
            time.monotonic() - started_wall > plan["wall_seconds"]):
        raise TimeoutError("reference unit exceeded sealed whole-unit ceiling")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("unit-id", "lock-path", "amendment-path", "amendment-sha256",
                 "reference-id", "role", "repair-manifest-path", "source-audit-path",
                 "historical-audit-path", "portfolio-package", "historical-cache-root",
                 "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = vars(parser.parse_args(argv))
    result = run(**{key.replace("-", "_"): value for key, value in args.items()})
    print(json.dumps({"status": result["status"], "rows": result["rows"],
                      "reference_id": result["arm_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
