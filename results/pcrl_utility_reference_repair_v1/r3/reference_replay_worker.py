"""Independent queued parity of the frozen U0 and Ucal references.

Only admitted role-only features and reference arrays are opened. The old
mixed-role release archive, assessment files, and labels are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit
from results.pcrl_utility_reference_repair_v1.r3.source_reference import (
    apply_inverse_temperature, file_sha, forward_u0, restore_source,
)


ROLES = ("OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT", "INNER_SELECTION")
TOLERANCE = 2e-7  # Registered before Adult forward, from synthetic batch geometry.
# SRC|U in the committed hcal MODEL_MANIFEST.json, independently of the
# mutable private COMPLETE receipt. The present campaign uses paired 0/1.
SOURCE_MODEL_SHA = (
    "5061a0074af74d21b53aca80b96d8b83b85a3e43cdb9e6758b2551be9045b9a7",
    "f52bef54ef77faf51cb6f55273e54f75ea094c4379bbf73b57a3941ce2f7a74c",
)


def array_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def require_input(spec: dict, path: Path) -> str:
    resolved = path.resolve(strict=True)
    expected = spec["inputs"].get(str(resolved))
    if not isinstance(expected, str) or file_sha(resolved) != expected:
        raise PermissionError("reference replay input absent or changed")
    return expected


def role_arrays(export_root: Path, manifest: dict, role: str, seed: int,
                spec: dict) -> tuple[np.ndarray, dict, dict]:
    if role not in ROLES or seed not in (0, 1):
        raise PermissionError("reference replay role or seed outside admitted stage")
    paths = {
        "features": export_root / role / "features.npz",
        "references": export_root / role / f"references_s{seed}.npz",
    }
    role_keys = {"features": f"{role}:features",
                 "references": f"{role}:references_s{seed}"}
    for kind, path in paths.items():
        expected = manifest["role_file_sha256"][role_keys[kind]]
        if require_input(spec, path) != expected:
            raise PermissionError("role file differs from admission manifest")
    with np.load(paths["features"], allow_pickle=False) as archive:
        if set(archive.files) != {"X", "row_id", "unit", "feature_names"}:
            raise PermissionError("feature archive member set differs")
        x = archive["X"]
        row_id = archive["row_id"]
        unit = archive["unit"]
        names = archive["feature_names"]
    with np.load(paths["references"], allow_pickle=False) as archive:
        expected_members = {"row_id", "unit", "U0_1", "U0_2", "Ucal_1",
                            "Ucal_2", "d_1", "d_2"}
        if set(archive.files) != expected_members:
            raise PermissionError("reference archive member set differs")
        if (not np.array_equal(row_id, archive["row_id"])
                or not np.array_equal(unit, archive["unit"])):
            raise PermissionError("reference people or order differ")
        refs = {key: archive[key] for key in expected_members - {"row_id", "unit"}}
    if (x.dtype != np.float32 or x.shape != (manifest["roles"][role]["rows"], 83)
            or row_id.dtype != np.int64 or unit.dtype != np.int64
            or len(np.unique(row_id)) != len(row_id)
            or len(np.unique(unit)) != manifest["roles"][role]["groups"]
            or len(names) != 83 or not np.isfinite(x).all()):
        raise PermissionError("role feature or identity schema differs")
    schema_sha = hashlib.sha256("\n".join(map(str, names)).encode()).hexdigest()
    if schema_sha != manifest["source_schema_sha256"]:
        raise PermissionError("role feature order differs")
    identity = {"rows": len(row_id), "groups": len(np.unique(unit)),
                "row_sha256": array_sha(row_id),
                "unit_sha256": array_sha(unit)}
    return x, refs, identity


def replay(spec: dict, export_root: Path, hcal_private: Path, manifest: dict,
           role: str, seed: int) -> dict:
    x, refs, identity = role_arrays(export_root, manifest, role, seed, spec)
    source = hcal_private / "admitted" / f"rel__s{seed}__U"
    complete_path = source / "COMPLETE.json"
    cal_dir = hcal_private / "run" / "units" / f"calU__s{seed}"
    cal_complete_path = cal_dir / "COMPLETE.json"
    cal_record_path = cal_dir / "record.json"
    for path in (complete_path, cal_complete_path, cal_record_path):
        require_input(spec, path)
    complete = json.loads(complete_path.read_text())
    if complete.get("id") != f"rel__s{seed}__U":
        raise PermissionError("source U unit identity differs")
    names = ("model.pt", "head_0.joblib", "head_1.joblib")
    if complete["files"].get("model.pt") != SOURCE_MODEL_SHA[seed]:
        raise PermissionError("source U model differs from committed hcal manifest")
    for name in names:
        path = source / name
        if require_input(spec, path) != complete["files"][name]:
            raise PermissionError("source U file differs from receipt")
    cal_complete = json.loads(cal_complete_path.read_text())
    if (cal_complete.get("id") != f"calU__s{seed}"
            or cal_complete.get("files", {}).get("record.json") !=
            file_sha(cal_record_path)):
        raise PermissionError("calibration unit identity or record differs")
    cal = json.loads(cal_record_path.read_text())
    if cal.get("schema") != "hcal-calU-v1" or cal.get("seed") != seed:
        raise PermissionError("Ucal fit record identity differs")
    model, heads = restore_source(
        source / "model.pt",
        (source / "head_0.joblib", source / "head_1.joblib"),
        tuple(complete["files"][name] for name in names[:3]),
    )
    for i, k in enumerate((2, 6)):
        if not np.array_equal(heads[i].classes_, np.arange(k)):
            raise PermissionError("source head class order differs")
    restored = forward_u0(model, heads, x)
    result = {
        "schema": "pcrl-repair-r3-reference-parity-v1", "role": role, "seed": seed,
        "identity": identity, "probability_tolerance": TOLERANCE,
        "batch_convention": "whole admitted role; original source used whole mixed-role forward",
        "source_complete_sha256": file_sha(complete_path),
        "calibration_complete_sha256": file_sha(cal_complete_path),
        "calibration_record_sha256": file_sha(cal_record_path),
        "tasks": {},
    }
    for i, task in enumerate(("income", "occupation_group"), 1):
        alpha = cal["tables"][f"H-GLOBAL-TEMP|r{i}"]["alpha"]
        actual = {"U0": restored[i - 1],
                  "Ucal": apply_inverse_temperature(restored[i - 1], alpha)}
        task_result = {"alpha": float(alpha), "references": {}}
        for name, value in actual.items():
            stored = np.asarray(refs[f"{name}_{i}"], dtype=np.float64)
            if value.shape != stored.shape:
                raise PermissionError("stored reference shape differs")
            decision = np.asarray(refs[f"d_{i}"])
            if (decision.dtype != np.int64 or decision.shape != (len(x),)
                    or np.any(decision < 0) or np.any(decision >= value.shape[1])
                    or not np.isfinite(stored).all()
                    or np.any(stored < 0) or np.any(stored > 1)
                    or np.max(np.abs(stored.sum(axis=1) - 1)) > 1e-8):
                raise PermissionError("stored reference or decision values differ")
            delta = np.abs(value - stored)
            task_result["references"][name] = {
                "stored_array_sha256": array_sha(stored),
                "restored_array_sha256": array_sha(value),
                "bitwise_equal": bool(np.array_equal(value, stored)),
                "max_abs_difference": float(delta.max(initial=0.0)),
                "values_above_tolerance": int(np.count_nonzero(delta > TOLERANCE)),
                "restored_decision_agreement": bool(np.array_equal(
                    value.argmax(axis=1), decision)),
                "stored_decision_agreement": bool(np.array_equal(
                    stored.argmax(axis=1), decision)),
            }
        result["tasks"][task] = task_result
    result["pass"] = all(
        cell["values_above_tolerance"] == 0
        and cell["restored_decision_agreement"]
        and cell["stored_decision_agreement"]
        for task in result["tasks"].values()
        for cell in task["references"].values()
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--hcal-private", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = assert_running_unit(args.unit_id, args.queue_root, args.lock,
                               "reference_parity", args.output)
    require_input(spec, args.manifest)
    manifest = json.loads(args.manifest.read_text())
    result = replay(spec, args.export_root, args.hcal_private, manifest,
                    args.role, args.seed)
    if args.output.exists():
        raise FileExistsError("independent reference report is write-once")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["pass"]:
        raise SystemExit("frozen reference parity beyond predeclared tolerance")


if __name__ == "__main__":
    main()
