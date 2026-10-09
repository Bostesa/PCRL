"""Restore frozen U0/Ucal on admitted role-only features, without labels.

The script opens only an explicit role's `features.npz` and
`references_s{k}.npz`, the admitted U deployment unit, and its fixed H-GLOBAL
calibrator record. It refuses other roles and never calls the old mixed-role
loader or reads an old assessment array.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import joblib
import torch

from dpc.deploy import teacher_probs
from hcal.calib import temp_apply_probs
from jcv.train import Model
from results.pcrl_utility_reference_repair_v1.r0.admission import assert_running_unit


ROLES = ("OSF_DEFENSE_FIT", "CALIBRATION_HELDOUT", "INNER_SELECTION")
# Pre-adult tolerance from independent synthetic batch-geometry test:
# float32 Torch GEMM can vary by <1e-7 when the original all-role forward is
# replayed on role-only batches. Preserve bitwise counts and the exact delta.
TOLERANCE = 2e-7


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _array_sha(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def safe_load_u(source: Path, seed: int):
    """Restore only U weights/heads; never touch the all-role release.npz."""
    complete = json.loads((source / "COMPLETE.json").read_text())
    expected = {"model.pt", "head_0.joblib", "head_1.joblib", "record.json"}
    if complete.get("id") != f"rel__s{seed}__U" or not expected <= set(complete.get("files", {})):
        raise PermissionError("frozen U complete receipt identity differs")
    for name in expected:
        if _sha_file(source / name) != complete["files"][name]:
            raise PermissionError("frozen U permitted file hash differs")
    record = json.loads((source / "record.json").read_text())
    if record.get("seed") != seed:
        raise PermissionError("frozen U model seed differs")
    model = Model(83, [2, 6], seed)
    model.load_state_dict(torch.load(source / "model.pt", weights_only=True))
    model.eval()
    heads = tuple(joblib.load(source / f"head_{i}.joblib") for i in (0, 1))
    return model, heads, complete["files"]["model.pt"]


def _role_arrays(export_root: Path, manifest: dict, role: str, seed: int):
    if role not in ROLES or seed not in (0, 1):
        raise PermissionError("reference parity role or seed is outside the admitted stage")
    paths = {kind: export_root / role / f"{kind}.npz"
             for kind in ("features", f"references_s{seed}")}
    for kind, path in paths.items():
        if _sha_file(path) != manifest["role_file_sha256"][f"{role}:{kind}"]:
            raise PermissionError("role-only reference export hash differs from admission")
    with np.load(paths["features"], allow_pickle=False) as z:
        if set(z.files) != {"X", "row_id", "unit", "feature_names"}:
            raise PermissionError("role-only feature member set differs")
        x = z["X"]
        row, unit = z["row_id"], z["unit"]
        names = z["feature_names"]
    with np.load(paths[f"references_s{seed}"], allow_pickle=False) as z:
        expected = {"row_id", "unit", "U0_1", "U0_2", "Ucal_1",
                    "Ucal_2", "d_1", "d_2"}
        if set(z.files) != expected:
            raise PermissionError("role-only reference member set differs")
        if not np.array_equal(row, z["row_id"]) or not np.array_equal(unit, z["unit"]):
            raise PermissionError("role-only reference people differ from feature order")
        refs = {kind: z[kind] for kind in sorted(expected - {"row_id", "unit"})}
    if (x.dtype != np.float32 or x.shape != (manifest["roles"][role]["rows"], 83)
            or len(names) != 83 or not np.isfinite(x).all()):
        raise PermissionError("role-only feature schema differs from source")
    schema = hashlib.sha256("\n".join(map(str, names)).encode()).hexdigest()
    if schema != manifest["source_schema_sha256"]:
        raise PermissionError("83-column feature order differs from source")
    return x, row, unit, names, refs, paths


def replay_one(export_root: Path, hcal_private: Path, manifest: dict,
               role: str, seed: int) -> dict:
    x, row, unit, names, refs, paths = _role_arrays(
        export_root, manifest, role, seed)
    source = hcal_private / "admitted" / f"rel__s{seed}__U"
    cal_record_path = hcal_private / "run" / "units" / f"calU__s{seed}" / "record.json"
    model, heads, model_sha = safe_load_u(source, seed)
    for i, classes in enumerate((2, 6)):
        if not np.array_equal(heads[i].classes_, np.arange(classes)):
            raise PermissionError("frozen source head class order differs")
    cal = json.loads(cal_record_path.read_text())
    if cal.get("schema") != "hcal-calU-v1" or cal.get("seed") != seed:
        raise PermissionError("frozen U calibration record differs")
    restored = teacher_probs(model, heads, x)
    result = {
        "schema": "pcrl-utility-reference-parity-v1", "role": role,
        "seed": seed, "rows": len(row), "groups": len(np.unique(unit)),
        "feature_file_sha256": _sha_file(paths["features"]),
        "reference_file_sha256": _sha_file(paths[f"references_s{seed}"]),
        "source_model_sha256": model_sha,
        "source_head_sha256": [_sha_file(source / f"head_{i}.joblib")
                               for i in (0, 1)],
        "source_complete_sha256": _sha_file(source / "COMPLETE.json"),
        "calibration_record_sha256": _sha_file(cal_record_path),
        "probability_tolerance": TOLERANCE,
        "batch_convention": "full admitted role at once; original source used full mixed-role forward",
        "tasks": {},
    }
    for i, task in enumerate(("income", "occupation_group"), 1):
        u0 = np.asarray(restored[i - 1], dtype=np.float64)
        alpha = cal["tables"][f"H-GLOBAL-TEMP|r{i}"]["alpha"]
        uc = temp_apply_probs(u0, alpha, np.asarray(refs[f"d_{i}"], dtype=np.int64))
        row_result = {}
        for name, actual in (("U0", u0), ("Ucal", uc)):
            stored = np.asarray(refs[f"{name}_{i}"], dtype=np.float64)
            if actual.shape != stored.shape:
                raise PermissionError("restored reference shape differs")
            delta = np.abs(actual - stored)
            row_result[name] = {
                "stored_sha256": _array_sha(stored),
                "restored_sha256": _array_sha(actual),
                "bitwise_equal": bool(np.array_equal(actual, stored)),
                "max_abs_difference": float(delta.max(initial=0)),
                "values_above_tolerance": int(np.count_nonzero(delta > TOLERANCE)),
                "decision_agreement": bool(np.array_equal(actual.argmax(1), refs[f"d_{i}"])),
            }
        row_result["Ucal_alpha"] = float(alpha)
        result["tasks"][task] = row_result
    result["pass"] = all(
        item["values_above_tolerance"] == 0 and item["decision_agreement"]
        for task in result["tasks"].values()
        for key, item in task.items() if key in ("U0", "Ucal"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--hcal-private", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    assert_running_unit(args.unit_id, args.queue_root, args.lock,
                        "reference_parity", args.output)
    manifest = json.loads(args.manifest.read_text())
    result = replay_one(args.export_root, args.hcal_private, manifest,
                        args.role, args.seed)
    if args.output.exists():
        raise FileExistsError("reference parity result is write-once")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["pass"]:
        raise SystemExit("frozen U0/Ucal parity differs beyond predeclared tolerance")


if __name__ == "__main__":
    main()
