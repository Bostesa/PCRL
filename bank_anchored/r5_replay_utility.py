"""Queue-only independent replay of one saved bank law's FIT/INNER utility.

No candidate, runner endpoint, or checkpoint-selection code is imported.
Only role-only task labels and pinned bank/selected-law files are opened.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .r5_independent import expected_costs, saved_law_to_pairs


ROLE_MEMBERS = {
    "income": {"row_id", "unit", "income"},
    "occupation": {"row_id", "unit", "occupation"},
    "references": {"row_id", "unit", "U0_1", "U0_2", "Ucal_1",
                   "Ucal_2", "d_1", "d_2"},
}
LAW_MEMBERS = {"row_id", "unit", "code_r1", "code_r2", "mass", "valid",
               "b0_code_r1", "b0_code_r2"}
METRICS = ("accuracy", "nll", "brier")


def _point_status(margin: float, cap: float, metric: str) -> str:
    if metric not in METRICS or not (math.isfinite(margin) and math.isfinite(cap)):
        raise ValueError("nonfinite or unregistered utility boundary")
    passed = margin >= cap if metric == "accuracy" else margin <= cap
    return "PASS" if passed else "FAIL"


def _boundary_diagnostic(independent: float, reported: float,
                         cap: float, metric: str) -> tuple[str, str, bool]:
    own, prior = (_point_status(value, cap, metric)
                  for value in (independent, reported))
    if own != prior and not min(independent, reported) <= cap <= max(independent, reported):
        raise ValueError("independent status differs away from cap boundary")
    return own, prior, own != prior


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pinned(item: dict) -> Path:
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise PermissionError("source pin inventory differs")
    original = Path(item["path"])
    if original.is_symlink():
        raise PermissionError("source symlink refused")
    path = original.resolve(strict=True)
    if _sha(path) != item["sha256"]:
        raise PermissionError("source digest differs")
    return path


def _archive(path: Path, members: set[str]) -> dict[str, np.ndarray]:
    with ZipFile(path) as z:
        if (len(z.infolist()) != len(members)
                or {x.filename for x in z.infolist()} !=
                    {name + ".npy" for name in members}):
            raise PermissionError("role-only archive member inventory differs")
    with np.load(path, allow_pickle=False) as z:
        return {name: np.array(z[name], copy=True) for name in members}


def _task_costs(prob: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    p = np.asarray(prob)
    y = np.asarray(truth)
    if (p.dtype != np.float64 or p.ndim != 2 or y.dtype != np.int64
            or y.shape != (len(p),) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(y < 0) or np.any(y >= p.shape[1])
            or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-10)):
        raise ValueError("task reference schema differs")
    total = {key: 0.0 for key in METRICS}
    for row, label in zip(p, y):
        target = int(label)
        total["accuracy"] += int(np.argmax(row) == target)
        total["nll"] -= math.log(max(float(row[target]), 1e-12))
        total["brier"] += sum((float(value) - int(k == target)) ** 2
                              for k, value in enumerate(row))
    return {key: value / len(y) for key, value in total.items()}


def replay(config: Path, config_sha256: str, receipt_path: Path) -> dict:
    if _sha(config) != config_sha256:
        raise PermissionError("utility job config changed")
    job = json.loads(config.read_text())
    role, arm, seed = job["role"], job["arm"], job["seed"]
    if (job.get("schema") != "pcrl-bank-r3-utility-job-v1"
            or job.get("stage") != "protection"
            or role not in ("OSF_DEFENSE_FIT", "INNER_SELECTION")
            or arm not in ("B0", "T", "D", "P", "L", "J", "S12", "S21")
            or seed not in (0, 1)):
        raise PermissionError("replay role/arm/seed differs")
    if job["law"] is None:
        raise PermissionError("fitted-arm replay requires an exact saved law")
    manifest_path = _pinned(job["role_manifest"])
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent.parent
    records = {}
    for field in ("income", "occupation", "references"):
        source_field = f"references_s{seed}" if field == "references" else field
        key = f"{role}:{source_field}"
        relative = manifest["role_file_relative_path"][key]
        path = root / relative
        if _sha(path) != manifest["role_file_sha256"][key]:
            raise PermissionError("role-only task source differs")
        records[field] = _archive(path, ROLE_MEMBERS[field])
    row, unit = records["income"]["row_id"], records["income"]["unit"]
    if (row.dtype != np.int64 or unit.dtype != np.int64 or
            not np.all(np.diff(row) > 0) or
            any(not np.array_equal(value[name], reference)
                for value in records.values()
                for name, reference in (("row_id", row), ("unit", unit)))):
        raise PermissionError("task/reference ordered people differ")
    if (len(row) != manifest["roles"][role]["rows"]
            or len(np.unique(unit)) != manifest["roles"][role]["groups"]):
        raise PermissionError("role-only row/group counts differ")
    table_doc = json.loads(_pinned(job["public_table"]).read_text())
    if table_doc["seed"] != seed:
        raise PermissionError("table source seed differs")
    bank = table_doc["families"]["W_TSTAR"]
    tables = tuple(np.asarray(bank[f"table{i}"], dtype=np.float64) for i in (0, 1))
    law = _archive(_pinned(job["law"]), LAW_MEMBERS)
    pairs = saved_law_to_pairs(law, tables, row, unit)
    truth = np.column_stack((records["income"]["income"],
                             records["occupation"]["occupation"]))
    candidate = expected_costs(pairs, tables, truth)
    b0_pairs = [{(int(a), int(b)): 1.0} for a, b in zip(
        law["b0_code_r1"], law["b0_code_r2"])]
    b0 = expected_costs(b0_pairs, tables, truth)
    refs = {}
    for purpose, task in enumerate(("income", "occupation"), 1):
        for name in ("U0", "Ucal"):
            refs[(task, name)] = _task_costs(
                records["references"][f"{name}_{purpose}"],
                truth[:, purpose - 1])
    receipt = json.loads(receipt_path.read_text())
    csv_path = Path(job["outputs"]["csv"])
    if (receipt["output_csv_sha256"] != _sha(csv_path)
            or receipt["law_sha256"] != job["law"]["sha256"]):
        raise PermissionError("R3 output/law receipt differs")
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 18:
        raise ValueError("R3 utility clause count differs")
    checked = set()
    max_gap = 0.0
    reported_primary_pass = 0
    independent_primary_pass = 0
    boundary_disagreements = []
    for item in rows:
        task, ref, metric = item["task"], item["reference"], item["metric"]
        key = task, ref, metric
        if key in checked or task not in candidate or ref not in ("B0", "U0", "Ucal") or metric not in METRICS:
            raise ValueError("R3 utility row key duplicated or unknown")
        checked.add(key)
        candidate_value = candidate[task][metric]
        reference_value = (b0[task][metric] if ref == "B0" else refs[(task, ref)][metric])
        margin = candidate_value - reference_value
        for field, value in (("candidate_value", candidate_value),
                             ("reference_value", reference_value),
                             ("margin", margin)):
            gap = abs(float(item[field]) - value)
            max_gap = max(max_gap, gap)
            if gap > 1e-10:
                raise ValueError("independent exact-law utility differs from R3")
        primary = (ref == "B0" if role == "OSF_DEFENSE_FIT"
                   else ref in ("U0", "Ucal"))
        if item["primary_gate"] != str(primary):
            raise ValueError("primary FIT/INNER gate was misidentified")
        regime = "fit" if role == "OSF_DEFENSE_FIT" and ref == "B0" else "inner"
        cap = ((-.002 if regime == "fit" else -.010) if metric == "accuracy"
               else (.001 if regime == "fit" else .010) if metric == "nll"
               else .0005 if regime == "fit" else .005)
        if not math.isclose(float(item["cap"]), cap, rel_tol=0, abs_tol=1e-15):
            raise ValueError("prospective utility cap changed")
        reported_margin = float(item["margin"])
        status, reported_status, disputed = _boundary_diagnostic(
            margin, reported_margin, cap, metric)
        if primary:
            if item["point_status"] != reported_status or item["diagnostic_point_status"]:
                raise ValueError("R3 primary status differs from its own margin")
            reported_primary_pass += reported_status == "PASS"
            independent_primary_pass += status == "PASS"
        elif (item["point_status"] != "DIAGNOSTIC_ONLY" or
              item["diagnostic_point_status"] != reported_status):
            raise ValueError("R3 diagnostic status differs from its own margin")
        if disputed:
            boundary_disagreements.append({
                "task": task, "reference": ref, "metric": metric,
                "primary": primary, "cap": cap,
                "r3_margin": reported_margin, "r5_margin": margin,
                "r3_status": reported_status, "r5_status": status,
                "absolute_margin_gap": abs(margin - reported_margin),
            })
    expected_primary = 6 if role == "OSF_DEFENSE_FIT" else 12
    if receipt["primary_clause_count"] != expected_primary or \
            receipt["primary_clause_pass_count"] != reported_primary_pass or \
            receipt["primary_point_pass"] != (reported_primary_pass == expected_primary):
        raise ValueError("R3 primary point-gate receipt differs")
    return {"arm": arm, "seed": seed, "role": role, "rows": len(row),
            "groups": len(np.unique(unit)), "checked_clauses": len(checked),
            "primary_clauses": expected_primary,
            "primary_point_passed_reported": reported_primary_pass,
            "primary_point_passed_independent": independent_primary_pass,
            "boundary_disagreements": boundary_disagreements,
            "status": ("NUMERIC_BOUNDARY_UNRESOLVED" if boundary_disagreements
                       else "PASS"),
            "max_absolute_gap": max_gap, "law_sha256": job["law"]["sha256"],
            "r3_receipt_sha256": _sha(receipt_path),
            "r3_csv_sha256": _sha(csv_path)}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--r3-receipt", type=Path, required=True)
    parser.add_argument("--r3-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env
    active = assert_active_from_env(stage="verification")
    if _sha(args.r3_receipt) != args.r3_receipt_sha256:
        raise PermissionError("R3 receipt differs from independent queue pin")
    result = replay(args.config, args.config_sha256, args.r3_receipt)
    result.update({"schema": "pcrl-bank-r5-independent-law-utility-v2",
                   "unit_id": active["unit_id"],
                   "scope": "development role only; candidate/runner scoring not imported"})
    target = args.output.resolve(strict=False)
    expected = Path(__file__).resolve().parents[1] / "results/pcrl_bank_anchored_release_v1/r5"
    if target.parent != expected or target.exists():
        raise PermissionError("R5 aggregate receipt path differs")
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
