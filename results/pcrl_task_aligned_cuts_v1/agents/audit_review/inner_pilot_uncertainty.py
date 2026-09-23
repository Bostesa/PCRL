"""Descriptive one-anchor paired household bootstrap for the 2018 inner pilot.

No outer assessment or model fitting. Input receipts and contribution archives
are hash checked before loading. Public output has aggregates only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm


ROLES = ("utility:A/same_residence", "attack:A/SEX", "attack:A/RAC1P",
         "attack:AB/SEX", "attack:AB/RAC1P")
WEIGHTS = ("U", "PWGTP")
N_BOOT = 10000
SEED = 20260923


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_unit(root: Path, directory: Path, *, center: bool, sidecar_sha: str,
              queue_sha: str) -> dict:
    receipt_path = directory / ("COMPLETE.json" if center else "SIDECAR_COMPLETE.json")
    receipt = json.loads(receipt_path.read_text())
    if center:
        if receipt.get("unit_id") != "a0_u1p1_z000_audit" or receipt.get("queue_sha256") != queue_sha:
            raise ValueError("center completion receipt differs from registered queue")
    elif receipt.get("sidecar_sha256") != sidecar_sha:
        raise ValueError("control completion receipt differs from registered sidecar")
    relative = "INNER_PILOT_CONTRIBUTIONS.npz"
    archive = directory / relative
    if receipt.get("artifacts", {}).get(relative) != sha(archive):
        raise ValueError("person-level contribution archive changed")
    report_path = directory / "INNER_PANEL.json"
    if receipt.get("artifacts", {}).get("INNER_PANEL.json") != sha(report_path):
        raise ValueError("inner panel report changed")
    report = json.loads(report_path.read_text())
    if (report.get("anchor") != 0 or report.get("outer_pool_opened") is not False
            or report.get("score_pool") != "downstream_validation:inner_pilot"):
        raise ValueError("unexpected scoring scope")
    arrays = {}
    with np.load(archive, allow_pickle=False) as stored:
        for role in ROLES:
            slug = role.replace(":", "_").replace("/", "_")
            arrays[role] = {
                key: np.asarray(stored[f"{slug}_{key}"]).copy()
                for key in ("ids", "households", "weights", "candidate_loss", "H_loss")}
    return {"report": report, "arrays": arrays, "receipt_sha256": sha(receipt_path),
            "contributions_sha256": sha(archive), "report_sha256": sha(report_path)}


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(weights / weights.sum(), values))


def build(root: Path, *, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[dict, dict]:
    if n_boot < 2 or seed < 0:
        raise ValueError("bootstrap count/seed invalid")
    base = root / "results/pcrl_task_aligned_cuts_v1"
    protocol = json.loads((base / "PROTOCOL_LOCK.json").read_text())
    sidecar_path = base / "agents/audit_review/CONTROL_AUDIT_SIDECAR.json"
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("outer_pool_opened") is not False:
        raise ValueError("outer pool was opened")
    gradient_path = base / "agents/controls_data/GRADIENT_SELECTION.json"
    gradient = json.loads(gradient_path.read_text())
    if gradient.get("outer_pool_opened") is not False:
        raise ValueError("gradient selection opened outer")
    chosen = [v for v in gradient["full_curve"] if v["unit_id"] == gradient["selected_unit_id"]]
    if len(chosen) != 1 or not chosen[0]["eligible"] or chosen[0]["exact_Q_alias"]:
        raise ValueError("selected gradient is not one distinct eligible map")
    gradient_label = chosen[0]["label"]
    registrations = {**sidecar["simple_map_audit_units"], **sidecar["fitted_control_audits"]}
    names = ("D17", "D_U1", "MILP", gradient_label)
    if len(set(names)) != 4:
        raise ValueError("four distinct named controls required")
    center_dir = base / "private/run/a0_u1p1_z000_audit"
    center = load_unit(root, center_dir, center=True, sidecar_sha=sha(sidecar_path),
                       queue_sha=protocol["run_queue_sha256"])
    if center["report"]["release_id"] != "a0_u1p1_z000":
        raise ValueError("center release ID changed")
    controls = {}
    for name in names:
        entry = registrations[name]
        if entry["release_id"] != entry["canonical_release_id"]:
            raise ValueError(f"selected control is an alias: {name}")
        directory = root / entry["output_relative_dir"]
        controls[name] = load_unit(root, directory, center=False,
                                   sidecar_sha=sha(sidecar_path),
                                   queue_sha=protocol["run_queue_sha256"])
        if controls[name]["report"]["release_id"] != entry["release_id"]:
            raise ValueError(f"control release ID differs: {name}")
    control_rows = list(csv.DictReader((base / "MATCHED_CONTROLS.csv").open(newline="")))
    core_rows = list(csv.DictReader((base / "CORE_FACTORIAL.csv").open(newline="")))
    control_table = {(v["control_label"], v["role"], v["weighting"]): v
                     for v in control_rows if v["anchor"] == "0"}
    core_table = {(v["configuration"], v["role"], v["weighting"]): v
                  for v in core_rows if v["anchor"] == "0"}
    members = set()
    endpoints = []
    raw = []
    for name in names:
        for role in ROLES:
            left = center["arrays"][role]
            right = controls[name]["arrays"][role]
            for field in ("ids", "households", "weights", "H_loss"):
                if not np.array_equal(left[field], right[field]):
                    raise ValueError(f"paired {field} differs for {name}/{role}")
            members.update(str(v) for v in left["households"])
            if len(np.unique(left["ids"])) != len(left["ids"]):
                raise ValueError("duplicate original-person IDs")
            for weighting in WEIGHTS:
                weights = (np.ones(len(left["ids"]), dtype=np.float64)
                           if weighting == "U" else np.asarray(left["weights"], dtype=np.float64))
                if (not np.isfinite(weights).all() or np.min(weights) < 0
                        or weights.sum() <= 0):
                    raise ValueError("invalid original-person weights")
                sign = 1 if role == ROLES[0] else -1
                diff = sign * (left["candidate_loss"] - right["candidate_loss"])
                estimate = _weighted(diff, weights)
                cent = _weighted(left["candidate_loss"], weights)
                comp = _weighted(right["candidate_loss"], weights)
                c_cell = core_table[("a0_u1p1_z000", role, weighting)]
                m_cell = control_table[(name, role, weighting)]
                if name in ("MILP", gradient_label) and m_cell["fixed_bank_feasible"] != "true":
                    raise ValueError("matched fitted control is not bank feasible")
                if (abs(cent - float(c_cell["release_loss"])) > 1e-10
                        or abs(comp - float(m_cell["candidate_loss"])) > 1e-10):
                    raise ValueError(f"public aggregate table differs from private replay: {name}/{role}/{weighting}")
                ident = f"center|{name}|{role}|{weighting}"
                endpoints.append({"id": ident, "comparator": name, "role": role,
                                  "weighting": weighting,
                                  "orientation": ("CE_center-CE_control" if sign == 1 else
                                                  "CE_control-CE_center = recovery_center-recovery_control"),
                                  "estimate": estimate,
                                  "original_people": len(left["ids"]),
                                  "households": len(np.unique(left["households"])),
                                  "identical_pairs": bool(np.all(diff == 0))})
                raw.append((np.asarray(left["households"]).astype(str), weights, diff))
    if len(endpoints) != 40:
        raise AssertionError("four controls × five roles × two weightings required")
    households = np.asarray(sorted(members), dtype=str)
    index = {h: j for j, h in enumerate(households)}
    h, m = len(households), len(endpoints)
    numerator = np.zeros((h, m), dtype=np.float64)
    denominator = np.zeros((h, m), dtype=np.float64)
    for j, (ids, weights, diff) in enumerate(raw):
        codes = np.fromiter((index[v] for v in ids), dtype=np.int64)
        numerator[:, j] = np.bincount(codes, weights=weights * diff, minlength=h)
        denominator[:, j] = np.bincount(codes, weights=weights, minlength=h)
    rng = np.random.default_rng(seed)
    count = 0
    means = np.zeros(m)
    m2 = np.zeros(m)
    rejected = 0
    attempted = 0
    while count < n_boot:
        if attempted >= max(10000, 100 * n_boot):
            raise RuntimeError("insufficient globally valid household bootstrap draws")
        batch = min(128, n_boot - count)
        draws = rng.multinomial(h, np.full(h, 1 / h), size=batch)
        attempted += batch
        den = draws @ denominator
        valid = np.all(den > 0, axis=1)
        rejected += int(np.sum(~valid))
        if not np.any(valid):
            continue
        values = (draws[valid] @ numerator) / den[valid]
        size = len(values)
        local_mean = values.mean(axis=0)
        local_m2 = np.sum((values - local_mean) ** 2, axis=0)
        new_count = count + size
        delta = local_mean - means
        m2 += local_m2 + delta * delta * count * size / new_count
        means += delta * size / new_count
        count = new_count
    se = np.sqrt(np.maximum(0., m2) / (n_boot - 1))
    z_point = float(norm.isf(.025))
    z_family = float(norm.isf(.05 / (2 * m)))
    for j, endpoint in enumerate(endpoints):
        endpoint.update({"bootstrap_se": float(se[j]),
                         "pointwise_95_lower": endpoint["estimate"] - z_point * float(se[j]),
                         "pointwise_95_upper": endpoint["estimate"] + z_point * float(se[j]),
                         "descriptive_family40_95_lower": endpoint["estimate"] - z_family * float(se[j]),
                         "descriptive_family40_95_upper": endpoint["estimate"] + z_family * float(se[j])})
    public = {"schema": "pcrl-inner-pilot-household-uncertainty-v1",
              "scope": "2018 historically reused anchor-0 inner_pilot; descriptive only, no outer assessment",
              "limitations": ["These intervals condition on fitted/selected models and do not cover adaptive model or study selection.",
                              "The 2018 pool was used in earlier studies; these are not confirmatory intervals.",
                              "Only anchor 0 is resampled; anchors 1/2 are not imputed or treated as independent."],
              "sampling_unit": "household; one shared multinomial draw across all 40 endpoints",
              "center_release_id": "a0_u1p1_z000", "selected_gradient_label": gradient_label,
              "comparators": list(names), "n_endpoints": m,
              "households_union": h, "bootstrap": {"seed": seed, "requested": n_boot,
              "accepted": count, "attempted": attempted, "rejected_zero_denominator": rejected,
              "pointwise_95_z": z_point, "descriptive_family40_95_z": z_family},
              "endpoints": endpoints,
              "input_sha256": {"CORE_FACTORIAL.csv": sha(base / "CORE_FACTORIAL.csv"),
                               "MATCHED_CONTROLS.csv": sha(base / "MATCHED_CONTROLS.csv"),
                               "GRADIENT_SELECTION.json": sha(gradient_path),
                               "CONTROL_AUDIT_SIDECAR.json": sha(sidecar_path)}}
    private = {"schema": "pcrl-inner-pilot-uncertainty-replay-v1",
               "script_sha256": sha(Path(__file__)),
               "bootstrap_seed": seed, "bootstrap_replicates": n_boot,
               "private_inputs": {"center": {k: center[k] for k in ("receipt_sha256", "contributions_sha256", "report_sha256")},
                                  **{name: {k: controls[name][k] for k in ("receipt_sha256", "contributions_sha256", "report_sha256")}
                                     for name in names}},
               "public_input_sha256": public["input_sha256"],
               "person_or_household_rows_copied": False}
    return public, private


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    public, private = build(args.root.resolve())
    encoded = json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.verify_only:
        if args.public.read_text() != encoded:
            raise ValueError("public aggregate uncertainty differs from replay")
        receipt = json.loads(args.private.read_text())
        if receipt != {**private, "public_output_sha256": sha(args.public)}:
            raise ValueError("private replay receipt differs")
        print(json.dumps({"status": "verified", "public_sha256": sha(args.public)}))
        return
    if args.public.exists() or args.private.exists():
        raise FileExistsError("uncertainty output already exists; use --verify-only")
    args.private.parent.mkdir(parents=True, exist_ok=True)
    args.public.write_text(encoded)
    args.private.write_text(json.dumps({**private, "public_output_sha256": sha(args.public)},
                                       indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "endpoints": public["n_endpoints"],
                      "public_sha256": sha(args.public)}))


if __name__ == "__main__":
    main()
