"""Same-coefficient-row reference calibration for the new development study.

Every input coefficient already contains the normalized joint state mass.  A
cut is a *fixed fitted-predictor loss* floor, not an MI or population bound.
The historical cross-pool selected-reference target is deliberately not used.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1.method import validate_channel
from experiments.pcrl_task_aligned_cuts_v1.solver import (
    PRIMAL_TOL,
    bank_sha256,
    replay_p1,
    solve_p1,
)


def _role(cut: Mapping) -> str:
    role = cut.get("role")
    if role is None and "view" in cut and "target" in cut:
        role = f"{cut['view']}/{cut['target']}"
    if not isinstance(role, str) or "/" not in role:
        raise ValueError("cut requires recipient/target role")
    return role


def _allowances(delta, groups):
    if isinstance(delta, Mapping):
        if set(delta) != set(groups):
            raise ValueError("allowance map must contain exactly every role|weighting")
        values = dict(delta)
    else:
        values = {group: delta for group in groups}
    for group, value in values.items():
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"allowance for {group} must be finite and nonnegative")
        values[group] = value
    return values


def calibrate_reference(q_ref: np.ndarray, bank: Sequence[Mapping], delta,
                        *, require_provenance: bool = True):
    """Rebase *all* cuts to min-bank D17 risk on their coefficient rows.

    ``bank`` contains fixed attack coefficients with `id`, `role` (or legacy
    `view` + `target`), `weighting`, and `coeff`.  All attacks in a role/weighting
    must use the same coefficient people, class order, and normalization when
    their provenance metadata are supplied.  The returned ordinary cuts feed
    the inherited LP/MILP with floor = rho - delta.  Reinvoke this function on
    the entire retained bank after *every* bank expansion, never only on new
    cuts.  At delta >= 0, q_ref is a constructive feasible witness. Production
    use requires row/class/weight provenance for every attack; the explicit
    opt-out is for small synthetic mathematical fixtures only.
    """
    q_ref = validate_channel(q_ref)
    if not isinstance(bank, Sequence) or isinstance(bank, (str, bytes)) or not bank:
        raise ValueError("nonempty attack bank required")
    groups = defaultdict(list)
    metadata = {}
    role_people = {}
    seen = set()
    shape = q_ref.shape
    for source in bank:
        if not isinstance(source, Mapping):
            raise ValueError("cut must be a mapping")
        cid = source.get("id")
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("cut IDs must be unique nonempty strings")
        seen.add(cid)
        role = _role(source)
        weighting = source.get("weighting")
        if weighting not in ("U", "W"):
            raise ValueError("cut weighting must be U or W")
        group = f"{role}|{weighting}"
        coeff = np.asarray(source.get("coeff"), dtype=np.float64)
        if coeff.shape != shape or not np.isfinite(coeff).all() or np.min(coeff) < 0:
            raise ValueError("nonnegative finite cut coefficient shape must match Q_ref")
        required = ("coefficient_pool_sha256", "class_order", "weight_normalization")
        if require_provenance and any(source.get(key) is None for key in required):
            raise ValueError("production cut requires coefficient-pool/class/weight provenance")
        people_signature = (source.get("coefficient_pool_sha256"),
                            tuple(source["class_order"]) if source.get("class_order") is not None else None)
        if role in role_people and people_signature != role_people[role]:
            raise ValueError(f"coefficient pool mismatch: U and W for {role} must use same original coefficient people and class order")
        role_people[role] = people_signature
        signature = tuple(
            (key, tuple(value) if key == "class_order" and value is not None else value)
            for key in ("coefficient_pool_sha256", "class_order", "weight_normalization")
            if (value := source.get(key)) is not None
        )
        if group in metadata and signature != metadata[group]:
            raise ValueError(f"coefficient pool/class/weight normalization differs within {group}")
        metadata[group] = signature
        groups[group].append((cid, source, coeff, float(np.sum(coeff * q_ref))))
    allowances = _allowances(delta, groups)
    rho = {}
    cuts = []
    for group in sorted(groups):
        attacks = sorted(groups[group], key=lambda item: item[0])
        best = min(attacks, key=lambda item: (item[3], item[0]))
        floor = float(best[3] - allowances[group])
        rho[group] = {"value": best[3], "attaining_attack_id": best[0],
                      "reference_attack_losses": {cid: risk for cid, _, _, risk in attacks},
                      "allowance": allowances[group]}
        for cid, source, coeff, _ in attacks:
            cleaned = {k: v for k, v in source.items()
                       if k not in ("coeff", "floor", "rho", "delta")
                       and not k.startswith("reference_")}
            cuts.append({**cleaned, "id": cid, "coeff": coeff,
                         "rho": best[3], "delta": allowances[group], "floor": floor,
                         "calibration": "same_coefficient_rows_min_bank_D17_v1",
                         "reference_attack_id": best[0]})
    cuts.sort(key=lambda item: item["id"])
    witness = replay_p1(q_ref, np.zeros(shape), cuts)
    if witness["maximum_cut_violation"] > 1e-10:
        raise AssertionError("constructive Q_ref witness contradicts calibrated cuts")
    return {"cuts": cuts, "rho": rho, "witness": witness,
            "bank_sha256": bank_sha256(cuts, shape),
            "reference_objective": "min fixed attack loss on same coefficient rows",
            "population_privacy_guarantee": False}


def solve_calibrated(cost, q_ref, bank, delta, *, time_limit=None):
    """Calibrate, replay witness, then solve one fixed-decoder LP.

    A phase-I infeasibility report despite the witness is a defect requiring
    diagnosis, not permission to relax a role or silently fit a fallback.
    """
    q_ref = validate_channel(q_ref)
    cost = np.asarray(cost, dtype=np.float64)
    if cost.shape != q_ref.shape or not np.isfinite(cost).all():
        raise ValueError("fixed decoder cost must match reference channel")
    calibration = calibrate_reference(q_ref, bank, delta)
    solution = solve_p1(cost, calibration["cuts"], time_limit=time_limit)
    phase = solution["phase_one"]
    if (solution["status"] == "registered_bank_infeasible" or
            (phase["minimum_common_violation"] is not None and
             phase["minimum_common_violation"] > PRIMAL_TOL)):
        raise RuntimeError("phase I contradicts constructive Q_ref witness; freeze science and inspect routing")
    return {**calibration, "solution": solution,
            "reference_cost": float(np.sum(cost * q_ref)),
            "fixed_decoder_only": True}
