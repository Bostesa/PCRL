"""Optional, fixed-bank privacy-first LP on the registered 17-token contract.

This is a training-bank diagnostic, not an attacker refit, new release fit,
independent audit, or population privacy guarantee. The frozen reference risks
and coefficient people are supplied by the completed adaptive center.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from . import controls


OTHER_ALLOWANCE = .001
TASK_ALLOWANCE = .001
TOL = 1e-7
ROLES = ("A/SEX", "A/RAC1P", "AB/SEX", "AB/RAC1P")
WEIGHTS = ("U", "W")


def _problem(cost_pair: Mapping, cuts: Sequence[Mapping], d17: np.ndarray):
    if not isinstance(cost_pair, Mapping) or set(cost_pair) != set(WEIGHTS):
        raise ValueError("separate U and W fixed-decoder task costs required")
    cost = {weight: controls._cost(cost_pair[weight]) for weight in WEIGHTS}
    q_ref = controls._channel(d17, deterministic=True)
    shape = q_ref.shape
    if any(cost[weight].shape != shape for weight in WEIGHTS):
        raise ValueError("task costs and D17 must use identical states and tokens")
    if not isinstance(cuts, Sequence) or isinstance(cuts, (str, bytes)) or not cuts:
        raise ValueError("nonempty frozen attack bank required")
    parsed, groups, reference_losses, identifiers = [], {}, {}, set()
    for source in cuts:
        if not isinstance(source, Mapping):
            raise ValueError("every cut must be a mapping")
        cid, role, weighting = source.get("id"), source.get("role"), source.get("weighting")
        if not isinstance(cid, str) or not cid or cid in identifiers:
            raise ValueError("unique nonempty cut IDs required")
        identifiers.add(cid)
        if role not in ROLES or weighting not in WEIGHTS:
            raise ValueError("unexpected role or weighting in privacy-first bank")
        coeff = np.asarray(source.get("coeff"), dtype=np.float64)
        rho = float(source.get("rho"))
        delta = float(source.get("delta"))
        floor = float(source.get("floor"))
        if (coeff.shape != shape or not np.isfinite(coeff).all() or np.min(coeff) < 0
                or not np.isfinite(rho) or not np.isfinite(delta) or delta < 0
                or not np.isfinite(floor)):
            raise ValueError("invalid fixed attack coefficients or reference risk")
        if abs(floor - (rho-delta)) > 1e-10:
            raise ValueError("stored floor must equal rho minus registered delta")
        group = (role, weighting)
        if group in groups and abs(groups[group]-rho) > 1e-10:
            raise ValueError("frozen rho differs within role and weighting")
        groups[group] = rho
        reference_losses.setdefault(group, []).append(float(np.sum(coeff*q_ref)))
        parsed.append({"id": cid, "role": role, "weighting": weighting,
                       "coeff": coeff, "rho": rho})
    expected = {(role, weight) for role in ROLES for weight in WEIGHTS}
    if set(groups) != expected:
        raise ValueError("every role and weighting requires at least one frozen attack cut")
    for group, rho in groups.items():
        if abs(rho-min(reference_losses[group])) > 1e-10:
            raise ValueError(f"rho must equal minimum retained attack loss on D17 coefficient rows: {group}")
    return cost, parsed, q_ref


def replay_privacy_first(q: np.ndarray, tau: float, cost_pair: Mapping,
                         cuts: Sequence[Mapping], d17: np.ndarray) -> dict:
    """Independently recompute every registered floor and both task caps."""
    cost, bank, q_ref = _problem(cost_pair, cuts, d17)
    q = controls._channel(q)
    if q.shape != q_ref.shape or not np.isfinite(tau):
        raise ValueError("candidate channel or tau does not match frozen problem")
    reference_task = {weight: float(np.sum(cost[weight]*q_ref)) for weight in WEIGHTS}
    task = {weight: float(np.sum(cost[weight]*q)) for weight in WEIGHTS}
    task_violation = {weight: max(0., task[weight]-reference_task[weight]-TASK_ALLOWANCE)
                      for weight in WEIGHTS}
    slacks = {}
    for cut in bank:
        floor = cut["rho"] + tau if cut["role"] == "AB/SEX" else cut["rho"]-OTHER_ALLOWANCE
        slacks[cut["id"]] = float(np.sum(cut["coeff"]*q)-floor)
    max_cut = max((max(0., -slack) for slack in slacks.values()), default=0.)
    simplex = float(np.max(np.abs(q.sum(axis=1)-1.)))
    min_entry = float(np.min(q))
    return {"feasible": bool(tau >= -TOL and max_cut <= TOL and
                              max(task_violation.values()) <= TOL and
                              simplex <= TOL and min_entry >= -TOL),
            "tau": float(tau), "reference_task_loss": reference_task,
            "task_loss": task, "task_cap": {weight: reference_task[weight]+TASK_ALLOWANCE
                                               for weight in WEIGHTS},
            "task_cap_violation": task_violation,
            "minimum_cut_slack": float(min(slacks.values())),
            "maximum_cut_violation": float(max_cut),
            "simplex_residual": simplex, "minimum_entry": min_entry,
            "cut_slacks": slacks}


def _matrices(cost, bank, q_ref, *, phase_one: bool):
    n_states, n_tokens = q_ref.shape
    n_q = n_states*n_tokens
    width = n_q+1
    rows, rhs = [], []
    for cut in bank:
        row = np.zeros(width, dtype=np.float64)
        row[:n_q] = -cut["coeff"].ravel()
        if phase_one:
            row[-1] = -1.  # common nonnegative phase-I violation
        elif cut["role"] == "AB/SEX":
            row[-1] = 1.  # L(Q) >= rho + tau
        rows.append(row)
        rhs.append(-cut["rho"] if cut["role"] == "AB/SEX"
                   else -(cut["rho"]-OTHER_ALLOWANCE))
    for weight in WEIGHTS:
        row = np.zeros(width, dtype=np.float64)
        row[:n_q] = cost[weight].ravel()
        if phase_one:
            row[-1] = -1.
        rows.append(row)
        rhs.append(float(np.sum(cost[weight]*q_ref))+TASK_ALLOWANCE)
    equality = np.zeros((n_states, width), dtype=np.float64)
    for state in range(n_states):
        equality[state, state*n_tokens:(state+1)*n_tokens] = 1.
    bounds = [(0., 1.)]*n_q + [(0., None)]
    return np.vstack(rows), np.asarray(rhs), equality, np.ones(n_states), bounds


def phase_one_privacy_first(cost_pair: Mapping, cuts: Sequence[Mapping],
                            d17: np.ndarray, *, time_limit_seconds: float = 30.) -> dict:
    """At tau=0, minimize one common slack over all attack and task clauses."""
    cost, bank, q_ref = _problem(cost_pair, cuts, d17)
    if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("positive finite phase-I time limit required")
    aub, bub, aeq, beq, bounds = _matrices(cost, bank, q_ref, phase_one=True)
    objective = np.zeros(aub.shape[1]); objective[-1] = 1.
    raw = linprog(objective, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=beq,
                  bounds=bounds, method="highs", options={"time_limit": time_limit_seconds})
    value = float(raw.x[-1]) if raw.x is not None else None
    residual = None
    if raw.x is not None:
        residual = {"maximum_inequality_violation": float(np.max(np.maximum(aub@raw.x-bub, 0.))),
                    "maximum_simplex_residual": float(np.max(np.abs(aeq@raw.x-beq))),
                    "minimum_entry": float(np.min(raw.x))}
    return {"status": "OPTIMAL" if raw.status == 0 else "UNRESOLVED",
            "solver_status": int(raw.status), "solver_message": raw.message,
            "minimum_common_violation": value, "residual": residual,
            "time_limit_seconds": float(time_limit_seconds)}


def _dual_report(raw, aub, bub, aeq, beq, n_q, objective):
    if raw.status != 0 or raw.x is None:
        return None
    yi = np.asarray(raw.ineqlin.marginals)
    ye = np.asarray(raw.eqlin.marginals)
    yl = np.asarray(raw.lower.marginals)
    yu = np.asarray(raw.upper.marginals)
    stationarity = objective-aub.T@yi-aeq.T@ye-yl-yu
    value = float(bub@yi+beq@ye+np.sum(yu[:n_q]))
    return {"minimization_dual_lower_bound": value,
            "tau_upper_bound": -value,
            "stationarity_residual": float(np.max(np.abs(stationarity))),
            "inequality_multiplier_maximum": float(np.max(yi)),
            "lower_bound_multiplier_minimum": float(np.min(yl)),
            "upper_bound_multiplier_maximum": float(np.max(yu[:n_q]))}


def solve_privacy_first(cost_pair: Mapping, cuts: Sequence[Mapping],
                        d17: np.ndarray, *, time_limit_seconds: float = 60.) -> dict:
    """Maximize common AB/SEX fitted-risk gain with unchanged registered caps."""
    cost, bank, q_ref = _problem(cost_pair, cuts, d17)
    if not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("positive finite LP time limit required")
    witness = replay_privacy_first(q_ref, 0., cost_pair, cuts, q_ref)
    if not witness["feasible"]:
        raise ValueError("D17,tau=0 witness fails frozen calibration; inspect input bank")
    phase = phase_one_privacy_first(cost_pair, cuts, q_ref,
                                    time_limit_seconds=min(30., time_limit_seconds))
    if phase["status"] != "OPTIMAL" or phase["minimum_common_violation"] is None:
        return {"status": "PHASE_ONE_UNRESOLVED", "witness": witness,
                "phase_one": phase, "Q": None, "tau": None}
    if phase["minimum_common_violation"] > TOL:
        raise RuntimeError("phase I contradicts independently verified D17 witness")
    aub, bub, aeq, beq, bounds = _matrices(cost, bank, q_ref, phase_one=False)
    objective = np.zeros(aub.shape[1]); objective[-1] = -1.
    raw = linprog(objective, A_ub=aub, b_ub=bub, A_eq=aeq, b_eq=beq,
                  bounds=bounds, method="highs", options={"time_limit": time_limit_seconds})
    q = raw.x[:-1].reshape(q_ref.shape) if raw.x is not None else None
    tau = float(raw.x[-1]) if raw.x is not None else None
    replay = replay_privacy_first(q, tau, cost_pair, cuts, q_ref) if q is not None else None
    residual = None
    if raw.x is not None:
        residual = {"maximum_inequality_violation": float(np.max(np.maximum(aub@raw.x-bub, 0.))),
                    "maximum_simplex_residual": float(np.max(np.abs(aeq@raw.x-beq))),
                    "minimum_entry": float(np.min(raw.x))}
    dual = _dual_report(raw, aub, bub, aeq, beq, q_ref.size, objective)
    status = "OPTIMAL" if raw.status == 0 and replay and replay["feasible"] else (
        "FEASIBLE_UNRESOLVED" if replay and replay["feasible"] else "UNRESOLVED")
    if status != "OPTIMAL":
        dual = None
    return {"status": status, "Q": q if replay and replay["feasible"] else None,
            "tau": tau if replay and replay["feasible"] else None,
            "witness": witness, "phase_one": phase, "replay": replay,
            "solver_status": int(raw.status), "solver_message": raw.message,
            "solver_residual": residual,
            "dual_tau_upper_bound": dual["tau_upper_bound"] if dual else None,
            "dual_gap": dual["tau_upper_bound"]-tau if dual else None,
            "dual_residual": dual,
            "time_limit_seconds": float(time_limit_seconds),
            "interpretation": "finite fixed-bank fitted-risk diagnostic; no independent or population guarantee"}


def load_frozen_inputs(branch: str, center_dir: Path, controls_dir: Path, *,
                       anchor: int, delta: float, a_center_dir: Path | None = None):
    """Load a completed center and its hash-pinned D17 control without fitting."""
    from . import fit_controls
    problem = fit_controls.load_final_problem(branch, center_dir, anchor=anchor,
                                              delta=delta, a_center_dir=a_center_dir)
    root = controls_dir.resolve()
    complete_path = root/"COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    center_source = problem["source"]
    control_source = complete.get("source")
    provenance_keys = {"d17_member_sha256", "historical_q_member_sha256",
                       "fit_controls_source_sha256", "controls_source_sha256"}
    source_matches = (isinstance(control_source, dict)
                      and set(control_source) == set(center_source) | provenance_keys
                      and all(control_source[key] == value
                              for key, value in center_source.items())
                      and all(isinstance(control_source[key], str)
                              and len(control_source[key]) == 64
                              and all(ch in "0123456789abcdef"
                                      for ch in control_source[key])
                              for key in provenance_keys))
    if ("private" not in root.parts or complete.get("status") != "COMPLETE"
            or complete.get("artifact_sha256") != fit_controls._inventory(root)
            or not source_matches):
        raise ValueError("controls receipt or frozen center source differs")
    path = root/"channels/D17/Q.npz"
    with np.load(path, allow_pickle=False) as archive:
        q_ref = archive["Q"].copy()
    if (fit_controls._array_sha(q_ref) != complete.get("q_ref_sha256")
            or fit_controls._bank_sha(.5*(problem["cost_pair"]["U"]+
                                       problem["cost_pair"]["W"]),
                                      problem["cuts"]) != complete.get("fixed_bank_sha256")):
        raise ValueError("D17 or final frozen bank differs from controls receipt")
    return problem["cost_pair"], problem["cuts"], q_ref, center_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("A", "B"), required=True)
    parser.add_argument("--center-dir", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--a-center-dir", type=Path)
    parser.add_argument("--anchor", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--delta", type=float, choices=(0., .001, .003), required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=60.)
    parser.add_argument("--witness-only", action="store_true")
    args = parser.parse_args(argv)
    cost, cuts, d17, source = load_frozen_inputs(
        args.branch, args.center_dir, args.controls_dir, anchor=args.anchor,
        delta=args.delta, a_center_dir=args.a_center_dir)
    if args.witness_only:
        report = {"status": "WITNESS_REPLAY", "source": source,
                  "witness": replay_privacy_first(d17, 0., cost, cuts, d17)}
    else:
        report = solve_privacy_first(cost, cuts, d17,
                                     time_limit_seconds=args.time_limit_seconds)
        report.pop("Q", None)  # no channel artifact or person-level output from CLI
        report["source"] = source
        report["witness"]["cut_slacks"] = {"count": len(cuts),
             "minimum": report["witness"]["minimum_cut_slack"]}
        if report.get("replay"):
            report["replay"]["cut_slacks"] = {"count": len(cuts),
                 "minimum": report["replay"]["minimum_cut_slack"]}
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
