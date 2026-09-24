"""Re-solve a frozen finite bank for a same-cost LP/MILP certificate.

This calculation reuses fitted coefficients and channels. It fits no encoder,
decoder, attack, or release, and it opens no person-level assessment pool.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import solver
from . import controls, fit_controls

_array_sha = fit_controls._array_sha
_file_sha = fit_controls._file_sha


def certify_fixed_bank(cost: np.ndarray, cuts: Sequence[Mapping],
                       selected_q: np.ndarray, milp: Mapping,
                       milp_q: np.ndarray | None) -> dict:
    """Compare a freshly solved LP upper bound to a valid MILP lower bound.

    Both use exactly these coefficient matrices, floors, and frozen decoder
    cost. A result is numerical, not interval-certified or an oracle claim.
    """
    cost = np.asarray(cost, dtype=np.float64)
    bank = controls.make_bank(cost, cuts)
    declared_cut_ids = milp.get("fixed_bank_cut_ids")
    if declared_cut_ids is not None and list(declared_cut_ids) != list(bank.ids):
        raise ValueError("MILP fixed-bank cut IDs differ from final frozen bank")
    selected_q = controls._channel(selected_q)
    if selected_q.shape != cost.shape:
        raise ValueError("selected channel differs from final bank dimensions")
    selected = solver.replay_p1(selected_q, cost, cuts)
    if (selected["maximum_cut_violation"] > solver.PRIMAL_TOL or
            selected["simplex_residual"] > solver.SIMPLEX_TOL):
        raise ValueError("selected channel is infeasible on the final frozen bank")
    lp = solver.solve_p1(cost, cuts)
    if not lp.get("feasible") or lp.get("Q") is None:
        raise ValueError("final-bank LP did not return a replayed feasible channel")
    lp_q = controls._channel(lp["Q"])
    lp_replay = solver.replay_p1(lp_q, cost, cuts)
    if (lp_replay["maximum_cut_violation"] > solver.PRIMAL_TOL or
            lp_replay["bank_sha256"] != selected["bank_sha256"]):
        raise ValueError("LP replay fails the selected channel's frozen bank")
    lp_dual = lp.get("dual_lower_bound")
    if lp_dual is not None and float(lp_dual) > lp_replay["objective"] + 1e-7:
        raise ValueError("LP dual lower bound exceeds feasible LP objective")

    lower = milp.get("lower_bound")
    lower = float(lower) if lower is not None else None
    if lower is not None and not np.isfinite(lower):
        raise ValueError("MILP lower bound is nonfinite")
    incumbent_valid = bool(milp.get("incumbent_valid"))
    incumbent = None
    if incumbent_valid:
        if milp_q is None:
            raise ValueError("valid MILP incumbent requires its pinned channel")
        milp_q = controls._channel(milp_q, deterministic=True)
        if milp_q.shape != cost.shape:
            raise ValueError("MILP incumbent dimensions differ from final bank")
        incumbent = controls.replay(milp_q, bank, deterministic=True)
        claimed = milp.get("incumbent_objective")
        if (not incumbent["feasible"] or claimed is None or
                abs(incumbent["objective"] - float(claimed)) > 1e-7):
            raise ValueError("MILP incumbent fails frozen-bank objective or feasibility replay")
        if lower is not None and lower > incumbent["objective"] + 1e-7:
            raise ValueError("MILP lower bound exceeds valid incumbent objective")
    elif milp_q is not None:
        raise ValueError("unvalidated MILP channel cannot enter certificate")

    gap = lower - lp_replay["objective"] if lower is not None else None
    status = ("DETERMINISTIC_LOWER_BOUND_UNRESOLVED" if lower is None else
              "NUMERICAL_FIXED_BANK_INTEGRALITY_GAP" if gap > 1e-7 else
              "NO_CERTIFIED_FIXED_BANK_GAP")
    return {
        "schema": "pcrl-adaptive-final-bank-certificate-v1",
        "status": status,
        "scope": "fixed bank and frozen decoder only",
        "oracle_error": "UNRESOLVED",
        "population_guarantee": False,
        "cost_array_sha256": _array_sha(cost),
        "fixed_bank_sha256": fit_controls._bank_sha(cost, cuts),
        "lp": {"status": lp["status"], "objective": lp_replay["objective"],
               "dual_lower_bound": lp_dual, "fixed_bank_gap": lp.get("fixed_bank_gap"),
               "maximum_cut_violation": lp_replay["maximum_cut_violation"],
               "simplex_residual": lp_replay["simplex_residual"],
               "bank_sha256": lp_replay["bank_sha256"],
               "Q_array_sha256": _array_sha(lp_q)},
        "selected_channel": {"objective": selected["objective"],
                             "maximum_cut_violation": selected["maximum_cut_violation"],
                             "bank_sha256": selected["bank_sha256"],
                             "Q_array_sha256": _array_sha(selected_q)},
        "milp": {"solver_status_code": milp.get("status_code"),
                 "lower_bound": lower,
                 "incumbent_valid": incumbent_valid,
                 "incumbent_objective": incumbent["objective"] if incumbent else None,
                 "incumbent_Q_array_sha256": _array_sha(milp_q) if incumbent else None,
                 "incumbent_maximum_cut_violation":
                     incumbent["maximum_cut_violation"] if incumbent else None,
                 "bank_sha256": lp_replay["bank_sha256"]},
        "deterministic_lower_bound_minus_stochastic_feasible": gap,
        "_lp_Q_private": lp_q,
    }


def certify_from_completed_controls(branch: str, center_dir: str | Path,
                                    controls_dir: str | Path, output_dir: str | Path,
                                    *, anchor: int, delta: float,
                                    a_center_dir: str | Path | None = None) -> dict:
    """Verify immutable inputs and write a separate private certificate unit."""
    controls_root = Path(controls_dir).resolve()
    output_root = Path(output_dir).resolve()
    if "private" not in controls_root.parts or "private" not in output_root.parts:
        raise ValueError("control and certificate directories must be private")
    problem = fit_controls.load_final_problem(branch, center_dir, anchor=anchor,
                                              delta=delta, a_center_dir=a_center_dir)
    complete_path = controls_root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    if (complete.get("schema") != "pcrl-adaptive-controls-v1" or
            complete.get("status") != "COMPLETE" or
            complete.get("artifact_sha256") != fit_controls._inventory(controls_root)):
        raise ValueError("control completion or artifact inventory differs")
    source = complete.get("source", {})
    if any(source.get(key) != value for key, value in problem["source"].items()):
        raise ValueError("control source differs from selected center and bank")
    cost_pair = problem["cost_pair"]
    cost = .5*(np.asarray(cost_pair["U"])+np.asarray(cost_pair["W"]))
    if (complete.get("cost_pair_sha256") != {
            "U": _array_sha(cost_pair["U"]), "W": _array_sha(cost_pair["W"])} or
            complete.get("fixed_bank_sha256") !=
            fit_controls._bank_sha(cost, problem["cuts"])):
        raise ValueError("control final cost or cut bank differs from frozen center")
    controls_meta = json.loads((controls_root / "CONTROLS.json").read_text())
    if controls_meta.get("schema") != 1:
        raise ValueError("control registry schema differs")
    milp = controls_meta["milp"]
    if (milp.get("lower_bound") != complete.get("mip_lower_bound") or
            milp.get("incumbent_objective") != complete.get("mip_incumbent_objective") or
            bool(milp.get("incumbent_valid")) != bool(complete.get("mip_incumbent_valid"))):
        raise ValueError("MILP receipt and registry differ")
    milp_q = None
    if milp.get("incumbent_valid"):
        entry = controls_meta["controls"]["MILP"]
        q_path = (controls_root / entry["relative_path"]).resolve()
        if not q_path.is_relative_to(controls_root) or _file_sha(q_path) != entry["Q_file_sha256"]:
            raise ValueError("MILP channel file hash or path differs")
        with np.load(q_path, allow_pickle=False) as archive:
            milp_q = archive["Q"].copy()
        if _array_sha(milp_q) != entry["Q_array_sha256"]:
            raise ValueError("MILP channel array hash differs")
    result = certify_fixed_bank(cost, problem["cuts"],
                                problem["selected_channel"], milp, milp_q)
    lp_q = result.pop("_lp_Q_private")
    result["source"] = {"branch": branch, "anchor": anchor, "delta": delta,
                        "center_complete_sha256": problem["source"]["center_sha256"],
                        "control_complete_sha256": _file_sha(complete_path),
                        "control_registry_sha256": _file_sha(controls_root / "CONTROLS.json"),
                        "selected_bank_sha256": problem["source"]["selected_bank_sha256"],
                        "certifier_source_sha256": _file_sha(Path(__file__))}
    if output_root.exists() and any(output_root.iterdir()):
        prior = json.loads((output_root / "COMPLETE.json").read_text())
        if prior.get("artifact_sha256") != fit_controls._inventory(output_root):
            raise ValueError("prior certificate artifact inventory differs")
        if prior.get("source") != result["source"] or prior.get("certificate") != result:
            raise ValueError("prior certificate differs from exact frozen inputs")
        return prior
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_root, 0o700)
    fit_controls._save_q(output_root / "LP_Q.npz", lp_q)
    fit_controls._write_json(output_root / "CERTIFICATE.json", result)
    receipt = {"schema": "pcrl-adaptive-final-bank-certificate-receipt-v1",
               "status": "COMPLETE", "source": result["source"],
               "certificate": result,
               "artifact_sha256": fit_controls._inventory(output_root)}
    fit_controls._write_json(output_root / "COMPLETE.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=("A", "B"), required=True)
    parser.add_argument("--anchor", choices=(0, 1, 2), type=int, required=True)
    parser.add_argument("--delta", choices=(0., .001, .003), type=float, required=True)
    parser.add_argument("--center-dir", required=True)
    parser.add_argument("--controls-dir", required=True)
    parser.add_argument("--a-center-dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    receipt = certify_from_completed_controls(
        args.branch, args.center_dir, args.controls_dir, args.output_dir,
        anchor=args.anchor, delta=args.delta, a_center_dir=args.a_center_dir)
    print(json.dumps({"status": receipt["certificate"]["status"],
                      "complete_sha256": _file_sha(Path(args.output_dir) / "COMPLETE.json")},
                     sort_keys=True))
    return receipt


if __name__ == "__main__":
    main()
