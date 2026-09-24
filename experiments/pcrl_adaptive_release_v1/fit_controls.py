"""Immutable same-bank deterministic, gradient and simple 17-token controls.

This module fits only fixed finite programmes supplied by a frozen A/B center.
It neither opens assessment labels nor selects a control on assessment results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import controls


PUBLISH_RATES = (.50, .75, .90, 1.00)


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as stream:
        stream.write(encoded)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _save_q(path: Path, q: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    np.savez_compressed(path, Q=np.asarray(q, dtype=np.float64))
    os.chmod(path, 0o600)


def _inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): _file_sha(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "COMPLETE.json"}


def _bank_sha(cost: np.ndarray, cuts: Sequence[Mapping]) -> str:
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(_array_sha(cost)))
    for cut in cuts:
        metadata = {key: value for key, value in cut.items() if key != "coeff"}
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        digest.update(bytes.fromhex(_array_sha(cut["coeff"])))
    return digest.hexdigest()


def load_final_problem(branch: str, center_dir: str | Path, *, anchor: int,
                       delta: float, a_center_dir: str | Path | None = None) -> dict:
    """Load the selected A/B channel and its last frozen bank, checking hashes.

    The returned cost is the *final* bank cost. An earlier selected channel is
    replayed against that bank before it can enter a matched comparison.
    """
    from experiments.pcrl_task_aligned_cuts_v1 import solver
    from . import fit_a, fit_b, refinement

    root = Path(center_dir).resolve()
    if branch not in ("A", "B") or anchor not in (0, 1, 2) or delta not in (0., .001, .003):
        raise ValueError("registered branch, anchor and allowance required")
    if "private" not in root.parts:
        raise ValueError("frozen center must be in a private output path")
    if branch == "A":
        selected = fit_b.load_a_selected(root, anchor=anchor, delta=delta)
        return {"cost_pair": {key: selected["cost"][key] for key in ("U", "W")},
                "cuts": selected["cuts"], "selected_channel": selected["Q"],
                "parent_of_leaf": np.arange(32, dtype=np.int64),
                "source": {"branch": "A", "anchor": anchor, "delta": delta,
                           "center_sha256": selected["complete_receipt_sha256"],
                           "selected_bank_sha256": selected["selected_bank_sha256"]}}
    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    if (complete.get("schema") != "pcrl-adaptive-B-center-v1" or
            complete.get("anchor") != anchor or complete.get("delta") != delta or
            complete.get("artifact_sha256") != _inventory(root)):
        raise ValueError("B center input or artifact hash inventory differs")
    if complete.get("status") == "SUPPORT_LIMITED_ALIAS_A":
        if a_center_dir is None:
            raise ValueError("B support-limited alias requires its frozen A center")
        a = load_final_problem("A", a_center_dir, anchor=anchor, delta=delta)
        channel_path = (root / complete["selected_channel_relative"]).resolve()
        if not channel_path.is_relative_to(root):
            raise ValueError("B alias channel path escaped private center")
        with np.load(channel_path, allow_pickle=False) as archive:
            alias_q = archive["Q"].copy()
        if not np.array_equal(alias_q, a["selected_channel"]):
            raise ValueError("B alias channel differs from frozen A release")
        return {**a, "source": {"branch": "B_ALIAS_A", "anchor": anchor,
                                "delta": delta, "center_sha256": _file_sha(complete_path),
                                "a_center_sha256": a["source"]["center_sha256"],
                                "selected_bank_sha256": a["source"]["selected_bank_sha256"]}}
    if complete.get("status") != "COMPLETE":
        raise ValueError("B center has no complete refined channel")
    partition_path = root / "PARTITION.json"
    if _file_sha(partition_path) != complete.get("partition_sha256"):
        raise ValueError("B partition hash differs")
    partition = refinement.NestedPartition.from_record(json.loads(partition_path.read_text()))
    selected_meta = json.loads((root / "SELECTED.json").read_text())
    selection_path = root / "FINAL_BANK_SELECTION.json"
    final_selection = json.loads(selection_path.read_text())
    selected_round = selected_meta["selected_round"]
    if (selected_round != complete["selected_round"] or
            selected_round != final_selection["selected_round"] or
            _file_sha(selection_path) != selected_meta["final_bank_selection_sha256"] or
            selected_meta["final_bank_sha256"] != complete["final_bank_sha256"] or
            final_selection["final_bank_sha256"] != complete["final_bank_sha256"] or
            final_selection["final_bank_checks"][selected_round]["feasible"] is not True):
        raise ValueError("B selected round or final bank hash differs")
    selected_path = (root / selected_meta["selected_channel_relative"]).resolve()
    if (not selected_path.is_relative_to(root) or
            selected_meta["selected_channel_relative"] !=
            complete["selected_channel_relative"]):
        raise ValueError("B selected channel path differs")
    with np.load(selected_path, allow_pickle=False) as archive:
        q = archive["Q"].copy()
    selected_record = complete["rounds"][selected_round]
    if (_file_sha(selected_path) != selected_record["channel_file_sha256"] or
            fit_b._array_sha256(q) != selected_record["channel_array_sha256"]):
        raise ValueError("B selected channel hash differs")
    final_round = len(complete["rounds"])-1
    final_root = root / f"round_r{final_round:02d}"
    final_record = complete["rounds"][final_round]
    if final_record["bank_sha256"] != complete["final_bank_sha256"]:
        raise ValueError("B final round bank differs")
    with np.load(final_root / "COSTS.npz", allow_pickle=False) as archive:
        cost_pair = {"U": archive["cost_U"].copy(), "W": archive["cost_W"].copy()}
    bank_meta = json.loads((final_root / "BANK.json").read_text())
    cut_path = final_root / "CUTS.npz"
    if (_file_sha(cut_path) != bank_meta["cut_coefficients_sha256"] or
            bank_meta["bank_sha256"] != complete["final_bank_sha256"]):
        raise ValueError("B final cut coefficient hash differs")
    with np.load(cut_path, allow_pickle=False) as archive:
        cuts = [{**meta, "coeff": archive[f"cut_{i:04d}"].copy()}
                for i, meta in enumerate(bank_meta["cuts"])]
    cost = .5*(cost_pair["U"]+cost_pair["W"])
    replay = solver.replay_p1(q, cost, cuts)
    if (q.shape != (len(partition.parents), 17) or
            replay["bank_sha256"] != complete["final_bank_sha256"] or
            replay["maximum_cut_violation"] > solver.PRIMAL_TOL or
            replay["simplex_residual"] > solver.SIMPLEX_TOL):
        raise ValueError("B selected channel fails final-bank replay")
    return {"cost_pair": cost_pair, "cuts": cuts, "selected_channel": q,
            "parent_of_leaf": partition.parent_of_leaf,
            "source": {"branch": "B", "anchor": anchor, "delta": delta,
                       "center_sha256": _file_sha(complete_path),
                       "selected_bank_sha256": complete["final_bank_sha256"],
                       "partition_sha256": complete["partition_sha256"],
                       "amendment_01_sha256": complete["amendment_01_sha256"]}}


def fit_controls_from_frozen(
    cost_pair: Mapping[str, np.ndarray], cuts: Sequence[Mapping],
    q_ref: np.ndarray, historical_q: np.ndarray, output_dir: str | Path, *,
    source: Mapping, milp_seconds: float = 120.0,
    heuristic_seconds: float = 60.0, gradient_steps: int = 1000,
    gradient_penalties: Sequence[float] = (10., 100., 1000.),
    gradient_seeds: Sequence[int] = (17, 29),
) -> dict:
    """Fit and replay all declared controls on one frozen decoder/cut bank.

    A completed unit is reused only after exact input and artifact verification.
    A partial unit remains preserved for explicit technical recovery.
    """
    root = Path(output_dir)
    if "private" not in root.parts:
        raise ValueError("fitted control channels require a private output path")
    if set(cost_pair) != {"U", "W"}:
        raise ValueError("separate U and PWGTP task costs required")
    cost_u = np.asarray(cost_pair["U"], dtype=np.float64)
    cost_w = np.asarray(cost_pair["W"], dtype=np.float64)
    if cost_u.shape != cost_w.shape or not np.isfinite(cost_u).all() or not np.isfinite(cost_w).all():
        raise ValueError("invalid matched task cost matrices")
    cost = .5*(cost_u+cost_w)
    bank = controls.make_bank(cost, cuts)
    q_ref = controls._channel(q_ref, deterministic=True)
    historical_q = controls._channel(historical_q)
    if q_ref.shape != cost.shape or historical_q.shape != cost.shape:
        raise ValueError("reference/historical release state or token contract differs")
    witness = controls.replay(q_ref, bank, deterministic=True)
    if not witness["feasible"]:
        raise ValueError("constructive D17 witness violates frozen reference bank")
    if not isinstance(source, Mapping) or not source:
        raise ValueError("hash-pinned frozen center source required")
    settings = {"milp_seconds": float(milp_seconds),
                "heuristic_seconds": float(heuristic_seconds),
                "gradient_steps": int(gradient_steps),
                "gradient_penalties": [float(x) for x in gradient_penalties],
                "gradient_seeds": [int(x) for x in gradient_seeds],
                "publish_rates": list(PUBLISH_RATES),
                "gradient_initializations": ["D17", "D_task"],
                "constant_token_rule": "registered fixed existing token 0"}
    expected = {"schema": "pcrl-adaptive-controls-v1", "source": dict(source),
                "settings": settings,
                "cost_pair_sha256": {"U": _array_sha(cost_u), "W": _array_sha(cost_w)},
                "fixed_bank_sha256": _bank_sha(cost, cuts),
                "q_ref_sha256": _array_sha(q_ref),
                "historical_q_sha256": _array_sha(historical_q),
                "reference_witness_maximum_cut_violation": witness["maximum_cut_violation"]}
    complete_path = root / "COMPLETE.json"
    if complete_path.exists():
        receipt = json.loads(complete_path.read_text())
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("completed control unit source or scientific inputs differ")
        if receipt.get("artifact_sha256") != _inventory(root):
            raise ValueError("completed control artifact hash inventory differs")
        return receipt
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("partial control unit retained for technical review")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    _write_json(root / "INPUTS.json", expected)
    np.savez_compressed(root / "FROZEN_BANK.npz", cost_U=cost_u, cost_W=cost_w,
                        **{f"cut_{i:04d}": cut["coeff"] for i, cut in enumerate(cuts)})
    os.chmod(root / "FROZEN_BANK.npz", 0o600)
    _write_json(root / "BANK.json", {"schema": 1,
        "cuts": [{key: value for key, value in cut.items() if key != "coeff"}
                 for cut in cuts],
        "coefficient_archive_sha256": _file_sha(root / "FROZEN_BANK.npz"),
        "fixed_bank_sha256": expected["fixed_bank_sha256"]})

    records = {}
    aliases = {}

    def add(identifier: str, q: np.ndarray, *, family: str, detail: dict | None = None,
            deterministic: bool = False) -> None:
        if identifier in records:
            raise ValueError("duplicate immutable control ID")
        q = controls._channel(q, deterministic=deterministic)
        report = controls.replay(q, bank, deterministic=deterministic)
        q_sha = _array_sha(q)
        alias = aliases.get(q_sha)
        aliases.setdefault(q_sha, identifier)
        path = root / "channels" / identifier / "Q.npz"
        _save_q(path, q)
        records[identifier] = {"family": family, "detail": detail or {},
                               "Q_array_sha256": q_sha,
                               "Q_file_sha256": _file_sha(path),
                               "relative_path": str(path.relative_to(root)),
                               "alias_of": alias, "replay": report}

    task_only = controls.rowwise_minimizer(cost)
    add("D17", q_ref, family="constructive_reference", deterministic=True)
    add("historical_Q", historical_q, family="historical_context")
    add("D_task", task_only, family="final_frozen_decoder_rowwise_minimum",
        detail={"scope": "fixed decoder and coefficient people; not a separately fitted raw-input task predictor"},
        deterministic=True)
    constant_token = 0
    for base_name, base in (("D17", q_ref), ("D_task", task_only)):
        for publish in PUBLISH_RATES:
            suffix = f"{round(100*publish):03d}"
            add(f"{base_name}_constant_replace_publish_{suffix}",
                controls.constant_replacement(base, replace_probability=1-publish,
                                              token=constant_token),
                family="constant_replacement", detail={"publish_probability": publish,
                "replacement_probability": 1-publish, "constant_token": constant_token})
            add(f"{base_name}_randomized_response_publish_{suffix}",
                controls.randomized_response(base, publish_probability=publish),
                family="randomized_response", detail={"publish_probability": publish})

    milp = controls.solve_deterministic_p1(cost, cuts,
        time_limit_seconds=milp_seconds)
    if milp["incumbent_valid"]:
        add("MILP", milp["Q"], family="same_bank_deterministic_milp",
            deterministic=True)
    heuristic = controls.deterministic_coordinate_search(cost, cuts,
        seconds=heuristic_seconds, seed=20260924,
        starts=(q_ref, task_only))
    if heuristic["replay"]["feasible"]:
        add("deterministic_search", heuristic["Q"],
            family="bounded_deterministic_search", deterministic=True)
    gradient = controls.gradient_control_grid(cost, cuts,
        steps_per_run=gradient_steps, seeds=gradient_seeds,
        penalties=gradient_penalties,
        initializations={"D17": q_ref, "D_task": task_only})
    gradient_receipts = []
    for index, item in enumerate(gradient):
        identifier = f"GRADIENT_{index:03d}"
        add(identifier, item["Q"], family="matched_exact_token_gradient",
            detail={key: item[key] for key in ("steps", "learning_rate", "penalty",
                   "seed", "dual_learning_rate", "initialization",
                   "initialization_smoothing")})
        gradient_receipts.append({"id": identifier,
                                  "feasible": records[identifier]["replay"]["feasible"]})
    _write_json(root / "CONTROLS.json", {"schema": 1, "controls": records,
        "milp": {key: value for key, value in milp.items()
                 if key not in ("Q", "replay")},
        "heuristic": {key: value for key, value in heuristic.items()
                      if key not in ("Q", "replay")},
        "gradient_runs": gradient_receipts,
        "selection_status": "UNSELECTED_INNER_AUDIT_REQUIRED",
        "external_oracle_error": "UNRESOLVED",
        "gradient_comparator_scope": "fixed bank only; fresh responses, decoder adaptation and common inner selection pending"})
    receipt = {**expected, "status": "COMPLETE",
               "control_ids": list(records),
               "constant_token": constant_token,
               "mip_lower_bound": milp["lower_bound"],
               "mip_incumbent_objective": milp["incumbent_objective"],
               "mip_incumbent_valid": bool(milp["incumbent_valid"]),
               "artifact_sha256": _inventory(root)}
    _write_json(complete_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True, choices=("A", "B"))
    parser.add_argument("--anchor", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--delta", required=True, type=float, choices=(0., .001, .003))
    parser.add_argument("--index", required=True, help="pinned private 2018 input index")
    parser.add_argument("--center-dir", required=True, help="completed frozen A/B center")
    parser.add_argument("--a-center-dir", help="required only for support-limited B alias")
    parser.add_argument("--output-dir", required=True, help="new private control unit")
    parser.add_argument("--milp-seconds", type=float, default=120.)
    parser.add_argument("--heuristic-seconds", type=float, default=60.)
    parser.add_argument("--gradient-steps", type=int, default=1000)
    args = parser.parse_args(argv)

    from experiments.pcrl_task_aligned_cuts_v1 import data
    from . import refinement

    index_value = data.index(args.index)
    problem = load_final_problem(args.branch, args.center_dir,
                                 anchor=args.anchor, delta=args.delta,
                                 a_center_dir=args.a_center_dir)
    d17 = data.load_map(index_value, args.anchor, "D17")
    historical_q = data.load_map(index_value, args.anchor, "Q")
    parent_of_leaf = np.asarray(problem["parent_of_leaf"])
    d17 = refinement.copy_parent_kernel(d17, parent_of_leaf)
    historical_q = refinement.copy_parent_kernel(historical_q, parent_of_leaf)
    source = {**problem["source"],
              "d17_member_sha256": data.member_record(index_value, args.anchor,
                  "map", "D17", "Q.npz")["sha256"],
              "historical_q_member_sha256": data.member_record(index_value, args.anchor,
                  "map", "Q", "Q.npz")["sha256"],
              "fit_controls_source_sha256": _file_sha(Path(__file__)),
              "controls_source_sha256": _file_sha(Path(controls.__file__))}
    result = fit_controls_from_frozen(
        problem["cost_pair"], problem["cuts"], d17, historical_q,
        args.output_dir, source=source, milp_seconds=args.milp_seconds,
        heuristic_seconds=args.heuristic_seconds,
        gradient_steps=args.gradient_steps)
    print(json.dumps({"status": result["status"],
                      "control_count": len(result["control_ids"]),
                      "fixed_bank_sha256": result["fixed_bank_sha256"],
                      "complete_sha256": _file_sha(Path(args.output_dir) / "COMPLETE.json")},
                     sort_keys=True))
    return result


if __name__ == "__main__":
    main()
