"""Aggregate-only replay of frozen 2018 PCRL development objects.

This module never fits or writes a channel, encoder, task probe, or attacker.
The four-cell service partition is loaded from the historical private archive;
it cannot alter Q, D17, or the confirmatory evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments.pcrl_task_directed_release_v1.encoding import ServicePartitions
from experiments.pcrl_task_directed_release_v1.finite import joint_table, privacy_matrix
from .math import affine_dual_certificate, cmi, information_radius_bound

NAMES = {"Q": "T0_L_0.01_a17", "D17": "T0_U_unconstrained_a17",
         "D33": "T0_U_unconstrained_a33"}
ROLES = [f"{v}/{s}/{w}" for v in ("A", "AB") for s in ("SEX", "RAC1P") for w in ("U", "W")]
LOCAL = [r for r in ROLES if r.startswith("A/")]
N_S = {"SEX": 2, "RAC1P": 9}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def _valid(s: np.ndarray, ns: int) -> np.ndarray:
    return np.isfinite(s) & (s == np.floor(s)) & (s >= 0) & (s < ns)


def _laws(prepared: dict, pool_name: str, partition: ServicePartitions) -> tuple[dict, dict]:
    pool = prepared["ctx"]["pools"][pool_name]
    code = np.asarray(prepared["encoded"][pool_name]["codes"]["T0"], int)
    weight = np.asarray(pool["weights"], float)
    ca, cab = partition.assign(pool["ha"], pool["hb"])
    contexts = {"A": (ca, len(partition.centers)), "AB": (cab, 2*len(partition.centers))}
    laws, support = {}, {}
    for protected, ns in N_S.items():
        s = np.asarray(pool["labels"][protected]); valid = _valid(s, ns)
        for view, (context, nc) in contexts.items():
            counts = np.zeros((ns, nc), dtype=int)
            np.add.at(counts, (s[valid].astype(int), context[valid]), 1)
            key = f"{view}/{protected}"
            support[key] = {"valid_people": int(valid.sum()),
                            "context_class_cells": int(counts.size),
                            "empty_context_class_cells": int(np.sum(counts == 0)),
                            "minimum_nonzero_context_class_count": int(counts[counts > 0].min()),
                            "empty_protected_classes": int(np.sum(counts.sum(axis=1) == 0))}
            for suffix, w in (("U", None), ("W", weight[valid])):
                laws[f"{key}/{suffix}"] = joint_table(s[valid].astype(int), context[valid],
                     code[valid], ns, nc, 32, w)
    return laws, support


def _conditional_descriptives(prepared: dict, channels: dict[str, np.ndarray], anchor: int,
                              expected_coarse: dict[str, np.ndarray], fine_payload: dict) -> dict:
    rf = prepared["ctx"]["pools"]["representation_fit"]
    roles = prepared["roles"]
    train_rows = np.sort(np.r_[roles["teacher_fit"], roles["teacher_internal_validation"]])
    fine = fine_payload["partitions"]
    coarse = prepared["encoder"].partitions
    if len(fine.centers) != 4 or len(coarse.centers) != 2:
        raise AssertionError("historical partition sizes differ")
    if fine.b_median != coarse.b_median:
        raise AssertionError("Historical fine partition changed B median")
    result = {"fine_partition_centers_sha256": hashlib.sha256(np.asarray(fine.centers).tobytes()).hexdigest(),
              "fine_partition_teacher_codebook_rows": int(len(train_rows)), "pools": {}}
    for pool in ("representation_fit:mechanism", "attacker_validation"):
        if pool.endswith(":mechanism"):
            # Use only the historical mechanism-estimation rows; do not mix teacher rows.
            subset = np.asarray(roles["mechanism"], int)
            data = {"ctx": {"pools": {"mechanism": {k: (v[subset] if isinstance(v, np.ndarray) else
                    {label: a[subset] for label, a in v.items()}) for k, v in rf.items()}}},
                    "encoded": {"mechanism": {"codes": {"T0":
                               prepared["encoded"]["representation_fit"]["codes"]["T0"][subset]}}}}
            name = "mechanism"
        else:
            data, name = prepared, pool
        result["pools"][pool] = {}
        for setting, partition in (("coarse2", coarse), ("fine4", fine)):
            laws, support = _laws(data, name, partition)
            if pool.endswith(":mechanism") and setting == "coarse2":
                maximum = max(float(np.max(np.abs(laws[key]-expected_coarse[key]))) for key in ROLES)
                if maximum > 1e-12:
                    raise AssertionError(f"reconstructed original coarse law differs: {maximum}")
            if pool.endswith(":mechanism") and setting == "fine4":
                historical = fine_payload["tables"]["T0"]["roles"]
                maximum = max(float(np.max(np.abs(laws[key]-historical[key+"/fineC"]))) for key in ROLES)
                if maximum > 1e-12:
                    raise AssertionError(f"historical fine law differs: {maximum}")
            result["pools"][pool][setting] = {
                "context_cells_A": len(partition.centers), "context_cells_AB": 2*len(partition.centers),
                "support": support,
                "cmi_nats": {name: {key: cmi(law, q) for key, law in laws.items()}
                             for name, q in channels.items()},
            }
    result["full_H"] = "unidentified: continuous service vectors have inadequate repeated joint support"
    return result


def _weighted_quantiles(x: np.ndarray, weights: np.ndarray) -> dict:
    order = np.argsort(x); values = x[order]; w = weights[order]
    cum = np.cumsum(w)/w.sum()
    return {str(p): float(values[min(np.searchsorted(cum, p), len(values)-1)])
            for p in (.0, .25, .5, .75, 1.)}


def _profile(q: np.ndarray, d: np.ndarray, cost: np.ndarray, mass: np.ndarray,
             joints: dict[str, np.ndarray]) -> dict:
    weight = mass/mass.sum()
    entropy = -np.sum(np.where(q > 0, q*np.log(np.maximum(q, 1e-300)), 0), axis=1)
    tv = .5*np.abs(q-d).sum(axis=1)
    best = np.eye(q.shape[1])[np.argmin(cost, axis=1)]
    tv_best = .5*np.abs(q-best).sum(axis=1)
    premium = np.sum(cost*(q-d), axis=1)
    sensitivity = np.array([max(np.abs(privacy_matrix(joints[key])[:, t]).sum()
                                for key in LOCAL) for t in range(len(q))])
    top = sensitivity >= np.quantile(sensitivity, .75)
    changed = tv > 1e-6
    total_tv = float(np.sum(weight*tv))
    p = {"state_count": len(q), "state_mass_total": int(mass.sum()),
         "state_mass_min": int(mass.min()), "state_mass_max": int(mass.max()),
         "unsupported_state_count": int(np.sum(mass == 0)),
         "rare_states_below_one_percent_mass": int(np.sum(weight < .01)),
         "zero_probabilities_exact": int(np.sum(q == 0)),
         "probabilities_below_1e_minus_8": int(np.sum(q < 1e-8)),
         "nondeterministic_rows_1e_minus_6": int(np.sum(q.max(axis=1) < 1-1e-6)),
         "nondeterministic_state_mass_fraction_1e_minus_6": float(np.sum(weight[q.max(axis=1) < 1-1e-6])),
         "mass_weighted_entropy_nats": float(weight@entropy),
         "mass_weighted_effective_actions": float(np.exp(weight@entropy)),
         "row_entropy_weighted_quantiles": _weighted_quantiles(entropy, weight),
         "tv_Q_D17_mass_weighted": total_tv,
         "tv_Q_D17_weighted_quantiles": _weighted_quantiles(tv, weight),
         "tv_Q_rowwise_cost_minimizer_mass_weighted": float(weight@tv_best),
         "rowwise_cost_minimizer_mass_weighted_Q_probability": float(weight@q[np.arange(len(q)), np.argmin(cost, axis=1)]),
         "changed_rows_over_1e_minus_6": int(changed.sum()),
         "fixed_objective_premium_sum": float(premium.sum()),
         "fixed_objective_positive_row_premium_sum": float(np.maximum(premium, 0).sum()),
         "fixed_objective_negative_row_premium_sum": float(np.minimum(premium, 0).sum()),
         "row_premium_quantiles": _weighted_quantiles(premium, weight),
         "top_quartile_coarse_privacy_sensitivity_state_mass_fraction": float(weight[top].sum()),
         "top_quartile_coarse_privacy_sensitivity_share_of_weighted_TV":
             float(np.sum(weight[top]*tv[top])/total_tv) if total_tv else None,
         "sensitivity_TV_weighted_pearson": float(np.corrcoef(sensitivity, tv)[0, 1]),
         "information_radius_upper_bound_nats": information_radius_bound(q, mass),
         "D17_information_radius_upper_bound_nats": information_radius_bound(d, mass)}
    return p


def _fixed_decoder_ce(prepared: dict, pool_name: str, channels: dict[str, np.ndarray]) -> dict:
    """Exact token expectation under historical fixed action decoder, no fit."""
    pool = prepared["ctx"]["pools"][pool_name]
    enc = prepared["encoded"][pool_name]
    y = np.asarray(pool["labels"]["same_residence"])
    valid = np.isin(y, (0, 1))
    weight = np.asarray(pool["weights"], float)[valid]
    code = np.asarray(enc["codes"]["T0"], int)[valid]
    out = {"valid_people": int(valid.sum()), "channels": {}}
    for name, q in channels.items():
        action = np.asarray(enc["actions"][17 if name != "D33" else 33], float)[valid]
        loss = -y[valid, None]*np.log(action)-(1-y[valid, None])*np.log1p(-action)
        expected = np.sum(q[code]*loss, axis=1)
        out["channels"][name] = {"U": float(np.mean(expected)),
                                  "W": float(np.average(expected, weights=weight))}
    out["Q_minus_D17"] = {w: out["channels"]["Q"][w]-out["channels"]["D17"][w]
                            for w in ("U", "W")}
    return out


def run(private_root: Path, finec_root: Path, output_root: Path) -> None:
    slacks, profiles, certificates, fixtures = {}, {}, {}, {"anchors": {}}
    reusable = {"source_commit": "f4bdf4cd5bf74c634feeec50aef78bff249667e4",
                "completed_prospective_evidence_commit": "5e154e5c4fdaeb23d327a0ebefe838525f1a19cb",
                "private_object_root": str(private_root), "anchors": {},
                "historical_archive_index": "results/pcrl_task_directed_release_v1/ARCHIVE_INDEX.json",
                "private_inputs_must_not_be_committed": True}
    for anchor in range(3):
        base = private_root/f"anchor_{anchor}"
        fine_path = finec_root/f"fineC_anchor_{anchor}.joblib"
        fine_payload = joblib.load(fine_path)
        prepared_path = base/"prepared.joblib"
        prepared = joblib.load(prepared_path)
        encoder = joblib.load(base/"encoder/encoder.joblib")
        if prepared["ctx"]["anchor"] != anchor:
            raise AssertionError("anchor mismatch")
        paths = {name: base/"maps"/candidate for name, candidate in NAMES.items()}
        sol = {name: joblib.load(path/"solution.joblib") for name, path in paths.items()}
        tables = {name: np.load(path/"tables.npz") for name, path in paths.items()}
        q = {name: np.asarray(value["Q"], float) for name, value in sol.items()}
        cost = np.asarray(tables["Q"]["cost"], float)
        if not np.array_equal(cost, tables["D17"]["cost"]):
            raise AssertionError("Q and D17 fixed cost matrices differ")
        if not np.array_equal(tables["Q"]["state_mass"], tables["D17"]["state_mass"]):
            raise AssertionError("Q and D17 state mass differs")
        if any(not np.array_equal(tables["Q"]["joint/"+key], tables["D17"]["joint/"+key]) for key in ROLES):
            raise AssertionError("Q and D17 empirical privacy laws differ")
        laws = {key: tables["Q"]["joint/"+key] for key in LOCAL}
        mass = np.asarray(tables["Q"]["state_mass"])
        objective = {name: float(np.sum(sol[name]["cost"]*q[name])) for name in NAMES}
        if any(abs(objective[name]-sol[name]["objective"]) > 1e-12 for name in NAMES):
            raise AssertionError("stored objectives fail replay")
        row_min = float(np.min(cost, axis=1).sum())
        d17_support = int(np.sum(np.abs(q["D17"]-np.eye(cost.shape[1])[np.argmin(cost, axis=1)]).sum(axis=1) > 1e-10))
        cmi_values = {name: {key: cmi(law, q[name]) for key, law in laws.items()}
                      for name in ("Q", "D17")}
        for key in LOCAL:
            if abs(cmi_values["Q"][key]-sol["Q"]["independent_cmi"][key]) > 1e-10:
                raise AssertionError("CMI replay differs")
        slacks[str(anchor)] = {
            "fixed_objectives_nats": objective, "Q_minus_D17_fixed_objective": objective["Q"]-objective["D17"],
            "D17_rowwise_lower_bound": row_min, "D17_minus_rowwise_lower_bound": objective["D17"]-row_min,
            "D17_rows_differing_from_first_cost_argmin": d17_support,
            "Q_budget_nats": .01,
            "local_constraints": {key: {"Q_CMI": cmi_values["Q"][key], "Q_slack": .01-cmi_values["Q"][key],
                                         "D17_CMI": cmi_values["D17"][key],
                                         "D17_excess": cmi_values["D17"][key]-.01}
                                  for key in LOCAL},
            "Q_solver_status": sol["Q"]["status"], "Q_stored_feasible": bool(sol["Q"]["feasible"]),
            "Q_stored_residuals": {k: sol["Q"]["residuals"][k]
                                    for k in ("simplex", "nonnegative", "support")},
        }
        certs = [affine_dual_certificate(cost, laws, {key: .01 for key in LOCAL}, q["Q"], e)
                 for e in (1e-6, 1e-8, 1e-10, 1e-12)]
        good = [c for c in certs if c["status"] == "dual_affine_bound"]
        certificates[str(anchor)] = max(good, key=lambda c: c["lower_bound"]) if good else certs[0]
        if certificates[str(anchor)]["status"] == "dual_affine_bound":
            certificates[str(anchor)]["conservative_numeric_guard"] = 1e-7
            certificates[str(anchor)]["conservative_lower_bound"] = certificates[str(anchor)]["lower_bound"]-1e-7
            certificates[str(anchor)]["guarded_primal_gap"] = certificates[str(anchor)]["absolute_gap"]+1e-7
        certificates[str(anchor)]["alternate_mixtures"] = [
            {"mixture": c.get("interior_mixture"), "bound": c.get("lower_bound"), "status": c["status"]} for c in certs]
        profiles[str(anchor)] = _profile(q["Q"], q["D17"], cost, mass, laws)
        profiles[str(anchor)]["conditional_laws"] = _conditional_descriptives(
            prepared, q, anchor, {key: tables["Q"]["joint/"+key] for key in ROLES}, fine_payload)
        profiles[str(anchor)]["fixed_decoder_CE"] = {
            pool: _fixed_decoder_ce(prepared, pool, q)
            for pool in ("downstream_fit", "downstream_validation", "attacker_validation")}
        reusable["anchors"][str(anchor)] = {
            "historical_fineC_tables": {"path": str(fine_path), "sha256": digest(fine_path)},
            "prepared": {"path": str(prepared_path), "sha256": digest(prepared_path)},
            "encoder": {"path": str(base/"encoder/encoder.joblib"), "sha256": digest(base/"encoder/encoder.joblib")},
            "maps": {name: {file: {"path": str(path/file), "sha256": digest(path/file)}
                             for file in ("solution.joblib", "tables.npz", "Q.npz")}
                     for name, path in paths.items()},
            "2018_split_rows_and_households": "inside prepared.joblib; never copy person rows into public reports"}
        fixtures["anchors"][str(anchor)] = {
            "encoder": {"T0_states": int(encoder.code.n_states("T0")),
                        "T0_cuts": int(len(encoder.code.cuts)),
                        "17_action_count": int(len(encoder.dictionaries[17]["offsets"])),
                        "33_action_count": int(len(encoder.dictionaries[33]["offsets"])),
                        "17_action_offset_range": [float(np.min(encoder.dictionaries[17]["offsets"])),
                                                   float(np.max(encoder.dictionaries[17]["offsets"]))],
                        "17_action_offsets_sha256": hashlib.sha256(
                            np.asarray(encoder.dictionaries[17]["offsets"]).tobytes()).hexdigest(),
                        "zero_action": int(encoder.dictionaries[17]["zero_action"]),
                        "teacher_train_rows": int(encoder.metadata["teacher_train_rows"]),
                        "teacher_internal_validation_rows": int(encoder.metadata["teacher_internal_validation_rows"]),
                        "mechanism_rows": int(encoder.metadata["mechanism_rows"])},
            "cost_matrix_sha256": hashlib.sha256(cost.tobytes()).hexdigest(),
            "state_mass_sha256": hashlib.sha256(mass.tobytes()).hexdigest(),
            "channel_sha256": {name: hashlib.sha256(q[name].tobytes()).hexdigest() for name in NAMES},
            "fixed_objective_replay": objective,
            "Q_cost_U": float(np.sum(tables["Q"]["cost_U"]*q["Q"])),
            "Q_cost_W": float(np.sum(tables["Q"]["cost_W"]*q["Q"])),
            "D17_cost_U": float(np.sum(tables["D17"]["cost_U"]*q["D17"])),
            "D17_cost_W": float(np.sum(tables["D17"]["cost_W"]*q["D17"])),
            "n_households_mechanism": len(np.unique(prepared["ctx"]["pools"]["representation_fit"]["households"][prepared["roles"]["mechanism"]])),
        }
    write_json(output_root/"CONSTRAINT_SLACKS.json", slacks)
    write_json(output_root/"RANDOMIZATION_PROFILE.json", profiles)
    write_json(output_root/"SOLVER_CERTIFICATE.json", certificates)
    write_json(output_root/"EXACT_DIAGNOSTIC_FIXTURES.json", fixtures)
    write_json(output_root/"REUSABLE_INPUTS.json", reusable)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--finec-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.private_root, arguments.finec_root, arguments.output_root)
