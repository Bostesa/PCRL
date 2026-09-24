"""PROTOCOL sections 3-5: frozen basis, one objective, every arm and control.

Objective (identical for every arm, fixed decoder and bank, coefficient_split):

    maximize t  s.t.  L_c(q) >= rho_c + t       every AB/SEX cut (U and W)
                      L_c(q) >= rho_c - 0.001   every other cut (the stored floor)
                      C_v(q) <= C_v(D17) + 0.001, v in {U, W}

Arms on the nested parameterisation q = B[T0] + sum_m A[k, m] 1{d_m = z}:
    D4 / D1   B = 0, eta = 1, one-hot rows of A; exact enumeration (5^K)
    R4 / R1   B = 0, eta = 1, A >= 0 row-stochastic; LP (HiGHS)
    NM4PF     B, A, eta free (SC `channel.solve_privacy_first`)
Controls:
    TASK_SEL4 argmin 0.5U+0.5W task over the 625 assignments, no constraint
    DET_SEL4  SC `fit_nm.deterministic_selector` replay; must equal the archive

K=1 blocks are the K=4 A blocks summed over contexts (exact).  Nothing here
loads person rows or labels; every quantity is a stored coefficient block.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from experiments.pcrl_shared_context_release_v1 import channel, fit_nm, laws
from experiments.pcrl_task_aligned_cuts_v1.solver import HIGHS_OPTIONS, PRIMAL_TOL, SIMPLEX_TOL

STUDY = "pcrl_privacy_first_selector_v1"
SCHEMA_BASIS = "pcrl-pfs-basis-v1"
SCHEMA_SOLVE = "pcrl-pfs-solve-v1"
SPEC_SCHEMA = "pcrl-sc-nested-release-v1"      # the SC loader's schema (release.load_law)
TARGET = channel.P_TARGET_ROLE                 # "AB/SEX"
TASK_ALLOWANCE = channel.TASK_ALLOWANCE         # .001
DELTA = fit_nm.DELTA                            # .001 (floors = rho - DELTA, stored in the bank)
TIE = 1e-12
POLICY_COLUMNS = ("D17", "task_only", "local_priced", "coalition_priced", "all_priced_x2")
BASIS_FILES = ("closing/TASK_BLOCKS.npz", "closing/CALIBRATED_BLOCKS.npz",
               "closing/CALIBRATED_BANK.json", "COMPLETE.json", "INPUTS.json")
RELEASES = ("D4", "R4", "NM4PF", "TASK_SEL4", "D1", "R1")


def _sha(path) -> str:
    return laws.sha256_file(path)


def _json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_default) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"{path} is write-once and differs")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(encoded)
    os.replace(temporary, path)


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value))


# ---------------------------------------------------------------------------
# frozen basis
# ---------------------------------------------------------------------------

def load_basis(nm4u_dir, bank_dir) -> dict:
    """The NM4_U closing basis (task blocks + calibrated bank) DET_SEL4 was chosen on."""
    src = Path(nm4u_dir).resolve()
    bank_root = Path(bank_dir).resolve()
    receipt = json.loads((src / "COMPLETE.json").read_text())
    if (receipt.get("variant") != "NM4_U" or receipt.get("K") != 4 or receipt.get("smoke")
            or receipt.get("status") != "COMPLETE"):
        raise ValueError("basis must be a completed, non-smoke NM4_U unit with K=4")
    if receipt.get("artifact_sha256") != fit_nm._inventory(src):
        raise ValueError("NM4_U unit inventory differs from its receipt")
    if tuple(receipt["policy_columns"]) != POLICY_COLUMNS:
        raise ValueError("policy columns differ from the registered bank")
    if _sha(bank_root / "BANK_COMPLETE.json") != receipt["bank_manifest_sha256"]:
        raise ValueError("bank manifest differs from the NM4_U receipt")
    closing = src / "closing"
    with np.load(closing / "TASK_BLOCKS.npz", allow_pickle=False) as t:
        cost = {v: {"B": t[f"{v}_B"].copy(), "A": t[f"{v}_A"].copy()} for v in channel.WEIGHTINGS}
    meta = json.loads((closing / "CALIBRATED_BANK.json").read_text())
    cuts = fit_nm._load_cut_blocks(closing / "CALIBRATED_BLOCKS.npz", meta["cuts"])
    if channel.nested_bank_sha256(cuts) != meta["bank_sha256"]:
        raise ValueError("calibrated bank hash differs from its stored hash")
    for cut in cuts:
        if abs(float(cut["floor"]) - (float(cut["rho"]) - DELTA)) > 1e-12:
            raise ValueError(f"cut {cut['id']}: floor is not rho - delta")
    roles = sorted({cut["role"] for cut in cuts})
    if TARGET not in roles:
        raise ValueError("bank has no AB/SEX cut")
    manifest = json.loads((bank_root / "BANK_COMPLETE.json").read_text())
    return {"anchor": int(receipt["anchor"]), "K": 4, "names": list(POLICY_COLUMNS),
            "cost": cost, "cuts": cuts, "nm4u_dir": src, "bank_dir": bank_root,
            "bank_manifest": manifest, "bank_sha256": meta["bank_sha256"],
            "files_sha256": {name: _sha(src / name) for name in BASIS_FILES},
            "bank_manifest_sha256": receipt["bank_manifest_sha256"],
            "roles": roles, "cut_count": len(cuts),
            "cut_count_by_role": {r: sum(c["role"] == r for c in cuts) for r in roles}}


def collapse_k1(cost, cuts):
    """K=1 blocks: sum the K=4 context rows (weights are normalised over all rows)."""
    c1 = {v: {"B": cost[v]["B"].copy(), "A": cost[v]["A"].sum(0, keepdims=True)} for v in cost}
    k1 = [{**cut, "coeff_A": np.asarray(cut["coeff_A"]).sum(0, keepdims=True)} for cut in cuts]
    return c1, k1


def task_caps(cost, d17):
    return {v: float(np.sum(cost[v]["B"] * np.asarray(d17))) + TASK_ALLOWANCE for v in channel.WEIGHTINGS}


# ---------------------------------------------------------------------------
# objective evaluation (shared by every arm)
# ---------------------------------------------------------------------------

def evaluate(B, A, cost, cuts, caps) -> dict:
    B = np.asarray(B, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    task = {v: channel.block_value(cost[v], B, A) for v in channel.WEIGHTINGS}
    target, guard, t_by_weighting = {}, {}, {}
    for cut in cuts:
        value = float(np.sum(cut["coeff_B"] * B) + np.sum(cut["coeff_A"] * A))
        if cut["role"] == TARGET:
            target[cut["id"]] = value - float(cut["rho"])
            v = cut["weighting"]
            t_by_weighting[v] = min(t_by_weighting.get(v, np.inf), target[cut["id"]])
        else:
            guard[cut["id"]] = value - float(cut["floor"])
    if set(t_by_weighting) != set(channel.WEIGHTINGS):
        raise ValueError("AB/SEX cuts are required in both weightings")
    t = min(target.values())
    task_slack = {v: caps[v] - task[v] for v in channel.WEIGHTINGS}
    min_guard = min(guard.values()) if guard else float("inf")
    feasible = min_guard >= -PRIMAL_TOL and min(task_slack.values()) >= -PRIMAL_TOL
    return {"t": float(t), "t_by_weighting": t_by_weighting,
            "task_U": task["U"], "task_W": task["W"], "task_balanced": 0.5 * (task["U"] + task["W"]),
            "task_slack": task_slack, "minimum_guard_slack": float(min_guard),
            "worst_target_cut": min(target, key=target.get),
            "feasible": bool(feasible)}


def _assignment_params(index, K, M):
    A = np.zeros((K, M))
    A[np.arange(K), list(index)] = 1.
    return np.zeros((channel.N_STATES, channel.N_TOKENS)), A, 1.


def enumerate_assignments(cost, cuts, caps, K, M):
    rows = []
    for index in itertools.product(range(M), repeat=K):
        B, A, _ = _assignment_params(index, K, M)
        rec = evaluate(B, A, cost, cuts, caps)
        rows.append({"index": list(index), "non_d17_contexts": int(sum(m != 0 for m in index)), **rec})
    return rows


def best_privacy(rows):
    """Exact D: max t among feasible; ties (1e-12) -> lower balanced task, fewer non-D17, lexicographic."""
    feasible = [r for r in rows if r["feasible"]]
    if not any(r["index"] == [0] * len(r["index"]) for r in feasible):
        raise AssertionError("all-D17 assignment must be feasible")
    top = max(r["t"] for r in feasible)
    tied = [r for r in feasible if r["t"] >= top - TIE]
    return min(tied, key=lambda r: (r["task_balanced"], r["non_d17_contexts"], r["index"])), len(feasible)


def best_task(rows):
    return min(rows, key=lambda r: (r["task_balanced"], r["non_d17_contexts"], r["index"]))


def _lp_rows(cost, cuts, caps, K, M, full):
    """Inequality rows over x = [vec(B) if full] + vec(A) + [eta if full] + [t]."""
    S, Z = channel.N_STATES, channel.N_TOKENS
    nb = S * Z if full else 0
    n = nb + K * M + (1 if full else 0) + 1

    def vec(gb, ga):
        v = np.zeros(n)
        if full:
            v[:nb] = np.asarray(gb, dtype=np.float64).ravel()
        v[nb:nb + K * M] = np.asarray(ga, dtype=np.float64).ravel()
        return v
    rows, rhs = [], []
    for cut in cuts:
        row = -vec(cut["coeff_B"], cut["coeff_A"])
        if cut["role"] == TARGET:
            row[-1] = 1.
            rhs.append(-float(cut["rho"]))
        else:
            rhs.append(-float(cut["floor"]))
        rows.append(row)
    task = {v: vec(cost[v]["B"], cost[v]["A"]) for v in channel.WEIGHTINGS}
    for v in channel.WEIGHTINGS:
        rows.append(task[v]); rhs.append(caps[v])
    if full:
        eq_b = sparse.hstack((sparse.kron(sparse.eye(S), np.ones((1, Z))), sparse.csr_matrix((S, K * M)),
                              np.ones((S, 1)), sparse.csr_matrix((S, 1))), format="csr")
        eq_a = sparse.hstack((sparse.csr_matrix((K, nb)), sparse.kron(sparse.eye(K), np.ones((1, M))),
                              -np.ones((K, 1)), sparse.csr_matrix((K, 1))), format="csr")
        a_eq, b_eq = sparse.vstack((eq_b, eq_a), format="csr"), np.concatenate((np.ones(S), np.zeros(K)))
    else:
        a_eq = sparse.hstack((sparse.kron(sparse.eye(K), np.ones((1, M))), sparse.csr_matrix((K, 1))), format="csr")
        b_eq = np.ones(K)
    return np.vstack(rows), np.asarray(rhs), a_eq, b_eq, 0.5 * (task["U"] + task["W"]), n, nb


def solve_lp(cost, cuts, caps, K, M, *, full=False, time_limit=300.):
    """Stage 1: max t. Stage 2 (amendment A1): min balanced task s.t. t >= t* (HiGHS tolerances).

    full=False: R arms (B = 0, eta = 1, A row-stochastic). full=True: NM4PF (B, A, eta free).
    """
    a_ub, b_ub, a_eq, b_eq, task_vec, n, nb = _lp_rows(cost, cuts, caps, K, M, full)
    options = dict(HIGHS_OPTIONS); options["time_limit"] = float(time_limit)
    bounds = [(0., 1.)] * (n - 1) + [(None, None)]
    c1 = np.zeros(n); c1[-1] = -1.
    raw1 = linprog(c1, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs", options=options)
    if raw1.status != 0 or raw1.x is None:
        raise RuntimeError(f"stage-1 LP unresolved: {raw1.message}")
    t_star = float(raw1.x[-1])
    y_ub = np.asarray(raw1.ineqlin.marginals, dtype=float)
    y_eq = np.asarray(raw1.eqlin.marginals, dtype=float)
    reduced = c1 - a_ub.T @ y_ub - a_eq.T @ y_eq
    dual_value = float(b_ub @ y_ub + b_eq @ y_eq + np.minimum(reduced[:-1], 0.).sum())
    bounds2 = bounds[:-1] + [(t_star, None)]
    raw2 = linprog(task_vec, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds2, method="highs",
                   options=options)
    if raw2.status != 0 or raw2.x is None:
        raise RuntimeError(f"stage-2 LP unresolved: {raw2.message}")
    x = raw2.x
    if full:
        cleaned = channel._clean_params(x[:nb].reshape(channel.N_STATES, channel.N_TOKENS),
                                        x[nb:nb + K * M].reshape(K, M), float(x[nb + K * M]))
        if cleaned is None:
            raise RuntimeError("full-mixture solution failed the simplex cleaning")
        B, A, eta, repair = cleaned
    else:
        A = np.maximum(x[:K * M].reshape(K, M), 0.)
        mass = A.sum(1)
        if np.max(np.abs(mass - 1)) > SIMPLEX_TOL:
            raise RuntimeError("mixture LP rows are not stochastic within tolerance")
        A = A / mass[:, None]
        repair = float(np.max(np.abs(A - x[:K * M].reshape(K, M))))
        B, eta = np.zeros((channel.N_STATES, channel.N_TOKENS)), 1.
    rec = evaluate(B, A, cost, cuts, caps)
    if not rec["feasible"] or rec["t"] < t_star - 1e-9:
        raise RuntimeError("cleaned LP solution violates the objective's constraints or loses t")
    law = np.asarray(A)
    return {"B": B, "A": A, "eta": float(eta), "record": {
        **rec, "solver": "HiGHS", "stage1_status": int(raw1.status), "stage1_t": t_star,
        "stage1_task_balanced": float(task_vec @ raw1.x), "stage2_status": int(raw2.status),
        "stage2_message": raw2.message, "lexicographic_tie_rule": "amendment A1",
        "repair_max": repair, "dual_t_upper_bound": -dual_value, "dual_gap": -dual_value - t_star,
        "free_variable_reduced_cost_t": float(reduced[-1]),
        "one_hot_contexts": int(np.sum(law.max(1) >= 1 - 1e-12)),
        "fractional_contexts": int(np.sum(law.max(1) < 1 - 1e-12))}}


def solve_mixture(cost, cuts, caps, K, M):
    return solve_lp(cost, cuts, caps, K, M, full=False)


def solve_full(cost, cuts, d17):
    caps = task_caps(cost, d17)
    K, M = np.asarray(cost["U"]["A"]).shape
    sol = solve_lp(cost, cuts, caps, K, M, full=True)
    reference = channel.solve_privacy_first(cost, cuts, d17)
    if not reference.get("feasible") or abs(reference["tau"] - sol["record"]["stage1_t"]) > 1e-7:
        raise RuntimeError("NM4PF stage-1 t differs from the SC privacy-first LP")
    sol["record"]["sc_solver_tau"] = float(reference["tau"])
    sol["record"]["sc_solver_status"] = reference.get("status")
    return sol


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def _write_unit(root, release, anchor, K, params, record, basis):
    root = Path(root)
    if (root / "COMPLETE.json").exists():
        raise FileExistsError(f"{root} already complete")
    root.mkdir(parents=True, exist_ok=True)
    B, A, eta = channel.validate_params(params[0], params[1], params[2])
    fit_nm._save_npz(root / "PARAMS.npz", B=B, A=A, eta=np.asarray(eta))
    _json(root / "SOLVE.json", record)
    manifest = basis["bank_manifest"]
    bank_rel = os.path.relpath(basis["bank_dir"], root.resolve())
    spec = {"schema": SPEC_SCHEMA, "kind": "nested", "variant": release, "anchor": int(anchor),
            "K": int(K), "eta_fixed_zero": False, "policy_columns": list(basis["names"]),
            "params_relative": "PARAMS.npz", "params_sha256": _sha(root / "PARAMS.npz"),
            "bank_dir": str(basis["bank_dir"]), "bank_dir_relative": bank_rel,
            "bank_manifest_sha256": basis["bank_manifest_sha256"],
            "policy_bank_sha256": manifest["policy_bank_sha256"],
            "contexts_sha256": manifest["contexts_sha256"],
            "legal_inputs": list(fit_nm.policies.LEGAL_INPUTS),
            "wire": "H_A unchanged plus one keyed-persistent 17-token draw",
            "study": STUDY}
    _json(root / "RELEASE_SPEC.json", spec)
    _json(root / "COMPLETE.json", {"schema": SCHEMA_SOLVE, "status": "COMPLETE", "release": release,
                                   "anchor": int(anchor), "K": int(K),
                                   "params_sha256": spec["params_sha256"],
                                   "solve_sha256": _sha(root / "SOLVE.json"),
                                   "basis_bank_sha256": basis["bank_sha256"],
                                   "outer_labels_accessed": False})
    pins = {name: _sha(root / name) for name in ("RELEASE_SPEC.json", "PARAMS.npz", "COMPLETE.json", "SOLVE.json")}
    laws.write_descriptor(root, "nested", release_id=f"a{int(anchor)}_{release}", anchor=int(anchor),
                          pins=pins, extra={"variant": release, "K": int(K), "study": STUDY})


def solve_anchor(nm4u_dir, bank_dir, det_sel4_dir, d17, out_dir) -> dict:
    """Solve every arm/control for one anchor and write release units + SOLVE_REPORT.json."""
    basis = load_basis(nm4u_dir, bank_dir)
    anchor, names = basis["anchor"], basis["names"]
    M = len(names)
    d17 = np.asarray(d17, dtype=np.float64)
    cost4, cuts4 = basis["cost"], basis["cuts"]
    cost1, cuts1 = collapse_k1(cost4, cuts4)
    caps = task_caps(cost4, d17)
    if task_caps(cost1, d17) != caps:
        raise AssertionError("K=1 collapse changed the D17 task caps")
    witness = evaluate(d17, np.zeros((4, M)), cost4, cuts4, caps)
    compact = evaluate(*_assignment_params([0] * 4, 4, M)[:2], cost4, cuts4, caps)
    if abs(witness["t"]) > 1e-12 or abs(compact["t"]) > 1e-10 or not witness["feasible"]:
        raise AssertionError("D17 witness must have t = 0 and be feasible")
    out = Path(out_dir)
    report = {"schema": SCHEMA_SOLVE, "study": STUDY, "anchor": anchor,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "basis": {k: basis[k] for k in ("files_sha256", "bank_sha256", "bank_manifest_sha256",
                                               "cut_count", "cut_count_by_role", "names")},
              "task_caps": caps, "d17_witness": witness, "releases": {}}
    rows4 = enumerate_assignments(cost4, cuts4, caps, 4, M)
    rows1 = enumerate_assignments(cost1, cuts1, caps, 1, M)
    d4, feasible4 = best_privacy(rows4)
    d1, feasible1 = best_privacy(rows1)
    task4 = best_task(rows4)
    det_best, _ = fit_nm.deterministic_selector(cost4, cuts4, 4, names)
    archived = json.loads((Path(det_sel4_dir) / "DET_SEL.json").read_text())
    if det_best["index"] != archived["selected"]["index"]:
        raise AssertionError("DET_SEL4 replay on the pinned basis differs from the archived selection")
    with np.load(Path(det_sel4_dir) / "PARAMS.npz") as p:
        if not np.array_equal(p["A"], _assignment_params(det_best["index"], 4, M)[1]):
            raise AssertionError("archived DET_SEL4 PARAMS differ from the replayed assignment")
    det_eval = evaluate(*_assignment_params(det_best["index"], 4, M)[:2], cost4, cuts4, caps)
    r4 = solve_mixture(cost4, cuts4, caps, 4, M)
    r1 = solve_mixture(cost1, cuts1, caps, 1, M)
    nm = solve_full(cost4, cuts4, d17)
    tol = 1e-9
    checks = {"t_R4_ge_t_D4": r4["record"]["t"] >= d4["t"] - tol,
              "t_D4_ge_t_D1": d4["t"] >= d1["t"] - tol,
              "t_R4_ge_t_R1": r4["record"]["t"] >= r1["record"]["t"] - tol,
              "t_R1_ge_t_D1": r1["record"]["t"] >= d1["t"] - tol,
              "t_NM4PF_ge_t_R4": nm["record"]["t"] >= r4["record"]["t"] - tol,
              "t_D1_ge_0": d1["t"] >= -tol}
    if not all(checks.values()):
        raise AssertionError(f"nesting inequalities violated: {checks}")
    deterministic = {
        "D4": (d4, 4, {"assignments_evaluated": len(rows4), "feasible_assignments": feasible4,
                       "exact_enumeration": True, "gap": 0.0}),
        "D1": (d1, 1, {"assignments_evaluated": len(rows1), "feasible_assignments": feasible1,
                       "exact_enumeration": True, "gap": 0.0}),
        "TASK_SEL4": (task4, 4, {"assignments_evaluated": len(rows4), "privacy_constraint": False})}
    for release, (row, K, extra) in deterministic.items():
        params = _assignment_params(row["index"], K, M)
        record = {**row, **extra, "assignment": [names[m] for m in row["index"]],
                  "form": "one policy per context (B=0, eta=1)"}
        _write_unit(out / f"a{anchor}_{release}", release, anchor, K, params, record, basis)
        report["releases"][release] = record
    for release, (sol, K) in {"R4": (r4, 4), "R1": (r1, 1), "NM4PF": (nm, 4)}.items():
        record = {**sol["record"], "A": sol["A"].tolist(), "eta": sol["eta"],
                  "form": "full nested mixture" if release == "NM4PF" else "per-context mixture (B=0, eta=1)"}
        _write_unit(out / f"a{anchor}_{release}", release, anchor, K,
                    (sol["B"], sol["A"], sol["eta"]), record, basis)
        report["releases"][release] = record
    report["releases"]["DET_SEL4"] = {**det_eval, "index": det_best["index"],
                                      "assignment": [names[m] for m in det_best["index"]],
                                      "archived_unit": str(Path(det_sel4_dir).resolve()),
                                      "replay_matches_archive": True}
    report["nesting_checks"] = checks
    report["value_of_randomization_fixed_bank"] = {"t_R4_minus_t_D4": r4["record"]["t"] - d4["t"],
                                                   "t_R1_minus_t_D1": r1["record"]["t"] - d1["t"]}
    report["top_assignments_D4"] = sorted(
        [r for r in rows4 if r["feasible"]], key=lambda r: (-r["t"], r["task_balanced"]))[:10]
    _json(out / f"a{anchor}_SOLVE_REPORT.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nm4u", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--det-sel4", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--anchor", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if "private" not in Path(args.out).resolve().parts:
        raise ValueError("release units go to a private directory")
    from experiments.pcrl_task_aligned_cuts_v1 import data
    d17 = data.load_map(data.index(args.index), args.anchor, "D17")
    report = solve_anchor(args.nm4u, args.bank, args.det_sel4, d17, args.out)
    if report["anchor"] != args.anchor:
        raise ValueError("basis anchor differs from --anchor")
    print(json.dumps({"anchor": args.anchor,
                      "t": {r: report["releases"][r]["t"] for r in report["releases"]}}, sort_keys=True))


if __name__ == "__main__":
    main()
