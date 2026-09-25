"""Independent verifier for the privacy-first selector v1 study.

Written from PROTOCOL.md, PROTOCOL_AMENDMENTS.md and agents/verifier/VERIFIER_BRIEF.md
only. It does not import or read solve.py, host.py, outer.py, lockgate.py, or any
SC/TAC inference/channel code. Allowed dependencies: numpy, scipy, json, hashlib, subprocess (+ stdlib argparse/datetime).

Part A (solves): re-derive D4, D1, TASK_SEL4, DET_SEL4 by own enumeration and R4, R1,
NM4PF by own LP formulations (stage 1 primal via HiGHS interior point, cross-checked
with the explicitly constructed dual LP via dual simplex; amendment A1 stage 2 via
interior point), then compare with the implementation's PARAMS/SOLVE outputs and
SOLVE_REPORTS.json.

Part B: recompute every outer endpoint (point estimate, own household bootstrap with the
verifier's own seed, bounds at the locked z, decisions, labels). Part C: custody chain.

Usage:
    python verify_independent.py part_a --data <V> [--out <json>]
    python -m experiments.pcrl_privacy_first_selector_v1.verify_independent part_bc \
        --outer-root <private/outer> --lock <SELECTION_LOCK.json> --endpoint-table <json> \
        --inference <json> --lockcheck <git checkout> --inner-root <private/inner_panels> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

STUDY = "pcrl_privacy_first_selector_v1"
REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / STUDY
POLICIES = ["D17", "task_only", "local_priced", "coalition_priced", "all_priced_x2"]
TARGET_ROLE = "AB/SEX"
TASK_DELTA = 0.001
FEAS_TOL = 1e-7        # brief: tolerance 1e-7
T_TOL = 1e-6           # brief: compare t to 1e-6
TIE_TOL = 1e-12        # protocol: ties within 1e-12
ANCHORS = (0, 1, 2)


# ----------------------------------------------------------------------------- basis
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_identity(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x, dtype=np.float64)
    return hashlib.sha256((str(x.dtype) + str(x.shape)).encode() + x.tobytes()).hexdigest()


class Basis:
    """Frozen coefficient blocks for one anchor."""

    def __init__(self, data: Path, anchor: int):
        closing = data / "prior_units" / f"a{anchor}_NM4_U" / "closing"
        self.files = {
            "closing/TASK_BLOCKS.npz": closing / "TASK_BLOCKS.npz",
            "closing/CALIBRATED_BLOCKS.npz": closing / "CALIBRATED_BLOCKS.npz",
            "closing/CALIBRATED_BANK.json": closing / "CALIBRATED_BANK.json",
        }
        tb = np.load(closing / "TASK_BLOCKS.npz")
        self.task = {v: (np.asarray(tb[f"{v}_B"], float), np.asarray(tb[f"{v}_A"], float)) for v in ("U", "W")}
        bank = json.loads((closing / "CALIBRATED_BANK.json").read_text())
        blocks = np.load(closing / "CALIBRATED_BLOCKS.npz")
        self.cuts = bank["cuts"]
        n = len(self.cuts)
        self.GB = np.stack([np.asarray(blocks[f"cut_{i:04d}_B"], float) for i in range(n)])  # n x 32 x 17
        self.GA = np.stack([np.asarray(blocks[f"cut_{i:04d}_A"], float) for i in range(n)])  # n x 4 x 5
        self.rho = np.array([c["rho"] for c in self.cuts], float)
        self.floor = np.array([c["floor"] for c in self.cuts], float)
        self.role = np.array([c["role"] for c in self.cuts])
        self.weighting = np.array([c["weighting"] for c in self.cuts])
        self.ids = [c["id"] for c in self.cuts]
        self.target = self.role == TARGET_ROLE
        self.d17 = np.asarray(np.load(data / f"d17_a{anchor}.npy"), float)
        self.d17_identity = array_identity(self.d17)
        self.T, self.Z = self.d17.shape
        self.K, self.M = self.GA.shape[1:]
        # D17 witness (B = D17, A = 0)
        w = self.evaluate(self.d17, np.zeros((self.K, self.M)))
        self.task_d17 = {"U": w["task_U"], "W": w["task_W"]}
        self.cap = {v: self.task_d17[v] + TASK_DELTA for v in ("U", "W")}
        self.witness = self.evaluate(self.d17, np.zeros((self.K, self.M)))  # re-evaluate with caps set

    # K=1 variants: A blocks summed over contexts
    def ga(self, K: int) -> np.ndarray:
        return self.GA if K == self.K else self.GA.sum(axis=1, keepdims=True)

    def ta(self, v: str, K: int) -> np.ndarray:
        A = self.task[v][1]
        return A if K == self.K else A.sum(axis=0, keepdims=True)

    def evaluate(self, B: np.ndarray, A: np.ndarray) -> dict:
        K = A.shape[0]
        L = np.einsum("ntz,tz->n", self.GB, B) + np.einsum("nkm,km->n", self.ga(K), A)
        tU = float(np.sum(self.task["U"][0] * B) + np.sum(self.ta("U", K) * A))
        tW = float(np.sum(self.task["W"][0] * B) + np.sum(self.ta("W", K) * A))
        return self._summarise(L, tU, tW)

    def _summarise(self, L: np.ndarray, tU: float, tW: float) -> dict:
        gap = L - self.rho
        t_by = {wt: float(gap[self.target & (self.weighting == wt)].min()) for wt in ("U", "W")}
        tgt = np.where(self.target)[0]
        worst = int(tgt[np.argmin(gap[tgt])])
        guard_slack = L[~self.target] - self.floor[~self.target]
        out = {
            "t": float(gap[self.target].min()),
            "t_by_weighting": t_by,
            "worst_target_cut": self.ids[worst],
            "task_U": tU,
            "task_W": tW,
            "task_balanced": 0.5 * (tU + tW),
            "task_slack": {"U": self.cap["U"] - tU if hasattr(self, "cap") else None,
                           "W": self.cap["W"] - tW if hasattr(self, "cap") else None},
            "minimum_guard_slack": float(guard_slack.min()),
            "minimum_all_cut_floor_slack": float((L - self.floor).min()),
        }
        if hasattr(self, "cap"):
            out["feasible"] = bool(out["minimum_guard_slack"] >= -FEAS_TOL
                                   and out["task_slack"]["U"] >= -FEAS_TOL
                                   and out["task_slack"]["W"] >= -FEAS_TOL)
        return out


# ----------------------------------------------------------------------- enumeration
def enumerate_assignments(bs: Basis, K: int) -> dict:
    """All M^K one-policy-per-context assignments, vectorised."""
    idx = np.array(list(itertools.product(range(bs.M), repeat=K)), int)  # lexicographic order
    GA = bs.ga(K)  # n x K x M
    L = np.zeros((len(idx), GA.shape[0]))
    tU = np.zeros(len(idx))
    tW = np.zeros(len(idx))
    aU, aW = bs.ta("U", K), bs.ta("W", K)
    for k in range(K):
        L += GA[:, k, idx[:, k]].T
        tU += aU[k, idx[:, k]]
        tW += aW[k, idx[:, k]]
    gap = L - bs.rho[None, :]
    t = gap[:, bs.target].min(axis=1)
    guard = (L[:, ~bs.target] - bs.floor[None, ~bs.target]).min(axis=1)
    allcut = (L - bs.floor[None, :]).min(axis=1)
    bal = 0.5 * (tU + tW)
    nonD17 = (idx != 0).sum(axis=1)
    return dict(idx=idx, L=L, t=t, guard=guard, allcut=allcut, tU=tU, tW=tW, bal=bal, nonD17=nonD17)


def pick(cands: np.ndarray, primary: np.ndarray, maximize: bool, e: dict) -> int:
    """Tie rule: best primary; within 1e-12 -> lower balanced task (if primary is not task)
    -> fewer non-D17 contexts -> lexicographic index (enumeration order is lexicographic)."""
    vals = primary[cands]
    best = vals.max() if maximize else vals.min()
    near = cands[np.abs(vals - best) <= TIE_TOL]
    if primary is not e["bal"]:
        b = e["bal"][near]
        near = near[np.abs(b - b.min()) <= TIE_TOL]
    nd = e["nonD17"][near]
    near = near[nd == nd.min()]
    return int(near[0])  # lexicographic: enumeration order is lexicographic in index tuple


def det_params(bs: Basis, index, K: int):
    A = np.zeros((K, bs.M))
    A[np.arange(K), index] = 1.0
    return np.zeros((bs.T, bs.Z)), A, 1.0


def solve_det_arm(bs: Basis, K: int, kind: str, tol: float = FEAS_TOL) -> dict:
    e = enumerate_assignments(bs, K)
    if kind == "privacy_first":  # D4 / D1
        feas = (e["guard"] >= -tol) & (e["tU"] <= bs.cap["U"] + tol) & (e["tW"] <= bs.cap["W"] + tol)
        cands = np.where(feas)[0]
        i = pick(cands, e["t"], True, e)
    elif kind == "task_only":  # TASK_SEL4: no constraints
        cands = np.arange(len(e["idx"]))
        feas = np.ones(len(cands), bool)
        i = pick(cands, e["bal"], False, e)
    elif kind == "det_sel":  # DET_SEL4: every cut >= floor, no task cap
        feas = e["allcut"] >= -tol
        cands = np.where(feas)[0]
        i = pick(cands, e["bal"], False, e)
    else:
        raise ValueError(kind)
    index = [int(v) for v in e["idx"][i]]
    B, A, eta = det_params(bs, index, K)
    ev = bs.evaluate(B, A)
    return {"index": index, "assignment": [POLICIES[j] for j in index],
            "feasible_assignments": int(feas.sum()), "assignments_evaluated": int(len(e["idx"])),
            "non_d17_contexts": int(sum(j != 0 for j in index)), **ev, "_e": e, "_params": (B, A, eta)}


# ------------------------------------------------------------------------------ LPs
class LP:
    """max c^T x s.t. A_ub x <= b_ub, A_eq x = b_eq; lower bounds 0 except free vars."""

    def __init__(self, bs: Basis, arm: str):
        self.bs, self.arm = bs, arm
        K = 1 if arm == "R1" else bs.K
        self.Kx = K
        GA, M = bs.ga(K), bs.M
        n = len(bs.cuts)
        nB = bs.T * bs.Z if arm == "NM4PF" else 0
        nA = K * M
        neta = 1 if arm == "NM4PF" else 0
        self.nB, self.nA, self.neta = nB, nA, neta
        self.nx = nB + nA + neta + 1  # last = t
        it = self.nx - 1
        G = np.zeros((n, self.nx))
        if nB:
            G[:, :nB] = bs.GB.reshape(n, -1)
        G[:, nB:nB + nA] = GA.reshape(n, -1)
        tv = {}
        for v in ("U", "W"):
            row = np.zeros(self.nx)
            if nB:
                row[:nB] = bs.task[v][0].ravel()
            row[nB:nB + nA] = bs.ta(v, K).ravel()
            tv[v] = row
        self.task_rows = tv
        rows, rhs = [], []
        tgt = bs.target
        # target: -(G x) + t <= -rho
        R = -G[tgt].copy(); R[:, it] = 1.0
        rows.append(R); rhs.append(-bs.rho[tgt])
        # guards: -(G x) <= -floor
        rows.append(-G[~tgt]); rhs.append(-bs.floor[~tgt])
        # task caps
        rows.append(np.stack([tv["U"], tv["W"]])); rhs.append(np.array([bs.cap["U"], bs.cap["W"]]))
        self.A_ub = np.vstack(rows)
        self.b_ub = np.concatenate(rhs)
        eq, beq = [], []
        if arm == "NM4PF":
            ie = nB + nA
            for s in range(bs.T):  # sum_z B[s,z] + eta = 1
                r = np.zeros(self.nx); r[s * bs.Z:(s + 1) * bs.Z] = 1.0; r[ie] = 1.0
                eq.append(r); beq.append(1.0)
            for k in range(K):  # sum_m A[k,m] - eta = 0
                r = np.zeros(self.nx); r[nB + k * M:nB + (k + 1) * M] = 1.0; r[ie] = -1.0
                eq.append(r); beq.append(0.0)
        else:
            for k in range(K):  # rows of A sum to 1 (eta = 1, B = 0)
                r = np.zeros(self.nx); r[nB + k * M:nB + (k + 1) * M] = 1.0
                eq.append(r); beq.append(1.0)
        self.A_eq = np.array(eq)
        self.b_eq = np.array(beq)
        self.free = np.zeros(self.nx, bool); self.free[it] = True
        self.bounds = [(None, None) if f else (0, None) for f in self.free]
        self.it = it

    def unpack(self, x):
        bs = self.bs
        B = x[:self.nB].reshape(bs.T, bs.Z) if self.nB else np.zeros((bs.T, bs.Z))
        A = x[self.nB:self.nB + self.nA].reshape(self.Kx, bs.M)
        eta = float(x[self.nB + self.nA]) if self.neta else 1.0
        return B, A, eta

    def stage1_primal_ipm(self):
        c = np.zeros(self.nx); c[self.it] = -1.0
        r = linprog(c, A_ub=self.A_ub, b_ub=self.b_ub, A_eq=self.A_eq, b_eq=self.b_eq,
                    bounds=self.bounds, method="highs-ipm",
                    options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10,
                             "ipm_optimality_tolerance": 1e-12})
        return r

    def stage1_dual(self):
        """Explicit dual of max t: min b_ub^T y + b_eq^T z, y >= 0, z free,
        A_ub^T y + A_eq^T z >= c on nonneg columns, = c on the free column (t)."""
        c = np.zeros(self.nx); c[self.it] = 1.0
        m_ub, m_eq = self.A_ub.shape[0], self.A_eq.shape[0]
        obj = np.concatenate([self.b_ub, self.b_eq])
        MT = np.hstack([self.A_ub.T, self.A_eq.T])  # nx x (m_ub+m_eq)
        nonneg = ~self.free
        # >= c  ->  -MT y <= -c
        r = linprog(obj, A_ub=-MT[nonneg], b_ub=-c[nonneg], A_eq=MT[self.free], b_eq=c[self.free],
                    bounds=[(0, None)] * m_ub + [(None, None)] * m_eq, method="highs-ds",
                    options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
        return r

    def stage2_ipm(self, t_star: float, slack: float = 0.0):
        c = 0.5 * (self.task_rows["U"] + self.task_rows["W"])
        bounds = list(self.bounds)
        bounds[self.it] = (t_star - slack, None)
        r = linprog(c, A_ub=self.A_ub, b_ub=self.b_ub, A_eq=self.A_eq, b_eq=self.b_eq,
                    bounds=bounds, method="highs-ipm",
                    options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10,
                             "ipm_optimality_tolerance": 1e-12})
        return r


def structure_check(bs: Basis, arm: str, B, A, eta) -> dict:
    """Parameterisation constraints of the implementation's law."""
    out = {"min_entry": float(min(B.min(), A.min(), eta))}
    if arm in ("R4", "R1"):
        out["B_zero"] = bool(np.all(B == 0))
        out["eta"] = float(eta)
        out["row_sum_err"] = float(np.abs(A.sum(1) - 1).max())
        out["ok"] = bool(out["B_zero"] and abs(eta - 1) <= FEAS_TOL and out["row_sum_err"] <= FEAS_TOL
                         and out["min_entry"] >= -FEAS_TOL)
    else:
        out["B_row_err"] = float(np.abs(B.sum(1) + eta - 1).max())
        out["A_row_err"] = float(np.abs(A.sum(1) - eta).max())
        out["ok"] = bool(out["B_row_err"] <= FEAS_TOL and out["A_row_err"] <= FEAS_TOL
                         and out["min_entry"] >= -FEAS_TOL)
    return out


def solve_lp_arm(bs: Basis, arm: str) -> dict:
    lp = LP(bs, arm)
    p = lp.stage1_primal_ipm()
    d = lp.stage1_dual()
    t_primal = -p.fun if p.status == 0 else None
    t_dual = d.fun if d.status == 0 else None
    t_star = t_primal
    # stage 2 (A1): min balanced task s.t. t >= t*. IPM may leave t* marginally above the
    # true optimum; if infeasible retry with tiny slack and record it.
    s2, used_slack = None, 0.0
    for slack in (0.0, 1e-10, 1e-9, 1e-8):
        s2 = lp.stage2_ipm(t_star, slack)
        used_slack = slack
        if s2.status == 0:
            break
    B, A, eta = lp.unpack(s2.x)
    ev = bs.evaluate(B, A)
    return {"stage1_status_primal_ipm": int(p.status), "stage1_status_dual_ds": int(d.status),
            "t_star_primal_ipm": t_primal, "t_star_dual": t_dual,
            "primal_dual_gap": None if (t_primal is None or t_dual is None) else float(t_dual - t_primal),
            "stage2_status": int(s2.status), "stage2_t_slack_used": used_slack,
            "stage2_task_balanced_objective": float(s2.fun),
            "law": {"A": A.tolist(), "eta": eta, "B_nonzero": int((np.abs(B) > 1e-12).sum())},
            "structure": structure_check(bs, arm, B, A, eta),
            **ev, "_params": (B, A, eta)}


# --------------------------------------------------------------------------- compare
def load_theirs(data: Path, anchor: int, arm: str):
    unit = data / "units" / f"a{anchor}_{arm}"
    z = np.load(unit / "PARAMS.npz")
    solve = json.loads((unit / "SOLVE.json").read_text())
    return np.asarray(z["B"], float), np.asarray(z["A"], float), float(z["eta"]), solve


def num_diff(a, b):
    if a is None or b is None:
        return None
    return float(abs(a - b))


def part_a(data: Path, out: Path | None = None) -> dict:
    reports = json.loads((RESULTS / "SOLVE_REPORTS.json").read_text())
    manifest = json.loads((RESULTS / "BASIS_MANIFEST.json").read_text())
    result = {"schema": "pcrl-pfs-independent-verification-part-a-v1", "study": STUDY,
              "method": {
                  "enumeration": "numpy vectorised over all M^K assignments (lexicographic order); "
                                 "feasibility tolerance 1e-7; ties within 1e-12",
                  "lp_stage1": "primal via scipy highs-ipm (own matrices) + explicit dual LP via highs-ds",
                  "lp_stage2": "A1: min 0.5(task_U+task_W) s.t. same constraints and t >= t* (highs-ipm)",
                  "t_compare_tol": T_TOL, "feasibility_tol": FEAS_TOL},
              "anchors": {}}
    all_pass = True
    max_diff = {"t_det": 0.0, "task_det": 0.0, "t_lp": 0.0, "task_lp": 0.0, "report_vs_params_t": 0.0,
                "lp_law_A_maxabs": 0.0}
    for a in ANCHORS:
        bs = Basis(data, a)
        rep = reports["anchors"][str(a)]["releases"]
        man = manifest["anchors"][str(a)]
        anc = {"basis": {}, "releases": {}}
        # basis hashes
        hb = {k: sha256_file(p) == man["nm4u_files_sha256"][k] for k, p in bs.files.items()}
        hb["d17_map_array_sha256"] = bs.d17_identity == man["d17_map_array_sha256"]
        for k in ("DET_SEL.json", "PARAMS.npz"):
            hb[f"DET_SEL4/{k}"] = sha256_file(data / "prior_units" / f"a{a}_DET_SEL4" / k) == man["det_sel4_files_sha256"][k]
        anc["basis"] = {"sha256_matches": hb, "all_match": all(hb.values()),
                        "cut_count": len(bs.cuts),
                        "d17_witness": {k: bs.witness[k] for k in ("t", "task_U", "task_W", "minimum_guard_slack")},
                        "d17_witness_feasible": bs.witness["feasible"],
                        "d17_equals_policy_column_0": {
                            "max_abs_diff_all_cuts_and_tasks": float(max(
                                np.abs(np.einsum("ntz,tz->n", bs.GB, bs.d17) - bs.GA[:, :, 0].sum(1)).max(),
                                abs(np.sum(bs.task["U"][0] * bs.d17) - bs.task["U"][1][:, 0].sum()),
                                abs(np.sum(bs.task["W"][0] * bs.d17) - bs.task["W"][1][:, 0].sum())))}}
        all_pass &= anc["basis"]["all_match"] and bs.witness["feasible"]

        # ---- deterministic arms
        det_specs = {"D4": (bs.K, "privacy_first"), "D1": (1, "privacy_first"),
                     "TASK_SEL4": (bs.K, "task_only")}
        mine_by_arm = {}
        for arm, (K, kind) in det_specs.items():
            m = solve_det_arm(bs, K, kind)
            # tolerance sensitivity of the selection
            sens = {str(tol): solve_det_arm(bs, K, kind, tol)["index"] for tol in (0.0, 1e-9, 1e-7, 1e-6)}
            B, A, eta, solve = load_theirs(data, a, arm)
            mB, mA, meta = m["_params"]
            theirs_ev = bs.evaluate(B, A)
            r = rep[arm]
            chk = {
                "params_exact": bool(np.array_equal(B, mB) and np.array_equal(A, mA) and eta == meta),
                "index_match_report": r["index"] == m["index"],
                "index_match_solve_json": solve.get("index") == m["index"],
                "t_diff_report": num_diff(r["t"], m["t"]),
                "task_balanced_diff_report": num_diff(r["task_balanced"], m["task_balanced"]),
                "feasible_match": bool(r["feasible"]) == bool(m["feasible"]) == bool(theirs_ev["feasible"]),
                "feasible_count_match": (r.get("feasible_assignments") == m["feasible_assignments"])
                if "feasible_assignments" in r else None,
                "report_matches_their_params_t": num_diff(r["t"], theirs_ev["t"]),
                "selection_stable_across_tol": len({tuple(v) for v in sens.values()}) == 1,
            }
            ok = (chk["params_exact"] and chk["index_match_report"] and chk["index_match_solve_json"]
                  and chk["t_diff_report"] <= T_TOL and chk["task_balanced_diff_report"] <= T_TOL
                  and chk["feasible_match"] and chk["feasible_count_match"] is not False
                  and chk["report_matches_their_params_t"] <= T_TOL)
            max_diff["t_det"] = max(max_diff["t_det"], chk["t_diff_report"])
            max_diff["task_det"] = max(max_diff["task_det"], chk["task_balanced_diff_report"])
            max_diff["report_vs_params_t"] = max(max_diff["report_vs_params_t"], chk["report_matches_their_params_t"])
            anc["releases"][arm] = {"pass": bool(ok), "checks": chk, "tol_sensitivity_index": sens,
                                    "mine": {k: m[k] for k in ("index", "assignment", "t", "t_by_weighting", "task_U",
                                                               "task_W", "task_balanced", "feasible",
                                                               "feasible_assignments", "non_d17_contexts",
                                                               "worst_target_cut", "minimum_guard_slack")},
                                    "theirs": {k: r.get(k) for k in ("index", "assignment", "t", "t_by_weighting",
                                                                     "task_U", "task_W", "task_balanced", "feasible",
                                                                     "feasible_assignments", "non_d17_contexts",
                                                                     "worst_target_cut")}}
            mine_by_arm[arm] = m
            all_pass &= ok

        # ---- DET_SEL4 re-derivation vs archive
        m = solve_det_arm(bs, bs.K, "det_sel")
        e = m["_e"]
        arch = json.loads((data / "prior_units" / f"a{a}_DET_SEL4" / "DET_SEL.json").read_text())
        z = np.load(data / "prior_units" / f"a{a}_DET_SEL4" / "PARAMS.npz")
        aB, aA, aeta = np.asarray(z["B"], float), np.asarray(z["A"], float), float(z["eta"])
        pos = {tuple(v): i for i, v in enumerate(e["idx"].tolist())}
        feas_mismatch, obj_diff = 0, 0.0
        for row in arch["assignments"]:
            i = pos[tuple(row["index"])]
            feas_mismatch += int(bool(row["feasible"]) != bool(e["allcut"][i] >= -FEAS_TOL))
            obj_diff = max(obj_diff, abs(row["objective"] - e["bal"][i]), abs(row["objective_U"] - e["tU"][i]),
                           abs(row["objective_W"] - e["tW"][i]))
        r = rep["DET_SEL4"]
        det_ev = bs.evaluate(aB, aA)
        sens = {str(tol): solve_det_arm(bs, bs.K, "det_sel", tol)["index"] for tol in (0.0, 1e-9, 1e-7, 1e-6)}
        chk = {
            "index_match_archive": arch["selected"]["index"] == m["index"],
            "archive_params_exact": bool(np.array_equal(aB, m["_params"][0]) and np.array_equal(aA, m["_params"][1])
                                         and aeta == m["_params"][2]),
            "index_match_report": r["index"] == m["index"],
            "feasible_count_mine": m["feasible_assignments"],
            "feasible_count_archive": arch["feasible_assignments"],
            "per_assignment_feasibility_mismatches": feas_mismatch,
            "per_assignment_objective_maxdiff": float(obj_diff),
            "t_diff_report": num_diff(r["t"], m["t"]),
            "task_balanced_diff_report": num_diff(r["task_balanced"], m["task_balanced"]),
            "report_matches_archive_params_t": num_diff(r["t"], det_ev["t"]),
            "selection_stable_across_tol": len({tuple(v) for v in sens.values()}) == 1,
        }
        ok = (chk["index_match_archive"] and chk["archive_params_exact"] and chk["index_match_report"]
              and chk["feasible_count_mine"] == chk["feasible_count_archive"] and feas_mismatch == 0
              and obj_diff <= 1e-12 and chk["t_diff_report"] <= T_TOL and chk["task_balanced_diff_report"] <= T_TOL)
        anc["releases"]["DET_SEL4"] = {"pass": bool(ok), "checks": chk, "tol_sensitivity_index": sens,
                                       "mine": {k: m[k] for k in ("index", "assignment", "t", "task_U", "task_W",
                                                                  "task_balanced", "feasible")},
                                       "theirs": {k: r.get(k) for k in ("index", "assignment", "t", "task_U",
                                                                        "task_W", "task_balanced", "feasible")}}
        all_pass &= ok

        # ---- LP arms
        lp_res = {}
        for arm in ("R4", "R1", "NM4PF"):
            m = solve_lp_arm(bs, arm)
            B, A, eta, solve = load_theirs(data, a, arm)
            th = bs.evaluate(B, A)
            r = rep[arm]
            struct = structure_check(bs, arm, B, A, eta)
            mA = m["_params"][1]
            law_diff = float(np.abs(A - mA).max())
            if arm == "NM4PF":
                law_diff = max(law_diff, float(np.abs(B - m["_params"][0]).max()), abs(eta - m["_params"][2]))
            chk = {
                "mine_stage1_primal_dual_agree": m["primal_dual_gap"] is not None and abs(m["primal_dual_gap"]) <= 1e-8,
                "t_theirs_params_vs_my_optimum": num_diff(th["t"], m["t_star_primal_ipm"]),
                "t_report_vs_my_optimum": num_diff(r["t"], m["t_star_primal_ipm"]),
                "stage1_t_report_vs_my_optimum": num_diff(r.get("stage1_t"), m["t_star_primal_ipm"]),
                "task_balanced_theirs_params_vs_mine": num_diff(th["task_balanced"], m["task_balanced"]),
                "task_balanced_report_vs_mine": num_diff(r["task_balanced"], m["task_balanced"]),
                "task_U_theirs_vs_mine": num_diff(th["task_U"], m["task_U"]),
                "task_W_theirs_vs_mine": num_diff(th["task_W"], m["task_W"]),
                "report_matches_their_params_t": num_diff(r["t"], th["t"]),
                "report_matches_their_params_task": num_diff(r["task_balanced"], th["task_balanced"]),
                "their_params_feasible": th["feasible"],
                "their_params_structure_ok": struct["ok"],
                "their_structure": struct,
                "my_law_feasible": m["feasible"],
                "my_law_structure_ok": m["structure"]["ok"],
                "my_law_t_minus_tstar": float(m["t"] - m["t_star_primal_ipm"]),
                "law_param_max_abs_diff": law_diff,
                "law_identical_within_1e-6": law_diff <= 1e-6,
            }
            if arm == "NM4PF" and "sc_solver_tau" in r:
                chk["sc_tau_vs_my_optimum"] = num_diff(r["sc_solver_tau"], m["t_star_primal_ipm"])
            ok = (chk["mine_stage1_primal_dual_agree"] and chk["t_theirs_params_vs_my_optimum"] <= T_TOL
                  and chk["t_report_vs_my_optimum"] <= T_TOL and chk["task_balanced_theirs_params_vs_mine"] <= T_TOL
                  and chk["task_balanced_report_vs_mine"] <= T_TOL and th["feasible"] and struct["ok"]
                  and chk["report_matches_their_params_t"] <= T_TOL
                  and chk["report_matches_their_params_task"] <= T_TOL
                  and m["feasible"] and m["structure"]["ok"])
            if "sc_tau_vs_my_optimum" in chk:
                ok = ok and chk["sc_tau_vs_my_optimum"] <= T_TOL
            max_diff["t_lp"] = max(max_diff["t_lp"], chk["t_theirs_params_vs_my_optimum"], chk["t_report_vs_my_optimum"])
            max_diff["task_lp"] = max(max_diff["task_lp"], chk["task_balanced_theirs_params_vs_mine"],
                                      chk["task_balanced_report_vs_mine"])
            max_diff["lp_law_A_maxabs"] = max(max_diff["lp_law_A_maxabs"], law_diff)
            mine_out = {k: v for k, v in m.items() if not k.startswith("_")}
            anc["releases"][arm] = {"pass": bool(ok), "checks": chk, "mine": mine_out,
                                    "theirs": {"t": r["t"], "stage1_t": r.get("stage1_t"),
                                               "t_by_weighting": r.get("t_by_weighting"),
                                               "task_U": r["task_U"], "task_W": r["task_W"],
                                               "task_balanced": r["task_balanced"], "feasible": r["feasible"],
                                               "A": A.tolist(), "eta": eta,
                                               "their_params_evaluated": {k: th[k] for k in (
                                                   "t", "task_U", "task_W", "task_balanced", "feasible",
                                                   "minimum_guard_slack", "task_slack")}}}
            lp_res[arm] = m
            all_pass &= ok

        # ---- structural nesting (my values)
        tD1, tD4 = mine_by_arm["D1"]["t"], mine_by_arm["D4"]["t"]
        tR1, tR4, tNM = (lp_res[k]["t_star_primal_ipm"] for k in ("R1", "R4", "NM4PF"))
        e_ = 1e-9
        nest = {"t_D1_ge_0": tD1 >= -e_, "t_D4_ge_t_D1": tD4 >= tD1 - e_, "t_R4_ge_t_D4": tR4 >= tD4 - e_,
                "t_R4_ge_t_R1": tR4 >= tR1 - e_, "t_R1_ge_t_D1": tR1 >= tD1 - e_, "t_NM4PF_ge_t_R4": tNM >= tR4 - e_}
        anc["nesting_checks_mine"] = nest
        anc["value_of_randomization_fixed_bank_mine"] = tR4 - tD4
        all_pass &= all(nest.values())
        result["anchors"][str(a)] = anc

    result["max_differences"] = max_diff
    result["summary"] = {str(a): {arm: v["pass"] for arm, v in result["anchors"][str(a)]["releases"].items()}
                         for a in ANCHORS}
    result["pass"] = bool(all_pass)
    if out is not None:
        out.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    return result


# ============================================================================ Part B
FAMILIES = ("family_manifest", "secondary_manifest", "capability_manifest")
DEFAULT_SEED = 918273645          # verifier's own seed (NOT the registered 20260926)
DEFAULT_DRAWS = 10_000
H_NAME = "H"


def role_stem(prefix: str, role: str) -> str:
    return f"{prefix}__{role.replace(':', '_').replace('/', '_')}"


class OuterAnchor:
    """One anchor's outer contribution archive, addressed by logical release name."""

    def __init__(self, outer_root: Path, anchor: int, l2c: dict):
        d = Path(outer_root) / f"a{anchor}"
        self.anchor, self.dir, self.l2c = anchor, d, l2c
        self.audit = json.loads((d / "OUTER_AUDIT.json").read_text())
        self.complete = json.loads((d / "COMPLETE.json").read_text())
        try:
            self.npz = np.load(d / "OUTER_CONTRIBUTIONS.npz", allow_pickle=False)
            self.npz_pickle = False
        except ValueError:  # pragma: no cover - object-dtype string arrays
            self.npz = np.load(d / "OUTER_CONTRIBUTIONS.npz", allow_pickle=True)
            self.npz_pickle = True
        self._cache: dict = {}
        confirmed = self.audit.get("outer_alias_confirmed", {}) or {}
        self.alias_disagreements = {k: {"lock": l2c.get(k), "outer": v} for k, v in confirmed.items()
                                    if l2c.get(k) != v}
        self.alias_breaks = self.audit.get("outer_alias_breaks", {}) or {}

    def arrays(self, logical: str, role: str) -> dict:
        canonical = self.l2c[logical]
        key = (canonical, role)
        if key not in self._cache:
            prefix = self.audit["releases"][canonical]["private_contribution_prefix"]
            s = role_stem(prefix, role)
            self._cache[key] = {
                "ids": np.asarray(self.npz[f"{s}_ids"]).astype(str),
                "households": np.asarray(self.npz[f"{s}_households"]).astype(str),
                "weights": np.asarray(self.npz[f"{s}_weights"], float),
                "candidate": np.asarray(self.npz[f"{s}_candidate_loss"], float),
                "H": np.asarray(self.npz[f"{s}_H_loss"], float),
                "canonical": canonical, "stem": s}
        return self._cache[key]

    def contrast(self, plus: str, minus: str, role: str):
        """Per-person D = loss(plus) - loss(minus); 'H' = H-only ancestor loss."""
        ref_name = plus if plus != H_NAME else minus
        ref = self.arrays(ref_name, role)
        def loss(name):
            if name == H_NAME:
                return ref["H"], ref
            x = self.arrays(name, role)
            if not np.array_equal(x["ids"], ref["ids"]):
                raise AssertionError(f"a{self.anchor} {role}: person ids differ between {name} and {ref_name}")
            if not np.array_equal(x["households"], ref["households"]):
                raise AssertionError(f"a{self.anchor} {role}: households differ between {name} and {ref_name}")
            if not np.array_equal(x["weights"], ref["weights"]):
                raise AssertionError(f"a{self.anchor} {role}: weights differ between {name} and {ref_name}")
            return x["candidate"], x
        lp, _ = loss(plus)
        lm, _ = loss(minus)
        return lp - lm, ref["households"], ref["weights"]


def household_counts(rng: np.random.Generator, n_households: int, batch: int) -> np.ndarray:
    """One multinomial resample of the household union per draw (batch x n_households)."""
    return rng.multinomial(n_households, np.full(n_households, 1.0 / n_households), size=batch).astype(float)


def bonferroni_z(m: int, alpha: float = 0.05) -> float:
    from scipy.stats import norm
    return float(norm.isf(alpha / (2 * m)))


def compute_estimates(lock: dict, outer_root: Path, seed: int = DEFAULT_SEED, draws: int = DEFAULT_DRAWS,
                      batch: int = 250, counts_fn=household_counts) -> dict:
    anchors = {a: OuterAnchor(outer_root, a, lock["anchors"][str(a)]["logical_to_canonical"]) for a in ANCHORS}
    endpoints = []
    for fam in FAMILIES:
        for e in lock[fam]["endpoints"]:
            endpoints.append((fam, e))
    # per-person contrasts; household union
    per = {}
    hh_all = set()
    for fam, e in endpoints:
        for a in ANCHORS:
            D, hh, w_p = anchors[a].contrast(e["plus"], e["minus"], e["role"])
            w = np.ones_like(D) if e["weighting"] == "U" else w_p
            if e["weighting"] not in ("U", "PWGTP"):
                raise ValueError(e["weighting"])
            per[(e["id"], a)] = (D, hh, w)
            hh_all.update(hh.tolist())
    union = np.array(sorted(hh_all))
    H = len(union)
    # H-loss identity across releases (diagnostic)
    h_identity = 0.0
    for a, oa in anchors.items():
        by_role: dict = {}
        for (canonical, role), x in list(oa._cache.items()):
            if role in by_role and np.array_equal(by_role[role]["ids"], x["ids"]):
                h_identity = max(h_identity, float(np.abs(by_role[role]["H"] - x["H"]).max()))
            by_role.setdefault(role, x)
    # point estimates and household aggregates (deduplicated columns)
    num_cols, den_cols, num_key, den_key = [], [], {}, {}
    ests = {}
    col_of = {}
    for fam, e in endpoints:
        anc = []
        for a in ANCHORS:
            D, hh, w = per[(e["id"], a)]
            anc.append(float(np.sum(w * D) / np.sum(w)))
            idx = np.searchsorted(union, hh)
            nv = np.bincount(idx, weights=w * D, minlength=H)
            dv = np.bincount(idx, weights=w, minlength=H)
            kn, kd = hashlib.sha256(nv.tobytes()).hexdigest(), hashlib.sha256(dv.tobytes()).hexdigest()
            if kn not in num_key:
                num_key[kn] = len(num_cols); num_cols.append(nv)
            if kd not in den_key:
                den_key[kd] = len(den_cols); den_cols.append(dv)
            col_of[(e["id"], a)] = (num_key[kn], den_key[kd])
        ests[e["id"]] = {"anchor_estimates": anc, "estimate": float(np.mean(anc))}
    N = np.stack(num_cols, axis=1)
    Dn = np.stack(den_cols, axis=1)
    ids = [e["id"] for _, e in endpoints]
    ni = np.array([[col_of[(i, a)][0] for a in ANCHORS] for i in ids])
    di = np.array([[col_of[(i, a)][1] for a in ANCHORS] for i in ids])
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, len(ids)))
    got, attempted, rejected = 0, 0, 0
    while got < draws:
        C = counts_fn(rng, H, min(batch, draws - got))
        attempted += C.shape[0]
        num = C @ N
        den = C @ Dn
        ok = np.all(den != 0, axis=1)
        rejected += int((~ok).sum())
        num, den = num[ok], den[ok]
        ratio = num[:, ni] / den[:, di]           # draws x endpoints x anchors
        b = ratio.mean(axis=2)
        k = min(b.shape[0], draws - got)
        boot[got:got + k] = b[:k]
        got += k
        if attempted > 100 * draws:
            raise RuntimeError("too many zero-denominator draws")
    se = boot.std(axis=0, ddof=1)
    for j, i in enumerate(ids):
        ests[i]["bootstrap_se"] = float(se[j])
    meta = {"seed": seed, "draws": draws, "attempted": attempted, "rejected_zero_denominator": rejected,
            "household_union_size": H, "numerator_columns": N.shape[1], "denominator_columns": Dn.shape[1],
            "max_H_loss_diff_across_releases": h_identity,
            "alias_disagreements": {str(a): oa.alias_disagreements for a, oa in anchors.items()},
            "alias_breaks": {str(a): oa.alias_breaks for a, oa in anchors.items()},
            "npz_needed_pickle": {str(a): oa.npz_pickle for a, oa in anchors.items()}}
    return {"estimates": ests, "meta": meta, "completed_utc": {str(a): oa.complete.get("completed_utc")
                                                              for a, oa in anchors.items()}}


def decision_labels(rows_by_id: dict, lock: dict, passed_key: str, est_key: str) -> dict:
    prim = [e for e in lock["family_manifest"]["endpoints"]]
    out, counts = {}, {}
    for slot, label in (("D", "ARM_D_MEETS_CRITERIA"), ("R", "ARM_R_MEETS_CRITERIA"),
                        ("R_vs_D", "RANDOMIZATION_ADDS")):
        es = [e for e in prim if e["candidate"] == slot]
        passed = [bool(rows_by_id[e["id"]][passed_key]) for e in es]
        out[label] = bool(len(es) == 10 and all(passed))
        counts[slot] = {"passed": int(sum(passed)), "total": len(es)}
    lead = []
    for e in prim:
        if e["candidate"] != "D":
            continue
        est = rows_by_id[e["id"]][est_key]
        if e["role"] == "attack:AB/SEX":
            lead.append(est <= -0.002)
        elif e["role"].startswith("utility:"):
            lead.append(est <= 0.001)
        else:
            lead.append(est <= 0.001)
    out["LEAD_REPRODUCED"] = bool(len(lead) == 10 and all(lead))
    return {"labels": out, "counts": counts}


def part_b(lock_path: Path, outer_root: Path, endpoint_table: Path, inference: Path,
           seed: int = DEFAULT_SEED, draws: int = DEFAULT_DRAWS, counts_fn=household_counts) -> dict:
    lock = json.loads(Path(lock_path).read_text())
    table = json.loads(Path(endpoint_table).read_text())
    inf = json.loads(Path(inference).read_text())
    theirs = {r["id"]: r for r in table["rows"]}
    comp = compute_estimates(lock, outer_root, seed=seed, draws=draws, counts_fn=counts_fn)
    mine = comp["estimates"]
    rows, flips = [], []
    mx = {"estimate": 0.0, "anchor_estimate": 0.0, "their_bounds_vs_their_se": 0.0, "se_rel": 0.0, "se_abs": 0.0}
    fam_checks = {}
    lock_ids = set()
    for fam in FAMILIES:
        F = lock[fam]
        z = float(F["critical_value_two_sided"])
        m = len(F["endpoints"])
        z_re = bonferroni_z(m)
        fam_checks[fam] = {"n_endpoints": m, "locked_z": z, "recomputed_z": z_re,
                           "z_match": abs(z - z_re) <= 1e-12,
                           "n_endpoints_field_match": F.get("n_endpoints", m) == m}
        for e in F["endpoints"]:
            lock_ids.add(e["id"])
            me, th = mine[e["id"]], theirs.get(e["id"])
            lo, up = me["estimate"] - z * me["bootstrap_se"], me["estimate"] + z * me["bootstrap_se"]
            passed = bool(up <= e["threshold"])
            row = {"id": e["id"], "family": fam, "threshold": e["threshold"], "z": z,
                   "estimate": me["estimate"], "anchor_estimates": me["anchor_estimates"],
                   "bootstrap_se": me["bootstrap_se"], "lower": lo, "upper": up, "passed_upper_bound": passed,
                   "margin_upper_minus_threshold": up - e["threshold"]}
            if th is None:
                row["missing_in_endpoint_table"] = True
                flips.append({"id": e["id"], "reason": "missing in ENDPOINT_TABLE"})
                rows.append(row)
                continue
            meta_match = all(th.get(k) == e.get(k) for k in ("plus", "minus", "role", "weighting", "threshold",
                                                             "candidate", "clause"))
            d_est = abs(th["estimate"] - me["estimate"])
            d_anc = max(abs(x - y) for x, y in zip(th["anchor_estimates"], me["anchor_estimates"]))
            their_b = max(abs(th["lower"] - (th["estimate"] - z * th["bootstrap_se"])),
                          abs(th["upper"] - (th["estimate"] + z * th["bootstrap_se"])))
            their_pass_consistent = bool(th["passed_upper_bound"]) == bool(th["upper"] <= e["threshold"])
            se_abs = abs(th["bootstrap_se"] - me["bootstrap_se"])
            se_rel = se_abs / th["bootstrap_se"] if th["bootstrap_se"] > 0 else (0.0 if se_abs <= 1e-15 else float("inf"))
            mx["estimate"] = max(mx["estimate"], d_est)
            mx["anchor_estimate"] = max(mx["anchor_estimate"], d_anc)
            mx["their_bounds_vs_their_se"] = max(mx["their_bounds_vs_their_se"], their_b)
            mx["se_rel"] = max(mx["se_rel"], se_rel)
            mx["se_abs"] = max(mx["se_abs"], se_abs)
            row.update({"theirs": {k: th.get(k) for k in ("estimate", "anchor_estimates", "bootstrap_se", "lower",
                                                         "upper", "passed_upper_bound")},
                        "estimate_abs_diff": d_est, "anchor_estimate_max_abs_diff": d_anc,
                        "se_abs_diff": se_abs, "se_rel_diff": se_rel,
                        "their_bounds_recomputed_max_abs_diff": their_b,
                        "their_pass_consistent_with_their_upper": their_pass_consistent,
                        "endpoint_metadata_match_lock": meta_match,
                        "decision_flip": passed != bool(th["passed_upper_bound"])})
            if row["decision_flip"]:
                flips.append({"id": e["id"], "mine": passed, "theirs": bool(th["passed_upper_bound"]),
                              "my_margin": up - e["threshold"], "their_margin": th["upper"] - e["threshold"]})
            rows.append(row)
    by_id = {r["id"]: r for r in rows}
    my_labels = decision_labels(by_id, lock, "passed_upper_bound", "estimate")
    their_labels = inf["decision_labels"]
    label_agree = {k: (my_labels["labels"][k] == their_labels.get(k)) for k in my_labels["labels"]}
    their_lab_from_table = decision_labels(theirs, lock, "passed_upper_bound", "estimate")
    counts_agree = all(their_labels.get("clauses_passed", {}).get(s) == v["passed"]
                       for s, v in my_labels["counts"].items())
    near = sorted(rows, key=lambda r: abs(r["margin_upper_minus_threshold"]))
    near = [{"id": r["id"], "my_margin": r["margin_upper_minus_threshold"],
             "their_margin": (r["theirs"]["upper"] - r["threshold"]) if "theirs" in r else None}
            for r in near if r["bootstrap_se"] > 0][:8]
    ok_rows = [r for r in rows if "theirs" in r]
    checks = {
        "endpoint_ids_match_lock": set(theirs) == lock_ids,
        "endpoint_metadata_match_lock": all(r["endpoint_metadata_match_lock"] for r in ok_rows),
        "point_estimates_within_1e-12": mx["estimate"] <= 1e-12 and mx["anchor_estimate"] <= 1e-12,
        "their_bounds_consistent_with_their_se_at_locked_z": mx["their_bounds_vs_their_se"] <= 1e-12,
        "their_pass_consistent_with_their_upper": all(r["their_pass_consistent_with_their_upper"] for r in ok_rows),
        "se_within_5pct_mc_tolerance": mx["se_rel"] <= 0.05,
        "locked_z_recomputed": all(v["z_match"] for v in fam_checks.values()),
        "no_decision_flips": len(flips) == 0,
        "labels_agree": all(label_agree.values()),
        "clause_counts_agree": counts_agree,
        "their_labels_consistent_with_their_table": their_lab_from_table["labels"] == {
            k: their_labels.get(k) for k in their_lab_from_table["labels"]},
        "no_alias_disagreements": all(not v for v in comp["meta"]["alias_disagreements"].values()),
        "selection_lock_sha_in_table_and_inference": {
            "endpoint_table": table.get("selection_lock_sha256"), "inference": inf.get("selection_lock_sha256"),
            "lock_file": sha256_file(Path(lock_path))},
    }
    s = checks["selection_lock_sha_in_table_and_inference"]
    checks["selection_lock_sha_consistent"] = s["endpoint_table"] == s["inference"] == s["lock_file"]
    passed_all = all(v for k, v in checks.items() if isinstance(v, bool))
    return {"pass": bool(passed_all), "checks": checks, "families": fam_checks, "max_differences": mx,
            "decision_flips": flips, "nearest_to_threshold": near,
            "labels": {"mine": my_labels, "theirs": {k: their_labels.get(k) for k in my_labels["labels"]},
                       "theirs_clauses_passed": their_labels.get("clauses_passed"), "agree": label_agree},
            "bootstrap": comp["meta"], "outer_completed_utc": comp["completed_utc"], "endpoints": rows}


# ============================================================================ Part C
LOCK_COMMIT = "7fcfaf1a8a70b6fa93b47d8f826ea504c1fe08ed"
LOCK_SHA256 = "d89cfe1bfb3c625ca279bfe256b884f482371ad510aa0892d53026538fc0bc7a"
BRANCH = "research/pcrl-privacy-first-selector-v1"
LOCK_REL = f"results/{STUDY}/SELECTION_LOCK.json"


def _git(repo: Path, *args, binary=False):
    import subprocess
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    return r.returncode, (r.stdout if binary else r.stdout.decode().strip()), r.stderr.decode().strip()


def _utc(s: str):
    from datetime import datetime, timezone
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError(f"naive timestamp {s!r}")
    return d.astimezone(timezone.utc)


def part_c(lock_path: Path, lockcheck: Path, outer_root: Path, inner_root: Path, unlock: Path | None = None,
           restore: Path | None = None, lock_commit: str = LOCK_COMMIT, lock_sha256: str = LOCK_SHA256,
           branch: str = BRANCH, fetch: bool = False, units_root: Path | None = None) -> dict:
    lock = json.loads(Path(lock_path).read_text())
    private = Path(outer_root).parent
    unlock = Path(unlock) if unlock else private / "OUTER_UNLOCK.json"
    restore = Path(restore) if restore else private / "ORIGINAL_RESTORE.json"
    c, notes = {}, []
    # git: lock commit bytes
    rc, blob, err = _git(lockcheck, "show", f"{lock_commit}:{LOCK_REL}", binary=True)
    c["lock_commit_blob_sha256"] = hashlib.sha256(blob).hexdigest() if rc == 0 else f"ERROR: {err}"
    c["lock_commit_bytes_match_pin"] = rc == 0 and c["lock_commit_blob_sha256"] == lock_sha256
    c["local_lock_file_matches_pin"] = sha256_file(Path(lock_path)) == lock_sha256
    rc, ctime, err = _git(lockcheck, "show", "-s", "--format=%cI", lock_commit)
    c["lock_commit_time"] = ctime if rc == 0 else f"ERROR: {err}"
    # on origin
    rc, out, err = _git(lockcheck, "ls-remote", "origin", f"refs/heads/{branch}")
    remote_sha = out.split()[0] if rc == 0 and out else None
    c["remote_branch_sha"] = remote_sha or f"ERROR: {err or 'branch not found'}"
    if remote_sha:
        if _git(lockcheck, "cat-file", "-e", f"{remote_sha}^{{commit}}")[0] != 0 and fetch:
            _git(lockcheck, "fetch", "origin", branch)
        rc, _, err = _git(lockcheck, "merge-base", "--is-ancestor", lock_commit, remote_sha)
        c["lock_commit_on_origin"] = rc == 0
        if rc not in (0, 1):
            notes.append(f"merge-base failed ({err}); remote tip may be missing locally (use --fetch)")
    else:
        c["lock_commit_on_origin"] = False
    # receipts
    u = json.loads(unlock.read_text())
    r = json.loads(restore.read_text())
    c["unlock"] = {k: u.get(k) for k in ("verified_utc", "remote_commit_sha", "lock_sha256")}
    c["unlock_lock_sha_matches_pin"] = u.get("lock_sha256") == lock_sha256
    if u.get("remote_commit_sha"):
        rc, _, err = _git(lockcheck, "merge-base", "--is-ancestor", lock_commit, u["remote_commit_sha"])
        c["lock_commit_ancestor_of_unlock_remote_sha"] = rc == 0
    else:
        c["lock_commit_ancestor_of_unlock_remote_sha"] = False
    c["restore_utc"] = r.get("utc")
    completed = {}
    for a in ANCHORS:
        completed[str(a)] = json.loads((Path(outer_root) / f"a{a}" / "COMPLETE.json").read_text()).get("completed_utc")
    c["outer_completed_utc"] = completed
    try:
        t_lock, t_unlock, t_restore = _utc(ctime), _utc(u["verified_utc"]), _utc(r["utc"])
        t_outer = [_utc(v) for v in completed.values()]
        c["order_lock_before_unlock"] = t_lock < t_unlock
        c["order_unlock_before_restore"] = t_unlock < t_restore
        c["order_restore_before_every_outer_complete"] = all(t_restore < t for t in t_outer)
    except Exception as ex:  # malformed or missing timestamps
        notes.append(f"timestamp parse failed: {ex!r}")
        c["order_lock_before_unlock"] = c["order_unlock_before_restore"] = False
        c["order_restore_before_every_outer_complete"] = False
    # inner pins
    inner = {}
    for a in ANCHORS:
        A = lock["anchors"][str(a)]
        d = Path(inner_root) / f"a{a}"
        got_c = sha256_file(d / "COMPLETE.json") if (d / "COMPLETE.json").exists() else None
        got_i = sha256_file(d / "INNER_AUDIT.json") if (d / "INNER_AUDIT.json").exists() else None
        inner[str(a)] = {"COMPLETE.json": got_c == A["inner_panel_complete_sha256"],
                         "INNER_AUDIT.json": got_i == A["inner_audit_sha256"]}
    c["inner_pins"] = inner
    c["inner_pins_match"] = all(all(v.values()) for v in inner.values())
    # unit pins (optional, if unit dirs exist)
    units_root = Path(units_root) if units_root else Path(inner_root).parent / "units"
    unit_res, unit_missing = {}, []
    for a in ANCHORS:
        for name, rel in lock["anchors"][str(a)].get("releases", {}).items():
            pins = rel.get("pins") or {}
            rid = rel.get("release_id")
            if not pins or not rid:
                continue
            for fname, want in pins.items():
                p = units_root / rid / fname
                if p.exists():
                    unit_res[f"{rid}/{fname}"] = sha256_file(p) == want
                else:
                    unit_missing.append(f"{rid}/{fname}")
    c["unit_pins_checked"] = len(unit_res)
    c["unit_pins_missing"] = unit_missing
    c["unit_pins_match"] = all(unit_res.values()) if unit_res else None
    c["unit_pin_mismatches"] = [k for k, v in unit_res.items() if not v]
    bool_keys = ["lock_commit_bytes_match_pin", "local_lock_file_matches_pin", "lock_commit_on_origin",
                 "unlock_lock_sha_matches_pin", "lock_commit_ancestor_of_unlock_remote_sha",
                 "order_lock_before_unlock", "order_unlock_before_restore",
                 "order_restore_before_every_outer_complete", "inner_pins_match"]
    ok = all(c[k] for k in bool_keys) and c["unit_pins_match"] is not False
    return {"pass": bool(ok), "checks": c, "notes": notes}


def part_bc(args) -> dict:
    b = part_b(args.lock, args.outer_root, args.endpoint_table, args.inference, seed=args.seed, draws=args.draws)
    c = part_c(args.lock, args.lockcheck, args.outer_root, args.inner_root, unlock=args.unlock,
               restore=args.restore, lock_commit=args.lock_commit, lock_sha256=args.lock_sha256,
               branch=args.branch, fetch=args.fetch, units_root=args.units_root)
    res = {"schema": "pcrl-pfs-independent-verification-part-bc-v1", "study": STUDY,
           "pass": bool(b["pass"] and c["pass"]), "part_b": b, "part_c": c}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, sort_keys=True, default=float) + "\n")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="part", required=True)
    pa = sub.add_parser("part_a", help="re-solve the fixed-bank arms")
    pa.add_argument("--data", required=True, type=Path)
    pa.add_argument("--out", type=Path, default=RESULTS / "INDEPENDENT_VERIFICATION_PART_A.json")
    pb = sub.add_parser("part_bc", help="outer estimates/decisions (B) and custody (C)")
    pb.add_argument("--outer-root", required=True, type=Path, help="dir containing a0/ a1/ a2/")
    pb.add_argument("--lock", required=True, type=Path)
    pb.add_argument("--endpoint-table", required=True, type=Path)
    pb.add_argument("--inference", required=True, type=Path)
    pb.add_argument("--lockcheck", required=True, type=Path, help="git checkout with origin remote")
    pb.add_argument("--inner-root", required=True, type=Path, help="inner_panels dir")
    pb.add_argument("--units-root", type=Path, default=None, help="default: <inner-root>/../units")
    pb.add_argument("--unlock", type=Path, default=None, help="default: <outer-root>/../OUTER_UNLOCK.json")
    pb.add_argument("--restore", type=Path, default=None, help="default: <outer-root>/../ORIGINAL_RESTORE.json")
    pb.add_argument("--lock-commit", default=LOCK_COMMIT)
    pb.add_argument("--lock-sha256", default=LOCK_SHA256)
    pb.add_argument("--branch", default=BRANCH)
    pb.add_argument("--fetch", action="store_true", help="git fetch origin <branch> if remote tip is missing")
    pb.add_argument("--seed", type=int, default=DEFAULT_SEED)
    pb.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    pb.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.part == "part_a":
        res = part_a(args.data, args.out)
        print(json.dumps({"pass": res["pass"], "summary": res["summary"], "max_differences": res["max_differences"]},
                         indent=2))
        return 0 if res["pass"] else 1
    if args.part == "part_bc":
        res = part_bc(args)
        b, c = res["part_b"], res["part_c"]
        print(json.dumps({"pass": res["pass"], "part_b_pass": b["pass"],
                          "part_b_checks": {k: v for k, v in b["checks"].items() if isinstance(v, bool)},
                          "max_differences": b["max_differences"], "decision_flips": b["decision_flips"],
                          "labels": b["labels"]["mine"]["labels"], "labels_agree": b["labels"]["agree"],
                          "part_c_pass": c["pass"],
                          "part_c_checks": {k: v for k, v in c["checks"].items() if isinstance(v, bool)},
                          "part_c_notes": c["notes"]}, indent=2, default=float))
        return 0 if res["pass"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
