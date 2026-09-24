"""Independent verifier for the privacy-first selector v1 study.

Written from PROTOCOL.md, PROTOCOL_AMENDMENTS.md and agents/verifier/VERIFIER_BRIEF.md
only. It does not import or read solve.py, host.py, outer.py, lockgate.py, or any
SC/TAC inference/channel code. Allowed dependencies: numpy, scipy, json, hashlib.

Part A (solves): re-derive D4, D1, TASK_SEL4, DET_SEL4 by own enumeration and R4, R1,
NM4PF by own LP formulations (stage 1 primal via HiGHS interior point, cross-checked
with the explicitly constructed dual LP via dual simplex; amendment A1 stage 2 via
interior point), then compare with the implementation's PARAMS/SOLVE outputs and
SOLVE_REPORTS.json.

Parts B (outer estimates) and C (custody) are placeholders to be added later.

Usage:
    python verify_independent.py part_a --data <V> [--out <json>]
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


def part_b(*_args, **_kwargs):
    raise NotImplementedError("Part B (outer point estimates and decisions) not yet implemented")


def part_c(*_args, **_kwargs):
    raise NotImplementedError("Part C (custody) not yet implemented")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="part", required=True)
    pa = sub.add_parser("part_a", help="re-solve the fixed-bank arms")
    pa.add_argument("--data", required=True, type=Path)
    pa.add_argument("--out", type=Path, default=RESULTS / "INDEPENDENT_VERIFICATION_PART_A.json")
    args = ap.parse_args(argv)
    if args.part == "part_a":
        res = part_a(args.data, args.out)
        print(json.dumps({"pass": res["pass"], "summary": res["summary"], "max_differences": res["max_differences"]},
                         indent=2))
        return 0 if res["pass"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
