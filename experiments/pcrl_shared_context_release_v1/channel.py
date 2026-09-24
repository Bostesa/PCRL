"""Nested contextual policy-mixture channel, coefficient blocks and LPs.

For person x with stored old state t=T0(x), hard context k(x) and frozen
deterministic policy tokens d_0(x)..d_{M-1}(x) (d_0 = D17 exactly):

    q(z|x) = B[t,z] + sum_m A[k(x),m] 1{d_m(x)=z}
    B, A >= 0;  sum_z B[t,z] + eta = 1 (every t);  sum_m A[k,m] - eta = 0 (every k)

With a frozen decoder/attack every expected loss is linear in (B, A):

    L = <G_B, B> + <G_A, A>,  G_B[t,z] = sum_{i:T_i=t} w_i l_i(z),
                              G_A[k,m] = sum_{i:k_i=k} w_i l_i(d_m(x_i)).

Cuts are fixed fitted-predictor loss floors (rho - delta) calibrated at the
exact D17 witness (B=D17, A=0, eta=0) on the same coefficient rows.  They are
not population privacy guarantees.  Nothing here loads ACS data.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from experiments.pcrl_adaptive_release_v1.reference import calibrate_reference
from experiments.pcrl_task_aligned_cuts_v1.solver import (
    HIGHS_OPTIONS,
    NONNEGATIVE_TOL,
    PRIMAL_TOL,
    SIMPLEX_TOL,
)

N_STATES = 32
N_TOKENS = 17
LAW_ROW_TOL = 1e-9
SUPPORT_TOL = 1e-12
TASK_ALLOWANCE = .001        # privacy-first task cap relative to D17 (AR mirror)
P_TARGET_ROLE = "AB/SEX"     # privacy-first target role (AR mirror)
WEIGHTINGS = ("U", "W")


# ---------------------------------------------------------------------------
# Parameters and per-person law
# ---------------------------------------------------------------------------

def d17_token_map(d17) -> np.ndarray:
    """Return the one token each old state emits under the deterministic D17."""
    d = np.asarray(d17, dtype=np.float64)
    if (d.ndim != 2 or d.shape[1] != N_TOKENS or not np.isfinite(d).all()
            or not np.all((d == 0) | (d == 1)) or not np.all(d.sum(1) == 1)):
        raise ValueError("D17 must be an exact one-hot state x 17 map")
    return np.argmax(d, axis=1).astype(np.int64)


def validate_params(B, A, eta, *, n_states=N_STATES, tol=LAW_ROW_TOL):
    """Check the nested parameter polytope (shape, sign, both equality blocks)."""
    B = np.asarray(B, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    eta = float(eta)
    if (B.shape != (n_states, N_TOKENS) or A.ndim != 2 or min(A.shape) < 1
            or not np.isfinite(B).all() or not np.isfinite(A).all()
            or not np.isfinite(eta)):
        raise ValueError("nested parameters must be finite with B (S,17) and A (K,M)")
    if B.min() < 0 or A.min() < 0 or not 0 <= eta <= 1:
        raise ValueError("nested parameters must be nonnegative with eta in [0,1]")
    if (np.max(np.abs(B.sum(1) + eta - 1)) > tol
            or np.max(np.abs(A.sum(1) - eta)) > tol):
        raise ValueError("nested equality structure violated")
    return B, A, eta


def d17_params(d17, n_contexts, n_policies):
    """Exact reference witness: eta=0, A=0, B=D17."""
    d17_token_map(d17)
    return (np.asarray(d17, dtype=np.float64).copy(),
            np.zeros((int(n_contexts), int(n_policies))), 0.)


def _aligned(codes, contexts, tokens, n_states, n_contexts, n_policies):
    t = np.asarray(codes)
    k = np.asarray(contexts)
    d = np.asarray(tokens)
    n = len(t)
    if (t.ndim != 1 or t.dtype.kind not in "iu" or np.any(t < 0) or np.any(t >= n_states)
            or k.shape != (n,) or k.dtype.kind not in "iu" or np.any(k < 0)
            or np.any(k >= n_contexts) or d.shape != (n, n_policies)
            or d.dtype.kind not in "iu" or np.any(d < 0) or np.any(d >= N_TOKENS)):
        raise ValueError("codes/contexts/policy tokens must be aligned legal integers")
    return t.astype(np.int64), k.astype(np.int64), d.astype(np.int64)


def validate_law(q, *, tol=LAW_ROW_TOL):
    """Per-person law check: finite, nonnegative, rows sum to one within tol."""
    q = np.asarray(q, dtype=np.float64)
    if (q.ndim != 2 or q.shape[1] != N_TOKENS or not np.isfinite(q).all()
            or q.min() < 0 or np.max(np.abs(q.sum(1) - 1)) > tol):
        raise ValueError("per-person token law must be finite, nonnegative and row-stochastic")
    return q


def person_law(B, A, eta, codes, contexts, tokens):
    """q(z|x_i) = B[t_i] + sum_m A[k_i,m] e_{d_m(x_i)}; validated rows."""
    B, A, eta = validate_params(B, A, eta, n_states=np.asarray(B).shape[0])
    t, k, d = _aligned(codes, contexts, tokens, B.shape[0], A.shape[0], A.shape[1])
    q = B[t].copy()
    rows = np.arange(len(t))
    for m in range(A.shape[1]):
        np.add.at(q, (rows, d[:, m]), A[k, m])
    return validate_law(q)


def deterministic_emission(q, *, tol=SUPPORT_TOL):
    """Report whether every emitted row is one-hot (a deterministic release)."""
    q = validate_law(q)
    one_hot = q.max(1) >= 1 - tol
    return {"all_rows_one_hot": bool(one_hot.all()),
            "one_hot_fraction": float(one_hot.mean()),
            "stochastic_rows": int((~one_hot).sum()),
            "tolerance": tol}


def parameter_sparsity(B, A, eta, *, tol=SUPPORT_TOL):
    """Vertex excess e(B), e(A) and eta, as recommended by the math review."""
    B = np.asarray(B)
    A = np.asarray(A)
    rows_b = (B > tol).sum(1)
    rows_a = (A > tol).sum(1)
    return {"eta": float(eta),
            "excess_B": int(np.sum(np.maximum(rows_b - 1, 0))),
            "excess_A": int(np.sum(np.maximum(rows_a - 1, 0))),
            "fractional_B_states": int(np.sum(rows_b > 1)),
            "fractional_A_contexts": int(np.sum(rows_a > 1)),
            "A_column_mass": np.asarray(A.sum(0)).tolist(),
            "support_tolerance": tol}


# ---------------------------------------------------------------------------
# Coefficient blocks
# ---------------------------------------------------------------------------

def normalized_weights(pwgtp, weighting):
    """Explicit U (1/n) or W (PWGTP/sum PWGTP) weights on stated valid rows."""
    w = np.asarray(pwgtp, dtype=np.float64)
    if w.ndim != 1 or len(w) == 0 or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("finite nonnegative person weights with positive sum required")
    if weighting == "U":
        return np.full(len(w), 1. / len(w))
    if weighting == "W":
        return w / w.sum()
    raise ValueError("weighting must be U or W")


def coefficient_blocks(losses, codes, contexts, tokens, weights, *,
                       n_states=N_STATES, n_contexts):
    """Group per-person token losses into the B (state) and A (context,policy) blocks.

    ``weights`` must already be normalized (sum to one) on exactly these rows.
    """
    losses = np.asarray(losses, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n_policies = np.asarray(tokens).shape[1] if np.asarray(tokens).ndim == 2 else -1
    t, k, d = _aligned(codes, contexts, tokens, n_states, n_contexts, n_policies)
    if (losses.shape != (len(t), N_TOKENS) or not np.isfinite(losses).all()
            or np.any(losses < 0) or w.shape != (len(t),) or not np.isfinite(w).all()
            or np.any(w < 0) or abs(w.sum() - 1) > 1e-12):
        raise ValueError("aligned finite losses and explicitly normalized weights required")
    weighted = losses * w[:, None]
    block_b = np.zeros((n_states, N_TOKENS), dtype=np.float64)
    np.add.at(block_b, t, weighted)
    gathered = np.take_along_axis(weighted, d, axis=1)          # (n, M)
    block_a = np.zeros((n_contexts, n_policies), dtype=np.float64)
    np.add.at(block_a, k, gathered)
    return {"B": block_b, "A": block_a}


def coefficient_pair_blocks(losses, codes, contexts, tokens, pwgtp, *,
                            n_states=N_STATES, n_contexts):
    """U and W blocks on the same valid rows."""
    return {v: coefficient_blocks(losses, codes, contexts, tokens,
                                  normalized_weights(pwgtp, v),
                                  n_states=n_states, n_contexts=n_contexts)
            for v in WEIGHTINGS}


def block_value(blocks, B, A):
    return float(np.sum(blocks["B"] * B) + np.sum(blocks["A"] * A))


def direct_expected_loss(q, losses, weights):
    """Independent replay: sum_i w_i sum_z q_i(z) l_i(z)."""
    q = validate_law(q)
    return float(np.dot(np.asarray(weights, dtype=np.float64),
                        np.einsum("nz,nz->n", q, np.asarray(losses, dtype=np.float64))))


# ---------------------------------------------------------------------------
# Calibration (min-bank D17 reference on the same coefficient rows)
# ---------------------------------------------------------------------------

def _cut_arrays(cut, n_states, n_contexts, n_policies):
    gb = np.asarray(cut.get("coeff_B"), dtype=np.float64)
    ga = np.asarray(cut.get("coeff_A"), dtype=np.float64)
    if (gb.shape != (n_states, N_TOKENS) or ga.shape != (n_contexts, n_policies)
            or not np.isfinite(gb).all() or not np.isfinite(ga).all()
            or gb.min() < 0 or ga.min() < 0):
        raise ValueError("cut blocks must be finite nonnegative (S,17) and (K,M)")
    return gb, ga


def nested_bank_sha256(cuts):
    """Order-independent hash of both coefficient blocks, floors and provenance."""
    digest = hashlib.sha256()
    digest.update(b"SC_NESTED_LOSS_FLOOR_BANK_V1")
    for cut in sorted(cuts, key=lambda item: item["id"]):
        meta = {k: v for k, v in cut.items() if k not in ("coeff_B", "coeff_A", "coeff")}
        digest.update(json.dumps(meta, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False, default=str).encode())
        for key in ("coeff_B", "coeff_A"):
            a = np.ascontiguousarray(cut[key], dtype="<f8")
            digest.update(np.asarray(a.shape, dtype="<i8").tobytes())
            digest.update(a.tobytes())
    return digest.hexdigest()


def calibrate_nested(d17, bank: Sequence[Mapping], delta, *,
                     require_provenance: bool = True, column0_is_d17: bool = True):
    """Rebase every cut to min-bank D17 risk and replay both D17 witnesses.

    rho is computed by the inherited AR routine on the B blocks (the witness
    B=D17, A=0, eta=0 never touches A).  Witness (ii) B=0, A[:,0]=1, eta=1 must
    give the same losses when column 0 is D17 and contexts cover all rows.
    """
    if not bank:
        raise ValueError("nonempty nested attack bank required")
    d17 = np.asarray(d17, dtype=np.float64)
    d17_token_map(d17)
    first = bank[0]
    n_contexts, n_policies = np.asarray(first["coeff_A"]).shape
    blocks = {}
    b_view = []
    for cut in bank:
        gb, ga = _cut_arrays(cut, d17.shape[0], n_contexts, n_policies)
        blocks[cut["id"]] = (gb, ga)
        b_view.append({**{k: v for k, v in cut.items() if k not in ("coeff_B", "coeff_A")},
                       "coeff": gb})
    calibration = calibrate_reference(d17, b_view, delta,
                                      require_provenance=require_provenance)
    cuts = []
    for cut in calibration["cuts"]:
        gb, ga = blocks[cut["id"]]
        cuts.append({**{k: v for k, v in cut.items() if k != "coeff"},
                     "coeff_B": gb, "coeff_A": ga})
    B0, A0, eta0 = d17_params(d17, n_contexts, n_policies)
    witness = replay(B0, A0, eta0, None, cuts)
    if witness["maximum_cut_violation"] > 1e-10:
        raise AssertionError("constructive D17 witness contradicts calibrated nested cuts")
    record = {"witness_eta0": witness}
    if column0_is_d17:
        A1 = np.zeros((n_contexts, n_policies)); A1[:, 0] = 1.
        compact = replay(np.zeros_like(d17), A1, 1., None, cuts)
        mismatch = max(abs(compact["losses"][cid] - witness["losses"][cid]) for cid in witness["losses"])
        if mismatch > 1e-10:
            raise AssertionError("compact D17-column witness disagrees with B=D17 witness; "
                                 "row/weight/context accounting differs between blocks")
        record["witness_eta1_d17_column_max_abs_difference"] = float(mismatch)
    return {"cuts": cuts, "rho": calibration["rho"], **record,
            "bank_sha256": nested_bank_sha256(cuts),
            "reference_objective": "min fixed attack loss at B=D17,A=0,eta=0 on same coefficient rows",
            "population_privacy_guarantee": False}


# ---------------------------------------------------------------------------
# LP assembly
# ---------------------------------------------------------------------------

class _Layout:
    def __init__(self, n_states, n_contexts, n_policies, fix_eta_zero, with_tau):
        self.S, self.K, self.M = n_states, n_contexts, n_policies
        self.fix = bool(fix_eta_zero)
        self.nb = n_states * N_TOKENS
        self.na = 0 if self.fix else n_contexts * n_policies
        self.neta = 0 if self.fix else 1
        self.tau = bool(with_tau)
        self.n = self.nb + self.na + self.neta + (1 if with_tau else 0)

    def vec(self, gb, ga, tau=0.):
        parts = [np.asarray(gb, dtype=np.float64).ravel()]
        if not self.fix:
            parts += [np.asarray(ga, dtype=np.float64).ravel(), np.zeros(1)]
        if self.tau:
            parts.append(np.array([float(tau)]))
        return np.concatenate(parts)

    def equality(self):
        if self.fix:
            base = sparse.kron(sparse.eye(self.S, format="csr"),
                               np.ones((1, N_TOKENS)), format="csr")
            if self.tau:
                base = sparse.hstack((base, sparse.csr_matrix((self.S, 1))), format="csr")
            return base, np.ones(self.S)
        rows_b = sparse.hstack((
            sparse.kron(sparse.eye(self.S), np.ones((1, N_TOKENS))),
            sparse.csr_matrix((self.S, self.na)),
            np.ones((self.S, 1))), format="csr")
        rows_a = sparse.hstack((
            sparse.csr_matrix((self.K, self.nb)),
            sparse.kron(sparse.eye(self.K), np.ones((1, self.M))),
            -np.ones((self.K, 1))), format="csr")
        eq = sparse.vstack((rows_b, rows_a), format="csr")
        if self.tau:
            eq = sparse.hstack((eq, sparse.csr_matrix((eq.shape[0], 1))), format="csr")
        return eq, np.concatenate((np.ones(self.S), np.zeros(self.K)))

    def upper(self):
        u = np.ones(self.n)
        if self.tau:
            u[-1] = np.inf
        return u

    def split(self, x):
        x = np.asarray(x, dtype=np.float64)
        B = x[:self.nb].reshape(self.S, N_TOKENS)
        if self.fix:
            A, eta = np.zeros((self.K, self.M)), 0.
        else:
            A = x[self.nb:self.nb + self.na].reshape(self.K, self.M)
            eta = float(x[self.nb + self.na])
        tau = float(x[-1]) if self.tau else None
        return B, A, eta, tau


def _clean_params(B, A, eta):
    """Clip tiny negatives, snap eta, renormalize rows; return repair size."""
    raw_b, raw_a, raw_eta = B.copy(), A.copy(), float(eta)
    if (not np.isfinite(B).all() or not np.isfinite(A).all() or not np.isfinite(eta)
            or min(B.min(), A.min(), eta) < -NONNEGATIVE_TOL or eta > 1 + NONNEGATIVE_TOL):
        return None
    B = np.maximum(B, 0.)
    A = np.maximum(A, 0.)
    eta = min(max(eta, 0.), 1.)
    if eta < SUPPORT_TOL:
        eta = 0.
    elif eta > 1 - SUPPORT_TOL:
        eta = 1.
    mass_b = B.sum(1)
    mass_a = A.sum(1)
    if (np.max(np.abs(mass_b - (1 - raw_eta))) > SIMPLEX_TOL
            or np.max(np.abs(mass_a - raw_eta)) > SIMPLEX_TOL):
        return None
    if eta == 1.:
        B = np.zeros_like(B)
    else:
        if np.any(mass_b <= 0):
            return None
        B = B / mass_b[:, None] * (1 - eta)     # division first: bit-identical to TAC _clean at eta=0
    if eta == 0.:
        A = np.zeros_like(A)
    else:
        if np.any(mass_a <= 0):
            return None
        A = A / mass_a[:, None] * eta
    repair = float(max(np.max(np.abs(B - raw_b)), np.max(np.abs(A - raw_a), initial=0.),
                       abs(eta - raw_eta)))
    return B, A, eta, repair


def replay(B, A, eta, cost_blocks, cuts, *, tau=None, task_caps=None):
    """Independent primal replay of stored parameters against stored blocks."""
    B = np.asarray(B, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    losses, slacks = {}, {}
    for cut in cuts:
        value = float(np.sum(cut["coeff_B"] * B) + np.sum(cut["coeff_A"] * A))
        losses[cut["id"]] = value
        floor = float(cut["floor"])
        if tau is not None and cut.get("role") == P_TARGET_ROLE:
            floor = float(cut["rho"]) + float(tau)
        slacks[cut["id"]] = value - floor
    out = {"losses": losses, "slacks": slacks,
           "maximum_cut_violation": float(max((max(0., -s) for s in slacks.values()), default=0.)),
           "minimum_cut_slack": float(min(slacks.values())) if slacks else None,
           "B_simplex_residual": float(np.max(np.abs(B.sum(1) + eta - 1))),
           "A_simplex_residual": float(np.max(np.abs(A.sum(1) - eta))),
           "minimum_entry": float(min(B.min(), A.min(), eta)),
           "eta": float(eta)}
    if cost_blocks is not None:
        out["objective_U"] = block_value(cost_blocks["U"], B, A)
        out["objective_W"] = block_value(cost_blocks["W"], B, A)
        out["objective"] = 0.5 * (out["objective_U"] + out["objective_W"])
    if task_caps is not None:
        out["task_cap_violation"] = float(max(
            max(0., out[f"objective_{v}"] - task_caps[v]) for v in WEIGHTINGS))
    return out


def _feasible(rep):
    ok = (rep["maximum_cut_violation"] <= PRIMAL_TOL
          and rep["B_simplex_residual"] <= SIMPLEX_TOL
          and rep["A_simplex_residual"] <= SIMPLEX_TOL
          and rep["minimum_entry"] >= -NONNEGATIVE_TOL)
    if "task_cap_violation" in rep:
        ok = ok and rep["task_cap_violation"] <= PRIMAL_TOL
    return bool(ok)


def _dual_bound(raw, c, a_ub, b_ub, a_eq, b_eq, upper):
    """Reconstructed Lagrangian lower bound from HiGHS marginals (0<=x<=u)."""
    if raw.status != 0 or raw.x is None:
        return None
    y_eq = np.asarray(raw.eqlin.marginals, dtype=float)
    y_ub = (np.minimum(np.asarray(raw.ineqlin.marginals, dtype=float), 0.)
            if a_ub is not None else np.zeros(0))
    reduced = c - a_eq.T @ y_eq
    if a_ub is not None:
        reduced = reduced - a_ub.T @ y_ub
    negative = np.minimum(reduced, 0.)
    if np.any((negative < -1e-12) & ~np.isfinite(upper)):
        bound = None
    else:
        finite = np.isfinite(upper)
        bound = float(np.dot(b_eq, y_eq) + (np.dot(b_ub, y_ub) if a_ub is not None else 0.)
                      + np.dot(negative[finite], upper[finite]))
    return {"lower_bound": bound, "minimum_reduced_cost": float(np.min(reduced)),
            "inequality_multipliers": (-y_ub).tolist(),
            "maximum_inequality_multiplier": float(np.max(-y_ub, initial=0.)),
            "source": "reconstructed HiGHS dual marginals; box-bounded reduced-cost guard",
            "not_interval_arithmetic": True}


def _shape(cost_blocks, cuts, d17):
    d17 = np.asarray(d17, dtype=np.float64)
    d17_token_map(d17)
    if set(cost_blocks) != set(WEIGHTINGS):
        raise ValueError("task cost blocks require exactly U and W")
    K, M = np.asarray(cost_blocks["U"]["A"]).shape
    for v in WEIGHTINGS:
        if (np.asarray(cost_blocks[v]["B"]).shape != d17.shape
                or np.asarray(cost_blocks[v]["A"]).shape != (K, M)
                or not np.isfinite(cost_blocks[v]["B"]).all()
                or not np.isfinite(cost_blocks[v]["A"]).all()):
            raise ValueError("task cost block shapes differ")
    ids = set()
    for cut in cuts:
        _cut_arrays(cut, d17.shape[0], K, M)
        if cut["id"] in ids or not np.isfinite(float(cut["floor"])):
            raise ValueError("unique cut IDs with finite floors required")
        ids.add(cut["id"])
    return d17, K, M


def _cut_matrix(layout, cuts, *, tau_role=None, slack=False):
    """Rows for -L_a(x) [+ tau for target] [- s] <= -floor."""
    width = layout.n + (1 if slack else 0)
    rows, rhs = [], []
    for cut in cuts:
        row = np.zeros(width)
        row[:layout.n] = -layout.vec(cut["coeff_B"], cut["coeff_A"])
        if slack:
            row[-1] = -1.
        if tau_role is not None and cut.get("role") == tau_role:
            if not slack:
                row[layout.n - 1] = 1.
            rhs.append(-float(cut["rho"]))
        else:
            rhs.append(-float(cut["floor"]))
        rows.append(row)
    if not rows:
        return None, None
    return np.vstack(rows), np.asarray(rhs)


def phase_one(cost_blocks, cuts, d17, *, fix_eta_zero=False, privacy_first=False,
              time_limit=None):
    """Minimize a common nonnegative relaxation of every clause (tau=0 for P)."""
    d17, K, M = _shape(cost_blocks, cuts, d17)
    layout = _Layout(d17.shape[0], K, M, fix_eta_zero, privacy_first)
    a_ub, b_ub = _cut_matrix(layout, cuts, tau_role=P_TARGET_ROLE if privacy_first else None,
                             slack=True)
    rows = [a_ub] if a_ub is not None else []
    rhs = [b_ub] if b_ub is not None else []
    if privacy_first:
        for v in WEIGHTINGS:
            row = np.zeros(layout.n + 1)
            row[:layout.n] = layout.vec(cost_blocks[v]["B"], cost_blocks[v]["A"])
            row[-1] = -1.
            rows.append(row[None, :])
            rhs.append(np.array([float(np.sum(cost_blocks[v]["B"] * d17)) + TASK_ALLOWANCE]))
    a_ub = np.vstack(rows) if rows else None
    b_ub = np.concatenate(rhs) if rhs else None
    a_eq, b_eq = layout.equality()
    a_eq = sparse.hstack((a_eq, sparse.csr_matrix((a_eq.shape[0], 1))), format="csr")
    c = np.zeros(layout.n + 1); c[-1] = 1.
    upper = [(0., 1.)] * layout.n + [(0., None)]
    if privacy_first:
        upper[layout.n - 1] = (0., 0.)     # tau pinned at zero in phase I
    options = dict(HIGHS_OPTIONS)
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    raw = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=upper,
                  method="highs", options=options)
    residual = None
    if raw.x is not None:
        residual = {"maximum_inequality_violation": float(np.max(np.maximum(a_ub @ raw.x - b_ub, 0.), initial=0.)) if a_ub is not None else 0.,
                    "maximum_equality_residual": float(np.max(np.abs(a_eq @ raw.x - b_eq)))}
    return {"status": "optimal" if raw.status == 0 else
            {1: "time_or_iteration_limit", 2: "infeasible", 3: "unbounded"}.get(raw.status, "solver_error"),
            "minimum_common_violation": float(raw.x[-1]) if raw.x is not None else None,
            "solver_message": raw.message, "residual": residual, "solver": "HiGHS"}


def _solve(c, a_ub, b_ub, layout, bounds, time_limit):
    a_eq, b_eq = layout.equality()
    options = dict(HIGHS_OPTIONS)
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    raw = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds,
                  method="highs", options=options)
    return raw, a_eq, b_eq


def solve_utility(cost_blocks, cuts, d17, *, fix_eta_zero=False, time_limit=None):
    """min 0.5 C_U + 0.5 C_W over (B,A,eta) s.t. calibrated cuts and equalities.

    With fix_eta_zero the variable vector is exactly vec(B) and the matrices are
    those of the inherited T32 LP (`TAC/solver.solve_p1`), so identical inputs
    reproduce the parent-restricted T32 solution.
    """
    d17, K, M = _shape(cost_blocks, cuts, d17)
    layout = _Layout(d17.shape[0], K, M, fix_eta_zero, False)
    witness = replay(*d17_params(d17, K, M), cost_blocks, cuts)
    if not _feasible(witness):
        raise ValueError("D17 witness infeasible on supplied calibrated cuts")
    phase = phase_one(cost_blocks, cuts, d17, fix_eta_zero=fix_eta_zero, time_limit=time_limit)
    base = {"form": "U", "fix_eta_zero": bool(fix_eta_zero), "witness": _brief(witness),
            "phase_one": phase, "bank_sha256": nested_bank_sha256(cuts),
            "variables": layout.n}
    if phase["status"] != "optimal" or phase["minimum_common_violation"] is None:
        return {**base, "status": "phase_one_unresolved", "feasible": False, "params": None}
    if phase["minimum_common_violation"] > PRIMAL_TOL:
        raise RuntimeError("phase I contradicts constructive D17 witness; inspect routing")
    cost = {v: layout.vec(cost_blocks[v]["B"], cost_blocks[v]["A"]) for v in WEIGHTINGS}
    c = 0.5 * (cost["U"] + cost["W"])
    a_ub, b_ub = _cut_matrix(layout, cuts)
    bounds = (0, None) if fix_eta_zero else [(0., None)] * layout.n
    raw, a_eq, b_eq = _solve(c, a_ub, b_ub, layout, bounds, time_limit)
    return _finish(base, raw, layout, c, a_ub, b_ub, a_eq, b_eq, cost_blocks, cuts, None)


def solve_privacy_first(cost_blocks, cuts, d17, *, fix_eta_zero=False, time_limit=60.):
    """max tau s.t. AB/SEX cuts >= rho + tau, others >= rho - delta, task caps.

    Task caps are C_v(q) <= C_v(D17) + 0.001 separately for U and W (AR mirror).
    """
    d17, K, M = _shape(cost_blocks, cuts, d17)
    layout = _Layout(d17.shape[0], K, M, fix_eta_zero, True)
    caps = {v: float(np.sum(cost_blocks[v]["B"] * d17)) + TASK_ALLOWANCE for v in WEIGHTINGS}
    witness = replay(*d17_params(d17, K, M), cost_blocks, cuts, tau=0., task_caps=caps)
    if not _feasible(witness):
        raise ValueError("D17,tau=0 witness infeasible on supplied calibrated cuts")
    phase = phase_one(cost_blocks, cuts, d17, fix_eta_zero=fix_eta_zero,
                      privacy_first=True, time_limit=time_limit)
    base = {"form": "P", "fix_eta_zero": bool(fix_eta_zero), "witness": _brief(witness),
            "phase_one": phase, "task_caps": caps, "bank_sha256": nested_bank_sha256(cuts),
            "variables": layout.n}
    if phase["status"] != "optimal" or phase["minimum_common_violation"] is None:
        return {**base, "status": "phase_one_unresolved", "feasible": False, "params": None}
    if phase["minimum_common_violation"] > PRIMAL_TOL:
        raise RuntimeError("phase I contradicts constructive D17,tau=0 witness")
    a_ub, b_ub = _cut_matrix(layout, cuts, tau_role=P_TARGET_ROLE)
    cap_rows = np.vstack([layout.vec(cost_blocks[v]["B"], cost_blocks[v]["A"]) for v in WEIGHTINGS])
    a_ub = np.vstack((a_ub, cap_rows)) if a_ub is not None else cap_rows
    b_ub = np.concatenate((b_ub if b_ub is not None else np.zeros(0),
                           [caps[v] for v in WEIGHTINGS]))
    c = np.zeros(layout.n); c[-1] = -1.
    bounds = [(0., None)] * layout.n
    raw, a_eq, b_eq = _solve(c, a_ub, b_ub, layout, bounds, time_limit)
    return _finish(base, raw, layout, c, a_ub, b_ub, a_eq, b_eq, cost_blocks, cuts, caps)


def _brief(rep):
    return {k: v for k, v in rep.items() if k not in ("losses", "slacks")}


def _finish(base, raw, layout, c, a_ub, b_ub, a_eq, b_eq, cost_blocks, cuts, caps):
    if raw.x is None:
        return {**base, "status": "solver_failed", "feasible": False, "params": None,
                "solver_status": int(raw.status), "solver_message": raw.message}
    B, A, eta, tau = layout.split(raw.x)
    cleaned = _clean_params(B, A, eta)
    if cleaned is None:
        return {**base, "status": "solver_failed", "feasible": False, "params": None,
                "solver_status": int(raw.status), "solver_message": raw.message}
    B, A, eta, repair = cleaned
    if tau is not None:
        tau = max(tau, 0.)
    rep = replay(B, A, eta, cost_blocks, cuts, tau=tau, task_caps=caps)
    feasible = _feasible(rep) and repair <= SIMPLEX_TOL and raw.status == 0
    residual = {"maximum_inequality_violation": float(np.max(np.maximum(a_ub @ raw.x - b_ub, 0.), initial=0.)) if a_ub is not None else 0.,
                "maximum_equality_residual": float(np.max(np.abs(a_eq @ raw.x - b_eq)))}
    dual = _dual_bound(raw, c, a_ub, b_ub, a_eq, b_eq, layout.upper())
    record = {**base, "status": "optimal" if feasible else
              ("infeasible_replay" if raw.status == 0 else "solver_unresolved"),
              "feasible": bool(feasible), "params": {"B": B, "A": A, "eta": eta},
              "tau": tau, "solver": "HiGHS", "solver_status": int(raw.status),
              "solver_message": raw.message, "solver_reported_objective": float(raw.fun),
              "solver_residual": residual, "repair_max": repair,
              "replay": _brief(rep), "cut_losses": rep["losses"],
              "sparsity": parameter_sparsity(B, A, eta)}
    if dual is not None:
        multipliers = dual.pop("inequality_multipliers")
        record["dual"] = dual
        record["cut_multipliers"] = {cut["id"]: float(m) for cut, m in zip(cuts, multipliers[:len(cuts)])}
        if base["form"] == "U":
            record["dual_lower_bound"] = dual["lower_bound"]
            record["fixed_bank_gap"] = (rep["objective"] - dual["lower_bound"]
                                        if dual["lower_bound"] is not None else None)
        else:
            record["dual_tau_upper_bound"] = (-dual["lower_bound"]
                                              if dual["lower_bound"] is not None else None)
            record["dual_gap"] = (record["dual_tau_upper_bound"] - tau
                                  if record["dual_tau_upper_bound"] is not None else None)
    return record


def check_feasible(params, cost_blocks, cuts, *, tau=None, task_caps=None):
    """Final-bank replay used by round selection and the deterministic selector."""
    rep = replay(params["B"], params["A"], params["eta"], cost_blocks, cuts,
                 tau=tau, task_caps=task_caps)
    return {**_brief(rep), "slacks": rep["slacks"], "feasible": _feasible(rep)}


# ---------------------------------------------------------------------------
# Nonalias diagnostics (never eta): TV to D17 and within-T32 spread
# ---------------------------------------------------------------------------

def t32_projection(q, codes, weights, *, n_states=N_STATES):
    """State means q_bar_t = sum_{i in t} w_i q_i / W_t (a T32 kernel).

    Unvisited states keep a uniform row so the result is a valid kernel.
    """
    q = validate_law(q)
    t = np.asarray(codes, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    mass = np.bincount(t, weights=w, minlength=n_states)
    total = np.zeros((n_states, N_TOKENS))
    np.add.at(total, t, q * w[:, None])
    proj = np.full((n_states, N_TOKENS), 1. / N_TOKENS)
    seen = mass > 0
    proj[seen] = total[seen] / mass[seen, None]
    proj /= proj.sum(1, keepdims=True)
    return proj, seen


def _max_pairwise_tv(rows):
    unique = np.unique(np.round(rows, 12), axis=0)
    if len(unique) < 2:
        return 0.
    best = 0.
    for start in range(0, len(unique), 256):
        block = unique[start:start + 256]
        tv = 0.5 * np.abs(block[:, None, :] - unique[None, :, :]).sum(2)
        best = max(best, float(tv.max()))
    return best


def nonalias_diagnostic(q, codes, d17, pwgtp, households, *, tol=1e-9,
                        n_states=N_STATES):
    """Weighted/unweighted TV to D17 and within-T32 variation, affected households."""
    q = validate_law(q)
    t = np.asarray(codes, dtype=np.int64)
    w = np.asarray(pwgtp, dtype=np.float64)
    hh = np.asarray(households).astype(str)
    d17 = np.asarray(d17, dtype=np.float64)
    if t.shape != (len(q),) or w.shape != (len(q),) or hh.shape != (len(q),):
        raise ValueError("aligned codes/weights/households required")
    wn = w / w.sum()
    tv_d17 = 0.5 * np.abs(q - d17[t]).sum(1)
    proj_u, seen = t32_projection(q, t, np.ones(len(q)), n_states=n_states)
    proj_w, _ = t32_projection(q, t, w, n_states=n_states)
    tv_mean_u = 0.5 * np.abs(q - proj_u[t]).sum(1)
    tv_mean_w = 0.5 * np.abs(q - proj_w[t]).sum(1)
    per_state = []
    for s in np.flatnonzero(seen):
        members = t == s
        per_state.append({"state": int(s), "people": int(members.sum()),
                          "V_unweighted": float(tv_mean_u[members].mean()),
                          "V_weighted": float(np.dot(w[members] / w[members].sum(), tv_mean_w[members])),
                          "max_pairwise_tv": _max_pairwise_tv(q[members]),
                          "varying_households": int(len(np.unique(hh[members & (tv_mean_u > tol)])))})
    v_states = np.asarray([row["V_unweighted"] for row in per_state])
    det = deterministic_emission(q)
    return {
        "people": int(len(q)), "households": int(len(np.unique(hh))), "tolerance": tol,
        "tv_to_d17": {"unweighted_mean": float(tv_d17.mean()),
                      "weighted_mean": float(np.dot(wn, tv_d17)),
                      "max": float(tv_d17.max()),
                      "affected_people": int((tv_d17 > tol).sum()),
                      "affected_unique_households": int(len(np.unique(hh[tv_d17 > tol])))},
        "within_t32": {"V_unweighted": float(tv_mean_u.mean()),
                       "V_weighted": float(np.dot(wn, tv_mean_w)),
                       "max_tv_to_state_mean": float(max(tv_mean_u.max(), tv_mean_w.max())),
                       "max_pairwise_tv": float(max((r["max_pairwise_tv"] for r in per_state), default=0.)),
                       "affected_people": int((tv_mean_u > tol).sum()),
                       "affected_unique_households": int(len(np.unique(hh[tv_mean_u > tol]))),
                       "states_with_variation": int(sum(r["max_pairwise_tv"] > tol for r in per_state)),
                       "per_state_V_quantiles": {str(p): float(np.quantile(v_states, p))
                                                 for p in (0, .25, .5, .75, .9, 1.)} if len(v_states) else {},
                       "per_state": per_state},
        "deterministic_emission": det,
        "stochastic_weight_fraction": float(np.dot(wn, (q.max(1) < 1 - SUPPORT_TOL))),
        "interpretation": "eta is not identified; within-T32 spread is the registered nonalias gate",
    }, {"U": proj_u, "W": proj_w}


def projection_rescore(projections, cost_blocks, cuts):
    """Re-score the within-T32-averaged law with the same frozen decoder/bank.

    The projection is a T32 kernel, so only B blocks are used (A=0, eta=0).
    U coefficients use the unweighted projection and W the PWGTP projection.
    """
    out = {}
    for v in WEIGHTINGS:
        out[f"task_{v}"] = float(np.sum(cost_blocks[v]["B"] * projections[v]))
    out["task"] = 0.5 * (out["task_U"] + out["task_W"])
    slacks = {}
    for cut in cuts:
        proj = projections[cut["weighting"]]
        slacks[cut["id"]] = float(np.sum(cut["coeff_B"] * proj)) - float(cut["floor"])
    out["minimum_cut_slack"] = float(min(slacks.values())) if slacks else None
    out["maximum_cut_violation"] = float(max((max(0., -s) for s in slacks.values()), default=0.))
    out["feasible_on_bank"] = bool(out["maximum_cut_violation"] <= PRIMAL_TOL)
    return out
