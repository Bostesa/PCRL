"""Finite local score channels on p(S,T1,T2,Y1,Y2).

Each sampler reads only its own input bin and uses an independent random
draw. Public outputs contain the sampled bank ID and probability vector.
The coordinate LP is exact for one channel given the other; alternating it
is a global heuristic, and finite-law Bayes accuracy is not Adult attack AUC.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

CLIP = 1e-12
TOL = 1e-9
VIEWS = ("r1", "r2", "pair")


@dataclass(frozen=True)
class FiniteLaw:
    mass: np.ndarray
    bank: tuple[np.ndarray, np.ndarray]
    input_class: tuple[np.ndarray, np.ndarray]
    output_class: tuple[np.ndarray, np.ndarray]
    n_input: tuple[int, int]
    n_output: tuple[int, int]

    def __init__(self, mass, bank1, bank2, input_class1, input_class2):
        p = np.asarray(mass, dtype=np.float64)
        if p.ndim != 5 or p.shape[0] != 2 or min(p.shape) < 1:
            raise ValueError("mass must have shape (2,T1,T2,Y1,Y2)")
        if (not np.all(np.isfinite(p)) or np.any(p < 0)
                or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0)):
            raise ValueError("mass must be a normalized nonnegative finite law")
        banks = tuple(np.asarray(b, dtype=np.float64) for b in (bank1, bank2))
        classes = tuple(np.asarray(c, dtype=np.int64) for c in (input_class1, input_class2))
        for r, (b, c) in enumerate(zip(banks, classes)):
            k = p.shape[3 + r]
            if b.ndim != 2 or b.shape[1] != k or b.shape[0] < 1:
                raise ValueError("bank shape does not match task classes")
            if c.shape != (p.shape[1 + r],) or np.any((c < 0) | (c >= k)):
                raise ValueError("input class shape or value invalid")
            if (np.any(~np.isfinite(b)) or np.any(b < 0)
                    or not np.allclose(b.sum(axis=1), 1, atol=1e-12, rtol=0)):
                raise ValueError("bank rows must be probability vectors")
            out_class = np.argmax(b, axis=1)
            if k > 1:
                others = np.where(np.eye(k, dtype=bool)[out_class], -np.inf, b)
                if np.any(b[np.arange(len(b)), out_class] <= np.max(others, axis=1)):
                    raise ValueError("every bank vector needs a strict predicted class")
            if not set(c).issubset(set(out_class)):
                raise ValueError("bank lacks a code for an input predicted class")
        object.__setattr__(self, "mass", p)
        object.__setattr__(self, "bank", banks)
        object.__setattr__(self, "input_class", classes)
        object.__setattr__(self, "output_class", tuple(np.argmax(b, axis=1) for b in banks))
        object.__setattr__(self, "n_input", (p.shape[1], p.shape[2]))
        object.__setattr__(self, "n_output", (len(banks[0]), len(banks[1])))


def _validate_q(law: FiniteLaw, r: int, q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (law.n_input[r], law.n_output[r]):
        raise ValueError("channel shape does not match finite law")
    allowed = law.input_class[r][:, None] == law.output_class[r][None, :]
    if (not np.all(np.isfinite(q)) or np.any(q < 0)
            or not np.allclose(q.sum(axis=1), 1, atol=TOL, rtol=0)
            or np.any(np.abs(q[~allowed]) > TOL)):
        raise ValueError("channel violates stochastic or predicted-class contract")
    return q


def _bayes(p_s_code):
    """Optimal binary SEX classification accuracy from disclosed codes."""
    return float(np.max(p_s_code, axis=0).sum())


def _accuracy_from_counts(p):
    return {"r1": _bayes(p.sum(axis=2)),
            "r2": _bayes(p.sum(axis=1)), "pair": _bayes(p)}


def input_recovery(law: FiniteLaw):
    """Bayes accuracy if both private input-bin identities were disclosed."""
    return _accuracy_from_counts(law.mass.sum(axis=(3, 4)))


def class_floor(law: FiniteLaw):
    """Bayes accuracy from the disclosed decisions alone."""
    c1, c2 = law.input_class
    p = law.mass.sum(axis=(3, 4))
    grouped = np.zeros((2, law.bank[0].shape[1], law.bank[1].shape[1]))
    for t1 in range(law.n_input[0]):
        for t2 in range(law.n_input[1]):
            grouped[:, c1[t1], c2[t2]] += p[:, t1, t2]
    return _accuracy_from_counts(grouped)


def _costs(law: FiniteLaw, r: int):
    """Expected task-loss coefficient for each input/output code edge."""
    py = (law.mass.sum(axis=(0, 2, 4)) if r == 0
          else law.mass.sum(axis=(0, 1, 3)))
    bank = law.bank[r]
    neg_log = -np.log(np.clip(bank, CLIP, 1))
    brier_by_y = np.sum(
        (bank[:, None, :] - np.eye(bank.shape[1])[None, :, :]) ** 2, axis=2)
    return {"nll": py @ neg_log.T, "brier": py @ brier_by_y.T}


def evaluate(law: FiniteLaw, q1, q2):
    """Exact fitted-law expectation; never NLL of an unsent mean vector."""
    q1 = _validate_q(law, 0, q1)
    q2 = _validate_q(law, 1, q2)
    p = law.mass.sum(axis=(3, 4))
    released = np.einsum("sab,ai,bj->sij", p, q1, q2, optimize=True)
    utility = {}
    for r, q in enumerate((q1, q2)):
        utility[f"r{r + 1}"] = {
            name: float(np.sum(q * c)) for name, c in _costs(law, r).items()}
    return {"accuracy": _accuracy_from_counts(released), "utility": utility}


def sample_release(law: FiniteLaw, *, recipient: str, input_tokens, q, rng):
    """Generate one local draw per input and expose its bank code identity."""
    r = {"r1": 0, "r2": 1}.get(recipient)
    if r is None:
        raise ValueError("recipient must be r1 or r2")
    q = _validate_q(law, r, q)
    tokens = np.asarray(input_tokens, dtype=np.int64)
    if tokens.ndim != 1 or np.any((tokens < 0) | (tokens >= law.n_input[r])):
        raise ValueError("input tokens are out of range")
    cdf = np.cumsum(q[tokens], axis=1)
    z = np.sum(rng.random(len(tokens))[:, None] > cdf, axis=1)
    z = np.minimum(z, law.n_output[r] - 1).astype(np.int64)
    return {"code_id": z, "probability": law.bank[r][z],
            "decision": law.output_class[r][z]}


def solve_coordinate(law: FiniteLaw, q1, q2, *, recipient: str, caps: dict,
                     time_limit: float | None = None,
                     deterministic: bool = False):
    """Minimize normalized max local/pair Bayes recovery in one LP coordinate.

    The other recipient's channel is fixed. This program is convex and solved
    by HiGHS; alternating coordinates is not a global certificate.
    """
    q1 = _validate_q(law, 0, q1)
    q2 = _validate_q(law, 1, q2)
    r = {"r1": 0, "r2": 1}.get(recipient)
    if r is None:
        raise ValueError("recipient must be r1 or r2")
    for name in ("r1", "r2"):
        if name not in caps or set(caps[name]) != {"nll", "brier"}:
            raise ValueError("both task NLL/Brier caps are required")
    original = evaluate(law, q1, q2)
    other_name = f"r{2 - r}"
    for metric in ("nll", "brier"):
        if original["utility"][other_name][metric] > caps[other_name][metric] + TOL:
            return {"success": False, "status": "fixed-other utility infeasible", "q": None}

    n_t, n_z = law.n_input[r], law.n_output[r]
    q_other = q2 if r == 0 else q1
    n_other_z = q_other.shape[1]
    allowed = law.input_class[r][:, None] == law.output_class[r][None, :]
    ti, zi = np.nonzero(allowed)
    n_q = len(ti)
    q_index = np.full((n_t, n_z), -1, dtype=int)
    q_index[ti, zi] = np.arange(n_q)
    loc_u = np.arange(n_q, n_q + n_z)
    pair_u = np.arange(n_q + n_z, n_q + n_z + n_z * n_other_z).reshape(n_z, n_other_z)
    t_index = n_q + n_z + n_z * n_other_z
    n_var = t_index + 1

    floor, raw = class_floor(law), input_recovery(law)
    gap = {v: raw[v] - floor[v] for v in VIEWS}
    local_name = f"r{r + 1}"
    delta = law.mass[1].sum(axis=(2, 3)) - law.mass[0].sum(axis=(2, 3))
    if r == 0:
        local_coeff = delta.sum(axis=1)
        pair_coeff = delta @ q_other  # T1,Z2
    else:
        local_coeff = delta.sum(axis=0)
        pair_coeff = delta.T @ q_other  # T2,Z1

    rows, cols, vals, rhs = [], [], [], []

    def add_row(pairs, bound):
        j = len(rhs)
        for col, val in pairs:
            if val != 0:
                rows.append(j)
                cols.append(int(col))
                vals.append(float(val))
        rhs.append(float(bound))

    # Every signed SEX-mass difference is bounded by its absolute-value
    # auxiliary. The bank ID, including identical-vector duplicate IDs, is
    # the released symbol; pair auxiliaries index ordered bank-ID pairs.
    for z in range(n_z):
        for sign in (1, -1):
            entries = [(q_index[t, z], sign * local_coeff[t])
                       for t in range(n_t) if allowed[t, z]]
            add_row(entries + [(loc_u[z], -1)], 0)
    for z in range(n_z):
        for zo in range(n_other_z):
            for sign in (1, -1):
                entries = [(q_index[t, z], sign * pair_coeff[t, zo])
                           for t in range(n_t) if allowed[t, z]]
                add_row(entries + [(pair_u[z, zo], -1)], 0)

    # G_v = (1 + sum |p(S=1,z)-p(S=0,z)|)/2. For a zero gap, this
    # enforces G_v <= F_v directly instead of dividing by zero.
    add_row([(j, 1) for j in loc_u] + [(t_index, -2 * gap[local_name])],
            2 * floor[local_name] - 1)
    add_row([(j, 1) for j in pair_u.flat] + [(t_index, -2 * gap["pair"])],
            2 * floor["pair"] - 1)
    add_row([(t_index, -gap[other_name])],
            floor[other_name] - original["accuracy"][other_name])
    costs = _costs(law, r)
    for metric in ("nll", "brier"):
        add_row([(q_index[t, z], costs[metric][t, z])
                 for t, z in zip(ti, zi)], caps[local_name][metric])

    aub = sparse.coo_matrix((vals, (rows, cols)), shape=(len(rhs), n_var)).tocsr()
    aeq = sparse.coo_matrix((np.ones(n_q), (ti, np.arange(n_q))),
                            shape=(n_t, n_var)).tocsr()
    objective = np.zeros(n_var)
    objective[t_index] = 1
    bounds = [(0, 1)] * n_q + [(0, None)] * (n_var - n_q - 1) + [(0, 1)]
    started = time.monotonic()

    def optimize(c, ineq_matrix, ineq_rhs):
        remaining = None if time_limit is None else float(time_limit) - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            return None
        options = {"time_limit": remaining} if remaining is not None else None
        if deterministic:
            integrality = np.zeros(n_var, dtype=int)
            integrality[:n_q] = 1
            all_rows = sparse.vstack((aeq, ineq_matrix), format="csr")
            constraints = LinearConstraint(
                all_rows,
                np.r_[np.ones(n_t), np.full(len(ineq_rhs), -np.inf)],
                np.r_[np.ones(n_t), ineq_rhs],
            )
            lb = np.array([lo if lo is not None else -np.inf for lo, _ in bounds])
            ub = np.array([hi if hi is not None else np.inf for _, hi in bounds])
            return milp(c, integrality=integrality, bounds=Bounds(lb, ub),
                        constraints=constraints, options=options)
        return linprog(c, A_ub=ineq_matrix, b_ub=ineq_rhs, A_eq=aeq,
                       b_eq=np.ones(n_t), bounds=bounds, method="highs", options=options)

    result = optimize(objective, aub, np.array(rhs))
    if result is None:
        return {"success": False, "status": "coordinate time limit before primary solve",
                "q": None}
    if not result.success or result.x is None:
        return {"success": False, "status": result.message, "q": None,
                "solver_status": int(result.status)}
    primary_t = float(result.x[t_index])
    primary_status = int(result.status)
    primary_mip_gap = getattr(result, "mip_gap", None)
    # Predeclared lexicographic tie rule: among primary optima, reduce the
    # variable local recovery. The fixed other local recovery is constant.
    second_objective = np.zeros(n_var)
    second_objective[loc_u] = 1
    t_cap = sparse.coo_matrix(([1.0], ([0], [t_index])), shape=(1, n_var)).tocsr()
    second = optimize(second_objective, sparse.vstack((aub, t_cap), format="csr"),
                      np.r_[rhs, primary_t + 1e-9])
    if second is None:
        return {"success": False, "status": "coordinate time limit before tie solve",
                "q": None, "primary_t": primary_t,
                "primary_solver_status": primary_status,
                "primary_mip_gap": primary_mip_gap}
    if not second.success or second.x is None:
        return {"success": False, "status": "lexicographic tie solve: " + second.message,
                "q": None, "solver_status": int(second.status),
                "primary_t": primary_t,
                "primary_solver_status": primary_status,
                "primary_mip_gap": primary_mip_gap}
    q = np.zeros((n_t, n_z))
    q[ti, zi] = second.x[:n_q]
    measured = evaluate(law, q if r == 0 else q1, q2 if r == 0 else q)
    ratios = [(measured["accuracy"][v] - floor[v]) / gap[v]
              for v in VIEWS if gap[v] > TOL]
    actual_t = max([0.0] + ratios)
    residuals = [np.max(np.abs(q.sum(axis=1) - 1))]
    residuals += [measured["utility"][name][metric] - caps[name][metric]
                  for name in ("r1", "r2") for metric in ("nll", "brier")]
    residuals += [measured["accuracy"][v] - floor[v]
                  for v in VIEWS if gap[v] <= TOL]
    residuals += [actual_t - float(second.x[t_index]),
                  float(second.x[t_index]) - primary_t - 1e-9]
    max_residual = float(max(residuals))
    return {"success": bool(max_residual <= 1e-7), "status": second.message,
            "solver_status": int(second.status), "q": q, "t": float(second.x[t_index]),
            "primary_t": primary_t,
            "primary_solver_status": primary_status,
            "primary_mip_gap": primary_mip_gap,
            "tie_objective": float(second.fun),
            "replayed_t": float(actual_t), "max_residual": max_residual,
            "solver": "HiGHS-MILP" if deterministic else "HiGHS-LP",
            "tie_mip_gap": getattr(second, "mip_gap", None),
            "elapsed_seconds": float(time.monotonic() - started),
            "scope": "one fixed-other finite-law coordinate"}


def fit_alternating(law: FiniteLaw, q1, q2, *, caps: dict,
                    max_rounds: int, time_limit_per_step: float | None = None,
                    deterministic: bool = False):
    """Run a bounded sequence of local coordinate solves with full receipts."""
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    q1, q2 = _validate_q(law, 0, q1).copy(), _validate_q(law, 1, q2).copy()
    history = []
    for round_index in range(max_rounds):
        for recipient in ("r1", "r2"):
            step = solve_coordinate(
                law, q1, q2, recipient=recipient, caps=caps,
                time_limit=time_limit_per_step, deterministic=deterministic)
            receipt = {k: v for k, v in step.items() if k != "q"}
            receipt.update(round=round_index, recipient=recipient)
            history.append(receipt)
            if not step["success"]:
                return {"success": False, "q1": q1, "q2": q2,
                        "history": history, "status": step["status"]}
            if recipient == "r1":
                q1 = step["q"]
            else:
                q2 = step["q"]
    return {"success": True, "q1": q1, "q2": q2,
            "history": history, "status": "bounded rounds complete",
            "final": evaluate(law, q1, q2)}
