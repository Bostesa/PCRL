"""Finite-map builders and bounded solvers for registered matched controls.

The product and local kernels are bilinear families; only a coordinate with
the other factor frozen is affine. Deterministic fitting uses binary weights,
never a rounded stochastic fit. This module loads no scientific role file.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse


def _groups(group):
    group = np.asarray(group)
    if group.dtype.kind not in "iu" or group.size == 0 or np.any(group < 0):
        raise ValueError("groups must be nonempty nonnegative integer indices")
    unique = np.unique(group)
    return group.astype(np.int64, copy=False), int(unique[-1]) + 1


def _factor(probability, width):
    probability = np.asarray(probability, dtype=np.float64)
    if (probability.ndim != 2 or probability.shape[1] != width
            or not np.isfinite(probability).all() or np.any(probability < 0)
            or not np.allclose(probability.sum(1), 1, rtol=0, atol=1e-12)):
        raise ValueError("frozen factor is not a finite normalized law")
    return probability


def product_map(group, frozen_other, fitted_recipient, width=4):
    """CSR mapping of variable marginal weights to an independent joint law."""
    group, count = _groups(group)
    frozen_other = _factor(frozen_other, width)
    if group.ndim != 1 or len(group) != len(frozen_other) or fitted_recipient not in (1, 2):
        raise ValueError("product coordinate contract differs")
    n = len(group)
    row, a, b = np.indices((n, width, width))
    coordinate = a if fitted_recipient == 1 else b
    fixed_coordinate = b if fitted_recipient == 1 else a
    values = frozen_other[row, fixed_coordinate]
    matrix = sparse.coo_matrix((values.ravel(),
        ((row * width * width + a * width + b).ravel(),
         (group[row] * width + coordinate).ravel())),
        shape=(n * width * width, count * width)).tocsr()
    matrix.eliminate_zeros()
    return matrix


def sequential_map(frozen_first, second_group, first_recipient, width=4):
    """Second-stage affine map, grouped by public emitted-first code.

    ``second_group[i,a]`` must be computed from input context and the actual
    emitted first-recipient global code, so aliased slots receive one group.
    No fine action position is a deployment input or transmitted field.
    """
    frozen_first = _factor(frozen_first, width)
    second_group, count = _groups(second_group)
    n = len(frozen_first)
    if second_group.shape != (n, width) or first_recipient not in (1, 2):
        raise ValueError("sequential coordinate contract differs")
    row, a, b = np.indices((n, width, width))
    first_action = a if first_recipient == 1 else b
    second_action = b if first_recipient == 1 else a
    values = frozen_first[row, first_action]
    return sparse.coo_matrix((values.ravel(),
        ((row * width * width + a * width + b).ravel(),
         (second_group[row, first_action] * width + second_action).ravel())),
        shape=(n * width * width, count * width)).tocsr()


def categorical_map(group, actions=16):
    """Context-shared joint categorical kernel; integrality is a solver option."""
    group, count = _groups(group)
    if group.ndim != 1 or actions < 1:
        raise ValueError("categorical map contract differs")
    row, action = np.indices((len(group), actions))
    return sparse.coo_matrix((np.ones(row.size),
        ((row * actions + action).ravel(),
         (group[row] * actions + action).ravel())),
        shape=(len(group) * actions, count * actions)).tocsr()


def simplex_rows(groups, width):
    """Two <= rows per simplex, expressing exact normalization."""
    if groups < 1 or width < 1:
        raise ValueError("empty simplex")
    index = np.arange(groups * width)
    positive = sparse.coo_matrix((np.ones(len(index)), (index // width, index)),
        shape=(groups, groups * width)).tocsr()
    return sparse.vstack((positive, -positive), format="csr"), np.r_[np.ones(groups), -np.ones(groups)]


def template_bounds(group, valid):
    """Only templates valid for every row in a shared group can receive mass."""
    group, count = _groups(group)
    valid = np.asarray(valid)
    if valid.dtype != bool or valid.ndim != 2 or group.shape != (len(valid),):
        raise ValueError("grouped support contract differs")
    allowed = np.ones((count, valid.shape[1]), bool)
    np.logical_and.at(allowed, group, valid)
    # A retained deployment parent may have no realized FIT support after its
    # children are selected. It has exact KEEP weights, not arbitrary free mass.
    absent = np.bincount(group, minlength=count) == 0
    allowed[absent] = False
    allowed[absent, 0] = True
    if not allowed.any(1).all():
        raise ValueError("shared group has no executable action")
    return np.c_[np.zeros(allowed.size), allowed.ravel().astype(np.float64)]


@dataclass(frozen=True)
class ControlCeilings:
    """Bounds must be frozen in the campaign lock, not changed after outcomes."""
    starts: int = 2
    rounds: int = 4
    seconds: float = 600.
    feasibility_tolerance: float = 1e-8
    relative_mip_gap: float = 1e-6


def append_simplexes(rows, count, width):
    from .lp import ConstraintRows
    matrix, rhs = simplex_rows(count, width)
    names = tuple(f"simplex/{direction}/{g}" for direction in ("upper", "lower") for g in range(count))
    return ConstraintRows(tuple(rows.names) + names,
        sparse.vstack((rows.matrix, matrix), format="csr"), np.r_[rows.rhs, rhs])


def map_law(emitted1, emitted2, identity1, identity2, mapping, bounds, weights=None, width=4):
    """Create the shared R2 affine law without fitting or reading any labels."""
    from .lp import AffineLaw
    if mapping.shape[1] % width:
        raise ValueError("mapping parameter width differs")
    eq, rhs = simplex_rows(mapping.shape[1] // width, width)
    count = len(rhs) // 2
    return AffineLaw(emitted1, emitted2, identity1, identity2,
        np.zeros(np.asarray(emitted1).shape, np.float64), mapping, bounds, weights,
        eq[:count], rhs[:count])


def initial_factor(bounds, mode, width=4):
    """Two predeclared starts: exact KEEP and uniform executable templates."""
    bounds = np.asarray(bounds, dtype=np.float64).reshape(-1, width, 2)
    allowed = bounds[:, :, 1] > 0
    if mode == "keep":
        if not allowed[:, 0].all():
            raise ValueError("KEEP is not executable in each group")
        answer = np.zeros(allowed.shape, np.float64)
        answer[:, 0] = 1.
    elif mode == "uniform":
        answer = allowed / allowed.sum(1, keepdims=True)
    else:
        raise ValueError("unregistered factor start")
    return answer


def apply_second_groups(contexts, emitted_first, keys):
    contexts = np.asarray(contexts, np.int64)
    emitted_first = np.asarray(emitted_first, np.int64)
    keys = np.asarray(keys, np.int64)
    if contexts.shape != (len(emitted_first),) or emitted_first.ndim != 2 or keys.ndim != 2 or keys.shape[1] != 2:
        raise ValueError("second-stage key contract differs")
    lookup = {tuple(key): index for index, key in enumerate(keys.tolist())}
    answer = np.empty(emitted_first.shape, np.int64)
    for row, context in enumerate(contexts):
        if (int(context), -1) not in lookup:
            raise ValueError("second-stage parent missing")
        parent = lookup[int(context), -1]
        for action, code in enumerate(emitted_first[row]):
            answer[row, action] = lookup.get((int(context), int(code)), parent)
    return answer


def register_second_groups(contexts, emitted_first, valid_first, floor=64., ceiling=512):
    """Label-free fixed hierarchy using uniform potential-template counts."""
    contexts = np.asarray(contexts, np.int64)
    emitted_first = np.asarray(emitted_first, np.int64)
    valid_first = np.asarray(valid_first, bool)
    if emitted_first.ndim != 2 or emitted_first.shape != valid_first.shape or contexts.shape != (len(emitted_first),):
        raise ValueError("second-stage source support differs")
    if not valid_first.any(1).all():
        raise ValueError("first-stage support empty")
    parents = [(int(c), -1) for c in np.unique(contexts)]
    if len(parents) > ceiling:
        raise ValueError("second-stage parent ceiling exceeded")
    counts = {}
    uniform = valid_first / valid_first.sum(1, keepdims=True)
    for row, action in zip(*np.nonzero(valid_first)):
        key = (int(contexts[row]), int(emitted_first[row, action]))
        counts[key] = counts.get(key, 0.) + float(uniform[row, action])
    children = sorted((key for key, count in counts.items() if count >= floor),
                      key=lambda key: (-counts[key], key))[:ceiling-len(parents)]
    keys = np.asarray(parents + children, np.int64)
    mapped = apply_second_groups(contexts, emitted_first, keys)
    return mapped, keys, np.asarray([counts.get(tuple(key), 0.) for key in keys], np.float64)


def fit_independent_product(emitted1, emitted2, identity1, identity2, groups1, groups2,
                            valid1, valid2, sensitive, constraints_builder,
                            ceilings=ControlCeilings(), weights=None):
    """Bounded alternating LPs, not a joint LP or global bilinear certificate.

    Local controls pass groups determined only by each own P0 input. Central
    product controls pass the same input-context groups for both recipients.
    Each failed frozen-factor subproblem is recorded; no global family
    infeasibility claim follows. No outcome-dependent restart is added.
    """
    import time
    from .lp import solve_protection, direct_bayes
    groups1, _ = _groups(groups1)
    groups2, _ = _groups(groups2)
    bounds = (template_bounds(groups1, valid1), template_bounds(groups2, valid2))
    if ceilings.starts != 2:
        raise ValueError("exactly the two registered starts are required")
    begun = time.monotonic()
    attempts, best = [], None
    for start in ("keep", "uniform"):
        factor = [initial_factor(bounds[0], start), initial_factor(bounds[1], start)]
        stopped = False
        for iteration in range(ceilings.rounds):
            for recipient in (1, 2):
                remaining = ceilings.seconds - (time.monotonic() - begun)
                if remaining <= 0:
                    stopped = True
                    break
                own = recipient - 1
                other = 1 - own
                group = (groups1, groups2)[own]
                other_group = (groups1, groups2)[other]
                mapping = product_map(group, factor[other][other_group], recipient)
                law = map_law(emitted1, emitted2, identity1, identity2, mapping, bounds[own], weights)
                rows = constraints_builder(law)
                result = solve_protection(law, sensitive, rows, time_limit=remaining,
                                          tolerance=ceilings.feasibility_tolerance)
                attempts.append({"start": start, "round": iteration, "recipient": recipient,
                                 "result": result})
                if not result["success"]:
                    stopped = True
                    break
                factor[own] = result["theta"].reshape(-1, 4)
                objective, views = direct_bayes(law, result["theta"], sensitive)
                if best is None or objective < best["objective"] - 1e-12:
                    best = {"objective": objective, "views": views, "factors": [v.copy() for v in factor],
                            "law": law, "theta": result["theta"].copy(), "result": result,
                            "start": start, "round": iteration, "recipient": recipient}
            if stopped:
                break
    return {"success": best is not None, "selected": best, "attempts": attempts,
            "solver_class": "bounded alternating coordinate LPs for a bilinear family",
            "global_optimality": "UNRESOLVED", "global_bilinear_gap": None,
            "elapsed_seconds": time.monotonic() - begun}


def solve_binary_control(law, sensitive, constraints, ceilings=ControlCeilings()):
    """Fit binary context/action weights with a genuine mixed-integer solver."""
    import time
    from scipy.optimize import milp, Bounds, LinearConstraint
    from .lp import protection_program, solve_program
    begun = time.monotonic()
    program = protection_program(law, sensitive, constraints)
    # The continuous relaxation is a lower bound for this exact MIP family.
    # It is separately counted by the caller, not reported as the MIP fit.
    relaxation = solve_program(program, time_limit=max(.001, (ceilings.seconds - (time.monotonic()-begun)) / 4),
                               tolerance=ceilings.feasibility_tolerance)
    linear = [LinearConstraint(program.inequality_matrix, -np.inf, program.inequality_rhs)]
    if len(program.equality_rhs):
        linear.append(LinearConstraint(program.equality_matrix, program.equality_rhs, program.equality_rhs))
    integrality = np.r_[np.ones(program.parameters, dtype=np.int32),
                       np.zeros(len(program.objective) - program.parameters, dtype=np.int32)]
    result = milp(program.objective, integrality=integrality,
                  bounds=Bounds(program.bounds[:, 0], program.bounds[:, 1]), constraints=linear,
                  options={"time_limit": max(.001, min(ceilings.seconds*.75, ceilings.seconds - (time.monotonic()-begun))), "threads": 1,
                           "mip_rel_gap": ceilings.relative_mip_gap})
    cert = {"scope": "registered binary context-action kernel and fixed empirical constraints",
            "solver_status": int(result.status), "solver_message": str(result.message),
            "optimality_certified": False, "feasible": False,
            "mip_gap": getattr(result, "mip_gap", None),
            "mip_dual_bound": getattr(result, "mip_dual_bound", None),
            "mip_nodes": getattr(result, "mip_node_count", None),
            "relaxation_objective": relaxation["certificate"].get("primal_objective"),
            "relaxation_certified": relaxation["success"]}
    theta = None
    if result.x is not None:
        raw_x = np.asarray(result.x, np.float64)
        integer_error = float(np.abs(raw_x[:program.parameters] - np.rint(raw_x[:program.parameters])).max(initial=0))
        x = raw_x.copy()
        # This snaps only a genuine binary-solver incumbent within its declared
        # integer tolerance. It is never rounding a stochastic LP solution.
        if integer_error <= ceilings.feasibility_tolerance:
            x[:program.parameters] = np.rint(x[:program.parameters])
        theta = x[:program.parameters]
        violation = max(0., float((program.inequality_matrix @ x - program.inequality_rhs).max(initial=0)),
                        float(np.abs(program.equality_matrix @ x - program.equality_rhs).max(initial=0)),
                        float((program.bounds[:, 0] - x).max(initial=0)),
                        float((x - program.bounds[:, 1]).max(initial=0)))
        cert.update(maximum_primal_violation=violation, maximum_integer_error=integer_error,
                    primal_objective=float(program.objective @ x),
                    feasible=bool(max(violation, integer_error) <= ceilings.feasibility_tolerance))
        cert["optimality_certified"] = bool(cert["feasible"] and result.success
                and getattr(result, "mip_gap", np.inf) <= ceilings.relative_mip_gap)
    return {"success": cert["feasible"], "theta": theta,
            "primal": None if result.x is None else x.copy(),
            "unsnapped_primal": None if result.x is None else result.x.copy(), "certificate": cert,
            "program": program, "relaxation": relaxation, "solver_class": "binary MILP"}


def fit_sequential(emitted1, emitted2, identity1, identity2, contexts, valid1, valid2,
                   sensitive, constraints_builder, first_recipient, ceilings=ControlCeilings(), weights=None):
    """Two genuinely frozen stages; grouping is registered before stage one."""
    import time
    from .lp import solve_protection, direct_bayes
    begun = time.monotonic()
    contexts, count = _groups(contexts)
    n = len(contexts)
    valid = (np.asarray(valid1, bool), np.asarray(valid2, bool))
    first = first_recipient-1
    if first not in (0, 1):
        raise ValueError("first recipient invalid")
    other = 1-first
    first_emitted = np.asarray((emitted1, emitted2)[first]).reshape(n, 4, 4)
    first_emitted = first_emitted[:, :, 0] if first == 0 else first_emitted[:, 0, :]
    second_groups, keys, counts = register_second_groups(contexts, first_emitted, valid[first])
    keep = np.zeros((n, 4), np.float64)
    keep[:, 0] = 1.
    bounds1 = template_bounds(contexts, valid[first])
    map1 = product_map(contexts, keep, first_recipient)
    law1 = map_law(emitted1, emitted2, identity1, identity2, map1, bounds1, weights)
    rows1 = constraints_builder(law1)
    result1 = solve_protection(law1, sensitive, rows1,
                              time_limit=max(.001, ceilings.seconds/2-(time.monotonic()-begun)),
                              tolerance=ceilings.feasibility_tolerance)
    record = {"solver_class": "two freeze-first LP stages", "first_result": result1,
              "first_recipient": first_recipient, "second_group_keys": keys,
              "second_group_uniform_counts": counts, "second_result": None,
              "global_sequential_optimality": "not claimed", "success": False}
    if not result1["success"]:
        record["failure_scope"] = "first-stage KEEP-other LP; not every sequential mechanism"
        return record
    theta1 = result1["theta"].reshape(-1, 4)
    fixed_first = theta1[contexts]
    mapping2 = sequential_map(fixed_first, second_groups, first_recipient)
    # The same second key can occur in many first slots; all row/template
    # validity conditions must hold for that shared conditional distribution.
    flat_group = second_groups.ravel()
    expanded_valid = np.repeat(valid[other][:, None, :], 4, axis=1).reshape(-1, 4)
    bounds2 = template_bounds(flat_group, expanded_valid)
    law2 = map_law(emitted1, emitted2, identity1, identity2, mapping2, bounds2, weights)
    rows2 = constraints_builder(law2)
    result2 = solve_protection(law2, sensitive, rows2,
                              time_limit=max(.001, min(ceilings.seconds/2, ceilings.seconds-(time.monotonic()-begun))),
                              tolerance=ceilings.feasibility_tolerance)
    record["second_result"] = result2
    if result2["success"]:
        objective, views = direct_bayes(law2, result2["theta"], sensitive)
        record.update(success=True, first_theta=theta1, second_theta=result2["theta"].reshape(-1, 4),
                      objective=objective, views=views, law=law2, theta=result2["theta"])
    return record


@dataclass
class ControlPolicy:
    """Deployment adapter reads P0 inputs only; returns original bank tuples."""
    space: object
    kind: str
    context_map: object
    first_weights: np.ndarray
    second_weights: object = None
    second_keys: object = None
    first_recipient: int = 1

    def __post_init__(self):
        if self.kind not in ("LOCAL", "PRODUCT", "DETERMINISTIC", "SEQUENTIAL"):
            raise ValueError("unregistered control policy")
        width = 16 if self.kind == "DETERMINISTIC" else 4
        self.first_weights = _factor(self.first_weights, width).copy()
        if self.kind == "DETERMINISTIC":
            if not np.isin(self.first_weights, [0.,1.]).all():
                raise ValueError("deterministic deployment requires exact binary masses")
        else:
            self.second_weights = _factor(self.second_weights, 4).copy()
        if self.kind != "LOCAL" and self.context_map is None:
            raise ValueError("central control requires frozen context map")
        if self.kind == "LOCAL":
            if len(self.first_weights) != len(self.space.probability1) or len(self.second_weights) != len(self.space.probability2):
                raise ValueError("local policy own-bank table differs")
        if self.kind == "SEQUENTIAL":
            self.second_keys = np.asarray(self.second_keys, np.int64).copy()
            if self.second_keys.shape != (len(self.second_weights), 2) or self.first_recipient not in (1,2):
                raise ValueError("sequential public-code key table differs")

    def law_codes(self, c1, c2):
        p = self.space.prepare(c1, c2)
        return p["code_r1"], p["code_r2"], self.law_prepared(p)

    def law_prepared(self, p):
        from incumbent_view.r2_space import check_law
        c1, c2 = p["b0_code_r1"], p["b0_code_r2"]
        n = len(c1)
        if self.kind == "LOCAL":
            first, second = self.first_weights[c1], self.second_weights[c2]
            q = (first[:,:,None]*second[:,None,:]).reshape(n,16)
        else:
            context = self.context_map.apply(self.space.probability1[c1], self.space.probability2[c2])
            supported = context >= 0
            q = np.zeros((n,16),np.float64);q[:,0] = 1.
            selected = context[supported]
            if self.kind == "DETERMINISTIC":
                q[supported] = self.first_weights[selected]
            elif self.kind == "PRODUCT":
                first, second = self.first_weights[selected], self.second_weights[selected]
                q[supported] = (first[:,:,None]*second[:,None,:]).reshape(-1,16)
            else:
                first_code = (p["code_r1"].reshape(n,4,4)[:,:,0] if self.first_recipient == 1
                              else p["code_r2"].reshape(n,4,4)[:,0,:])
                groups = apply_second_groups(selected, first_code[supported], self.second_keys)
                first, second = self.first_weights[selected], self.second_weights[groups]
                if self.first_recipient == 1:
                    q[supported] = (first[:,:,None]*second).reshape(-1,16)
                else:
                    q[supported] = (first[:,:,None]*second).transpose(0,2,1).reshape(-1,16)
        return check_law(p,q)


def save_control_policy(directory, policy, *, seed, arm_id, budget, source_manifest):
    from pathlib import Path
    from .fit import write_npz
    from .source import write_once
    directory = Path(directory)
    if directory.exists() or ".private" not in directory.parts:
        raise PermissionError("new private control package required")
    directory.mkdir(parents=True,mode=0o700)
    arrays = {"first_weights":policy.first_weights}
    if policy.second_weights is not None:
        arrays["second_weights"] = policy.second_weights
    if policy.second_keys is not None:
        arrays["second_keys"] = policy.second_keys
    if policy.context_map is not None:
        arrays.update(context_mapping=policy.context_map.mapping,
                      raw_fit_counts=policy.context_map.raw_fit_counts,
                      merged_fit_counts=policy.context_map.merged_fit_counts)
    package_sha = write_npz(directory/"control.npz",arrays)
    record = {"schema":"CBR1-control-policy-v1","seed":int(seed),"arm_id":arm_id,"budget":budget,
              "kind":policy.kind,"first_recipient":policy.first_recipient,"package_sha256":package_sha,
              "context_labels":None if policy.context_map is None else list(policy.context_map.labels),
              "public_bank_sha256":source_manifest["seeds"][str(seed)]["public_bank"]["sha256"],
              "parameter_count":int(policy.first_weights.size + (0 if policy.second_weights is None else policy.second_weights.size)),
              "runtime_sensitive_or_task_labels":False,"public_wire":"original code tuples and frozen source metadata"}
    return record,write_once(directory/"POLICY_RECEIPT.json",record)


def load_control_policy(directory,manifest,*,expected_receipt_sha256=None):
    import json
    from pathlib import Path
    from .source import sha_file,read_exact,load_historical_policy
    from .contexts import ContextMap
    directory = Path(directory)
    receipt = directory/"POLICY_RECEIPT.json"
    if expected_receipt_sha256 and sha_file(receipt)!=expected_receipt_sha256:
        raise PermissionError("control receipt differs")
    record = json.loads(receipt.read_text())
    if record["schema"] != "CBR1-control-policy-v1":
        raise ValueError("wrong control package schema")
    bank_sha = manifest["seeds"][str(record["seed"])]["public_bank"]["sha256"]
    if bank_sha != record["public_bank_sha256"]:
        raise PermissionError("control bank source differs")
    keys = {"first_weights"}
    if record["kind"] != "DETERMINISTIC":keys.add("second_weights")
    if record["kind"] == "SEQUENTIAL":keys.add("second_keys")
    if record["context_labels"] is not None:
        keys.update({"context_mapping","raw_fit_counts","merged_fit_counts"})
    arrays = read_exact({"path":str(directory/"control.npz"),"sha256":record["package_sha256"]},keys)
    context = None if record["context_labels"] is None else ContextMap(arrays["context_mapping"],arrays["raw_fit_counts"],
                                  arrays["merged_fit_counts"],tuple(record["context_labels"]))
    space = load_historical_policy(manifest,record["seed"],"V_J").space
    return ControlPolicy(space,record["kind"],context,arrays["first_weights"],arrays.get("second_weights"),
                         arrays.get("second_keys"),record["first_recipient"])
