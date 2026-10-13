"""Context-shared NARROW/WIDE action-template affine kernels.

Action positions are internal; recipients see original emitted global codes.
KEEP has sentinel -1 only inside the public parameterization description.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from .lp import AffineLaw


def build_template_law(space, code1, code2, contexts, support, max_input_bytes):
    c1, c2, ctx = np.asarray(code1), np.asarray(code2), np.asarray(contexts)
    if (c1.dtype != np.int64 or c2.dtype != np.int64 or ctx.dtype != np.int64
            or c1.ndim != 1 or c2.shape != c1.shape or ctx.shape != c1.shape or len(c1) == 0
            or np.any(c1 < 0) or np.any(c1 >= len(space.probability1))
            or np.any(c2 < 0) or np.any(c2 >= len(space.probability2))
            or np.any(ctx < -1) or support not in ("NARROW", "WIDE")):
        raise ValueError("template input contract differs")
    active = np.unique(ctx[ctx >= 0])
    if len(active) and not np.array_equal(active, np.arange(active.max() + 1)):
        raise ValueError("supported context IDs must be contiguous")
    predicted1 = space.probability1.argmax(1)
    predicted2 = space.probability2.argmax(1)
    descriptions, offsets, counts, parameter_count = [], [], [], 0
    maximum_actions = 1
    for g in active:
        mask = ctx == g
        classes1, classes2 = np.unique(predicted1[c1[mask]]), np.unique(predicted2[c2[mask]])
        if len(classes1) != 1 or len(classes2) != 1:
            raise ValueError("each context must preserve one predicted class pair")
        if support == "NARROW":
            left, right = np.arange(4), np.arange(4)
        else:
            left = np.r_[-1, np.flatnonzero(predicted1 == classes1[0])]
            right = np.r_[-1, np.flatnonzero(predicted2 == classes2[0])]
        count = len(left) * len(right)
        descriptions.append({"context": int(g), "class_pair": [int(classes1[0]), int(classes2[0])],
                             "recipient1_templates": left.tolist(), "recipient2_templates": right.tolist(),
                             "parameter_offset": parameter_count, "action_count": count})
        offsets.append(parameter_count); counts.append(count)
        parameter_count += count
        maximum_actions = max(maximum_actions, count)
    if parameter_count == 0:
        # An entirely unsupported context map is an identity diagnostic, not
        # an informative free-kernel experiment. One fixed zero parameter
        # makes generic LP serialization work without inventing a context.
        parameter_count = 1
    e = len(c1) * maximum_actions
    # Conservative peak estimate includes dense emitted arrays/base, temporary
    # row/column/value COO arrays, CSR storage and masks. Solver workspace is
    # separately reserved by scheduler and LP matrix-size receipt.
    estimated_input_bytes = e * (8 * 10 + 4) + parameter_count * 64 + len(c1) * 64
    if estimated_input_bytes > int(max_input_bytes):
        raise MemoryError(f"template input estimate {estimated_input_bytes} exceeds declared ceiling {int(max_input_bytes)}")
    emitted1 = np.repeat(c1[:, None], maximum_actions, axis=1)
    emitted2 = np.repeat(c2[:, None], maximum_actions, axis=1)
    base = np.zeros((len(c1), maximum_actions), np.float64)
    base[ctx < 0, 0] = 1.
    bounds = np.zeros((parameter_count, 2), np.float64)
    rows, cols = [], []
    eqrows, eqcols = [], []
    for g, desc in zip(active, descriptions):
        selected = np.flatnonzero(ctx == g)
        n = len(selected); count = desc["action_count"]; offset = desc["parameter_offset"]
        left = np.asarray(desc["recipient1_templates"], np.int64)
        right = np.asarray(desc["recipient2_templates"], np.int64)
        if support == "NARROW":
            action1 = space.actions1[c1[selected]][:, np.repeat(left, len(right))]
            action2 = space.actions2[c2[selected]][:, np.tile(right, len(left))]
            valid = (space.valid1[c1[selected]][:, np.repeat(left, len(right))]
                     & space.valid2[c2[selected]][:, np.tile(right, len(left))])
            allowed = valid.all(0)
        else:
            template1, template2 = np.repeat(left, len(right)), np.tile(right, len(left))
            action1 = np.broadcast_to(template1, (n, count)).copy()
            action2 = np.broadcast_to(template2, (n, count)).copy()
            action1[:, template1 == -1] = c1[selected, None]
            action2[:, template2 == -1] = c2[selected, None]
            allowed = np.ones(count, dtype=bool)
        if not allowed[0]:
            raise ValueError("KEEP must be executable in every supported context")
        emitted1[selected, :count] = action1
        emitted2[selected, :count] = action2
        bounds[offset:offset + count, 1] = allowed
        action = np.arange(count)
        rows.append((selected[:, None] * maximum_actions + action).ravel())
        cols.append(np.broadcast_to(offset + action, (n, count)).ravel())
        eqrows.extend([int(g)] * count); eqcols.extend(range(offset, offset + count))
    if rows:
        row = np.concatenate(rows); col = np.concatenate(cols)
        delta = sparse.coo_matrix((np.ones(len(row)), (row, col)), shape=(e, parameter_count)).tocsr()
    else:
        base[:, 0] = 1.
        delta = sparse.csr_matrix((e, parameter_count))
    equality = sparse.coo_matrix((np.ones(len(eqrows)), (eqrows, eqcols)),
                                 shape=(len(active), parameter_count)).tocsr()
    law = AffineLaw(emitted1, emitted2, c1, c2, base, delta, bounds,
                    equality_matrix=equality, equality_rhs=np.ones(len(active)))
    metadata = {"schema": "CBR1-context-template-v1", "support": support,
                "parameters": parameter_count, "supported_contexts": len(active),
                "maximum_actions_per_row": maximum_actions, "contexts": descriptions,
                "unsupported_rows": int((ctx < 0).sum()), "identity_only": len(active) == 0,
                "estimated_input_bytes": estimated_input_bytes,
                "declared_input_ceiling_bytes": int(max_input_bytes),
                "actions_observed": "only original emitted global code tuples",
                "wide_scope": "every same-predicted-class original global bank ID plus explicit KEEP",
                "pricing": "all declared template columns included; no column generation"}
    return law, metadata


def template_probabilities(metadata, theta, space, code1, code2, contexts):
    """Deployment law builder from frozen descriptors, without any fitting."""
    theta = np.asarray(theta, np.float64)
    c1, c2, ctx = np.asarray(code1), np.asarray(code2), np.asarray(contexts)
    if (theta.shape != (metadata["parameters"],) or not np.isfinite(theta).all()
            or c1.dtype != np.int64 or c2.dtype != np.int64 or ctx.dtype != np.int64
            or c1.ndim != 1 or c2.shape != c1.shape or ctx.shape != c1.shape
            or np.any(c1 < 0) or np.any(c1 >= len(space.probability1))
            or np.any(c2 < 0) or np.any(c2 >= len(space.probability2))):
        raise ValueError("template deployment arrays invalid")
    width = metadata["maximum_actions_per_row"]
    emitted1, emitted2 = np.repeat(c1[:, None], width, axis=1), np.repeat(c2[:, None], width, axis=1)
    mass = np.zeros((len(c1), width), np.float64)
    known = np.zeros(len(c1), bool)
    for desc in metadata["contexts"]:
        selected = np.flatnonzero(ctx == desc["context"])
        if not len(selected):
            continue
        if (np.any(space.probability1[c1[selected]].argmax(1) != desc["class_pair"][0])
                or np.any(space.probability2[c2[selected]].argmax(1) != desc["class_pair"][1])):
            raise ValueError("template context changed predicted class pair")
        left = np.asarray(desc["recipient1_templates"], np.int64)
        right = np.asarray(desc["recipient2_templates"], np.int64)
        a, b = np.repeat(left, len(right)), np.tile(right, len(left))
        count, offset = desc["action_count"], desc["parameter_offset"]
        if metadata["support"] == "NARROW":
            aa, bb = space.actions1[c1[selected]][:, a], space.actions2[c2[selected]][:, b]
            valid = space.valid1[c1[selected]][:, a] & space.valid2[c2[selected]][:, b]
            row_mass = np.broadcast_to(theta[offset:offset + count], valid.shape).copy()
            # A source code outside FIT can lack a particular neighbor. The
            # frozen rule is exact-P0 fallback for the entire affected row.
            executable = ~(~valid & (row_mass > 0)).any(1)
        else:
            aa = np.broadcast_to(a, (len(selected), count)).copy()
            bb = np.broadcast_to(b, (len(selected), count)).copy()
            aa[:, a == -1] = c1[selected, None]
            bb[:, b == -1] = c2[selected, None]
            executable = np.ones(len(selected), bool)
            row_mass = np.broadcast_to(theta[offset:offset + count], aa.shape)
        selected = selected[executable]
        emitted1[selected, :count] = aa[executable]
        emitted2[selected, :count] = bb[executable]
        mass[selected, :count] = row_mass[executable]
        known[selected] = True
    mass[~known, 0] = 1.
    if np.any(mass < 0) or not np.allclose(mass.sum(1), 1, atol=1e-8, rtol=0):
        raise ValueError("template deployment law not normalized")
    return emitted1, emitted2, mass
