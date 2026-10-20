"""Utility-recovery frontier comparison without an exact cost match (challenger; see nrm/ctl/__init__.py).

Inputs are numbers already produced by the coordinator's evaluators (nrm.evaluate, nrm.screen / nrm.devreaders);
nothing here reads data or labels.

A control path is a list of (cost, recovery) points from ONE control family, ordered by strength and starting at the
twin itself (strength 0). cost is one predeclared scalar task-spending axis, larger = more spending: the primary axis is
the maximum normalized clause usage max_i excess_i / limit_i over the 24 clauses + 4 guards (clause_usage below);
secondary axes are the per-task NLL excess vs U0. recovery is a reader AUC, lower = more protective. For linked
recipients use the linked recovery max(pair reader, ignore-one-recipient readers), never the pair reader alone.

  frontier_step(c)  lowest recovery over REALIZED control points with cost <= c (a staircase: the control is never
                    credited with a point it did not produce)
  frontier_lin(c)   linear interpolation between staircase corners (time-sharing heuristic), reported beside the
                    staircase, never instead of it

Verdict for a candidate point (c, r), with a predeclared recovery margin m (from the statistical reviewer's rules,
e.g. two paired SEs, or 0 for a descriptive read) and a predeclared cost tolerance tol (the control may spend up to
tol more than the candidate and still count as matched):
  CONTROL_EXPLAINS      a realized control point has cost <= c + tol and recovery <= r + m
  BELOW_CONTROL_RANGE   the candidate spends less than every control point (the path must start at the twin; a
                        candidate cheaper than its own twin signals an undertrained twin)
  EXTEND_CONTROL        not explained, but no control point reaches cost >= c: the control was never pushed to the
                        candidate's spending (the SCM failure); no claim until the path is extended
  AHEAD_VS_SPARSE_PATH  not explained and bracketed, but ahead on the staircase only (interpolated gap <= m): densify
  CANDIDATE_AHEAD       not explained, bracketed, staircase gap > m and interpolated gap > m

Selftest (invented numbers only): python3 -m nrm.ctl.frontier selftest
"""
from __future__ import annotations

import json
import sys

import numpy as np

EPS_COST = 1e-12


def clause_usage(excess, limits):
    """Maximum normalized usage over matching clause names: max_i excess_i / limit_i (negative excess allowed)."""
    keys = sorted(set(excess) & set(limits))
    if not keys or set(excess) != set(limits):
        raise ValueError('excess and limits must name the same clauses')
    return max(float(excess[k]) / float(limits[k]) for k in keys)


def _path(points):
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2 or len(P) == 0 or not np.isfinite(P).all():
        raise ValueError('a path is a non-empty list of finite (cost, recovery) pairs')
    return P[np.lexsort((P[:, 1], P[:, 0]))]


def staircase(points):
    """Pareto corners (increasing cost, strictly decreasing recovery)."""
    out, best = [], np.inf
    for c, r in _path(points):
        if r < best:
            best = r
            out.append((float(c), float(r)))
    return out


def frontier_step(c, points):
    P = _path(points)
    ok = P[:, 0] <= c + EPS_COST
    return float(P[ok, 1].min()) if ok.any() else None


def frontier_lin(c, points):
    S = staircase(points)
    if c < S[0][0] - EPS_COST or c > _path(points)[:, 0].max() + EPS_COST:
        return None
    for (c0, r0), (c1, r1) in zip(S, S[1:]):
        if c0 - EPS_COST <= c <= c1 + EPS_COST:
            return float(r0 + (r1 - r0) * (c - c0) / (c1 - c0)) if c1 > c0 else float(min(r0, r1))
    return float(S[-1][1])          # beyond the last corner but within the path's cost range: flat


def compare(cand, control, margin=0.0, tol=0.0):
    """One candidate point (cost, recovery) against one control path on one cost axis."""
    c, r = float(cand[0]), float(cand[1])
    P = _path(control)
    out = {'cost': c, 'recovery': r, 'margin': margin, 'tol': tol, 'gap_step': None, 'gap_lin': None,
           'extra_cost': None, 'explaining_point': None}
    hit = P[(P[:, 0] <= c + tol + EPS_COST) & (P[:, 1] <= r + margin + EPS_COST)]
    reach = P[P[:, 1] <= r + EPS_COST]
    if len(reach):
        out['extra_cost'] = float(reach[:, 0].min() - c)
    fs, fl = frontier_step(c, P), frontier_lin(c, P)
    out['gap_step'] = None if fs is None else fs - r
    out['gap_lin'] = None if fl is None else fl - r
    if len(hit):
        out['explaining_point'] = [float(x) for x in hit[np.argmin(hit[:, 0])]]
        out['verdict'] = 'CONTROL_EXPLAINS'
    elif fs is None:
        out['verdict'] = 'BELOW_CONTROL_RANGE'
    elif P[:, 0].max() < c - EPS_COST:
        out['verdict'] = 'EXTEND_CONTROL'
    elif out['gap_lin'] is None or out['gap_lin'] <= margin:
        out['verdict'] = 'AHEAD_VS_SPARSE_PATH'
    else:
        out['verdict'] = 'CANDIDATE_AHEAD'
    return out


def compare_axes(cand, controls, margins, tols, primary):
    """cand: {axis: (cost, recovery)}; controls: {axis: path}. CANDIDATE_AHEAD overall only if ahead on the primary axis
    and explained on no axis; CONTROL_EXPLAINS if explained on any axis; otherwise the primary axis verdict."""
    per = {a: compare(cand[a], controls[a], margins[a], tols[a]) for a in controls}
    v = [x['verdict'] for x in per.values()]
    if 'CONTROL_EXPLAINS' in v:
        overall = 'CONTROL_EXPLAINS'
    else:
        overall = per[primary]['verdict']
    return {'overall': overall, 'per_axis': per}


# ----------------------------------------------------------------------------------------------- selftest (invented)

def selftest():
    res = []

    def ok(name, cond):
        res.append(bool(cond))
        print(('PASS ' if cond else 'FAIL ') + name)

    ctl = [(0.0, 0.65), (0.2, 0.63), (0.4, 0.61), (0.6, 0.60), (0.8, 0.59), (0.5, 0.62)]   # one dominated point
    ok('staircase drops the dominated point', staircase(ctl) == [(0.0, 0.65), (0.2, 0.63), (0.4, 0.61), (0.6, 0.60),
                                                                  (0.8, 0.59)])
    ok('step frontier uses only realized cheaper points', frontier_step(0.3, ctl) == 0.63)
    ok('interpolated frontier', abs(frontier_lin(0.3, ctl) - 0.62) < 1e-12)
    cases = [((0.3, 0.620), 0.002, 0.0, 'AHEAD_VS_SPARSE_PATH'),
             ((0.3, 0.600), 0.002, 0.0, 'CANDIDATE_AHEAD'),
             ((0.5, 0.612), 0.002, 0.0, 'CONTROL_EXPLAINS'),
             ((0.35, 0.614), 0.002, 0.05, 'CONTROL_EXPLAINS'),    # control may spend tol more (0.35 + 0.05 = 0.40)
             ((0.35, 0.614), 0.002, 0.0, 'AHEAD_VS_SPARSE_PATH'),  # staircase gap 0.016, interpolated gap 0.001
             ((0.35, 0.612), 0.002, 0.0, 'CANDIDATE_AHEAD'),       # interpolated gap 0.003 > margin
             ((-0.1, 0.600), 0.002, 0.0, 'BELOW_CONTROL_RANGE'),
             ((1.0, 0.550), 0.002, 0.0, 'EXTEND_CONTROL'),
             ((1.0, 0.589), 0.002, 0.0, 'CONTROL_EXPLAINS')]
    for (pt, m, tol, want) in cases:
        got = compare(pt, ctl, m, tol)
        ok(f'candidate {pt} margin {m} tol {tol}: {want}', got['verdict'] == want)
    g = compare((0.3, 0.600), ctl, 0.002)
    ok('extra cost the control needs to reach the candidate recovery', abs(g['extra_cost'] - 0.3) < 1e-12)
    ok('gaps', abs(g['gap_step'] - 0.03) < 1e-12 and abs(g['gap_lin'] - 0.02) < 1e-12)
    lim = {'occ_nll_U0': 0.010, 'occ_brier_U0': 0.005, 'men_occ_brier': 0.005}
    ok('clause usage', abs(clause_usage({'occ_nll_U0': 0.004, 'occ_brier_U0': 0.003, 'men_occ_brier': -0.001}, lim)
                           - 0.6) < 1e-12)
    two = compare_axes({'usage': (0.3, 0.600), 'occ_nll': (0.006, 0.600)},
                       {'usage': ctl, 'occ_nll': [(0.0, 0.65), (0.004, 0.598)]},
                       {'usage': 0.002, 'occ_nll': 0.002}, {'usage': 0.0, 'occ_nll': 0.0}, primary='usage')
    ok('explained on a secondary axis -> CONTROL_EXPLAINS overall', two['overall'] == 'CONTROL_EXPLAINS')
    try:
        clause_usage({'a': 0.1}, {'a': 0.1, 'b': 0.2})
        ok('mismatched clause sets refused', False)
    except ValueError:
        ok('mismatched clause sets refused', True)
    print('  example:', json.dumps(g))
    n_fail = res.count(False)
    print(f'SELFTEST {"PASS" if n_fail == 0 else "FAIL"}: {len(res) - n_fail}/{len(res)} checks')
    return n_fail == 0


if __name__ == '__main__':
    if sys.argv[1:] == ['selftest']:
        sys.exit(0 if selftest() else 1)
    print(__doc__)
