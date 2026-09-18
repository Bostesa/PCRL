"""Stage 5a: is the existing marginal spectral arm already a SARL adaptation?

The aim is to **avoid fits**. If the repository's marginal spectral arms already are
SARL-style residual-teacher adaptations under their exact parametrisation, they are
credited and reused; no duplicate model is fitted merely to attach a published name.

What SARL (arXiv 1910.07423, Appendix B eq. 24) solves, in its empirical form, is
``min Tr[G' B G]`` over ``G'G = I`` with ``B = lambda*S~'S~ - (1-lambda)*Y~'Y~`` in the
coordinates of an orthonormal basis of ``range(X~')`` -- which *is* the whitening, not
a step before it. Writing ``U`` and ``P`` for the whitened cross-covariance Grams,
that is ``B = lambda*P - (1-lambda)*U``, so SARL takes the top eigenvectors of
``U - lambda_bar*P`` with ``lambda_bar = lambda/(1-lambda)``.

This module checks that correspondence **numerically on the saved training matrices**
rather than asserting it, and measures the three places a reimplementation diverges:
eigenvalue ordering, the fixed-``r`` versus sign-test rank rule, and the gap at the
truncation boundary (repeated-eigenspace ambiguity).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.acs_residual_spectral import ARMS, _trace_normalize, moment_penalty
from experiments.pcrl_nonlinear_rank_v1.inputs import (Registry, load_matrix_diagnostics,
                                                       load_pools, load_representation_labels,
                                                       load_spectral_model, write_json)
from experiments.pcrl_nonlinear_rank_v1.run_fit import limit_threads

from .objective import spectral_solution

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/pcrl_invariant_baselines_v1'
SCHEMA = {'SEX': 2, 'RAC1P': 9, 'public_coverage': 2}
MARGINAL_ARMS = ('spectral_M025', 'spectral_M1')
RANK = 16


def subspace_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Projector Frobenius distance -- the numerically reliable 'same subspace' test.

    Bases are compared this way and never entrywise: an eigenbasis is defined only up
    to sign and up to rotation inside a repeated eigenspace.
    """
    return float(np.linalg.norm(a @ a.T - b @ b.T))


def marginal_grams(seed: int, registry: Registry):
    """Rebuild the marginal sensitive Grams two ways: the repository's and SARL's."""
    model = load_spectral_model(seed, registry)
    diag = load_matrix_diagnostics(seed, registry)
    _, teacher, anchors = load_pools(seed, registry)
    labels, _ = load_representation_labels(seed, registry)

    t = np.asarray(teacher['representation_fit'], dtype=np.float64)
    ha = np.asarray(anchors['representation_fit/A'], dtype=np.float64)
    v = model.features(t, ha)
    n, q = v.shape
    basis = np.ones((n, 1))

    # Utility matrix, rebuilt exactly as the frozen study did.
    q_local = model.qA.transform(ha[:, [1, 3]])
    r = t - q_local @ np.linalg.lstsq(q_local, t, rcond=1e-10)[0]
    r -= r.mean(0)
    cross = v.T @ r / n
    utility, utility_trace = _trace_normalize(cross @ cross.T)

    repository = np.zeros((q, q))          # per-attribute trace-normalised, then averaged
    raw_sum = np.zeros((q, q))             # SARL: one raw sum over every attribute and class
    per_attribute = {}
    for name, classes in SCHEMA.items():
        y = labels[name]
        prior = np.bincount(y[y >= 0], minlength=classes) / max(int((y >= 0).sum()), 1)
        predicted = np.tile(prior, (n, 1))
        normalized, record = moment_penalty(v, basis, y, predicted, classes)
        repository += normalized / len(SCHEMA)
        valid = y >= 0
        attribute_raw = np.zeros((q, q))
        for c in range(classes):
            residual = (y[valid] == c).astype(float) - predicted[valid, c]
            g = v[valid].T @ (basis[valid] * residual[:, None]) / max(int(valid.sum()), 1)
            attribute_raw += g @ g.T
        raw_sum += attribute_raw
        per_attribute[name] = {'raw_trace': float(np.trace(attribute_raw)),
                               'stored_raw_trace': record['raw_trace'],
                               'raw_trace_abs_error': abs(float(np.trace(attribute_raw))
                                                          - record['raw_trace']),
                               'classes': classes,
                               'unsupported_classes': record['unsupported_classes']}
    sarl, sarl_trace = _trace_normalize(raw_sum)
    return {'V': v, 'U': utility, 'utility_trace': utility_trace,
            'P_repository': repository, 'P_sarl': sarl, 'P_sarl_trace': sarl_trace,
            'per_attribute': per_attribute, 'model': model, 'diagnostics': diag,
            'rows': n, 'whitened_rank': q}


def audit_seed(seed: int, registry: Registry) -> dict:
    state = marginal_grams(seed, registry)
    utility, repository, sarl = state['U'], state['P_repository'], state['P_sarl']
    model = state['model']

    checks = []
    for arm in MARGINAL_ARMS:
        role, coefficient = ARMS[arm]
        assert role == 'marginal', arm
        a_repo = utility - coefficient * repository
        w_repo, values_repo = spectral_solution(a_repo, RANK)

        # (0) the rebuilt matrix must reproduce the stored historical arm bitwise
        stored = model.maps[arm]
        # (1) ordering: the SAME eigenvectors, ascending-vs-descending, must give the
        #     same SUBSPACE; truncating by index in the other order would not.
        ascending = np.linalg.eigh((a_repo + a_repo.T) / 2)[1][:, :RANK]
        # (2) rank rule: SARL keeps only strictly positive eigenvalues of U - lambda_bar P
        tolerance = 1e-10 * max(1.0, float(abs(values_repo).max()))
        positive = int((values_repo > tolerance).sum())
        # (3) truncation-boundary gap: a tie here makes the selected subspace ill-posed
        gap = float(values_repo[RANK - 1] - values_repo[RANK]) if RANK < len(values_repo) else None

        # SARL's own gram, with lambda_bar re-fitted so the two parametrisations are
        # compared at matched sensitive-penalty STRENGTH rather than at matched lambda.
        scale = float(np.trace(repository) / max(np.trace(sarl), 1e-300))
        w_sarl, _ = spectral_solution(utility - coefficient * scale * sarl, RANK)

        checks.append({
            'arm': arm, 'coefficient': coefficient, 'rank': RANK,
            'stored_map_bitwise_identical': bool(np.array_equal(w_repo, stored)),
            'stored_map_max_abs_difference': float(np.max(abs(w_repo - stored))),
            'ordering_ascending_subspace_distance': subspace_distance(w_repo, ascending),
            'ordering_note': ('Ascending truncation selects the BOTTOM 16 eigenvectors. A large '
                              'distance here is the expected, correct result: it quantifies the '
                              'bug a reimplementation ships if it copies SARL sort order without '
                              'reversing the sign convention.'),
            'positive_eigenvalue_count': positive,
            'sarl_rank_rule_would_select': min(RANK, positive),
            'fixed_rank_equals_sign_rule': bool(min(RANK, positive) == RANK),
            'truncation_boundary_gap': gap,
            'eigenvalues_around_boundary': values_repo[max(0, RANK - 3):RANK + 3].tolist(),
            'sarl_gram_subspace_distance': subspace_distance(w_repo, w_sarl),
            'sarl_gram_trace_rescale': scale,
        })

    traces = {k: v['raw_trace'] for k, v in state['per_attribute'].items()}
    return {
        'seed': seed, 'rows': state['rows'], 'whitened_rank': state['whitened_rank'],
        'per_attribute': state['per_attribute'],
        'per_attribute_raw_traces': traces,
        'per_attribute_trace_ratio_max_over_min': (max(traces.values()) / min(traces.values())
                                                   if min(traces.values()) > 0 else None),
        'arms': checks,
    }


def run(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    limit_threads()
    tick = time.perf_counter()
    results = {}
    for seed in seeds:
        registry = Registry.new()
        results[str(seed)] = audit_seed(seed, registry)
        print('ALIAS_AUDIT', seed, 'done', flush=True)
    record = {
        'question': ('Are the existing MARGINAL spectral arms already SARL-style '
                     'residual-teacher adaptations under their exact parametrisation?'),
        'derivation': (
            "SARL Appendix B eq. 24 minimises Tr[G'BG] over G'G=I with "
            "B = lambda*S~'S~ - (1-lambda)*Y~'Y~ in the coordinates of an orthonormal basis of "
            "range(X~'), which IS the whitening. With U and P the whitened cross-covariance "
            'Grams this is B = lambda*P - (1-lambda)*U, so SARL takes the TOP eigenvectors of '
            'U - lambda_bar*P with lambda_bar = lambda/(1-lambda). The repository computes '
            'A = U - coefficient*P and takes its top-r eigenvectors, with V whitened, U the '
            'trace-normalised Gram of the residualised teacher R, and the marginal P built from '
            'an intercept-only basis and the class PRIOR as the nuisance -- so its per-class '
            "moment is V'e_c/n with e_c the CENTRED one-hot, i.e. exactly the whitened "
            'sensitive cross-covariance SARL uses.'),
        'scope': (
            "Only the MARGINAL arms are candidate SARL aliases. The local and coalition arms use "
            'the residualised moment e_j = onehot(S_j) - m_j(H_c), which is a CONDITIONAL '
            'construction (U-FaTE-adjacent), not SARL. K-TOpt and U-FaTE are related '
            'formulations, not automatically distinct implemented baselines, and are not fitted.'),
        'known_source_hazard': (
            'The official SARL code takes the r algebraically smallest eigenvectors with no sign '
            'test, which its own Theorem 3 excludes. Any equivalence claimed here is to the '
            'PAPER, not to that code path.'),
        'seeds': results,
        'runtime_seconds': time.perf_counter() - tick,
    }
    write_json(out / 'ALIAS_AUDIT.json', record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    run(a.out, tuple(a.seeds))


if __name__ == '__main__':
    main()
