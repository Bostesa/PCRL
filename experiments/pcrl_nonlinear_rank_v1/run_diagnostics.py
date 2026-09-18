"""Phase 2b: mechanism diagnostics. Uses fitted maps only; fits no attacker.

Two diagnostics:

1. The rotation decomposition (see :mod:`diagnostics`): how much of the nonlinear
   refinement's training-objective gain is reachable by rotating within the
   original subspace, which provably changes nothing an attacker can recover.
2. A held-out nuisance calibration diagnostic on already-used rows, permitted by
   the protocol. It is a calibration check, not a new nuisance sweep: no nuisance
   model is refitted or retuned anywhere in this study.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .diagnostics import rotation_invariance_check, rotation_share
from .inputs import OUT, Registry, load_pools, load_representation_labels, read_json, write_json
from .maps import (ROLE_SPEC, build_rank_context, build_roles, condition_name, make_objective,
                   oof_predictions)
from .objective import POLICIES, spectral_solution
from .run_fit import limit_threads, machine_state


def nuisance_calibration(seed: int, registry: Registry) -> dict:
    """Do the frozen OOF nuisances still calibrate on rows they did not fit?

    The out-of-fold predictions are already held out by construction (fold f's
    rows are predicted by the models fitted on the other two folds). This reports
    their calibration so that a nonzero penalty is not read as evidence of
    disclosure when it could be nuisance error. No model is refitted.
    """
    from .inputs import load_matrix_diagnostics, load_spectral_model
    model = load_spectral_model(seed, registry)
    diag = load_matrix_diagnostics(seed, registry)
    _, _teacher, anchors = load_pools(seed, registry)
    labels, _ = load_representation_labels(seed, registry)
    ha = np.asarray(anchors['representation_fit/A'], dtype=np.float64)
    hb = np.asarray(anchors['representation_fit/B'], dtype=np.float64)
    bases = {'local': model.qA.transform(ha[:, [1, 3]]),
             'coalition': model.qAB.transform(np.column_stack((ha[:, [1, 3]], hb[:, 1])))}
    folds = np.asarray(diag['fold_assignments'], dtype=int)
    out = {}
    for name, basis_key, attribute, classes in ROLE_SPEC:
        basis = bases[basis_key]
        y = labels[attribute]
        predicted = oof_predictions(model, basis, folds, basis_key, attribute, classes)
        valid = y >= 0
        p = np.clip(predicted[valid], 1e-12, 1.)
        p = p / p.sum(1, keepdims=True)
        yy = y[valid]
        onehot = np.eye(classes)[yy]
        log_loss = float(-np.mean(np.log(p[np.arange(len(yy)), yy])))
        prior = onehot.mean(0)
        prior_loss = float(-np.mean(np.log(np.clip(prior[yy], 1e-12, 1.))))
        # Reliability: mean predicted vs observed frequency, per class, 10 equal-count bins.
        reliability = []
        for c in range(classes):
            if onehot[:, c].sum() == 0:
                continue
            order = np.argsort(p[:, c])
            bins = np.array_split(order, 10)
            gaps = [abs(p[b, c].mean() - onehot[b, c].mean()) for b in bins if len(b)]
            reliability.append({'class': c, 'support': int(onehot[:, c].sum()),
                                'max_bin_calibration_gap': float(max(gaps)),
                                'mean_bin_calibration_gap': float(np.mean(gaps))})
        out[name] = {'attribute': attribute, 'classes': classes, 'valid_rows': int(valid.sum()),
                     'oof_log_loss': log_loss, 'prior_log_loss': prior_loss,
                     'improvement_over_prior': prior_loss - log_loss,
                     'mean_residual_abs_max': float(np.max(np.abs((onehot - p).mean(0)))),
                     'per_class_reliability': reliability}
    out['scope'] = ('Out-of-fold predictions on already-used representation-fit rows. A '
                    'calibration diagnostic only: no nuisance model was refitted or retuned, and '
                    'residual moments can reflect nuisance error as well as disclosure.')
    return out


def run(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    limit_threads()
    out = Path(out)
    summary = {}
    for seed in seeds:
        marker = out / f'seed_{seed}' / 'diagnostics.json'
        if marker.exists():
            summary[str(seed)] = read_json(marker)
            print('DIAG_REUSED', seed, flush=True)
            continue
        tick = time.perf_counter()
        registry = Registry.new()
        state = build_roles(seed, registry)
        import joblib
        model = joblib.load(out / f'seed_{seed}' / 'maps.joblib')
        record = {'seed': seed, 'rotation': {}, 'invariance': {}}
        for rank in (16, 8):
            context = build_rank_context(state, rank, seed)
            for policy in POLICIES:
                linear = make_objective(state, context, 'original', policy)
                nonlinear = make_objective(state, context, 'nonlinear', policy)
                w_lin, _ = spectral_solution(linear.spectral_matrix(), rank)
                name = condition_name('nonlinear', rank, policy)
                record['rotation'][name] = rotation_share(nonlinear, w_lin, model.maps[name])
                record['invariance'][name] = rotation_invariance_check(nonlinear, w_lin, seed)
                print('DIAG', seed, name,
                      'rotation_share=%s' % (
                          'undefined' if record['rotation'][name]['rotation_only_share'] is None
                          else '%.3f' % record['rotation'][name]['rotation_only_share']),
                      flush=True)
        record['nuisance_calibration'] = nuisance_calibration(seed, registry)
        record['runtime_seconds'] = time.perf_counter() - tick
        record['inputs'] = registry.dump()
        write_json(marker, record)
        summary[str(seed)] = record
    write_json(out / 'DIAGNOSTICS_SUMMARY.json', {'seeds': summary, 'machine': machine_state()})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    run(a.out, tuple(a.seeds))


if __name__ == '__main__':
    main()
