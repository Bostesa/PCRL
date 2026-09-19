"""Registered validation checks, including an independently written replay scorer.

The replay scorer below does not import the predecessor's loss helpers: it reloads the
stored selected-candidate probabilities and recomputes clipped, renormalised,
optionally person-weighted log loss from scratch, then compares with the audit's own
reported value.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import torch

from experiments.pcrl_direct_adversarial_v1 import inputs as dax
from experiments.pcrl_direct_adversarial_v1 import report as rep
from experiments.run_acs_residual_spectral import load_labels
from experiments.pcrl_nonlinear_rank_v1.inputs import HIST_ROOT

from . import evidence as ev
from . import projection as pj
from .common import ADVERSARIAL_RESULTS, OUT, read_json, utcnow, write_json_atomic

SEEDS = (0, 1, 2)
SENS = ('A/SEX', 'AB/SEX', 'A/RAC1P', 'AB/RAC1P')


def independent_log_loss(p, y, w=None) -> float:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y)
    keep = y >= 0
    y = y[keep].astype(int)
    if len(p) != len(y):
        raise ValueError('rows misaligned with valid labels')
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    nll = -np.log(p[np.arange(len(y)), y])
    if w is None:
        return float(nll.mean())
    w = np.asarray(w, dtype=np.float64)[keep]
    return float(np.dot(nll, w) / w.sum())


def replay(units, frames) -> dict:
    worst, checks = 0.0, 0
    for seed in SEEDS:
        _f, _p, labels, weights = frames[seed]
        for unit in units:
            path = rep.condition_file(OUT, seed, unit, 'predictions.npz')
            with np.load(path) as store:
                for weight in rep.WEIGHTS:
                    point = ev.point(seed, unit, 'test', weight)
                    wt = weights['test'] if weight == 'person_weighted' else None
                    for endpoint in SENS:
                        row = point['selected']['audit/' + endpoint]
                        target = endpoint.split('/')[1]
                        key = (f"audit/{row['view']}/{target}/{row['audit_budget']}/"
                               f"{row['candidate_id']}/test")
                        mine = independent_log_loss(store[key], labels['test'][target], wt)
                        worst = max(worst, abs(mine - point['losses'][endpoint]))
                        checks += 1
                    row = point['selected']['utility/same_residence']
                    key = f"utility/{row['view']}/same_residence/None/{row['candidate_id']}/test"
                    mine = independent_log_loss(store[key], labels['test']['same_residence'], wt)
                    worst = max(worst, abs(mine - point['utility']['same_residence']))
                    checks += 1
    return {'checks': checks, 'max_abs_difference': worst, 'pass': bool(worst < 1e-9)}


def reference_identity() -> dict:
    out = {}
    for ref, hist in (('ref_A0', 'A0'), ('ref_J', 'J')):
        worst = 0.0
        for seed in SEEDS:
            for weight in rep.WEIGHTS:
                a = ev.point(seed, ref, 'test', weight)
                b = ev.point(seed, hist, 'test', weight)
                for e in a['gains']:
                    worst = max(worst, abs(a['gains'][e] - b['gains'][e]))
                for t in a['utility']:
                    worst = max(worst, abs(a['utility'][t] - b['utility'][t]))
        out[ref] = {'historical': hist, 'max_abs_difference': worst, 'exact': worst == 0.0}
    return out


def historical_gamma1_reproduction() -> dict:
    rows = []
    for seed in SEEDS:
        for policy in ('L1', 'L2', 'C1'):
            for beta in ('b100', 'b300'):
                new = read_json(OUT / f'seed_{seed}' / 'fits' / f'N_A0_g100_{policy}_{beta}'
                                / 'fit_record.json')
                old = read_json(ADVERSARIAL_RESULTS / f'seed_{seed}' / 'fits'
                                / f'dax16_{policy}_{beta}' / 'fit_record.json')
                rows.append({'seed': seed, 'unit': f'N_A0_g100_{policy}_{beta}',
                             'identical': new['channel_hash'] == old['channel_hash'],
                             'selected_step_new': new['selected_step'],
                             'selected_step_old': old['selected_step']})
    return {'units': len(rows), 'identical': sum(r['identical'] for r in rows), 'rows': rows}


def selection_reconstruction() -> dict:
    total, match = 0, 0
    for seed in SEEDS:
        for path in sorted((OUT / f'seed_{seed}' / 'fits').glob('N_*/fit_record.json')):
            r = read_json(path)
            scores = [m['source'] + r['gamma'] * m['distortion'] + r['beta'] * m['penalty']
                      for m in r['monitor_scores']]
            total += 1
            match += int(int(np.argmin(scores)) == r['selected_index'])
    return {'units': total, 'reconstructed': match, 'pass': total == match}


def serialization_replay() -> dict:
    """Rebuild a sample of releases from serialised objects: every family, every anchor."""
    from experiments.pcrl_direct_adversarial_v1.channel import build_channel
    rows = []
    for seed in SEEDS:
        registry = dax.Registry.new()
        state = dax.load_frozen_state(seed, registry)
        from .run_fit_e import frozen_channel
        base = {'A0': state['channel'],
                'J': frozen_channel(seed, registry, state, 'J')['channel']}
        maps = joblib.load(OUT / f'seed_{seed}' / 'track_e_maps.joblib')['maps']
        sample = ['E_A0_C_k4', 'E_J_C_k2', 'E_J_LX_k8', 'E_A0_marginal_k6', 'E_J_pca_k4',
                  'E_A0_rand_k2', 'E_J_L_full', 'leace_J']
        for unit in sample:
            record = maps.get(unit)
            if record is None:
                continue
            channel = 'J' if unit.startswith('E_J') or unit == 'leace_J' else 'A0'
            z = base[channel]['test']
            if record['kind'] == 'whitened_projection':
                out = record['mu'] + (z - record['mu']) @ record['R'] @ record['P'] @ record['S']
            elif record['kind'] == 'original_metric_projection':
                out = record['mu'] + (z - record['mu']) @ record['Q']
            else:
                out = (z - record['mean']) @ record['projection'].T + record['mean']
            stored = np.load(OUT / f'seed_{seed}' / 'releases' / unit / 'releases.npz')['wire/A/test'][:, 4:]
            rows.append({'seed': seed, 'unit': unit,
                         'max_abs_difference': float(np.abs(out - stored).max())})
        for unit in ('N_J_g000_C1_b100', 'N_A0_g010_L2_b300', 'N_J_g100_C1_b300'):
            path = OUT / f'seed_{seed}' / 'fits' / unit / 'checkpoints.pt'
            if not path.exists():
                continue
            model = build_channel(16, state['a0'], state['channel']['representation_fit'])['model']
            model.load_state_dict(torch.load(path, map_location='cpu',
                                             weights_only=False)['selected_state'])
            x = dax.standardize(state['pca']['test'], state['a0']['input_mean'],
                                state['a0']['input_scale'])
            out = model.eval().release(x)
            stored = np.load(OUT / f'seed_{seed}' / 'releases' / unit / 'releases.npz')['wire/A/test'][:, 4:]
            rows.append({'seed': seed, 'unit': unit,
                         'max_abs_difference': float(np.abs(out - stored).max())})
    worst = max(r['max_abs_difference'] for r in rows)
    return {'replays': len(rows), 'max_abs_difference': worst, 'pass': worst < 1e-9,
            'rows': rows}


def exposure_match(units) -> dict:
    counts = {}
    for seed in SEEDS:
        for unit in units:
            canonical = ev.canonical(seed, unit)
            c = read_json(OUT / f'seed_{seed}' / canonical / 'complete.json')['audit_counts']
            counts[f'{seed}/{unit}'] = json.dumps(c, sort_keys=True)
    distinct = sorted(set(counts.values()))
    return {'units': len(counts), 'distinct_audit_count_profiles': len(distinct),
            'pass': len(distinct) == 1}


def incidents() -> dict:
    found = []
    for seed in SEEDS:
        for q in (OUT / f'seed_{seed}' / 'quarantine').glob('*/quarantine.json'):
            found.append(read_json(q))
    return {'quarantined_attempts': len(found), 'records': found}


def run(units) -> dict:
    frames = {seed: load_labels(HIST_ROOT, seed) for seed in SEEDS}
    result = {'generated_utc': utcnow(),
              'reference_identity': reference_identity(),
              'historical_gamma1_reproduction': historical_gamma1_reproduction(),
              'selection_reconstruction': selection_reconstruction(),
              'serialization_replay': serialization_replay(),
              'exposure_match': exposure_match([u for u in units if u.startswith(('N_', 'E_', 'ref_', 'leace_J', 'splince_J'))]),
              'independent_replay': replay(units, frames),
              'incidents': incidents()}
    write_json_atomic(OUT / 'VALIDATION.json', result)
    return result
