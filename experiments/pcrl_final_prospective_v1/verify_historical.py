"""Tier A: verify the frozen panel objects against the historical 2018 evidence.

1. Rebuild every panel release on each anchor's 2018 test rows with this study's
   loader and compare the token law / fixed decoder with the archived evaluation.
2. For anchor 0 (audits restored), replay each archived selected predictor on the
   rebuilt release and compare probabilities and per-person losses.
3. Recompute the headline Q/D17/D33 distinctions versus J, and D17 versus Q, from
   archived per-person losses with the predecessor's household bootstrap, and
   compare with PAIRED_BOUNDS.json. Historical scores are never rewritten.
"""
from __future__ import annotations
import json

import joblib
import numpy as np
import pandas as pd

from experiments.pcrl_task_directed_release_v1.evaluation import routed_probabilities, _subset_pool, _subset_release
from experiments.pcrl_task_directed_release_v1.audits import expected_token_loss
from experiments.pcrl_task_directed_release_v1.uncertainty import paired_household_bounds
from .common import ANCHORS, OUT, PANEL, PRIMARY_ROLES, RESTORED, ROOT, TASK_ROLE, atomic_json, now, sha
from .releases import FrozenReleases
from .transport import CSV_2018, FIXED

ROLES = PRIMARY_ROLES+(TASK_ROLE,)
PRED_OUT = ROOT/'results/pcrl_task_directed_release_v1'


def safe(role):
    return role.replace(':', '__').replace('/', '__')


def eval_npz(anchor, short, role):
    base = RESTORED/f'anchor_{anchor}'/'evaluation'/PANEL[short]
    receipt = json.loads((base/'COMPLETE.json').read_text())
    rel = f'test/{safe(role)}.npz'
    if sha(base/rel) != receipt['artifact_hashes'][rel]:
        raise ValueError(f'Archived evaluation hash mismatch {anchor} {short} {role}')
    with np.load(base/rel, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def test_inputs(anchor):
    base = FIXED/f'seed_{anchor}'
    import torch
    with np.load(base/'split_rows.npz') as z:
        rows = z['test']
    with np.load(base/'pca.npz') as z:
        pca = z['test']
    with np.load(base/'anchors.npz') as z:
        ha, hb = z['test/A'], z['test/B']
    with np.load(base/'training/J/releases.npz') as z:
        j = z['wire/A/test'][:, 4:].copy()
    state = torch.load(base/'training/J/final.pt', map_location='cpu', weights_only=False)['model_state']
    x = np.asarray((np.asarray(pca, np.float64)-state['input_mean'].numpy())/state['input_scale'].numpy(), np.float32)
    keys = pd.read_csv(CSV_2018, usecols=['SERIALNO', 'SPORDER', 'PWGTP'], dtype={'SERIALNO': str})
    keys = keys.iloc[rows]
    ids = (keys.SERIALNO.astype(str)+':'+keys.SPORDER.astype(str)).to_numpy()
    return {'x': x, 'ha': ha, 'hb': hb, 'J': j, 'ids': ids, 'households': keys.SERIALNO.to_numpy(),
            'weights': keys.PWGTP.to_numpy(float)}


def release_checks(anchor, frozen, data):
    enc = {'test': frozen.encode(data['x'], data['ha'])}
    pools = {'test': {k: data[k] for k in ('x', 'ha', 'J')}}
    out, releases = {}, {}
    for short in PANEL:
        release = frozen.release(short, pools, enc)['test']
        releases[short] = release
        for role in ROLES:
            stored = eval_npz(anchor, short, role)
            pos = {v: i for i, v in enumerate(data['ids'])}
            idx = np.array([pos[v] for v in stored['ids']])
            p = release['token_probs'][idx]
            record = {'rows': len(idx), 'token_law_bitwise': bool(np.array_equal(p, stored['token_probs'])),
                      'token_law_max_abs_error': float(np.max(np.abs(p-stored['token_probs'])))}
            out[f'{short}|{role}'] = record
    return out, releases, enc


def predictor_replay(anchor, data, releases):
    """Anchor 0 only: archived registries are restored for the core panel."""
    out = {}
    for short in PANEL:
        reg = joblib.load(RESTORED/f'anchor_{anchor}'/'audits'/PANEL[short]/'registry.joblib')
        for role in ROLES:
            stored = eval_npz(anchor, short, role)
            record = reg['roles'][role]
            pos = {v: i for i, v in enumerate(data['ids'])}
            idx = np.array([pos[v] for v in stored['ids']])
            rows = {'ha': data['ha'][idx], 'hb': data['hb'][idx]}
            rel = {k: (None if v is None else np.asarray(v)[idx]) for k, v in releases[short].items()}
            q, p = routed_probabilities(record['candidates'][record['selection']], rows, rel)
            loss = expected_token_loss(q, p, stored['y'])
            out[f'{short}|{role}'] = {
                'selection': record['selection'],
                'probabilities_max_abs_error': float(np.max(np.abs(q-stored['probabilities']))),
                'loss_max_abs_error': float(np.max(np.abs(loss-stored['loss']))),
                'loss_bitwise': bool(np.array_equal(loss, stored['loss']))}
    return out


def paired(anchor_arrays):
    """anchor_arrays: list of dicts with ids/households/weights/diff (already aligned)."""
    return anchor_arrays


def headline_contrasts():
    """Recompute point estimates and SEs from archived per-person losses."""
    specs = []
    for m in ('Q', 'D17', 'D33'):
        for w in ('unweighted', 'PWGTP'):
            specs.append({'id': f'task|{m}-J|{w}', 'role': TASK_ROLE, 'w': w, 'plus': m, 'minus': 'J'})
            for role in PRIMARY_ROLES:
                specs.append({'id': f'sens|J-{m}|{role}|{w}', 'role': role, 'w': w, 'plus': 'J', 'minus': m})
    for w in ('unweighted', 'PWGTP'):
        specs.append({'id': f'task|Q-D17|{w}', 'role': TASK_ROLE, 'w': w, 'plus': 'Q', 'minus': 'D17'})
        for role in PRIMARY_ROLES:
            specs.append({'id': f'sens|D17-Q|{role}|{w}', 'role': role, 'w': w, 'plus': 'D17', 'minus': 'Q'})
    contrasts = []
    for s in specs:
        anchors = []
        for a in ANCHORS:
            plus, minus = eval_npz(a, s['plus'], s['role']), eval_npz(a, s['minus'], s['role'])
            if not np.array_equal(plus['ids'], minus['ids']):
                raise ValueError('Archived role masks differ between interfaces')
            weights = np.ones(len(plus['ids'])) if s['w'] == 'unweighted' else plus['weights']
            anchors.append({'household': plus['households'].astype(str), 'difference': plus['loss']-minus['loss'],
                            'weights': weights})
        contrasts.append({'id': s['id'], 'anchors': anchors})
    result = paired_household_bounds(contrasts, n_boot=10000, seed=20260921, alpha=.05)
    return specs, result


def match_paired_bounds(specs, result):
    """Match each recomputed contrast with the archived registered endpoint, if any."""
    pb = json.loads((PRED_OUT/'PAIRED_BOUNDS.json').read_text())
    cs = json.loads((PRED_OUT/'CONTRASTS.json').read_text())
    names = {k: v for k, v in PANEL.items()}
    index = {}
    for e in cs['endpoints']:
        terms = tuple(sorted((t['configuration'], t['coefficient']) for t in e['terms']))
        index.setdefault((terms, e['role'], e['weighting']), e['id'])
    rows = []
    for s in specs:
        terms = tuple(sorted(((names[s['plus']], 1.0), (names[s['minus']], -1.0))))
        key = (terms, s['role'], s['w'])
        mine = result['bounds'][s['id']]
        row = {'id': s['id'], 'estimate': mine['estimate'], 'bootstrap_se': mine['bootstrap_se'],
               'anchor_estimates': mine['anchor_estimates']}
        if key in index:
            old = pb['bounds'][index[key]]
            row.update(archived_endpoint=index[key], archived_estimate=old['estimate'],
                       archived_se=old['bootstrap_se'],
                       estimate_abs_error=abs(old['estimate']-mine['estimate']),
                       se_abs_error=abs(old['bootstrap_se']-mine['bootstrap_se']))
        rows.append(row)
    return rows


def main():
    report = {'created_utc': now(), 'scope': '2018 development evidence already used; verification only'}
    releases_by_anchor = {}
    for anchor in ANCHORS:
        frozen = FrozenReleases(anchor)
        data = test_inputs(anchor)
        checks, releases, _ = release_checks(anchor, frozen, data)
        report[f'release_checks_anchor{anchor}'] = checks
        if anchor == 0:
            report['predictor_replay_anchor0'] = predictor_replay(anchor, data, releases)
        releases_by_anchor[anchor] = releases
    specs, result = headline_contrasts()
    report['headline'] = match_paired_bounds(specs, result)
    report['bootstrap'] = result['bootstrap']
    atomic_json(OUT/'private/HISTORICAL_VERIFICATION_FULL.json', report)
    summary = {'all_token_laws_bitwise': all(v['token_law_bitwise'] for a in ANCHORS for v in report[f'release_checks_anchor{a}'].values()),
               'max_replay_probability_error': max(v['probabilities_max_abs_error'] for v in report['predictor_replay_anchor0'].values()),
               'max_replay_loss_error': max(v['loss_max_abs_error'] for v in report['predictor_replay_anchor0'].values()),
               'max_headline_estimate_error': max(r.get('estimate_abs_error', 0.) for r in report['headline']),
               'max_headline_se_error': max(r.get('se_abs_error', 0.) for r in report['headline']),
               'matched_endpoints': sum('archived_endpoint' in r for r in report['headline']),
               'recomputed_endpoints': len(report['headline'])}
    print(json.dumps(summary, indent=1))
    return report, summary


if __name__ == '__main__':
    main()
