"""Stage D: FIT2 24-clause rule for every checkpoint/service and the latest-common-positive selection."""
from __future__ import annotations
import argparse
import json

from .common import bind, Source, bases, role_forward, npz, write_out, task_metrics_np

CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
ARMS = ('T_BOTH', 'L_BOTH', 'J_BOTH', 'S12', 'S21', 'I_ONLY')
SEQ = ('S12', 'S21')


def selectable(arm):
    return [f'P2E{e}' for e in (1, 3, 5, 10)] if arm in SEQ else [f'E{e}' for e in (1, 3, 5, 10)]


def fallback(arm):
    return 'P2E10' if arm in SEQ else 'E10'


def seed_clauses(y, wires, refs):
    out = []
    for i, task in enumerate(('income', 'occupation')):
        K = (2, 6)[i]
        own = {k: float(v.mean()) for k, v in task_metrics_np(y[task], wires[i][:, 16 + K:16 + 2 * K], K).items()}
        for ref in ('U0', 'Ucal'):
            r = {k: float(v.mean()) for k, v in task_metrics_np(y[task], refs[f'{ref}_{i + 1}'], K).items()}
            vals = {'accuracy_loss': r['accuracy'] - own['accuracy'], 'nll_excess': own['nll'] - r['nll'], 'brier_excess': own['brier'] - r['brier']}
            for m, cap in CAPS.items():
                out.append({'task': task, 'reference': ref, 'metric': m, 'value': vals[m], 'pass': bool(vals[m] <= cap)})
    return out


def run(spec):
    import numpy as np
    import torch
    active, cfg, get, guard = bind(spec)
    fit = json.loads(get(cfg['fit2_utility']).read_text())
    sel = json.loads(get(cfg['selection_lock_item']).read_text())
    V = cfg['validation']
    feat = npz(get(V['features']))
    y = {}
    for t in ('income', 'occupation'):
        z = npz(get(V[t])); assert np.array_equal(z['row_id'], feat['row_id']); y[t] = z[t].astype(np.int64)
    table = {}; maxdiff = 0.0; mism = []
    for seed in ('0', '1'):
        guard()
        S = cfg['seeds'][seed]
        refs = npz(get(V[f'references_s{seed}'])); assert np.array_equal(refs['row_id'], feat['row_id'])
        src = Source(get(S['checkpoint'])); B, _ = bases(get(S['geometry']))
        scales = [float(x) for x in json.loads(get(S['adapter_binding']).read_text())['scales']]
        X, names = feat['X'], feat['feature_names']
        services = {('RAW_PROJ', 'E0'): ('raw', None)}
        pp = torch.load(get(S['o_previous']), map_location='cpu', weights_only=False)
        services[('O_PREVIOUS', 'E10')] = ('adapter', [None, {'R': pp['R'], 'b': pp['b_res']}])
        for arm, labels in cfg['checkpoint_paths'][seed].items():
            for lab, it in labels.items():
                p = torch.load(get(it), map_location='cpu', weights_only=False)
                st = [{'R': p['states'][i]['R'], 'b': p['states'][i]['b']} for i in (0, 1)]
                services[(arm, lab)] = ('adapter', st)
                if arm == 'J_BOTH':
                    services[(f'HYB_JI_RO@{lab}', lab)] = ('adapter', [st[0], None])
                    services[(f'HYB_RI_JO@{lab}', lab)] = ('adapter', [None, st[1]])
        for (name, lab), (mode, st) in services.items():
            guard()
            w = role_forward(src, B, X, names, mode='raw_proj') if mode == 'raw' else role_forward(src, B, X, names, states=st, scales=scales)
            cl = seed_clauses(y, w, refs)
            key = f'{seed}|{name}|{lab}'
            table[key] = {'all_point_pass': all(c['pass'] for c in cl), 'clauses': cl}
            camp = fit['table'].get(key)
            if camp is None:
                mism.append(f'{key} missing in campaign table'); continue
            cmap = {(c['task'], c['reference'], c['metric']): c for c in camp['clauses']}
            for c in cl:
                cc = cmap[(c['task'], c['reference'], c['metric'])]
                maxdiff = max(maxdiff, abs(cc['value'] - c['value']))
                if cc['point_pass'] != c['pass']:
                    mism.append(f'{key} {c["task"]} {c["reference"]} {c["metric"]} pass differs')
            if camp['all_point_pass'] != table[key]['all_point_pass']:
                mism.append(f'{key} all_point_pass differs')
    missing = sorted(set(fit['table']) - set(table))
    selection = {}
    for arm in ARMS:
        feas = [l for l in selectable(arm) if all(table[f'{s}|{arm}|{l}']['all_point_pass'] for s in ('0', '1'))]
        chosen = max(feas, key=lambda l: int(l.split('E')[-1])) if feas else None
        selection[arm] = {'feasible_common_labels': feas, 'selected_label': chosen, 'attacked_label': chosen or fallback(arm),
                          'clauses_checked_per_label': 24}
        for src_name, s in (('FIT2_UTILITY', fit['selection'][arm]), ('SELECTION_LOCK', sel['selection'][arm])):
            if s['selected_label'] != chosen or s['attacked_label'] != selection[arm]['attacked_label'] or s['feasible_common_labels'] != feas:
                mism.append(f'{arm} selection differs from {src_name}')
    ref_ok = all(table[f'{s}|RAW_PROJ|E0']['all_point_pass'] for s in ('0', '1'))
    prev_ok = all(table[f'{s}|O_PREVIOUS|E10']['all_point_pass'] for s in ('0', '1'))
    if ref_ok != fit['RAW_PROJ_point_eligible'] or prev_ok != fit['O_PREVIOUS_point_eligible']:
        mism.append('reference eligibility differs')
    worst = {}
    for k, v in table.items():
        worst[k] = min(CAPS[c['metric']] - c['value'] for c in v['clauses'])
    res = {'schema': 'TRA1-independent-stage-D-v1', 'unit_id': active['unit_id'], 'rows': int(len(feat['row_id'])),
           'entries_recomputed': len(table), 'campaign_entries_missing_here': missing, 'max_abs_clause_value_diff': maxdiff,
           'selection': selection, 'RAW_PROJ_point_eligible': ref_ok, 'O_PREVIOUS_point_eligible': prev_ok,
           'min_headroom_by_entry': worst, 'mismatches': mism}
    res['pass'] = bool(not mism and not missing and maxdiff <= 1e-12)
    return write_out(cfg, res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    print(json.dumps(run(ap.parse_args().spec)['pass']))
