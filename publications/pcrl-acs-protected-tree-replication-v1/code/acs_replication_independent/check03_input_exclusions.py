"""Independent check 3 (INPUT EXCLUSIONS).

Allow-list check on every exported development role (SEALED_ASSESSMENT has no export): model features are only
AGEP, WKHP and one-hot levels of COW, SCHL, MAR, POBP (+ '<VAR>=__OTHER__'); no forbidden variable appears; widths
agree across roles, with the extraction receipt, the saved tree teachers and the U0 networks (both seeds); and the
feature matrix X is fully reproduced from the six permitted raw inputs alone (so X carries nothing else).
Only names, shapes, counts and max deviations are reported (no person-level values). No acs_replication import.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[1]
RES = WT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
PRIV = WT / '.private' / 'acs_replication'
EXP = PRIV / 'exports'
OUT = RES / 'independent' / 'check03_input_exclusions.json'

PERMITTED_NUM = ['AGEP', 'WKHP']
PERMITTED_CAT = ['COW', 'SCHL', 'MAR', 'POBP']
EXTRA_FORBIDDEN = ['SEX', 'RAC1P', 'RAC2P', 'RAC3P', 'HISP', 'OCCP', 'SOCP', 'INDP', 'NAICSP', 'PINCP', 'PERNP', 'WAGP',
                   'SEMP', 'INTP', 'RETP', 'SSP', 'SSIP', 'PAP', 'OIP', 'POVPIP', 'RELP', 'RELSHIPP', 'SERIALNO', 'SPORDER',
                   'PUMA', 'POWPUMA', 'POWSP', 'ST', 'PWGTP', 'ADJINC', 'ESR', 'FSEXP', 'FOCCP', 'FPINCP']
DEV_ROLES = ['MODEL_FIT', 'MODEL_SELECT', 'CALIBRATION', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT']


def base_var(name: str) -> str:
    return name.split('=', 1)[0]


def main():
    lock = json.loads((RES / 'SCIENCE_LOCK.json').read_text())
    xrec = json.loads((PRIV / 'extraction' / 'EXTRACTION_RECEIPT.json').read_text())
    forbidden = sorted(set(lock['benchmark']['forbidden_inputs']) | set(EXTRA_FORBIDDEN) | {f'PWGTP{i}' for i in range(1, 81)})
    fails, ev = [], {}

    roles_present = sorted(p.name for p in EXP.iterdir() if p.is_dir())
    ev['export_roles_present'] = roles_present
    if any('SEALED' in r for r in roles_present):
        fails.append('a SEALED export directory exists')

    enc = xrec['encoding']
    names_ref = list(enc['feature_names'])
    # receipt names themselves follow the allow-list and the registered level lists
    expected = PERMITTED_NUM + [f'{v}={lvl}' for v in PERMITTED_CAT for lvl in enc['categorical'][v] + ['__OTHER__']]
    ev['receipt_feature_names_equal_allowlist_construction'] = names_ref == expected
    if names_ref != expected:
        fails.append('receipt feature names differ from AGEP,WKHP + registered one-hot levels')
    ev['receipt_input_width'] = xrec['input_width']
    ev['levels_per_categorical'] = {v: len(enc['categorical'][v]) for v in PERMITTED_CAT}

    # MODEL_FIT statistics for the independent reconstruction
    mf = np.load(EXP / 'MODEL_FIT' / 'raw_inputs.npz', allow_pickle=False)
    stats = {}
    for v in PERMITTED_NUM:
        x = mf[v].astype(np.float64)
        stats[v] = (float(x.mean()), float(x.std(ddof=0)))
    ev['model_fit_numeric_stats_match_receipt'] = {v: {'abs_diff_mean': abs(stats[v][0] - enc['numeric'][v]['mean']),
                                                       'abs_diff_sd': abs(stats[v][1] - enc['numeric'][v]['sd'])}
                                                   for v in PERMITTED_NUM}
    if any(d['abs_diff_mean'] > 1e-9 or d['abs_diff_sd'] > 1e-9 for d in ev['model_fit_numeric_stats_match_receipt'].values()):
        fails.append('MODEL_FIT numeric stats differ from receipt')
    # category levels = MODEL_FIT levels with count >= min_count
    lvl_check = {}
    for v in PERMITTED_CAT:
        vals, cnt = np.unique(mf[v], return_counts=True)
        kept = sorted(str(a) for a, c in zip(vals, cnt) if c >= enc['min_count'])
        lvl_check[v] = {'n_kept_independent': len(kept), 'n_registered': len(enc['categorical'][v]),
                        'equal': kept == sorted(enc['categorical'][v])}
        if not lvl_check[v]['equal']:
            fails.append(f'{v} levels differ from MODEL_FIT count>={enc["min_count"]} rule')
    ev['category_levels_min_count_rule'] = lvl_check

    per_role = {}
    for role in DEV_ROLES:
        r = {}
        fz = np.load(EXP / role / 'features.npz', allow_pickle=False)
        names = [str(s) for s in fz['feature_names']]
        X = fz['X']
        bad = [n for n in names if not (n in PERMITTED_NUM or re.fullmatch(r'(COW|SCHL|MAR|POBP)=.+', n))]
        hits = sorted({base_var(n) for n in names} & set(forbidden))
        r['n_features'] = len(names)
        r['X_shape'] = list(X.shape)
        r['X_dtype'] = str(X.dtype)
        r['names_outside_allowlist'] = bad
        r['forbidden_variable_hits'] = hits
        r['names_equal_receipt_order'] = names == names_ref
        r['base_variables'] = sorted({base_var(n) for n in names})
        rz = np.load(EXP / role / 'raw_inputs.npz', allow_pickle=False)
        raw_cols = sorted(set(rz.files) - {'row_id', 'unit'})
        r['raw_inputs_columns'] = raw_cols
        r['raw_inputs_forbidden_hits'] = sorted(set(raw_cols) & set(forbidden))
        r['raw_rows_aligned_with_features'] = bool(np.array_equal(rz['row_id'], fz['row_id']) and np.array_equal(rz['unit'], fz['unit']))
        # reconstruct X from the six raw inputs only
        Xr = np.zeros((len(rz['row_id']), len(names_ref)), dtype=np.float32)
        for j, v in enumerate(PERMITTED_NUM):
            Xr[:, j] = ((rz[v].astype(np.float64) - stats[v][0]) / stats[v][1]).astype(np.float32)
        col = 2
        onehot_ok = True
        for v in PERMITTED_CAT:
            lv = enc['categorical'][v]
            idx = {s: i for i, s in enumerate(lv)}
            vals = rz[v]
            pos = np.array([idx.get(str(s), len(lv)) for s in vals])
            Xr[np.arange(len(vals)), col + pos] = 1.0
            blk = X[:, col:col + len(lv) + 1]
            onehot_ok &= bool(np.all(blk.sum(1) == 1.0) and np.all((blk == 0) | (blk == 1)))
            col += len(lv) + 1
        r['numeric_max_abs_dev'] = float(np.max(np.abs(X[:, :2].astype(np.float64) - Xr[:, :2].astype(np.float64))))
        r['numeric_exact_float32_equal'] = bool(np.array_equal(X[:, :2], Xr[:, :2]))
        r['onehot_blocks_valid'] = onehot_ok
        r['onehot_cells_mismatch'] = int(np.sum(X[:, 2:] != Xr[:, 2:]))
        r['X_reproduced_from_permitted_raw_inputs'] = bool(r['onehot_cells_mismatch'] == 0 and r['numeric_max_abs_dev'] <= 1e-6)
        per_role[role] = r
        if bad or hits or r['raw_inputs_forbidden_hits'] or not r['names_equal_receipt_order'] or X.shape[1] != xrec['input_width'] \
                or raw_cols != sorted(PERMITTED_NUM + PERMITTED_CAT) or not r['raw_rows_aligned_with_features'] \
                or not r['X_reproduced_from_permitted_raw_inputs'] or not onehot_ok:
            fails.append(f'{role}: input exclusion/width/reconstruction problem')
    ev['per_role'] = per_role

    # model-side widths: tree teachers and U0 (both seeds)
    models = {}
    for s in (0, 1):
        d = PRIV / 'runs' / f'STAGE_A_{s}'
        seedj = json.loads((d / 'STAGE_A_SEED.json').read_text())
        m = {'STAGE_A_SEED.input_width': seedj.get('input_width')}
        for task in ('income', 'occupation'):
            t = json.loads((d / f'TREE_TEACHER_{task}.json').read_text())
            sf = np.asarray(t['split_features'], dtype=object)
            flat = [int(v) for v in np.concatenate([np.ravel(np.asarray(a)) for a in t['split_features']])] if len(sf) else []
            m[f'teacher_{task}'] = {'n_features': t['n_features'], 'feature_names_equal_receipt': list(t['feature_names']) == names_ref,
                                    'max_split_feature_index': max(flat) if flat else None}
        try:
            import torch
            sd = torch.load(d / 'U0_MODEL.pt', map_location='cpu', weights_only=True)
            widths = {}
            def walk(o, p=''):
                if isinstance(o, dict):
                    for k, v in o.items():
                        walk(v, f'{p}/{k}')
                elif hasattr(o, 'shape') and len(o.shape) == 2:
                    widths[p] = list(o.shape)
            walk(sd)
            first = {k: v for k, v in widths.items() if v[1] not in (64, 16)}
            m['U0_first_layer_weight_shapes'] = first
            m['U0_input_width_ok'] = bool(first) and all(v[1] == xrec['input_width'] for v in first.values())
        except Exception as e:  # report, do not hide
            m['U0_error'] = repr(e)[:300]
            m['U0_input_width_ok'] = False
        models[f'seed{s}'] = m
        ok = (m['STAGE_A_SEED.input_width'] == xrec['input_width'] and m['U0_input_width_ok']
              and all(m[f'teacher_{t}']['n_features'] == xrec['input_width'] and m[f'teacher_{t}']['feature_names_equal_receipt']
                      and (m[f'teacher_{t}']['max_split_feature_index'] or 0) < xrec['input_width'] for t in ('income', 'occupation')))
        if not ok:
            fails.append(f'seed {s}: model input width / names inconsistent')
    ev['model_widths'] = models
    ev['forbidden_list_checked'] = len(forbidden)

    out = {'id': 3, 'name': 'INPUT EXCLUSIONS', 'status': 'PASS' if not fails else 'FAIL', 'fail_reasons': fails,
           'evidence': ev, 'notes': 'allow-list (not only deny-list) test; X reconstructed from the six permitted raw inputs '
                                    'with MODEL_FIT z-scores (ddof=0, float64->float32) and registered one-hot levels; '
                                    'row_id/unit in exports are derived integer keys, not raw identifiers or model inputs'}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({'status': out['status'], 'fails': fails,
                      'per_role': {k: {kk: v[kk] for kk in ('n_features', 'forbidden_variable_hits', 'names_outside_allowlist',
                                                            'numeric_max_abs_dev', 'onehot_cells_mismatch', 'raw_inputs_columns')}
                                   for k, v in per_role.items()},
                      'models': models, 'levels': lvl_check}, indent=1))


if __name__ == '__main__':
    main()
