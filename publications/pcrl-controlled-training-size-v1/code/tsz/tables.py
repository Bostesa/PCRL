"""Readable tables from the saved size records (label-free; no label is read).

    python3 -m tsz.tables size        size/PAIRED_CONTRASTS.csv, size/CONTROL_COMPARISON.csv, size/ENCODING.csv,
                                      size/SUBSET_VARIATION.csv from size/SIZE_RESULTS.json
    python3 -m tsz.tables identity    IDENTITY_CHECKS.json: reproduction, aliases, control identity, correction-base gates, deployment,
                                      subset membership and fitting exposure from the access log
"""
from __future__ import annotations
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from . import common as C

SIZE = C.RESULTS / 'size'


def _j(p):
    return json.loads(Path(p).read_text())


def _write(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def size_tables():
    res = _j(SIZE / 'SIZE_RESULTS.json')
    rows = []
    for recipe, summ in res['size_summary'].items():
        for c, per_m in summ['contrasts'].items():
            for m, blk in per_m.items():
                for v, s in (blk.get('per_version') or {}).items():
                    vals = s.get('values') or {}
                    rows.append({'recipe': recipe, 'contrast': c, 'metric': m, 'version': v,
                                 'orientation': 'improvement (positive = larger training set better)',
                                 **{f'ordering_{o}': vals.get(o) for o in C.ORDERINGS}, 'mean': s.get('mean'), 'sd_ord': s.get('sd_ord'),
                                 'se_eval': s.get('se_eval'), 'se_comb': s.get('se_comb'), 'label': s.get('label'),
                                 'sign_reversal': s.get('sign_reversal'), 'across_versions': blk.get('across')})
    _write(SIZE / 'PAIRED_CONTRASTS.csv', rows)
    crow = []
    for ctl, d in res['control'].items():
        for f, per_m in d['level'].items():
            for m, s in per_m.items():
                for v, pv in (s.get('per_version') or {}).items():
                    crow.append({'control': ctl, 'kind': 'level (A1_PO_16 improvement over the control)', 'size_or_contrast': f,
                                 'metric': m, 'version': v, 'mean': pv.get('mean'), 'se_comb': pv.get('se_comb'),
                                 'label': pv.get('label'), 'across_versions': s.get('label')})
        for c, per_m in d['did'].items():
            for m, s in per_m.items():
                for v, pv in (s.get('per_version') or {}).items():
                    crow.append({'control': ctl, 'kind': 'difference in size gains (A1_PO_16 minus control)', 'size_or_contrast': c,
                                 'metric': m, 'version': v, 'mean': pv.get('mean'), 'se_comb': pv.get('se_comb'),
                                 'label': pv.get('label'), 'across_versions': s.get('across')})
    _write(SIZE / 'CONTROL_COMPARISON.csv', crow)
    erow = []
    for m, s in res['encoding']['summary'].items():
        vals = s.get('values') or {}
        erow.append({'metric': m, 'orientation': 'loss(FIT-only dictionary) - loss(admitted dictionary); positive = admitted better',
                     **{f'ordering_{o}': vals.get(o) for o in C.ORDERINGS}, 'mean': s.get('mean'), 'se_comb': s.get('se_comb'),
                     'label': s.get('encoding_label')})
    _write(SIZE / 'ENCODING.csv', erow)
    vrow = []
    for k, d in res['subset_to_subset_pairwise'].items():
        vrow.append({'pair': k, 'kind': 'ordering difference at a fixed size (loss orientation)',
                     **{f'{m}': d[m]['value'] for m in d}, **{f'{m}_se': d[m]['se'] for m in d}})
    for k, d in res['version_difference_same_rows'].items():
        vrow.append({'pair': f'{k}|V1-V0', 'kind': 'version difference on the same rows (procedure repeat)',
                     **{f'{m}': d[m]['value'] for m in d}, **{f'{m}_se': d[m]['se'] for m in d}})
    _write(SIZE / 'SUBSET_VARIATION.csv', vrow)
    print(json.dumps({'paired_rows': len(rows), 'control_rows': len(crow), 'encoding_rows': len(erow), 'variation_rows': len(vrow)}))


def identity():
    out = {'schema': 'TSZ1-identity-checks-v1', 'utc': C.now()}
    size = {}
    for d in sorted((C.RUNS / 'SIZE').iterdir()):
        if (d / 'FIT.json').is_file():
            r = _j(d / 'FIT.json')
            e = {'subset': r.get('subset'), 'version': r.get('version'), 'people': r.get('n_people'), 'households': r.get('n_households'),
                 'T_star': r.get('T_star'), 'rows_sha256': r.get('rows_sha256') or r.get('row_set_sha256'),
                 'verify': (_j(d / 'VERIFY.json').get('verify') if (d / 'VERIFY.json').is_file() else None)}
            if (d / 'REPRO.json').is_file():
                rp = _j(d / 'REPRO.json')
                e['reproduction'] = {'reproduced': rp.get('reproduced'), 'all_roles_bitwise': rp.get('all_roles_bitwise'),
                                     'T_star': rp.get('T_star'), 'model_pkl_equal': (rp.get('model_pkl_sha256') or {}).get('equal')}
            size[d.name] = e
    out['size_fits'] = size
    ctl = {}
    for d in sorted((C.RUNS / 'CONTROL').iterdir()):
        e = {}
        for sub, rn in (('TEACHERS', 'TEACHERS.json'), ('STAGE_B', 'STAGE_B.json')):
            if (d / sub / rn).is_file():
                r = _j(d / sub / rn)
                e[sub] = {'people': r.get('n_people') or r.get('n'), 'rows_sha256': r.get('rows_sha256'),
                          'tree_counts': {t: x.get('tree_count') for t, x in (r.get('teachers') or {}).items()} or None,
                          'epoch_order': r.get('epoch_order')}
        for p in sorted(d.glob('REPRO*.json')) + sorted(d.glob('VERIFY*.json')):
            r = _j(p)
            e[p.stem] = {k: r.get(k) for k in ('reproduced', 'verify', 'all_bitwise', 'teachers_identical', 'theta_bitwise', 'summary')
                         if k in r}
        ctl[d.name] = e
    out['controls'] = ctl
    base = {}
    for v in C.VERSIONS:
        p = C.RUNS / 'CORRECT' / f'V{v}' / 'BASE' / 'BASE.json'
        if p.is_file():
            r = _j(p)
            base[str(v)] = {'checks_pass': r.get('checks_pass'), 'checks': r.get('checks')}
    out['correction_base'] = base
    acc = C.access_events()
    fit_reads = [e for e in acc if e.get('operation') in ('fit', 'select_ncv', 'calibrate')]
    out['exposure'] = {
        'label_reads': len(acc), 'by_operation_role_subset': dict(Counter(f"{e.get('operation')}|{e.get('role')}|{e.get('subset')}"
                                                                          for e in acc)),
        'fit_reads_touching_MODEL_SELECT': [e for e in fit_reads if e.get('role') == 'MODEL_SELECT'],
        'closed_role_reads': [e for e in acc if e.get('role') in C.CLOSED_LABEL_ROLES],
        'fit_reads_outside_queue': [e for e in fit_reads if not e.get('unit_id')],
        'subset_rows_match_registered': all(
            e.get('rows_sha256') == _j(C.RESULTS / 'SUBSETS.json')['row_set_sha256'].get(e.get('subset'))
            for e in acc if e.get('role') == 'MODEL_FIT' and e.get('subset'))}
    for f in ('deploy/DEPLOYMENT_CHECKS_W100.json', 'deploy/DEPLOYMENT_CHECKS.json', 'roles/verifier/DEPLOY_W100.json'):
        p = C.RESULTS / f
        if p.is_file():
            r = _j(p)
            out.setdefault('deployment', {})[f] = {k: r.get(k) for k in ('all_bitwise', 'all_refused')} if 'all_bitwise' in r else \
                {k: r.get(k) for k in ('0', '1')}
    C.write_json(C.RESULTS / 'IDENTITY_CHECKS.json', out)
    print(json.dumps({'size_fits': len(size), 'controls': len(ctl), 'base': {v: b['checks_pass'] for v, b in base.items()},
                      'label_reads': len(acc), 'subset_rows_match_registered': out['exposure']['subset_rows_match_registered'],
                      'fit_reads_outside_queue': len(out['exposure']['fit_reads_outside_queue'])}))


if __name__ == '__main__':
    cmd = sys.argv[1:]
    if cmd == ['size']:
        size_tables()
    elif cmd == ['identity']:
        identity()
    else:
        raise SystemExit(__doc__)
