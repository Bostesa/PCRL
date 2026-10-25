"""Machine-readable frontier and endpoint tables of the direct-recovery frontier campaign (DRF1; coordinator; label-free: saved
records only).

    python3 -m drf.frontier write

FRONTIER.json / FRONTIER.csv: one row per (operating point, version): usefulness on MODEL_SELECT (G1 point pass, worst value / cap,
men's and women's occupation guard, total NLL excess vs U0) and recovery for the declared readers on the registered DRF1 cells
(registered-winner SCORE AUC of the pair and each view, mean over the 4 cells of the version), beside the MODEL_FIT diagnostics of
the finite-family points (in-sample exact optimal-reader AUC under the estimated and the empirical law; nested held-out LOOKUP AUC
and nested spends). The SEX-blind coarsening path appears with its MODEL_FIT values for every step. ENDPOINTS.json: per reading,
the decision labels, the three outcomes, FC, TWIN, the failure stage and the decisive comparisons. Development evidence on reused
people; every number is conditional on the declared readers and splits.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import common as C
from nrm import rules as NR

REFS = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100', 'U0', 'Ucal')
REF_SPEC = {'TR100': 'REF:H3+TASK_REFIT', 'TT100': 'REF:H3+T_TREE', 'W': 'W'}


def _eval(spec):
    R = C.RUNS
    if spec.startswith('REF:') or spec == 'W':
        refs = json.loads((R / 'REFERENCES' / 'EVAL_MS.json').read_text())['references']
        return refs.get(spec)
    if spec.startswith('SRCFRC:'):
        p = R / 'SOURCE_FRC' / 'FA' / spec.split('/')[-1] / 'EVAL_MS.json'
    else:
        fam, cfg = spec.split('/')
        p = R / fam / cfg / 'EVAL_MS.json'
    return json.loads(p.read_text())['versions'] if p.is_file() else None


def _util(ev, v):
    if not ev or str(v) not in ev:
        return None
    e = ev[str(v)]
    u, pt = e['utility'], e['point']
    tot = sum(float(c['value']) for c in u['clauses'] if c['reference'] == 'U0' and c['metric'] == 'nll_excess')
    g = {x['group']: x['value'] for x in u['guards']}
    return {'g1_point_pass_this_version': bool(pt['point_pass_12_plus_2']), 'max_value_over_cap': pt['max_value_over_cap'],
            'failed': pt['failed'], 'men_guard': g.get('sex0'), 'women_guard': g.get('sex1'), 'total_nll_excess_vs_U0': tot}


def _grid(bank):
    out = {}
    for v in C.VERSIONS:
        p = C.RUNS / 'GRID' / bank / f'S{v}' / 'GRID.json'
        if not p.is_file():
            continue
        g = json.loads(p.read_text())
        for cell, cres in g['cells'].items():
            if '@' in cell:
                continue
            vv, rep, k = NR.parse_cell(cell)
            if cell not in NR.REGISTERED:
                continue
            for m, ent in cres.items():
                for view in ('pair', 'income', 'occupation'):
                    out.setdefault((m, vv), {}).setdefault(view, []).append(ent[view]['score_auc'])
    return {key: {view: float(np.mean(x)) for view, x in d.items()} for key, d in out.items()}


def _fit_rows(tag, sfx):
    rows = []
    for v in C.VERSIONS:
        p = C.RESULTS / 'dr' / f'FIT_{tag}V{v}.json'
        if not p.is_file():
            continue
        f = json.loads(p.read_text())
        for cid, ins in f['insample'].items():
            ne = f['nested']['entries'].get(cid) or {}
            rows.append({'point': cid, 'version': v, 'source': 'fit', 'revision': f.get('revision', 'v1'),
                         'mf_insample_estimated_auc': {k: ins['estimated'][k] for k in ('auc_pair', 'auc_inc', 'auc_occ')},
                         'mf_insample_empirical_auc': {k: ins['empirical'][k] for k in ('auc_pair', 'auc_inc', 'auc_occ')},
                         'mf_insample_spend': ins['estimated']['spend'], 'binding': ins['estimated'].get('binding'),
                         'mf_nested_lookup_auc': ne.get('lookup_auc'),
                         'mf_nested_spend': {k: x[0] for k, x in (ne.get('spend_over_identity') or {}).items()}})
        for p_ in f['path_full']:
            st = str(p_['step'])
            nl = f['nested']['path']['lookup_auc'].get(st)
            ns = f['nested']['path']['spend_over_identity'].get(st)
            rows.append({'point': f'CTRL_PATH{sfx}_STEP_{p_["step"]}', 'version': v, 'source': 'fit_path', 'revision': f.get('revision', 'v1'),
                         'mf_insample_estimated_auc': {k: p_['insample']['estimated'][k] for k in ('auc_pair', 'auc_inc', 'auc_occ')},
                         'mf_insample_empirical_auc': {k: p_['insample']['empirical'][k] for k in ('auc_pair', 'auc_inc', 'auc_occ')},
                         'mf_insample_total_nll_spend': p_['spend_total_nll_insample'], 'mf_nested_lookup_auc': nl,
                         'mf_nested_spend': {k: x[0] for k, x in (ns or {}).items()}})
    return rows


def _opt_rows():
    rows = []
    for v in C.VERSIONS:
        p = C.RESULTS / 'dr' / f'OPTIMISTIC_V{v}.json'
        if not p.is_file():
            continue
        o = json.loads(p.read_text())
        for cid, e in o['entries'].items():
            ne = o['nested']['entries'].get(cid) or {}
            emp = e['insample']['empirical']
            rows.append({'point': cid, 'version': v, 'source': 'optimistic (MODEL_FIT only)', 'construction': e['construction'],
                         'mf_insample_empirical_auc': {k: emp[k] for k in ('auc_pair', 'auc_inc', 'auc_occ')},
                         'mf_insample_spend': emp['spend'], 'within_budget': emp['feasible'],
                         'mf_nested_lookup_auc': ne.get('lookup_auc'),
                         'mf_nested_spend': {k: x[0] for k, x in (ne.get('spend_over_identity') or {}).items()}})
    return rows


def write():
    grids = {}
    for bank in sorted(p.name for p in (C.RUNS / 'GRID').iterdir()) if (C.RUNS / 'GRID').is_dir() else []:
        if bank.endswith('_ROB'):
            continue
        grids[bank] = _grid(bank)
    ms_points = {'DR_AUC': 'DR/DR_AUC', 'DR_MI': 'DR/DR_MI', 'DR_CTRL_LO': 'DR/DR_CTRL_LO', 'DR_CTRL_HI': 'DR/DR_CTRL_HI',
                 'DR_AUC_R1': 'DR/DR_AUC_R1', 'DR_MI_R1': 'DR/DR_MI_R1', 'DR_CTRL_LO_R1': 'DR/DR_CTRL_LO_R1',
                 'DR_CTRL_HI_R1': 'DR/DR_CTRL_HI_R1', 'DR_AUC_R2': 'DR/DR_AUC_R2', 'DR_MI_R2': 'DR/DR_MI_R2',
                 'DR_CTRL_LO_R2': 'DR/DR_CTRL_LO_R2', 'DR_CTRL_HI_R2': 'DR/DR_CTRL_HI_R2', 'TWB_C64': 'SRCFRC:FA/TWB_C64',
                 **{r: REF_SPEC.get(r, f'REF:{r}') for r in REFS}}
    rows = []
    for name, spec in ms_points.items():
        ev = _eval(spec)
        for v in C.VERSIONS:
            u = _util(ev, v)
            rec = {}
            for bank, g in grids.items():
                if (name, v) in g:
                    rec[bank] = g[(name, v)]
            if u is None and not rec:
                continue
            rows.append({'point': name, 'version': v, 'source': 'model_select', 'usefulness': u, 'grid_registered_cells_mean_auc': rec})
    rows += _fit_rows('', '') + _fit_rows('R1_', '_R1') + _fit_rows('R2_', '_R2') + _opt_rows()
    out = {'schema': 'DRF1-frontier-v1', 'utc': C.now(), 'rows': rows,
           'note': 'MODEL_SELECT rows: usefulness under the original 28 conditions and registered-cell recovery for the declared readers; '
                   'MODEL_FIT rows: exact finite-law values of the fitted people (in-sample) and nested held-out values; optimistic rows '
                   'are MODEL_FIT-only diagnostics (OPT_S routes by the true SEX and is not deployable). Not a population guarantee.'}
    C.write_json(C.RESULTS / 'FRONTIER.json', out)
    with open(C.RESULTS / 'FRONTIER.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['point', 'version', 'source', 'g1_point_pass_this_version', 'max_value_over_cap', 'men_guard', 'total_nll_excess_vs_U0',
                    'grid_bank', 'grid_pair', 'grid_income', 'grid_occupation', 'mf_insample_estimated_pair', 'mf_insample_empirical_pair',
                    'mf_nested_lookup_pair', 'mf_nested_total_nll_spend'])
        for r in rows:
            u = r.get('usefulness') or {}
            gb = r.get('grid_registered_cells_mean_auc') or {}
            bank = next(iter(gb), '')
            g = gb.get(bank, {})
            w.writerow([r['point'], r['version'], r['source'], u.get('g1_point_pass_this_version', ''), u.get('max_value_over_cap', ''),
                        u.get('men_guard', ''), u.get('total_nll_excess_vs_U0', ''), bank, g.get('pair', ''), g.get('income', ''),
                        g.get('occupation', ''), (r.get('mf_insample_estimated_auc') or {}).get('auc_pair', ''),
                        (r.get('mf_insample_empirical_auc') or {}).get('auc_pair', ''),
                        (r.get('mf_nested_lookup_auc') or {}).get('pair', ''), (r.get('mf_nested_spend') or {}).get('total_nll', '')])
    return out


def _lab(x):
    return x.get('label') if isinstance(x, dict) else x


def endpoints():
    """ENDPOINTS.json: per reading (claims/<id>/DECISIONS.json, with D2R.json when present) the labels and decisive numbers."""
    out = {}
    for d in sorted((C.RESULTS / 'claims').glob('*/DECISIONS.json')):
        rec = json.loads(d.read_text())
        dec = rec['decisions']
        d2 = dec.get('D2') or {}
        comps = {}
        for k, c in (d2.get('comparisons') or {}).items():
            c = c or {}
            fr = c.get('frontier') or {}
            comps[k] = {'label': _lab(c), 'bracketing_control': fr.get('bracketing_control'),
                        'benefit': (c.get('benefit') or {}).get('aggregate') if isinstance(c.get('benefit'), dict) else c.get('benefit'),
                        'spend_state': c.get('spend_state'), 'passed': c.get('passed'), 'statement': c.get('statement')}
        rob = d.parent / 'D2R.json'
        out[d.parent.name] = {'plan': rec.get('plan'), 'bank': rec.get('bank'), 'reader_stage': rec.get('reader_stage', 'RUN'),
                              'D1': _lab(dec.get('D1')), 'D2': _lab(d2), 'D2_comparisons': comps, 'FC': _lab(dec.get('FC')),
                              'TWIN': _lab(dec.get('TWIN')), 'D3': _lab(dec.get('D3')),
                              'D3_passed': (dec.get('D3') or {}).get('passed'), 'D3_failing': (dec.get('D3') or {}).get('failing'),
                              'D4': _lab(dec.get('D4')), 'D5': _lab(dec.get('D5')), 'D6': _lab(dec.get('D6')),
                              'stage': _lab(dec.get('failure_stage')), 'legal_stage': (dec.get('failure_stage') or {}).get('legal_stage'),
                              'class': _lab(dec.get('outcome_class')), 'registered': dec.get('registered'), 'adaptive_tag': dec.get('adaptive_tag'),
                              'outcomes': (dec.get('outcomes') or {}).get('statement'),
                              'D2R': (json.loads(rob.read_text())['d2r'].get('label') if rob.is_file() else 'NOT_RUN'),
                              'decisions_sha256': C.sha(d)}
    C.write_json(C.RESULTS / 'ENDPOINTS.json', {'schema': 'DRF1-endpoints-v1', 'utc': C.now(), 'readings': out,
                                                'frontier': 'FRONTIER.json / FRONTIER.csv'})
    return out


def main(argv=None):
    a = (argv or sys.argv[1:])
    if not a or a[0] not in ('write', 'endpoints'):
        raise SystemExit(__doc__)
    if a[0] == 'endpoints':
        print(json.dumps(endpoints(), indent=1, default=str)[:4000])
        return 0
    out = write()
    print(json.dumps({'rows': len(out['rows'])}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
