"""Tables of the PCRL strong-anchor release campaign, generated from the saved records only (label-free; coordinator).

    python3 -m sar.report tables [--main SAR1_MAIN] [--extra-bank SAR1_R1 ...] [--rob SAR1_ROB]

Writes results/<campaign>/TABLES.md and TABLES.json: utility per release (original point status, worst value / cap, total NLL excess
vs U0, two-SE reserve), registered-cell mean SCORE AUC of the registered winner per view, pair comparisons (aggregate benefit, se_agg,
screen labels, recipient gates; nrm.rules.comparison), the decisions of every reading (claims/*/DECISIONS.json), the data-rich reader
benefits (rich/RICH_*.json) and the robustness labels (claims/*/D2R.json). Development evidence on adaptively used MODEL_SELECT
records, for the declared readers only.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import common as C
from . import claims as CL
from nrm import rules as NR

REFS = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100', 'U0', 'Ucal')


TWIN_SPEC = 'CTL/CTL_TWIN'


def _eval(spec):
    if spec.startswith('REF:'):
        refs = json.loads((C.RUNS / 'REFERENCES' / 'EVAL_MS.json').read_text())['references']
        return refs.get(spec[4:]) or refs.get('REF:' + spec[4:])
    if spec.startswith('DDR:'):
        fam, cfg = spec[4:].split('/')
        p = C.RUNS / 'SOURCE_DDR' / fam / cfg / 'EVAL_MS.json'
        return json.loads(p.read_text())['versions'] if p.is_file() else None
    if spec.startswith('ROLE:'):
        p = C.RUNS / 'ROLES' / spec[5:] / 'EVAL_MS.json'
        return json.loads(p.read_text())['versions'] if p.is_file() else None
    fam, cfg = spec.split('/')
    p = C.RUNS / fam / cfg / 'EVAL_MS.json'
    return json.loads(p.read_text())['versions'] if p.is_file() else None


def _util_cells(ev):
    out = []
    for v in ('0', '1'):
        e = ev.get(v) if ev else None
        if not e or e.get('missing'):
            out.append('-'); continue
        u = e['utility']
        tot = sum(float(c['value']) for c in u['clauses'] if c['reference'].startswith('U0') and c['metric'].startswith('nll'))
        res = (e.get('reserve_without_mc') or e.get('reserve') or {}).get('min_whole_ratio')
        out.append(f"{'pass' if e['point']['point_pass_12_plus_2'] else 'FAIL'} {e['point']['max_value_over_cap']:.3f} / {tot:+.4f} / "
                   f"{res:.2f}" if res is not None else '')
    return out


def mean_auc(grid, member, view, cells=NR.REGISTERED):
    xs = [grid['cells'][c][member][view]['score_auc'] for c in cells if c in grid['cells'] and member in grid['cells'][c]]
    return float(np.mean(xs)) if len(xs) == len(cells) else None


def comparison_row(grid, cand, comp):
    e = grid['paired'].get(cand, {}).get(comp)
    if not e:
        return None
    cmp = NR.comparison(e)
    agg = cmp['benefit'].get('aggregate') or {}
    vm = {}
    for v in C.VERSIONS:
        xs = [e['pair']['cells'][c] for c in NR.REGISTERED if c.startswith(f'S{v}')]
        vm[v] = float(np.mean(xs)) if xs else None
    return {'benefit': agg.get('mean'), 'se_agg': agg.get('se_agg'), 'v0': vm[0], 'v1': vm[1], 'screen': cmp['benefit']['label'],
            'income': cmp['recipient']['income']['label'], 'occupation': cmp['recipient']['occupation']['label'],
            'recipient_gate_pass': cmp['recipient_gate_pass']}


def _merge(grid, other):
    """Member-level merge of two loaded grids (sar.claims.merge_grids replaces a whole cell, which is harmless for its own use
    with one bank's minimum set but would drop the first bank's members of a shared cell here). Identical cache keys give identical
    cell entries, so the order does not matter."""
    for cell, mem in other['cells'].items():
        grid['cells'].setdefault(cell, {}).update(mem)
    for v, m in other['members'].items():
        grid['members'].setdefault(v, {}).update(m)
    for cand, comps in other['paired'].items():
        for comp, views in comps.items():
            for view, e in views.items():
                d = grid['paired'].setdefault(cand, {}).setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                d['cells'].update(e['cells'])
                d['cell_se'].update(e['cell_se'])
    return grid


def _spend_vs_twin(member):
    """MODEL_SELECT paired total-NLL cost of a family release against the shared twin per version (income + occupation NLL)."""
    if member == TWIN_SPEC:
        return [0.0, 0.0]
    if member.startswith('ROLE:'):
        p = C.RUNS / 'ROLES' / member[5:] / 'EVAL_MS.json'
    else:
        fam, cfg = member.split('/')
        p = C.RUNS / fam / cfg / 'EVAL_MS.json'
    if not p.is_file():
        return None
    r = json.loads(p.read_text())['versions']
    out = []
    for v in ('0', '1'):
        c = ((r.get(v) or {}).get('paired_costs') or {}).get(TWIN_SPEC)
        out.append(float(c['costs']['income|nll']['value'] + c['costs']['occupation|nll']['value']) if c else None)
    return out


def tables(main='SAR1_MAIN', extra=(), rob=None):
    spec = json.loads((C.RESULTS / 'grid_specs' / f'{main}.json').read_text())
    grid = CL.load_grid(main)
    for b in extra:
        grid = _merge(grid, CL.load_grid(b))
        bs = json.loads((C.RESULTS / 'grid_specs' / f'{b}.json').read_text())
        spec['members'].update(bs['members'])
        spec['candidates'] = list(dict.fromkeys(list(spec.get('candidates', [])) + list(bs.get('candidates', []))))
    rec = {'utc': C.now(), 'bank': main, 'extra_banks': list(extra), 'releases': {}, 'comparisons': {}, 'decisions': {}, 'rich': {},
           'd2r': {}}
    lines = ['# Tables (campaign `pcrl_strong_anchor_release_v1`)', '',
             'Generated by `python3 -m sar.report tables` from the saved records. Development evidence on adaptively used MODEL_SELECT '
             'records, for the declared readers only; lower recovery for these readers is not protection against every reader. Versions '
             'share every training row and the MODEL_SELECT people (procedure repeats, not replications).', '',
             '## 1. Utility and recovery per release', '',
             'Per version: original point status (12 clauses + 2 guards), worst value over clauses and guards / cap, total NLL excess vs U0 (income + occupation, '
             'nats), whole-release two-SE reserve ratio (>= 2 meets it; reported, never gating). AUC = registered-cell mean SCORE AUC (8 '
             'cells) of the registered winner, per view.', '',
             '| Release | Version 0 | Version 1 | Pair AUC | Income AUC | Occupation AUC |', '|---|---|---|---|---|---|']
    for name, m in spec['members'].items():
        if m.get('type') in ('sar',):
            evs = f"{m['family']}/{m['config']}"
        elif m.get('type') == 'ddr':
            evs = f"DDR:{m['family']}/{m['config']}"
        elif m.get('type') == 'arrays' and m.get('role'):
            evs = f"ROLE:{m['role']}"
        elif name in ('W',):
            evs = 'REF:W'
        elif m.get('type') == 'reference':
            evs = 'REF:' + m['name']
        else:
            evs = None
        ev = _eval(evs) if evs else None
        u = _util_cells(ev) if ev else ['-', '-']
        a = {vw: mean_auc(grid, name, vw) for vw in ('pair', 'income', 'occupation')}
        rec['releases'][name] = {'utility': u, 'auc': a}
        f = lambda x: f'{x:.4f}' if x is not None else '-'
        lines.append(f"| {name} | {u[0]} | {u[1]} | {f(a['pair'])} | {f(a['income'])} | {f(a['occupation'])} |")
    headline = set()
    for pp in sorted((C.RESULTS / 'claims').glob('PLAN_*.json')):
        c0 = json.loads(pp.read_text()).get('cand')
        if c0:
            headline.add(c0)
    lines += ['', '## 2. Pair comparisons (benefit = comparator AUC - candidate AUC; positive favours the candidate)', '',
              '| Candidate | Comparator | Benefit | se_agg | V0 | V1 | Screen | Income recipient | Occupation recipient |',
              '|---|---|---|---|---|---|---|---|---|']
    for cand in spec.get('candidates', []):
        for comp in spec['members']:
            if comp == cand:
                continue
            r = comparison_row(grid, cand, comp)
            if r is None:
                continue
            rec['comparisons'][f'{cand}__{comp}'] = r
            if cand in headline or comp in ('CTL_TWIN',):
                g = lambda x: f'{x:+.4f}' if x is not None else '-'
                lines.append(f"| {cand} | {comp} | {g(r['benefit'])} | {r['se_agg']:.4f} | {g(r['v0'])} | {g(r['v1'])} | {r['screen']} | "
                             f"{r['income']} | {r['occupation']} |")
    lines += ['', '## 3. Decisions (sar.rules, SAR1-rules-v1; claims/*/DECISIONS.json)', '']
    for d in sorted((C.RESULTS / 'claims').glob('*/DECISIONS.json')):
        dec = json.loads(d.read_text())
        rec['decisions'][d.parent.name] = dec['decisions']
        labs = {k: (v.get('label') if isinstance(v, dict) else v) for k, v in dec['decisions'].items() if k.startswith('D') or k == 'outcome_class'}
        lines.append(f"- **{d.parent.name}** (grid valid: {dec['grid_validity']['label']}): " + '; '.join(f'{k} `{v}`' for k, v in labs.items()))
    lines += ['', '## 4. Data-rich readers (descriptive; rich/RICH_*.json)', '']
    for p in sorted((C.RESULTS / 'rich').glob('RICH_*.json')):
        if p.name.startswith('RICH_SPEC'):
            continue
        r = json.loads(p.read_text())
        rec['rich'][p.stem] = r['benefits']
        for k, x in r['benefits'].items():
            vm = {kk: (vv.get('version_means') if isinstance(vv, dict) else None) for kk, vv in x.items()}
            lines.append(f'- {k}: ' + '; '.join(f'{kk} ' + ', '.join(f'{a} {b:+.4f}' for a, b in (vm[kk] or {}).items()) for kk in vm))
    lines += ['', '## 5. Robustness to classifier refitting (claims/*/D2R.json; descriptive)', '']
    for d in sorted((C.RESULTS / 'claims').glob('*/D2R.json')):
        r = json.loads(d.read_text())
        rec['d2r'][d.parent.name] = r
        lines.append(f"- {d.parent.name}: D2R `{r['d2r'].get('label')}`; refit `{r['refit_effectiveness']['label']}`; per comparator "
                     + ', '.join(f"{k} {v.get('label')}" for k, v in (r['d2r'].get('per_comparator') or {}).items()))
    lines += ['', '## 6. Strongest feasible nonzero release per family (claims/STRONGEST.json; RULES.md 8; descriptive)', '']
    sp = C.RESULTS / 'claims' / 'STRONGEST.json'
    if sp.is_file():
        st = json.loads(sp.read_text())['families']
        rec['strongest'] = st
        for fam, s_ in st.items():
            rows = ', '.join(f"{r['config_id']} {r.get('status')}" + (f" {r['x']:+.4f}" if r.get('x') is not None else '')
                             for r in s_.get('records', []))
            x = s_.get('x')
            lines.append(f"- {fam}: `{s_.get('label')}` {s_.get('selected')}" + (f' (pair benefit vs TWIN {x:+.4f})' if x is not None else '')
                         + f'; candidates: {rows}')
    lines += ['', '## 7. Beside comparisons (claims/BESIDE_*.json; registered cells; descriptive, never a decision input)', '']
    for p in sorted((C.RESULTS / 'claims').glob('BESIDE_*.json')):
        r = json.loads(p.read_text())
        rec.setdefault('beside', {})[p.stem] = r
        for key, x in r['comparisons'].items():
            if x.get('missing'):
                lines.append(f'- {key}: missing')
                continue
            vm = x.get('version_means') or {}
            f4 = lambda y: f'{y:+.4f}' if isinstance(y, (int, float)) else '-'
            lines.append(f"- {x.get('cand', r.get('cand'))} vs {x.get('comparator', key)} ({r['bank']}): {x['mean']:+.4f} (se_agg "
                         f"{x['se_agg']:.4f}; V0 {f4(vm.get('0'))}, V1 {f4(vm.get('1'))}) `{x['label']}`; recipient gate "
                         f"{x.get('recipient_gate_pass')}")
    lines += ['', '## 8. MODEL_SELECT spend: paired total-NLL cost against CTL_TWIN (income + occupation, nats; V0 / V1)', '']
    for name, m in spec['members'].items():
        if m.get('type') in ('sar', 'arrays') and (m.get('type') == 'sar' or m.get('role')):
            sv = _spend_vs_twin(f"{m['family']}/{m['config']}" if m.get('type') == 'sar' else f"ROLE:{m['role']}")
            if sv:
                rec.setdefault('spend_vs_twin', {})[name] = sv
                lines.append(f"- {name}: " + ' / '.join('-' if x is None else f'{x:+.5f}' for x in sv))
    cs = C.RESULTS / 'claims' / 'CAMPAIGN_SUMMARY.json'
    if cs.is_file():
        summ = json.loads(cs.read_text())
        rec['campaign_summary'] = summ
        lines += ['', '## 9. Campaign summary (sar.claims summary; RULES.md 7)', '', '```',
                  json.dumps(summ.get('summary'), indent=1, default=str)[:6000], '```']
    (C.RESULTS / 'TABLES.md').write_text('\n'.join(lines) + '\n')
    C.write_json(C.RESULTS / 'TABLES.json', rec)
    print(f'TABLES.md written: {len(rec["releases"])} releases, {len(rec["comparisons"])} comparisons, {len(rec["decisions"])} readings')
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('tables'); t.add_argument('--main', default='SAR1_MAIN'); t.add_argument('--extra-bank', action='append', default=[])
    t.add_argument('--rob', default=None)
    a = ap.parse_args(argv)
    tables(a.main, a.extra_bank, a.rob)
    return 0


if __name__ == '__main__':
    sys.exit(main())
