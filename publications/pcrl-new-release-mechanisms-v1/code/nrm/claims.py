"""Claims assembly of the PCRL new-release-mechanisms campaign (coordinator). Label-free: it reads only saved evaluation records
(runs/<FAMILY>/<config>/EVAL_MS*.json, runs/REFERENCES/EVAL_MS*.json), saved reader-grid outputs (runs/GRID/<bank>/S<v>/GRID.json)
and the committed plan, and applies the statistical reviewer's frozen rules (nrm.rules, NRM1-rules-v1) unchanged.

    python3 -m nrm.claims run --plan results/<campaign>/claims/PLAN_<id>.json --bank <bank_id> --out results/<campaign>/claims/<id>

PLAN (committed before the grid outputs it governs are read):
  {"plan_id", "label_sentences": {...}, "references_eval": ["runs/REFERENCES/EVAL_MS__<tag>.json"],
   "configs": {"<member>": {"family": F, "config": C, "kind": K, "twin": member|null, "alternatives": [member, ...],
                            "paths": {"<path id>": [member, ...]}, "replicates": [member, ...], "outputs_jointly_designed": bool,
                            "novelty": label, "label_sentence": key|null}}}
Members are the bank's member names (family members by config id; Monte Carlo replicate members '<member>__d2', '__d4').
G2 is computed once per declared alternative and holds only if it holds for all (DECLARATIONS_V1 D1). G3 is
nrm.rules.frontier_comparison against EACH declared path; DOMINATES is required against every path (D2); the record passed to
nrm.rules.claims is the first non-dominating path record, else the family path's. Randomized members with replicate members get
per-cell Monte Carlo cells through nrm.rules.mc_cell (draw offset paired with offset when both sides have replicates).
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

from . import common as C
from . import rules as R

REF_SPEC = {'NEURAL_RAW': 'REF:NEURAL_RAW', 'H_LJ': 'REF:H_LJ', 'B2TWIN': 'REF:B2TWIN', 'RETIRED_B3CTL': 'REF:RETIRED_B3CTL',
            'SUCC': 'REF:SUCC', 'TWJ': 'REF:TWJ', 'W': 'W', 'TR100': 'REF:H3+TASK_REFIT', 'TT100': 'REF:H3+T_TREE'}
SECONDARY_MAP = {'income|acc_loss': 'income|acc_loss', 'income|brier': 'income|brier', 'income|nll': 'income|nll',
                 'occupation|acc_loss': 'occupation|acc_loss', 'occupation|brier': 'occupation|brier',
                 'occupation|nll': 'occupation|nll', 'occupation|men_brier': 'occupation|men_brier',
                 'occupation|women_brier': 'occupation|women_brier'}


def load_grid(bank):
    """Merge the per-version grid runs: {'paired': {cand: {comp: {view: entry}}}, 'cells': {...}, 'members': {v: {...}},
    'extras': {cell: {member: {view: extras}}}}."""
    out = {'paired': {}, 'cells': {}, 'members': {}, 'seeds': {}}
    for v in C.VERSIONS:
        p = C.RUNS / 'GRID' / bank / f'S{v}' / 'GRID.json'
        g = json.loads(p.read_text())
        out['members'][v] = g.get('members') or {}
        for cell, cres in (g.get('cells') or {}).items():
            out['cells'][cell] = cres
        for cand, comps in (g.get('paired') or {}).items():
            for comp, views in comps.items():
                for view, e in views.items():
                    d = out['paired'].setdefault(cand, {}).setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                    d['cells'].update(e['cells']); d['cell_se'].update(e['cell_se'])
    return out


def grid_entry(grid, cand, comp, reps):
    """Grid entry of cand vs comp ({view: {'cells', 'cell_se'[, 'cell_se_mc']}}); Monte Carlo cells when replicate members exist."""
    rc, rp = reps.get(cand, []), reps.get(comp, [])
    base = grid['paired'].get(cand, {}).get(comp)
    if base is None:
        return None
    if not rc and not rp:
        return base
    out = {}
    for view, e in base.items():
        cells, ses, mcs = {}, {}, {}
        for cell in e['cells']:
            vals, ss = [e['cells'][cell]], [e['cell_se'][cell]]
            for i, a in enumerate(rc or [None] * len(rp)):
                c2 = a or cand
                k2 = rp[i] if i < len(rp) else comp
                x = grid['paired'].get(c2, {}).get(k2, {}).get(view)
                if x is None or cell not in x['cells']:
                    raise KeyError(f'replicate pairing {c2} vs {k2} missing for {view} {cell}')
                vals.append(x['cells'][cell]); ss.append(x['cell_se'][cell])
            m = R.mc_cell(vals, ss)
            cells[cell], ses[cell], mcs[cell] = m['value'], m['se'], m['se_mc']
        out[view] = {'cells': cells, 'cell_se': ses, 'cell_se_mc': mcs, 'mc_draws': len(vals)}
    return out


def eval_records(family, config):
    """All EVAL_MS*.json of a configuration, merged: utility and wires from the most recent record that carries the selection wire
    hashes (records written before the ENG1-05 fix lack them; a tagged re-evaluation supplies them), paired costs from all."""
    d = C.RUNS / family / config
    recs = [(p, json.loads(p.read_text())) for p in sorted(d.glob('EVAL_MS*.json'))]
    withw = [(p, r) for p, r in recs if all('wire_sha256_eval' in (r['versions'].get(str(v)) or {}) for v in C.VERSIONS)]
    if not withw:
        raise KeyError(f'{family}/{config}: no evaluation record carries wire_sha256_eval (re-evaluate with a tag)')
    base = max(withw, key=lambda pr: pr[1].get('utc', ''))[1]
    costs = {v: {} for v in C.VERSIONS}
    for _, r in sorted(recs, key=lambda pr: pr[1].get('utc', '')):       # later records win (ENG2-03)
        for v in C.VERSIONS:
            for spec, c in ((r['versions'].get(str(v)) or {}).get('paired_costs') or {}).items():
                costs[v][spec] = c['costs']
    return base, costs


def ref_wires(paths):
    out = {}
    for p in paths:
        r = json.loads((C.ROOT / p).read_text())
        for name, spec in REF_SPEC.items():
            if spec in r['references']:
                out[name] = {v: r['references'][spec][str(v)]['wire_sha256'] for v in C.VERSIONS}
    return out


def grid_wires(grid, member):
    return {v: (grid['members'][v].get(member) or {}).get('wire_sha256') for v in C.VERSIONS}


def distinct_inputs(bank):
    """Label-free distinct released-vector counts per (member, version) and view, from the bank's own members (A-SC1 explanations);
    randomized members: the count of the instance R1 reads (draw 0)."""
    import numpy as np
    from . import devreaders as DR
    spec = json.loads((C.RESULTS / 'grid_specs' / f'{bank}.json').read_text())
    out = {}
    for name, m in spec['members'].items():
        for v in C.VERSIONS:
            rel, _ = DR.load_member(m, v)
            inc, occ = rel['income'], rel['occupation']
            inc = inc[0] if inc.ndim == 3 else inc
            occ = occ[0] if occ.ndim == 3 else occ
            out[f'{name}|V{v}'] = {'income': int(np.unique(inc, axis=0).shape[0]), 'occupation': int(np.unique(occ, axis=0).shape[0]),
                                   'pair': int(np.unique(np.column_stack((inc, occ)), axis=0).shape[0])}
    return out


def validity(grid, minimum_set, manifest, distinct=None):
    """nrm.rules.grid_validity from the extras' score fingerprints (k0 vs k1 per recipe, view, member, version, repetition)."""
    seeds = {f'S{v}k{k}': manifest['reader_seeds'][f'V{v}k{k}'] for v in C.VERSIONS for k in (0, 1)}
    checks = []                       # (V1 also reset `distinct` here, dropping the counts the rule needs; deviation 10)
    for cell, cres in grid['cells'].items():
        if '@' in cell or not cell.endswith('k0'):
            continue
        v, rep = int(cell[1]), cell[2:4]
        other = cell[:-2] + 'k1'
        for member, ent in cres.items():
            for view, e in ent.items():
                f0 = (e.get('extras') or {}).get('score_sha256') or {}
                f1 = (((grid['cells'].get(other) or {}).get(member) or {}).get(view, {}).get('extras') or {}).get('score_sha256') or {}
                for recipe in R.SEED_DEPENDENT + R.SEED_INVARIANT:
                    if recipe in f0 and recipe in f1:
                        checks.append({'recipe': recipe, 'view': view, 'wire': member, 'v': v, 'R': rep,
                                       'equal_k0_k1': f0[recipe] == f1[recipe]})
    distinct = distinct or {}
    return R.grid_validity(seeds, checks, minimum_set, distinct), len(checks)


def run(plan_path, bank, out_dir):
    plan = json.loads(Path(plan_path).read_text())
    if not C.committed(Path(plan_path)):
        raise PermissionError('the claims plan must be committed before it governs any read of grid outputs')
    grid = load_grid(bank)
    manifest = json.loads((C.RESULTS / 'READER_SPLITS.json').read_text())
    reps = {m: c.get('replicates', []) for m, c in plan['configs'].items()}
    rwires = ref_wires(plan['references_eval'])
    dinputs = distinct_inputs(bank)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    records, ledger, rows = {}, [], []
    for entry_no, (member, cfg) in enumerate(plan['configs'].items()):
        base, costs = eval_records(cfg['family'], cfg['config'])
        evals = {v: base['versions'][str(v)]['utility'] for v in C.VERSIONS}
        comps = list(R.BENCHMARK) + [x for x in [cfg.get('twin')] + list(cfg.get('alternatives') or []) if x] + \
            [q for path in (cfg.get('paths') or {}).values() for q in path]
        gentries = {}
        for comp in dict.fromkeys(comps):
            e = grid_entry(grid, member, comp, reps)
            if e is not None:
                gentries[comp] = e
        minimum = [member] + [x for x in [cfg.get('twin')] + list(cfg.get('alternatives') or []) if x]
        gv, nchecks = validity(grid, minimum, manifest, dinputs)
        # G3: one frontier record per declared path; DOMINATES against every path required
        fr_all = {}
        for pid, path in (cfg.get('paths') or {}).items():
            pts = []
            for q in path:
                pa = [grid['cells'][c][q]['pair']['score_auc'] for c in R.REGISTERED if q in grid['cells'].get(c, {})]
                pts.append({'id': q, 'pair_auc': sum(pa) / len(pa) if pa else None})
            spec_of = {q: f"{plan['configs'][q]['family']}/{plan['configs'][q]['config']}" if q in plan['configs'] else q for q in path}
            # every declared path member is passed: a missing paired cost makes the record INCOMPLETE (ENG2-01), never a
            # label computed on a shorter path
            point = {'id': member, 'costs_vs': {q: {v: costs[v][spec_of[q]] for v in C.VERSIONS if costs[v].get(spec_of[q])}
                                                 for q in path},
                     'grid_vs': {q: gentries.get(q) for q in path if gentries.get(q)}}
            fr_all[pid] = R.frontier_comparison(point, pts)
        fr_pass = all(f.get('label') == 'DOMINATES_CONTROL_FRONTIER' for f in fr_all.values()) and bool(fr_all)
        fr_main = (next((f for f in fr_all.values() if f.get('label') != 'DOMINATES_CONTROL_FRONTIER'), None)
                   or (next(iter(fr_all.values())) if fr_all else None))
        joint = {'outputs_changed': [t for t in C.TASKS if cfg.get('twin') and any(
                     (base['versions'][str(v)]['wire_sha256_eval'][t] !=
                      eval_records(plan['configs'][cfg['twin']]['family'], plan['configs'][cfg['twin']]['config'])[0]
                      ['versions'][str(v)]['wire_sha256_eval'][t]) for v in C.VERSIONS)],
                 'jointly_designed': bool(cfg.get('outputs_jointly_designed'))} if cfg.get('twin') in plan['configs'] else None
        per_alt = {}
        for alt in (cfg.get('alternatives') or [None]):
            rec = {'evals': evals, 'grid': gentries, 'matched': {'twin': cfg.get('twin'), 'alternative': alt},
                   'frontier': fr_main, 'joint': joint, 'novelty': cfg.get('novelty', 'NOT_CLAIMED'), 'grid_valid': gv['effective']}
            per_alt[alt or 'NONE'] = R.claims(rec)
        cl = next(iter(per_alt.values()))
        g2 = ('INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS'
              if per_alt and all(x['G2']['label'] == 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS' for x in per_alt.values())
              else next((x['G2']['label'] for x in per_alt.values() if x['G2']['label'] != 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS'),
                        'NO_MATCHED_CONTROLS'))
        summary = {k: cl[k]['label'] for k in ('G1', 'G3', 'G4', 'G5', 'G6', 'G7')}
        summary['G2'] = g2
        summary['G3_all_paths'] = {pid: f.get('label') for pid, f in fr_all.items()}
        if not fr_pass and summary['G3'] == 'STABLE_FRONTIER_IMPROVEMENT':
            summary['G3'] = 'NOT_DOMINATING_EVERY_DECLARED_PATH'
        records[member] = {'config': cfg, 'summary': summary, 'claims_per_alternative': per_alt, 'frontier_per_path': fr_all,
                           'grid_validity': gv, 'grid_validity_checks': nchecks, 'reserve': cl.get('reserve'),
                           'label_sentence': (plan.get('label_sentences') or {}).get(cfg.get('label_sentence'))}
        nfit = {}
        for v in C.VERSIONS:
            rj = json.loads((C.RUNS / cfg['family'] / cfg['config'] / f'V{v}' / 'RELEASE.json').read_text())
            nfit[v] = int(rj.get('n_model_fits', 0)) + int(rj.get('n_calibrator_fits', 0)) + int(rj.get('n_critic_fits', 0))
        lrow = {'config_id': member, 'family': cfg['family'], 'kind': cfg.get('kind'), 'entry': entry_no, 'n_fitted_components': nfit,
                'wire_sha256_eval': {v: base['versions'][str(v)]['wire_sha256_eval'] for v in C.VERSIONS},
                'wire_sha256_grid': grid_wires(grid, member), 'evals': evals, 'grid': gentries,
                'matched': {'twin': cfg.get('twin'), 'alternative': (cfg.get('alternatives') or [None])[0]},
                'frontier': fr_main, 'joint': joint, 'novelty': cfg.get('novelty', 'NOT_CLAIMED'), 'grid_valid': gv['effective']}
        if len(cfg.get('alternatives') or []) > 1 and g2 != 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS':
            # conjunction failed -> no tier through G2; the selection reads the first failing alternative (ENG2-02)
            failing = next(a for a, x in per_alt.items() if x['G2']['label'] != 'INCREMENTAL_REDUCTION_BEYOND_MATCHED_CONTROLS')
            lrow['matched'] = {'twin': cfg.get('twin'), 'alternative': failing}
        ledger.append(lrow)
        rows.append({'member': member, 'kind': cfg.get('kind'), **summary, 'G3_all_paths': json.dumps(summary['G3_all_paths']),
                     'grid_valid': gv['effective'], 'reserve_qualified': (cl.get('reserve') or {}).get('qualified')})
    sel = R.select_candidate(ledger, rwires) if all(r in rwires for r in R.BENCHMARK) else {
        'outcome': 'NOT_COMPUTED_REFERENCE_WIRES_MISSING', 'missing': [r for r in R.BENCHMARK if r not in rwires]}
    res = {'schema': 'NRM1-claims-v1', 'utc': C.now(), 'plan': str(Path(plan_path)), 'plan_sha256': C.sha(Path(plan_path)),
           'bank': bank, 'records': records, 'selection': sel, 'rules_sha256': R.load_rules()['_sha256'],
           'note': 'development evidence on adaptively used MODEL_SELECT; frozen NRM1 rules applied unchanged'}
    C.write_json_x(out_dir / 'CLAIMS.json', res)
    keys = sorted({k for r in rows for k in r})
    with open(out_dir / 'CLAIMS_SUMMARY.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(json.dumps({'members': len(records), 'selection': sel.get('outcome'), 'selected': sel.get('selected'),
                      'summary': {m: r['summary'] for m, r in records.items()}}, indent=1, default=str))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--plan', required=True); r.add_argument('--bank', required=True)
    r.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    run(a.plan, a.bank, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
