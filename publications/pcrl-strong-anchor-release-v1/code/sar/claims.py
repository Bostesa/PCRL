"""Claims assembly of the PCRL strong-anchor release campaign (`pcrl_strong_anchor_release_v1`; coordinator; label-free). It reads only
saved records (evaluation records EVAL_MS*.json, the family fit records, reader-grid outputs GRID.json, the access log for the
registration status) and a committed plan, and applies the statistical reviewer's frozen rules (sar.rules, SAR1-rules-v1) unchanged.

    python3 -m sar.claims run --plan results/<campaign>/claims/PLAN_<id>.json --bank <bank_id> --out results/<campaign>/claims/<id>
    python3 -m sar.claims d2r --plan PLAN.json --main <bank> --rob <bank> --out DIR      (D2R of the strongest release; descriptive)
    python3 -m sar.claims summary --decisions DIR1 DIR2 ... --out FILE                      (sar.rules.campaign_summary)

PLAN (committed before the reading's first MODEL_SELECT utility or reader computation; RULES.md 10, sar.plan):
  {"plan_id", "family", "card", "revision", "written_utc",
   "start": {"id": <reference name: SUCC | NEURAL_RAW | ...>, "binding": {"0": cond, "1": cond}, "wires": {...}},
   "cand": member, "strengths": [{"config_id", "strength": {"0": x, "1": y}}], "every_nonzero_strength_alias_of_start": bool,
   "cont_declared": bool, "cont": member | null, "ctrl_origin": "START" | "CONT", "alt_declared": bool,
   "paths": {"CTRL": [members], "ALT": [members] | null}, "selection_data": {"CTRL": ..., "ALT": ...},
   "start_in_sample_on_model_fit": bool, "controls_not_run": {"CONT": bool, "CTRL": bool, "ALT": bool},
   "members": {member: "<family>/<config>"} (this campaign's saved releases used by the reading), "alias_of": {member: id},
   "fineness_inputs": {...} (sar.rules.grid_fineness input), "joint": null | {...}, "novelty": null}
Grid members carry the plan's member names; references (START among them) keep their benchmark names. In the rules' record START is
keyed by its reference name in grid, costs_vs and nolookup.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import common as C
from . import rules as RU
from nrm import rules as NR

REFS = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'W', 'TR100', 'TT100')
PATH_KEYS = ('SBC', 'SPEND', 'RO')                 # the three gating control paths (PROTOCOL section 4)
JOINT_KEYS = ('separate', 'income_first', 'occupation_first')


def load_grid(bank):
    """Merge the per-version grid runs of a bank: {'paired': {cand: {comp: {view: {'cells', 'cell_se'}}}}, 'cells': {...}}."""
    out = {'paired': {}, 'cells': {}, 'members': {}}
    for v in C.VERSIONS:
        p = C.RUNS / 'GRID' / bank / f'S{v}' / 'GRID.json'
        if not p.is_file():
            continue
        g = json.loads(p.read_text())
        out['members'][v] = g.get('members') or {}
        for cell, cres in (g.get('cells') or {}).items():
            out['cells'][cell] = cres
        for cand, comps in (g.get('paired') or {}).items():
            for comp, views in comps.items():
                for view, e in views.items():
                    d = out['paired'].setdefault(cand, {}).setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                    d['cells'].update(e['cells'])
                    d['cell_se'].update(e['cell_se'])
    return out


def merge_grids(*grids):
    out = {'paired': {}, 'cells': {}, 'members': {}}
    for g in grids:
        out['cells'].update(g['cells'])
        for v, m in g['members'].items():
            out['members'].setdefault(v, {}).update(m)
        for cand, comps in g['paired'].items():
            for comp, views in comps.items():
                for view, e in views.items():
                    d = out['paired'].setdefault(cand, {}).setdefault(comp, {}).setdefault(view, {'cells': {}, 'cell_se': {}})
                    d['cells'].update(e['cells'])
                    d['cell_se'].update(e['cell_se'])
    return out


def distinct_inputs(bank):
    """Label-free distinct released-vector counts per (member, version) and view over the MODEL_SELECT people."""
    from . import devreaders as DR
    spec = json.loads((C.RESULTS / 'grid_specs' / f'{bank}.json').read_text())
    out = {}
    for name, m in spec['members'].items():
        for v in C.VERSIONS:
            rel, _ = DR.load_member(m, v)
            inc, occ = rel['income'], rel['occupation']
            if inc.ndim == 3:
                inc = inc[0]
            if occ.ndim == 3:
                occ = occ[0]
            out[f'{name}|V{v}'] = {'income': int(np.unique(inc, axis=0).shape[0]), 'occupation': int(np.unique(occ, axis=0).shape[0]),
                                   'pair': int(np.unique(np.column_stack((inc, occ)), axis=0).shape[0])}
    return out


def seed_records(ks=(0, 1), reps=('R1', 'R2')):
    """Reader seeds from READER_SPLITS.json and bootstrap seeds from the locked grid code (sar/devreaders.py constants, pinned by
    DEVGRID_CODE_LOCK.json): the inputs of sar.rules.grid_validity."""
    from . import devreaders as DR
    man = json.loads((C.RESULTS / 'READER_SPLITS.json').read_text())
    seeds = {f'S{v}k{k}': int(man['reader_seeds'][f'V{v}k{k}']) for v in C.VERSIONS for k in ks}
    boot = {f'S{v}{r}': int(DR.BOOT_SEED + 100 * v + DR.REP_INDEX[r]) for v in C.VERSIONS for r in reps}
    return seeds, boot


def seed_checks(grid, pairs=((0, 1),)):
    """Score fingerprints of every recipe per (member, view, version, repetition) for each reader-seed pair."""
    checks = []
    for cell, cres in grid['cells'].items():
        if '@' in cell:
            continue
        v, rep, k = NR.parse_cell(cell)
        for a, b in pairs:
            if k != a:
                continue
            other = cell[:-1] + str(b)
            for member, ent in cres.items():
                for view, e in ent.items():
                    f0 = (e.get('extras') or {}).get('score_sha256') or {}
                    f1 = (((grid['cells'].get(other) or {}).get(member) or {}).get(view, {}).get('extras') or {}).get('score_sha256') or {}
                    for recipe in NR.SEED_DEPENDENT + NR.SEED_INVARIANT:
                        if recipe in f0 and recipe in f1:
                            checks.append({'recipe': recipe, 'view': view, 'wire': member, 'v': v, 'R': rep, 'k_pair': [a, b],
                                           'equal_k0_k1': f0[recipe] == f1[recipe]})
    return checks


def extra_benefit(grid, cand, comp, key='nolookup', cells=NR.REGISTERED):
    """Per cell: the extra reader's (non-table winner 'nolookup' / enhanced winner 'enhanced') SCORE AUC of comp minus cand."""
    out = {}
    for cell in cells:
        try:
            a = grid['cells'][cell][cand]['pair']['extras'][key]['score_auc']
            b = grid['cells'][cell][comp]['pair']['extras'][key]['score_auc']
        except (KeyError, TypeError):
            return None
        out[cell] = float(b) - float(a)
    return {'cells': out}


def evals_of(member_spec):
    if member_spec.startswith('REF:'):                  # a frozen reference: this campaign's reference utility record
        r = json.loads((C.RUNS / 'REFERENCES' / 'EVAL_MS.json').read_text())['references'].get(member_spec)
        return None if r is None else {v: r[str(v)] for v in C.VERSIONS if str(v) in r}
    if member_spec.startswith('ROLE:'):                 # a role member (sar.roles): its record under runs/ROLES/<name>
        p = C.RUNS / 'ROLES' / member_spec[5:] / 'EVAL_MS.json'
    else:
        fam, cfg = member_spec.split('/')
        p = C.RUNS / fam / cfg / 'EVAL_MS.json'
    if not p.is_file():
        return None
    r = json.loads(p.read_text())
    return {v: r['versions'][str(v)] for v in C.VERSIONS if not r['versions'][str(v)].get('missing')}


def costs_between(cand_spec, comp_spec):
    ev = evals_of(cand_spec)
    if ev is None:
        return None
    out = {}
    for v, e in ev.items():
        c = (e.get('paired_costs') or {}).get(comp_spec)
        if c is None or c.get('missing'):
            return None
        out[v] = c['costs']
    return out


def _git(*a):
    r = subprocess.run(['git', '-C', str(C.ROOT), *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def registration_status(card):
    """card_committed_before_any_model_select_outcome: the card's first commit time against the first MODEL_SELECT read in the
    access log (RULES.md 10.1)."""
    first_ms = next((e['utc'] for e in C.access_events() if e.get('role') == 'MODEL_SELECT'), None)
    log = _git('log', '--diff-filter=A', '--format=%cI', '--', card) or ''
    first_card = log.splitlines()[-1] if log else None
    if first_card is None:
        return None, {'card_first_commit': None, 'first_model_select_read': first_ms}
    import datetime as dt
    ok = first_ms is None or dt.datetime.fromisoformat(first_card) < dt.datetime.fromisoformat(first_ms)
    return bool(ok), {'card_first_commit': first_card, 'first_model_select_read': first_ms}


def reader_budget_flags(bank, cand, refs=REFS):
    """Label-free pair-view budget contrast of CAND against each reference (sar.readerbudget.contrast); the flag the rules read."""
    from . import devreaders as DR
    from . import readerbudget as RB
    spec = json.loads((C.RESULTS / 'grid_specs' / f'{bank}.json').read_text())
    unit = C.features('MODEL_SELECT')['unit']
    flags = {}
    for v in (0,):
        rc, _ = DR.load_member(spec['members'][cand], v)
        bc = RB.release_budget(rc, unit)
        for ref in refs:
            rr, _ = DR.load_member(spec['members'][ref], v)
            con = RB.contrast(bc, RB.release_budget(rr, unit), view='pair')
            flags[ref] = con.get('flag') or con.get('label')
    return flags


def build(plan, grid, bank):
    """The SAR1 rules' record of one family reading (sar.rules docstring) from the committed plan, the saved MODEL_SELECT
    evaluation records (EVAL_MS.json, with paired costs against START = 'REF:<start>', CONT and every CTRL / ALT point) and the grid."""
    mem = plan['members']
    cand, start = plan['cand'], plan['start']['id']
    ok, reg = registration_status(plan['card'])
    rec = {'family': plan['family'], 'card': plan['card'], 'revision': int(plan.get('revision', 0)),
           'card_committed_before_any_model_select_outcome': ok, 'registration_status_detail': reg}
    ev_s = evals_of(f'REF:{start}')
    rec['start'] = {'id': start, 'evals': {v: e['utility'] for v, e in (ev_s or {}).items()},
                    'binding': {int(k): b for k, b in (plan['start'].get('binding') or {}).items()}}
    if cand is None:
        rec['cand'] = {'status': 'NO_FEASIBLE_STRENGTH'}
        return rec
    ev_c = evals_of(mem[cand])
    rec['cand'] = {'status': 'FITTED' if ev_c else 'MISSING', 'config_id': cand,
                   'evals': {v: e['utility'] for v, e in (ev_c or {}).items()},
                   'strength': (plan.get('strengths') or [{}])[0].get('strength'),
                   'every_nonzero_strength_alias_of_start': bool(plan.get('every_nonzero_strength_alias_of_start', False))}
    rec['strengths'] = [{'config_id': x['config_id'], 'strength': x['strength'],
                         'evals': {v: e['utility'] for v, e in (evals_of(mem[x['config_id']]) or {}).items()}}
                        for x in plan.get('strengths', [])]
    rec['cont_declared'] = bool(plan.get('cont_declared', False))
    cont = plan.get('cont')
    if cont:
        rec['cont'] = {'config_id': cont, 'evals': {v: e['utility'] for v, e in (evals_of(mem[cont]) or {}).items()}}
    rec['ctrl_origin'] = plan.get('ctrl_origin', 'START')
    rec['alt_declared'] = bool(plan.get('alt_declared', False))
    paths = {k: plan['paths'].get(k) for k in ('CTRL', 'ALT')}
    rec['paths'] = paths
    rec['selection_data'] = plan.get('selection_data') or {}
    rec['start_in_sample_on_model_fit'] = plan.get('start_in_sample_on_model_fit')
    rec['controls_not_run'] = plan.get('controls_not_run') or {'CONT': False, 'CTRL': False, 'ALT': False}
    rec['alias_of'] = plan.get('alias_of') or {}
    if plan.get('fineness_inputs'):
        rec['fineness'] = RU.grid_fineness(plan['fineness_inputs'], plan.get('start_in_sample_on_model_fit'))
    pc = grid['paired'].get(cand, {})
    ids = ([cont] if cont else []) + [m for k in ('CTRL', 'ALT') for m in (paths.get(k) or [])]
    joint = plan.get('joint')
    joint_ids = [joint[k] for k in JOINT_KEYS if joint and joint.get(k)]
    rec['grid'] = {k: pc[k] for k in [start] + ids + joint_ids + list(REFS) if k in pc}
    rec['costs_vs'] = {start: costs_between(mem[cand], f'REF:{start}'), **{m: costs_between(mem[cand], mem[m]) for m in ids}}
    rec['nolookup'] = {start: extra_benefit(grid, cand, start), **{m: extra_benefit(grid, cand, m) for m in ids},
                       **{r: extra_benefit(grid, cand, r) for r in REFS}}
    rec['enhanced'] = {start: extra_benefit(grid, cand, start, 'enhanced'), **{m: extra_benefit(grid, cand, m, 'enhanced') for m in ids}}
    rec['reader_budget'] = reader_budget_flags(bank, cand)
    rec['joint'] = ({**joint, 'costs_vs': {m: costs_between(mem[cand], mem[m]) for m in joint_ids if m in mem}} if joint else None)
    rec['novelty'] = plan.get('novelty')
    return rec


def run(plan_path, bank, out):
    plan = json.loads(Path(plan_path).read_text())
    if not C.committed(Path(plan_path)):
        raise PermissionError('the claims plan must be committed before the grid outputs it governs are read')
    grid = load_grid(bank)
    seeds, boot = seed_records()
    minimum = [m for m in [plan['cand'], plan['start']['id'], plan.get('cont') if plan.get('cont_declared') else None]
               + [x for k in ('CTRL', 'ALT') for x in (plan['paths'].get(k) or [])] if m]
    distinct = distinct_inputs(bank)
    checks = seed_checks(grid)
    gv = RU.grid_validity(seeds, checks, minimum, distinct, boot_seeds=boot)
    rec = build(plan, grid, bank)
    rec['grid_valid'] = gv['effective']
    dec = RU.decisions(rec)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    C.write_json_x(out / 'DECISIONS.json', {'schema': 'SAR1-decisions-v1', 'utc': C.now(), 'plan': plan.get('plan_id'), 'bank': bank,
                                            'plan_sha256': C.sha(Path(plan_path)),
                                            'rules_sha256': C.sha(RU.RULES_PATH), 'decisions': dec, 'grid_validity': gv,
                                            'grid_validity_checks': len(checks), 'seeds': seeds, 'boot_seeds': boot,
                                            'registration_status': rec.get('registration_status_detail'),
                                            'reader_budget': rec.get('reader_budget'),
                                            'inputs_summary': {'grid_entries': sorted(rec.get('grid') or {}),
                                                               'costs_vs': sorted(k for k, v in (rec.get('costs_vs') or {}).items()
                                                                                  if v is not None),
                                                               'nolookup': sorted(k for k, v in (rec.get('nolookup') or {}).items()
                                                                                  if v is not None)}})
    print(json.dumps({k: (v.get('label') if isinstance(v, dict) else v) for k, v in dec.items() if k != 'scope'}, indent=1,
                     default=str))
    return dec


def no_feasible(plan_path, out):
    """Decision record of a reading whose card rule saved no candidate in both versions (D1 NO_FEASIBLE_STRENGTH): no grid and no
    MODEL_SELECT input is needed; the record carries the plan's per-version candidate status."""
    plan = json.loads(Path(plan_path).read_text())
    if not C.committed(Path(plan_path)):
        raise PermissionError('the plan must be committed before its decision record is written')
    if plan.get('cand') is not None:
        raise PermissionError('the plan has a candidate: use `run`')
    ok, reg = registration_status(plan['card'])
    rec = {'family': plan['family'], 'card': plan['card'], 'revision': int(plan.get('revision', 0)),
           'card_committed_before_any_model_select_outcome': ok,
           'start': {'id': plan['start']['id'], 'evals': {}, 'binding': {int(k): b for k, b in plan['start']['binding'].items()}},
           'cand': {'status': 'NO_FEASIBLE_STRENGTH'}, 'cont_declared': bool(plan.get('cont_declared')),
           'ctrl_origin': plan.get('ctrl_origin', 'START'), 'alt_declared': bool(plan.get('alt_declared')),
           'paths': plan.get('paths') or {}, 'selection_data': plan.get('selection_data') or {},
           'start_in_sample_on_model_fit': plan.get('start_in_sample_on_model_fit'),
           'controls_not_run': {'CONT': False, 'CTRL': False, 'ALT': False}, 'alias_of': {}, 'grid': {}, 'costs_vs': {}, 'nolookup': {}}
    dec = RU.decisions(rec)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    C.write_json_x(out / 'DECISIONS.json', {'schema': 'SAR1-decisions-v1', 'utc': C.now(), 'plan': plan.get('plan_id'), 'bank': None,
                                            'plan_sha256': C.sha(Path(plan_path)),
                                            'rules_sha256': C.sha(RU.RULES_PATH), 'decisions': dec,
                                            'cand_status_by_version': plan.get('cand_status_by_version'),
                                            'registration_status': reg, 'note': plan.get('note')})
    print(json.dumps({k: (v.get('label') if isinstance(v, dict) else v) for k, v in dec.items() if k != 'scope'}, indent=1, default=str))
    return dec


def split_records():
    """Label-free split identities of the SAR1 repetitions (assignment hash; SCORE household overlaps)."""
    import hashlib
    from . import devreaders as DR
    unit = C.features('MODEL_SELECT')['unit']
    roles = {r: DR.roles_of(unit, r) for r in DR.ALL_REPS}
    out = {}
    for r, m in roles.items():
        a = np.where(m['FIT'], 0, np.where(m['VALIDATION'], 1, 2)).astype(np.int8)
        sc = set(np.unique(unit[m['SCORE']]).tolist())
        ov = {}
        for q, mq in roles.items():
            if q != r:
                sq = set(np.unique(unit[mq['SCORE']]).tolist())
                ov[q] = len(sc & sq) / len(sc)
        out[r] = {'assignment_sha256': hashlib.sha256(a.tobytes()).hexdigest(), 'score_overlap': ov, 'score_households': len(sc)}
    return out


def d2r(plan_path, main_bank, rob_bank, out):
    """D2R of a reading's CAND (RULES.md 8; descriptive; never changes D2): the registered and robustness cells merged, refit
    effectiveness on the eight reader seeds and four repetitions, and the registered decisive comparators (START, CONT when gating,
    the registered CTRL / ALT brackets Q* read from the reading's DECISIONS.json; never re-chosen on robustness cells), with the
    material-target references beside."""
    plan = json.loads(Path(plan_path).read_text())
    dec = json.loads((Path(out).parent / plan['plan_id'] / 'DECISIONS.json').read_text())['decisions']
    gm, gr = load_grid(main_bank), load_grid(rob_bank)
    g = merge_grids(gm, gr)
    cand, start = plan['cand'], plan['start']['id']
    cont = plan.get('cont') if plan.get('cont_declared') else None
    d2 = dec.get('D2') or {}
    comps = d2.get('comparisons') or {}

    def q_of(key):
        c = comps.get(key) or {}
        fr = c.get('frontier') or {}
        return fr.get('bracketing_control')            # None when the frontier had no bracket on MODEL_SELECT
    comparators = {'START': start, 'CONT': cont, 'CTRL': q_of('CTRL_FRONTIER')}
    if plan.get('alt_declared'):
        comparators['ALT'] = q_of('ALT_FRONTIER')
    seeds, boot = seed_records(ks=(0, 1, 2, 3), reps=('R1', 'R2', 'R3', 'R4'))
    checks = seed_checks(g, pairs=((0, 1), (2, 3)))
    minimum = [m for m in [cand, start, cont] + [comparators.get('CTRL'), comparators.get('ALT')] if m]
    distinct = distinct_inputs(rob_bank)
    eff = RU.refit_effectiveness(split_records(), seeds, checks, minimum, distinct, boot_seeds=boot)
    keys = [m for m in [start, cont, comparators.get('CTRL'), comparators.get('ALT')] if m] + [r for r in REFS if r != start]
    rob = {k: g['paired'].get(cand, {}).get(k) for k in dict.fromkeys(keys)}
    s_reg, b_reg = seed_records()
    gv = RU.grid_validity(s_reg, seed_checks(gm), minimum, distinct_inputs(main_bank), boot_seeds=b_reg)
    rec = {'grid_valid': gv['effective'], 'release': cand, 'refit': eff, 'rob': {k: v for k, v in rob.items() if v is not None},
           'comparators': comparators, 'alt_declared': bool(plan.get('alt_declared'))}
    res = RU.d2r(rec)
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=True)
    C.write_json_x(outd / 'D2R.json', {'schema': 'SAR1-d2r-v1', 'utc': C.now(), 'plan': plan.get('plan_id'), 'main_bank': main_bank,
                                       'rob_bank': rob_bank, 'comparators': comparators, 'd2r': res, 'refit_effectiveness': eff,
                                       'refit_checks': len(checks), 'seeds': seeds, 'boot_seeds': boot})
    print(json.dumps({'d2r': res.get('label'), 'refit': eff.get('label'), 'comparators': comparators}, indent=1, default=str))
    return res


def summary(dirs, out):
    recs = [json.loads((Path(d) / 'DECISIONS.json').read_text())['decisions'] for d in dirs]
    s = RU.campaign_summary(recs)
    C.write_json(Path(out), {'schema': 'SAR1-campaign-summary-v1', 'utc': C.now(), 'readings': [str(d) for d in dirs], 'summary': s})
    print(json.dumps(s, indent=1, default=str))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    nf = sub.add_parser('no_feasible'); nf.add_argument('--plan', required=True); nf.add_argument('--out', required=True)
    r = sub.add_parser('run'); r.add_argument('--plan', required=True); r.add_argument('--bank', required=True)
    r.add_argument('--out', required=True)
    d = sub.add_parser('d2r'); d.add_argument('--plan', required=True); d.add_argument('--main', required=True)
    d.add_argument('--rob', required=True); d.add_argument('--out', required=True)
    s = sub.add_parser('summary'); s.add_argument('--decisions', nargs='+', required=True); s.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'no_feasible':
        no_feasible(a.plan, a.out); return 0
    if a.cmd == 'run':
        run(a.plan, a.bank, a.out)
    elif a.cmd == 'd2r':
        d2r(a.plan, a.main, a.rob, a.out)
    else:
        summary(a.decisions, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
