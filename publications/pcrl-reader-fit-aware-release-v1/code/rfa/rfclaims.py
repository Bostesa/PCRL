"""Claims assembly of the PCRL reader-fit-aware release campaign (RFA1; coordinator; label-free: it reads saved records only and
applies the statistical reviewer's frozen rules `rfa.rules` (RFA1-rules-v1) unchanged).

    python3 -m rfa.rfclaims plan  --name NAME --fit-record 'FIT_RF_V{v}.json' --card CARD [--revision 0|1] [--prefix P]
                                  [--config-map JSON]                  label-free; writes results/<campaign>/claims/PLAN_<NAME>.json
    python3 -m rfa.rfclaims run   --plan PLAN.json --bank BANK [--lc-bank LC] --out DIR      DECISIONS.json (rfa.rules.decisions)
    python3 -m rfa.rfclaims d2r   --plan PLAN.json --main BANK --rob ROB --out DIR            D2R.json (rfa.rules.d2r; descriptive)

The plan is written from the MODEL_FIT fit records only (saved configurations, operating points, bracket spends, the F0 gate and the
ALTCRIT lag) and must be committed before the reading's first MODEL_SELECT utility or reader computation (`run` refuses an
uncommitted plan). Grid members carry the plan's member names (CAND, CONT, STOP_LO, STOP_HI, ALTC_HI, RAWJ_HI); references keep their
benchmark names (START = NEURAL_RAW). Reused helpers: `rfa.claims` (grid loading, seed records and checks, extra readers, evaluation
records, paired costs, registration status, reader-budget flags, split records).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import claims as CL
from . import common as C
from . import rules as RU

REFS = CL.REFS
FAMILY_DIR = 'FAM'
FIT_DIR = C.RESULTS / 'fam'
CLAIMS_DIR = C.RESULTS / 'claims'
DEFAULT_MAP = {'CAND': 'RF_FULL', 'CONT': 'RF_TASK', 'STOP_LO': 'RF_STOP_LO', 'STOP_HI': 'RF_STOP_HI', 'ALTC_HI': 'RF_ALTC_HI',
               'RAWJ_HI': 'RF_RAWJ_HI'}
PATH_OF = {'STOP_LO': 'STOPFIT', 'STOP_HI': 'STOPFIT', 'ALTC_HI': 'ALTCRIT', 'RAWJ_HI': 'RAWJ'}
FIT_PATH_KEY = {'STOPFIT': 'STOPFIT', 'ALTCRIT': 'ALTCRIT', 'RAWJ': 'ALT'}      # rules key -> fit-record path key
NEAR_ALIAS_LAG = 0.01


def _fineness(recs):
    """rfa.rules.grid_fineness input from the fit records' MODEL_FIT out-of-fold brackets (label-free)."""
    out = {}
    for key, fk in FIT_PATH_KEY.items():
        src = {'selection_data': 'MODEL_FIT_OOF'}
        for v, r in recs.items():
            p = (r.get('paths') or {}).get(fk)
            if not p or r.get('cand', {}).get('spend_over_start') is None:
                continue
            g = p['grid']
            hi, lo = p.get('hi'), p.get('lo')
            src[v] = {'cand': r['cand']['spend_over_start']['value'],
                      'lower': None if lo is None else g[str(lo)]['spend_over_start'],
                      'upper': None if hi is None else g[str(hi)]['spend_over_start'],
                      'upper_se': None if hi is None else g[str(hi)]['se_diff_vs_cand'],
                      'max': max(x['spend_over_start'] for x in g.values()),
                      'grid': [{'spend': x['spend_over_start'], 'se': x['se_diff_vs_cand']} for x in g.values()]}
        out[key] = src
    return out


def write_plan(name, fit_record, card, revision=0, prefix='', config_map=None, bank=None, lc_bank=None, rob_bank=None,
               inner_size=None):
    from . import devreaders as DR
    cmap = config_map or DEFAULT_MAP
    recs = {v: json.loads((FIT_DIR / fit_record.format(v=v)).read_text()) for v in C.VERSIONS}
    saved = {v: set((recs[v].get('saved') or {})) for v in C.VERSIONS}
    members, missing = {}, []
    for m, cfg in cmap.items():
        if all(cfg in saved[v] for v in C.VERSIONS):
            members[m] = f'{FAMILY_DIR}/{cfg}'
        elif any(cfg in saved[v] for v in C.VERSIONS):
            missing.append({'member': m, 'config': cfg, 'saved_in_versions': [v for v in C.VERSIONS if cfg in saved[v]]})
    cand = 'CAND' if 'CAND' in members else None
    paths = {k: [m for m in ('STOP_LO', 'STOP_HI', 'ALTC_HI', 'RAWJ_HI') if PATH_OF[m] == k and m in members] or None
             for k in ('STOPFIT', 'ALTCRIT', 'RAWJ')}
    lags = {v: recs[v].get('altcrit_rel_lag_median_deployed') for v in C.VERSIONS}
    near_alias = all(isinstance(x, (int, float)) and x < NEAR_ALIAS_LAG for x in lags.values())
    unit = C.features('MODEL_SELECT')['unit']
    fit_sizes = {r: int(DR.roles_of(unit, r)['FIT'].sum()) for r in DR.REPS}
    half_sizes = {r: int(DR.lc_fit(unit, DR.roles_of(unit, r)['FIT'], 0.5).sum()) for r in DR.REPS}
    rr = recs[0].get('recipe_rf') or {}
    plan = {'plan_id': name, 'family': 'RFA', 'card': card, 'revision': int(revision), 'label_prefix': prefix, 'written_utc': C.now(),
            'fit_records': {v: f'results/{C.CAMPAIGN}/fam/{fit_record.format(v=v)}' for v in C.VERSIONS},
            'fit_records_sha256': {v: C.sha(FIT_DIR / fit_record.format(v=v)) for v in C.VERSIONS},
            'start': {'id': 'NEURAL_RAW'}, 'cand': cand, 'cont': 'CONT' if 'CONT' in members else None, 'members': members,
            'missing_in_one_version': missing, 'paths': paths,
            'control_kind': {'ALTCRIT': 'PATH'}, 'altcrit_near_alias': near_alias, 'altcrit_rel_lag_median': lags,
            'selection_data': {'STOPFIT': 'MODEL_FIT_OOF', 'ALTCRIT': 'MODEL_FIT_OOF', 'RAWJ': 'MODEL_FIT_OOF'},
            'cand_meta': {'selection_data': 'MODEL_FIT_OOF', 'selection_uses_sex_diagnostic': False,
                          'operating_point': {v: (recs[v].get('cand') or {}).get('checkpoint') for v in C.VERSIONS},
                          'alias_of_start': False,
                          'disclosure': 'the F0 mechanism gate (SEX-based, MODEL_FIT only) decided whether the family was fitted; it '
                                        'selected no operating point'},
            'mechanism_gate': {v: {k: (recs[v].get('mechanism_gate') or {}).get(k)
                                   for k in ('label', 'systematic_share', 'orthogonal_share', 'cos_tot_dir', 'n')} for v in C.VERSIONS},
            'controls_not_run': {'CONT': False, 'STOPFIT': False, 'ALTCRIT': False, 'RAWJ': False}, 'alias_of': {},
            'fineness_inputs': _fineness(recs), 'bank': bank, 'lc_bank': lc_bank, 'rob_bank': rob_bank,
            'learning_curve_budgets': {'FULL': int(round(np.mean(list(fit_sizes.values())))),
                                       'HALF': int(round(np.mean(list(half_sizes.values())))),
                                       'per_repetition': {'FULL': fit_sizes, 'HALF': half_sizes}},
            'inner_episode_fit_size': int(inner_size or rr.get('n_A') or 0), 'joint': None, 'novelty': None}
    out = CLAIMS_DIR / f'PLAN_{name}.json'
    if out.exists():
        raise PermissionError(f'{out.name} exists (write-once)')
    C.write_json(out, plan)
    print(json.dumps({'plan': str(out.relative_to(C.ROOT)), 'cand': cand, 'members': members, 'paths': paths, 'missing': missing,
                      'altcrit_near_alias': near_alias}, indent=1))
    return out


def _half_entries(lc_grid, cand, comps):
    """Learning-curve entries: CAND vs each comparator on the half-FIT cells, keyed by the registered cell names."""
    out = {}
    pc = (lc_grid.get('paired') or {}).get(cand) or {}
    for c in comps:
        e = pc.get(c)
        if not e:
            continue
        out[c] = {view: {'cells': {k.split('@')[0]: x for k, x in d['cells'].items() if k.endswith('@0.5')},
                         'cell_se': {k.split('@')[0]: x for k, x in d['cell_se'].items() if k.endswith('@0.5')}}
                  for view, d in e.items()}
    return out


def _diff_se(lc_bank, cand, comps):
    """Paired-bootstrap SE per registered cell of b(FULL) - b(HALF) for the pair view (the same SCORE households and bootstrap
    weights at both FIT fractions: boot_weights depends on version and repetition only)."""
    out = {c: {} for c in comps}
    for v in C.VERSIONS:
        p = C.RUNS / 'GRID' / lc_bank / f'S{v}' / 'BOOT.npz'
        if not p.is_file():
            continue
        with np.load(p) as z:
            keys = set(z.files)
            for cell in [k.split('__')[0] for k in keys if k.endswith(f'__{cand}__pair') and '@' not in k]:
                half = f'{cell}@0.5'
                for c in comps:
                    need = [f'{cell}__{cand}__pair', f'{cell}__{c}__pair', f'{half}__{cand}__pair', f'{half}__{c}__pair']
                    if all(k in keys for k in need):
                        d = (z[need[1]] - z[need[0]]) - (z[need[3]] - z[need[2]])
                        out[c][cell] = float(np.std(d, ddof=1))
    return {c: x for c, x in out.items() if x}


def build(plan, grid, bank, lc_grid=None, lc_bank=None, gv_effective=None):
    mem = plan['members']
    cand, start = plan['cand'], plan['start']['id']
    ok, reg = CL.registration_status(plan['card'])
    rec = {'family': plan['family'], 'card': plan['card'], 'revision': int(plan.get('revision', 0)),
           'card_committed_before_any_model_select_outcome': ok, 'registration_status_detail': reg}
    ev_s = CL.evals_of(f'REF:{start}')
    rec['start'] = {'id': start, 'evals': {v: e['utility'] for v, e in (ev_s or {}).items()}}
    if cand is None:
        rec['cand'] = {'status': 'NO_FEASIBLE_POINT'}
        return rec
    ev_c = CL.evals_of(mem[cand])
    cm = plan.get('cand_meta') or {}
    rec['cand'] = {'status': 'FITTED' if ev_c else 'MISSING', 'config_id': cand,
                   'evals': {v: e['utility'] for v, e in (ev_c or {}).items()},
                   'operating_point': cm.get('operating_point'), 'selection_data': cm.get('selection_data'),
                   'selection_uses_sex_diagnostic': bool(cm.get('selection_uses_sex_diagnostic')),
                   'alias_of_start': bool(cm.get('alias_of_start'))}
    cont = plan.get('cont')
    if cont:
        rec['cont'] = {'config_id': cont, 'evals': {v: e['utility'] for v, e in (CL.evals_of(mem[cont]) or {}).items()}}
    rec['paths'] = plan['paths']
    rec['control_kind'] = plan.get('control_kind') or {}
    rec['selection_data'] = plan.get('selection_data') or {}
    rec['controls_not_run'] = plan.get('controls_not_run') or {}
    rec['alias_of'] = plan.get('alias_of') or {}
    if plan.get('fineness_inputs'):
        rec['fineness'] = RU.grid_fineness(plan['fineness_inputs'])
    ids = ([cont] if cont else []) + [m for k in ('STOPFIT', 'ALTCRIT', 'RAWJ') for m in (plan['paths'].get(k) or [])]
    pc = grid['paired'].get(cand, {})
    rec['grid'] = {k: pc[k] for k in [start] + ids + list(REFS) if k in pc}
    rec['costs_vs'] = {start: CL.costs_between(mem[cand], f'REF:{start}'),
                       **{m: CL.costs_between(mem[cand], mem[m]) for m in ids}}
    rec['nolookup'] = {k: CL.extra_benefit(grid, cand, k) for k in [start] + ids + list(REFS)}
    rec['enhanced'] = {k: CL.extra_benefit(grid, cand, k, 'enhanced') for k in [start] + ids}
    rec['control_evals'] = {m: {v: e['utility'] for v, e in (CL.evals_of(mem[m]) or {}).items()} for m in ids}
    rec['reader_budget'] = CL.reader_budget_flags(bank, cand)
    rec['joint'] = None
    rec['novelty'] = None
    if lc_grid is not None:
        comps = [start] + ids + ['H_LJ']
        rec['learning_curve'] = {'grid_valid': gv_effective, 'budgets': plan.get('learning_curve_budgets', {}),
                                 'inner_episode_fit_size': plan.get('inner_episode_fit_size'),
                                 'half': _half_entries(lc_grid, cand, comps), 'diff_cell_se': _diff_se(lc_bank, cand, comps)}
    return rec


def run(plan_path, bank, out, lc_bank=None):
    plan = json.loads(Path(plan_path).read_text())
    if not C.committed(Path(plan_path)):
        raise PermissionError('the claims plan must be committed before the grid outputs it governs are read')
    grid = CL.load_grid(bank)
    seeds, boot = CL.seed_records()
    minimum = [m for m in [plan['cand'], plan['start']['id'], plan.get('cont')]
               + [x for k in ('STOPFIT', 'ALTCRIT', 'RAWJ') for x in (plan['paths'].get(k) or [])] if m]
    distinct = CL.distinct_inputs(bank)
    checks = CL.seed_checks(grid)
    gv = RU.grid_validity(seeds, checks, minimum, distinct, boot_seeds=boot, boot_B=400)
    lc_grid = CL.load_grid(lc_bank) if lc_bank else None
    rec = build(plan, grid, bank, lc_grid=lc_grid, lc_bank=lc_bank, gv_effective=gv['effective'])
    rec['grid_valid'] = gv['effective']
    dec = RU.decisions(rec)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    C.write_json_x(out / 'DECISIONS.json', {'schema': 'RFA1-decisions-v1', 'utc': C.now(), 'plan': plan.get('plan_id'), 'bank': bank,
                                            'lc_bank': lc_bank, 'plan_sha256': C.sha(Path(plan_path)),
                                            'rules_sha256': C.sha(RU.RULES_PATH), 'decisions': dec, 'grid_validity': gv,
                                            'grid_validity_checks': len(checks), 'seeds': seeds, 'boot_seeds': boot,
                                            'registration_status': rec.get('registration_status_detail'),
                                            'reader_budget': rec.get('reader_budget'),
                                            'altcrit_near_alias': plan.get('altcrit_near_alias'),
                                            'mechanism_gate': plan.get('mechanism_gate'),
                                            'inputs_summary': {'grid_entries': sorted(rec.get('grid') or {}),
                                                               'costs_vs': sorted(k for k, v in (rec.get('costs_vs') or {}).items()
                                                                                  if v is not None),
                                                               'learning_curve_half': sorted(((rec.get('learning_curve') or {})
                                                                                              .get('half') or {}))}})
    print(json.dumps({k: (v.get('label') if isinstance(v, dict) else v) for k, v in dec.items() if k != 'scope'}, indent=1,
                     default=str))
    return dec


def d2r(plan_path, main_bank, rob_bank, out):
    """D2R of CAND (descriptive; never changes D2): registered and robustness cells merged, refit effectiveness on k0-k3 x R1-R4, and
    the registered decisive comparators read from the reading's DECISIONS.json (never re-chosen on robustness cells)."""
    plan = json.loads(Path(plan_path).read_text())
    dec = json.loads((CLAIMS_DIR / plan['plan_id'] / 'DECISIONS.json').read_text())['decisions']
    gm, gr = CL.load_grid(main_bank), CL.load_grid(rob_bank)
    g = CL.merge_grids(gm, gr)
    cand, start = plan['cand'], plan['start']['id']
    comps = ((dec.get('D2') or {}).get('comparisons')) or {}
    comparators = {'START': start, 'CONT': plan.get('cont')}
    for k in ('STOPFIT', 'ALTCRIT', 'RAWJ'):
        c = comps.get(k) or {}
        comparators[k] = 'ALIAS_OF_STOPFIT' if c.get('kind') == 'ALIAS_OF_STOPFIT' else c.get('point')
    seeds, boot = CL.seed_records(ks=(0, 1, 2, 3), reps=('R1', 'R2', 'R3', 'R4'))
    checks = CL.seed_checks(g, pairs=((0, 1), (2, 3)))
    minimum = [m for m in [cand, start, plan.get('cont')] + [comparators.get(k) for k in ('STOPFIT', 'ALTCRIT', 'RAWJ')]
               if m and m != 'ALIAS_OF_STOPFIT']
    distinct = CL.distinct_inputs(rob_bank)
    eff = RU.refit_effectiveness(CL.split_records(), seeds, checks, minimum, distinct, boot_seeds=boot, boot_B=400)
    keys = list(dict.fromkeys([m for m in minimum if m != cand] + [r for r in REFS]))
    rob = {k: g['paired'].get(cand, {}).get(k) for k in keys}
    s_reg, b_reg = CL.seed_records()
    gv = RU.grid_validity(s_reg, CL.seed_checks(gm), minimum, CL.distinct_inputs(main_bank), boot_seeds=b_reg, boot_B=400)
    rec = {'grid_valid': gv['effective'], 'release': cand, 'refit': eff, 'rob': {k: v for k, v in rob.items() if v is not None},
           'comparators': comparators}
    res = RU.d2r(rec)
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=True)
    C.write_json_x(outd / 'D2R.json', {'schema': 'RFA1-d2r-v1', 'utc': C.now(), 'plan': plan.get('plan_id'), 'main_bank': main_bank,
                                       'rob_bank': rob_bank, 'comparators': comparators, 'd2r': res, 'refit_effectiveness': eff,
                                       'refit_checks': len(checks), 'seeds': seeds, 'boot_seeds': boot})
    print(json.dumps({'d2r': res.get('label'), 'refit': eff.get('label'), 'comparators': comparators}, indent=1, default=str))
    return res


LC_REFS = ('NEURAL_RAW', 'H_LJ')
ROB_REFS = ('NEURAL_RAW', 'H_LJ', 'SUCC', 'RETIRED_B3CTL')


def _manual_spec(bank_id, purpose, members, cand, reps, ks, fractions=None):
    from . import banks as BK
    from . import devreaders as DR
    spec = {'bank_id': bank_id, 'purpose': purpose, 'members': members, 'candidates': [cand],
            'comparators': [m for m in members if m != cand], 'reps': list(reps), 'reader_seeds': list(ks), 'written_utc': C.now(),
            'benchmark_references': [r for r in BK.REFERENCE_MEMBERS if r in members], 'diagnostics': []}
    if fractions:
        spec['fractions'] = list(fractions)
    DR.validate_spec(spec)
    out = C.RESULTS / 'grid_specs' / f'{bank_id}.json'
    if out.exists():
        raise PermissionError(f'{out.name} exists (write-once)')
    C.write_json(out, spec)
    return out


def write_banks(plan_path):
    """The main bank (every reference and diagnostic: rfa.banks.write) and the learning-curve bank (FIT fractions 1.0 and 0.5 on
    R1 / R2, k0 / k1; CAND, START, CONT, every saved control point and H_LJ), both written label-free from the plan before any reader
    output exists."""
    from . import banks as BK
    plan = json.loads(Path(plan_path).read_text())
    members = {m: s for m, s in plan['members'].items()}
    main_out = BK.write(plan['bank'], f"RFA1 reading {plan['plan_id']}: CAND, CONT and the saved control points with every reference",
                        members, [plan['cand']], reps=['R1', 'R2'], reader_seeds=[0, 1])
    lc_mem = {m: BK.family_member(s) for m, s in members.items()}
    for r in LC_REFS:
        lc_mem[r] = BK.REFERENCE_MEMBERS[r]
    lc_out = _manual_spec(plan['lc_bank'], f"RFA1 learning curve of reading {plan['plan_id']}: FIT fractions 1.0 (cache) and 0.5 "
                                           '(the fixed RFA1_devgrid_LC half), R1 / R2, k0 / k1', lc_mem, plan['cand'], ['R1', 'R2'],
                          [0, 1], fractions=[1.0, 0.5])
    print(json.dumps({'main': str(main_out.relative_to(C.ROOT)), 'lc': str(lc_out.relative_to(C.ROOT))}))
    return main_out, lc_out


def write_rob_bank(plan_path):
    """The second-split bank (R1-R4 x k0-k3; registered cells are cache hits): CAND, START, CONT, the registered D2 bracket points
    read from DECISIONS.json (never re-chosen) and four references beside; written before any robustness output exists."""
    from . import banks as BK
    plan = json.loads(Path(plan_path).read_text())
    dec = json.loads((CLAIMS_DIR / plan['plan_id'] / 'DECISIONS.json').read_text())['decisions']
    comps = ((dec.get('D2') or {}).get('comparisons')) or {}
    pts = [((comps.get(k) or {}).get('point')) for k in ('STOPFIT', 'ALTCRIT', 'RAWJ')]
    mem = {}
    for m in [plan['cand'], plan.get('cont')] + [p for p in pts if p]:
        if m and m in plan['members']:
            mem[m] = BK.family_member(plan['members'][m])
    for r in ROB_REFS:
        mem[r] = BK.REFERENCE_MEMBERS[r]
    out = _manual_spec(plan['rob_bank'], f"RFA1 second split of reading {plan['plan_id']}: R1-R4 x k0-k3, the registered decisive "
                                         'comparators and references beside', mem, plan['cand'], ['R1', 'R2', 'R3', 'R4'], [0, 1, 2, 3])
    print(json.dumps({'rob': str(out.relative_to(C.ROOT)), 'members': sorted(mem)}))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='rfa.rfclaims')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('banks').add_argument('--plan', required=True)
    sub.add_parser('rob_bank').add_argument('--plan', required=True)
    p = sub.add_parser('plan'); p.add_argument('--name', required=True); p.add_argument('--fit-record', required=True)
    p.add_argument('--card', required=True); p.add_argument('--revision', type=int, default=0); p.add_argument('--prefix', default='')
    p.add_argument('--config-map', default=None); p.add_argument('--bank'); p.add_argument('--lc-bank'); p.add_argument('--rob-bank')
    r = sub.add_parser('run'); r.add_argument('--plan', required=True); r.add_argument('--bank', required=True)
    r.add_argument('--lc-bank'); r.add_argument('--out', required=True)
    d = sub.add_parser('d2r'); d.add_argument('--plan', required=True); d.add_argument('--main', required=True)
    d.add_argument('--rob', required=True); d.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'banks':
        write_banks(a.plan); return 0
    if a.cmd == 'rob_bank':
        write_rob_bank(a.plan); return 0
    if a.cmd == 'plan':
        write_plan(a.name, a.fit_record, a.card, a.revision, a.prefix, json.loads(a.config_map) if a.config_map else None,
                   a.bank, a.lc_bank, a.rob_bank)
    elif a.cmd == 'run':
        run(a.plan, a.bank, a.out, a.lc_bank)
    else:
        d2r(a.plan, a.main, a.rob, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
