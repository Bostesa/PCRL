"""Label-free plan writer of the PCRL strong-anchor release campaign (coordinator; SAR1 rules). After a reading's fits it reads the
family fit records and the saved release records and writes, before any MODEL_SELECT utility or reader computation of that reading:
- the role members of its version-specific bracket points (`paths/SHARED_ROLES.json`, merged; sar.roles): a CTRL / ALT side that is a
  saved configuration in one version and the origin (START) or the continuation in the other is one role member;
- the reading's plan `claims/PLAN_<reading>.json` (RULES.md 10 and the reviewer's report section 7: START with its frozen wires and
  declared binding condition, CAND and its strengths, CONT, the CTRL / ALT paths with origin, selection data and the full grid, aliases,
  the minimum set, the budget count).

    python3 -m sar.plan reading --reading FA_REG          (or FB_REG, or a revision key of READINGS)

The bank spec and the evaluation lane of the readings are written by `bank` / `lanes` once every plan exists.
"""
from __future__ import annotations
import argparse
import json
import sys

from . import common as C
from . import release as RL

FROZEN_BINDING = {'SUCC': {'0': 'guard|sex0', '1': 'guard|sex0'}, 'NEURAL_RAW': {'0': 'guard|sex0', '1': 'income|U0|brier_excess'}}

READINGS = {
    'FA_REG': {'family': 'FA', 'revision': 0, 'card': 'results/pcrl_strong_anchor_release_v1/cards/FA_TFR_v1.md', 'start': 'SUCC',
               'kind': 'tfr', 'record': 'fa/FIT_V{v}.json', 'cand': 'FA_TFR', 'cont': 'FA_TFR_CONT', 'alt_declared': False,
               'roles': {'CTRL': ('FA_CTRL_LO', 'FA_CTRL_HI')}},
    'FB_REG': {'family': 'FB', 'revision': 0, 'card': 'results/pcrl_strong_anchor_release_v1/cards/FB_PFD_v1.md', 'start': 'NEURAL_RAW',
               'kind': 'pfd', 'record': 'fb/FIT_PFD_V{v}.json', 'cand': 'FB_PFD', 'cont': 'FB_PFD_TASK', 'alt_declared': True,
               'roles': {'CTRL': ('FB_CTRL_LO', 'FB_CTRL_HI'), 'ALT': ('FB_ALT_LO', 'FB_ALT_HI')}},
    'FA_R1': {'family': 'FA', 'revision': 1, 'card': 'results/pcrl_strong_anchor_release_v1/cards/FA_TFRO_R1.md', 'start': 'SUCC',
              'kind': 'tfr', 'record': 'fa/FIT_R1_V{v}.json', 'cand': 'FA_TFRO_R1', 'cont': 'FA_TFRO_R1_CONT', 'alt_declared': False,
              'roles': {'CTRL': ('FA_R1_CTRL_LO', 'FA_R1_CTRL_HI')}},
    'FA_R2': {'family': 'FA', 'revision': 2, 'card': 'results/pcrl_strong_anchor_release_v1/cards/FA_TFRB_R2.md', 'start': 'SUCC',
              'kind': 'tfr', 'record': 'fa/FIT_R2_V{v}.json', 'cand': 'FA_TFRB_R2', 'cont': 'FA_TFRO_R1_CONT', 'alt_declared': False,
              'roles': {'CTRL': ('FA_R2_CTRL_LO', 'FA_R2_CTRL_HI')}},
    'FS_CPI': {'family': 'FS_CPI', 'release_family': 'FS', 'revision': 0, 'card': 'results/pcrl_strong_anchor_release_v1/cards/FS_CPI_v1.md',
               'start': 'SUCC', 'kind': 'pfd', 'record': 'fs/FIT_CPI_V{v}.json', 'cand': 'FS_CPI', 'cont': None, 'alt_declared': True,
               'roles': {'CTRL': ('FS_CTRL_LO', 'FS_CTRL_HI'), 'ALT': ('FS_ALT_LO', 'FS_ALT_HI')}},
    'FB_R1': {'family': 'FB', 'revision': 1, 'card': 'results/pcrl_strong_anchor_release_v1/cards/FB_PFO_R1.md', 'start': 'NEURAL_RAW',
              'kind': 'pfd', 'record': 'fb/FIT_PFO_V{v}.json', 'cand': 'FB_PFO', 'cont': 'FB_PFO_TASK', 'alt_declared': True,
              'roles': {'CTRL': ('FB_R1_CTRL_LO', 'FB_R1_CTRL_HI'), 'ALT': ('FB_R1_ALT_LO', 'FB_R1_ALT_HI')}},
}


def _rec(R, v):
    p = C.RESULTS / R['record'].format(v=v)
    return json.loads(p.read_text()) if p.is_file() else None


def _wires(fam, cfg, v):
    rec = json.loads((RL.rdir(fam, cfg, v) / 'RELEASE.json').read_text())
    return {'income': rec['wire_sha256']['MODEL_SELECT__income'], 'occupation': rec['wire_sha256']['MODEL_SELECT__occupation'],
            'release_sha256': rec['release_sha256']}


def side_specs(R, rec, key):
    """Per path side: {'LO': spec, 'HI': spec} for one version's fit record, a spec being 'FAM/CONFIG', 'REF:<start>' (the origin) or
    'FAM/<cont>' (the continuation); plus the grid rows, the flags and the selection data."""
    fam, start = R['family'], R['start']
    if R['kind'] == 'tfr':
        c = rec.get('ctrl') or {}
        out = {}
        for side in ('LO', 'HI'):
            sv = (c.get('saved') or {}).get(side) or {}
            if sv.get('config'):
                out[side] = f"{fam}/{sv['config']}"
            elif sv.get('alias') == 'START':
                out[side] = f'REF:{start}'
            elif sv.get('alias'):
                out[side] = f"{fam}/{sv['alias']}"
            else:
                out[side] = f'REF:{start}'               # no qualifying grid point: the origin closes the path on that side
        return out, {'grid': c.get('path'), 'flags': c.get('flags'), 'selection_data': c.get('selection_data'),
                     'cand_spend': (rec.get('cand') or {}).get('cal_spend_over_SUCC'), 'origin': c.get('origin')}
    p = (rec.get('paths') or {}).get(key) or {}
    out = {}
    for side in ('LO', 'HI'):
        cfg = p.get(f'{side.lower()}_config')
        if cfg:
            out[side] = f'{fam}/{cfg}'
        elif side == 'LO' and p.get('hi_config') and 'LO_EQUALS_HI' in (p.get('flags') or []):
            out[side] = f"{fam}/{p['hi_config']}"
        else:
            out[side] = f'REF:{start}'
    return out, {'grid': p.get('grid'), 'flags': p.get('flags'), 'selection_data': p.get('selection_data'),
                 'cand_spend': (rec.get('cand') or {}).get('spend_over_start'), 'origin': p.get('origin'),
                 'fineness': p.get('fineness')}


def fineness_inputs(R, recs, paths_saved):
    """sar.rules.grid_fineness input from the fit records: per frontier path and version CAND's spend over START on the declared
    selection data, the saved lower point's spend (the origin = 0.0 when no lower point is saved), the upper point's spend and the SE of
    its paired difference against CAND, the largest grid spend and the full grid."""
    out = {}
    keys = ['CTRL'] + (['ALT'] if R['alt_declared'] else [])
    for key in keys:
        d = {}
        for v, rec in recs.items():
            if R['kind'] == 'tfr':
                c = rec.get('ctrl') or {}
                grid = c.get('path') or {}
                cand = ((rec.get('cand') or {}).get('cal_spend_over_SUCC') or {}).get('spend_over_start')
                lo, hi = c.get('lo'), c.get('hi')
                sd = c.get('selection_data')
            else:
                c = (rec.get('paths') or {}).get(key) or {}
                grid = c.get('grid') or {}
                cand = ((rec.get('cand') or {}).get('spend_over_start') or {}).get('value')
                lo, hi = c.get('lo'), c.get('hi')
                sd = c.get('selection_data')
                if not c.get('lo_config'):
                    lo = None                                # the lower point is not saved: the origin serves as the lower point
            row = lambda g: grid.get(str(g)) or grid.get(g) or {}
            pts = [{'spend': r.get('spend_over_start'), 'se': r.get('se_diff_vs_cand')} for r in grid.values()]
            d[str(v)] = {'cand': cand, 'lower': (row(lo).get('spend_over_start') if lo is not None else 0.0),
                         'upper': row(hi).get('spend_over_start') if hi is not None else None,
                         'upper_se': row(hi).get('se_diff_vs_cand') if hi is not None else None,
                         'max': max([x['spend'] for x in pts if x['spend'] is not None], default=None), 'grid': pts}
            d['selection_data'] = sd
        out[key] = d
    return out


def reading(name):
    R = READINGS[name]
    fam, start = R.get('release_family', R['family']), R['start']
    recs = {v: _rec(R, v) for v in C.VERSIONS}
    cand_ok = all(r is not None and (r.get('cand') or {}).get('status') == 'FITTED' for r in recs.values())
    plan = {'plan_id': name, 'family': R['family'], 'card': R['card'], 'revision': R['revision'], 'written_utc': C.now(),
            'start': {'id': start, 'binding': FROZEN_BINDING[start], 'wires': None},
            'start_in_sample_on_model_fit': True, 'alt_declared': R['alt_declared'], 'cont_declared': R.get('cont') is not None,
            'ctrl_origin': 'START', 'release_family': fam,
            'novelty': None, 'joint': None, 'written_before_model_select': True}
    adm = json.loads((C.RESULTS / 'SOURCE_ADMISSION.json').read_text())['reference_releases_MODEL_SELECT_wire_sha256']
    plan['start']['wires'] = {str(v): adm[f'{start}/S{v}'] for v in C.VERSIONS}
    if not cand_ok:
        plan.update({'cand': None, 'cand_status_by_version': {str(v): ((r or {}).get('cand') or {}).get('status') for v, r in recs.items()},
                     'members': {}, 'paths': {'CTRL': [], 'ALT': [] if R['alt_declared'] else None}, 'strengths': [],
                     'note': 'no candidate saved in both versions (NO_FEASIBLE_STRENGTH): the reading ends at D1'})
        return plan, {}
    members = {R['cand']: f"{fam}/{R['cand']}", **({R['cont']: f"{fam}/{R['cont']}"} if R.get('cont') else {})}
    roles, paths, sel, grids = {}, {}, {}, {}
    for key, (lo_name, hi_name) in R['roles'].items():
        per = {}
        for v, rec in recs.items():
            sides, info = side_specs(R, rec, key)
            per[v] = sides
            grids.setdefault(key, {})[str(v)] = info
            sel[key] = info['selection_data']
        ids = []
        for side, rname in (('LO', lo_name), ('HI', hi_name)):
            m = {str(v): per[v][side] for v in C.VERSIONS}
            if all(x == f'REF:{start}' for x in m.values()):
                continue                                   # the origin in both versions: not a separate path point
            roles[rname] = m
            members[rname] = f'ROLE:{rname}'
            ids.append(rname)
        paths[key] = ids
    if not R['alt_declared']:
        paths['ALT'] = None
    strengths = {str(v): (rec['cand'].get('rho_star') if R['kind'] == 'tfr' else rec['cand'].get('strength')) for v, rec in recs.items()}
    plan.update({'cand': R['cand'], 'cont': R['cont'], 'members': members, 'paths': paths, 'selection_data': sel,
                 'controls_not_run': {'CONT': False, 'CTRL': False, 'ALT': False}, 'alias_of': {},
                 'strengths': [{'config_id': R['cand'], 'strength': strengths}],
                 'every_nonzero_strength_alias_of_start': False,
                 'path_records': grids, 'roles': roles, 'fineness_inputs': fineness_inputs(R, recs, paths),
                 'wires': {m: {str(v): (_wires(fam, spec.split('/')[1], v) if not spec.startswith('ROLE:') else
                                        (None if roles[m][str(v)].startswith('REF:') else _wires(*roles[m][str(v)].split('/'), v)))
                               for v in C.VERSIONS} for m, spec in members.items()},
                 'minimum_set': [R['cand'], start] + ([R['cont']] if R.get('cont') else []) + [x for k in ('CTRL', 'ALT') for x in (paths.get(k) or [])],
                 'disclosures': (['H3 (SUCC income) and Ucal are in-sample on CALIBRATION: the income-side CALIBRATION spends over SUCC are '
                                  'biased in SUCC\'s favour (conservative for CAND)'] if start == 'SUCC' else [])})
    # exact aliases among saved points (identical MODEL_SELECT wires in both versions)
    seen = {}
    for m in plan['minimum_set']:
        if m == start:
            continue
        w = plan['wires'].get(m) or {}
        key = tuple((w.get(str(v)) or {}).get('release_sha256') for v in C.VERSIONS)
        if None not in key and key in seen:
            plan['alias_of'][m] = seen[key]
        elif None not in key:
            seen[key] = m
    return plan, roles


def write_reading(name):
    plan, roles = reading(name)
    rp = C.RESULTS / 'paths' / 'SHARED_ROLES.json'
    cur = json.loads(rp.read_text()) if rp.is_file() else {'schema': 'SAR1-roles-v1', 'roles': {}}
    for k, m in roles.items():
        if k in cur['roles'] and cur['roles'][k] != m:
            raise PermissionError(f'role {k} exists with a different mapping (roles are write-once)')
        cur['roles'][k] = m
    cur['utc'] = C.now()
    C.write_json(rp, cur)
    out = C.RESULTS / 'claims' / f'PLAN_{name}.json'
    if out.exists():
        raise PermissionError(f'{out.name} exists (a plan is write-once; a revision gets its own reading key)')
    C.write_json(out, plan)
    print(json.dumps({'plan': str(out.relative_to(C.ROOT)), 'cand': plan.get('cand'), 'paths': plan.get('paths'), 'roles': roles,
                      'alias_of': plan.get('alias_of')}, indent=1))
    return plan


# ---------------------------------------------------------------- bank spec and lanes of a set of readings (label-free)
def lanes(names, bank_id, purpose):
    """Write the reader bank spec of the readings' members (candidates = each reading's CAND) and the evaluation, grid and claims lanes.
    Every role member enters the bank as a hash-pinned 'arrays' member (sar.roles.array_member)."""
    from . import banks as BK
    from . import roles as RO
    plans = [json.loads((C.RESULTS / 'claims' / f'PLAN_{n}.json').read_text()) for n in names]
    members, cands = {}, []
    for p in plans:
        if p.get('cand') is None:
            continue
        cands.append(p['cand'])
        for m, spec in p['members'].items():
            members[m] = RO.array_member(m) if spec.startswith('ROLE:') else spec
    spec_path = BK.write(bank_id, purpose, members, cands)
    ev, cl = [], []
    for p in plans:
        if p.get('cand') is None:
            continue
        start = p['start']['id']
        cont = p['members'].get(p.get('cont')) if p.get('cont') else None
        comps = [f'REF:{start}'] + ([cont] if cont else []) + [p['members'][x] for k in ('CTRL', 'ALT') for x in (p['paths'].get(k) or [])]
        rels = [p['members'][p['cand']]] + ([cont] if cont else [])
        ev.append({'unit': f"SAR_EVAL_{p['plan_id']}_V1", 'stage': 'evaluate', 'rss_gib': 2.5,
                   'argv': ['PY', '-m', 'sar.evaluate', 'many'] + [a for r in rels for a in ('--release', r)]
                           + [a for c in comps for a in ('--compare', c)]})
        roles = [x for k in ('CTRL', 'ALT') for x in (p['paths'].get(k) or []) if p['members'][x].startswith('ROLE:')]
        if roles:
            ev.append({'unit': f"SAR_EVAL_{p['plan_id']}_ROLES_V1", 'stage': 'evaluate', 'rss_gib': 2.5,
                       'argv': ['PY', '-m', 'sar.evaluate', 'roles'] + [a for r in roles for a in ('--role', r)]
                               + ['--compare', f'REF:{start}'] + (['--compare', cont] if cont else [])})
        cl.append({'unit': f"SAR_CLAIMS_{p['plan_id']}_V1", 'stage': 'claims', 'rss_gib': 1.5,
                   'argv': ['PY', '-m', 'sar.claims', 'run', '--plan', f"results/{C.CAMPAIGN}/claims/PLAN_{p['plan_id']}.json",
                            '--bank', bank_id, '--out', f"results/{C.CAMPAIGN}/claims/{p['plan_id']}"]})
    L = C.RESULTS / 'lanes'
    C.write_json(L / f'EVAL_{bank_id}.json', ev)
    for v in C.VERSIONS:
        C.write_json(L / f'GRID_{bank_id}_V{v}.json', [{'unit': f'SAR_GRID_{bank_id}_V{v}', 'stage': 'readers', 'rss_gib': 3.0,
                                                       'max_wall_s': 10800,
                                                       'argv': ['TORCH', '-m', 'sar.devreaders', 'run', '--spec',
                                                                str(spec_path.relative_to(C.ROOT)), '--seed', str(v)]}])
    C.write_json(L / f'CLAIMS_{bank_id}.json', cl)
    print(json.dumps({'bank': str(spec_path.relative_to(C.ROOT)), 'candidates': cands, 'members': sorted(members),
                      'eval_units': [u['unit'] for u in ev], 'claims_units': [u['unit'] for u in cl]}, indent=1))
    return spec_path


def rob_bank(name, bank_id, extra_refs=('H_LJ', 'SUCC', 'RETIRED_B3CTL')):
    """The D2R robustness bank of a reading (RULES.md 8): repetitions R1-R4 and reader seeds k0-k3 (the registered cells are cache
    hits), members = CAND, START, CONT, the registered CTRL / ALT brackets Q* (read from the reading's DECISIONS.json; never re-chosen)
    and a few references beside (the material-target references among them); written before any robustness output exists."""
    from . import banks as BK
    from . import devreaders as DR
    from . import roles as RO
    plan = json.loads((C.RESULTS / 'claims' / f'PLAN_{name}.json').read_text())
    dec = json.loads((C.RESULTS / 'claims' / name / 'DECISIONS.json').read_text())['decisions']
    comps = (dec.get('D2') or {}).get('comparisons') or {}
    qs = [((comps.get(k) or {}).get('frontier') or {}).get('bracketing_control') for k in ('CTRL_FRONTIER', 'ALT_FRONTIER')]
    start = plan['start']['id']
    mem = {}
    for m in [plan['cand'], plan.get('cont')] + [q for q in qs if q]:
        if m and m not in BK.REFERENCE_MEMBERS:
            spec = plan['members'][m]
            mem[m] = RO.array_member(m) if spec.startswith('ROLE:') else BK.family_member(spec)
    for r in dict.fromkeys([start, 'NEURAL_RAW'] + list(extra_refs)):
        mem[r] = BK.REFERENCE_MEMBERS[r]
    spec = {'bank_id': bank_id, 'purpose': f'D2R robustness cells of reading {name}: repetitions R1-R4, reader seeds k0-k3 (registered '
                                            'cells re-used from the cache); the registered decisive comparators and references beside',
            'members': mem, 'candidates': [plan['cand']], 'comparators': [m for m in mem if m != plan['cand']],
            'reps': ['R1', 'R2', 'R3', 'R4'], 'reader_seeds': [0, 1, 2, 3], 'written_utc': C.now(),
            'benchmark_references': [r for r in BK.REFERENCE_MEMBERS if r in mem], 'diagnostics': []}
    DR.validate_spec(spec)
    out = C.RESULTS / 'grid_specs' / f'{bank_id}.json'
    if out.exists():
        raise PermissionError(f'{out.name} exists')
    C.write_json(out, spec)
    L = C.RESULTS / 'lanes'
    units = [{'unit': f'SAR_GRID_{bank_id}_V{v}', 'stage': 'readers', 'rss_gib': 3.0, 'max_wall_s': 10800,
              'argv': ['TORCH', '-m', 'sar.devreaders', 'run', '--spec', str(out.relative_to(C.ROOT)), '--seed', str(v)]} for v in C.VERSIONS]
    units.append({'unit': f'SAR_D2R_{name}_V1', 'stage': 'claims', 'rss_gib': 1.5,
                  'argv': ['PY', '-m', 'sar.claims', 'd2r', '--plan', f'results/{C.CAMPAIGN}/claims/PLAN_{name}.json', '--main',
                           plan.get('bank') or f'SAR1_{name}', '--rob', bank_id, '--out', f'results/{C.CAMPAIGN}/claims/{name}_D2R']})
    C.write_json(L / f'D2R_{name}.json', units)
    print(json.dumps({'bank': str(out.relative_to(C.ROOT)), 'members': sorted(mem), 'units': [u['unit'] for u in units]}, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('reading'); r.add_argument('--reading', required=True)
    b = sub.add_parser('lanes'); b.add_argument('--reading', action='append', required=True); b.add_argument('--bank', required=True)
    b.add_argument('--purpose', required=True)
    rb = sub.add_parser('rob'); rb.add_argument('--reading', required=True); rb.add_argument('--bank', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'rob':
        rob_bank(a.reading, a.bank)
        return 0
    if a.cmd == 'reading':
        write_reading(a.reading)
    elif a.cmd == 'lanes':
        lanes(a.reading, a.bank, a.purpose)
    return 0


if __name__ == '__main__':
    sys.exit(main())
