"""Reading plan of a DR reading (campaign `pcrl_direct_recovery_frontier_v1`, DRF1; coordinator; label-free). Written from the DR fit
records (`results/<campaign>/dr/FIT_V<v>.json`) and the saved releases, and committed BEFORE the reading's first MODEL_SELECT utility
or reader computation (RULES.md 11; reviewer R2-R8). It also writes the reader-bank spec, the version-specific CTRL role members
when a bracket point is the family origin in one version, and the evaluation and grid lanes.

    python3 -m drf.drplan write --plan-id DRF_MAIN --card results/<campaign>/cards/DR_DRF_v1.md [--revision 0] [--suffix '']
"""
from __future__ import annotations
import argparse
import json
import sys

import numpy as np

from . import common as C
from . import dr as DR
from . import release as RL

TWIN = 'TWB_C64'
TWIN_SPEC = 'FRC:FA/TWB_C64'


def _ms(spec, v):
    from . import evaluate as EV
    arr, _ = EV.comparator_release(spec, v)
    return arr


def _eq(a, b):
    return all(np.array_equal(a[t], b[t]) for t in C.TASKS)


def _distinct(rel):
    inc, occ = rel['income'], rel['occupation']
    return {'income': int(np.unique(inc, axis=0).shape[0]), 'occupation': int(np.unique(occ, axis=0).shape[0]),
            'pair': int(np.unique(np.column_stack((inc, occ)), axis=0).shape[0])}


def write(plan_id, card, revision=0, suffix='', revision_note=None, fit_tag='', optimistic_ctrl_fit_tag=''):
    """fit_tag '' reads dr/FIT_V<v>.json (card DR_DRF_v1), 'R1_' reads dr/FIT_R1_V<v>.json (card DR_DRF_R1); the optimistic
    diagnostic's CTRL path comes from the fit records of `optimistic_ctrl_fit_tag` (the registered v1 path: realized in-sample
    labels, the same estimator as OPT_L / OPT_S)."""
    fits = {v: json.loads((C.RESULTS / 'dr' / f'FIT_{fit_tag}V{v}.json').read_text()) for v in C.VERSIONS}
    cand, obj = f'DR_AUC{suffix}', f'DR_MI{suffix}'
    lo, hi = f'DR_CTRL_LO{suffix}', f'DR_CTRL_HI{suffix}'
    # CTRL points per version: a saved configuration, the origin (the twin) or absent
    per = {}
    for name, key in ((lo, f'DR_CTRL_LO{suffix}'), (hi, f'DR_CTRL_HI{suffix}')):
        per[name] = {}
        for v, f in fits.items():
            a = f['ctrl_alias'].get(key)
            if a is None and RL.exists('DR', name, v):
                per[name][v] = f'DR/{name}'
            elif a == 'TWIN':
                per[name][v] = TWIN_SPEC
            else:
                per[name][v] = None
    members, specs, roles, paths, alias_of, notes = {}, {}, {}, [], {}, []
    members[cand], specs[cand] = f'DR/{cand}', f'DR/{cand}'
    members[obj], specs[obj] = f'DR/{obj}', f'DR/{obj}'
    members[TWIN], specs[TWIN] = 'SRCFRC:FA/TWB_C64', TWIN_SPEC
    for name, pv in per.items():
        vals = set(pv.values())
        if vals == {f'DR/{name}'}:
            members[name] = specs[name] = f'DR/{name}'
            paths.append(name)
        elif vals == {TWIN_SPEC}:
            alias_of[name] = 'TWIN'                                # the origin in both versions: not a new point
            notes.append(f'{name} is the origin (the twin) in both versions')
        elif None in vals:
            notes.append(f'{name} is not available in version(s) {[v for v, x in pv.items() if x is None]} '
                         f'(path may not bracket there)')
            avail = {v: x for v, x in pv.items() if x is not None}
            if avail:
                roles[name] = {str(v): x for v, x in avail.items()}
                members[name] = specs[name] = f'ROLE:{name}'
                paths.append(name)
        else:                                                     # a configuration in one version, the origin in the other
            roles[name] = {str(v): x for v, x in pv.items()}
            members[name] = specs[name] = f'ROLE:{name}'
            paths.append(name)
            notes.append(f'{name} is the origin (the twin) in version(s) '
                         f'{[v for v, x in pv.items() if x == TWIN_SPEC]}: a version-specific role member')
    if roles:
        from . import roles as RO
        prev = RO.role_map()
        C.write_json(RO.ROLES_PATH, {'schema': 'DRF1-roles-v1', 'utc': C.now(), 'roles': {**prev, **roles}})
    specs['SUCC'] = 'REF:SUCC'
    # fineness inputs (RULES.md 4.4) from the nested bracket records
    fin = {'CTRL': {'selection_data': 'MODEL_FIT_NESTED'}}
    for v, f in fits.items():
        br = f['brackets']
        grid = br['grid']
        sp = {g['step']: g for g in grid}
        lo_s, hi_s = br['lower_step'], br['upper_step']
        fin['CTRL'][str(v)] = {'cand': br['candidate_spend'],
                               'lower': 0.0 if lo_s in (None, 0) else sp[lo_s]['spend'],
                               'upper': None if hi_s is None else sp[hi_s]['spend'],
                               'upper_se': None if hi_s is None else sp[hi_s]['se_vs_candidate'],
                               'max': max(g['spend'] for g in grid),
                               'grid': [{'spend': g['spend'], 'se': g['se_vs_candidate']} for g in grid if g['step'] > 0]}
    # exact-alias relations and distinct-input counts on MODEL_SELECT wires (R4, R7)
    alias_rec, distinct = {}, {}
    for v in C.VERSIONS:
        w = {cand: _ms(specs[cand], v), obj: _ms(specs[obj], v), TWIN: _ms(TWIN_SPEC, v), 'SUCC': _ms('REF:SUCC', v)}
        for name in paths:
            spec = per[name].get(v)
            if spec:
                w[name] = _ms(spec, v)
        alias_rec[str(v)] = {'cand_equals_twin': _eq(w[cand], w[TWIN]), 'cand_equals_obj_mi': _eq(w[cand], w[obj]),
                             'obj_mi_equals_twin': _eq(w[obj], w[TWIN]),
                             **{f'{n}_equals_twin': _eq(w[n], w[TWIN]) for n in paths if n in w},
                             **{f'{n}_equals_cand': _eq(w[n], w[cand]) for n in paths if n in w}}
        for n, rel in w.items():
            distinct[f'{n}|V{v}'] = _distinct(rel)
    alias_rec['cand_alias_of_twin'] = all(alias_rec[str(v)]['cand_equals_twin'] for v in C.VERSIONS)
    alias_rec['cand_alias_of_obj_mi'] = all(alias_rec[str(v)]['cand_equals_obj_mi'] for v in C.VERSIONS)
    plan = {'plan_id': plan_id, 'family': 'DR', 'card': card, 'revision': int(revision), 'written_utc': C.now(),
            'cand': cand, 'twin': {'id': TWIN}, 'obj_mi': {'config_id': obj}, 'start': {'id': 'SUCC'},
            'members': members, 'compare_specs': specs, 'paths': {'CTRL': paths}, 'alias_of': alias_of,
            'ctrl_points_by_version': {n: {str(v): x for v, x in pv.items()} for n, pv in per.items()}, 'ctrl_notes': notes,
            'selection_data': {'CTRL': 'MODEL_FIT_NESTED'}, 'controls_not_run': {'CTRL': False, 'OBJ': False},
            'fineness_inputs': fin, 'fineness': None, 'reader_stage': 'RUN',
            'budgets': json.loads((C.ROOT / DR.BUDGETS_REL).read_text())['budgets'], 'budgets_sha256': C.sha(C.ROOT / DR.BUDGETS_REL),
            'fit_records': {str(v): {'path': f'results/{C.CAMPAIGN}/dr/FIT_{fit_tag}V{v}.json',
                                     'sha256': C.sha(C.RESULTS / 'dr' / f'FIT_{fit_tag}V{v}.json')} for v in C.VERSIONS},
            'optimistic_ctrl_fit_tag': optimistic_ctrl_fit_tag,
            'optimistic_records': {str(v): f'results/{C.CAMPAIGN}/dr/OPTIMISTIC_V{v}.json (written by DRF_OPT_V{v}_* after this plan; '
                                           f'read by drf.drclaims; reported, never gating)' for v in C.VERSIONS},
            'alias_records': alias_rec, 'distinct_inputs_model_select': distinct,
            'equal_exposure': {k: {f'{cid}|V{v}': (fits[v]['traces'].get(cid) or {}).get(k) for cid in (cand, obj)
                                   for v in C.VERSIONS} for k in ('restarts', 'max_sweeps', 'seed', 'seeds', 'perturb_moves')},
            'statement': ('No preliminary significance, optimistic forecast or novelty gate prevents any fit or reading: DR_AUC, DR_MI and '
                          'the CTRL points are fitted and read whatever the optimistic diagnostic shows. Every G1 statement about DR '
                          'carries the disclosure that its MODEL_FIT constraints were tuned on the twin\'s published MODEL_SELECT '
                          'aggregates. FC (DR vs SUCC) and TWIN are reported apart from D2.')}
    if revision_note:
        plan['revision_note'] = revision_note
    from . import rules as RU
    plan['fineness'] = RU.grid_fineness(fin)
    out = C.RESULTS / 'claims' / f'PLAN_{plan_id}.json'
    C.write_json_x(out, plan)
    return plan, out


def bank_and_lanes(plan, bank_id):
    from . import banks as BK
    from . import roles as RO
    members = {}
    for k, spec in plan['members'].items():
        if spec.startswith('ROLE:'):
            members[k] = RO.array_member(spec[5:])
        elif spec.startswith('SRCFRC:'):
            members[k] = 'FRC:' + spec[7:]
        else:
            members[k] = spec
    cands = [plan['cand'], plan['obj_mi']['config_id']]
    out = BK.write(bank_id, f'registered DRF1 cells (R1, R2 x k0, k1, both versions) for the reading {plan["plan_id"]}: DR_AUC '
                   f'(candidate), DR_MI (previous objective), the CTRL points, the twin TWB_C64, the nine references and U0 / Ucal; '
                   f'written after the plan and before any MODEL_SELECT computation of the reading', members, cands)
    lanes = C.RESULTS / 'lanes'
    pid = plan['plan_id']
    comps = [plan['compare_specs'][k] for k in plan['compare_specs'] if k != plan['cand']] + ['REF:NEURAL_RAW', 'REF:H_LJ']
    rel = [plan['members'][plan['cand']], plan['members'][plan['obj_mi']['config_id']]] + \
          [plan['members'][n] for n in plan['paths']['CTRL'] if plan['members'][n].startswith('DR/')]
    ev = [{'unit': f'DRF_EVAL_{pid}_V1', 'stage': 'evaluate', 'rss_gib': 2.5,
           'argv': ['PY', '-m', 'drf.evaluate', 'many'] + sum([['--release', r] for r in rel], []) + sum([['--compare', c] for c in
                                                                                                          dict.fromkeys(comps)], [])}]
    role_names = [plan['members'][n][5:] for n in plan['paths']['CTRL'] if plan['members'][n].startswith('ROLE:')]
    if role_names:
        ev.append({'unit': f'DRF_EVAL_{pid}_ROLES_V1', 'stage': 'evaluate', 'rss_gib': 2.5,
                   'argv': ['PY', '-m', 'drf.evaluate', 'roles'] + sum([['--role', r] for r in role_names], []) +
                           sum([['--compare', c] for c in dict.fromkeys(comps) if not c.startswith('ROLE:')], [])})
    if not (C.RUNS / 'SOURCE_FRC' / 'FA' / TWIN / 'EVAL_MS.json').is_file():
        ev.append({'unit': 'DRF_EVAL_TWIN_V1', 'stage': 'evaluate', 'rss_gib': 2.0,
                   'argv': ['PY', '-m', 'drf.evaluate', 'source_frc', '--family', 'FA', '--config', TWIN, '--compare', 'REF:SUCC']})
    if not (C.RUNS / 'REFERENCES' / 'EVAL_MS.json').is_file():
        ev.append({'unit': 'DRF_REF_EVAL_V1', 'stage': 'evaluate', 'rss_gib': 2.0,
                   'argv': ['PY', '-m', 'drf.evaluate', 'references', '--name', 'W', '--name', 'REF:NEURAL_RAW', '--name', 'REF:H_LJ',
                            '--name', 'REF:B2TWIN', '--name', 'REF:RETIRED_B3CTL', '--name', 'REF:SUCC', '--name', 'REF:TWJ',
                            '--name', 'REF:H3+TASK_REFIT', '--name', 'REF:H3+T_TREE', '--name', 'REF:U0', '--name', 'REF:Ucal']})
    C.write_json(lanes / f'EVAL_{pid}.json', ev)
    for v in C.VERSIONS:
        C.write_json(lanes / f'GRID_{pid}_V{v}.json', [{'unit': f'DRF_GRID_{pid}_V{v}_V1', 'stage': 'readers', 'rss_gib': 3.0,
                                                         'max_wall_s': 10800,
                                                         'argv': ['TORCH', '-m', 'drf.devreaders', 'run', '--spec',
                                                                  str(out.relative_to(C.ROOT)), '--seed', str(v)]}])
    return out


def rob_bank(plan_id, bank_id, decisions_dir):
    """D2R robustness bank of a reading (RULES.md 8; written after the reading's decisions, before any robustness output): the
    candidate, its registered decisive comparators (the CTRL bracket Q* chosen on MODEL_SELECT and OBJ_MI) and the references beside
    (TWIN, SUCC, NEURAL_RAW, H_LJ, RETIRED_B3CTL), on repetitions R1-R4 with reader seeds k0-k3 (registered cells re-used from the
    cache); nothing is re-chosen. Also writes the two robustness grid lanes."""
    from . import banks as BK
    from . import roles as RO
    plan = json.loads((C.RESULTS / 'claims' / f'PLAN_{plan_id}.json').read_text())
    dec = json.loads((C.RESULTS / 'claims' / decisions_dir / 'DECISIONS.json').read_text())['decisions']
    fr = (((dec.get('D2') or {}).get('comparisons') or {}).get('CTRL_FRONTIER') or {}).get('frontier') or {}
    q = fr.get('bracketing_control')
    keep = [plan['cand'], plan['obj_mi']['config_id'], plan['twin']['id']] + ([q] if q and q in plan['members'] else [])
    members = {}
    for k in keep:
        spec = plan['members'][k]
        members[k] = (RO.array_member(spec[5:]) if spec.startswith('ROLE:') else
                      ('FRC:' + spec[7:] if spec.startswith('SRCFRC:') else spec))
    spec = {'bank_id': bank_id, 'purpose': f'D2R robustness cells of reading {plan_id}: repetitions R1-R4, reader seeds k0-k3 '
                                           f'(registered cells re-used from the cache); the registered decisive comparators (CTRL '
                                           f'bracket {q}, OBJ_MI {plan["obj_mi"]["config_id"]}) and references beside; no retuning',
            'members': {**{r: BK.REFERENCE_MEMBERS[r] for r in ('NEURAL_RAW', 'H_LJ', 'RETIRED_B3CTL', 'SUCC')},
                        **{k: (BK.family_member(m) if isinstance(m, str) else m) for k, m in members.items()}},
            'candidates': [plan['cand']], 'comparators': [k for k in keep if k != plan['cand']] + ['SUCC', 'NEURAL_RAW', 'H_LJ',
                                                                                                  'RETIRED_B3CTL'],
            'written_utc': C.now(), 'benchmark_references': ['NEURAL_RAW', 'H_LJ', 'RETIRED_B3CTL', 'SUCC'], 'diagnostics': [],
            'reps': ['R1', 'R2', 'R3', 'R4'], 'reader_seeds': [0, 1, 2, 3], 'decisive': {'CTRL': q, 'OBJ': plan['obj_mi']['config_id']}}
    from . import devreaders as DRD
    DRD.validate_spec(spec)
    out = BK.OUT / f'{bank_id}.json'
    if out.exists():
        raise PermissionError(f'{out.name} exists (write-once)')
    C.write_json(out, spec)
    for v in C.VERSIONS:
        C.write_json(C.RESULTS / 'lanes' / f'GRID_{bank_id}_V{v}.json',
                     [{'unit': f'DRF_GRID_{bank_id}_V{v}_V1', 'stage': 'readers', 'rss_gib': 3.0, 'max_wall_s': 10800,
                       'argv': ['TORCH', '-m', 'drf.devreaders', 'run', '--spec', str(out.relative_to(C.ROOT)), '--seed', str(v)]}])
    return out, q


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('write'); w.add_argument('--plan-id', required=True); w.add_argument('--card', required=True)
    w.add_argument('--revision', type=int, default=0); w.add_argument('--suffix', default='')
    w.add_argument('--fit-tag', default=''); w.add_argument('--revision-note', default=None)
    w.add_argument('--bank-id', required=True)
    rb = sub.add_parser('rob'); rb.add_argument('--plan-id', required=True); rb.add_argument('--bank-id', required=True)
    rb.add_argument('--decisions', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'rob':
        out, q = rob_bank(a.plan_id, a.bank_id, a.decisions)
        print(json.dumps({'bank': str(out.relative_to(C.ROOT)), 'decisive_ctrl_bracket': q}, indent=1))
        return 0
    note = json.loads(a.revision_note) if a.revision_note else None
    plan, out = write(a.plan_id, a.card, a.revision, a.suffix, note, a.fit_tag)
    spec = bank_and_lanes(plan, a.bank_id)
    print(json.dumps({'plan': str(out.relative_to(C.ROOT)), 'bank': str(spec.relative_to(C.ROOT)), 'paths': plan['paths'],
                      'alias_of': plan['alias_of'], 'notes': plan['ctrl_notes'], 'fineness': plan['fineness'],
                      'alias_records': plan['alias_records']}, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
