"""Label-free plan writer of the PCRL strong-anchor release campaign (coordinator): after the family fits, it reads the saved
configuration ledger and the family fit records and writes, before any grid output is read,
- `paths/SHARED_ROLES.json`: the role members of the shared SPEND / RO bracket points (sar.roles; per version the CTL configuration
  each role names);
- the registered reader bank `grid_specs/<bank>.json` (every saved configuration of both families, the role members as hash-pinned
  'arrays' members, the shared twin, the references and the dual-design beside references);
- the claims plans `claims/PLAN_<family>_REG.json` (CAND, TWIN, the gating paths, aliases by wire hash, the card declarations
  operation_sex_blind / ro / joint, the per-version SBC construction and the MODEL_FIT fineness inputs);
- the data-rich reader scoring spec `rich/RICH_SPEC_<bank>.json`.

    python3 -m sar.plan registered --bank SAR1_MAIN
    python3 -m sar.plan revision --bank SAR1_R1 --name IA_R1      # a revision reading (REVISIONS; card IA_OSG_R1)

Derived from `ddr.plan` with three control paths, the shared CTL path points and the card declarations of the SAR1 rules.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from . import common as C
from . import release as RL

PATH_KEYS = ('SBC', 'SPEND', 'RO')
TWIN = ('CTL', 'CTL_TWIN')
BESIDE_REFS = {'DDR_TWIN': 'DDR:DA/DA_TWIN', 'DDR_PTC_R1': 'DDR:DB/DB_PTC_R1', 'DDR_ZTILT_R1': 'DDR:DB/DB_ZTILT_R1'}
CARDS = C.RESULTS / 'cards'
FAMILIES = {
    'IA': {'card': 'results/pcrl_strong_anchor_release_v1/cards/IA_SFG_v1.md', 'cand': 'IA_SFG', 'fit_dir': 'ia',
           'operation_sex_blind': False, 'ro': {'kind': 'GENERIC_ZTILT', 'operation_release_only': False},
           'sbc': ('IA_SBC_LO', 'IA_SBC_HI'), 'shared': {'SPEND': 'IA_SPD', 'RO': 'IA_RO'}, 'ablations': ('IA_BLIND',),
           'joint': None},
    'IB': {'card': 'results/pcrl_strong_anchor_release_v1/cards/IB_CSF_v1.md', 'cand': 'IB_CSF', 'fit_dir': 'ib',
           'operation_sex_blind': True, 'ro': None, 'sbc': (), 'shared': {'SPEND': 'IB_SPD'}, 'ablations': ('IB_CSFT', 'IB_IND'),
           'joint': {'outputs_changed': ['income', 'occupation'], 'jointly_designed': True, 'separate': 'IB_IND',
                     'collapsed': {'income_first': 'JOINT', 'occupation_first': 'JOINT'}}},
}


REVISIONS = {        # revision readings (PROTOCOL 7): one entry per revision card, its controls saved under the family
    'IA_R1': {'family': 'IA', 'revision': 1, 'card': 'results/pcrl_strong_anchor_release_v1/cards/IA_OSG_R1.md', 'cand': 'IA_OSG',
              'fit_record': 'ia/FIT_R1_V{v}.json', 'operation_sex_blind': False,
              'ro': {'kind': 'GENERIC_ZTILT', 'operation_release_only': False},
              'paths': {'SBC': ('IA_OSG_SBC_LO', 'IA_OSG_SBC_HI'), 'SPEND': ('IA_OSG_SPD_LO', 'IA_OSG_SPD_HI'),
                        'RO': ('IA_OSG_RO_LO', 'IA_OSG_RO_HI')},
              'ablations': ('IA_OSG_BLIND',), 'joint': None, 'readers_of': 'IA:R1', 'beside': {'IA_SFG': 'IA/IA_SFG'}},
}


def saved():
    out = {}
    for e in RL.configurations():
        out.setdefault((e['family'], e['config_id']), set()).add(int(e['version']))
    return out


def _both(sv, fam, cfg):
    return sv.get((fam, cfg)) == set(C.VERSIONS)


def fit_record(fam, v):
    p = C.RESULTS / FAMILIES[fam]['fit_dir'] / f'FIT_V{int(v)}.json'
    return json.loads(p.read_text()) if p.is_file() else None


def roles_from_fits():
    """{role name: {version: 'CTL/<config>'}} for every shared bracket point of every family (present in both versions only)."""
    roles = {}
    for fam, P in FAMILIES.items():
        for key, prefix in P['shared'].items():
            for side in ('LO', 'HI'):
                per = {}
                for v in C.VERSIONS:
                    r = fit_record(fam, v)
                    cfg = (((r or {}).get('shared_points') or {}).get(key) or {}).get(side)
                    if cfg and RL.exists('CTL', cfg, v):
                        per[str(v)] = f'CTL/{cfg}'
                if len(per) == len(C.VERSIONS):
                    roles[f'{prefix}_{side}'] = per
    return roles


def member_wires(spec, roles):
    """Per-version (income, occupation) MODEL_SELECT wire hashes of 'FAMILY/CONFIG' or a role name."""
    w = {}
    for v in C.VERSIONS:
        real = roles[spec][str(v)] if spec in roles else spec
        fam, cfg = real.split('/')
        rec = json.loads((RL.rdir(fam, cfg, v) / 'RELEASE.json').read_text())
        w[v] = (rec['wire_sha256']['MODEL_SELECT__income'], rec['wire_sha256']['MODEL_SELECT__occupation'])
    return w


def _strength(fam, cfg):
    out = {}
    for v in C.VERSIONS:
        rec = json.loads((RL.rdir(fam, cfg, v) / 'RELEASE.json').read_text())
        out[str(v)] = rec.get('strength', rec.get('k', rec.get('pi')))
    return out


def family_plan(fam, P, sv, members, roles, plan_id, revision=0):
    tf, tc = TWIN
    base = {'plan_id': plan_id, 'family': fam, 'card': P['card'], 'revision': revision, 'twin': tc,
            'operation_sex_blind': P['operation_sex_blind'], 'ro': P['ro'], 'novelty': None, 'written_before_grid_outputs': True}
    if not _both(sv, fam, P['cand']):
        return {**base, 'cand': None, 'members': {tc: f'{tf}/{tc}'},
                'paths': {k: ([] if k != 'SBC' or not P['operation_sex_blind'] else None) for k in PATH_KEYS},
                'note': 'no candidate saved in both versions (NO_FEASIBLE_STRENGTH)'}
    paths = {'SBC': None if P['operation_sex_blind'] else [c for c in P['sbc'] if c in members],
             'SPEND': [n for n in (f"{P['shared']['SPEND']}_LO", f"{P['shared']['SPEND']}_HI") if n in members],
             'RO': ([n for n in (f"{P['shared']['RO']}_LO", f"{P['shared']['RO']}_HI") if n in members]
                    if 'RO' in P['shared'] else None)}
    tw = member_wires(f'{tf}/{tc}', roles)
    alias_of, seen = {}, {tw[0]: 'TWIN'}
    for c in [x for k in PATH_KEYS for x in (paths[k] or [])]:
        w = member_wires(members[c] if c not in roles else c, roles)
        if w[0] in seen and all(w[v] == (tw if seen[w[0]] == 'TWIN' else member_wires(members[seen[w[0]]] if seen[w[0]] not in roles
                                                                                     else seen[w[0]], roles))[v] for v in C.VERSIONS):
            alias_of[c] = seen[w[0]]
        else:
            seen[w[0]] = c
    cw = member_wires(f'{fam}/{P["cand"]}', roles)
    sbc_con, fineness = {}, {}
    for v in C.VERSIONS:
        r = fit_record(fam, v) or {}
        if not P['operation_sex_blind']:
            sbc_con[str(v)] = r.get('sbc_construction')
        for key, inp in (r.get('fineness_inputs') or {}).items():
            fineness.setdefault(key, {})[str(v)] = inp
    plan_members = [tc, P['cand']] + [x for k in PATH_KEYS for x in (paths[k] or [])]
    joint = dict(P['joint']) if P['joint'] else None
    if joint and joint.get('separate'):
        plan_members.append(joint['separate'])
    return {**base, 'cand': P['cand'],
            'members': {k: members[k] for k in dict.fromkeys(plan_members)},
            'paths': paths, 'controls_not_run': {k: False for k in PATH_KEYS},
            'alias_of': alias_of, 'strengths': [{'config_id': P['cand'], 'strength': _strength(fam, P['cand'])}],
            'every_nonzero_strength_alias_of_twin': all(cw[v] == tw[v] for v in C.VERSIONS),
            'sbc_construction': sbc_con or None, 'fineness_inputs': fineness,
            'ablations': [c for c in P['ablations'] if c in members], 'joint': joint}


def revision_plan(name, sv, plan_id):
    """The claims plan of a revision reading (REVISIONS[name]): its candidate, the shared twin and its own control points (each
    present in both versions), aliases by wire hash, the card declarations, the per-version SBC construction and fineness inputs."""
    R = REVISIONS[name]
    fam = R['family']
    tf, tc = TWIN
    base = {'plan_id': plan_id, 'family': fam, 'card': R['card'], 'revision': R['revision'], 'twin': tc,
            'operation_sex_blind': R['operation_sex_blind'], 'ro': R['ro'], 'novelty': None, 'written_before_grid_outputs': True}
    if not _both(sv, fam, R['cand']):
        return {**base, 'cand': None, 'members': {tc: f'{tf}/{tc}'},
                'paths': {k: ([] if k != 'SBC' or not R['operation_sex_blind'] else None) for k in PATH_KEYS},
                'note': 'no candidate saved in both versions (NO_FEASIBLE_STRENGTH)'}
    members = {tc: f'{tf}/{tc}', R['cand']: f"{fam}/{R['cand']}"}
    paths = {}
    for key in PATH_KEYS:
        cfgs = R['paths'].get(key)
        if cfgs is None or (key == 'SBC' and R['operation_sex_blind']):
            paths[key] = None
            continue
        paths[key] = [c for c in cfgs if _both(sv, fam, c)]
        for c in paths[key]:
            members[c] = f'{fam}/{c}'
    tw = member_wires(f'{tf}/{tc}', {})
    alias_of, seen = {}, {tw[0]: 'TWIN'}
    for c in [x for k in PATH_KEYS for x in (paths[k] or [])]:
        w = member_wires(members[c], {})
        ref = seen.get(w[0])
        if ref is not None and all(w[v] == (tw if ref == 'TWIN' else member_wires(members[ref], {}))[v] for v in C.VERSIONS):
            alias_of[c] = ref
        else:
            seen[w[0]] = c
    cw = member_wires(members[R['cand']], {})
    sbc_con, fineness = {}, {}
    for v in C.VERSIONS:
        p = C.RESULTS / R['fit_record'].format(v=int(v))
        r = json.loads(p.read_text()) if p.is_file() else {}
        if not R['operation_sex_blind']:
            sbc_con[str(v)] = r.get('sbc_construction')
        for key, inp in (r.get('fineness_inputs') or {}).items():
            fineness.setdefault(key, {})[str(v)] = inp
    return {**base, 'cand': R['cand'], 'members': members, 'paths': paths, 'controls_not_run': {k: False for k in PATH_KEYS},
            'alias_of': alias_of, 'strengths': [{'config_id': R['cand'], 'strength': _strength(fam, R['cand'])}],
            'every_nonzero_strength_alias_of_twin': all(cw[v] == tw[v] for v in C.VERSIONS),
            'sbc_construction': sbc_con or None, 'fineness_inputs': fineness,
            'ablations': [c for c in R['ablations'] if _both(sv, fam, c)], 'joint': R['joint']}


def revision(bank, name):
    """Bank, claims plan and data-rich reader spec of a revision reading, written from the configuration ledger and its fit records
    before any of its grid output: the registered cells (R1, R2 x k0, k1; the SAR1_MAIN seeds)."""
    sv = saved()
    R = REVISIONS[name]
    fam = R['family']
    plan = revision_plan(name, sv, f"{fam}_R{R['revision']}")
    members = dict(plan['members'])
    for c in plan.get('ablations') or []:
        members[c] = f'{fam}/{c}'
    members.update(R['beside'])
    cands = ([plan['cand']] + list(plan.get('ablations') or [])) if plan.get('cand') else []
    from . import banks as BK
    spec = BK.write(bank, f"{name} grid (R1, R2 x k0, k1, both versions; the registered cells): the revision's candidate (card "
                          f"{Path(R['card']).name}) with its own SBC, SPEND and RO bracket points and its ablation, the shared twin, the "
                          f"nine references + U0 / Ucal, the registered candidate and the dual-design references beside; written from the "
                          f"configuration ledger and the fit records before any of its grid output", {**members, **BESIDE_REFS}, cands)
    path = C.RESULTS / 'claims' / f"PLAN_{fam}_R{R['revision']}.json"
    C.write_json_x(path, plan)
    rich_members, comps = {}, []
    if plan.get('cand'):
        names = [TWIN[1], plan['cand']] + [x for k in PATH_KEYS for x in (plan['paths'][k] or [])] + list(plan['ablations'])
        for nm in dict.fromkeys(names):
            f_, c_ = members[nm].split('/')
            rich_members[f'{nm}@{name}'] = {'family': f_, 'config': c_, 'readers_of': R['readers_of'], 'reader_key': c_}
        for nm in dict.fromkeys(names):
            if nm != plan['cand']:
                comps.append([f"{plan['cand']}@{name}", f'{nm}@{name}'])
    rspec = {'id': f'{bank}_RICH', 'members': rich_members, 'comparisons': comps,
             'note': "each member is scored with the readers trained in the revision's fit unit on its MODEL_FIT out-of-fold releases"}
    rpath = C.RESULTS / 'rich' / f'RICH_SPEC_{bank}.json'
    C.write_json_x(rpath, rspec)
    out = {'bank': str(spec.relative_to(C.ROOT)), 'plan': str(path.relative_to(C.ROOT)), 'rich_spec': str(rpath.relative_to(C.ROOT)),
           'cand': plan.get('cand'), 'paths': plan.get('paths'), 'alias_of': plan.get('alias_of')}
    print(json.dumps(out, indent=1))
    return out


def registered(bank):
    sv = saved()
    roles = roles_from_fits()
    rp = C.RESULTS / 'paths' / 'SHARED_ROLES.json'
    C.write_json_x(rp, {'schema': 'SAR1-shared-roles-v1', 'utc': C.now(), 'roles': roles,
                        'note': 'role member -> per-version shared CTL configuration (card CTL_PATHS_v1), from the family fit records; '
                                'written before any grid output'})
    members = {TWIN[1]: f'{TWIN[0]}/{TWIN[1]}'}
    cands = []
    for fam, P in FAMILIES.items():
        for cfg in (P['cand'],) + tuple(P['sbc']) + tuple(P['ablations']):
            if _both(sv, fam, cfg):
                members[cfg] = f'{fam}/{cfg}'
    for name in roles:
        members[name] = f'ROLE:{name}'
    plans = {fam: family_plan(fam, P, sv, members, roles, f'{fam}_REG') for fam, P in FAMILIES.items()}
    for fam, P in FAMILIES.items():
        if plans[fam].get('cand'):
            cands += [P['cand']] + [c for c in P['ablations'] if c in members]
    from . import banks as BK
    from . import roles as RO
    spec_members = {}
    for name, spec in members.items():
        spec_members[name] = RO.array_member(name) if spec.startswith('ROLE:') else spec
    spec = BK.write(bank, 'registered SAR1 grid (R1, R2 x k0, k1, both versions): every saved configuration of families IA and IB '
                          '(the shared twin, candidates, control paths with the shared CTL points as role members, ablations), the '
                          'nine references + U0 / Ucal and the dual-design beside references; written from the configuration ledger '
                          'and the fit records before any grid output', {**spec_members, **BESIDE_REFS}, cands)
    out = {'bank': str(spec.relative_to(C.ROOT)), 'roles': str(rp.relative_to(C.ROOT)), 'plans': {}}
    for fam, p_ in plans.items():
        path = C.RESULTS / 'claims' / f'PLAN_{fam}_REG.json'
        C.write_json_x(path, p_)
        out['plans'][fam] = str(path.relative_to(C.ROOT))
    rich_members, comps = {}, []
    for fam, P in FAMILIES.items():
        if not plans[fam].get('cand'):
            continue
        fam_members = [TWIN[1], P['cand']] + [x for k in PATH_KEYS for x in (plans[fam]['paths'][k] or [])] + list(plans[fam]['ablations'])
        for name in dict.fromkeys(fam_members):
            key = f'{name}@{fam}'
            if name in roles:
                byv = {}
                for v in C.VERSIONS:
                    f_, c_ = roles[name][str(v)].split('/')
                    byv[str(v)] = {'family': f_, 'config': c_, 'reader_key': c_}
                rich_members[key] = {'readers_of': fam, 'by_version': byv}
            else:
                f_, c_ = members[name].split('/')
                rich_members[key] = {'family': f_, 'config': c_, 'readers_of': fam, 'reader_key': c_}
        for c in [TWIN[1]] + [x for k in PATH_KEYS for x in (plans[fam]['paths'][k] or [])] + list(plans[fam]['ablations']):
            comps.append([f"{P['cand']}@{fam}", f'{c}@{fam}'])
    rspec = {'id': f'{bank}_RICH', 'members': rich_members, 'comparisons': comps,
             'note': "each member is scored with the readers trained in its family's fit unit on its MODEL_FIT out-of-fold releases; a "
                     "shared CTL point is keyed per version (by_version)"}
    rpath = C.RESULTS / 'rich' / f'RICH_SPEC_{bank}.json'
    C.write_json_x(rpath, rspec)
    out['rich_spec'] = str(rpath.relative_to(C.ROOT))
    print(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('registered'); r.add_argument('--bank', required=True)
    rv = sub.add_parser('revision'); rv.add_argument('--bank', required=True); rv.add_argument('--name', required=True, choices=sorted(REVISIONS))
    a = ap.parse_args(argv)
    if a.cmd == 'revision':
        revision(a.bank, a.name)
    else:
        registered(a.bank)
    return 0


if __name__ == '__main__':
    sys.exit(main())
