"""Label-free plan writer of the PCRL dual-design release campaign (coordinator): after the family fits, it reads the saved
configuration ledger and the fit records and writes, before any grid output is read,
- the registered reader bank `grid_specs/<bank>.json` (every saved configuration of both families + the references);
- the claims plans `claims/PLAN_<family>_REG.json` (CAND, TWIN, the two control paths, aliases by wire hash);
- the data-rich reader scoring spec `rich/RICH_SPEC_<id>.json`.

    python3 -m ddr.plan registered --bank DDR1_MAIN
"""
from __future__ import annotations
import argparse
import json
import sys

from . import common as C
from . import release as RL

FAMILY_PLAN = {
    'DA': {'card': 'results/pcrl_dual_design_release_v1/cards/DA_STD_v1.md', 'cand': 'DA_STD', 'twin': ('DA', 'DA_TWIN'),
           'sbc': ('DA_SBC_LO', 'DA_SBC_HI'), 'spend': ('DA_SPD_LO', 'DA_SPD_HI'), 'ablations': ('DA_COMP', 'DA_STD_OCC'),
           'beside': (), 'strength_key': 'lambda'},
    'DB': {'card': 'results/pcrl_dual_design_release_v1/cards/DB_PTC_v1.md', 'cand': 'DB_PTC', 'twin': ('DA', 'DA_TWIN'),
           'sbc': ('DB_SBC_LO', 'DB_SBC_HI'), 'spend': ('DB_SPD_LO', 'DB_SPD_HI'), 'ablations': ('DB_PTC_OCC',),
           'beside': ('DB_ZTILT', 'DB_SHUF'), 'strength_key': 'alpha'},
}


def saved():
    out = {}
    for e in RL.configurations():
        out.setdefault((e['family'], e['config_id']), set()).add(int(e['version']))
    return out


def _both(sv, fam, cfg):
    return sv.get((fam, cfg)) == set(C.VERSIONS)


def wires(fam, cfg):
    w = {}
    for v in C.VERSIONS:
        rec = json.loads((RL.rdir(fam, cfg, v) / 'RELEASE.json').read_text())
        w[v] = (rec['wire_sha256']['MODEL_SELECT__income'], rec['wire_sha256']['MODEL_SELECT__occupation'])
    return w


def registered(bank):
    sv = saved()
    members, plans, cands = {}, {}, []
    for fam, P in FAMILY_PLAN.items():
        tf, tc = P['twin']
        members[tc] = f'{tf}/{tc}'
        for cfg in (P['cand'],) + P['sbc'] + P['spend'] + P['ablations'] + P['beside']:
            if _both(sv, fam, cfg):
                members[cfg] = f'{fam}/{cfg}'
    for fam, P in FAMILY_PLAN.items():
        tf, tc = P['twin']
        if not _both(sv, fam, P['cand']):
            plans[fam] = {'plan_id': f'{fam}_REG', 'family': fam, 'card': P['card'], 'revision': 0, 'cand': None, 'twin': tc,
                          'members': {tc: f'{tf}/{tc}'}, 'paths': {'SBC': [], 'SPEND': []},
                          'note': 'no candidate saved in both versions (NO_FEASIBLE_STRENGTH)'}
            continue
        cands += [P['cand']] + [c for c in P['ablations'] + P['beside'] if c in members]
        tw = wires(tf, tc)
        alias_of, path = {}, {}
        for key, ids in (('SBC', P['sbc']), ('SPEND', P['spend'])):
            path[key] = [c for c in ids if c in members]
        seen = {tw[0]: 'TWIN'}
        for c in path['SBC'] + path['SPEND']:
            w = wires(fam, c)
            if w[0] in seen and all(w[v] == (tw if seen[w[0]] == 'TWIN' else wires(fam, seen[w[0]]))[v] for v in C.VERSIONS):
                alias_of[c] = seen[w[0]]
            else:
                seen[w[0]] = c
        strengths = []
        rec = {v: json.loads((RL.rdir(fam, P['cand'], v) / 'RELEASE.json').read_text()) for v in C.VERSIONS}
        strengths.append({'config_id': P['cand'], 'strength': {str(v): rec[v].get(P['strength_key']) for v in C.VERSIONS}})
        cw = wires(fam, P['cand'])
        plans[fam] = {'plan_id': f'{fam}_REG', 'family': fam, 'card': P['card'], 'revision': 0, 'cand': P['cand'], 'twin': tc,
                      'members': {k: members[k] for k in [tc, P['cand']] + path['SBC'] + path['SPEND']},
                      'paths': path, 'controls_not_run': {'SBC': not path['SBC'], 'SPEND': not path['SPEND']},
                      'alias_of': alias_of, 'strengths': strengths,
                      'every_nonzero_strength_alias_of_twin': all(cw[v] == tw[v] for v in C.VERSIONS),
                      'strength_differs_by_version': len({str(s) for s in strengths[0]['strength'].values()}) > 1,
                      'beside': [c for c in P['beside'] if c in members], 'ablations': [c for c in P['ablations'] if c in members],
                      'novelty': None, 'joint': None,
                      'written_before_grid_outputs': True}
    from . import banks as BK
    spec = BK.write(bank, 'registered DDR1 grid (R1, R2 x k0, k1, both versions): every saved configuration of families DA and DB '
                          '(twin, candidates, both control paths, ZTILT / SHUF, ablations) and the nine references + U0 / Ucal; '
                          'written from the configuration ledger before any grid output',
                    members, cands)
    out = {'bank': str(spec.relative_to(C.ROOT)), 'plans': {}}
    for fam, p in plans.items():
        path = C.RESULTS / 'claims' / f'PLAN_{fam}_REG.json'
        C.write_json_x(path, p)
        out['plans'][fam] = str(path.relative_to(C.ROOT))
    rich_members = {}
    for name, spec_m in members.items():
        fam, cfg = spec_m.split('/')
        rich_members[name] = {'family': fam, 'config': cfg, 'readers_of': fam if cfg != 'DA_TWIN' else 'DA',
                              'reader_key': cfg}
    comps = []
    for fam, P in FAMILY_PLAN.items():
        if plans[fam].get('cand'):
            for c in ['DA_TWIN'] + list(plans[fam]['paths']['SBC']) + list(plans[fam]['paths']['SPEND']) + list(P['beside']):
                if c in members:
                    comps.append([P['cand'], c])
    rspec = {'id': f'{bank}_RICH', 'members': rich_members, 'comparisons': comps,
             'note': 'DB members are scored with DB-trained readers; DA_TWIN and DA members with DA-trained readers (both families '
                     'trained their readers on the same twin out-of-fold release for TWIN)'}
    rp = C.RESULTS / 'rich' / f'RICH_SPEC_{bank}.json'
    C.write_json_x(rp, rspec)
    out['rich_spec'] = str(rp.relative_to(C.ROOT))
    print(json.dumps(out, indent=1))
    return out


def revision(bank, family, rev, card, cand, sbc, spend, extra_members=()):
    """Bank + claims plan of a revision reading (label-free; written before its grid outputs): CAND, the shared twin, the refitted
    SBC points and the reused SPEND points of the family (PROTOCOL 8), plus the references (cache hits)."""
    sv = saved()
    members = {'DA_TWIN': 'DA/DA_TWIN'}
    for cfg in (cand,) + tuple(sbc) + tuple(spend) + tuple(extra_members):
        if _both(sv, family, cfg):
            members[cfg] = f'{family}/{cfg}'
    from . import banks as BK
    spec = BK.write(bank, f'revision reading {family} {rev}: {cand}, the shared twin, its SBC and SPEND points and the references '
                          '(written before any output of this bank)', members, [cand] if cand in members else [])
    path = {'SBC': [c for c in sbc if c in members], 'SPEND': [c for c in spend if c in members]}
    tw = wires('DA', 'DA_TWIN')
    cw = wires(family, cand) if cand in members else None
    rec0 = {v: json.loads((RL.rdir(family, cand, v) / 'RELEASE.json').read_text()) for v in C.VERSIONS} if cand in members else {}
    plan = {'plan_id': f'{family}_{rev}', 'family': family, 'card': card, 'revision': int(rev.lstrip('R')),
            'cand': cand if cand in members else None, 'twin': 'DA_TWIN',
            'members': {k: members[k] for k in ['DA_TWIN'] + ([cand] if cand in members else []) + path['SBC'] + path['SPEND']},
            'paths': path, 'controls_not_run': {'SBC': not path['SBC'], 'SPEND': not path['SPEND']}, 'alias_of': {},
            'strengths': [{'config_id': cand, 'strength': {str(v): (rec0[v].get('lambda') if family == 'DA' else
                                                                    [rec0[v].get('alpha_inc'), rec0[v].get('alpha_occ')])
                                                             for v in C.VERSIONS}}] if rec0 else [],
            'every_nonzero_strength_alias_of_twin': bool(cw) and all(cw[v] == tw[v] for v in C.VERSIONS),
            'novelty': None, 'joint': None, 'written_before_grid_outputs': True}
    out = C.RESULTS / 'claims' / f'PLAN_{family}_{rev}.json'
    C.write_json_x(out, plan)
    print(json.dumps({'bank': str(spec.relative_to(C.ROOT)), 'plan': str(out.relative_to(C.ROOT)), 'members': sorted(members)}, indent=1))
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('registered'); r.add_argument('--bank', required=True)
    v = sub.add_parser('revision'); v.add_argument('--bank', required=True); v.add_argument('--family', required=True)
    v.add_argument('--rev', required=True); v.add_argument('--card', required=True); v.add_argument('--cand', required=True)
    v.add_argument('--sbc', nargs='*', default=[]); v.add_argument('--spend', nargs='*', default=[])
    a = ap.parse_args(argv)
    if a.cmd == 'registered':
        registered(a.bank)
    else:
        revision(a.bank, a.family, a.rev, a.card, a.cand, a.sbc, a.spend)
    return 0


if __name__ == '__main__':
    sys.exit(main())
