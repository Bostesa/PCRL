"""Label-free plan writer of the PCRL input-and-pair release campaign (coordinator): after the family fits, it reads the saved
configuration ledger and the fit records and writes, before any grid output is read,
- the registered reader bank `grid_specs/<bank>.json` (every saved configuration of both families, the shared twin, the references
  and the dual-design beside references);
- the claims plans `claims/PLAN_<family>_REG.json` (CAND, TWIN, the three control paths SBC / SPEND / RO, aliases by wire hash, the
  joint-design controls of a family whose card declares joint design);
- the data-rich reader scoring spec `rich/RICH_SPEC_<bank>.json`.

    python3 -m ipr.plan registered --bank IPR1_MAIN
    python3 -m ipr.plan revision --bank B --family F --rev R1 --card CARD --cand C [--sbc ...] [--spend ...] [--ro ...]

Derived from `ddr.plan` with three control paths and the D5 controls.
"""
from __future__ import annotations
import argparse
import json
import sys

from . import common as C
from . import release as RL

PATH_KEYS = ('SBC', 'SPEND', 'RO')
TWIN = ('CTL', 'CTL_TWIN')
BESIDE_REFS = {'DDR_TWIN': 'DDR:DA/DA_TWIN', 'DDR_PTC_R1': 'DDR:DB/DB_PTC_R1', 'DDR_ZTILT_R1': 'DDR:DB/DB_ZTILT_R1'}
# Filled from the committed cards before the registered fits (family -> its registered reading's configuration ids).
FAMILY_PLAN = {}


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


def _strength(fam, cfg, key):
    rec = {v: json.loads((RL.rdir(fam, cfg, v) / 'RELEASE.json').read_text()) for v in C.VERSIONS}
    return {str(v): rec[v].get(key) for v in C.VERSIONS}


def family_plan(fam, P, sv, members, plan_id, revision=0):
    tf, tc = P.get('twin', TWIN)
    if not _both(sv, fam, P['cand']):
        return {'plan_id': plan_id, 'family': fam, 'card': P['card'], 'revision': revision, 'cand': None, 'twin': tc,
                'members': {tc: f'{tf}/{tc}'}, 'paths': {k: [] for k in PATH_KEYS},
                'note': 'no candidate saved in both versions (NO_FEASIBLE_STRENGTH)'}
    tw = wires(tf, tc)
    path = {k: [c for c in P.get(k.lower(), ()) if c in members] for k in PATH_KEYS}
    alias_of, seen = {}, {tw[0]: 'TWIN'}
    for c in [x for k in PATH_KEYS for x in path[k]]:
        w = wires(fam, c)
        if w[0] in seen and all(w[v] == (tw if seen[w[0]] == 'TWIN' else wires(fam, seen[w[0]]))[v] for v in C.VERSIONS):
            alias_of[c] = seen[w[0]]
        else:
            seen[w[0]] = c
    cw = wires(fam, P['cand'])
    joint = None
    if P.get('joint'):
        j = P['joint']
        if all(j.get(k) in members for k in ('separate', 'income_first', 'occupation_first')):
            joint = {'outputs_changed': ['income', 'occupation'], 'jointly_designed': True,
                     **{k: j[k] for k in ('separate', 'income_first', 'occupation_first')}}
        else:
            joint = {'outputs_changed': ['income', 'occupation'], 'jointly_designed': True, 'separate': None, 'income_first': None,
                     'occupation_first': None, 'note': 'joint-design controls not saved in both versions'}
    plan_members = [tc, P['cand']] + [x for k in PATH_KEYS for x in path[k]]
    if joint:
        plan_members += [joint[k] for k in ('separate', 'income_first', 'occupation_first') if joint.get(k)]
    return {'plan_id': plan_id, 'family': fam, 'card': P['card'], 'revision': revision, 'cand': P['cand'], 'twin': tc,
            'members': {k: members[k] for k in dict.fromkeys(plan_members)},
            'paths': path, 'controls_not_run': {k: not path[k] for k in PATH_KEYS},
            'alias_of': alias_of, 'strengths': [{'config_id': P['cand'], 'strength': _strength(fam, P['cand'], P['strength_key'])}],
            'every_nonzero_strength_alias_of_twin': all(cw[v] == tw[v] for v in C.VERSIONS),
            'strength_differs_by_version': len({json.dumps(s) for s in _strength(fam, P['cand'], P['strength_key']).values()}) > 1,
            'beside': [c for c in P.get('beside', ()) if c in members], 'ablations': [c for c in P.get('ablations', ()) if c in members],
            'novelty': None, 'joint': joint, 'written_before_grid_outputs': True}


def registered(bank):
    sv = saved()
    members = {TWIN[1]: f'{TWIN[0]}/{TWIN[1]}'}
    cands = []
    for fam, P in FAMILY_PLAN.items():
        ids = (P['cand'],) + tuple(x for k in PATH_KEYS for x in P.get(k.lower(), ())) + tuple(P.get('ablations', ())) + \
            tuple(P.get('beside', ())) + tuple(v for k, v in (P.get('joint') or {}).items())
        for cfg in ids:
            if _both(sv, fam, cfg):
                members[cfg] = f'{fam}/{cfg}'
    plans = {fam: family_plan(fam, P, sv, members, f'{fam}_REG') for fam, P in FAMILY_PLAN.items()}
    for fam, P in FAMILY_PLAN.items():
        if plans[fam].get('cand'):
            cands += [P['cand']] + [c for c in tuple(P.get('ablations', ())) + tuple(P.get('beside', ())) if c in members]
    from . import banks as BK
    spec = BK.write(bank, 'registered IPR1 grid (R1, R2 x k0, k1, both versions): every saved configuration of families IA and IB '
                          '(the shared twin, candidates, the three control paths, joint-design controls, ablations), the nine '
                          'references + U0 / Ucal and the dual-design beside references; written from the configuration ledger '
                          'before any grid output', {**members, **BESIDE_REFS}, cands)
    out = {'bank': str(spec.relative_to(C.ROOT)), 'plans': {}}
    for fam, p in plans.items():
        path = C.RESULTS / 'claims' / f'PLAN_{fam}_REG.json'
        C.write_json_x(path, p)
        out['plans'][fam] = str(path.relative_to(C.ROOT))
    rich_members = {}
    for name, spec_m in members.items():
        fam, cfg = spec_m.split('/')
        rich_members[name] = {'family': fam, 'config': cfg, 'readers_of': fam, 'reader_key': cfg}
    comps = []
    for fam, P in FAMILY_PLAN.items():
        if plans[fam].get('cand'):
            for c in [TWIN[1]] + [x for k in PATH_KEYS for x in plans[fam]['paths'][k]]:
                if c in members:
                    comps.append([P['cand'], c])
    rspec = {'id': f'{bank}_RICH', 'members': rich_members, 'comparisons': comps,
             'note': 'each family member is scored with the readers trained in its own family fit unit; CTL_TWIN with each family\'s '
                     'readers trained on the twin out-of-fold release (scored once per family)'}
    rp = C.RESULTS / 'rich' / f'RICH_SPEC_{bank}.json'
    C.write_json_x(rp, rspec)
    out['rich_spec'] = str(rp.relative_to(C.ROOT))
    print(json.dumps(out, indent=1))
    return out


def revision(bank, family, rev, card, cand, sbc, spend, ro, strength_key, extra_members=()):
    """Bank + claims plan of a revision reading (label-free; written before its grid outputs): CAND, the shared twin, its control
    points of the three paths (reused or refitted, PROTOCOL 8), plus the references (cache hits)."""
    sv = saved()
    members = {TWIN[1]: f'{TWIN[0]}/{TWIN[1]}'}
    for cfg in (cand,) + tuple(sbc) + tuple(spend) + tuple(ro) + tuple(extra_members):
        if _both(sv, family, cfg):
            members[cfg] = f'{family}/{cfg}'
    from . import banks as BK
    spec = BK.write(bank, f'revision reading {family} {rev}: {cand}, the shared twin, its SBC, SPEND and RO points and the '
                          'references (written before any output of this bank)', {**members, **BESIDE_REFS},
                    [cand] if cand in members else [])
    P = {'card': card, 'cand': cand, 'sbc': tuple(sbc), 'spend': tuple(spend), 'ro': tuple(ro), 'strength_key': strength_key}
    plan = family_plan(family, P, sv, members, f'{family}_{rev}', revision=int(rev.lstrip('R')))
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
    v.add_argument('--ro', nargs='*', default=[]); v.add_argument('--strength-key', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'registered':
        registered(a.bank)
    else:
        revision(a.bank, a.family, a.rev, a.card, a.cand, a.sbc, a.spend, a.ro, a.strength_key)
    return 0


if __name__ == '__main__':
    sys.exit(main())
