"""Repair-stage plan helpers (label-free): the post-freeze unit lane and the R reader bank.

    python3 -m cts.rstage bank --bank-id ID [--strengths 0.2,0.4,0.6,0.8,1.0] [--trackr]
    python3 -m cts.rstage lane --out LANE.json [--trackr]

The R bank (stage R: R0 -> seed 0, R1 -> seed 1, H3 income) holds, for the matched frontier and the rule comparison:
the unchanged witness W; REFIT_PAIR, REFIT_LOC and REFIT_CONT at the fixed strengths (common correction magnitudes) and at every
nonzero strength a frozen rule selected (full deployment) on R0 / R1; the compression twins matched to REFIT_PAIR at those
strengths; and the six required comparators (frozen; cached). Every member is both a candidate and a comparator, so all paired
differences exist. With --trackr the Track R complete releases (refitted income H3R + REFIT_PAIR_TR) are added.
"""
from __future__ import annotations
import argparse
import json
import sys

from . import common as C
from . import trackf as TF
from . import releases as REL

RPARTS = ('R0', 'R1')
RULE_KEYS = ('OLD-B80|full', 'OLD-B50|full', 'SIMPLE|full', 'UNCERTAINTY|full')


def nonzero_rule_names(did):
    names = []
    for rk in RULE_KEYS:
        if all(float(json.loads((TF.pdir(P) / 'SELECT' / f'{did}.json').read_text())['selected'][rk]) > 0 for P in RPARTS):
            names.append(f'{did}@{rk}')
    return names


def selected_recipe():
    """(direction id, 'RULE|dep') of the frozen D-stage recipe (cts.dstage select), or None (NO_RECIPE / not selected)."""
    p = C.RESULTS / 'diagnosis' / 'RECIPE_SELECTION.json'
    sel = json.loads(p.read_text()).get('selected') if p.is_file() else None
    if not sel:
        return None
    did, key = sel.split('@')
    return did, key


def recipe_directions():
    """Directions the R stage must evaluate for the selected recipe beyond the REFIT arms (e.g. FROZEN_PAIR + FROZEN_CONT)."""
    rs = selected_recipe()
    if rs is None:
        return []
    did, _ = rs
    return [d for d in (did, f"{did.split('_')[0]}_CONT") if not d.startswith('REFIT_')]


def members(strengths, trackr=False):
    m = ['W']
    rs = selected_recipe()
    if rs is not None:                       # the primary endpoint (E3): the selected recipe and its matched CONT, any routing / deployment
        did, key = rs
        m += [n for n in (f'{did}@{key}', f"{did.split('_')[0]}_CONT@{key}") if n not in m]
    for arm in ('PAIR', 'LOC', 'CONT'):
        did = f'REFIT_{arm}'
        m += [f'{did}@s{s}' for s in strengths]
        m += [n for n in nonzero_rule_names(did) if n not in m]
    comps = []
    for s in strengths:
        comps.append(f'COMP:REFIT_PAIR@s{s}')
    for n in nonzero_rule_names('REFIT_PAIR'):
        comps.append(f'COMP:REFIT_PAIR@{n.split("@")[1]}')
    return m, comps


def bank(bank_id, strengths, trackr=False):
    m, comps = members(strengths, trackr)
    have = []
    for c in comps:
        did, rk = c[5:].split('@')
        ok = all((TF.pdir(P) / 'COMP' / f'COMP__{did}__{rk.replace("|", "_")}' / 'COMP.json').is_file() for P in RPARTS)
        if ok:
            have.append(c)
    names = m + have
    p, spec = REL.bank('R', bank_id, names, names, references=True,
                       purpose='R-stage development readers: matched frontier (common strengths), rule selections, compression twins, '
                               'required comparators', reps=('R1', 'R2'))
    print(json.dumps({'spec': str(p.relative_to(C.ROOT)), 'members': len(spec['members']), 'missing_comp': sorted(set(comps) - set(have))}))
    return spec


def lane(out, strengths, trackr=False):
    units = []
    arms = ['PAIR', 'LOC', 'CONT']
    if trackr and all((TF.pdir(P) / 'DIR' / 'REFIT_PAIR_TR' / 'DIR.json').is_file() for P in RPARTS):
        arms.append('PAIR_TR')
    dids = [f'REFIT_{arm}' for arm in arms] + recipe_directions()     # + the selected recipe's FROZEN directions, if any
    for P in RPARTS:
        for did in dids:
            units.append({'unit': f'CTS_{P}_REALIZED_{did}_V1', 'stage': 'repair_check', 'rss_gib': 1.5,
                          'argv': ['PY', '-m', 'cts.select', 'check', '--partition', P, '--direction', did]})
            units.append({'unit': f'CTS_{P}_LADDER_CHECK_{did}_V1', 'stage': 'repair_check', 'rss_gib': 1.5,
                          'argv': ['PY', '-m', 'cts.trackf', 'ladder', '--partition', P, '--direction', did, '--check']})
            for dep in ('full', 'ens'):
                units.append({'unit': f'CTS_{P}_QA_{did}_{dep}_V1', 'stage': 'repair_check', 'rss_gib': 1.5,
                              'argv': ['PY', '-m', 'cts.questiona', 'run', '--partition', P, '--direction', did, '--deployment', dep]})
        for s in strengths:
            units.append({'unit': f'CTS_{P}_COMP_REFIT_PAIR_s{s}_V1', 'stage': 'repair_fit', 'rss_gib': 1.5,
                          'argv': ['PY', '-m', 'cts.compression', 'fit', '--partition', P, '--direction', 'REFIT_PAIR', '--rule', f's{s}']})
        for n in nonzero_rule_names('REFIT_PAIR'):
            rk = n.split('@')[1]
            units.append({'unit': f'CTS_{P}_COMP_REFIT_PAIR_{rk.replace("|", "_")}_V1', 'stage': 'repair_fit', 'rss_gib': 1.5,
                          'argv': ['PY', '-m', 'cts.compression', 'fit', '--partition', P, '--direction', 'REFIT_PAIR', '--rule', rk]})
        units.append({'unit': f'CTS_{P}_COMP_REALIZED_V1', 'stage': 'repair_check', 'rss_gib': 1.5,
                      'argv': ['PY', '-m', 'cts.compression', 'realized', '--partition', P]})
    C.write_json(out, units)
    print(json.dumps({'units': len(units), 'out': str(out)}))
    return units


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    x = sub.add_parser('bank'); x.add_argument('--bank-id', required=True); x.add_argument('--strengths', default='0.2,0.4,0.6,0.8,1.0')
    x.add_argument('--trackr', action='store_true')
    y = sub.add_parser('lane'); y.add_argument('--out', required=True); y.add_argument('--strengths', default='0.2,0.4,0.6,0.8,1.0')
    y.add_argument('--trackr', action='store_true')
    a = ap.parse_args(argv)
    st = tuple(float(s) for s in a.strengths.split(','))
    if a.cmd == 'bank':
        bank(a.bank_id, st, a.trackr)
    else:
        lane(a.out, st, a.trackr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
