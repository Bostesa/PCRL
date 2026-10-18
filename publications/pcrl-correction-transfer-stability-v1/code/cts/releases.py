"""Materialize selected Track F releases (MODEL_SELECT probability arrays, per associated seed) and build reader bank specs
(PCRL correction-transfer campaign). Label-free: strengths come from the saved selections (cts.select), arrays from the saved
witness / direction / compression records; nothing here reads a label.

A release name is resolved per partition P:
    W                              the unchanged witness (s = 0)
    <DIR>@<RULE>                   direction DIR (e.g. REFIT_PAIR) at the strength RULE selected for DIR on P
    <DIR>@<RULE>:<SRC_DIR>         direction DIR at the strength RULE selected for SRC_DIR (e.g. the continuation twin at the PAIR
                                   strength: REFIT_CONT@UNCERTAINTY:REFIT_PAIR)
    <DIR>@s<x>                     direction DIR at the fixed ladder strength x (frontier points)
    COMP:<DIR>@<RULE>              the compression code matched to DIR at RULE (cts.compression)
A stage-pair release joins the seed-0 partition (D0 / R0) and the seed-1 partition (D1 / R1); the income wire is H3's frozen
income (Track F).

    python3 -m cts.releases bank --stage D|R --bank-id ID --candidates N1,N2 --comparators M1,M2 [--references]
"""
from __future__ import annotations
import argparse
import json
import sys

import numpy as np

from . import common as C
from . import trackf as TF

STAGES = {'D': ('D0', 'D1'), 'R': ('R0', 'R1')}
REQUIRED = ('NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ')


def strength(P, did, rule):
    srec = json.loads((TF.pdir(P) / 'SELECT' / f'{did}.json').read_text())
    return float(srec['selected'][rule])


def resolve(P, name):
    """(MODEL_SELECT occupation array, description) of a release name on partition P."""
    wrec, W, O = TF.load_witness(P)
    if name == 'W':
        return W['dep__MODEL_SELECT'], {'kind': 'witness', 'strength': 0.0}
    if name.startswith('COMP:'):
        did, rule = name[5:].split('@')
        d = TF.pdir(P) / 'COMP' / f'COMP__{did}__{rule.replace("|", "_")}'
        rec = json.loads((d / 'COMP.json').read_text())
        if C.sha(d / 'CODE.npz') != rec['npz_sha256']:
            raise PermissionError('compression record differs')
        with np.load(d / 'CODE.npz') as z:
            return z['rel__MODEL_SELECT'].copy(), {'kind': 'compression', 'K': rec['K'], 'matched_to': rec['matched_to']}
    did, spec = name.split('@')
    drec, T, D = TF.load_direction(P, did)
    if spec.startswith('s'):
        sv, _, dep = spec[1:].partition('|')
        s, src, dep = float(sv), None, dep or 'full'
    else:
        rule, _, src = spec.partition(':')
        dep = rule.split('|')[1] if '|' in rule else 'full'
        s = strength(P, src or did, rule if '|' in rule else f'{rule}|full')
    if dep == 'ens':
        p = TF.ensemble_at(W['fold__MODEL_SELECT'], D['folds__MODEL_SELECT'], s)
    else:
        p = TF.release_at(W['dep__MODEL_SELECT'], D['full__MODEL_SELECT'], s)
    return p, {'kind': 'direction', 'direction': did, 'strength': s, 'strength_source': src or did, 'deployment': dep}


def materialize(stage, name, parts=None):
    parts = tuple(parts or STAGES[stage])
    suffix = '' if parts == STAGES[stage] else '__' + '_'.join(parts)
    out = C.RUNS / 'RELEASES' / stage / (name.replace(':', '__').replace('@', '_AT_').replace('|', '_') + suffix + '.npz')
    arrays, desc = {}, {}
    for P in parts:
        s = str(TF.seed_of(P))
        arrays[f'occ_S{s}'], desc[P] = resolve(P, name)
    if out.exists():
        with np.load(out) as z:
            for k, v in arrays.items():
                if not np.array_equal(z[k], v):
                    raise PermissionError(f'{out.name}: existing file differs from the recomputed release')
    else:
        TF._save_npz(out, **arrays)
    return out, desc


def _trackr_income(P, tag):
    """Track R refitted income (H3R or the task-only twin HCR) on MODEL_SELECT: path, key and file hash for a bank member."""
    d = C.RUNS / 'TRACK_R' / P / 'INCOME'
    rec = json.loads((d / 'INCOME.json').read_text())
    p = d / 'RELEASE.npz'
    if C.sha(p) != rec['files_sha256']['RELEASE.npz']:
        raise PermissionError('Track R release differs from its record')
    return {'path': str(p.relative_to(C.ROOT)), 'key': f'{tag}__MODEL_SELECT__income', 'sha256': C.sha(p),
            'row_id_key': 'row_id__MODEL_SELECT'}


def member(stage, name, parts=None):
    """A bank member. Names 'H3R:<release>' / 'HCR:<release>' pair the Track R refitted income (or its task-only twin) of each
    repair partition with the occupation release <release>; all other names use H3's frozen income."""
    parts = tuple(parts or STAGES[stage])
    income_tag, occ_name = (name.split(':', 1) if name.split(':', 1)[0] in ('H3R', 'HCR') else (None, name))
    out, desc = materialize(stage, occ_name, parts)
    sha = C.sha(out)
    rel = str(out.relative_to(C.ROOT))
    seeds = {str(TF.seed_of(P)): {'income': ({'frozen_anchor': 'H3'} if income_tag is None else _trackr_income(P, income_tag)),
                                  'occupation': {'path': rel, 'key': f'occ_S{TF.seed_of(P)}', 'sha256': sha}} for P in parts}
    return {'type': 'arrays', 'seeds': seeds, 'description': {**desc, 'income': income_tag or 'H3 (frozen)'}}


def bank(stage, bank_id, candidates, comparators, references=False, purpose='', reps=('R1', 'R2'), fractions=(1.0,), parts=None):
    members = {}
    for n in list(candidates) + list(comparators):
        if n in REQUIRED:
            members[n] = {'type': 'reference', 'name': n}
        else:
            members[n] = member(stage, n, parts)
    comps = list(comparators) + ([r for r in REQUIRED if r not in comparators] if references else [])
    for r in comps:
        if r in REQUIRED and r not in members:
            members[r] = {'type': 'reference', 'name': r}
    spec = {'bank_id': bank_id, 'purpose': purpose or f'{stage}-stage development readers', 'members': members,
            'candidates': list(candidates), 'comparators': comps, 'fractions': list(fractions), 'reps': list(reps)}
    p = C.RESULTS / 'grid_specs' / f'{bank_id}.json'
    C.write_json(p, spec)
    return p, spec


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    x = sub.add_parser('bank'); x.add_argument('--stage', required=True, choices=('D', 'R')); x.add_argument('--bank-id', required=True)
    x.add_argument('--candidates', required=True); x.add_argument('--comparators', default=''); x.add_argument('--references', action='store_true')
    x.add_argument('--reps', default='R1,R2'); x.add_argument('--fractions', default='1.0'); x.add_argument('--purpose', default='')
    x.add_argument('--parts', default=None, help='restrict to these partitions (e.g. R0: a seed-0-only bank)')
    a = ap.parse_args(argv)
    p, spec = bank(a.stage, a.bank_id, [c for c in a.candidates.split(',') if c], [c for c in a.comparators.split(',') if c],
                   a.references, a.purpose, tuple(a.reps.split(',')), tuple(float(f) for f in a.fractions.split(',')),
                   parts=tuple(a.parts.split(',')) if a.parts else None)
    print(json.dumps({'spec': str(p.relative_to(C.ROOT)), 'members': list(spec['members']), 'comparators': spec['comparators']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
