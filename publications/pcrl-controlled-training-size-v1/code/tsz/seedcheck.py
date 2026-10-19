"""Reader-seed effectiveness check of the fresh development grid (RULES.md section 9; computed before any stability label).

    python3 -m tsz.seedcheck run --spec results/pcrl_controlled_training_size_v1/grid_specs/COND.json

For every member release of the bank (at least W100_v and every selected release), every view (income, occupation, pair), every
version v and repetition R, it reads the two cached reader entries of the grid (reader seeds k0 and k1; tsz.devreaders cache, keyed
by the released wire and the MODEL_SELECT SEX array hash) and records, per recipe, whether the SCORE scores are bitwise equal.
SEX is read only to recompute the cache key (operation 'readers', logged; authorized by a dated ACCESS_AMENDMENTS entry); no reader is
fitted here and nothing is scored. Output: results/<campaign>/readers/READER_SEED_CHECK.json with tsz.rules.reader_seed_check.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import common as C
from . import devreaders as DR


def run(spec_path):
    from . import rules as R
    out = C.RESULTS / 'readers' / 'READER_SEED_CHECK.json'
    if out.exists():
        raise PermissionError('READER_SEED_CHECK.json exists (write-once)')
    spec = DR.validate_spec(json.loads(Path(spec_path).read_text()))
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'readers', 'reader-seed effectiveness check: recompute the grid cache keys '
               '(MODEL_SELECT SEX array hash) to compare the k0 and k1 readers of the same wire; no reader is fitted or scored')
    sex, unit, row_id = S['sex'], S['unit'], S['row_id']
    del S
    sx = C.arr_sha(np.asarray(sex))
    checks, missing = [], []
    seeds = {f'S{v}k{k}': DR.reader_seed(v, k) for v in C.VERSIONS for k in DR.READER_SEEDS}
    for v in C.VERSIONS:
        for name, member in spec['members'].items():
            rel, _ = DR.load_member(member, v, n=len(sex), row_id=row_id)
            for view in DR.VIEWS:
                wh = DR.wire_hash(view, rel['income'], rel['occupation'])
                for rep in DR.REPS:
                    n_sc = int(DR.roles_of(unit, rep)['SCORE'].sum())
                    ents = {}
                    for k in DR.READER_SEEDS:
                        p = DR.CACHE / f'{DR.cache_key(rep, DR.reader_seed(v, k), 1.0, wh, sx)}.npz'
                        ents[k] = DR._read_entry(p, n_sc) if p.is_file() else None
                    if ents[0] is None or ents[1] is None:
                        missing.append({'member': name, 'view': view, 'v': v, 'R': rep}); continue
                    for recipe in DR.RECIPES:
                        checks.append({'recipe': recipe, 'view': view, 'wire': name, 'v': v, 'R': rep,
                                       'equal_k0_k1': bool(np.array_equal(ents[0][recipe][0], ents[1][recipe][0]))})
    res = R.reader_seed_check(seeds, checks)
    per_recipe = {}
    for c in checks:
        d = per_recipe.setdefault(c['recipe'], {'checks': 0, 'equal_k0_k1': 0})
        d['checks'] += 1; d['equal_k0_k1'] += int(c['equal_k0_k1'])
    rec = {'schema': 'TSZ1-reader-seed-check-v1', 'utc': C.now(), 'bank_id': spec['bank_id'], 'spec_sha256': C.sha(Path(spec_path)),
           'seeds': seeds, 'n_checks': len(checks), 'missing_entries': missing, 'per_recipe': per_recipe, 'rule': res,
           'note': 'seed-dependent recipes must differ under k0 / k1 in every check; seed-invariant recipes are expected to be equal'}
    C.write_json_x(out, rec)
    print(json.dumps({'label': res['label'], 'errors': res['errors'], 'flags': res['flags'], 'missing': len(missing),
                      'per_recipe': per_recipe}))
    return rec


def cardinality(spec_path):
    """Label-free companion record required by adjudication A-SC1 (iii-c): the number of distinct input vectors each reader view
    receives on MODEL_SELECT, for every bank member and version (income: the income vectors; occupation: the occupation vectors;
    pair: the concatenated pair). No label is read."""
    out = C.RESULTS / 'readers' / 'WIRE_CARDINALITY.json'
    if out.exists():
        raise PermissionError('WIRE_CARDINALITY.json exists (write-once)')
    spec = DR.validate_spec(json.loads(Path(spec_path).read_text()))
    F = C.features('MODEL_SELECT')
    res = {}
    for v in C.VERSIONS:
        for name, member in spec['members'].items():
            rel, _ = DR.load_member(member, v, n=len(F['row_id']), row_id=F['row_id'])
            inc = np.ascontiguousarray(rel['income']); occ = np.ascontiguousarray(rel['occupation'])
            res[f'{name}|V{v}'] = {'income': int(np.unique(inc, axis=0).shape[0]), 'occupation': int(np.unique(occ, axis=0).shape[0]),
                                   'pair': int(np.unique(np.hstack([inc, occ]), axis=0).shape[0])}
    rec = {'schema': 'TSZ1-wire-cardinality-v1', 'utc': C.now(), 'bank_id': spec['bank_id'], 'spec_sha256': C.sha(Path(spec_path)),
           'labels_read': False, 'people': int(len(F['row_id'])), 'distinct_input_vectors': res,
           'purpose': 'adjudication A-SC1 condition (iii-c): input cardinality beside every seed-equality check'}
    C.write_json_x(out, rec)
    print(json.dumps({k: v for k, v in res.items() if min(v.values()) < 100}))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--spec', required=True)
    c = sub.add_parser('cardinality'); c.add_argument('--spec', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'run':
        run(a.spec)
    else:
        cardinality(a.spec)
    return 0


if __name__ == '__main__':
    sys.exit(main())
