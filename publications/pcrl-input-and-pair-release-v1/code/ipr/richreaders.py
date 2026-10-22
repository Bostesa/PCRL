"""Data-rich reader check of the PCRL input-and-pair release campaign (challenger findings 3 and 18; PROTOCOL_AMENDMENT_1.md;
descriptive, never a decision input). EXPLORATORY DEVELOPMENT on previously and adaptively used records.

Readers (LOG_LR on standardized log-probability features for the income, occupation and pair views; HGB15 and MLP64 on the pair
view) are TRAINED inside each family's fit unit on the 20,000 MODEL_FIT people's out-of-fold (DA) or nested (DB) releases with the
MODEL_FIT SEX label (operation 'fit', under the family's card): five times the grid's reader-FIT size. They are SCORED here, by the
readers-operation entry point `ipr.richreaders` (added by ACCESS_AMENDMENTS.json before first use), on the deployed MODEL_SELECT
releases of the IPR1 SCORE parts of repetitions R1 and R2 with the MODEL_SELECT SEX label (no MODEL_FIT label is read here; the
trained readers are loaded from their hash-checked files). Output: aggregate AUCs per release, view, reader, version and repetition,
and benefits (comparator AUC - candidate AUC) for the declared comparisons. No person-level output is written.

    python3 -m ipr.richreaders score --spec results/<campaign>/rich/RICH_SPEC.json
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from . import common as C

FLOOR = 1e-6
VIEWS = ('income', 'occupation', 'pair')
OUT = C.RESULTS / 'rich'


def features(pI, pO, view):
    li = np.log(np.maximum(pI[:, 1:], FLOOR) / np.maximum(pI[:, :1], FLOOR))
    lo = np.log(np.maximum(pO, FLOOR))
    co = lo - lo.mean(1, keepdims=True)
    if view == 'income':
        return li
    if view == 'occupation':
        return co
    return np.column_stack([li, co])


def train(views_by_config, sex, version):
    """{config: {view: {reader: fitted pipeline}}} on MODEL_FIT out-of-fold releases (called inside a fit unit)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = {}
    t0 = time.process_time()
    for cfg, (pI, pO) in views_by_config.items():
        out[cfg] = {}
        for view in VIEWS:
            F = features(pI, pO, view)
            m = {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m['LOG_LR'] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(F, sex)
                if view == 'pair':
                    m['HGB15'] = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.05, max_iter=300,
                                                                min_samples_leaf=40, random_state=int(version)).fit(F, sex)
                    m['MLP64'] = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
                                                                               early_stopping=True, random_state=int(version))).fit(F, sex)
            out[cfg][view] = m
    return out, time.process_time() - t0


def save(family, version, models, extra):
    d = C.RUNS / family
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    p = d / f'RICH_READERS_V{int(version)}.pkl'
    buf = pickle.dumps({'models': models, **extra}, protocol=5)
    with open(p, 'xb') as f:
        f.write(buf)
    import hashlib
    sha = hashlib.sha256(buf).hexdigest()
    (d / f'RICH_READERS_V{int(version)}.sha256').write_text(sha)
    return sha


def _load(family, version):
    """`family` is a family id (the registered fit's readers, RICH_READERS_V<v>) or 'FAMILY:R<n>' (a revision fit's readers,
    RICH_READERS_R<n>_V<v>, written by that revision's fit unit)."""
    import hashlib
    fam, _, rev = family.partition(':')
    d = C.RUNS / fam
    stem = f'RICH_READERS_{rev}_V{int(version)}' if rev else f'RICH_READERS_V{int(version)}'
    buf = (d / f'{stem}.pkl').read_bytes()
    if hashlib.sha256(buf).hexdigest() != (d / f'{stem}.sha256').read_text().strip():
        raise PermissionError(f'{family} rich readers V{version} differ from their hash')
    return pickle.loads(buf)


def _auc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    r = rankdata(score)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def score(spec_path):
    """spec = {'id', 'members': {name: {'family': F, 'config': C, 'readers_of': F2 (family whose readers were trained on this
    member's MODEL_FIT out-of-fold release) , 'reader_key': cfg key in that file}}, 'comparisons': [[cand, comp], ...]}."""
    from . import release as RL
    from . import devreaders as DR
    spec = json.loads(Path(spec_path).read_text())
    t0 = time.process_time()
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'readers',
               purpose=f"data-rich reader check {spec['id']} (MODEL_SELECT SEX on the IPR1 SCORE parts of R1 and R2; readers trained "
                       f"on MODEL_FIT out-of-fold releases)")
    sex, unit = S['sex'], S['unit']
    del S
    score_masks = {r: DR.roles_of(unit, r)['SCORE'] for r in ('R1', 'R2')}
    res = {}
    for v in C.VERSIONS:
        cache = {}
        for name, m in spec['members'].items():
            fam_r = m['readers_of']
            if fam_r not in cache:
                cache[fam_r] = _load(fam_r, v)
            readers = cache[fam_r]['models'].get(m['reader_key'])
            if readers is None:
                continue
            arr, rec = RL.load(m['family'], m['config'], v, roles=('MODEL_SELECT',))
            pI, pO = arr['MODEL_SELECT']['income'], arr['MODEL_SELECT']['occupation']
            for view, rd in readers.items():
                F = features(pI, pO, view)
                for rname, model in rd.items():
                    s = model.predict_proba(F)[:, 1]
                    for rep, msk in score_masks.items():
                        res.setdefault(name, {}).setdefault(view, {}).setdefault(rname, {})[f'S{v}{rep}'] = _auc(s[msk], sex[msk])
    comps = {}
    for cand, comp in spec.get('comparisons', []):
        if cand not in res or comp not in res:
            continue
        out = {}
        for view, rd in res[cand].items():
            for rname, cells in rd.items():
                other = (res[comp].get(view) or {}).get(rname) or {}
                d = {c: other[c] - cells[c] for c in cells if c in other}
                vm = {f'V{v}': float(np.mean([d[c] for c in d if c.startswith(f'S{v}')])) for v in C.VERSIONS
                      if any(c.startswith(f'S{v}') for c in d)}
                out[f'{view}|{rname}'] = {'cells': d, 'version_means': vm, 'mean': float(np.mean(list(d.values()))) if d else None}
        comps[f'{cand}__vs__{comp}'] = out
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {'schema': 'IPR1-rich-readers-v1', 'id': spec['id'], 'utc': C.now(), 'auc': res, 'benefits': comps,
           'note': 'descriptive: readers trained on 20,000 MODEL_FIT out-of-fold releases (MODEL_FIT SEX), scored on the IPR1 SCORE parts '
                   'of R1 / R2 (MODEL_SELECT SEX); benefit = comparator AUC - candidate AUC (positive favours the candidate); never a '
                   'decision input', 'cpu_s': time.process_time() - t0}
    C.write_json_x(OUT / f"RICH_{spec['id']}.json", rec)
    print(json.dumps({k: {kk: vv.get('pair|LOG_LR', {}).get('version_means') for kk, vv in {k: v}.items()}
                      for k, v in comps.items()}, indent=1, default=str))
    return rec


def selftest():
    rng = np.random.default_rng(0)
    n = 2000
    sex = rng.integers(0, 2, n)
    pI = np.column_stack([1 - (0.3 + 0.3 * sex), 0.3 + 0.3 * sex]) * 0.9 + 0.05
    pO = rng.dirichlet(np.ones(5), n)
    m, cpu = train({'X': (pI, pO)}, sex, 0)
    s = m['X']['pair']['LOG_LR'].predict_proba(features(pI, pO, 'pair'))[:, 1]
    ok = {'trained_all_views': set(m['X']) == set(VIEWS), 'pair_has_three': set(m['X']['pair']) == {'LOG_LR', 'HGB15', 'MLP64'},
          'separable_auc_high': _auc(s, sex) > 0.95}
    ok = {k: bool(v) for k, v in ok.items()}
    ok['all_ok'] = all(ok.values())
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    s = sub.add_parser('score'); s.add_argument('--spec', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'selftest':
        r = selftest(); print(json.dumps(r)); return 0 if r['all_ok'] else 1
    score(a.spec)
    return 0


if __name__ == '__main__':
    sys.exit(main())
