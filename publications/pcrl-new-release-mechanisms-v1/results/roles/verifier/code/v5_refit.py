"""V5 representative reader refits with the verifier's OWN readers (scikit-learn estimators configured here, not the grid's recipe
code) on the released vectors of named grid members, in chosen cells of the registered splits: fitted on reader FIT, oriented and
selected on VALIDATION, scored on SCORE. Purpose: order of magnitude and sign of key paired differences (comparator - candidate
pair AUC), never a substitute for the registered grid.

Readers (per view: income, occupation, pair = both outputs; each sees only the float64 probability vectors and argmax decisions):
  V_LR    logistic regression (C = 1, lbfgs, 2,000 iterations) on centred log-probabilities + one-hot decisions;
  V_GB    HistGradientBoostingClassifier (200 iterations, learning rate 0.05, 15 leaves, min 40 per leaf) on probabilities + decisions;
  V_TAB   smoothed frequency table on exact released vectors (finite cells; prior-mean smoothing with weight 2; unseen -> prior).
Pair selection: max VALIDATION AUC among the pair readers and the ignore-one alternatives (income-only, occupation-only).

Labels: MODEL_SELECT SEX only, via nrm.common.load(operation 'verify') called from THIS script (logged).

    python3 .../v5_refit.py --tag V1 --bank BANK --seed 0 --cells S0R1k0 [S0R2k0] --members CAND COMP1 COMP2 ... [--candidate CAND]
    python3 .../v5_refit.py --selftest
"""
from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

READERS = ('V_LR', 'V_GB', 'V_TAB')


def design(view, inc, occ):
    def part(p):
        lp = np.log(np.clip(p, 1e-12, None))
        lp = lp - lp.mean(axis=1, keepdims=True)
        dec = np.eye(p.shape[1])[V.argmax_first(p)]
        return np.column_stack((lp, dec)), np.column_stack((p, dec))
    if view == 'income':
        return part(inc)
    if view == 'occupation':
        return part(occ)
    a1, b1 = part(inc); a2, b2 = part(occ)
    return np.column_stack((a1, a2)), np.column_stack((b1, b2))


def table_reader(Xfit, yfit, Xall, weight=2.0):
    keys_fit = [r.tobytes() for r in np.ascontiguousarray(Xfit)]
    prior = float(yfit.mean())
    cnt, pos = {}, {}
    for k, y in zip(keys_fit, yfit):
        cnt[k] = cnt.get(k, 0) + 1; pos[k] = pos.get(k, 0) + int(y)
    out = np.empty(len(Xall))
    for i, r in enumerate(np.ascontiguousarray(Xall)):
        k = r.tobytes()
        n = cnt.get(k, 0)
        out[i] = (pos.get(k, 0) + weight * prior) / (n + weight)
    return out


def fit_view(view, inc, occ, sex, fit, val, rs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xlr, Xgb = design(view, inc, occ)
    raw = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xlr[fit], sex[fit])
        raw['V_LR'] = lr.predict_proba(Xlr)[:, 1]
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                            random_state=rs).fit(Xgb[fit], sex[fit])
        raw['V_GB'] = gb.predict_proba(Xgb)[:, 1]
    tab_in = inc if view == 'income' else occ if view == 'occupation' else np.column_stack((inc, occ))
    raw['V_TAB'] = table_reader(tab_in[fit], sex[fit], tab_in)
    out = {}
    for n, s in raw.items():
        a = V.auc(sex[val], s[val])
        sign = 1 if a >= 0.5 else -1
        so = s if sign == 1 else 1.0 - s
        out[n] = {'scores': so, 'val_auc': V.auc(sex[val], so[val]), 'orientation': sign}
    return out


def refit_member(inc, occ, sex, unit, rep, rs):
    r = V.reader_roles(unit, rep)
    fit, val, sc = r['FIT'], r['VALIDATION'], r['SCORE']
    views = {view: fit_view(view, inc, occ, sex, fit, val, rs) for view in V.VIEWS}
    res = {}
    for view in ('income', 'occupation'):
        best = max(READERS, key=lambda n: (views[view][n]['val_auc'], -READERS.index(n)))
        res[view] = {'winner': best, 'val_auc': views[view][best]['val_auc'], 'score_auc': V.auc(sex[sc], views[view][best]['scores'][sc]),
                     'per_reader_score_auc': {n: V.auc(sex[sc], views[view][n]['scores'][sc]) for n in READERS}}
    cands = [(f'pair:{n}', views['pair'][n]) for n in READERS] + [(f'income:{n}', views['income'][n]) for n in READERS] + \
            [(f'occupation:{n}', views['occupation'][n]) for n in READERS]
    bi = max(range(len(cands)), key=lambda i: (cands[i][1]['val_auc'], -i))
    res['pair'] = {'winner': cands[bi][0], 'val_auc': cands[bi][1]['val_auc'], 'score_auc': V.auc(sex[sc], cands[bi][1]['scores'][sc]),
                   'per_reader_score_auc': {n: V.auc(sex[sc], views['pair'][n]['scores'][sc]) for n in READERS}}
    return res


def load_sex(purpose):
    sys.path.insert(0, str(V.ROOT))
    import nrm.common as C
    S = C.load('MODEL_SELECT', ('features', 'sex'), 'verify', purpose)
    return np.asarray(S['sex']), np.asarray(S['unit'])


def selftest():
    rng = np.random.default_rng(5)
    n = 3000
    unit = np.repeat(np.arange(n // 2), 2)
    sex = (rng.random(n) < 0.47).astype(np.int64)
    z = rng.normal(size=(n, 5)); z[:, 1] += 0.8 * sex
    leaky = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    z2 = rng.normal(size=(n, 5))
    safe = np.exp(z2) / np.exp(z2).sum(1, keepdims=True)
    t = 1 / (1 + np.exp(-rng.normal(size=n))); inc = np.column_stack((1 - t, t))
    a = refit_member(inc, leaky, sex, unit, 'R1', 1)
    b = refit_member(inc, safe, sex, unit, 'R1', 1)
    levels = np.array([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
    fin = levels[np.clip((sex + rng.integers(0, 2, n)), 0, 2)]
    c = refit_member(fin, safe, sex, unit, 'R1', 1)
    ok = {'leaky_pair_auc_high': a['pair']['score_auc'] > 0.60,'safe_pair_auc_near_half': abs(b['pair']['score_auc'] - 0.5) < 0.05,
          'leaky_beats_safe': a['pair']['score_auc'] - b['pair']['score_auc'] > 0.1,
          'finite_income_table_reads': c['income']['per_reader_score_auc']['V_TAB'] > 0.6}
    ok['all_ok'] = all(ok.values())
    return ok, {'leaky': a['pair']['score_auc'], 'safe': b['pair']['score_auc'], 'finite_tab': c['income']['per_reader_score_auc']['V_TAB']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag'); ap.add_argument('--bank'); ap.add_argument('--seed', type=int); ap.add_argument('--cells', nargs='*', default=[])
    ap.add_argument('--members', nargs='*', default=[]); ap.add_argument('--candidate'); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok, vals = selftest()
        chk = {'id': f'V5.REFIT_SELFTEST__{a.tag or "V1"}', 'what': 'own reader refits on invented data: a leaky occupation vector reads '
               'high, a SEX-independent one near 0.5, a finite SEX-dependent income code is read by the frequency table',
               'inputs': V.inputs_sha([Path(__file__), V.CODE_DIR / 'vlib.py']), 'result': {**ok, 'values': vals},
               'numerical_status': 'PASS' if ok['all_ok'] else 'FAIL', 'procedural_status': 'NOT_APPLICABLE'}
        print(json.dumps(V.append_checks([chk]), indent=1)); print(json.dumps({**ok, **vals}, indent=1, default=V._jd))
        return 0 if ok['all_ok'] else 1
    import v4_grid as G
    sex, unit = load_sex(f'verify: representative own-reader refits on {a.bank} seed {a.seed} cells {a.cells} members {a.members}')
    d = V.RUNS / 'GRID' / a.bank / f'S{a.seed}'
    g = json.loads((d / 'GRID.json').read_text())
    out = {}
    for name in a.members:
        rel, prov = G.member_arrays(g['spec']['members'][name], a.seed)
        rel = {t: np.asarray(rel[t], dtype=np.float64) for t in V.TASKS}
        for cell in a.cells:
            rep, k = cell[2:4], int(cell.split('k')[1])
            out.setdefault(cell, {})[name] = refit_member(rel['income'], rel['occupation'], sex, unit, rep, V.reader_seed(a.seed, k))
    comp = {}
    if a.candidate:
        for cell, per in out.items():
            for name in per:
                if name == a.candidate:
                    continue
                mine = per[name]['pair']['score_auc'] - per[a.candidate]['pair']['score_auc']
                grid = g['paired'].get(a.candidate, {}).get(name, {}).get('pair', {}).get('cells', {}).get(cell)
                if grid is None and name in g['paired']:
                    rev = g['paired'][name].get(a.candidate, {}).get('pair', {}).get('cells', {}).get(cell)
                    grid = -rev if rev is not None else None
                comp[f'{cell}|{name}'] = {'own_refit_benefit': mine, 'grid_benefit': grid,
                                          'same_sign': (None if grid is None else bool(np.sign(mine) == np.sign(grid))),
                                          'own_pair_auc': {a.candidate: per[a.candidate]['pair']['score_auc'], name: per[name]['pair']['score_auc']},
                                          'grid_pair_auc': {x: g['cells'][cell][x]['pair']['score_auc'] for x in (a.candidate, name) if x in g['cells'].get(cell, {})}}
    rec = {'tag': a.tag, 'bank': a.bank, 'seed': a.seed, 'cells': a.cells, 'members': a.members, 'candidate': a.candidate,
           'refits': out, 'comparisons': comp, 'utc': V.utc()}
    V.write_record(f'V5_REFIT__{a.bank}__S{a.seed}__{a.tag}.json', rec)
    signs = [x['same_sign'] for x in comp.values() if x['same_sign'] is not None]
    mag = [abs(x['own_refit_benefit'] - x['grid_benefit']) for x in comp.values() if x['grid_benefit'] is not None]
    chk = {'id': f'V5.REFIT.{a.bank}.S{a.seed}__{a.tag}',
           'what': f'own representative reader refits (logistic, gradient boosting, frequency table) on {a.members} in {a.cells}: sign and '
                   'magnitude of comparator - candidate pair AUC versus the grid',
           'inputs': V.inputs_sha([d / 'GRID.json']),
           'criterion': 'PASS iff every compared benefit has the grid sign and |own - grid| < 0.03 AUC (order of magnitude); otherwise '
                        'ATTENTION (own readers differ from the locked recipes; never a FAIL of the grid by itself)',
           'result': {'comparisons': comp, 'n_same_sign': sum(signs), 'n_compared': len(signs), 'max_abs_gap_to_grid': max(mag) if mag else None},
           'numerical_status': ('PASS' if signs and all(signs) and max(mag) < 0.03 else 'ATTENTION') if signs else 'NOT_RUN',
           'procedural_status': 'NOT_APPLICABLE'}
    print(json.dumps(V.append_checks([chk]), indent=1))
    print(json.dumps(chk['result'], indent=1, default=V._jd)[:4000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
