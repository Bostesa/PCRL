"""Independent check 7: MODEL_SELECT clauses, OUT_LR proxy and the frozen nominations (own code; no acs_replication import).

Clauses per seed (2 tasks x {U0, Ucal} x {accuracy loss <= 0.010, NLL excess <= 0.010 (1e-12 floor), summed Brier excess <= 0.005})
on MODEL_SELECT; proxy = AUC on MODEL_SELECT of LogisticRegression(C=1, lbfgs, max_iter=500) trained on the MODEL_FIT pair wire
[p_inc, d_inc, p_occ, d_occ] with MODEL_FIT SEX after an own zero-mean / unit-sd scaler (sd == 0 -> 1). Selection rule re-derived from
SCIENCE_LOCK.selection text. Aggregates only.
"""
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
PRIV = ROOT / '.private' / 'acs_replication'
RES = ROOT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
OUT = RES / 'independent'
TASKS = ('income', 'occupation')
KCLS = {'income': 2, 'occupation': 5}
CAPS = {'accuracy_loss': 0.010, 'nll_excess': 0.010, 'brier_excess': 0.005}
FILES = ('STAGE_A_{s}/STAGE_A_OUTPUTS.npz', 'STAGE_B_{s}/STAGE_B_OUTPUTS_E20.npz', 'NEURAL_RAW_{s}/NEURAL_RAW_OUTPUTS.npz',
         'FARE_{s}/FARE_OUTPUTS.npz', 'FAGTB_{s}/FAGTB_OUTPUTS.npz')
MODEL_SEED_BASE = 2026101205


def auc_rank(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    r = rankdata(s, method='average')
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def role(name, field):
    with np.load(PRIV / 'exports' / name / f'{field}.npz', allow_pickle=False) as z:
        return z[field].copy(), z['row_id'].copy()


def metrics(y, p):
    p = np.asarray(p, dtype=np.float64); n = len(y)
    acc = np.mean(np.argmax(p, axis=1) == y)
    nll = np.mean(-np.log(np.maximum(p[np.arange(n), y], 1e-12)))
    onehot = np.zeros_like(p); onehot[np.arange(n), y] = 1.0
    brier = np.mean(np.sum((p - onehot) ** 2, axis=1))
    return acc, nll, brier


def clauses(lab, svc_probs, refs):
    out = {}
    for t in TASKS:
        a, n, b = metrics(lab[t], svc_probs[t])
        for r in ('U0', 'Ucal'):
            ra, rn, rb = metrics(lab[t], refs[r][t])
            out[f'{t}|{r}|accuracy_loss'] = ra - a
            out[f'{t}|{r}|nll_excess'] = n - rn
            out[f'{t}|{r}|brier_excess'] = b - rb
    return out


def wire(svc_role):
    cols = []
    for t in TASKS:
        p = np.asarray(svc_role[t], dtype=np.float64)
        cols.append(p); cols.append(np.argmax(p, axis=1).astype(np.float64)[:, None])
    return np.hstack(cols)


def scale_fit(x):
    mu = x.mean(axis=0); sd = np.sqrt(((x - mu) ** 2).mean(axis=0))
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def proxy(fit_role, sel_role, sex_fit, sex_sel):
    from sklearn.linear_model import LogisticRegression
    xf = wire(fit_role); xs = wire(sel_role)
    mu, sd = scale_fit(xf)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500).fit((xf - mu) / sd, sex_fit)
    assert list(m.classes_) == [0, 1]
    return auc_rank(sex_sel, m.predict_proba((xs - mu) / sd)[:, 1])


def load_outputs(seed, roles):
    out = {}
    for pat in FILES:
        p = PRIV / 'runs' / pat.format(s=seed)
        with np.load(p) as z:
            for k in z.files:
                parts = k.split('__')
                if len(parts) != 3:
                    continue
                svc, rl, t = parts
                if rl in roles:
                    out.setdefault(svc, {}).setdefault(rl, {})[t] = z[k].copy()
    return out


def main():
    detail = json.loads((RES / 'SELECTION_DETAIL.json').read_text())
    lock = json.loads((RES / 'SELECTION_LOCK.json').read_text())
    lab = {t: role('MODEL_SELECT', t)[0] for t in TASKS}
    sex_sel, rid_sel = role('MODEL_SELECT', 'sex')
    sex_fit, rid_fit = role('MODEL_FIT', 'sex')
    mine = {}
    cmp_clause, cmp_proxy = [], []
    for seed in (0, 1):
        outs = load_outputs(seed, ('MODEL_FIT', 'MODEL_SELECT'))
        with np.load(PRIV / 'runs' / f'STAGE_A_{seed}' / 'STAGE_A_OUTPUTS.npz') as z:
            assert np.array_equal(z['row_id__MODEL_SELECT'], rid_sel) and np.array_equal(z['row_id__MODEL_FIT'], rid_fit)
        refs = {r: outs[r]['MODEL_SELECT'] for r in ('U0', 'Ucal')}
        mine[seed] = {}
        for svc, roles in outs.items():
            if svc in ('U0', 'Ucal'):
                continue
            cl = clauses(lab, roles['MODEL_SELECT'], refs)
            px = proxy(roles['MODEL_FIT'], roles['MODEL_SELECT'], sex_fit, sex_sel)
            mine[seed][svc] = {'clauses': cl, 'proxy': px}
            lead = detail['per_seed'][str(seed)]['services'][svc]
            lv = {f"{c['task']}|{c['reference']}|{c['metric']}": c['value'] for c in lead['clauses']}
            for k, v in cl.items():
                cmp_clause.append(abs(v - lv[k]))
            cmp_proxy.append((svc, seed, px, lead['proxy_auc'], abs(px - lead['proxy_auc'])))
    common = sorted(set(mine[0]) & set(mine[1]))
    nr = sorted([s for s in common if s.startswith('NEURAL_RAW_rho')], key=lambda s: (float(s.split('rho')[1].split('_')[0]), int(s.split('_E')[1])))
    fg = sorted([s for s in common if s.startswith('FAGTB_lam')], key=lambda s: float(s.split('lam')[1]))
    fa = sorted([s for s in common if s.startswith('FARE_F')], key=lambda s: int(s.split('FARE_F')[1]))
    frozen, tables = {}, {}
    for fam, grid in (('NEURAL_RAW', nr), ('FAGTB', fg), ('FARE', fa)):
        rows = []
        for i, c in enumerate(grid):
            vals = [(k, v) for s in (0, 1) for k, v in mine[s][c]['clauses'].items()]
            feas = all(v <= CAPS[k.split('|')[2]] for k, v in vals)
            maxn = max(v / CAPS[k.split('|')[2]] for k, v in vals)
            failed = [f's{s}|{k}' for s in (0, 1) for k, v in mine[s][c]['clauses'].items() if v > CAPS[k.split('|')[2]]]
            rows.append({'configuration': c, 'grid_index': i, 'feasible': feas, 'max_normalized': maxn,
                         'proxy_mean': float(np.mean([mine[s][c]['proxy'] for s in (0, 1)])), 'n_failed_clauses': len(failed), 'failed': failed})
        feasible = [r for r in rows if r['feasible']]
        if feasible:
            w = min(feasible, key=lambda r: (r['proxy_mean'], r['grid_index'])); status = 'ELIGIBLE'
        else:
            w = min(rows, key=lambda r: (r['max_normalized'], r['grid_index'])); status = 'INELIGIBLE_FALLBACK'
        frozen[fam] = {'configuration': w['configuration'], 'status': status}
        tables[fam] = rows
    flags = []
    for seed in (0, 1):
        r = json.loads((PRIV / 'runs' / f'FAGTB_{seed}' / 'FAGTB_SEED.json').read_text())
        flags.append(bool(r.get('binary_valid')) and bool(r.get('mc_valid')))
    two_task = all(flags)
    fixed = {}
    for svc in ('T_TREE', 'LOCAL_TREE', 'TASK_REFIT', 'JOINT_TREE'):
        vals = [(f's{s}|{k}', v) for s in (0, 1) for k, v in mine[s][svc]['clauses'].items()]
        fixed[svc] = {'point_feasible_24': all(v <= CAPS[k.split('|')[3]] for k, v in vals),
                      'failed': [k for k, v in vals if v > CAPS[k.split('|')[3]]],
                      'max_normalized': max(v / CAPS[k.split('|')[3]] for k, v in vals)}
    # margin of the NEURAL_RAW nomination (proxy gap to the runner-up feasible configuration)
    nr_feas = sorted([r for r in tables['NEURAL_RAW'] if r['feasible']], key=lambda r: r['proxy_mean'])
    agree_frozen = all(frozen[f]['configuration'] == lock['frozen'][f]['configuration'] and frozen[f]['status'] == lock['frozen'][f]['status'] for f in frozen)
    agree_fixed = all(fixed[s]['point_feasible_24'] == lock['fixed_services'][s]['point_feasible_24'] and sorted(fixed[s]['failed']) == sorted(lock['fixed_services'][s]['failed_clauses']) for s in fixed)
    lead_tables = detail['family_tables']
    agree_tables = all([r['configuration'] for r in tables[f]] == [r['configuration'] for r in lead_tables[f]] and
                       all(a['feasible'] == b['feasible'] for a, b in zip(tables[f], lead_tables[f])) for f in tables)
    max_clause = float(max(cmp_clause)); max_proxy = float(max(x[4] for x in cmp_proxy))
    ok = (max_clause <= 1e-12 and max_proxy <= 1e-6 and agree_frozen and agree_fixed and agree_tables and two_task == lock['frozen']['FAGTB']['two_task_valid'])
    res = {'id': 7, 'name': 'NOMINATIONS', 'status': 'PASS' if ok else 'FAIL',
           'evidence': {'services_x_seeds_checked': len(cmp_proxy), 'clause_values_checked': len(cmp_clause), 'max_abs_clause_diff_vs_SELECTION_DETAIL': max_clause,
                        'max_abs_proxy_auc_diff_vs_SELECTION_DETAIL': max_proxy,
                        'frozen_recomputed': frozen, 'frozen_lead': {f: {'configuration': lock['frozen'][f]['configuration'], 'status': lock['frozen'][f]['status']} for f in frozen},
                        'frozen_agree': agree_frozen, 'family_feasibility_tables_agree': agree_tables,
                        'feasible_counts': {f: sum(r['feasible'] for r in tables[f]) for f in tables},
                        'neural_raw_feasible_ranked_by_proxy': [(r['configuration'], round(r['proxy_mean'], 6)) for r in nr_feas],
                        'fallback_max_normalized': {f: min(r['max_normalized'] for r in tables[f]) for f in ('FAGTB', 'FARE')},
                        'fixed_services': {s: {'point_feasible_24': v['point_feasible_24'], 'failed': v['failed']} for s, v in fixed.items()},
                        'fixed_agree': agree_fixed, 'FAGTB_two_task_valid_recomputed': two_task, 'FAGTB_two_task_valid_lead': lock['frozen']['FAGTB']['two_task_valid']},
           'tables': tables,
           'proxy_comparison': [{'service': a, 'seed': b, 'independent': c, 'lead': d, 'abs_diff': e} for a, b, c, d, e in cmp_proxy],
           'notes': 'Own metrics (accuracy argmax, NLL with 1e-12 floor, summed Brier), own scaler (population sd, zero-sd -> 1), sklearn LogisticRegression C=1 lbfgs '
                    'max_iter=500 (same estimator class as the lead; OUT_LR has no random_state dependence under lbfgs), own rank AUC. Rule re-derived from SCIENCE_LOCK.selection.'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'check07_nominations.json').write_text(json.dumps(res, indent=1, sort_keys=True, default=float))
    print(json.dumps(res['evidence'], indent=1, default=float), res['status'])


if __name__ == '__main__':
    main()
