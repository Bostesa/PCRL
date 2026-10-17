"""Independent verifier, Check 7: refit the pair-view winner readers of SCM, TWIN and NEURAL_RAW on both seeds.

Own code (sklearn estimators only; no scm_bench / task_conditioned_representation function is called for any verified
quantity):
- features: 'out' = [p, argmax(p)] ; 'log' = [log(max(p, 1e-300)), onehot(argmax p)] ; pair = [income kind, occupation kind]
- scaler: column mean and population std (ddof 0) from ATTACK_FIT; zero-std columns get scale 1
- readers: LR = LogisticRegression(C=1, lbfgs, max_iter=500); MLP = MLPClassifier((64, 32), relu, adam, alpha=1e-4,
  max_iter=300, early_stopping=False, random_state=MODEL_SEED_BASE + s); HGB = HistGradientBoosting(15 leaves, 200 iters,
  lr 0.05, no early stop, random_state=MODEL_SEED_BASE + s); MODEL_SEED_BASE = 2026101205 (source registry constant)
- AUC: Mann-Whitney with mid-ranks; orientation +1 iff ATTACK_VALIDATION AUC >= 0.5, else scores are 1 - s
- winner ids are read from runs/READERS_{s}/READERS_SEED.json; refit DEV_REPORT scores are compared with the stored
  runs/READERS_{s}/DEV_REPORT_SCORES.npz entries (max abs diff, bitwise)
- selection replay: every pair candidate (8 pair readers + 16 ignore-one aliases) is re-scored on ATTACK_VALIDATION from
  the STORED reader models (joblib; the stored scaler moments are applied by this script's own transform) and the winner
  rule (max oriented AUC, ties to the earlier candidate) is re-applied with this script's AUC.
Writes aggregates only (check7_refits.json).
"""
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

W = Path('<repository>/.claude/worktrees/pcrl-scm-engineering-benchmark-v1')
P = W / '.private' / 'pcrl_scm_engineering_benchmark_v1'
R = W / 'results' / 'pcrl_scm_engineering_benchmark_v1'
OUT = R / 'roles' / 'verifier' / 'check7_refits.json'
MODEL_SEED_BASE = 2026101205
ROLES3 = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
NAMES8 = ('OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP')
KIND = {'OUT_LR': 'out', 'OUT_MLP': 'out', 'OUT_HGB': 'out', 'LOG_LR': 'log', 'LOG_MLP': 'log', 'DEC_LR': 'dec', 'CONF_HGB': 'conf'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    r = rankdata(s, method='average'); n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def kind_table(p, kind):
    p = np.asarray(p, dtype=np.float64); d = p.argmax(1); k = p.shape[1]
    one = np.zeros((len(p), k)); one[np.arange(len(p)), d] = 1.0
    if kind == 'out':
        return np.column_stack((p, d.astype(np.float64)))
    if kind == 'log':
        return np.column_stack((np.log(np.maximum(p, 1e-300)), one))
    if kind == 'dec':
        return one
    if kind == 'conf':
        return p.max(1, keepdims=True)
    raise ValueError(kind)


def pair_table(pi, po, kind):
    return np.column_stack((kind_table(pi, kind), kind_table(po, kind)))


def scale_fit(x):
    m = x.mean(0); sd = x.std(0)
    return m, np.where(sd > 0, sd, 1.0)


def make_model(name, seed):
    if 'MLP' in name:
        return MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=1e-4, max_iter=300,
                             early_stopping=False, random_state=seed)
    if 'HGB' in name:
        return HistGradientBoostingClassifier(max_leaf_nodes=15, max_iter=200, learning_rate=.05, early_stopping=False, random_state=seed)
    return LogisticRegression(C=1, max_iter=500, solver='lbfgs')


def lookup_keys(*vecs):
    z = np.ascontiguousarray(np.column_stack([np.asarray(v, dtype=np.float64) for v in vecs]))
    return [r.tobytes() for r in z]


def load_export(role, name):
    pub = json.loads((W / 'results' / 'pcrl_acs_protected_tree_replication_v1' / 'EXTRACTION_SUMMARY.json').read_text())['export_file_sha256'][role]
    p = P / 'source' / 'exports' / role / f'{name}.npz'
    if sha(p) != pub[name]:
        raise SystemExit(f'export hash differs: {role} {name}')
    with np.load(p) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    t_start = time.process_time()
    res = {'schema': 'SCM1-verifier-check7-v1', 'model_seed_base': MODEL_SEED_BASE, 'seeds': {}}
    man = json.loads((R / 'SERVICE_BANK_MANIFEST.json').read_text())
    rl = json.loads((R / 'REPORTING_LOCK.json').read_text())
    all_ok = True
    for s in (0, 1):
        seed_model = MODEL_SEED_BASE + s
        bp = P / 'bank' / f'SERVICE_BANK_S{s}.npz'
        rec = {'bank_sha256_matches_manifest': sha(bp) == man['bank_files_sha256'][f'S{s}']}
        rs_path = P / 'runs' / f'READERS_{s}' / 'READERS_SEED.json'
        sc_path = P / 'runs' / f'READERS_{s}' / 'DEV_REPORT_SCORES.npz'
        rec['readers_seed_sha256_matches_reporting_lock'] = sha(rs_path) == rl['readers'][f'S{s}']['readers_seed_sha256']
        rec['dev_report_scores_sha256_matches_reporting_lock'] = sha(sc_path) == rl['readers'][f'S{s}']['dev_report_scores_sha256']
        rseed = json.loads(rs_path.read_text())
        rec['stored_reader_model_seed'] = rseed['reader_model_seed']
        rec['reader_model_seed_rule_ok'] = rseed['reader_model_seed'] == seed_model
        with np.load(bp) as z:
            bank = {k: z[k].copy() for k in z.files}
        with np.load(sc_path) as z:
            stored = {k: z[k].copy() for k in z.files}
        sex = {}
        for role in ('ATTACK_FIT', 'ATTACK_VALIDATION'):
            e = load_export(role, 'sex'); f = load_export(role, 'features')
            if not (np.array_equal(e['row_id'], f['row_id']) and np.array_equal(e['row_id'], bank[f'row_id__{role}'])
                    and np.array_equal(e['unit'], bank[f'unit__{role}'])):
                raise SystemExit('row pairing differs ' + role)
            sex[role] = e['sex'].astype(np.int64)
        f = load_export('DEV_REPORT', 'features')
        rec['dev_report_row_ids_aligned'] = bool(np.array_equal(f['row_id'], bank['row_id__DEV_REPORT']) and np.array_equal(stored['row_id'], f['row_id']))
        rec['services'] = {}
        for svc in ('SCM', 'TWIN', 'NEURAL_RAW'):
            w = rseed['services'][svc]['winner']['pair']
            name = w['candidate']; rid = w['reader_id']
            srec = {'stored_winner': name, 'stored_reader_id': rid, 'stored_attack_validation_auc': w['attack_validation_auc']}
            # ---- selection replay from stored models (all 24 pair candidates)
            import joblib  # noqa: E402  (stored models are pickles; FiniteScaler class import happens inside joblib.load)
            cands = []
            for nm in NAMES8:
                cands.append((nm, f'PAIR__{svc}__{nm}', 'pair'))
            for nm in NAMES8:
                cands.append((f'IGNORE_occupation__{nm}', f'MARG__{svc}|income__{nm}', 'income'))
            for nm in NAMES8:
                cands.append((f'IGNORE_income__{nm}', f'MARG__{svc}|occupation__{nm}', 'occupation'))
            replay = []
            stored_c = rseed['services'][svc]['pair_candidates_attack_validation_auc']
            maxdiff_c = 0.0
            for cname, crid, view in cands:
                base = cname.split('__')[-1]
                mobj = joblib.load(P / 'runs' / f'READERS_{s}' / f'MODEL__{crid}.joblib')
                vi = bank[f'{svc}__ATTACK_VALIDATION__income']; vo = bank[f'{svc}__ATTACK_VALIDATION__occupation']
                if base == 'LOOKUP':
                    keys = lookup_keys(vi, vo) if view == 'pair' else lookup_keys(vi) if view == 'income' else lookup_keys(vo)
                    t = mobj['table']; pi = mobj['pi']; a = mobj['pseudo']
                    raw = np.array([(t[k][0] + a * pi) / (t[k][1] + a) if k in t else pi for k in keys])
                else:
                    kd = KIND[base]
                    x = pair_table(vi, vo, kd) if view == 'pair' else kind_table(vi if view == 'income' else vo, kd)
                    sc = mobj['scaler']; mdl = mobj['model']
                    raw = mdl.predict_proba((x - sc.mean_) / sc.scale_)[:, 1]
                a_raw = auc(sex['ATTACK_VALIDATION'], raw)
                a_or = a_raw if a_raw >= .5 else auc(sex['ATTACK_VALIDATION'], 1 - raw)
                replay.append((cname, a_or))
                maxdiff_c = max(maxdiff_c, abs(a_or - stored_c[cname]))
            win = max(range(len(replay)), key=lambda i: (replay[i][1], -i))
            srec['selection_replay'] = {'n_candidates': len(replay), 'max_abs_auc_diff_vs_stored_candidates': maxdiff_c,
                                        'replayed_winner': replay[win][0], 'replayed_winner_auc': replay[win][1],
                                        'winner_matches_stored': replay[win][0] == name,
                                        'runner_up': sorted(replay, key=lambda c: -c[1])[1]}
            # ---- independent refit of the winner from ATTACK_FIT
            if name == 'LOOKUP' or name.startswith('IGNORE_'):
                raise SystemExit('unexpected winner kind (not handled): ' + name)
            kd = KIND[name]
            X = {r: pair_table(bank[f'{svc}__{r}__income'], bank[f'{svc}__{r}__occupation'], kd) for r in ROLES3}
            m, scl = scale_fit(X['ATTACK_FIT'])
            model = make_model(name, seed_model)
            t0 = time.process_time()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit((X['ATTACK_FIT'] - m) / scl, sex['ATTACK_FIT'])
            if list(model.classes_) != [0, 1]:
                raise SystemExit('class order')
            fit_cpu = time.process_time() - t0
            raw_v = model.predict_proba((X['ATTACK_VALIDATION'] - m) / scl)[:, 1].astype(np.float64)
            a_raw = auc(sex['ATTACK_VALIDATION'], raw_v)
            sign = 1 if a_raw >= .5 else -1
            raw_r = model.predict_proba((X['DEV_REPORT'] - m) / scl)[:, 1].astype(np.float64)
            s_rep = raw_r if sign == 1 else 1 - raw_r
            a_val = a_raw if sign == 1 else auc(sex['ATTACK_VALIDATION'], 1 - raw_v)
            st = stored[rid]
            srec.update({'refit_orientation': sign, 'stored_orientation': rseed['services'][svc]['pair_readers'][name]['orientation'],
                         'refit_attack_validation_auc': a_val,
                         'attack_validation_auc_abs_diff': abs(a_val - w['attack_validation_auc']),
                         'dev_report_scores_max_abs_diff': float(np.max(np.abs(s_rep - st))),
                         'dev_report_scores_bitwise': bool(s_rep.dtype == st.dtype and s_rep.tobytes() == st.tobytes()),
                         'n_iter': int(np.max(getattr(model, 'n_iter_', [0]))), 'fit_cpu_s': round(fit_cpu, 3),
                         'n_features': int(X['ATTACK_FIT'].shape[1])})
            ok = (srec['dev_report_scores_bitwise'] and srec['refit_orientation'] == srec['stored_orientation']
                  and srec['selection_replay']['winner_matches_stored'] and srec['attack_validation_auc_abs_diff'] < 1e-12)
            srec['PASS'] = bool(ok); all_ok &= ok
            rec['services'][svc] = srec
            print(s, svc, name, json.dumps({k: srec[k] for k in ('dev_report_scores_max_abs_diff', 'dev_report_scores_bitwise',
                                                                'attack_validation_auc_abs_diff', 'refit_orientation', 'fit_cpu_s')}),
                  'selection', json.dumps(srec['selection_replay']), flush=True)
        all_ok &= rec['bank_sha256_matches_manifest'] and rec['readers_seed_sha256_matches_reporting_lock'] \
            and rec['dev_report_scores_sha256_matches_reporting_lock'] and rec['reader_model_seed_rule_ok'] and rec['dev_report_row_ids_aligned']
        res['seeds'][f'S{s}'] = rec
    res['PASS'] = bool(all_ok)
    res['cpu_s_total'] = round(time.process_time() - t_start, 2)
    import sklearn
    res['sklearn_version'] = sklearn.__version__
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True))
    print('PASS', all_ok, 'cpu', res['cpu_s_total'])


if __name__ == '__main__':
    main()
