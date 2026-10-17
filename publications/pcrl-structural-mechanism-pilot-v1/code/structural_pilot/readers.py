"""Fresh recovery readers for one model seed over this campaign's SERVICE_BANK (exploratory development pilot).

Source slate and transformations (acs_replication.readers / acs_hybrids.readers; definitions in
task_conditioned_representation.attacks): OUT_LR, OUT_MLP, OUT_HGB on probabilities + decision; LOG_LR, LOG_MLP on
log-probabilities + one-hot decision; DEC_LR on the one-hot decision; CONF_HGB on the max probability. Views: income
recipient, occupation recipient, linked pair.
- Fit on ATTACK_FIT with SEX; orientation and every winner on ATTACK_VALIDATION only; DEV_REPORT scored from released
  outputs alone (no DEV_REPORT label is opened here). No reader sees MODEL_FIT, MODEL_SELECT, DEV_REPORT truth or sealed people.
- RNG rule: every reader uses model seed MODEL_SEED_BASE + s, independent of service, view and reader name.
- Marginal banks are fitted once per unique single-recipient wire; services sharing a wire alias it.
- Pair candidates per service: 8 pair readers (the inherited 7 plus the frozen LOOKUP reader) + 16 ignore-one aliases (its
  own income-only and occupation-only marginal readers). Winner: max ATTACK_VALIDATION AUC with fixed orientation; ties go to
  the earlier candidate. The winner over the inherited 7-reader slate (7 + 14 candidates) is recorded as a secondary figure.
- Controls on the nominee's wire, per view: shuffled-SEX null (fixed permutation of ATTACK_FIT SEX, seed 2026101602 + s;
  OUT_HGB) and a planted SEX bit (OUT_HGB on the wire plus the SEX column).
- Learning-curve diagnostic (pre-declared, never primary): the pair slate refitted on one household subset of ATTACK_FIT
  (LC_FRACTION of households, identical subset for every listed arm), winner by ATTACK_VALIDATION among the 7 pair readers.
- Label-free replay check: reader scores on wires the frozen-hybrids follow-up also attacked are compared with its stored
  DEV_REPORT reader scores (recorded only; never a reason to change a reader).
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

from . import common as C

SLATE = (('OUT_LR', 'out'), ('OUT_MLP', 'out'), ('OUT_HGB', 'out'), ('LOG_LR', 'log'), ('LOG_MLP', 'log'), ('DEC_LR', 'dec'), ('CONF_HGB', 'conf'))
NULL_SEED = 2026101602
LC_SEED = 2026101704
VIEWS = ('income', 'occupation', 'pair')
ROLES3 = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
HYB_LABEL = {'NEURAL_RAW': 'NEURAL_RAW', 'TASK_REFIT': 'TASK_REFIT', 'T_TREE': 'T_TREE', 'H_LL': 'H_LL', 'H_LJ': 'H_LJ', 'H_JJ': 'H_JJ'}


LOOKUP_PSEUDO = 2.0   # frozen before any fit (SERVICE_AND_THREAT_MODEL.md section 3)


def kinds(p, k):
    p = np.asarray(p, dtype=np.float64); d = p.argmax(1); one = np.eye(k)[d]
    return {'out': np.column_stack((p, d.astype(np.float64))), 'log': np.column_stack((np.log(np.maximum(p, 1e-300)), one)),
            'dec': one, 'conf': p.max(1, keepdims=True)}


def lookup_keys(*vectors):
    """Exact float64 bytes of each row's released vector(s), concatenated in recipient order."""
    z = np.ascontiguousarray(np.column_stack([np.asarray(v, dtype=np.float64) for v in vectors]))
    return [r.tobytes() for r in z]


def fit_lookup(keys, y):
    y = np.asarray(y); pi = float(y.mean()); table = {}
    for k, v in zip(keys, y):
        n1, n = table.get(k, (0, 0)); table[k] = (n1 + int(v), n + 1)
    return {'pi': pi, 'table': table, 'pseudo': LOOKUP_PSEUDO}


def predict_lookup(rec, keys):
    pi, a, t = rec['pi'], rec['pseudo'], rec['table']
    return np.array([(t[k][0] + a * pi) / (t[k][1] + a) if k in t else pi for k in keys], dtype=np.float64)


def load_service_bank(seed):
    man = json.loads((C.RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    p = C.PRIV / 'bank' / f'SERVICE_BANK_S{seed}.npz'
    if C.sha(p) != man['bank_files_sha256'][f'S{seed}']:
        raise PermissionError('service bank hash differs from SERVICE_BANK_MANIFEST')
    out, ids = {}, {}
    with np.load(p) as z:
        for key in z.files:
            parts = key.split('__')
            if parts[0] in ('row_id', 'unit'):
                ids[key] = z[key].copy(); continue
            svc, role, task = parts
            out.setdefault(svc, {}).setdefault(role, {})[task] = z[key].copy()
    return out, ids, man


def run(seed, out_dir):
    from task_conditioned_representation.attacks import MODEL_SEED_BASE, fit_reader, predict_reader, orientation, orient
    from task_conditioned_representation.inference import auc
    for name in ('ANALYSIS_LOCK.json', 'SELECTION_LOCK.json'):
        if not C.committed(C.RESULTS / name):
            raise PermissionError(f'reader fits require the committed {name}')
    lock = json.loads((C.RESULTS / 'ANALYSIS_LOCK.json').read_text())
    roster = lock['readers']['attack_roster']; nominee = lock['readers']['control_wire']; lc_arms = lock['readers']['learning_curve_arms']
    lc_fraction = lock['readers']['learning_curve_household_fraction']
    out = Path(out_dir)
    if out.exists():
        raise PermissionError('fresh reader directory required')
    out.mkdir(parents=True, mode=0o700)
    A = C.load_role('ATTACK_FIT', ('features', 'sex')); V = C.load_role('ATTACK_VALIDATION', ('features', 'sex'))
    R = C.load_role('DEV_REPORT', ('features',))
    bank, ids, man = load_service_bank(seed)
    for r, X in (('ATTACK_FIT', A), ('ATTACK_VALIDATION', V), ('DEV_REPORT', R)):
        if not (np.array_equal(ids[f'row_id__{r}'], X['row_id']) and np.array_equal(ids[f'unit__{r}'], X['unit'])):
            raise AssertionError(f'bank row order differs from the {r} export')
    seed_model = MODEL_SEED_BASE + seed
    rec = {'schema': 'SMP1-readers-v1', 'seed': seed, 'reader_model_seed': seed_model, 'rng_rule': 'MODEL_SEED_BASE + s for every service, view and reader',
           'roles': {'train': 'ATTACK_FIT', 'select': 'ATTACK_VALIDATION', 'score': 'DEV_REPORT'},
           'rows': {'ATTACK_FIT': int(len(A['row_id'])), 'ATTACK_VALIDATION': int(len(V['row_id'])), 'DEV_REPORT': int(len(R['row_id']))},
           'analysis_lock_sha256': C.sha(C.RESULTS / 'ANALYSIS_LOCK.json'), 'selection_lock_sha256': C.sha(C.RESULTS / 'SELECTION_LOCK.json'),
           'bank_sha256': man['bank_files_sha256'][f'S{seed}'], 'marginal_banks': {}, 'services': {}, 'controls': {}, 'learning_curve': {},
           'accounting': [], 'aliases': {}}
    scores, models = {}, {}

    def fit_one(x_fit, y_fit, x_val, x_rep, name, rid):
        t0 = time.process_time()
        if name == 'LOOKUP':
            m = fit_lookup(x_fit, y_fit); fit_pred = predict_lookup
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = fit_reader(x_fit, y_fit, name, seed_model)
            fit_pred = predict_reader
        mp = out / f'MODEL__{rid}.joblib'; joblib.dump(m, mp, compress=3); mp.chmod(0o600)
        raw = fit_pred(m, x_val); sign = orientation(V['sex'], raw)
        val_auc = auc(V['sex'], orient(raw, sign)); s_rep = orient(fit_pred(m, x_rep), sign)
        scores[rid] = s_rep; models[rid] = C.sha(mp)
        rec['accounting'].append({'reader_id': rid, 'reader': name, 'cpu_s': round(time.process_time() - t0, 4)})
        out_rec = {'attack_validation_auc': float(val_auc), 'orientation': int(sign), 'model_sha256': models[rid]}
        if name == 'LOOKUP':
            out_rec['distinct_fit_keys'] = len(m['table'])
            out_rec['share_val_keys_seen_in_fit'] = float(np.mean([k in m['table'] for k in x_val]))
        return out_rec

    # ---- marginal banks once per unique single-recipient wire
    wire_of, uniq = {}, {}
    for svc in roster:
        for t in C.TASKS:
            h = tuple(C.wire_sha(bank[svc][r][t]) for r in ROLES3)
            uniq.setdefault((t, h), []).append(svc)
    names8 = [n for n, _ in SLATE] + ['LOOKUP']
    for (t, h), svcs in uniq.items():
        k = C.K[t]; wid = f'{svcs[0]}|{t}'; svc0 = svcs[0]
        tf, tv, tr = (kinds(bank[svc0][r][t], k) for r in ROLES3)
        mb = {'task': t, 'services_aliasing_this_bank': svcs, 'readers': {}}
        for name, kind in SLATE:
            mb['readers'][name] = fit_one(tf[kind], A['sex'], tv[kind], tr[kind], name, f'MARG__{wid}__{name}')
        kf, kv, kr = (lookup_keys(bank[svc0][r][t]) for r in ROLES3)
        mb['readers']['LOOKUP'] = fit_one(kf, A['sex'], kv, kr, 'LOOKUP', f'MARG__{wid}__LOOKUP')
        rec['marginal_banks'][wid] = mb; rec['aliases'][wid] = svcs
        for svc in svcs:
            wire_of[(svc, t)] = wid
    # ---- pair readers per service and winners (primary: 8-reader bank incl. LOOKUP; secondary: inherited 7-reader slate)
    def pair_tables(svc, rows=None):
        tabs = {}
        for r in ROLES3:
            a, b = kinds(bank[svc][r]['income'], 2), kinds(bank[svc][r]['occupation'], 5)
            tabs[r] = {kk: np.column_stack((a[kk], b[kk])) for kk in a}
        return tabs

    def pick(cands):
        win = max(range(len(cands)), key=lambda i: (cands[i][1], -i))
        return {'candidate': cands[win][0], 'reader_id': cands[win][2], 'attack_validation_auc': cands[win][1], 'n_candidates': len(cands)}
    for svc in roster:
        srec = {'income_wire': wire_of[(svc, 'income')], 'occupation_wire': wire_of[(svc, 'occupation')], 'pair_readers': {}, 'winner': {}, 'winner_slate7': {}}
        tabs = pair_tables(svc)
        for name, kind in SLATE:
            srec['pair_readers'][name] = fit_one(tabs['ATTACK_FIT'][kind], A['sex'], tabs['ATTACK_VALIDATION'][kind], tabs['DEV_REPORT'][kind], name, f'PAIR__{svc}__{name}')
        kf, kv, kr = (lookup_keys(bank[svc][r]['income'], bank[svc][r]['occupation']) for r in ROLES3)
        srec['pair_readers']['LOOKUP'] = fit_one(kf, A['sex'], kv, kr, 'LOOKUP', f'PAIR__{svc}__LOOKUP')
        for names, key in ((names8, 'winner'), (names8[:7], 'winner_slate7')):
            for view in ('income', 'occupation'):
                mb = rec['marginal_banks'][wire_of[(svc, view)]]['readers']
                srec[key][view] = pick([(n, mb[n]['attack_validation_auc'], f"MARG__{wire_of[(svc, view)]}__{n}") for n in names])
            cands = [(n, srec['pair_readers'][n]['attack_validation_auc'], f'PAIR__{svc}__{n}') for n in names]
            cands += [(f'IGNORE_occupation__{n}', rec['marginal_banks'][wire_of[(svc, 'income')]]['readers'][n]['attack_validation_auc'],
                       f"MARG__{wire_of[(svc, 'income')]}__{n}") for n in names]
            cands += [(f'IGNORE_income__{n}', rec['marginal_banks'][wire_of[(svc, 'occupation')]]['readers'][n]['attack_validation_auc'],
                       f"MARG__{wire_of[(svc, 'occupation')]}__{n}") for n in names]
            srec[key]['pair'] = pick(cands)
            if key == 'winner':
                srec['pair_candidates_attack_validation_auc'] = {c[0]: c[1] for c in cands}
        rec['services'][svc] = srec
    # ---- controls on the nominee wire
    shuffled = np.random.default_rng(NULL_SEED + seed).permutation(A['sex'])
    for view in VIEWS:
        def tab(r):
            a, b = kinds(bank[nominee][r]['income'], 2)['out'], kinds(bank[nominee][r]['occupation'], 5)['out']
            return a if view == 'income' else b if view == 'occupation' else np.column_stack((a, b))
        x, xv, xr = tab('ATTACK_FIT'), tab('ATTACK_VALIDATION'), tab('DEV_REPORT')
        t0 = time.process_time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rs = fit_reader(x, shuffled, 'OUT_HGB', seed_model)
            rp = fit_reader(np.column_stack((x, A['sex'])), A['sex'], 'OUT_HGB', seed_model)
        scores[f'CONTROL__{view}__SHUFFLED'] = predict_reader(rs, xr)
        for bit in (0, 1):
            scores[f'CONTROL__{view}__PLANTED_{bit}'] = predict_reader(rp, np.column_stack((xr, np.full(len(xr), bit))))
        rec['controls'][view] = {'wire': nominee, 'shuffled_attack_validation_auc': auc(V['sex'], predict_reader(rs, xv)),
                                 'planted_attack_validation_auc': auc(V['sex'], predict_reader(rp, np.column_stack((xv, V['sex']))))}
        rec['accounting'].append({'reader_id': f'CONTROL__{view}', 'reader': 'OUT_HGB x2', 'cpu_s': round(time.process_time() - t0, 4)})
    # ---- learning-curve diagnostic: one household subset of ATTACK_FIT, identical across arms (pair slate only)
    groups = np.unique(A['unit']); perm = np.random.default_rng(LC_SEED).permutation(len(groups))
    keep = np.isin(A['unit'], groups[perm[:int(round(lc_fraction * len(groups)))]])
    rec['learning_curve'] = {'household_fraction': lc_fraction, 'households': int(round(lc_fraction * len(groups))), 'persons': int(keep.sum()),
                             'subset_sha256': C.arr_sha(A['row_id'][keep]), 'arms': {}}
    for svc in lc_arms:
        tabs = pair_tables(svc)
        res = {}
        for name, kind in SLATE:
            res[name] = fit_one(tabs['ATTACK_FIT'][kind][keep], A['sex'][keep], tabs['ATTACK_VALIDATION'][kind], tabs['DEV_REPORT'][kind], name, f'LC__{svc}__{name}')
        win = max(range(len(SLATE)), key=lambda i: (res[SLATE[i][0]]['attack_validation_auc'], -i))
        rec['learning_curve']['arms'][svc] = {'pair_readers': res, 'winner': {'candidate': SLATE[win][0], 'reader_id': f'LC__{svc}__{SLATE[win][0]}',
                                                                              'attack_validation_auc': res[SLATE[win][0]]['attack_validation_auc']}}
    # ---- label-free replay against the frozen-hybrids stored DEV_REPORT reader scores
    replay = {}
    hp = C.HYB_PRIV / 'runs' / f'READERS_{seed}' / 'DEV_REPORT_SCORES.npz'
    if hp.is_file():
        with np.load(hp) as z:
            hs = {k: z[k].copy() for k in z.files}
        if np.array_equal(hs['row_id'], R['row_id']):
            for svc, hl in HYB_LABEL.items():
                if svc not in roster:
                    continue
                for name, _ in SLATE:
                    key = f'PAIR__{hl}__{name}'
                    if key in hs:
                        a = scores[f'PAIR__{svc}__{name}']
                        replay[key] = {'max_abs_diff': float(np.max(np.abs(a - hs[key]))), 'bitwise': bool(np.array_equal(a, hs[key]))}
    rec['replay_vs_frozen_hybrids'] = {'compared': len(replay), 'bitwise': sum(v['bitwise'] for v in replay.values()),
                                       'max_abs_diff': max((v['max_abs_diff'] for v in replay.values()), default=None), 'detail': replay}
    sp = out / 'DEV_REPORT_SCORES.npz'
    with sp.open('xb') as f:
        np.savez(f, row_id=R['row_id'], unit=R['unit'], **scores)
    sp.chmod(0o600)
    rec['dev_report_scores_sha256'] = C.sha(sp); rec['model_files_sha256'] = models
    n_marg = sum(len(m['readers']) for m in rec['marginal_banks'].values()); n_pair = sum(len(s['pair_readers']) for s in rec['services'].values())
    rec['fit_counts'] = {'marginal_unique_wire_fits': n_marg, 'pair_fits': n_pair, 'control_fits': 6, 'learning_curve_fits': 7 * len(lc_arms),
                         'effective_numerical_fits': n_marg + n_pair + 6 + 7 * len(lc_arms)}
    rec['DEV_REPORT_labels_opened_or_hashed'] = False
    rec['sklearn_version'] = __import__('sklearn').__version__
    with (out / 'READERS_SEED.json').open('x') as f:
        json.dump(rec, f, indent=2, sort_keys=True)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); r = run(a.seed, a.out)
    print(json.dumps({'seed': r['seed'], 'fit_counts': r['fit_counts'],
                      'winners': {k: {v: w['candidate'] for v, w in s['winner'].items()} for k, s in r['services'].items()},
                      'controls': r['controls'], 'replay': {k: r['replay_vs_frozen_hybrids'][k] for k in ('compared', 'bitwise', 'max_abs_diff')}}, indent=1))
