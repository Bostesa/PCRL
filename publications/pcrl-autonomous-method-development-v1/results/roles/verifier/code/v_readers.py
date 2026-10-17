"""Verifier unit AMD_VERIFY_READERS_V1: check 7 (refit of the frozen 14-reader subset in ANALYSIS_LOCK.json) and a label-free
re-application of the ATTACK_VALIDATION winner rule to the recorded reader AUCs. Readers are refitted on ATTACK_FIT features of
the released outputs with SEX, using the source constructor task_conditioned_representation.attacks.fit_reader (seed
MODEL_SEED_BASE + s); features, LOOKUP, orientation and AUC are this verifier's own code. No DEV_REPORT label is read."""
from __future__ import annotations
import json
import sys
import time
import warnings

sys.path.insert(0, '<repository>/.claude/worktrees/pcrl-autonomous-method-development-v1/.private/pcrl_autonomous_method_development_v1/verifier/independent')
import numpy as np
import vcommon as V
from amd import common as C
from task_conditioned_representation.attacks import fit_reader, MODEL_SEED_BASE

KIND = {'OUT_LR': 'out', 'OUT_MLP': 'out', 'OUT_HGB': 'out', 'LOG_LR': 'log', 'LOG_MLP': 'log', 'DEC_LR': 'dec', 'CONF_HGB': 'conf'}
NAMES8 = ['OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP']


def feat(p, k, kind):
    p = np.asarray(p, dtype=np.float64); d = np.argmax(p, axis=1)
    one = np.zeros((len(p), k)); one[np.arange(len(p)), d] = 1.0
    if kind == 'out':
        return np.concatenate([p, d[:, None].astype(np.float64)], axis=1)
    if kind == 'log':
        return np.concatenate([np.log(np.maximum(p, 1e-300)), one], axis=1)
    if kind == 'dec':
        return one
    return p.max(axis=1)[:, None]


def keys(*vs):
    z = np.ascontiguousarray(np.concatenate([np.asarray(v, dtype=np.float64) for v in vs], axis=1))
    return [bytes(r.tobytes()) for r in z]


def lookup_fit(k, y):
    pi = float(np.mean(y)); tab = {}
    for kk, v in zip(k, y):
        a, n = tab.get(kk, (0, 0)); tab[kk] = (a + int(v), n + 1)
    return pi, tab


def lookup_pred(m, k):
    pi, tab = m
    return np.array([(tab[x][0] + 2.0 * pi) / (tab[x][1] + 2.0) if x in tab else pi for x in k], dtype=np.float64)


lock = json.loads((V.RES / 'ANALYSIS_LOCK.json').read_text())
A = C.load_role('ATTACK_FIT', ('features', 'sex')); VA = C.load_role('ATTACK_VALIDATION', ('features', 'sex')); R = C.load_role('DEV_REPORT', ('features',))
out = {'unit': 'AMD_VERIFY_READERS_V1', 'utc': C.now(), 'refits': [], 'winner_rule': {}}
for s in (0, 1):
    with np.load(V.PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz') as z:
        B = {k: z[k].copy() for k in z.files}
    for r, X in (('ATTACK_FIT', A), ('ATTACK_VALIDATION', VA), ('DEV_REPORT', R)):
        if not np.array_equal(B[f'row_id__{r}'], X['row_id']):
            raise AssertionError('bank order differs')
    rd = V.PRIV / 'runs' / f'READERS_{s}'
    rr = json.loads((rd / 'READERS_SEED.json').read_text())
    with np.load(rd / 'DEV_REPORT_SCORES.npz') as z:
        stored = {k: z[k].copy() for k in z.files}
    # ---- winner rule re-applied to the recorded ATTACK_VALIDATION AUCs (max; ties to the first candidate)
    wr = {}
    for svc, sv in rr['services'].items():
        for view in ('income', 'occupation'):
            mb = rr['marginal_banks'][sv[f'{view}_wire']]['readers']
            c = [(n, mb[n]['attack_validation_auc'], f"MARG__{sv[f'{view}_wire']}__{n}") for n in NAMES8]
            best = max(range(len(c)), key=lambda i: (c[i][1], -i))
            wr[f'{svc}|{view}'] = c[best][2] == sv['winner'][view]['reader_id']
        c = [(n, sv['pair_readers'][n]['attack_validation_auc'], f'PAIR__{svc}__{n}') for n in NAMES8]
        c += [(f'IGNORE_occupation__{n}', rr['marginal_banks'][sv['income_wire']]['readers'][n]['attack_validation_auc'], f"MARG__{sv['income_wire']}__{n}") for n in NAMES8]
        c += [(f'IGNORE_income__{n}', rr['marginal_banks'][sv['occupation_wire']]['readers'][n]['attack_validation_auc'], f"MARG__{sv['occupation_wire']}__{n}") for n in NAMES8]
        best = max(range(len(c)), key=lambda i: (c[i][1], -i))
        wr[f'{svc}|pair'] = c[best][2] == sv['winner']['pair']['reader_id']
    out['winner_rule'][f'S{s}'] = {'all_match': all(wr.values()), 'n': len(wr), 'detail': wr,
                                   'winners': {svc: {v: sv['winner'][v]['reader_id'] for v in ('income', 'occupation', 'pair')} for svc, sv in rr['services'].items()}}
    for item in lock['refit_subset']:
        if item['seed'] != s:
            continue
        rid = item['reader_id']; t0 = time.process_time()
        parts = rid.split('__')
        if parts[0] == 'PAIR':
            svc, name = parts[1], parts[2]
            if name == 'LOOKUP':
                kf, kv, kr = (keys(B[f'{svc}__{r}__income'], B[f'{svc}__{r}__occupation']) for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'))
            else:
                kd = KIND[name]
                xf, xv, xr = (np.concatenate([feat(B[f'{svc}__{r}__income'], 2, kd), feat(B[f'{svc}__{r}__occupation'], 5, kd)], axis=1)
                              for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'))
            rec_av = rr['services'][svc]['pair_readers'][name]
        else:
            wid, name = parts[1], parts[2]
            svc, t = wid.split('|')
            if name == 'LOOKUP':
                kf, kv, kr = (keys(B[f'{svc}__{r}__{t}']) for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'))
            else:
                kd = KIND[name]
                xf, xv, xr = (feat(B[f'{svc}__{r}__{t}'], V.KCLS[t], kd) for r in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'))
            rec_av = rr['marginal_banks'][wid]['readers'][name]
        if name == 'LOOKUP':
            m = lookup_fit(kf, A['sex']); sv_, sr_ = lookup_pred(m, kv), lookup_pred(m, kr)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = fit_reader(xf, A['sex'], name, MODEL_SEED_BASE + s)
            if not np.array_equal(m['model'].classes_, [0, 1]):
                raise AssertionError('class order')
            sv_ = m['model'].predict_proba(m['scaler'].transform(xv))[:, 1].astype(np.float64)
            sr_ = m['model'].predict_proba(m['scaler'].transform(xr))[:, 1].astype(np.float64)
        a_raw = V.auc_point(VA['sex'], sv_); sign = 1 if a_raw >= 0.5 else -1
        av = a_raw if sign == 1 else V.auc_point(VA['sex'], 1 - sv_)
        dev = sr_ if sign == 1 else 1 - sr_
        st = stored[rid]
        out['refits'].append({'seed': s, 'reader_id': rid, 'orientation': sign, 'recorded_orientation': rec_av['orientation'],
                              'attack_validation_auc': av, 'recorded_attack_validation_auc': rec_av['attack_validation_auc'],
                              'av_auc_abs_diff': abs(av - rec_av['attack_validation_auc']),
                              'dev_scores_bitwise': bool(np.array_equal(dev, st)), 'dev_scores_max_abs_diff': float(np.max(np.abs(dev - st))),
                              'cpu_s': round(time.process_time() - t0, 2)})
        print(out['refits'][-1], flush=True)
ok = all(r['dev_scores_max_abs_diff'] <= 1e-12 and r['orientation'] == r['recorded_orientation'] and r['av_auc_abs_diff'] <= 1e-12 for r in out['refits'])
out['pass'] = bool(ok and len(out['refits']) == 14)
out['winner_rule_pass'] = all(v['all_match'] for v in out['winner_rule'].values())
V.dump('V_READERS.json', out)
print(json.dumps({'pass': out['pass'], 'winner_rule_pass': out['winner_rule_pass'], 'n': len(out['refits'])}))
