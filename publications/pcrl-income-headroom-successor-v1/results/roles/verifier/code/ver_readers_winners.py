"""Independent verifier, check 8b: re-derive every ENHANCED and ORIGINAL final-reader winner (income, occupation, pair views)
for every service of the final bank, both seeds, from the stored candidate ATTACK_VALIDATION AUCs, and re-check every stored
candidate AUC against the stored oriented AV score arrays with an own rank-based (Mann-Whitney, midrank) AUC.

Context: public-use U.S. Census ACS 2019 1-Year PUMS Florida person records (development roles). Readers estimate the dataset's
recorded SEX column from released income-class / occupation-group probability vectors. ATTACK_VALIDATION SEX is read through the
logged loader (after the committed FINALIST_LOCK). No DEV_REPORT label is read here.

Own code: selection rule, wire aliasing, AUC. Not reused: ffv.readers.pick, ihs.final_readers, ihs.* selection code.
Run from the worktree root through the queue.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import rankdata

from ihs import common as C   # admitted, hash-checked loaders only

SRC8 = ('OUT_LR', 'OUT_MLP', 'OUT_HGB', 'LOG_LR', 'LOG_MLP', 'DEC_LR', 'CONF_HGB', 'LOOKUP')
NEW8 = ('ET5', 'ET25', 'HGB15', 'HGB31', 'MLP64', 'MLP6432', 'CATI01', 'CATI1')
ROLES3 = ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
NEAR = 0.010
OUT = ROOT / '.private' / C.CAMPAIGN / 'verifier'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def mw_auc(y, s):
    """Mann-Whitney AUC with midranks for ties: P(s_pos > s_neg) + 0.5 P(tie)."""
    y = np.asarray(y).astype(np.int64); s = np.asarray(s, dtype=np.float64)
    r = rankdata(s, method='average')
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def choose(cands):
    """cands: list of (name, auc, key). Max AUC; exact ties go to the EARLIER candidate (strict > when scanning in order)."""
    best = 0
    for i in range(1, len(cands)):
        if cands[i][1] > cands[best][1]:
            best = i
    rest = [c[1] for i, c in enumerate(cands) if i != best]
    gap = cands[best][1] - max(rest)
    return {'candidate': cands[best][0], 'key': cands[best][2], 'auc': cands[best][1], 'gap': gap, 'near': bool(gap < NEAR),
            'n': len(cands)}


def load_bank(seed):
    man = json.loads((C.RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    p = C.RUNS / 'FINAL' / f'SERVICE_BANK_S{seed}.npz'
    assert sha256(p) == man['bank_files_sha256'][f'S{seed}'], 'bank hash differs'
    bank = {}
    with np.load(p) as z:
        for k in z.files:
            parts = k.split('__')
            if len(parts) == 3 and parts[0] not in ('row_id', 'unit'):
                bank.setdefault(parts[0], {}).setdefault(parts[1], {})[parts[2]] = z[k].copy()
    return bank


def wires(bank):
    """Own wire aliasing: services whose task output is exactly equal on all three reader roles share one wire, named after
    the first service in sorted order."""
    out, alias = {}, {}
    svcs = sorted(bank)
    for t in ('income', 'occupation'):
        groups = []
        for s in svcs:
            for g in groups:
                if all(np.array_equal(bank[g[0]][r][t], bank[s][r][t]) and bank[g[0]][r][t].dtype == bank[s][r][t].dtype for r in ROLES3):
                    g.append(s); break
            else:
                groups.append([s])
        for g in groups:
            wid = f'{g[0]}|{t}'; alias[wid] = g
            for s in g:
                out[(s, t)] = wid
    return out, alias


def main():
    V = C.load_role('ATTACK_VALIDATION', ('features', 'sex'),
                    purpose='independent verification: final reader winner re-derivation (IHS verifier)')
    yV = V['sex']
    res = {'schema': 'IHS1-verifier-winners-v1', 'seeds': {}}
    tot = {'winners_compared': 0, 'winners_match': 0, 'auc_compared': 0, 'auc_max_abs_diff_stored_vs_own': 0.0,
           'auc_max_abs_diff_stored_vs_tcr': 0.0, 'mismatches': []}
    from task_conditioned_representation.inference import auc as tcr_auc   # allowed helper (second opinion only)
    for seed in (0, 1):
        d = C.RUNS / f'READERS_S{seed}'
        rec = json.loads((d / 'READERS_SEED.json').read_text())
        assert sha256(d / 'AV_SCORES.npz') == rec['av_scores_sha256']
        with np.load(d / 'AV_SCORES.npz') as z:
            av = {k: z[k].copy() for k in z.files}
        V_ok = bool(np.array_equal(av['row_id'], V['row_id']))
        bank = load_bank(seed)
        wire_of, alias = wires(bank)
        alias_ok = {w: alias.get(w) == rec['marginal_banks'][w]['services_aliasing_this_wire'] for w in rec['marginal_banks']}
        alias_ok['_same_wire_set'] = sorted(alias) == sorted(rec['marginal_banks'])
        # 1. every stored candidate AUC vs own AUC on stored oriented arrays
        aucs = {}
        diffs = []
        for w, mb in rec['marginal_banks'].items():
            for n, r in mb['readers'].items():
                key = f'MARG__{w}__{n}'; own = mw_auc(yV, av[key]); tcr = tcr_auc(yV, av[key])
                aucs[key] = own; diffs.append((key, r['attack_validation_auc'], own, tcr))
        for svc, pr in rec['pairs'].items():
            for n, r in pr.items():
                key = f'PAIR__{svc}__{n}'; own = mw_auc(yV, av[key]); tcr = tcr_auc(yV, av[key])
                aucs[key] = own; diffs.append((key, r['attack_validation_auc'], own, tcr))
        d_own = max(abs(a - b) for _, a, b, _ in diffs); d_tcr = max(abs(a - c) for _, a, _, c in diffs)
        below_half = [k for k, a, b, _ in diffs if b < 0.5 - 1e-12]
        tot['auc_compared'] += len(diffs)
        tot['auc_max_abs_diff_stored_vs_own'] = max(tot['auc_max_abs_diff_stored_vs_own'], d_own)
        tot['auc_max_abs_diff_stored_vs_tcr'] = max(tot['auc_max_abs_diff_stored_vs_tcr'], d_tcr)
        # 2. controls (AV): stored vs own
        ctrl = {}
        for view, cr in rec['controls'].items():
            for n, c in cr.items():
                ks, kp = f'CONTROL__{view}__{n}__SHUFFLED', f'CONTROL__{view}__{n}__PLANTED'
                ctrl[f'{view}|{n}'] = {'shuffled_stored': c['shuffled_attack_validation_auc'], 'shuffled_own': mw_auc(yV, av[ks]),
                                      'planted_stored': c['planted_attack_validation_auc'], 'planted_own': mw_auc(yV, av[kp]),
                                      'inside_bands_own': bool(mw_auc(yV, av[kp]) >= 0.95 and abs(mw_auc(yV, av[ks]) - 0.5) <= 0.05)}
        # 3. winners re-derived from the STORED candidate AUCs (rule check) and from own AUCs (data check)
        win = {}
        for svc in rec['roster']:
            st = rec['winners'][svc]; win[svc] = {}
            for src_name, A in (('stored_auc', None), ('own_auc', aucs)):
                def a_marg(w, n):
                    return rec['marginal_banks'][w]['readers'][n]['attack_validation_auc'] if A is None else A[f'MARG__{w}__{n}']
                def a_pair(n):
                    return rec['pairs'][svc][n]['attack_validation_auc'] if A is None else A[f'PAIR__{svc}__{n}']
                mine = {'original': {}, 'enhanced': {}}
                for view in ('income', 'occupation'):
                    w = wire_of[(svc, view)]
                    src = [(n, a_marg(w, n), f'MARG__{w}__{n}') for n in SRC8]
                    new = [(n, a_marg(w, n), f'MARG__{w}__{n}') for n in NEW8]
                    mine['original'][view] = choose(src); mine['enhanced'][view] = choose(src + new)
                wi, wo = wire_of[(svc, 'income')], wire_of[(svc, 'occupation')]
                src = [(n, a_pair(n), f'PAIR__{svc}__{n}') for n in SRC8]
                src += [(f'IGNORE_occupation__{n}', a_marg(wi, n), f'MARG__{wi}__{n}') for n in SRC8]
                src += [(f'IGNORE_income__{n}', a_marg(wo, n), f'MARG__{wo}__{n}') for n in SRC8]
                new = [(n, a_pair(n), f'PAIR__{svc}__{n}') for n in NEW8]
                new += [(f'IGNORE_occupation__{n}', a_marg(wi, n), f'MARG__{wi}__{n}') for n in NEW8]
                new += [(f'IGNORE_income__{n}', a_marg(wo, n), f'MARG__{wo}__{n}') for n in NEW8]
                mine['original']['pair'] = choose(src); mine['enhanced']['pair'] = choose(src + new)
                cmp = {}
                for bank_name in ('original', 'enhanced'):
                    for view in ('income', 'occupation', 'pair'):
                        m, s = mine[bank_name][view], st[bank_name][view]
                        ok = {'candidate': m['candidate'] == s['candidate'], 'key': m['key'] == s['reader_key'],
                              'auc_abs_diff': abs(m['auc'] - s['attack_validation_auc']), 'near_tie': m['near'] == s['AV_NEAR_TIE'],
                              'gap_abs_diff': abs(m['gap'] - s['gap_to_runner_up']), 'n_candidates': m['n'] == s['n_candidates']}
                        if bank_name == 'enhanced':
                            o = mine['original'][view]
                            same = m['key'] == o['key']; gap_o = m['auc'] - o['auc']
                            ok['same_as_original'] = same == s['same_as_original']
                            ok['gap_to_original_abs_diff'] = abs(gap_o - s['gap_to_original_winner'])
                            ok['near_tie_to_original'] = bool((not same) and gap_o < NEAR) == s['AV_NEAR_TIE_TO_ORIGINAL']
                        match = all(v for k, v in ok.items() if isinstance(v, bool)) and ok['auc_abs_diff'] <= 1e-12 and \
                            ok['gap_abs_diff'] <= 1e-12 and ok.get('gap_to_original_abs_diff', 0.0) <= 1e-12
                        cmp[f'{bank_name}|{view}'] = {'mine': [m['candidate'], m['auc'], m['near'], m['gap']],
                                                      'stored': [s['candidate'], s['attack_validation_auc'], s['AV_NEAR_TIE'], s['gap_to_runner_up']],
                                                      'checks': ok, 'match': bool(match)}
                        if src_name == 'stored_auc':
                            tot['winners_compared'] += 1; tot['winners_match'] += int(match)
                        if not match:
                            tot['mismatches'].append({'seed': seed, 'service': svc, 'bank': bank_name, 'view': view, 'basis': src_name,
                                                      'mine': cmp[f'{bank_name}|{view}']['mine'], 'stored': cmp[f'{bank_name}|{view}']['stored'], 'checks': ok})
                win[svc][src_name] = cmp
        res['seeds'][f'S{seed}'] = {'row_ids_equal_ATTACK_VALIDATION_export': V_ok, 'wire_aliasing_matches_stored': alias_ok,
                                    'n_candidate_aucs': len(diffs), 'max_abs_diff_stored_vs_own_auc': d_own,
                                    'max_abs_diff_stored_vs_tcr_auc': d_tcr, 'oriented_auc_below_half': below_half,
                                    'controls_av': ctrl, 'winners': win, 'not_converged_stored': rec['not_converged']}
    res['totals'] = tot
    OUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    (OUT / 'WINNERS_CHECK.json').write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    print(json.dumps({'totals': tot, 'per_seed': {s: {k: v for k, v in r.items() if k in ('row_ids_equal_ATTACK_VALIDATION_export',
                      'n_candidate_aucs', 'max_abs_diff_stored_vs_own_auc', 'max_abs_diff_stored_vs_tcr_auc', 'oriented_auc_below_half',
                      'not_converged_stored')} | {'alias_all_ok': all(r['wire_aliasing_matches_stored'].values()),
                      'controls_all_inside_own': all(c['inside_bands_own'] for c in r['controls_av'].values()),
                      'controls_max_abs_diff': max(max(abs(c['shuffled_stored'] - c['shuffled_own']), abs(c['planted_stored'] - c['planted_own'])) for c in r['controls_av'].values())}
                      for s, r in res['seeds'].items()}}, indent=1, default=float))


if __name__ == '__main__':
    main()
