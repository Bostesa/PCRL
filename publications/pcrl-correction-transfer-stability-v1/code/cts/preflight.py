"""Preflight of the PCRL correction-transfer campaign: source replay and representative source-output checks.

    python3 -m cts.preflight outputs     label-free: representative source occupation, income and correction outputs recomputed from
                                         admitted features (witness A1_PO_16 / B2_SINGLE, H3 income, P2 corrections) vs their stored
                                         releases; results/<campaign>/preflight/SOURCE_OUTPUTS.json
    python3 -m cts.preflight replay      operation 'source_replay' (after the registration receipt): the source cost curves and
                                         projection decisions of P2_PAIR / P2A_PAIR (seeds 0, 1) recomputed from the stored out-of-fold
                                         arrays with ohn.protect.project and compared with the recorded scales and statistics; and the
                                         source's realized MODEL_SELECT incremental costs of the selected B80 / B50 releases against their
                                         witnesses (the source's transfer error); results/<campaign>/preflight/SOURCE_REPLAY.json
No new release is fitted or evaluated here; the repair panels are read only as the source's own MODEL_FIT rows.
"""
from __future__ import annotations
import argparse
import json
import sys
import time

import numpy as np

from . import common as C

REPLAYS = (('P2_PAIR', 'B', 'B2_SINGLE'), ('P2A_PAIR', 'A', 'A1_PO_16'))


def outputs():
    from ohn import fam_a, fam_b, protect as PR, devrun as OD
    t0 = time.process_time()
    res = {}
    S = C.features('MODEL_SELECT')
    X, names = S['X'], S['feature_names']
    for fam, mod, cfg in (('A', fam_a, 'A1_PO_16'), ('B', fam_b, 'B2_SINGLE')):
        for s in C.SEEDS:
            occ, rec, _ = OD.load_occupation(fam, cfg, s)
            p = mod.predict(s, cfg, X, names)
            res[f'{fam}/{cfg}/S{s}'] = {'bitwise': bool(np.array_equal(p, occ['MODEL_SELECT'])), 'max_abs_diff': float(np.abs(p - occ['MODEL_SELECT']).max())}
    for cid in ('P2_PAIR_B80', 'P2A_PAIR_B80'):
        for s in C.SEEDS:
            occ, rec, _ = OD.load_occupation('PROT', cid, s)
            p = PR.predict(s, cid, X, names)
            res[f'PROT/{cid}/S{s}'] = {'bitwise': bool(np.array_equal(p, occ['MODEL_SELECT'])), 'max_abs_diff': float(np.abs(p - occ['MODEL_SELECT']).max())}
    from ihs import fam_h
    from ohn import common as OC
    for s in C.SEEDS:
        rel, rec = OC.ihs_release('H', 'H3_STRONG_LJDIR_PB', s)
        p = fam_h.predict(s, 'H3_STRONG_LJDIR_PB', X, names)
        stored = rel['MODEL_SELECT']['income']
        res[f'H/H3_STRONG_LJDIR_PB/S{s}'] = {'bitwise': bool(np.array_equal(p['income'] if isinstance(p, dict) else p, stored)),
                                             'max_abs_diff': float(np.abs((p['income'] if isinstance(p, dict) else p) - stored).max()),
                                             'release_sha256_checked': True}
    out = {'schema': 'CTS1-source-outputs-v1', 'role': 'MODEL_SELECT (features only)', 'checks': res,
           'all_within_1e-12': all(v['max_abs_diff'] <= 1e-12 for v in res.values()), 'interpreter': sys.executable,
           'labels_read': False, 'cpu_s': time.process_time() - t0, 'utc': C.now()}
    C.write_json(C.RESULTS / 'preflight' / 'SOURCE_OUTPUTS.json', out)
    print(json.dumps({'all_within_1e-12': out['all_within_1e-12'], 'checks': res}))
    return out


def replay():
    from ohn import protect as PR, devrun as OD
    t0 = time.process_time()
    F = C.load('MODEL_FIT', ('features', 'occupation', 'sex'), 'source_replay',
               'source replay: recompute the source P2 cost curves and B80 / B50 projection decisions from the stored out-of-fold arrays '
               '(rows = all MODEL_FIT, as the source used them)')
    y, sex, unit = F['occupation'], F['sex'], F['unit']
    S = C.load('MODEL_SELECT', ('features', 'occupation', 'sex'), 'source_replay',
               "source replay: the source's realized MODEL_SELECT incremental costs of its selected B80 / B50 releases against their witnesses")
    out = {'schema': 'CTS1-source-replay-v1', 'replays': {}}
    for stem, wf, wc in REPLAYS:
        for s in C.SEEDS:
            trec = json.loads((C.OHN_RUNS / 'PROT_TRAIN' / stem / f'S{s}' / 'TRAIN.json').read_text())
            with np.load(C.OHN_RUNS / 'PROT_TRAIN' / stem / f'S{s}' / 'TRAIN.npz') as z:
                delta_oof = z['delta_oof'].copy()
            wocc, wrec, wcf = OD.load_occupation(wf, wc, s)
            p_oof = np.asarray(wcf['MODEL_FIT__oof'], dtype=np.float64)
            if not np.array_equal(wcf['fold'], C.household_folds(unit, 5, 'OHN1_crossfit')):
                raise AssertionError('source cross-fitting folds differ')
            z_oof = PR.logp(p_oof)
            rep = {}
            for lvl in ('B80', 'B50'):
                if lvl not in trec['proj']:
                    continue
                budget = trec['budgets'][lvl]
                scale, st, curve, base = PR.project(z_oof, delta_oof, y, sex, budget)
                rec_scale = trec['proj'][lvl]['scale']
                rec_st = trec['proj'][lvl]['oof_stats']
                r = {'recorded_scale': rec_scale, 'replayed_scale': scale, 'scale_abs_diff': abs(scale - rec_scale),
                     'stats_max_abs_diff': max(abs(st[k] - rec_st[k]) for k in rec_st), 'budget': budget,
                     'binding': max(PR.STATS, key=lambda k: st[k] / budget[k] if budget[k] > 0 else 0.0),
                     'curve_every_0.05': [{k: c[k] for k in ('s', 'men_brier', 'women_brier', 'brier', 'nll', 'acc_loss', 'feasible')}
                                          for c in curve if abs(round(c['s'] * 20) - c['s'] * 20) < 1e-9]}
                cid = trec['proj'][lvl].get('config_id')
                if cid:
                    occ, _, _ = OD.load_occupation('PROT', cid, s)
                    ms = PR.cost_stats(occ['MODEL_SELECT'], C.task_vectors(S['occupation'], wocc['MODEL_SELECT'], 5), S['occupation'], S['sex'])
                    ms_se = PR.cost_stats_se(occ['MODEL_SELECT'], C.task_vectors(S['occupation'], wocc['MODEL_SELECT'], 5), S['occupation'],
                                             S['sex'], S['unit'])
                    r['release'] = cid
                    r['realized_MODEL_SELECT'] = ms
                    r['realized_MODEL_SELECT_se'] = ms_se
                    r['transfer_error_MS_minus_OOF'] = {k: ms[k] - st[k] for k in st}
                    r['transfer_error_over_budget'] = {k: (ms[k] - st[k]) / budget[k] if budget[k] > 0 else None for k in st}
                rep[lvl] = r
            out['replays'][f'{stem}/S{s}'] = rep
    diffs = [r['scale_abs_diff'] for v in out['replays'].values() for r in v.values()]
    out.update({'max_scale_abs_diff': max(diffs), 'reproduced': max(diffs) <= 1e-9, 'cpu_s': time.process_time() - t0, 'utc': C.now(),
                'interpreter': sys.executable})
    C.write_json(C.RESULTS / 'preflight' / 'SOURCE_REPLAY.json', out)
    print(json.dumps({'reproduced': out['reproduced'], 'max_scale_abs_diff': out['max_scale_abs_diff'],
                      'transfer': {k: {l: {m: round(x, 5) for m, x in r.get('transfer_error_MS_minus_OOF', {}).items()} for l, r in v.items()}
                                   for k, v in out['replays'].items()}}))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('outputs', 'replay'))
    a = ap.parse_args(argv)
    (outputs if a.cmd == 'outputs' else replay)()
    return 0


if __name__ == '__main__':
    sys.exit(main())
