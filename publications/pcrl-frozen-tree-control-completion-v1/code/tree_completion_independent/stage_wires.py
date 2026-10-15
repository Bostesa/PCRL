"""Stage WIRES: INNER releases of the new controls recomputed from their frozen artifacts.

RAW_J_NEQ / T_NEURAL: own torch forward of the saved per-task Sequential(83-64-ReLU-64-ReLU-16-K) state dicts
(float32 network), float64 centred softmax; decisions = argmax (first maximum).
FARE_TC / FARE_TC_Z: INNER_RELEASE arrays vs the official FARE release arrays mapped by row_id.
"""
from __future__ import annotations

import argparse
import io

import numpy as np

from . import core

NEW = ('T_REFIT_E20', 'RAW_J_NEQ', 'T_NEURAL', 'FARE_TC', 'FARE_TC_Z', 'T_TREE')
KEYS = {'RAW_J_NEQ': ('RAW_J', '1.0', 'E2'), 'T_NEURAL': ('T_NEURAL', 'none', 'E2')}


def net(k):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(83, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                               torch.nn.Linear(64, 16), torch.nn.Linear(16, k))


def forward(state, X):
    """Own forward: float32 logits, then float64 centred softmax."""
    import torch
    out = {}
    for task, k in (('income', 2), ('occupation', 6)):
        m = net(k); m.load_state_dict(state[task]); m.eval()
        with torch.no_grad():
            z = m(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))).numpy().astype(np.float64)
        z = z - z.mean(1, keepdims=True)
        e = np.exp(z - z.max(1, keepdims=True))
        out[task] = e / e.sum(1, keepdims=True)
    return out


def load_state(verify, item):
    import torch
    p = verify(item['path'], item['sha256'])
    return torch.load(io.BytesIO(p.read_bytes()), map_location='cpu', weights_only=True)


def run(spec):
    import torch
    torch.set_num_threads(1)
    from tree_completion.evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec, 'verification')
    R = core.Receipt('WIRES', active, cfg)
    cfg['_code_sha256'] = core.code_hashes(verify, cfg)
    J = lambda it: core.load_json(verify, it)
    slock = J(core.lock_item(cfg, 'selection_lock')); rlock = J(core.lock_item(cfg, 'reporting_lock')); manifest = J(cfg['manifest'])
    receipts = {s: J(cfg['receipts'][s]) for s in ('0', '1')}
    feats = core.load_npz(verify, cfg['inner_features'], ['row_id', 'unit', 'X', 'feature_names'])
    X, row_id = feats['X'], feats['row_id']
    names = [str(x) for x in feats['feature_names']]
    R.check('INNER_features_raw83_float32_finite', X.shape == (2235, 83) and X.dtype == np.float32 and np.isfinite(X).all())
    R.check('INNER_feature_names_equal_manifest', names == manifest['preprocessing']['feature_names'])

    worst = {}
    for s in ('0', '1'):
        guard()
        nrec = J(cfg['neural_record'][s]); frec = J(cfg['fare_record'][s])
        rel = {svc: core.load_npz(verify, cfg['new_releases'][svc][s]) for svc in NEW}
        for svc, z in rel.items():
            R.check(f'release_fields_only_row_id_p1_d1_p2_d2_{svc}_s{s}', sorted(z) == ['d1', 'd2', 'p1', 'p2', 'row_id'])
            R.check(f'release_rows_equal_INNER_order_{svc}_s{s}', np.array_equal(z['row_id'], row_id))
            ok = (z['p1'].shape == (2235, 2) and z['p2'].shape == (2235, 6) and z['p1'].dtype == np.float64 and z['p2'].dtype == np.float64
                  and np.isfinite(z['p1']).all() and np.isfinite(z['p2']).all() and (z['p1'] >= 0).all() and (z['p2'] >= 0).all()
                  and np.abs(z['p1'].sum(1) - 1).max() < 1e-9 and np.abs(z['p2'].sum(1) - 1).max() < 1e-9
                  and np.array_equal(z['d1'], z['p1'].argmax(1)) and np.array_equal(z['d2'], z['p2'].argmax(1)))
            R.check(f'release_probability_convention_and_argmax_decisions_{svc}_s{s}', ok)
            w = core.wire_sha(z['p1'], z['d1'], z['p2'], z['d2'])
            R.check(f'wire_sha_equals_receipt_and_REPORTING_LOCK_{svc}_s{s}',
                    w == receipts[s]['services'][svc]['wire_sha256']['INNER'] == rlock['new_banks'][s]['inner_wire_sha256'][svc])
        # neural recomputation
        for svc, (arm, tag, lab) in KEYS.items():
            it = cfg['neural'][s][svc]
            fname = it['path'].split('/')[-1]
            R.check(f'{svc}_s{s}_checkpoint_sha_equals_selection_lock_and_record',
                    it['sha256'] == slock['attack_roster_files'][s][svc]['files_sha256'][fname] ==
                    nrec['checkpoints'][f'{arm}|{tag}'][lab]['sha256'] and slock['attack_roster'][svc] == f'{arm}|{tag}|{lab}')
            state = load_state(verify, it)
            R.check(f'{svc}_s{s}_feature_names_equal_INNER', [str(x) for x in state['feature_names']] == names)
            p = forward(state, X)
            d1 = np.abs(p['income'] - rel[svc]['p1']).max(); d2 = np.abs(p['occupation'] - rel[svc]['p2']).max()
            dec = np.array_equal(p['income'].argmax(1), rel[svc]['d1']) and np.array_equal(p['occupation'].argmax(1), rel[svc]['d2'])
            worst[f'{svc}_s{s}'] = {'income_max_abs': float(d1), 'occupation_max_abs': float(d2), 'decisions_exact': bool(dec),
                                    'bitwise': bool(np.array_equal(p['income'], rel[svc]['p1']) and np.array_equal(p['occupation'], rel[svc]['p2']))}
            R.check(f'{svc}_s{s}_probabilities_within_1e-12', max(d1, d2) <= 1e-12, income=float(d1), occupation=float(d2))
            R.check(f'{svc}_s{s}_decisions_exact', dec)
        # FARE recomputation by row_id mapping
        for svc, lab in (('FARE_TC', 'F1'), ('FARE_TC_Z', 'Z1')):
            it = cfg['fare_release'][s][svc]
            fname = it['path'].split('/')[-1]
            R.check(f'{svc}_s{s}_release_sha_equals_selection_lock_and_record',
                    it['sha256'] == slock['attack_roster_files'][s][svc]['files_sha256'][fname] == frec['units'][lab]['release_sha256']
                    and slock['attack_roster'][svc] == f'FARE|{lab}')
            f = core.load_npz(verify, it, ['row_id', 'q1', 'hard1', 'q2', 'hard2'])
            pos = {int(r): i for i, r in enumerate(f['row_id'])}
            R.check(f'{svc}_s{s}_FARE_release_row_ids_unique_and_cover_INNER', len(pos) == len(f['row_id']) and all(int(r) in pos for r in row_id))
            idx = np.array([pos[int(r)] for r in row_id])
            same = all(np.array_equal(rel[svc][a], f[b][idx]) and rel[svc][a].tobytes() == np.ascontiguousarray(f[b][idx]).tobytes()
                       for a, b in (('p1', 'q1'), ('p2', 'q2')))
            dsame = np.array_equal(rel[svc]['d1'], f['hard1'][idx]) and np.array_equal(rel[svc]['d2'], f['hard2'][idx])
            R.check(f'{svc}_s{s}_INNER_release_equals_FARE_release_bitwise', same and dsame,
                    max_abs=float(max(np.abs(rel[svc]['p1'] - f['q1'][idx]).max(), np.abs(rel[svc]['p2'] - f['q2'][idx]).max())))
    # old-bank releases used by the report: fields only
    for svc, items in cfg['old_releases'].items():
        for s in ('0', '1'):
            flds = core.npz_fields(verify, items[s])
            R.check(f'old_release_fields_only_row_id_p1_d1_p2_d2_{svc}_s{s}', flds == ['d1', 'd2', 'p1', 'p2', 'row_id'])
    R.numbers['neural_recompute'] = worst
    return R.write()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    run(ap.parse_args().spec)
