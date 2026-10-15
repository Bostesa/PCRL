"""Stage F: from-scratch refits of a subset of pinned readers on own-recomputed ATT/INNER wires; compare to frozen INNER_SCORES."""
from __future__ import annotations
import argparse
import json
import warnings

from .common import bind, Source, bases, role_forward, npz, write_out, arr_sha


def run(spec):
    import numpy as np
    import torch
    active, cfg, get, guard = bind(spec)
    from task_conditioned_representation.attacks import fit_reader, predict_reader, orientation, orient, attack_split, MODEL_SEED_BASE
    from compressed_income_safeguard.readers import tables, kind_for
    att = npz(get(cfg['att_features'])); sex = npz(get(cfg['att_sex'])); inner = npz(get(cfg['inner_features']))
    assert np.array_equal(att['row_id'], sex['row_id']) and np.array_equal(att['unit'], sex['unit'])
    y = sex['sex'].astype(np.int64)
    train, val = attack_split(att['unit'])
    fits = 0; rows = []; fails = []
    for seed in ('0', '1'):
        S = cfg['seeds'][seed]
        src = Source(get(S['checkpoint'])); B, _ = bases(get(S['geometry']))
        scales = [float(x) for x in json.loads(get(S['adapter_binding']).read_text())['scales']]
        for svc, plan in S['services'].items():
            guard()
            if plan['checkpoint'] is None:
                wa = role_forward(src, B, att['X'], att['feature_names'], mode='raw_proj')
                wi = role_forward(src, B, inner['X'], inner['feature_names'], mode='raw_proj')
            else:
                p = torch.load(get(plan['checkpoint']), map_location='cpu', weights_only=False)
                assert p['arm'] == plan['arm'] and p['label'] == plan['label']
                st = [{'R': p['states'][i]['R'], 'b': p['states'][i]['b']} for i in (0, 1)]
                wa = role_forward(src, B, att['X'], att['feature_names'], states=st, scales=scales)
                wi = role_forward(src, B, inner['X'], inner['feature_names'], states=st, scales=scales)
            receipt = json.loads(get(plan['receipt']).read_text())
            frozen = npz(get(plan['scores']))
            hashes_ok = (arr_sha(wa[0]) == receipt['wire_hashes']['ATT_r1'] and arr_sha(wa[1]) == receipt['wire_hashes']['ATT_r2']
                         and arr_sha(wi[0]) == receipt['wire_hashes']['INNER_r1'] and arr_sha(wi[1]) == receipt['wire_hashes']['INNER_r2'])
            if not hashes_ok:
                fails.append(f'S{seed} {svc} own ATT/INNER wire hash differs from receipt')
            tab = tables(wa[0], wa[1]); itab = tables(wi[0], wi[1])
            for view, name in plan['readers']:
                guard()
                kind = kind_for(name)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rec = fit_reader(tab[view][kind][train], y[train], name, MODEL_SEED_BASE + int(seed))
                fits += 1
                raw = predict_reader(rec, tab[view][kind][val]); sign = orientation(y[val], raw)
                sc = orient(predict_reader(rec, itab[view][kind]), sign)
                ref = frozen[f'{view}__{name}']
                d = float(np.max(np.abs(sc - ref)))
                rv = receipt['ATT_validation'][view][name]
                row = {'seed': int(seed), 'service': svc, 'view': view, 'reader': name, 'inner_score_max_abs_diff': d,
                       'bitwise': bool(np.array_equal(sc, ref)), 'orientation_own': sign, 'orientation_receipt': rv['orientation'],
                       'reason': plan['reasons'].get(f'{view}__{name}')}
                rows.append(row)
                if d > 1e-12 or sign != rv['orientation']:
                    fails.append(f'S{seed} {svc} {view} {name} differs ({d:.2e})')
    res = {'schema': 'TRA1-independent-stage-F-v1', 'unit_id': active['unit_id'], 'reader_refits_used': fits, 'refits': rows,
           'failures': fails, 'att_split': {'train': int(len(train)), 'validation': int(len(val))}}
    res['pass'] = bool(not fails and fits <= 36)
    return write_out(cfg, res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--spec', required=True)
    print(json.dumps(run(ap.parse_args().spec)['pass']))
