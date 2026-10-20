"""V7 child process: runs INSIDE a restored backup copy (sys.path[0] = the restored root, so nrm.* and the admitted copies resolve
there; the imported module files are asserted to lie under the restored root). Label-free. For each package: the six-field wrapper
API, a CSV written by the verifier, and the underlying entry point on the admitted DEV_REPORT encoding must reproduce bitwise the
restored stored wire, the restored RELEASE.npz DEV_REPORT draw 0 and the LIVE evaluated RELEASE.npz DEV_REPORT draw 0; the package
must also reproduce the evaluated MODEL_SELECT wire (draw 0) from MODEL_SELECT features; decisions are the first argmax; every
refusal probe (registered ones plus the verifier's own) must raise; only probability and decision are exported.

    (called by v7_deploy.py) python3 v7_child.py --root RESTORED --live-runs LIVE_RUNS --packages P1 P2 --out OUT.json
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--live-runs', required=True)
    ap.add_argument('--packages', nargs='+', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    sys.path.insert(0, str(root))
    import nrm.common as C
    import nrm.deploy as D
    from nrm import release as RL
    mods = {m.__name__: str(Path(m.__file__).resolve()) for m in (C, D, RL)}
    under = all(Path(p).is_relative_to(root) for p in mods.values())
    sys.path.insert(1, str(Path(__file__).resolve().parent))
    import vlib as V
    perm = list(D.PERMITTED)
    R = C.features('DEV_REPORT', raw=True)
    raw = R['raw_inputs']
    rows = [{c: (raw[c][i].item() if hasattr(raw[c][i], 'item') else raw[c][i]) for c in perm} for i in range(len(R['row_id']))]
    MS = C.features('MODEL_SELECT')
    tmp = Path(tempfile.mkdtemp(prefix='v7_', dir=str(root)))
    own_csv = tmp / 'own.csv'
    with open(own_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(perm)
        for r in rows:
            w.writerow([r[c] for c in perm])
    out = {'modules': mods, 'modules_under_restored_root': under, 'packages': {}}
    for pk in a.packages:
        pkg = root / '.private' / 'pcrl_new_release_mechanisms_v1' / 'deploy' / pk
        res = {'package_dir': str(pkg.relative_to(root))}
        try:
            meta = D._load(pkg)
        except Exception as e:                     # noqa: BLE001  (a dependency-hash refusal is a FAIL of this package)
            res['load_error'] = f'{type(e).__name__}: {e}'
            out['packages'][pk] = res
            continue
        fam, cfg = meta['name'].split('/')
        v = int(meta['version'])
        rnd = bool(meta.get('randomized'))
        keys_dev = {'role': 'DEV_REPORT', 'row_id': np.asarray(R['row_id'])} if rnd else None
        keys_ms = {'role': 'MODEL_SELECT', 'row_id': np.asarray(MS['row_id'])} if rnd else None
        st = D.stored(meta)
        restored_rel, rec = RL.load(fam, cfg, v)
        live = np.load(Path(a.live_runs) / fam / cfg / f'V{v}' / 'RELEASE.npz')
        live_sha = V.file_sha(Path(a.live_runs) / fam / cfg / f'V{v}' / 'RELEASE.npz')

        def d0(x):
            return x[0] if x.ndim == 3 else x
        api = D.release(pkg, rows, keys=keys_dev)
        via_csv = D.release_csv(pkg, own_csv, keys=keys_dev)
        und = D.underlying(pkg, R['X'], R['feature_names'], keys=keys_dev)
        und_ms = D.underlying(pkg, MS['X'], MS['feature_names'], keys=keys_ms)
        eq = {}
        for t in V.TASKS:
            targets = {'stored_wire': st[t], 'restored_release_DEV_REPORT': d0(restored_rel['DEV_REPORT'][t]),
                       'live_release_DEV_REPORT': d0(live[f'DEV_REPORT__{t}'])}
            for nm, o in (('wrapper_api', api[t]['probability']), ('wrapper_own_csv', via_csv[t]['probability']), ('underlying', und[t])):
                for tn, ref in targets.items():
                    eq[f'{t}|{nm}|{tn}'] = bool(np.array_equal(o, ref))
            eq[f'{t}|underlying_MODEL_SELECT|live_release_MODEL_SELECT'] = bool(np.array_equal(und_ms[t], d0(live[f'MODEL_SELECT__{t}'])))
            eq[f'{t}|decision_first_argmax'] = bool(np.array_equal(api[t]['decision'], V.argmax_first(api[t]['probability'])))
        probes = {}

        def probe(label, fn):
            try:
                fn(); probes[label] = 'NOT_REFUSED'
            except (PermissionError, ValueError, KeyError, TypeError) as e:
                probes[label] = f'refused: {type(e).__name__}'
        r0 = rows[0]
        k3 = ({'role': 'DEV_REPORT', 'row_id': np.asarray(R['row_id'][:1])} if rnd else None)
        for label, bad in (('wrapper_SEX_field', {**r0, 'SEX': 1}), ('wrapper_lowercase_sex_field', {**r0, 'sex': 2}),
                           ('wrapper_target_OCCP_field', {**r0, 'OCCP': '0010'}), ('wrapper_target_PINCP_field', {**r0, 'PINCP': 50000}),
                           ('wrapper_extra_field', {**r0, 'PUMA': 1}), ('wrapper_reordered_fields', dict(reversed(list(r0.items())))),
                           ('wrapper_missing_field', {k: x for k, x in r0.items() if k != 'POBP'})):
            probe(label, lambda bad=bad: D.release(pkg, [bad], keys=k3))
        for label, hdr in (('csv_SEX_column', perm + ['SEX']), ('csv_lowercase_sex_column', perm + ['sex']),
                           ('csv_target_PINCP_column', perm + ['PINCP']), ('csv_extra_column', perm + ['PUMA']),
                           ('csv_reordered_columns', perm[::-1]), ('csv_missing_column', perm[:-1])):
            q = tmp / f'{label}.csv'
            with open(q, 'w', newline='') as f:
                cw = csv.writer(f); cw.writerow(hdr); cw.writerow([r0.get(h, 1) for h in hdr])
            probe(label, lambda q=q: D.release_csv(pkg, q, keys=k3))
        names = list(R['feature_names']); X1 = R['X'][:1]
        kx = ({'role': 'DEV_REPORT', 'row_id': np.asarray(R['row_id'][:1])} if rnd else None)
        for label, (Xb, nb) in (('underlying_SEX_column', (np.column_stack((X1, np.ones(1))), names + ['SEX'])),
                                ('underlying_extra_column', (np.column_stack((X1, np.ones(1))), names + ['EXTRA'])),
                                ('underlying_missing_column', (X1[:, 1:], names[1:])),
                                ('underlying_reordered_columns', (X1[:, ::-1], names[::-1])),
                                ('underlying_nonfinite_input', (np.where(np.arange(X1.shape[1]) == 0, np.nan, X1), names))):
            probe(label, lambda Xb=Xb, nb=nb: D.underlying(pkg, Xb, nb, keys=kx))
        probe('hidden_output_request_raw_score', lambda: D.release(pkg, rows[:1], requested=('probability', 'raw_score'), keys=k3))
        probe('hidden_output_request_cell_id', lambda: D.release(pkg, rows[:1], requested=('decision', 'cell_id'), keys=k3))
        if rnd:
            probe('randomized_without_draw_keys', lambda: D.release(pkg, rows[:1]))
        else:
            probe('deterministic_with_draw_keys', lambda: D.release(pkg, rows[:1], keys={'role': 'DEV_REPORT', 'row_id': np.arange(1)}))
        res.update({'name': meta['name'], 'version': v, 'randomized': rnd, 'live_release_sha256': live_sha,
                    'release_sha256_in_package': meta.get('release_sha256'),
                    'package_release_equals_live': meta.get('release_sha256') == live_sha,
                    'bitwise': eq, 'all_bitwise': all(eq.values()), 'refusals': probes,
                    'all_refused': all(x.startswith('refused') for x in probes.values()),
                    'exported_keys': {t: sorted(api[t]) for t in V.TASKS},
                    'only_probability_and_decision': all(sorted(api[t]) == ['decision', 'probability'] for t in V.TASKS),
                    'wire_sha256_DEV_REPORT': {t: V.wire_sha(api[t]['probability']) for t in V.TASKS}})
        out['packages'][pk] = res
    Path(a.out).write_text(json.dumps(out, indent=1, default=V._jd))
    return 0


if __name__ == '__main__':
    sys.exit(main())
