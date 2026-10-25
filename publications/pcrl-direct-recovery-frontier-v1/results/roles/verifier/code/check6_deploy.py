"""Verifier check 6 (DRF1; label-free): raw-input deployment packages against the evaluated wires, with my own emission as the
reference (separate code: vlib routing, bins, tables; frozen teachers applied with nrm.fa.prc._proba on ohn.fam_a.compact_po).

    python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/check6_deploy.py --packages DIR ... --out FILE

Per package: (a) every PACKAGE.json dependency hash recomputed; (b) the package's release hash = RELEASE.json = the evaluation record;
(c) the stored arrays (runs/STORED) = RELEASE.npz bitwise on all six roles; (d) my emission from MODEL.pkl on MODEL_FIT and MODEL_SELECT
features = the stored arrays bitwise; (e) the package used as a black box (`drf.deploy.release` on the six raw fields of every
MODEL_FIT row; `drf.deploy.underlying` on MODEL_SELECT features as a whole batch, in requests of 333 rows (all rows), 7 rows (first 700) and 1 row (first 100); bounded,
the coordinator's check runs every size) = my emission bitwise, decisions = argmax, exported keys = probability and decision only; (f) refusal probes (SEX field,
hidden-output request, reordered fields); (g) the coordinator's DEPLOYMENT_CHECK.json flags against mine.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import datetime as dt
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

CAMPAIGN = 'pcrl_direct_recovery_frontier_v1'
RES = REPO / 'results' / CAMPAIGN
PRIV = REPO / '.private' / CAMPAIGN
RUNS = PRIV / 'runs'
ROLES6 = ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')


def resolve(key):
    if key.startswith('DRF::'):
        return RUNS / key[5:]
    if key.startswith('REPO::'):
        return REPO / key[6:]
    if key.startswith('CODE::'):
        m = key[6:]
        p = REPO / (m.replace('.', '/') + '.py')
        return p if p.is_file() else REPO / m.replace('.', '/') / '__init__.py'
    raise KeyError(key)


def my_emission(M, Z, PRC):
    qI, qO = PRC._proba(M['teachers']['inc'], Z), PRC._proba(M['teachers']['occ'], Z)
    pi = np.clip(PRC._proba(M['sex_teacher'], Z)[:, 1], *V.PI_CLIP)
    k, _ = V.nearest_codeword(qI, M['book']['CI'])
    l, _ = V.nearest_codeword(qO, M['book']['CO'])
    m = V.pi_bins(pi, M['pi_cuts'])
    ic, oc = V.fine_cells(k, l, m)
    return {'income': np.asarray(M['T_inc'])[np.asarray(M['A_inc'])[ic]], 'occupation': np.asarray(M['T_occ'])[np.asarray(M['A_occ'])[oc]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--packages', nargs='+', required=True)
    ap.add_argument('--report', default=str(RES / 'deploy' / 'DEPLOYMENT_CHECK.json'))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    t0 = time.process_time()
    from drf import common as C                    # label-free features and raw inputs only
    from drf import deploy as D                    # the package under test, used as a black box
    from drf import release as RL                  # saved arrays only
    from nrm.fa import prc as PRC                  # frozen teachers applied with PRC._proba only
    from ohn import fam_a as FA                    # compact_po only
    items, info, per = [], {'utc_start': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}, {}
    F = {r: C.features(r) for r in ('MODEL_FIT', 'MODEL_SELECT')}
    R = C.features('MODEL_FIT', raw=True)
    raw = R['raw_inputs']
    rows = [{c: raw[c][i].item() if hasattr(raw[c][i], 'item') else raw[c][i] for c in D.PERMITTED} for i in range(len(R['row_id']))]
    Z = {r: FA.compact_po(F[r]['X'], F[r]['feature_names']) for r in F}
    for pkg in a.packages:
        p = Path(pkg)
        p = p if p.is_absolute() else REPO / p
        nm = p.name
        meta = json.loads((p / 'PACKAGE.json').read_text())
        fam, cfg = meta['name'].split('/')
        v = int(meta['version'])
        for key, h in meta['depends_sha256'].items():
            items.append(V.cmp_eq(f'{nm}:dependency_sha256:{key}', V.sha256_file(resolve(key)), h))
        rel_json = json.loads((RUNS / fam / cfg / f'V{v}' / 'RELEASE.json').read_text())
        ev = json.loads((RUNS / fam / cfg / 'EVAL_MS.json').read_text())['versions'][str(v)]
        items.append(V.cmp_eq(f'{nm}:release_sha256_package_equals_record', meta['release_sha256'], rel_json['release_sha256']))
        items.append(V.cmp_eq(f'{nm}:release_sha256_equals_evaluated', ev['release_sha256'], rel_json['release_sha256']))
        saved, _ = RL.load(fam, cfg, v)
        with np.load(RUNS / 'STORED' / f'{fam}__{cfg}__V{v}.npz') as z:
            stored = {k: z[k].copy() for k in z.files}
        for r in ROLES6:
            for t in ('income', 'occupation'):
                items.append(V.cmp_eq(f'{nm}:stored_{r}_{t}_bitwise_equal_RELEASE', bool(np.array_equal(stored[f'{t}__{r}'], saved[r][t])), True))
        d = RUNS / fam / cfg / f'V{v}'
        mj = json.loads((d / 'MODEL.json').read_text())
        items.append(V.cmp_eq(f'{nm}:MODEL.pkl_sha256_equals_MODEL.json', V.sha256_file(d / 'MODEL.pkl'), mj['model_sha256']))
        M = pickle.loads((d / 'MODEL.pkl').read_bytes())
        mine = {r: my_emission(M, Z[r], PRC) for r in Z}
        for r in Z:
            for t in ('income', 'occupation'):
                items.append(V.cmp_eq(f'{nm}:my_emission_{r}_{t}_bitwise_equal_stored', bool(np.array_equal(mine[r][t], stored[f'{t}__{r}'])), True))
        out = D.release(str(p), rows)
        for t in ('income', 'occupation'):
            items.append(V.cmp_eq(f'{nm}:package_raw_MODEL_FIT_{t}_bitwise_equal_mine', bool(np.array_equal(out[t]['probability'],
                                                                                                        mine['MODEL_FIT'][t])), True))
            items.append(V.cmp_eq(f'{nm}:package_raw_MODEL_FIT_{t}_decisions_argmax', bool(np.array_equal(out[t]['decision'],
                                                                                                      np.argmax(mine['MODEL_FIT'][t], 1))), True))
            items.append(V.cmp_eq(f'{nm}:package_exported_keys_{t}', sorted(out[t]), ['decision', 'probability']))
        Xs, ns = F['MODEL_SELECT']['X'], F['MODEL_SELECT']['feature_names']
        full = D.underlying(str(p), Xs, ns)
        for t in ('income', 'occupation'):
            items.append(V.cmp_eq(f'{nm}:package_MODEL_SELECT_full_batch_{t}_bitwise_equal_mine', bool(np.array_equal(full[t],
                                                                                                                   mine['MODEL_SELECT'][t])), True))
        for size, n_rows in ((333, len(Xs)), (7, 700), (1, 100)):        # bounded sample (the coordinator's check runs every size)
            parts = {'income': [], 'occupation': []}
            for s in range(0, n_rows, size):
                o = D.underlying(str(p), Xs[s:min(s + size, n_rows)], ns)
                for t in parts:
                    parts[t].append(o[t])
            for t in parts:
                items.append(V.cmp_eq(f'{nm}:package_MODEL_SELECT_requests_of_{size}_first_{n_rows}_{t}_bitwise_equal_mine',
                                      bool(np.array_equal(np.vstack(parts[t]), mine['MODEL_SELECT'][t][:n_rows])), True))
        probes = {}
        for label, call in (('SEX_field', lambda: D.release(str(p), [{**rows[0], 'SEX': 1}])),
                            ('hidden_output_request', lambda: D.release(str(p), rows[:2], requested=('probability', 'raw_score'))),
                            ('reordered_fields', lambda: D.release(str(p), [dict(reversed(list(rows[0].items())))]))):
            try:
                call()
                probes[label] = 'NOT_REFUSED'
            except (PermissionError, ValueError) as e:
                probes[label] = f'refused: {type(e).__name__}'
            items.append(V.cmp_eq(f'{nm}:refusal_{label}', probes[label].startswith('refused'), True))
        per[nm] = {'name': meta['name'], 'version': v, 'n_dependencies': len(meta['depends_sha256']), 'probes': probes}
    rp = Path(a.report)
    rp = rp if rp.is_absolute() else REPO / rp
    if not rp.is_file():
        items.append({'name': 'report:missing', 'result': 'FAIL', 'path': str(rp)})
    else:
        rep = json.loads(rp.read_text())
        info['report'] = {'path': str(rp.relative_to(REPO)), 'sha256': V.sha256_file(rp)}
        for flag in ('all_bitwise', 'all_attacked_full_batch_bitwise', 'all_decisions_equal_every_request_size', 'all_refused'):
            items.append(V.cmp_eq(f'report:{flag}', rep.get(flag), True))
        for nm in per:
            items.append(V.cmp_eq(f'report:{nm}:package_present', nm in rep.get('packages', {}), True))
    info['cpu_s'] = round(time.process_time() - t0, 1)
    groups = {}
    for it in items:
        groups.setdefault(it['name'].split(':', 1)[0], []).append(it)
    res = {'schema': 'DRF1-verifier-check6-v1', 'info': info, 'packages': per, 'summary': {g: V.summarize(x) for g, x in groups.items()},
           'overall': V.summarize(items), 'items': items}
    if a.out:
        V.dump(a.out, res)
    print(json.dumps(V.jsonable({'overall': res['overall'], 'summary': res['summary'], 'packages': per, 'cpu_s': info['cpu_s']}), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
