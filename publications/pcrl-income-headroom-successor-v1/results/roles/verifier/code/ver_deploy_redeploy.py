"""Independent verifier, check 7: actual transmitted outputs (label-free).

Public-use ACS 2019 Florida person records (DEV_REPORT, 5,943 people). The service predicts income class (2) and occupation
group (5) from AGEP, WKHP, COW, SCHL, MAR, POBP. This driver redeploys the finalist (and its twin and component control) from
those six raw fields ONLY, through the package wrapper `ihs.deploy.release`, and compares the produced float64 probability
vectors bitwise (np.array_equal and this script's own SHA-256 over dtype, shape and bytes) with (i) the stored release in the
run directory and (ii) the final service bank. No label of any role is read (DEV_REPORT is loaded with features and raw inputs
only); SEX is never read.

Own code: row construction, hashing, comparisons, dependency-hash checks, refusal probes. The package wrapper is the object under
test; the family predict is reached only through it.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import numpy as np                                    # noqa: E402

from ihs import common as C                           # noqa: E402  (hash-checked label-free loader only)
from ihs import deploy as DP                          # noqa: E402  (the package wrapper under test)

SIX = ('AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP')
TASKS = ('income', 'occupation')
OUT = C.PRIV / 'verifier'
BANK_NAME = {'H3_STRONG_LJDIR_PB': 'SUCCESSOR', 'HC_STRONG': 'TW', 'HC_TTREE_LJDIR_PB': 'CC'}


def fsha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def asha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(repr(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    res = {'schema': 'IHS1-verifier-redeploy-v1', 'labels_read': False, 'utc': C.now()}
    R = C.load_role('DEV_REPORT', ('features', 'raw_inputs'))
    assert not ({'income', 'occupation', 'sex'} & set(R)), 'a label was returned'
    raw = R['raw_inputs']
    res['raw_inputs_keys'] = sorted(raw)
    extra = sorted(set(raw) - set(SIX) - {'row_id', 'unit'})
    res['raw_input_fields_beyond_six_and_ids'] = extra
    n = len(R['row_id'])
    assert np.array_equal(raw['row_id'], R['row_id']) and np.array_equal(raw['unit'], R['unit'])
    rows = []
    for i in range(n):
        rows.append({c: str(raw[c][i]) for c in SIX})         # exactly the six fields, in the permitted order
    assert all(list(r) == list(SIX) for r in rows)
    res['n_rows'] = n

    # FINALIST_LOCK package hashes and the service bank manifest, re-hashed here
    lock = json.loads((C.RESULTS / 'FINALIST_LOCK.json').read_text())
    res['finalist_lock_package_files'] = {}
    for s in (0, 1):
        d = C.RUNS / 'H' / 'H3_STRONG_LJDIR_PB' / f'S{s}'
        res['finalist_lock_package_files'][f'S{s}'] = {name: {'locked': h, 'on_disk': fsha(d / name), 'equal': fsha(d / name) == h}
                                                        for name, h in lock['package_files_sha256'][f'S{s}'].items()}
    man = json.loads((C.RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    banks = {}
    res['service_bank'] = {}
    for s in (0, 1):
        p = C.RUNS / 'FINAL' / f'SERVICE_BANK_S{s}.npz'
        h = fsha(p)
        res['service_bank'][f'S{s}'] = {'manifest': man['bank_files_sha256'][f'S{s}'], 'on_disk': h, 'equal': h == man['bank_files_sha256'][f'S{s}']}
        banks[s] = load_npz(p)
        assert np.array_equal(banks[s]['row_id__DEV_REPORT'], R['row_id']), 'bank DEV_REPORT row order differs'

    res['packages'] = {}
    for cfg in ('H3_STRONG_LJDIR_PB', 'HC_STRONG', 'HC_TTREE_LJDIR_PB'):
        for s in (0, 1):
            key = f'{cfg}|S{s}'
            pkg = C.RUNS / 'DEPLOY' / f'{cfg}_S{s}'
            meta = json.loads((pkg / 'PACKAGE.json').read_text())
            deps = {}
            for rel, h in meta['depends_sha256'].items():
                p = (C.PRIV / rel) if rel.startswith('runs/') else (C.ADM / rel)
                deps[rel] = (fsha(p) == h)
            r = {'package': str(pkg.relative_to(ROOT)), 'package_json_sha256': fsha(pkg / 'PACKAGE.json'), 'meta_family': meta['family'],
                 'meta_config': meta['config_id'], 'meta_seed': meta['seed'], 'module': meta['module'],
                 'permitted_inputs': meta['permitted_inputs'], 'encoder_raw_fields': sorted(set(meta['encoder']['numeric']) | set(meta['encoder']['categorical'])),
                 'n_dependencies': len(deps), 'dependencies_all_hash_equal': all(deps.values()),
                 'dependencies_failed': [k for k, v in deps.items() if not v], 'dependency_names': sorted(deps)}
            # the 145-column features are a deterministic function of the six fields alone
            Xe = DP.SRC_DEPLOY.encode_raw(rows, meta['encoder'])
            r['encoded_from_six_equals_admitted_features'] = bool(Xe.dtype == R['X'].dtype and np.array_equal(Xe, R['X']))
            r['encoded_dtype'] = str(Xe.dtype); r['admitted_dtype'] = str(R['X'].dtype)
            out = DP.release(str(pkg), rows)
            rd = C.RUNS / 'H' / cfg / f'S{s}'
            rrec = json.loads((rd / 'RELEASE.json').read_text())
            r['release_npz_sha_matches_record'] = fsha(rd / 'RELEASE.npz') == rrec['release_sha256']
            stored = load_npz(rd / 'RELEASE.npz')
            svc = BANK_NAME[cfg]
            r['bank_service'] = svc
            for t in TASKS:
                prob = out[t]['probability']; dec = out[t]['decision']
                st = stored[f'DEV_REPORT__{t}']
                bk = banks[s][f'{svc}__DEV_REPORT__{t}']
                r[t] = {'dtype': str(prob.dtype), 'shape': list(prob.shape), 'redeployed_sha256': asha(prob),
                        'stored_release_sha256': asha(st), 'bank_sha256': asha(bk),
                        'bitwise_equal_stored_release': bool(prob.dtype == st.dtype and np.array_equal(prob, st)),
                        'bitwise_equal_bank': bool(prob.dtype == bk.dtype and np.array_equal(prob, bk)),
                        'record_wire_sha256_equal': C.wire_sha(prob) == rrec['wire_sha256'][f'DEV_REPORT__{t}'],
                        'decision_equals_argmax_stored': bool(np.array_equal(dec, st.argmax(1))),
                        'max_abs_diff_stored': float(np.max(np.abs(prob - st)))}
            # occupation output equal to H_LJ's JOINT_TREE occupation output (frozen bank and service bank)
            fb, fids = C.frozen_bank(s, roles=('DEV_REPORT',))
            assert np.array_equal(fids['row_id__DEV_REPORT'], R['row_id'])
            hlj_f = fb['H_LJ']['DEV_REPORT']['occupation']; hlj_b = banks[s]['H_LJ__DEV_REPORT__occupation']
            r['occupation_equals_H_LJ_frozen_bank'] = bool(np.array_equal(out['occupation']['probability'], hlj_f))
            r['occupation_equals_H_LJ_service_bank'] = bool(np.array_equal(out['occupation']['probability'], hlj_b))
            # refusal probes (own): forbidden, extra, reordered, missing fields; hidden-output request
            probes = {}
            for label, bad in (('SEX_added', [{**rows[0], 'SEX': '1'}]), ('PINCP_added', [{**rows[0], 'PINCP': '50000'}]),
                               ('OCCP_added', [{**rows[0], 'OCCP': '0010'}]), ('nonpermitted_extra_ZZZ', [{**rows[0], 'ZZZ': '1'}]),
                               ('reordered', [{c: rows[0][c] for c in reversed(SIX)}]), ('missing_POBP', [{c: rows[0][c] for c in SIX[:-1]}])):
                try:
                    DP.release(str(pkg), bad); probes[label] = 'NOT_REFUSED'
                except (PermissionError, ValueError) as e:
                    probes[label] = f'refused: {type(e).__name__}'
            try:
                DP.release(str(pkg), rows[:3], requested=('probability', 'logit')); probes['hidden_output_logit'] = 'NOT_REFUSED'
            except PermissionError as e:
                probes['hidden_output_logit'] = f'refused: {type(e).__name__}'
            # row-batch invariance: a 13-row batch and single rows reproduce the same bits
            small = DP.release(str(pkg), rows[100:113]); one = DP.release(str(pkg), rows[100:101])
            r['batch_invariance'] = {t: bool(np.array_equal(small[t]['probability'], stored[f'DEV_REPORT__{t}'][100:113])
                                             and np.array_equal(one[t]['probability'], stored[f'DEV_REPORT__{t}'][100:101])) for t in TASKS}
            r['refusals'] = probes
            res['packages'][key] = r
    # nominee vs H_LJ / twin relations
    res['all_bitwise_stored'] = all(v[t]['bitwise_equal_stored_release'] for v in res['packages'].values() for t in TASKS)
    res['all_bitwise_bank'] = all(v[t]['bitwise_equal_bank'] for v in res['packages'].values() for t in TASKS)
    res['all_occupation_equal_H_LJ'] = all(v['occupation_equals_H_LJ_frozen_bank'] and v['occupation_equals_H_LJ_service_bank'] for v in res['packages'].values())
    res['all_refused'] = all(x.startswith('refused') for v in res['packages'].values() for x in v['refusals'].values())
    res['all_lock_files_equal'] = all(x['equal'] for s in res['finalist_lock_package_files'].values() for x in s.values())
    OUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    (OUT / 'REDEPLOY_CHECK.json').write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: res[k] for k in ('all_bitwise_stored', 'all_bitwise_bank', 'all_occupation_equal_H_LJ', 'all_refused',
                                          'all_lock_files_equal', 'service_bank', 'raw_input_fields_beyond_six_and_ids')}, indent=1))


if __name__ == '__main__':
    main()
