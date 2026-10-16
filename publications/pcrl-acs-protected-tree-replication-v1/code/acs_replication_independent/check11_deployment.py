"""Independent check 11: deployment parity flags, refusals and package provenance.

Reads DEPLOYMENT_RESULTS.json and the private packages; recomputes the summary flags from the per-package records,
re-hashes every package file against PACKAGE.json and against the frozen run-directory sources, re-encodes the
DEV_REPORT raw permitted inputs with an independent encoder (from the package encoder spec) and compares with the
exported DEV_REPORT feature matrix, and checks the refusal-test inputs (headers; the SEX column of the forbidden-column
test file must be a constant placeholder, not a label). Does not import acs_replication. Aggregates only.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parent.parent
RES = WT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
PRIV = WT / '.private' / 'acs_replication'
OUT = RES / 'independent' / 'check11_deployment.json'
PERMITTED = ['AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


problems, ev = [], {}
dep_path = RES / 'DEPLOYMENT_RESULTS.json'
if not dep_path.exists():
    OUT.write_text(json.dumps({'id': 11, 'name': 'DEPLOYMENT', 'status': 'NOT_AVAILABLE', 'evidence': {}, 'notes': 'DEPLOYMENT_RESULTS.json absent'}, indent=2))
    raise SystemExit(0)
dep = json.loads(dep_path.read_text())
lock = json.loads((RES / 'SCIENCE_LOCK.json').read_text())
sel = json.loads((RES / 'SELECTION_LOCK.json').read_text())
receipt = json.loads((PRIV / 'extraction' / 'EXTRACTION_RECEIPT.json').read_text())
forbidden = set(lock['benchmark']['forbidden_inputs'])

# 1. flags recomputed from per-package records
parity_rows, refusal_rows = [], []
for name, p in dep['packages'].items():
    for task, v in p['parity'].items():
        parity_rows.append({'package': name, 'task': task, 'max_abs_prob_diff': v['max_abs_prob_diff'], 'decisions_identical': v['decisions_identical'],
                            'ok': v['max_abs_prob_diff'] <= 1e-12 and v['decisions_identical'] is True})
    for k, x in p['refusals'].items():
        refusal_rows.append({'package': name, 'probe': k, 'recorded': x, 'refused': str(x).startswith('REFUSED')})
expected_exc = {'forbidden_SEX_column': 'REFUSED: PermissionError', 'reordered_schema': 'REFUSED: ValueError', 'hidden_output_request': 'REFUSED: PermissionError'}
ev['flags'] = {
    'packages': sorted(dep['packages']), 'n_packages': len(dep['packages']),
    'parity_entries': len(parity_rows), 'parity_entries_ok': sum(r['ok'] for r in parity_rows),
    'max_abs_prob_diff_over_all': max(r['max_abs_prob_diff'] for r in parity_rows),
    'all_decisions_identical': all(r['decisions_identical'] for r in parity_rows),
    'refusal_entries': len(refusal_rows), 'refusal_entries_refused': sum(r['refused'] for r in refusal_rows),
    'refusal_exception_types_as_expected': all(r['recorded'] == expected_exc.get(r['probe']) for r in refusal_rows),
    'recorded_all_parity_ok': dep['all_parity_ok'], 'recomputed_all_parity_ok': all(r['ok'] for r in parity_rows),
    'recorded_all_refusals_ok': dep['all_refusals_ok'], 'recomputed_all_refusals_ok': all(r['refused'] for r in refusal_rows),
    'labels_used': dep['labels_used'], 'raw_rows_deployed': dep['raw_rows_deployed'],
}
f = ev['flags']
with np.load(PRIV / 'exports' / 'DEV_REPORT' / 'features.npz', allow_pickle=False) as z:
    X_dev = z['X']; feat_names = [str(x) for x in z['feature_names']]
n_dev = int(X_dev.shape[0])
f['DEV_REPORT_rows'] = n_dev
if not (f['recorded_all_parity_ok'] and f['recomputed_all_parity_ok'] and f['recorded_all_refusals_ok'] and f['recomputed_all_refusals_ok']
        and f['refusal_exception_types_as_expected'] and f['labels_used'] is False and f['raw_rows_deployed'] == n_dev):
    problems.append('deployment flags/refusals inconsistent')
expected_pkgs = {'T_TREE_S0', 'T_TREE_S1', 'LOCAL_TREE_S0', 'LOCAL_TREE_S1', f"{sel['frozen']['NEURAL_RAW']['configuration']}_S0"}
f['expected_packages'] = sorted(expected_pkgs)
f['package_set_as_expected'] = set(dep['packages']) == expected_pkgs
f['eligible_control_from_SELECTION_LOCK'] = sel['frozen']['NEURAL_RAW']['configuration']
if not f['package_set_as_expected']:
    problems.append('deployed package set differs from T_TREE/LOCAL_TREE x2 + eligible control')

# 2. package provenance
prov = {}
for name, p in dep['packages'].items():
    d = PRIV / 'deploy' / name
    meta = json.loads((d / 'PACKAGE.json').read_text())
    seed = meta['seed']
    rec = {'package_json_sha256_matches': sha(d / 'PACKAGE.json') == p['package_json_sha256'],
           'listed_files_match': all(sha(d / fn) == h for fn, h in meta['files'].items()),
           'encoder_equals_extraction_encoder': meta['encoder'] == receipt['encoding']}
    src = {}
    for fn in meta['files']:
        if fn.startswith('TREE_TEACHER_'):
            src[fn] = sha(d / fn) == sha(PRIV / 'runs' / f'STAGE_A_{seed}' / fn)
        elif fn == 'THETA.npz':
            src[fn] = sha(d / fn) == sha(PRIV / 'runs' / f'STAGE_B_{seed}' / 'LOCAL_TREE__E20.npz')
        elif fn == 'NEURAL_STATE.pt':
            cfg = sel['frozen']['NEURAL_RAW']['configuration']  # NEURAL_RAW_rho1.0_E2
            rho, ep = cfg.split('_rho')[1].split('_')
            src[fn] = sha(d / fn) == sha(PRIV / 'runs' / f'NEURAL_RAW_{seed}' / f'NEURAL_RAW__rho{rho}__{ep}.pt')
    rec['files_equal_frozen_run_sources'] = src
    rec['frozen_by_TRAINING_COMPLETE_LOCK'] = True
    tc = json.loads((RES / 'TRAINING_COMPLETE_LOCK.json').read_text())['frozen_files_sha256']
    for fn in meta['files']:
        rel = {'THETA.npz': f'STAGE_B_{seed}/LOCAL_TREE__E20.npz', 'NEURAL_STATE.pt': None}.get(fn, f'STAGE_A_{seed}/{fn}')
        if fn == 'NEURAL_STATE.pt':
            rel = f"NEURAL_RAW_{seed}/NEURAL_RAW__rho{rho}__{ep}.pt"
        if rel not in tc or tc[rel] != sha(d / fn):
            rec['frozen_by_TRAINING_COMPLETE_LOCK'] = False
    prov[name] = rec
    if not (rec['package_json_sha256_matches'] and rec['listed_files_match'] and rec['encoder_equals_extraction_encoder']
            and all(src.values()) and rec['frozen_by_TRAINING_COMPLETE_LOCK']):
        problems.append(f'{name}: package provenance failure')
ev['package_provenance'] = prov

# 3. independent re-encoding of DEV_REPORT raw permitted inputs
enc = receipt['encoding']
with np.load(PRIV / 'exports' / 'DEV_REPORT' / 'raw_inputs.npz', allow_pickle=False) as z:
    raw = {c: z[c] for c in PERMITTED}
names = enc['feature_names']
col = {n: i for i, n in enumerate(names)}
X = np.zeros((n_dev, len(names)), dtype=np.float64)
for c in ('AGEP', 'WKHP'):
    v = np.asarray([float(x) for x in raw[c]], dtype=np.float64)
    X[:, col[c]] = (v - enc['numeric'][c]['mean']) / enc['numeric'][c]['sd']
other_counts = {}
for c in ('COW', 'SCHL', 'MAR', 'POBP'):
    keep = set(enc['categorical'][c])
    idx = np.asarray([col[f'{c}={str(x).strip()}'] if str(x).strip() in keep else col[f'{c}=__OTHER__'] for x in raw[c]])
    X[np.arange(n_dev), idx] = 1.0
    other_counts[c] = int((idx == col[f'{c}=__OTHER__']).sum())
X32 = X.astype(np.float32)
ev['independent_reencoding'] = {
    'feature_names_equal_export': names == feat_names, 'width': len(names),
    'exact_equal_to_exported_DEV_REPORT_X': bool(np.array_equal(X32, X_dev)),
    'max_abs_diff': float(np.max(np.abs(X32.astype(np.float64) - X_dev.astype(np.float64)))),
    'rows_mapped_to_OTHER_per_variable': other_counts,
    'note': 'deployment parity of probabilities then reduces to the stored-output identity re-derived in check 6 (teacher routing / theta on DEV_REPORT X)',
}
if not (ev['independent_reencoding']['feature_names_equal_export'] and ev['independent_reencoding']['exact_equal_to_exported_DEV_REPORT_X']):
    problems.append('independent re-encoding of DEV_REPORT raw inputs differs from the exported features')

# 4. refusal-test inputs and the deployed raw CSV (headers/aggregates only)
tmp = PRIV / 'deploy_tmp'
csvinfo = {}
for fn in ('DEV_REPORT_raw_inputs.csv', 'reordered.csv', 'forbidden.csv'):
    with open(tmp / fn, newline='') as fh:
        rd = csv.reader(fh)
        header = next(rd)
        rows = list(rd)
    info = {'header': header, 'rows': len(rows), 'header_equals_permitted_order': header == PERMITTED,
            'forbidden_columns_in_header': [h for h in header if h.upper() in forbidden]}
    if 'SEX' in header:
        j = header.index('SEX')
        info['SEX_column_distinct_values'] = len({r[j] for r in rows})
    csvinfo[fn] = info
ok_csv = (csvinfo['DEV_REPORT_raw_inputs.csv']['header_equals_permitted_order'] and csvinfo['DEV_REPORT_raw_inputs.csv']['rows'] == n_dev
          and csvinfo['reordered.csv']['header'] == list(reversed(PERMITTED)) and csvinfo['forbidden.csv']['forbidden_columns_in_header'] == ['SEX']
          and csvinfo['forbidden.csv'].get('SEX_column_distinct_values') == 1)
csvinfo['refusal_inputs_well_formed_and_SEX_placeholder_constant'] = ok_csv
ev['deployment_inputs'] = csvinfo
if not ok_csv:
    problems.append('deployment input files malformed or forbidden-column test carries non-constant SEX values')

out = {'id': 11, 'name': 'DEPLOYMENT', 'status': 'PASS' if not problems else 'FAIL', 'evidence': ev, 'problems': problems,
       'scope': ('recorded parity/refusal flags re-derived from per-package records (not re-executed with the lead deploy code); package files re-hashed '
                 'against PACKAGE.json, the frozen run sources and TRAINING_COMPLETE_LOCK; raw->feature encoding re-implemented independently; '
                 'probability parity of the tree packages is covered by check 6 (routing on the identical DEV_REPORT X). Control package covers seed 0 only (as run).'),
       'sealed_assessment_touched': False}
OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
print(json.dumps({'status': out['status'], 'problems': problems, 'flags': {k: f[k] for k in ('n_packages', 'parity_entries_ok', 'parity_entries', 'max_abs_prob_diff_over_all', 'refusal_entries_refused', 'refusal_entries')},
                  'reencode_exact': ev['independent_reencoding']['exact_equal_to_exported_DEV_REPORT_X']}, indent=1))
