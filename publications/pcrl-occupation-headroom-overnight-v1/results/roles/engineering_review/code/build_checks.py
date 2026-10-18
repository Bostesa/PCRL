"""Engineering and deployment review: compose the machine-readable record roles/engineering_review/CHECKS.json.

LABEL-FREE. Sources: the review's private outputs under .private/<campaign>/runs/DEPLOY_REVIEW/ (CHECK.json, REVIEW_PY314.json,
REVIEW_VENV.json, INTEGRITY_PY314.json; written by `ohn.deploy check` and deploy_review.py through the queue), the campaign's
release / verify records, the queue ledger, the access logs (metadata only), git history of this worktree (read-only), and
negative gate tests that are refused before any data access (preconditions are checked first; a test whose precondition fails is
skipped, never run). Only aggregates are written: booleans, hashes, maximum absolute differences, counts and repository-relative
paths; no person-level value.

    python3 results/pcrl_occupation_headroom_overnight_v1/roles/engineering_review/code/build_checks.py
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from ohn import common as C  # noqa: E402
import code_history as CH  # noqa: E402

PRIV = C.RUNS / 'DEPLOY_REVIEW'
OUT = C.RESULTS / 'roles' / 'engineering_review' / 'CHECKS.json'
GATE_PURPOSE = 'engineering review negative gate test (must be refused before any read)'
RECONSTRUCTED = {'8be2baebb8f1': 'ohn/fam_a.py at cb89a8b51 plus ONLY the DIAG_SETTINGS D6-D8 block later committed in fda3a0282 '
                                 '(exact SHA-256 reconstruction); fit and predict code identical to the committed cb89a8b51'}


def load(p):
    return json.loads(Path(p).read_text())


def rel(p):
    p = str(p)
    r = str(C.ROOT.resolve())
    return p[len(r) + 1:] if p.startswith(r) else p


# ---------------------------------------------------------------- gates
def gate_tests():
    pre = {'registered_gate_open': C.registered(), 'finalist_lock_absent': not (C.RESULTS / 'FINALIST_LOCK.json').exists(),
           'reporting_lock_absent': not (C.RESULTS / 'REPORTING_LOCK.json').exists(),
           'dev_report_access_log_absent': not C.ACCESS_LOG.exists(), 'attack_label_log_absent': not C.LABEL_LOG.exists()}

    def t(fn):
        try:
            fn()
            return 'NOT_REFUSED'
        except PermissionError:
            return 'refused: PermissionError'
        except Exception as e:                                       # recorded, not hidden
            return f'refused_other: {type(e).__name__}'
    tests = {}
    if pre['finalist_lock_absent']:
        tests['ATTACK_FIT_sex_before_committed_FINALIST_LOCK'] = t(lambda: C.load_role('ATTACK_FIT', ('features', 'sex'), purpose=GATE_PURPOSE))
        tests['ATTACK_VALIDATION_sex_before_committed_FINALIST_LOCK'] = t(lambda: C.load_role('ATTACK_VALIDATION', ('sex',), purpose=GATE_PURPOSE))
    tests['ATTACK_FIT_income_not_admitted'] = t(lambda: C.load_role('ATTACK_FIT', ('features', 'income'), purpose=GATE_PURPOSE))
    tests['DEV_REPORT_sex_not_admitted'] = t(lambda: C.load_role('DEV_REPORT', ('features', 'sex'), purpose=GATE_PURPOSE))
    tests['DEV_REPORT_income_not_admitted'] = t(lambda: C.load_role('DEV_REPORT', ('features', 'income'), purpose=GATE_PURPOSE))
    tests['DEV_REPORT_occupation_not_admitted'] = t(lambda: C.load_role('DEV_REPORT', ('features', 'occupation'), purpose=GATE_PURPOSE))
    tests['CALIBRATION_sex_not_admitted'] = t(lambda: C.load_role('CALIBRATION', ('features', 'sex'), purpose=GATE_PURPOSE))
    tests['SEALED_ASSESSMENT_closed'] = t(lambda: C.load_role('SEALED_ASSESSMENT'))
    tests['unknown_role_closed'] = t(lambda: C.load_role('CAPPED_OUT', ('features',)))
    if pre['registered_gate_open']:
        tests['MODEL_SELECT_sex_without_logged_purpose'] = t(lambda: C.load_role('MODEL_SELECT', ('features', 'sex')))
    tests['dev_report_truth_unregistered_event'] = t(lambda: C.dev_report_truth(GATE_PURPOSE, 'UNREGISTERED_EVENT'))
    if pre['reporting_lock_absent'] and pre['dev_report_access_log_absent']:
        tests['dev_report_truth_before_committed_REPORTING_LOCK'] = t(lambda: C.dev_report_truth(GATE_PURPOSE, C.DEV_EVENTS[0]))
    traces = 0
    for log in (C.MS_LOG, C.TRAIN_LOG, C.LABEL_LOG, C.ACCESS_LOG):
        if log.is_file():
            traces += sum(GATE_PURPOSE in line for line in log.read_text().splitlines())
    return {'preconditions': pre, 'tests': tests, 'all_refused': all(v.startswith('refused: ') for v in tests.values()),
            'access_log_lines_written_by_tests': traces,
            'note': 'every test raises before any export file is opened; no label is read and no access-log line is written'}


def access_summary():
    ms = [json.loads(x) for x in C.MS_LOG.read_text().splitlines() if x.strip()]

    def cat(e):
        p, u = e['purpose'], e.get('unit_id') or ''
        if p.startswith('development utility evaluation'):
            return 'ohn.devrun score (registered path)'
        if p.startswith('reference utility'):
            return 'ohn.devrun references (registered path)'
        if p.startswith('development reader grid'):
            return 'ohn.devreaders run (registered path)'
        if u.startswith('VER_'):
            return 'independent verifier scripts (VER_*)'
        if u.startswith('OHN_RR_'):
            return 'reader-robustness scripts (OHN_RR_*)'
        return 'other script: ' + u
    tr = [json.loads(x) for x in C.TRAIN_LOG.read_text().splitlines() if x.strip()]
    fam_sex = sorted({e['unit_id'] for e in tr if 'sex' in e['fields'] and e['unit_id'].split('_')[1] in ('A', 'B', 'C', 'INC')})
    reg = load(C.REG_RECEIPT)
    return {'model_select_label_reads': len(ms), 'first_model_select_label_read_utc': ms[0]['utc'] if ms else None,
            'registration_receipt_verified_utc': reg['verified_utc'],
            'model_select_reads_by_path': dict(collections.Counter(cat(e) for e in ms)),
            'model_select_reads_without_unit_id': sum(1 for e in ms if not e.get('unit_id')),
            'model_select_reads_outside_registered_paths_units': sorted({e['unit_id'] for e in ms if 'registered path' not in cat(e)}),
            'training_role_label_reads': len(tr), 'training_reads_without_unit_id': sum(1 for e in tr if not e.get('unit_id')),
            'training_reads_by_role_fields': {f"{k[0]}|{'+'.join(k[1])}": v for k, v in collections.Counter((e['role'], tuple(e['fields'])) for e in tr).items()},
            'task_only_family_units_reading_MODEL_FIT_sex': len(fam_sex),
            'task_only_family_sex_reads_all_scoring_units': all('SCORE' in u for u in fam_sex),
            'attack_label_log_exists': C.LABEL_LOG.exists(), 'dev_report_truth_log_exists': C.ACCESS_LOG.exists()}


def receipt_summary():
    r = load(C.RESULTS / 'RECEIPTS.json')['receipts']
    out = []
    for x in r:
        blobs = {f: {'published': b, 'local_git_blob': CH.git('hash-object', str(C.RESULTS / f))} for f, b in x['published_blobs'].items()}
        out.append({'kind': x['kind'], 'receipt_commit': x['receipt_commit'], 'remote_sha_verified': x['remote_sha_verified'],
                    'verified_utc': x['verified_utc'], 'files': sorted(blobs),
                    'published_blobs_identical_to_local_files': all(v['published'] == v['local_git_blob'] for v in blobs.values())})
    return out


# ---------------------------------------------------------------- deployment
def env_summary(e, label):
    return {'interpreter': label, **{k: e.get(k) for k in ('python', 'numpy', 'sklearn', 'scipy', 'torch', 'xgboost', 'numpy_blas', 'machine', 'cpu')}}


def pkg_summary(r):
    a = r['audit']
    return {'anchor': r['anchor'], 'occupation': r['occupation'], 'seed': r['seed'], 'n_dependencies_in_manifest': r['n_dependencies'],
            'package_json_records_env_or_code_hashes': {k: v for k, v in r['package_records'].items()},
            'refusal_cases': len(r['refusals']), 'all_refused_cleanly': r['all_refused_cleanly'], 'refusals': r['refusals'],
            'value_level_behaviour': r['value_level_behaviour'], 'contract_ok': r['contract']['ok'], 'contract': r['contract'],
            'invariance': r['invariance'], 'all_six_roles_bitwise': r['roles6']['all_bitwise'],
            'all_six_roles': {role: r['roles6'][role] for role in C.ROLES6},
            'runtime_audit': {'label_bearing_files_opened': a.get('label_bearing_files_opened'), 'subprocesses': a.get('subprocesses'),
                              'sockets': a.get('sockets'), 'unlisted_runtime_data_files': a.get('unlisted_runtime_data_files'),
                              'repo_code_modules_not_in_manifest': [m for m in a.get('repo_code_modules_not_in_manifest', []) if not m.startswith('results/')],
                              'third_party_libraries_loaded': a.get('third_party_libraries_loaded'),
                              'cpu_s_release_all_dev_rows': a.get('cpu_s_release_all_dev_rows'), 'error': a.get('error')}}


def deployment():
    chk = load(PRIV / 'CHECK.json')
    py, ve, integ = load(PRIV / 'REVIEW_PY314.json'), load(PRIV / 'REVIEW_VENV.json'), load(PRIV / 'INTEGRITY_PY314.json')
    std = {'unit': 'OHN_ENG_DEPLOY_CHECK_V1', 'command': 'python3 -m ohn.deploy check (coordinator check code, unchanged)',
           'interpreter': 'python3 (Homebrew 3.14.3)', 'all_bitwise': chk['all_bitwise'], 'all_refused': chk['all_refused'],
           'labels_read': chk['labels_read'],
           'packages': {k: {'anchor': v['anchor'], 'occupation': v['occupation'], 'seed': v['seed'],
                            'bitwise_equal_to_evaluated_DEV_REPORT_release': v['bitwise_equal_to_evaluated_DEV_REPORT_release'],
                            'decisions_equal': v['decisions_equal'], 'exported_keys': v['exported_keys'], 'refusals': v['refusals'],
                            'small_batch_max_abs_diff': v['small_batch_max_abs_diff'], 'cpu_s': v['cpu_s'],
                            'n_dependencies': v['n_dependencies']} for k, v in chk['packages'].items()}}
    ext = {}
    for label, rec, unit in (('python3', py, 'OHN_ENG_DEPLOY_REVIEW_PY314_V2'), ('venv', ve, 'OHN_ENG_DEPLOY_REVIEW_VENV_V1')):
        ext[label] = {'unit': unit, 'env': env_summary(rec['env'], 'python3 (Homebrew)' if label == 'python3' else '.venv/bin/python'),
                      'standard_check_all_bitwise': rec['standard_check']['all_bitwise'],
                      'standard_check_all_refused': rec['standard_check']['all_refused'],
                      'standard_check_bitwise': {k: v['bitwise'] for k, v in rec['standard_check']['per_package'].items()},
                      'packages': {k: pkg_summary(v) for k, v in rec['packages'].items()},
                      'module_parity': rec['module_parity'], 'cpu_s': rec['cpu_s']}
    names = sorted(py['packages'])
    matrix = {}
    for n in names:
        ok = {}
        for label, rec in (('python3', py), ('venv', ve)):
            p = rec['packages'][n]
            ok[label] = bool(rec['standard_check']['per_package'][n]['bitwise']['wrapper_api']['income']
                             and all(all(v.values()) for v in rec['standard_check']['per_package'][n]['bitwise'].values())
                             and p['roles6']['all_bitwise'] and p['invariance']['ok_1e12'] and p['all_refused_cleanly'] and p['contract']['ok'])
        matrix[n] = ok
    mp = {}
    for label, rec in (('python3', py), ('venv', ve), ('python3_integrity', integ)):
        for k, v in rec.get('module_parity', {}).items():
            mp.setdefault(k, {})[label] = {'all_bitwise': v.get('all_bitwise'), 'module_sha256': (v.get('module_sha256') or '')[:12],
                                           'module_source': ('committed blob 651d57b7f:ohn/protect.py' if v.get('module_file_given') else 'working tree (= HEAD)'),
                                           'error': v.get('error')}
    extra = {}
    for key, fn in (('prot_inc_packages_python3', 'CHECK_PROT_INC_PY314_SUMMARY.json'), ('prot_inc_packages_venv', 'CHECK_PROT_INC_VENV_SUMMARY.json'),
                    ('proposed_deploy_patch_v1_test_python3_SUPERSEDED_based_on_651d57b7f', 'CHECK_PATCH_TEST_SUMMARY.json'),
                    ('proposed_deploy_patch_v2_test_python3_based_on_5fb9d494a', 'CHECK_PATCH2_TEST_SUMMARY.json')):
        p = PRIV / fn
        if not p.is_file():
            extra[key] = None
            continue
        r = load(p)
        e = {k: r.get(k) for k in ('interpreter', 'python', 'numpy', 'sklearn', 'deploy_module_sha256', 'deploy_module_source', 'all_bitwise',
                                   'all_refused', 'check_refusals', 'check_extra', 'check_environment', 'cpu_s')}
        e['packages'] = {n: {'spec': b['spec'], 'n_dependencies': b['n_dependencies'],
                             **({'dependency_keys': b['dependency_keys']} if key.startswith('proposed') else {})} for n, b in r['built'].items()}
        if 'audit' in r:
            e['runtime_audit'] = {n: {kk: ([m for m in (vv or []) if not m.startswith('results/')] if kk == 'repo_code_modules_not_in_manifest' else vv)
                                      for kk, vv in au.items()} for n, au in r['audit'].items()}
        extra[key] = e
    fb = PRIV / 'FAM_B_STATE_TEST.json'
    extra['fam_b_state_check_demonstration'] = load(fb) if fb.is_file() else None
    return {'standard_check_requested_packages': std, 'extended_checks': ext, 'interpreter_matrix_all_checks_pass': matrix,
            'module_parity_all_six_roles': mp, 'grid_partition_replay': integ.get('grid_replay'), **extra}


def scrub(obj):
    """No absolute machine path in the published record: interpreters become labels, other absolute paths their last parts."""
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, str) and obj.startswith('/'):
        if '/.venv/' in obj:
            return '.venv/bin/python'
        if 'homebrew' in obj.lower() or 'python3.14' in obj:
            return 'python3 (Homebrew)'
        return '<abs>/' + '/'.join(obj.split('/')[-2:])
    return obj


# ---------------------------------------------------------------- reproducibility
def reproducibility():
    hist = CH.history()
    sha2 = collections.defaultdict(list)
    for m, h in hist.items():
        if m == 'working_tree':
            continue
        for x in h:
            sha2[x['sha256'][:12]].append(f"{m}@{x['commit']}")
    rows = []
    paths = sorted(glob.glob(str(C.RUNS / '*' / '*' / 'S*' / 'OCC.json')) + glob.glob(str(C.RUNS / '*' / '*' / 'S*' / 'RELEASE.json'))
                   + glob.glob(str(C.RUNS / 'INC' / 'STEP1' / 'S*' / 'STEP1.json')))
    for p in paths:
        r = load(p); d = Path(p).parent
        fam, cfg, s = d.parent.parent.name, d.parent.name, d.name
        cs = (r.get('code_sha256') or '')[:12] or None
        ver = []
        for v in sorted(d.glob('VERIFY*.json')):
            vr = load(v)
            vc = (vr.get('code_sha256') or '')[:12] or None
            ver.append({'file': v.name, 'utc': vr.get('utc'), 'ok': vr.get('ok', vr.get('all_bitwise')), 'code_sha256': vc,
                        'code_in_git': sha2.get(vc) if vc else None, 'interpreter': os.path.basename(vr.get('interpreter') or '') or None})
        rows.append({'release': f'{fam}/{cfg}/{s}', 'record': Path(p).name, 'kind': r.get('kind'), 'utc': r.get('utc'), 'unit_id': r.get('unit_id'),
                     'interpreter': os.path.basename(r.get('interpreter') or '') or None, 'python': r.get('python'), 'numpy': r.get('numpy'),
                     'sklearn': r.get('sklearn'), 'code_sha256_recorded': cs, 'code_in_git': sha2.get(cs) if cs else None,
                     'reconstruction': RECONSTRUCTED.get(cs) if cs else None, 'verify_records': ver})
    by_fam = collections.defaultdict(lambda: {'n': 0, 'code_hash_recorded': 0, 'code_hash_in_git': 0, 'interpreters': set()})
    for x in rows:
        f = by_fam[x['release'].split('/')[0]]
        f['n'] += 1; f['code_hash_recorded'] += bool(x['code_sha256_recorded']); f['code_hash_in_git'] += bool(x['code_in_git'])
        f['interpreters'].add(f"{x['interpreter']} {x['python']} numpy {x['numpy']}")
    return {'records': rows, 'by_family': {k: {**v, 'interpreters': sorted(v['interpreters'])} for k, v in by_fam.items()},
            'module_history': {m: h for m, h in hist.items() if m != 'working_tree'},
            'working_tree_differs_from_HEAD': {m: (h[0]['sha256'] != hist['working_tree'][m]) if h else None
                                               for m, h in hist.items() if m != 'working_tree'}}


def proposed_changes():
    """The minimal changes sent to the coordinator (not applied by the reviewer), with their tests."""
    out = {}
    dp, fb = HERE / 'proposed_ohn_deploy.patch', HERE / 'proposed_ohn_fam_b.patch'
    t2 = PRIV / 'CHECK_PATCH2_TEST_SUMMARY.json'
    if dp.is_file():
        r = load(t2) if t2.is_file() else {}
        out['ohn/deploy.py'] = {
            'patch': rel(dp), 'patch_sha256': C.sha(dp), 'applies_to': 'ohn/deploy.py at 5fb9d494a (sha256 56edd7b28782...)',
            'patched_module_sha256': r.get('deploy_module_sha256'), 'test_unit': 'OHN_ENG_DEPLOY_PATCH_TEST_V2',
            'test_all_bitwise': r.get('all_bitwise'), 'test_all_refused': r.get('all_refused'),
            'test_unlisted_runtime_files': {n: a.get('unlisted_runtime_data_files') for n, a in (r.get('audit') or {}).items()},
            'contents': ['_upstream_deps: family-B base-learner files; recursion into the witness chain; INC stored-income-score mirror files',
                         '_run_deps: VERIFY* excluded (mutable)', 'REPO:: dependencies: both SOURCE_ADMISSION.json records (+ _load branch)',
                         'PACKAGE.json: environment_at_build and code_closure_sha256; check(): environment and code_closure_unchanged_since_build',
                         'check(): raw-input CSV under .private/ and removed; PINCP and six CSV refusal cases']}
    if fb.is_file():
        demo = PRIV / 'FAM_B_STATE_TEST.json'
        out['ohn/fam_b.py'] = {'patch': rel(fb), 'patch_sha256': C.sha(fb), 'applies_to': 'ohn/fam_b.py at 5c9c6cef5 (HEAD)',
                               'optional': True, 'contents': ['predict checks STATE.json against OCC.json state_sha256 (when OCC.json exists)'],
                               'demonstration_unit': 'OHN_ENG_FAM_B_STATE_TEST_V1', 'demonstration': load(demo) if demo.is_file() else None}
    return out


def queue_units():
    out = {}
    for line in (C.PRIV / 'compute' / 'ledger.jsonl').read_text().splitlines():
        e = json.loads(line)
        if e.get('event') == 'unit_end' and e['unit_id'].startswith('OHN_ENG_'):
            out[e['unit_id']] = {k: e[k] for k in ('stage', 'status', 'returncode', 'charged_cpu_s', 'wall_s', 'utc')} | \
                                {'peak_rss_gib': round(e['child_maxrss_bytes_platform'] / 2 ** 30, 2)}
        elif e.get('event') == 'charge' and e.get('label') == 'ENGINEERING_REVIEW_AGENT':
            out.setdefault('out_of_queue_charges', []).append({k: e[k] for k in ('charged_cpu_s', 'note', 'utc')})
    out['queue_cpu_s_total'] = round(sum(v['charged_cpu_s'] for k, v in out.items() if k.startswith('OHN_ENG_')), 3)
    return out


STATUS = {
    'E1': ('acknowledged by the coordinator (reply between 07:15Z and 07:27Z): label-free DEPLOYMENT_CHECKS_<name>.json for both seeds of every '
           'candidate before the formal nominate run (pending)'),
    'E2': 'fixed in code by the coordinator (7b80b1d5d: utility_rows reads RELEASE.json); UTILITY_FRONTIER.csv to be regenerated',
    'E4': ('partly fixed by the coordinator (5fb9d494a: PROT witness run directory and module; INC Step-1, witness run directory and modules); '
           'still missing per the runtime audit: B base-learner files for every chain containing B, the INC stored-income-score mirror files '
           '(income teacher, JOINT / LOCAL E20), both SOURCE_ADMISSION.json records; proposed_ohn_deploy.patch (re-based on 5fb9d494a) closes them'),
    'E5': 'mitigated by process (coordinator: packages built only after the last verify); VERIFY*.json still hashed (excluded by the proposed patch)',
    'E9': ('ohn/protect.py committed 06b0ca47a (07:21:30Z) before any P3 fit; the units that ran the uncommitted file are recorded by the '
           'coordinator as deviation 14'),
    'E3': 'open; proposed_ohn_deploy.patch records environment_at_build and code_closure_sha256 in PACKAGE.json and check() reports them',
    'E6': 'open; proposed_ohn_deploy.patch writes the CSV under .private/ and removes it; the two remaining system-temp copies belong to their owners',
    'E7': 'open; proposed_ohn_deploy.patch adds the PINCP case and six CSV refusal cases to check()',
    'E21': 'open; proposed_ohn_fam_b.patch adds the STATE.json check to fam_b.predict (demonstrated: fam_b_state_test.py)',
}


def F(id_, severity, area, location, finding, affects, fix):
    return {'id': id_, 'severity': severity, 'area': area, 'location': location, 'finding': finding,
            'affects_recorded_result': affects, 'required_fix': fix, 'status': STATUS.get(id_, 'open')}


FINDINGS = [
    F('E1', 'HIGH', 'nomination plumbing', 'ohn/nominate.py:40-52, :99; ohn/rules.py:457-460',
      "_deploy_ok only reads results/<campaign>/**/DEPLOYMENT_CHECKS*.json and needs BOTH seeds of a candidate; no such file exists "
      "(ohn.deploy check writes to --report, by convention under .private/). Every candidate is then NOT_ADMISSIBLE ('no runnable "
      "package'), and six_raw_fields is set equal to runnable, so a parity check is also taken as the six-field evidence.",
      'no recorded result yet (no NOMINATION.json); would decide the nomination outcome if run as is (coordinator notified 07:15Z)',
      'before nominate: build and check both seeds of every candidate package (including INC:...@COMPLETE) and save the label-free '
      'check JSON as results/<campaign>/deploy/DEPLOYMENT_CHECKS_<name>.json (or read the private reports); record the six-field '
      'evidence separately (code review + runtime audit of this record), not as parity'),
    F('E2', 'MEDIUM', 'reporting', 'ohn/tables.py:82',
      'utility_rows requires OCC.json, so income-branch complete releases (runs/INC/*/S*/RELEASE.json + EVAL_MS_COMPLETE.json) are '
      'silently omitted from UTILITY_FRONTIER.csv.', 'published table incomplete (no number wrong)',
      'also read RELEASE.json directories (anchor COMPLETE) in utility_rows; regenerate the table'),
    F('E3', 'MEDIUM', 'deployment record (PROTOCOL section 7)', 'ohn/deploy.py@5fb9d494a:66-103 (was 41-77 at 651d57b7f)',
      'PACKAGE.json records no interpreter, Python / numpy / scikit-learn / scipy versions, CPU or platform, and hashes only the '
      'occupation family module; the runtime audit shows the release path also loads ohn/{common,deploy,devrun}.py, '
      'ihs/{common,bases,fam_h}.py, acs_replication/{deploy,schema,trees}.py, scm_bench/common.py (plus ohn/fam_a.py and ohn/fam_b.py '
      'for the INC and PROT witness chain), none hashed in the package.', 'no; requirement gap',
      'record interpreter, library versions, platform/CPU and the SHA-256 of the full imported code closure at build; check them in _load'),
    F('E4', 'MEDIUM', 'deployment custody', 'ohn/deploy.py@5fb9d494a:41-63, 66-103 (was 56-72 at 651d57b7f); ohn/fam_b.py:678-689',
      'the dependency manifest omits runtime data files outside the configuration run directory (runtime audit): B -> '
      'runs/B/BASES/<learner>/S<s>/{MODEL.pkl,BASE.json}; INC complete release -> runs/INC/STEP1/S<s>/{STEP1.json,HGB_PO.pkl}, the '
      "witness's runs/B/B2_SINGLE/S<s>/STATE.json and BASES files, three mirror teacher / Stage-B files; PROT -> the witness's files "
      'and code; every package -> results/pcrl_income_headroom_successor_v1/SOURCE_ADMISSION.json (read by ihs admitted()). Runtime '
      "integrity still holds through each module's own hash chain, except that fam_b.predict never checks its own STATE.json, which "
      'INC and PROT packages reach unverified. A backup or restore driven by the manifest would miss these files.',
      'no; custody / restore gap', "derive the manifest from the runtime audit (or each module's declared dependencies), include the "
      "witness chain, and make fam_b.predict check STATE.json against OCC.json's state_sha256"),
    F('E5', 'LOW', 'deployment fragility', 'ohn/deploy.py@5fb9d494a:41-42 (_run_deps; was :59, :67)',
      'manifests hash mutable non-functional files (VERIFY.json in every run directory; EVAL_MS.json / VERIFY.json in the anchor '
      'mirror directory); A, B, INC and PROT verify overwrite VERIFY.json, so a verify after a build makes the package unloadable '
      '(fail-closed). A package built while its module is being edited goes stale the same way (observed: ohn/protect.py edited '
      '07:08:47Z, 3 s after a review build).', 'no', 'exclude VERIFY*/EVAL_* from deps; build packages only from committed code after the stage is frozen'),
    F('E6', 'LOW', 'custody hygiene', 'ohn/deploy.py@5fb9d494a:197-199 (was 171-173)',
      'check() writes the full DEV_REPORT raw-input CSV (5,943 people, six raw fields, no labels) to a system mkdtemp() directory and '
      'never deletes it. Three copies were found in the system temp directory (05:02Z coordinator OHN_DEPLOY_TEST_V1; 07:05:33Z '
      'verifier ver_deploy_*; 07:05:37Z this review, removed by the reviewer); directories are owner-only (0700).', 'no',
      'write the CSV under .private/ and delete it in a finally block; owners remove the two remaining copies'),
    F('E7', 'LOW', 'deployment test coverage', 'ohn/deploy.py@5fb9d494a:210-230 (was 184-204)',
      "check()'s refusal set never exercises release_csv refusals and tests one target (OCCP), not PINCP. Covered here: 42 cases "
      '(API, CSV, underlying; SEX / sex / Sex, PINCP, OCCP, extra, reordered, missing, duplicated, renamed, non-finite, mis-shaped, '
      'hidden-output requests) all refused with PermissionError / ValueError under both interpreters.', 'no',
      'add the CSV and PINCP cases to ohn.deploy.check'),
    F('E8', 'LOW', 'input validation', 'acs_replication/deploy.py:39-41 (unmodified source encoder); ohn/deploy.py:94-101',
      'value-level problems are not schema problems: None, empty or unseen categorical values are accepted and encoded as the block '
      "__OTHER__ level; a None numeric raises TypeError (outside the documented PermissionError / ValueError); out-of-population "
      'ages are extrapolated.', 'no', 'document in the package, or refuse empty / None values in check_rows'),
    F('E9', 'MEDIUM', 'reproducibility record', 'ohn/fam_b.py (records), ohn/fam_c.py, ohn/protect.py, ohn/devrun.py:73-75',
      'fam_b.py and fam_c.py were not in the registration commit and were first committed at 05:45:46Z (B) and 06:01:42Z (C), after '
      'their first fits (B bases from 05:13Z, B1 fits 05:43Z; C1 fits 05:41-05:44Z). B records no code hash at all (OCC.json, BASE.json) '
      'and no scikit-learn version in OCC.json; four recorded C hashes are absent from git (e89393f2 C1_RB_D1 / C1_RB_D2; 4c18d641 '
      'CC_CONT_D4; 9631a62b C2_XF_D2 S0; 81137716 C2_XF_D1 S0 / S1, CC_XF_CONT_D4); A1_PO_8 / A1_PO_16 seed 1 record 8be2baeb (absent '
      'from git, reconstructed exactly: committed cb89a8b51 plus an unused diagnostic block, so the fit code is the committed one); '
      'PROT OCC.json records no code hash (VERIFY.json has an end-of-fit hash) and the P2A seed-1 units started 07:10-07:11Z on an '
      'uncommitted ohn/protect.py (3d06c0b5). xgboost / torch versions are not in the C / PROT release records. Every release '
      'nevertheless replays bitwise from features with the committed code (family verify records and this review).',
      'no recorded number found affected; fit-time code of all B releases and of the seven C releases is identifiable only by timing',
      'hash each module at import time (plus git HEAD and a dirty flag) into every record; commit before any fit; record all library versions'),
    F('E10', 'LOW', 'record accuracy', 'ohn/fam_a.py:522; ohn/fam_inc.py:261; protect VERIFY',
      'code_sha256 is the file on disk when the record is written, not the code that ran: INC STEP1 S1 records bb7b3290 (print-fixed) '
      'although it executed 670a8cf6 and crashed in the old print (deviation 8).', 'record only (print-only difference)', 'as E9'),
    F('E11', 'LOW', 'documentation of a selected witness', 'ohn/fam_a.py:485-486, :499',
      "A1_PO_16 and A1_PO_8 OCC.json: 'operation' says the release is computed 'on the admitted 145-column coding of X' and that T* "
      "uses the 'OHN1_crossfit partition'; 'inputs_used' says 'used as is'. Actual: the six-column compact_po re-coding (deterministic "
      "from X) and the seed-specific inner partition OHN_A_cv_S<seed> (correct in 'inner_cv_tag'); the partition text is wrong for all A configurations.",
      'record text only', 'append an erratum (records are write-once) and fix the text for later fits'),
    F('E12', 'LOW', 'gate documentation', 'ohn/common.py:143-146; PROTOCOL_DEVIATIONS.md item 9',
      'load_role enforces the registration receipt and a logged purpose for MODEL_SELECT labels, not the registered call paths: '
      '{outside} of {total} logged MODEL_SELECT label reads (at {utc}) came through other scripts ({detail}); deviation 9 lists two '
      '(OHN_RR_PROBE_A_S0_V1 / S1_V1). All are logged with unit ids and diagnostic purposes (units in gates.access_logs).', 'no',
      'extend deviation 9 to the full unit list (or state that section 7 authorizes the verifier reads)'),
    F('E13', 'MEDIUM', 'future reporting gate', 'ohn/receipt.py:37-54',
      'verify writes the private record even when verification fails and refuses any second verification (write-once): one failed '
      'attempt (e.g. before the push is visible) permanently blocks dev_report_truth unless a private file is deleted by hand. verify '
      'also does not compare the published blob ids with the local files (checked here: all five registration files identical).',
      'no (registration verified first time)', 'write the record only on success (log failures separately) and compare blob ids'),
    F('E14', 'LOW', 'queue robustness', 'ohn/jobqueue.py:132-140, :150-157, :172-174, :184-188',
      'with psutil importable (the .venv interpreter has it) _peak_rss_loop blocks until the child exits, so max_wall_s is never '
      'enforced (Homebrew python3, used for every wrapper so far, lacks psutil: sampled_peak_rss is null for every unit); refusals '
      '(window, budget, RSS, duplicate id) leave no ledger record; id uniqueness is checked only against finished units; a killed '
      "waiter whose PID is reused would block the FIFO head.", 'no (no orphan, duplicate start or RSS overrun in the ledger)',
      'bound the sampler loop by max_wall_s; log refusals; check running ids'),
    F('E15', 'LOW', 'write-once discipline', 'ohn/devrun.py:71-78, :206',
      'save_occupation writes OCC.npz before validating the record keys (a missing key leaves an orphan OCC.npz and burns the id); '
      'EVAL_MS_<anchor>.json is overwritten on a re-score.', 'no (no configuration was scored twice; 142 ledger rows)',
      'validate before writing; refuse or version re-scores'),
    F('E16', 'INFO', 'inherited metric plumbing', 'ihs/evaluate.py:94',
      'the guard margin_se_units is +inf when se == 0 even if the guard is violated; inert because ohn/rules.py:146 floors the SE.', 'no', 'none required'),
    F('E17', 'INFO', 'witness selection', 'ohn/witness.py:38',
      "task-only status is a roles_read string test ('SEX' without 'NO SEX'); correct for every current record (the A / B / C fits never read SEX).",
      'no', 'prefer a structural field (e.g. sex_used) in the records'),
    F('E18', 'INFO', 'reader-grid lock', 'ohn/devreaders.py:270-282',
      'the lock does not cover ohn/common.py (household partition) and run() never re-checks the DEV_GRID household sets; replayed here: identical.',
      'no', 'add ohn/common.py to LOCK_FILES or compare household-set hashes in run()'),
    F('E19', 'INFO', 'protection stage', 'ohn/protect.py@651d57b7f:573-575',
      "fit uses the witness's predict under the fitting interpreter with a 1e-9 tolerance instead of the saved witness release; "
      'bitwise in practice (OHN_PROT_PARITY_* units and the replays here).', 'no', 'require bitwise parity or use the saved arrays'),
    F('E20', 'INFO', 'gate design', 'ohn/common.py:106-122, :147-152, :269-282',
      'any edit to the five receipt-pinned files (PROTOCOL.md, DEV_GRID.json, EXPOSURE_TABLE.md, RULES.json, SELECTION_RULE.md) closes '
      'every label read for the rest of the campaign; corrections must stay in PROTOCOL_DEVIATIONS.md (as done for deviation 4). The '
      'ATTACK gate checks only that a FINALIST_LOCK is committed (no content, no receipt); dev_report_truth does not require a FINALIST_LOCK.',
      'no', 'none required; keep in mind at the final stage'),
    F('E21', 'LOW', 'parameter-file anchoring', 'ohn/protect.py@651d57b7f:645-651, :700-705, :733-752; ohn/fam_c.py:540-551; ohn/fam_b.py:678-689',
      'the B, C and PROT predict functions do not check their parameter file (STATE.json / PARAMS.json) against the release record. '
      "PROT writes THETA.npz / CODEBOOK.npz and PARAMS.json (scale, witness) AFTER save_occupation, so OCC.json records none of them; "
      'outside a package nothing pins the release strength (PARAMS.json pins THETA / CODEBOOK). The family verify and the package '
      'manifest do catch a change.', 'no', 'write PARAMS / THETA / CODEBOOK before save_occupation and record their hashes in OCC.json; '
      'check them (and B STATE.json, C PARAMS.json) in predict'),
]


def findings_now(acc, utc):
    """FINDINGS with the live MODEL_SELECT read counts of this build in E12."""
    by = acc['model_select_reads_by_path']
    out_n = sum(v for k, v in by.items() if 'registered path' not in k)
    detail = ', '.join(f'{v} {k}' for k, v in sorted(by.items()) if 'registered path' not in k)
    res = []
    for f in FINDINGS:
        f = dict(f)
        if f['id'] == 'E12':
            f['finding'] = f['finding'].format(outside=out_n, total=acc['model_select_label_reads'], utc=utc, detail=detail)
        res.append(f)
    return res


def main():
    now = C.now()
    acc = access_summary()
    rec = {'schema': 'OHN1-engineering-review-checks-v1', 'utc': now, 'role': 'engineering and deployment reviewer (agent)',
           'labels_read_by_reviewer': False,
           'note': 'exploratory development campaign on previously used records; this record concerns code, gates, deployment and '
                   'provenance only and makes no statistical claim',
           'code_sha256': {f.name: C.sha(f) for f in sorted(HERE.glob('*.py'))},
           'queue_units': queue_units(),
           'queue_units_withdrawn_before_start': {'OHN_ENG_DEPLOY_REVIEW_PY314_V1': 'stopped with SIGINT while waiting for a slot (no unit_start, '
                                                  'no ledger record): its PROT package had become stale when ohn/protect.py was edited 3 s after the build'},
           'gates': {'negative_tests': gate_tests(), 'access_logs': acc, 'receipts': receipt_summary()},
           'deployment': deployment(), 'reproducibility': reproducibility(), 'findings': findings_now(acc, now),
           'proposed_changes': proposed_changes()}
    rec = scrub(rec)
    C.write_json(OUT, rec)
    print(json.dumps({'written': rel(OUT), 'gate_tests_all_refused': rec['gates']['negative_tests']['all_refused'],
                      'interpreter_matrix': rec['deployment']['interpreter_matrix_all_checks_pass'], 'n_findings': len(rec['findings']),
                      'E12': [f['finding'] for f in rec['findings'] if f['id'] == 'E12'][0][:260]}, indent=1))


if __name__ == '__main__':
    main()
