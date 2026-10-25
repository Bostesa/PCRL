"""CHECK7: provenance spot-checks (read-only; git is used only through `git show <commit>:<path>`).

  7a  SHA-256 of every vendored file (D/code/vendor, no caches) equals SOURCE_PROVENANCE.json, and the five lock-listed recipe
      files equal the DRF1 publication's DEVGRID_CODE_LOCK.json (also its drf/devreaders.py entry).
  7b  the vendored files equal the blobs of the DRF1 research tip 913eb5c91 they were copied from; nrm/fa/prc.py's blob hash equals
      SOURCE_PROVENANCE.json and the two copied functions in edg/source_prc.py are text-equal to the blob (own AST extraction).
  7c  the publication code (drf/__init__, devreaders, dr, finite_law, solver) equals SOURCE_PROVENANCE.json and the publication
      commit c4896479e's blobs.
  7d  the registration code lock (GRID.json code_sha256, commit c169705c2) against the current files: mismatches and unlocked files
      are listed and matched to the PROTOCOL.md section 9 amendments.
  7e  LAWS.json: recorded p / q hashes, supports and A values equal the laws rebuilt by edg.laws and the verifier's exact A."""
import ast
import hashlib
import json
import subprocess

import vlib as V

V.study_imports()
from edg import laws as L             # noqa: E402

T0 = V.utc_now()
D, W = V.D, V.W
rec = {'check': 7, 'name': 'provenance'}
fails = []


def git_blob_sha(commit, path):
    out = subprocess.run(['git', '-C', str(W), 'show', f'{commit}:{path}'], capture_output=True)
    return (hashlib.sha256(out.stdout).hexdigest() if out.returncode == 0 else None), out.stdout


prov = json.load(open(D / 'SOURCE_PROVENANCE.json'))
lock = json.load(open(W / 'publications' / 'pcrl-direct-recovery-frontier-v1' / 'results' / 'DEVGRID_CODE_LOCK.json'))['files_sha256']
# 7a / 7b
va = {}
vendor_files = sorted(p for p in (D / 'code' / 'vendor').rglob('*.py') if '__pycache__' not in p.parts)
for p in vendor_files:
    rel = str(p.relative_to(D / 'code' / 'vendor'))
    h = V.sha256_file(p)
    rp = prov['vendored'].get(rel, {})
    blob, _ = git_blob_sha('913eb5c91', rel)
    r = {'sha256': h, 'equals_SOURCE_PROVENANCE': h == rp.get('sha256'), 'in_DEVGRID_lock': rel in lock,
         'equals_DEVGRID_lock': (h == lock[rel]) if rel in lock else None, 'equals_blob_913eb5c91': h == blob,
         'provenance_equals_lock_flag': rp.get('equals_lock')}
    va[rel] = r
    if not (r['equals_SOURCE_PROVENANCE'] and r['equals_blob_913eb5c91'] and (r['equals_DEVGRID_lock'] in (True, None))):
        fails.append(f'7a/7b {rel}')
five = ['ffv/features.py', 'ffv/readers.py', 'scm_bench/readers.py', 'task_conditioned_representation/attacks.py',
        'task_conditioned_representation/inference.py']
rec['7a_vendored'] = va
rec['7a_five_lock_listed_all_equal'] = all(va[f]['equals_DEVGRID_lock'] is True for f in five)
rec['7a_unlisted_vendored_files'] = sorted(set(va) - set(prov['vendored']))
rec['7a_listed_but_missing'] = sorted(set(prov['vendored']) - set(va))
if not rec['7a_five_lock_listed_all_equal'] or rec['7a_unlisted_vendored_files'] or rec['7a_listed_but_missing']:
    fails.append('7a five / listing')
blob_sha, blob = git_blob_sha('913eb5c91', 'nrm/fa/prc.py')
src = open(D / 'code' / 'edg' / 'source_prc.py').read()


def fn_text(text, name):
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    return None


btxt = blob.decode()
rec['7b_prc'] = {'blob_sha256': blob_sha, 'equals_SOURCE_PROVENANCE': blob_sha == prov['nrm/fa/prc.py']['sha256'],
                 'auc_text_equal': fn_text(btxt, 'auc') is not None and fn_text(btxt, 'auc') == fn_text(src, 'auc'),
                 'lookup_scores_text_equal': fn_text(btxt, 'lookup_scores') is not None
                 and fn_text(btxt, 'lookup_scores') == fn_text(src, 'lookup_scores')}
if not all(rec['7b_prc'][k] for k in ('equals_SOURCE_PROVENANCE', 'auc_text_equal', 'lookup_scores_text_equal')):
    fails.append('7b prc')
# 7c publication code
pc = {}
for rel, meta in prov['publication_code'].items():
    path = W / 'publications' / 'pcrl-direct-recovery-frontier-v1' / rel
    h = V.sha256_file(path)
    b, _ = git_blob_sha('c4896479e', f'publications/pcrl-direct-recovery-frontier-v1/{rel}')
    k = rel.replace('code/', '')
    pc[rel] = {'sha256': h, 'equals_SOURCE_PROVENANCE': h == meta['sha256'], 'equals_blob_c4896479e': h == b,
               'equals_DEVGRID_lock': (h == lock[k]) if k in lock else None}
    if not (pc[rel]['equals_SOURCE_PROVENANCE'] and pc[rel]['equals_blob_c4896479e'] and pc[rel]['equals_DEVGRID_lock'] in (True, None)):
        fails.append(f'7c {rel}')
rec['7c_publication_code'] = pc
# 7d registration lock vs current files
grid_now = json.load(open(D / 'GRID.json'))
gsha, gblob = git_blob_sha('c169705c2', 'results/pcrl_estimator_diagnostic_v1/GRID.json')
reg_lock = json.loads(gblob.decode())['code_sha256']
cur = {str(p.relative_to(D)): V.sha256_file(p) for p in sorted((D / 'code').rglob('*.py')) if '__pycache__' not in p.parts
       and 'roles' not in p.parts}
cur['edg_run.py'] = V.sha256_file(D / 'edg_run.py')
rec['7d_GRID_json_unchanged_since_registration'] = grid_now['code_sha256'] == reg_lock and V.sha256_file(D / 'GRID.json') == gsha
rec['7d_locked_files_changed_since_registration'] = sorted(k for k in reg_lock if cur.get(k) != reg_lock[k])
rec['7d_files_not_in_registration_lock'] = sorted(k for k in cur if k not in reg_lock)
rec['7d_amendment_mapping'] = {'edg_run.py': 'A1/A2 (custody, publish dispatch; posthoc dispatch added with A4)',
                               'code/edg/selftest.py': 'A3 (skip the unavailable source blob in a bundle clone; 2 lines)',
                               'code/edg/custody.py': 'A1', 'code/edg/publish.py': 'A2', 'code/edg/posthoc.py': 'A4 (post hoc)'}
undisclosed = [k for k in rec['7d_locked_files_changed_since_registration'] + rec['7d_files_not_in_registration_lock']
               if k not in rec['7d_amendment_mapping']]
rec['7d_undisclosed_changes'] = undisclosed
estimator_files = ['code/edg/estimators.py', 'code/edg/laws.py', 'code/edg/simulate.py', 'code/edg/bank.py', 'code/edg/analyze.py',
                   'code/edg/source_prc.py', 'code/edg/__init__.py', 'code/edg/grid.py']
rec['7d_estimator_analysis_files_equal_registration_lock'] = all(cur[k] == reg_lock[k] for k in estimator_files)
if undisclosed or not rec['7d_estimator_analysis_files_equal_registration_lock'] or not rec['7d_GRID_json_unchanged_since_registration']:
    fails.append('7d')
# 7e LAWS.json vs rebuilt laws and the verifier's exact A
lj = json.load(open(D / 'LAWS.json'))
le, worst = {}, 0.0
for key, d in lj['laws'].items():
    fam, pi, occ = key.split('|')
    law = L.build(fam, float(pi), occ)
    p_ok = hashlib.sha256(law['p'].tobytes()).hexdigest() == d['p_sha256']
    q_ok = hashlib.sha256(law['q'].tobytes()).hexdigest() == d['q_sha256']
    vv = {}
    for name, vd in d['views'].items():
        wid, K, m1, m0 = V.study_law_exact_view(law, name)
        A_ex = float(V.pop_optimal_auc_exact(m1, m0))
        dif = abs(vd['A_population_optimal_auc'] - A_ex)
        worst = max(worst, dif)
        vv[name] = {'support_recorded': vd['observable_support'], 'support_own': K, 'A_recorded_minus_exact': vd['A_population_optimal_auc'] - A_ex}
        if K != vd['observable_support'] or dif > 1e-12:
            fails.append(f'7e {key} {name}')
    le[key] = {'p_sha256_ok': p_ok, 'q_sha256_ok': q_ok, 'checks_all_pass_recorded': d['checks']['all_pass'], 'views': vv}
    if not (p_ok and q_ok and d['checks']['all_pass']):
        fails.append(f'7e {key} hashes')
rec['7e_laws_json'] = le
rec['7e_max_abs_A_recorded_minus_exact'] = worst
rec['7e_all_checks_pass_recorded'] = lj['all_checks_pass']
rec['failures'] = fails
rec['status'] = 'PASS' if not fails else 'FAIL'
print(rec['status'], fails)
print('7a five', rec['7a_five_lock_listed_all_equal'], '7b', rec['7b_prc'], '7d changed', rec['7d_locked_files_changed_since_registration'],
      'unlocked', rec['7d_files_not_in_registration_lock'], 'undisclosed', undisclosed, 'est-equal', rec['7d_estimator_analysis_files_equal_registration_lock'],
      '7e worst', worst)
V.write_out('CHECK7_provenance.json', rec, started_utc=T0)
