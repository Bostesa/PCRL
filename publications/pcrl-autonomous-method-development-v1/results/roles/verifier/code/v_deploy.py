"""Verifier unit AMD_VERIFY_DEPLOY_V1 (label-free): check 8 (deployment parity and refusals through amd.deploy.release /
release_csv on the restored FINALIST and TASKONLY packages), bank integrity against the FINALIST_LOCK sources, the lock hash
chain, and the distance of DEV_REPORT / MODEL_SELECT income statistics to the coarse thresholds (numerical fragility)."""
from __future__ import annotations
import csv
import json
import shutil
import sys

sys.path.insert(0, '<repository>/.claude/worktrees/pcrl-autonomous-method-development-v1/.private/pcrl_autonomous_method_development_v1/verifier/independent')
import numpy as np
import vcommon as V
from amd import common as C
from amd import deploy as D            # entry points under test (release, release_csv, raw_rows)

out = {'unit': 'AMD_VERIFY_DEPLOY_V1', 'utc': C.now()}
ROLES4 = ('MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT')
PERM = ['AGEP', 'WKHP', 'COW', 'SCHL', 'MAR', 'POBP']

# ---------------------------------------------------------------- lock hash chain
sha = {n: C.sha(V.RES / n) for n in ('PROMOTION.json', 'FINALIST_LOCK.json', 'ANALYSIS_LOCK.json', 'REPORTING_LOCK.json', 'SERVICE_BANK_MANIFEST.json')}
fl = json.loads((V.RES / 'FINALIST_LOCK.json').read_text()); al = json.loads((V.RES / 'ANALYSIS_LOCK.json').read_text())
rl = json.loads((V.RES / 'REPORTING_LOCK.json').read_text()); man = json.loads((V.RES / 'SERVICE_BANK_MANIFEST.json').read_text())
rep = json.loads((V.RES / 'report' / 'DEV_REPORT_RESULTS.json').read_text())
ev = [json.loads(l) for l in (V.PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl').read_text().splitlines() if l.strip()]
chain = {'promotion_in_finalist_lock': fl['promotion_sha256'] == sha['PROMOTION.json'],
         'finalist_lock_in_analysis_lock': al['finalist_lock_sha256'] == sha['FINALIST_LOCK.json'],
         'finalist_lock_in_bank_manifest': man['finalist_lock_sha256'] == sha['FINALIST_LOCK.json'],
         'draft_hash_in_analysis_lock': al['frozen_design_sha256'] == C.sha(V.RES / 'roles' / 'statistical_reviewer' / 'CLAIMS_AND_ENDPOINTS_DRAFT.md'),
         'bank_files_in_manifest': all(C.sha(V.PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz') == man['bank_files_sha256'][f'S{s}'] for s in (0, 1)),
         'report_locks_match_files': all(rep['locks_sha256'][n] == sha[n] for n in rep['locks_sha256']),
         'access_events_reporting_lock_hash': all(e['reporting_lock_sha256'] == sha['REPORTING_LOCK.json'] for e in ev)}
for s in (0, 1):
    p = V.PRIV / 'runs' / f'READERS_{s}' / 'DEV_REPORT_SCORES.npz'
    chain[f'reader_scores_S{s}_in_reporting_lock'] = C.sha(p) == rl['readers'][f'S{s}']['dev_report_scores_sha256']
out['lock_chain'] = chain
out['reporting_lock_keys'] = list(rl.keys())

# ---------------------------------------------------------------- bank integrity vs registered sources
integ = {}
for s in (0, 1):
    with np.load(V.PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz') as z:
        B = {k: z[k].copy() for k in z.files}
    frozen, _ = C.frozen_bank(s, roles=ROLES4); um = C.um_service_bank(s, roles=ROLES4)
    for entry in fl['bank']:
        src = entry['source']; role = entry['role']
        if src['kind'] == 'release':
            relj = json.loads((V.PRIV / 'runs' / src['family'] / src['config_id'] / f'S{s}' / 'RELEASE.json').read_text())
            with np.load(V.PRIV / 'runs' / src['family'] / src['config_id'] / f'S{s}' / 'RELEASE.npz') as z:
                o = {r: {t: z[f'{r}__{t}'].copy() for t in V.TASKS} for r in ROLES4}
            sha_ok = relj['release_sha256'] == fl['release_sha256'][f"{src['family']}/{src['config_id']}"][f'S{s}']
        else:
            bank_src = frozen if src['kind'] == 'frozen' else um
            o = {r: {t: bank_src[src['name']][r][t] for t in V.TASKS} for r in ROLES4}
            sha_ok = None
        integ[f'S{s}|{role}'] = {'source': src, 'bitwise_all_roles': all(np.array_equal(np.asarray(o[r][t], dtype=np.float64), B[f'{role}__{r}__{t}']) for r in ROLES4 for t in V.TASKS),
                                 'release_sha256_matches_lock': sha_ok}
out['bank_integrity'] = integ

# ---------------------------------------------------------------- distance of the income statistic to the coarse thresholds
prox = {}
feats = {r: C.load_role(r, ('features',)) for r in ('MODEL_FIT', 'MODEL_SELECT', 'DEV_REPORT')}
for s in (0, 1):
    rel = json.loads((V.PRIV / 'runs' / 'B' / 'B3CTL_UNCOUPLED_MIX50_C6A' / f'S{s}' / 'RELEASE.json').read_text())
    thr = np.asarray(rel['thresholds'])
    te = V.Teacher(json.loads(C.admitted(f'stage_a/S{s}/TREE_TEACHER_income.json').read_text()))
    th = {a: np.load(C.admitted(f'stage_b/S{s}/{a}__E20.npz'))['theta_income'] for a in ('LOCAL_TREE', 'JOINT_TREE')}
    for role, F in feats.items():
        lv = te.leaves(F['X'], F['feature_names'])
        d = 0.5 * te.raw(lv, th['LOCAL_TREE']).reshape(-1) + 0.5 * te.raw(lv, th['JOINT_TREE']).reshape(-1)
        dist = np.abs(d[:, None] - thr[None, :]).min(1)
        prox[f'S{s}|{role}'] = {'min_abs_distance_to_threshold': float(dist.min()), 'n_within_1e-12': int((dist <= 1e-12).sum()),
                                'n_within_1e-9': int((dist <= 1e-9).sum())}
out['threshold_proximity'] = prox

# ---------------------------------------------------------------- deployment parity and refusals
rows = D.raw_rows()
tmp = V.OUT / 'tmp_deploy'
if tmp.exists():
    shutil.rmtree(tmp)
tmp.mkdir(parents=True)


def write_csv(path, rws, header, extra=None):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header)
        for r in rws:
            rr = dict(r, **(extra or {}))
            w.writerow([str(rr[c]) for c in header])


csv_path = tmp / 'dev_raw.csv'; write_csv(csv_path, rows, PERM)
dep = {}
for name, svc in (('FINALIST', 'FINALIST'), ('TASKONLY', 'TASKONLY')):
    for s in (0, 1):
        pkg = V.PRIV / 'deploy' / f'{name}_S{s}'
        with np.load(V.PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz') as z:
            ref = {t: z[f'{svc}__DEV_REPORT__{t}'].copy() for t in V.TASKS}
        api = D.release(pkg, rows); viacsv = D.release_csv(pkg, csv_path)
        rec = {}
        for lab, res in (('api', api), ('csv', viacsv)):
            for t in V.TASKS:
                rec[f'{lab}_{t}_prob_bitwise'] = bool(np.array_equal(res[t]['probability'], ref[t]))
                rec[f'{lab}_{t}_prob_max_abs'] = float(np.max(np.abs(res[t]['probability'] - ref[t])))
                rec[f'{lab}_{t}_decision_equal'] = bool(np.array_equal(res[t]['decision'], ref[t].argmax(1)))
                rec[f'{lab}_keys'] = sorted(res[t].keys())
        refu = {}
        few = rows[:5]
        csv_cases = {'extra_column': PERM + ['PUMA'], 'reordered_columns': PERM[::-1], 'missing_column': PERM[:-1], 'sex_column': PERM + ['SEX'],
                     'sex_replacing_field': PERM[:-1] + ['SEX'], 'sex_first': ['SEX'] + PERM, 'lowercase_sex_column': PERM + ['sex'], 'swap_two_columns': [PERM[1], PERM[0]] + PERM[2:]}
        for cname, header in csv_cases.items():
            p = tmp / f'{name}_{s}_{cname}.csv'; write_csv(p, few, header, extra={'SEX': '2', 'sex': '2', 'PUMA': '101'})
            try:
                D.release_csv(pkg, p); refu[f'csv_{cname}'] = 'ACCEPTED'
            except (PermissionError, ValueError, KeyError) as e:
                refu[f'csv_{cname}'] = type(e).__name__
        api_cases = {'sex_key': [dict(r, SEX=1) for r in few], 'lowercase_sex_key': [dict(r, sex=1) for r in few], 'extra_key': [dict(r, PUMA=101) for r in few],
                     'reordered_keys': [{c: r[c] for c in PERM[::-1]} for r in few], 'missing_key': [{c: r[c] for c in PERM[:-1]} for r in few]}
        for cname, rr in api_cases.items():
            try:
                D.release(pkg, rr); refu[f'api_{cname}'] = 'ACCEPTED'
            except (PermissionError, ValueError, KeyError) as e:
                refu[f'api_{cname}'] = type(e).__name__
        for h in ('leaf_ids', 'raw_score', 'theta', 'sex_specific_probability', 'teacher_probability', 'propensity', 'coarse_bin', 'income_statistic', 'h_logit'):
            try:
                D.release(pkg, few, ('probability', h)); refu[f'hidden_{h}'] = 'ACCEPTED'
            except (PermissionError, ValueError, KeyError) as e:
                refu[f'hidden_{h}'] = type(e).__name__
        rec['refusals'] = refu
        rec['all_refused'] = all(v != 'ACCEPTED' for v in refu.values())
        rec['parity'] = all(v for k, v in rec.items() if k.endswith('_bitwise') or k.endswith('_decision_equal'))
        dep[f'{name}_S{s}'] = rec
shutil.rmtree(tmp)
out['deployment'] = dep
out['deployment_pass'] = all(r['parity'] and r['all_refused'] for r in dep.values())
out['n_rows'] = len(rows)
V.dump('V_DEPLOY.json', out)
print(json.dumps({'chain': chain, 'integrity_all': all(v['bitwise_all_roles'] for v in integ.values()), 'deployment_pass': out['deployment_pass'],
                  'prox': prox}, default=str))
