"""Verifier check 1: source admission and household-role integrity (own code).

1a  every admitted copy re-hashed and compared with the hash its OWN manifest records (EXTRACTION_SUMMARY, source
    TRAINING_COMPLETE_LOCK -> FIT_SEED files_sha256, FROZEN_BANK_MANIFEST, MODEL_MANIFEST); the 12 source manifest checks
    re-done on the source private tree; unexpected files under source/ listed.
1b  lock chain: ANALYSIS_LOCK (protocol, inventory, admission, reviews, code), TRAJECTORY_COMPLETE_LOCK, SELECTION_LOCK.
1c  git facts: source tip, publication commit, public-package comparison (with the CRLF explanation tested).
1d  household (unit) and row disjointness between all six admitted roles; counts against the public extraction summary.
1e  routing: own oblivious-tree teacher evaluation (fold average), own KL-nearest-centroid routing on the four output roles,
    compared with the admitted ROUTING arrays; the 31 historical arm outputs rebuilt from (own routing, historical tables) and
    compared bitwise with ARM_OUTPUTS; centroids = pooled out-of-fold cell means.
1f  sealed assessment: no export anywhere in the admitted trees; the campaign loader refuses it and refuses DEV_REPORT truth
    fields (objects under test: scm_um.common.load_role and scm_bench.common.load_role, imported for that purpose only).
No label is read by this check.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

t0 = time.process_time()
out = {'check': 1, 'title': 'source admission and household-role integrity', 'discrepancies': [], 'notes': []}
adm = V.jload(V.RES / 'SOURCE_ADMISSION.json')

# ---------------------------------------------------------------- 1a copies vs their own manifests
pub = V.public_export_sha()
tcl = V.jload(V.SB_RES / 'TRAINING_COMPLETE_LOCK.json')
fbm = V.jload(V.HYB_RES / 'FROZEN_BANK_MANIFEST.json')['bank_files_sha256']
mm = V.jload(V.REPL_RES / 'MODEL_MANIFEST.json')['deployment_packages']['T_TREE_S0']['package_json_sha256']
fitseed = {}
for s in V.SEEDS:
    p = V.SRC / 'fit' / f'FIT_{s}' / 'FIT_SEED.json'
    fitseed[s] = V.jload(p)


def expected(rel):
    parts = rel.split('/')
    if parts[0] == 'exports':
        return pub[parts[1]][parts[2][:-4]], 'EXTRACTION_SUMMARY'
    if parts[0] == 'bank':
        return fbm['S' + parts[1].split('_S')[1][0]], 'FROZEN_BANK_MANIFEST'
    if parts[0] == 'encoder':
        return mm, 'MODEL_MANIFEST'
    if parts[0] == 'fit':
        s = int(parts[1].split('_')[1]); sub = '/'.join(parts[2:])
        if sub == 'FIT_SEED.json':
            return tcl['fit_seed_sha256'][f'S{s}'], 'source TRAINING_COMPLETE_LOCK'
        return fitseed[s]['files_sha256'][sub], 'FIT_SEED files_sha256'
    if parts[0] == 'routing':
        return None, 'computed (checked in 1e)'
    return 'UNKNOWN', 'UNKNOWN'


copies = {}
for rel, rec in sorted(adm['copies'].items()):
    p = V.SRC / rel
    got = V.sha_file(p) if p.is_file() else 'MISSING'
    exp, basis = expected(rel)
    ok_rec = got == rec['sha256']
    ok_man = (exp is None) or (got == exp)
    ok_bytes = p.is_file() and p.stat().st_size == rec['bytes']
    copies[rel] = {'file_equals_recorded': ok_rec, 'file_equals_manifest': ok_man, 'manifest': basis, 'bytes_equal': ok_bytes}
fit_seed_self = {s: V.sha_file(V.SRC / 'fit' / f'FIT_{s}' / 'FIT_SEED.json') == tcl['fit_seed_sha256'][f'S{s}'] for s in V.SEEDS}
on_disk = sorted(str(p.relative_to(V.SRC)) for p in V.SRC.rglob('*') if p.is_file())
unexpected = sorted(set(on_disk) - set(adm['copies']))
missing = sorted(set(adm['copies']) - set(on_disk))
out['1a_copies'] = {'n_recorded': len(adm['copies']), 'n_on_disk': len(on_disk), 'all_equal_recorded': all(v['file_equals_recorded'] for v in copies.values()),
                    'all_equal_manifest': all(v['file_equals_manifest'] for v in copies.values()), 'all_bytes_equal': all(v['bytes_equal'] for v in copies.values()),
                    'fit_seed_json_equals_TRAINING_COMPLETE_LOCK': fit_seed_self, 'unexpected_files_under_source': unexpected, 'missing_files': missing,
                    'n_by_kind': {k: sum(r.startswith(k) for r in adm['copies']) for k in ('exports', 'fit', 'bank', 'encoder', 'routing')},
                    'failures': {k: v for k, v in copies.items() if not (v['file_equals_recorded'] and v['file_equals_manifest'] and v['bytes_equal'])}}
# the 12 manifest checks on the source private tree
sel_src = V.jload(V.SB_RES / 'SELECTION_LOCK.json'); sbm = V.jload(V.SB_RES / 'SERVICE_BANK_MANIFEST.json'); rl = V.jload(V.SB_RES / 'REPORTING_LOCK.json')
man = {}
for s in V.SEEDS:
    fd = V.SB_PRIV / 'runs' / f'FIT_{s}'
    man[f'FIT_{s}/FIT_SEED.json'] = V.sha_file(fd / 'FIT_SEED.json') == tcl['fit_seed_sha256'][f'S{s}']
    man[f'FIT_{s}/ARM_OUTPUTS.npz'] = V.sha_file(fd / 'ARM_OUTPUTS.npz') == tcl['arm_outputs_sha256'][f'S{s}']
    man[f'FIT_{s}/ARM_OUTPUTS.npz (selection lock)'] = V.sha_file(fd / 'ARM_OUTPUTS.npz') == sel_src['arm_output_files_sha256'][f'runs/FIT_{s}/ARM_OUTPUTS.npz']
    man[f'SERVICE_BANK_S{s}.npz'] = V.sha_file(V.SB_PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz') == sbm['bank_files_sha256'][f'S{s}']
    rd = V.SB_PRIV / 'runs' / f'READERS_{s}'
    man[f'READERS_{s}/DEV_REPORT_SCORES.npz'] = V.sha_file(rd / 'DEV_REPORT_SCORES.npz') == rl['readers'][f'S{s}']['dev_report_scores_sha256']
    man[f'READERS_{s}/READERS_SEED.json'] = V.sha_file(rd / 'READERS_SEED.json') == rl['readers'][f'S{s}']['readers_seed_sha256']
out['1a_manifest_checks'] = {'n': len(man), 'all_match': all(man.values()), 'detail': man,
                             'recorded_n': len(adm['manifest_checks']), 'recorded_all_match': all(v['match'] for v in adm['manifest_checks'].values()),
                             'same_names_as_recorded': sorted(man) == sorted(adm['manifest_checks'])}
# admitted copy of every fit artifact = source private file
same_as_source = {}
for s in V.SEEDS:
    for rel in list(fitseed[s]['files_sha256']) + ['FIT_SEED.json']:
        same_as_source[f'FIT_{s}/{rel}'] = V.sha_file(V.SB_PRIV / 'runs' / f'FIT_{s}' / rel) == V.sha_file(V.SRC / 'fit' / f'FIT_{s}' / rel)
out['1a_fit_copies_equal_source_private_files'] = all(same_as_source.values())

# ---------------------------------------------------------------- 1b lock chain
al = V.jload(V.RES / 'ANALYSIS_LOCK.json')
chain = {'ANALYSIS_LOCK.protocol_sha256 == PROTOCOL.md': al['protocol_sha256'] == V.sha_file(V.RES / 'PROTOCOL.md'),
         'ANALYSIS_LOCK.unit_inventory_sha256 == UNIT_INVENTORY.json': al['unit_inventory_sha256'] == V.sha_file(V.RES / 'UNIT_INVENTORY.json'),
         'ANALYSIS_LOCK.source_admission_sha256 == SOURCE_ADMISSION.json': al['source_admission_sha256'] == V.sha_file(V.RES / 'SOURCE_ADMISSION.json')}
for f, h in al['prefit_reviews_sha256'].items():
    chain[f'ANALYSIS_LOCK.prefit_reviews {f}'] = h == V.sha_file(V.ROOT / f)
code_changed = sorted(f for f, h in al['code_sha256'].items() if V.sha_file(V.ROOT / f) != h)
tl = V.jload(V.RES / 'TRAJECTORY_COMPLETE_LOCK.json')
chain['TRAJECTORY_COMPLETE_LOCK.analysis_lock_sha256 == ANALYSIS_LOCK.json'] = tl['analysis_lock_sha256'] == V.sha_file(V.RES / 'ANALYSIS_LOCK.json')
for s in V.SEEDS:
    for f, h in tl['files_sha256'][f'S{s}'].items():
        chain[f'TRAJECTORY_COMPLETE_LOCK S{s} {f}'] = h == V.sha_file(V.PRIV / 'runs' / f'TRAJ_{s}' / f)
    ts = V.jload(V.PRIV / 'runs' / f'TRAJ_{s}' / 'TRAJ_SEED.json')
    chain[f'TRAJ_SEED S{s} analysis_lock_sha256'] = ts['analysis_lock_sha256'] == V.sha_file(V.RES / 'ANALYSIS_LOCK.json')
    for f, h in ts['files_sha256'].items():
        chain[f'TRAJ_SEED S{s} files_sha256 {f}'] = h == V.sha_file(V.PRIV / 'runs' / f'TRAJ_{s}' / f)
sl = V.jload(V.RES / 'SELECTION_LOCK.json')
chain['SELECTION_LOCK.trajectory_lock_sha256 == TRAJECTORY_COMPLETE_LOCK.json'] = sl['trajectory_lock_sha256'] == V.sha_file(V.RES / 'TRAJECTORY_COMPLETE_LOCK.json')
rlk = V.RES / 'REPORTING_LOCK.json'
code_in_reporting_lock = {}
if rlk.is_file():
    rj = V.jload(rlk)
    flat = {}

    def walk(o, pre=''):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f'{pre}{k}/')
        elif isinstance(o, str) and len(o) == 64:
            flat[pre.rstrip('/')] = o
    walk(rj)
    for f in code_changed:
        hits = [k for k, v in flat.items() if k.endswith(f) and v == V.sha_file(V.ROOT / f)]
        code_in_reporting_lock[f] = hits
out['1b_lock_chain'] = {'all_pass': all(chain.values()), 'detail': chain, 'code_files_in_ANALYSIS_LOCK': len(al['code_sha256']),
                        'code_changed_since_ANALYSIS_LOCK': code_changed, 'changed_code_hash_found_in_REPORTING_LOCK': code_in_reporting_lock}


# ---------------------------------------------------------------- 1c git facts
def git(*a):
    r = subprocess.run(['git', '-C', str(V.ROOT), *a], capture_output=True, text=True)
    return r.stdout.strip()


src_tip = git('rev-parse', 'research/pcrl-scm-engineering-benchmark-v1')
pub_ok = git('cat-file', '-t', '6fc06021d3ce42e1b281e92702b54bfe1cf384c8')
pub_time = git('log', '-1', '--format=%cI', '6fc06021d3ce42e1b281e92702b54bfe1cf384c8')
pp = adm['source']['public_package']
differs = {f: v for f, v in pp['detail'].items() if v.startswith('differs')}
crlf = {}
for f in differs:
    loc = (V.SB_RES / f).read_bytes()
    pubb = subprocess.run(['git', '-C', str(V.ROOT), 'show', f'6fc06021d3ce42e1b281e92702b54bfe1cf384c8:publications/pcrl-scm-engineering-benchmark-v1/results/{f}'],
                          capture_output=True).stdout
    last = git('log', '-1', '--format=%h %cI', 'research/pcrl-scm-engineering-benchmark-v1', '--', str((V.SB_RES / f).relative_to(V.ROOT)))
    from datetime import datetime
    after = (datetime.fromisoformat(last.split(' ')[1]) > datetime.fromisoformat(pub_time)) if last else None
    crlf[f] = {'identical_after_CRLF_to_LF': pubb.replace(b'\r\n', b'\n') == loc.replace(b'\r\n', b'\n'),
               'local_has_CRLF': b'\r\n' in loc, 'public_has_CRLF': b'\r\n' in pubb,
               'last_research_commit_touching_file': last, 'recorded_explanation': differs[f],
               'last_commit_after_publication': after}
out['1c_git'] = {'source_tip': src_tip, 'source_tip_matches_recorded': src_tip == adm['source']['recorded_tip'],
                 'publication_commit_exists': pub_ok == 'commit', 'publication_commit_time': pub_time,
                 'public_package_status_counts_recorded': pp['status_counts'], 'differs_files': crlf}
for f, c in crlf.items():
    if c['identical_after_CRLF_to_LF'] and c['last_commit_after_publication'] is False:
        out['discrepancies'].append({'id': 'C1-ADM-EXPLANATION', 'file': f, 'text':
                                     f"SOURCE_ADMISSION.json records '{c['recorded_explanation']}', but the last research commit touching the file "
                                     f"({c['last_research_commit_touching_file']}) predates the publication commit ({pub_time}); the file is identical to the "
                                     "published one after CRLF->LF normalisation (PROTOCOL.md section 1 states this correctly). Wording only."})

# ---------------------------------------------------------------- 1d household / row disjointness and counts
ex = V.jload(V.REPL_RES / 'EXTRACTION_SUMMARY.json')['roles']
units, rows = {}, {}
for r in V.ROLES6:
    f = V.export(r, 'features')
    units[r] = np.unique(f['unit']); rows[r] = np.unique(f['row_id'])
    assert len(rows[r]) == len(f['row_id']), 'duplicate row ids'
pairs = {}
for i, a in enumerate(V.ROLES6):
    for b in V.ROLES6[i + 1:]:
        pairs[f'{a}|{b}'] = {'shared_units': int(len(np.intersect1d(units[a], units[b]))), 'shared_row_ids': int(len(np.intersect1d(rows[a], rows[b])))}
counts = {r: {'persons': int(len(rows[r])), 'households': int(len(units[r])), 'public_persons': ex[r]['selected_persons'],
              'public_households': ex[r]['selected_households']} for r in V.ROLES6}
# label exports pair with the feature export (row_id and unit identical), for the label files this campaign admits
pairing = {}
for r in V.ROLES6:
    f = V.export(r, 'features')
    for fld in ('income', 'occupation', 'sex'):
        if r == 'DEV_REPORT':
            continue
        p = V.SRC / 'exports' / r / f'{fld}.npz'
        if p.is_file():
            with np.load(p) as z:          # only the id columns are compared here
                pairing[f'{r}/{fld}'] = bool(np.array_equal(z['row_id'], f['row_id']) and np.array_equal(z['unit'], f['unit']))
out['1d_roles'] = {'pairwise': pairs, 'all_disjoint_units': all(v['shared_units'] == 0 for v in pairs.values()),
                   'all_disjoint_rows': all(v['shared_row_ids'] == 0 for v in pairs.values()), 'counts': counts,
                   'counts_match_public': all(c['persons'] == c['public_persons'] and c['households'] == c['public_households'] for c in counts.values()),
                   'DEV_REPORT_G_3955_n_5943': counts['DEV_REPORT']['households'] == 3955 and counts['DEV_REPORT']['persons'] == 5943,
                   'label_exports_pair_with_features': pairing}

# ---------------------------------------------------------------- 1e routing by own teacher evaluation
def own_teacher(payload, X):
    feats = np.asarray(payload['split_features'], dtype=np.int64)
    borders = np.asarray(payload['split_borders'], dtype=np.float32)
    leaves = np.asarray(payload['leaf_values'], dtype=np.float64)
    T, depth = feats.shape
    Xf = np.ascontiguousarray(X, dtype=np.float32)
    raw = np.zeros((len(X), leaves.shape[2]))
    for t in range(T):                                  # sequential sum over trees (own order)
        leaf = np.zeros(len(X), dtype=np.int64)
        for i in range(depth):
            leaf += (Xf[:, feats[t, i]] > borders[t, i]).astype(np.int64) << i
        raw += leaves[t, leaf, :]
    raw = float(payload['scale']) * raw + np.asarray(payload['bias'], dtype=np.float64)
    if payload['task'] == 'income':
        d = raw[:, 0]; lg = np.column_stack((-d / 2.0, d / 2.0))
    else:
        lg = raw - raw.mean(axis=1, keepdims=True)
    lg = lg - lg.max(axis=1, keepdims=True); e = np.exp(lg)
    return e / e.sum(axis=1, keepdims=True)


def own_kl_route(t1, t2, c1, c2):
    l1 = np.log(np.maximum(c1, 1e-300)); l2 = np.log(np.maximum(c2, 1e-300))
    best = np.empty(len(t1), dtype=np.int64); gap = np.empty(len(t1))
    for a in range(0, len(t1), 1000):
        sc = (t1[a:a + 1000, None, :] * l1[None, :, :]).sum(-1) + (t2[a:a + 1000, None, :] * l2[None, :, :]).sum(-1)
        best[a:a + 1000] = np.argmax(sc, axis=1)
        srt = np.sort(sc, axis=1)
        gap[a:a + 1000] = srt[:, -1] - srt[:, -2]
    return best, gap


route_res = {}
for s in V.SEEDS:
    P = V.npz(V.SRC / 'fit' / f'FIT_{s}' / 'PARTITION.npz')
    Tb = V.npz(V.SRC / 'fit' / f'FIT_{s}' / 'ARM_TABLES.npz')
    AO = V.npz(V.SRC / 'fit' / f'FIT_{s}' / 'ARM_OUTPUTS.npz')
    RT = V.npz(V.SRC / 'routing' / f'ROUTING_S{s}.npz')
    k = P['cent1'].shape[0]
    n = np.bincount(P['fine_assign_MODEL_FIT'], minlength=k).astype(np.float64)
    s1 = np.zeros((k, 2)); s2 = np.zeros((k, 5))
    np.add.at(s1, P['fine_assign_MODEL_FIT'], P['oof_income']); np.add.at(s2, P['fine_assign_MODEL_FIT'], P['oof_occupation'])
    cent_exact = bool(np.array_equal(s1 / n[:, None], P['cent1']) and np.array_equal(s2 / n[:, None], P['cent2']))
    cent_close = float(max(np.abs(s1 / n[:, None] - P['cent1']).max(), np.abs(s2 / n[:, None] - P['cent2']).max()))
    teach = {t: [V.jload(V.SRC / 'fit' / f'FIT_{s}' / 'TEACHERS' / f'fold{f}_{t}.json') for f in range(5)] for t in V.TASKS}
    own_init, gaps, rid_ok, src_prob_diff = {}, {}, {}, {}
    for r in V.OUT_ROLES:
        fx = V.export(r, 'features')
        rid_ok[r] = bool(np.array_equal(RT[f'row_id__{r}'], fx['row_id']) and np.array_equal(RT[f'unit__{r}'], fx['unit'])
                         and np.array_equal(AO[f'row_id__{r}'], fx['row_id']) and np.array_equal(AO[f'unit__{r}'], fx['unit']))
        names = tuple(map(str, fx['feature_names']))
        assert all(tuple(p['feature_names']) == names for t in V.TASKS for p in teach[t])
        prob = {}
        for t in V.TASKS:
            acc = np.zeros((len(fx['X']), V.KCLS[t]))
            for f in range(5):
                acc = acc + own_teacher(teach[t][f], fx['X']) / 5.0
            prob[t] = acc
        own_init[r], gaps[r] = own_kl_route(prob['income'], prob['occupation'], P['cent1'], P['cent2'])
        # comparison with the source evaluator (object under test, for the probability difference only)
        try:
            from acs_replication import trees
            from scm_bench.teachers import teacher_probs
            d = 0.0
            for t in V.TASKS:
                acc = 0.0
                for f in range(5):
                    acc = acc + teacher_probs(trees.TreeTeacher(teach[t][f]), fx['X'], names) / 5.0
                d = max(d, float(np.abs(acc - prob[t]).max()))
            src_prob_diff[r] = d
        except Exception as e:      # pragma: no cover
            src_prob_diff[r] = f'not compared: {e!r}'
    routing_equal = {r: int((own_init[r] != RT[f'init__{r}']).sum()) for r in V.OUT_ROLES}
    min_gap = {r: float(gaps[r].min()) for r in V.OUT_ROLES}
    arms = sorted({kk.split('__')[0] for kk in Tb})
    repro = {}
    for arm in arms:
        v1, v2, ci, gi = (Tb[f'{arm}__{x}'] for x in ('v1', 'v2', 'cell_of_initial', 'group_of_initial'))
        okk = True
        for r in V.OUT_ROLES:
            init = own_init[r]
            okk &= bool(np.array_equal(v1[gi[init]], AO[f'{arm}__{r}__income']) and np.array_equal(v2[ci[init]], AO[f'{arm}__{r}__occupation']))
        repro[arm] = okk
    route_res[f'S{s}'] = {'occupied_fine_cells': int(k), 'all_cells_occupied_on_MODEL_FIT': bool((n > 0).all()),
                          'centroids_equal_pooled_oof_cell_means_bitwise': cent_exact, 'centroid_max_abs_diff': cent_close,
                          'row_ids_align': rid_ok, 'own_routing_mismatches_vs_admitted': routing_equal,
                          'min_score_gap_best_vs_second': min_gap, 'max_abs_prob_diff_own_vs_source_evaluator': src_prob_diff,
                          'historical_arms': len(arms), 'arms_reproduced_bitwise_4_roles_2_tasks': int(sum(repro.values())),
                          'arms_not_reproduced': [a for a, v in repro.items() if not v]}
out['1e_routing'] = route_res

# ---------------------------------------------------------------- 1f sealed assessment
sealed_paths = []
for base in (V.PRIV, V.SB_PRIV / 'source'):
    for p in base.rglob('*'):
        if 'SEALED' in p.name.upper():
            sealed_paths.append(str(p.relative_to(V.ROOT)))
refusals = {}
sys.path.insert(0, str(V.ROOT))
from scm_um import common as UMC          # object under test
from scm_bench import common as SBC       # object under test
for label, fn, args in (('scm_um.load_role(SEALED_ASSESSMENT)', UMC.load_role, ('SEALED_ASSESSMENT',)),
                        ('scm_bench.load_role(SEALED_ASSESSMENT)', SBC.load_role, ('SEALED_ASSESSMENT',)),
                        ('scm_um.load_role(DEV_REPORT, sex)', UMC.load_role, ('DEV_REPORT', ('features', 'sex'))),
                        ('scm_um.load_role(DEV_REPORT, income)', UMC.load_role, ('DEV_REPORT', ('features', 'income'))),
                        ('scm_um.load_role(DEV_REPORT, occupation)', UMC.load_role, ('DEV_REPORT', ('features', 'occupation'))),
                        ('scm_um.load_role(ATTACK_FIT, income)', UMC.load_role, ('ATTACK_FIT', ('features', 'income'))),
                        ('scm_um.load_role(ATTACK_VALIDATION, occupation)', UMC.load_role, ('ATTACK_VALIDATION', ('features', 'occupation'))),
                        ('scm_um.load_role(CALIBRATION, sex)', UMC.load_role, ('CALIBRATION', ('features', 'sex')))):
    try:
        fn(*args); refusals[label] = 'NOT_REFUSED'
    except PermissionError as e:
        refusals[label] = 'refused: ' + str(e)[:120]
out['1f_sealed'] = {'sealed_named_paths_in_admitted_trees': sealed_paths, 'public_export_roles': sorted(pub),
                    'sealed_export_in_public_summary': any('SEALED' in r for r in pub),
                    'extraction_summary_sealed_record': V.jload(V.REPL_RES / 'EXTRACTION_SUMMARY.json')['sealed_assessment'],
                    'refusals': refusals, 'all_refused': all(v.startswith('refused') for v in refusals.values())}

# ---------------------------------------------------------------- verdict
ok = (out['1a_copies']['all_equal_recorded'] and out['1a_copies']['all_equal_manifest'] and out['1a_copies']['all_bytes_equal']
      and not out['1a_copies']['unexpected_files_under_source'] and all(fit_seed_self.values()) and out['1a_manifest_checks']['all_match']
      and out['1a_fit_copies_equal_source_private_files'] and out['1c_git']['source_tip_matches_recorded'] and out['1c_git']['publication_commit_exists']
      and out['1d_roles']['all_disjoint_units'] and out['1d_roles']['all_disjoint_rows'] and out['1d_roles']['counts_match_public']
      and all(v['centroids_equal_pooled_oof_cell_means_bitwise'] and sum(v['own_routing_mismatches_vs_admitted'].values()) == 0
              and v['arms_reproduced_bitwise_4_roles_2_tasks'] == 31 and all(v['row_ids_align'].values()) for v in route_res.values())
      and not sealed_paths and out['1f_sealed']['all_refused'])
out['status'] = 'PASS' if ok else 'FAIL'
out['cpu_s'] = round(time.process_time() - t0, 2)
out['code_under_test_imported'] = ['scm_um.common.load_role', 'scm_bench.common.load_role (refusal tests only)',
                                   'acs_replication.trees + scm_bench.teachers.teacher_probs (probability difference only; routing is own code)',
                                   'mechanism_design not used']
V.write_json('check1_admission.json', out)
print(json.dumps({'status': out['status'], 'copies': {k: out['1a_copies'][k] for k in ('n_recorded', 'all_equal_recorded', 'all_equal_manifest', 'unexpected_files_under_source')},
                  'manifest': out['1a_manifest_checks']['all_match'], 'chain': out['1b_lock_chain']['all_pass'],
                  'code_changed': out['1b_lock_chain']['code_changed_since_ANALYSIS_LOCK'],
                  'roles_disjoint': (out['1d_roles']['all_disjoint_units'], out['1d_roles']['all_disjoint_rows'], out['1d_roles']['counts_match_public']),
                  'routing': {k: (v['own_routing_mismatches_vs_admitted'], v['arms_reproduced_bitwise_4_roles_2_tasks'], v['centroids_equal_pooled_oof_cell_means_bitwise'],
                                  v['max_abs_prob_diff_own_vs_source_evaluator']) for k, v in route_res.items()},
                  'sealed': (sealed_paths, out['1f_sealed']['all_refused']), 'discrepancies': len(out['discrepancies']), 'cpu_s': out['cpu_s']}, indent=1, default=str))
