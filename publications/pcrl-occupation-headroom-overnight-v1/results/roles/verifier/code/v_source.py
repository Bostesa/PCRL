"""VER check 1: source admission and household-role integrity (label-free; own hashing and partition code).

1a every SOURCE_ADMISSION copy in the mirror re-hashed (recorded and expected hashes); the original source private file re-hashed
   and compared with the mirror copy;
1b the copies against the source's COMMITTED records, independently of the admission record: the source SOURCE_ADMISSION copies,
   FINALIST_LOCK package hashes, SERVICE_BANK_MANIFEST (bank files, release records, every wire hash recomputed), the H3 deployment
   package dependencies, RELEASE.json / RELEASE.npz consistency, the replication's EXTRACTION_SUMMARY export hashes and the
   frozen-hybrids FROZEN_BANK_MANIFEST; copies without an expected hash searched in the committed source records;
1c household disjointness and counts of the six roles (features only);
1d the registered DEV_GRID split (FIT / VALIDATION / SCORE, learning-curve subsets, overlaps, reader seeds) recomputed;
1e the cross-fitting folds (tag OHN1_crossfit, k = 5) recomputed and compared with every stored CROSSFIT.npz fold vector;
1f registration receipt and reader-grid code lock against the current files and git objects.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcommon as V  # noqa: E402

from ohn import common as C  # noqa: E402  (hash-checked label-free loading only)

SRC_PRIV = C.IHS_SRC_PRIV


def git(*a):
    r = subprocess.run(['git', '-C', str(V.ROOT), *a], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def check_mirror():
    adm = json.loads((V.RES / 'SOURCE_ADMISSION.json').read_text())
    rows, bad, bad_orig, missing_orig = {}, [], [], []
    for rel, rec in adm['copies'].items():
        p = V.MIRROR / rel
        h = V.file_sha256(p) if p.is_file() else None
        ok = h == rec['sha256'] and (rec['expected_sha256'] is None or h == rec['expected_sha256'])
        orig = SRC_PRIV / rel
        ho = V.file_sha256(orig) if orig.is_file() else None
        if ho is None:
            missing_orig.append(rel)
        elif ho != h:
            bad_orig.append(rel)
        if not ok:
            bad.append(rel)
        rows[rel] = h
    return {'n_copies': len(adm['copies']), 'n_copies_recorded_n': adm['n_copies'], 'hash_mismatch_vs_admission': bad,
            'mirror_differs_from_original_source_file': bad_orig, 'original_source_file_missing': missing_orig,
            'admission_status': adm['status'], 'admission_problems': adm['problems'],
            'n_with_expected_hash': sum(r['expected_sha256'] is not None for r in adm['copies'].values())}, rows, adm


def check_committed_records(rows, adm):
    out, problems = {}, []
    src_adm = json.loads((V.SRC_RESULTS / 'SOURCE_ADMISSION.json').read_text())['copies']
    n_ok = 0
    for rel, rec in src_adm.items():
        h = rows.get(f'admitted/{rel}')
        if h is None:
            problems.append(f'source copy not admitted: {rel}')
        elif h != rec['sha256']:
            problems.append(f'source copy hash differs: {rel}')
        else:
            n_ok += 1
    out['source_SOURCE_ADMISSION'] = {'n': len(src_adm), 'n_equal': n_ok}
    fl = json.loads((V.SRC_RESULTS / 'FINALIST_LOCK.json').read_text())['package_files_sha256']
    n = n_ok = 0
    for s, files in fl.items():
        for name, h in files.items():
            n += 1
            if rows.get(f'runs/H/H3_STRONG_LJDIR_PB/{s}/{name}') == h:
                n_ok += 1
            else:
                problems.append(f'FINALIST_LOCK file differs: {s}/{name}')
    out['source_FINALIST_LOCK_package_files'] = {'n': n, 'n_equal': n_ok}
    sbm = json.loads((V.SRC_RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    wires = {'n': 0, 'n_equal': 0}
    for s in ('S0', 'S1'):
        if rows.get(f'runs/FINAL/SERVICE_BANK_{s}.npz') != sbm['bank_files_sha256'][s]:
            problems.append(f'SERVICE_BANK {s} differs')
        per = sbm['per_seed'][s]
        for svc, cfg in (('SUCCESSOR', 'H3_STRONG_LJDIR_PB'), ('TW', 'HC_STRONG')):
            if rows.get(f'runs/H/{cfg}/{s}/RELEASE.npz') != per['release_records'][svc]:
                problems.append(f'release record {svc} {s} differs')
        with np.load(V.MIRROR / f'runs/FINAL/SERVICE_BANK_{s}.npz') as z:
            keys = set(z.files)
            for svc, tv in per['wire_sha256'].items():
                for t, rv in tv.items():
                    for role, h in rv.items():
                        wires['n'] += 1
                        cand = [k for k in keys if k.split('__')[0] == svc and role in k.split('__') and k.split('__')[-1] == t]
                        if len(cand) != 1:
                            problems.append(f'bank key not found {s} {svc} {role} {t}: {cand}')
                            continue
                        if V.wire_hash(z[cand[0]]) == h:
                            wires['n_equal'] += 1
                        else:
                            problems.append(f'bank wire differs {s} {svc} {role} {t}')
        # the stand-alone release arrays of the successor, the twin and the component control against the bank wires
        for svc, cfg in (('SUCCESSOR', 'H3_STRONG_LJDIR_PB'), ('TW', 'HC_STRONG'), ('CC', 'HC_TTREE_LJDIR_PB')):
            with np.load(V.MIRROR / f'runs/H/{cfg}/{s}/RELEASE.npz') as z:
                for t in V.TASKS:
                    for role in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT'):
                        wires['n'] += 1
                        if V.wire_hash(z[f'{role}__{t}']) == per['wire_sha256'][svc][t][role]:
                            wires['n_equal'] += 1
                        else:
                            problems.append(f'release array wire differs from manifest {s} {svc} {role} {t}')
    out['source_SERVICE_BANK_MANIFEST_wires'] = wires
    # H3 deployment package dependencies (the package records themselves are admitted copies)
    deps = {'n': 0, 'n_equal': 0}
    for s in (0, 1):
        meta = json.loads((V.MIRROR / f'runs/DEPLOY/H3_STRONG_LJDIR_PB_S{s}/PACKAGE.json').read_text())
        for rel, h in meta['depends_sha256'].items():
            key = rel if rel.startswith('runs/') else f'admitted/{rel}'
            deps['n'] += 1
            if rows.get(key) == h:
                deps['n_equal'] += 1
            else:
                problems.append(f'H3 S{s} package dependency differs or not admitted: {rel}')
    out['H3_package_dependencies'] = deps
    # every admitted package record's dependencies
    pk = {'n': 0, 'n_equal': 0, 'not_admitted': []}
    for rel in rows:
        if rel.startswith('runs/DEPLOY/') and rel.endswith('PACKAGE.json'):
            meta = json.loads((V.MIRROR / rel).read_text())
            for dep, h in meta.get('depends_sha256', {}).items():
                key = dep if dep.startswith('runs/') else f'admitted/{dep}'
                pk['n'] += 1
                if key not in rows:
                    pk['not_admitted'].append(f'{rel}: {dep}')
                elif rows[key] == h:
                    pk['n_equal'] += 1
                else:
                    problems.append(f'{rel}: dependency {dep} differs')
    out['all_admitted_package_dependencies'] = pk
    rel_ok = {'n': 0, 'n_equal': 0}
    for rel in rows:
        if rel.endswith('/RELEASE.json') and rel.startswith('runs/'):
            rec = json.loads((V.MIRROR / rel).read_text())
            rel_ok['n'] += 1
            if rec.get('release_sha256') == rows.get(rel.replace('RELEASE.json', 'RELEASE.npz')):
                rel_ok['n_equal'] += 1
            else:
                problems.append(f'release record does not match array: {rel}')
    out['release_records_vs_arrays'] = rel_ok
    ext = json.loads((V.ROOT / 'results/pcrl_acs_protected_tree_replication_v1/EXTRACTION_SUMMARY.json').read_text())['export_file_sha256']
    ex = {'n': 0, 'n_equal': 0}
    for rel, h in rows.items():
        if rel.startswith('admitted/exports/'):
            role, f = rel.split('/')[2], rel.split('/')[3].replace('.npz', '')
            ex['n'] += 1
            if ext[role][f] == h:
                ex['n_equal'] += 1
            else:
                problems.append(f'export differs from EXTRACTION_SUMMARY: {rel}')
    out['exports_vs_replication_EXTRACTION_SUMMARY'] = ex
    fb = json.loads((V.ROOT / 'results/pcrl_acs_frozen_hybrids_v1/FROZEN_BANK_MANIFEST.json').read_text())['bank_files_sha256']
    out['frozen_bank_vs_FROZEN_BANK_MANIFEST'] = {s: rows.get(f'admitted/bank/FROZEN_BANK_{s}.npz') == h for s, h in fb.items()}
    if not all(out['frozen_bank_vs_FROZEN_BANK_MANIFEST'].values()):
        problems.append('frozen bank differs from FROZEN_BANK_MANIFEST')
    # copies without an expected hash: is the hash recorded in any committed source record?
    texts = []
    for p in sorted(V.SRC_RESULTS.rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.csv', '.md', '.txt', '.jsonl'):
            texts.append(p.read_text(errors='ignore'))
    blob = '\n'.join(texts)
    no_exp = [rel for rel, rec in adm['copies'].items() if rec['expected_sha256'] is None]
    pinned_elsewhere = [rel for rel in no_exp if rows[rel] in blob]
    out['copies_without_expected_hash'] = {'n': len(no_exp), 'hash_found_in_committed_source_records': len(pinned_elsewhere),
                                           'not_found': sorted(set(no_exp) - set(pinned_elsewhere))}
    # the committed source records are unchanged since the source tip and clean
    rc, diff = git('diff', '--stat', C.IHS_TIP, 'HEAD', '--', f'results/{C.IHS_CAMPAIGN}/')
    rc2, st = git('status', '--porcelain', '--', f'results/{C.IHS_CAMPAIGN}/')
    out['source_records_unchanged_since_source_tip'] = (rc == 0 and diff == '' and st == '')
    if not out['source_records_unchanged_since_source_tip']:
        problems.append('source records changed or dirty')
    out['problems'] = problems
    return out


def check_roles():
    expected = {'MODEL_FIT': (20000, 13337), 'CALIBRATION': (2982, 1974), 'MODEL_SELECT': (8001, 5306), 'ATTACK_FIT': (8001, 5319),
                'ATTACK_VALIDATION': (3000, 1974), 'DEV_REPORT': (5943, 3955)}
    units, rowids, out = {}, {}, {}
    for role in C.ROLES6:
        F = C.load_role(role, ('features',))
        units[role] = np.asarray(F['unit']); rowids[role] = np.asarray(F['row_id'])
        out[role] = {'people': int(len(units[role])), 'households': int(len(set(units[role].tolist()))),
                     'unit_dtype': str(units[role].dtype), 'X_shape': list(F['X'].shape), 'X_dtype': str(F['X'].dtype),
                     'expected': list(expected[role]),
                     'counts_ok': (int(len(units[role])), int(len(set(units[role].tolist())))) == expected[role],
                     'row_id_unique': bool(len(set(rowids[role].tolist())) == len(rowids[role]))}
    overl = {}
    roles = list(C.ROLES6)
    for i, a in enumerate(roles):
        for b in roles[i + 1:]:
            overl[f'{a}|{b}'] = {'households': len(set(units[a].tolist()) & set(units[b].tolist())),
                                 'row_ids': len(set(rowids[a].tolist()) & set(rowids[b].tolist()))}
    return {'roles': out, 'pairwise_overlap': overl,
            'all_disjoint': all(v['households'] == 0 and v['row_ids'] == 0 for v in overl.values()),
            'all_counts_ok': all(v['counts_ok'] for v in out.values())}, units


def check_devgrid(unit_ms):
    reg = json.loads((V.RES / 'DEV_GRID.json').read_text())
    out, problems = {}, []
    roles_by_rep = {}
    for rep in ('R1', 'R2'):
        part = V.household_parts(unit_ms, 10, f'OHN1_devgrid_{rep}')
        masks = {'FIT': part <= 4, 'VALIDATION': (part >= 5) & (part <= 6), 'SCORE': part >= 7}
        roles_by_rep[rep] = masks
        r = {}
        for k, m in masks.items():
            r[k] = {'people': int(m.sum()), 'households': len(set(unit_ms[m].tolist())), 'household_set_sha256': V.household_set_sha(unit_ms[m])}
            regk = reg['repetitions'][rep][k]
            for f in ('people', 'households', 'household_set_sha256'):
                if r[k][f] != regk[f]:
                    problems.append(f'{rep} {k} {f}: {r[k][f]} vs registered {regk[f]}')
        hh = {k: set(unit_ms[m].tolist()) for k, m in masks.items()}
        r['disjoint'] = not (hh['FIT'] & hh['VALIDATION'] or hh['FIT'] & hh['SCORE'] or hh['VALIDATION'] & hh['SCORE'])
        r['covers_all'] = bool((masks['FIT'] | masks['VALIDATION'] | masks['SCORE']).all())
        lc = V.household_parts(unit_ms, 4, 'OHN1_devgrid_LC')
        r['learning_curve_fit_people'] = {'0.25': int((masks['FIT'] & (lc == 0)).sum()), '0.5': int((masks['FIT'] & (lc <= 1)).sum()),
                                          '1.0': int(masks['FIT'].sum())}
        if r['learning_curve_fit_people'] != reg['repetitions'][rep]['learning_curve_fit_people']:
            problems.append(f'{rep} learning-curve counts differ')
        if not r['disjoint'] or not reg['repetitions'][rep]['disjoint']:
            problems.append(f'{rep} not disjoint')
        out[rep] = r
    ov = {}
    for k in ('FIT', 'VALIDATION', 'SCORE'):
        ov[f'R1_vs_R2_{k}'] = len(set(unit_ms[roles_by_rep['R1'][k]].tolist()) & set(unit_ms[roles_by_rep['R2'][k]].tolist()))
    out['household_overlap_between_repetitions'] = ov
    if ov != reg['household_overlap_between_repetitions']:
        problems.append('overlap counts differ')
    base = None
    for line in (V.ROOT / 'task_conditioned_representation/attacks.py').read_text().splitlines():
        if line.replace(' ', '').startswith('MODEL_SEED_BASE='):
            base = int(line.split('=')[1].split('#')[0].strip())
    seeds = {f'S{s}k{k}': base + 300 + 10 * s + k for s in (0, 1) for k in (0, 1)}
    out['reader_seeds'] = seeds
    if seeds != reg['reader_seeds']:
        problems.append('reader seeds differ')
    out['cells'] = [f'S{s}{rep}k{k}' for s in (0, 1) for rep in ('R1', 'R2') for k in (0, 1)]
    if out['cells'] != reg['cells']:
        problems.append('cell list differs')
    out['problems'] = problems
    return out


def check_crossfit(unit_fit):
    folds = V.household_parts(unit_fit, 5, 'OHN1_crossfit')
    sizes = {int(f): int((folds == f).sum()) for f in range(5)}
    hh_sizes = {int(f): len(set(unit_fit[folds == f].tolist())) for f in range(5)}
    stored = {}
    for p in sorted(V.RUNS.glob('*/*/S*/CROSSFIT.npz')):
        with np.load(p) as z:
            fv = z['fold'] if 'fold' in z.files else None
        rel = str(p.relative_to(V.RUNS))
        if fv is None:
            stored[rel] = {'has_fold': False}
            continue
        # households wholly within one fold in the stored vector
        whole = True
        first = {}
        for g, f in zip(unit_fit.tolist(), np.asarray(fv).tolist()):
            if first.setdefault(g, f) != f:
                whole = False
                break
        stored[rel] = {'has_fold': True, 'equal_to_registered': bool(np.array_equal(np.asarray(fv), folds)),
                       'households_wholly_within_fold': whole, 'length': int(len(fv))}
    return {'fold_people': sizes, 'fold_households': hh_sizes, 'stored': stored,
            'all_stored_equal': all(v.get('equal_to_registered', False) for v in stored.values()),
            'n_stored': len(stored)}


def check_locks():
    out, problems = {}, []
    lock = json.loads((V.RES / 'DEVGRID_CODE_LOCK.json').read_text())
    cur = {f: V.file_sha256(V.ROOT / f) for f in lock['files_sha256']}
    out['code_lock_files_unchanged'] = cur == lock['files_sha256']
    out['code_lock_changed_files'] = [f for f in cur if cur[f] != lock['files_sha256'][f]]
    out['dev_grid_sha_equals_lock'] = V.file_sha256(V.RES / 'DEV_GRID.json') == lock['dev_grid_sha256']
    rec = json.loads((V.RES / 'RECEIPTS.json').read_text())['receipts'][0]
    files_now = {f: V.file_sha256(V.RES / f) for f in rec['files_sha256']}
    out['registration_files_unchanged'] = files_now == rec['files_sha256']
    blobs = {}
    for f, b in rec['published_blobs'].items():
        rc, hb = git('hash-object', str(V.RES / f))
        blobs[f] = (hb == b)
    out['registration_blobs_equal_published'] = blobs
    rc, tree = git('ls-tree', '-r', rec['receipt_commit'])
    out['receipt_commit_present_locally'] = rc == 0
    out['receipt_tree_contains_blobs'] = {f: (b in tree) for f, b in rec['published_blobs'].items()}
    rc, anc = git('merge-base', '--is-ancestor', rec['research_commit'], 'HEAD')
    out['research_commit_is_ancestor'] = rc == 0
    rc, ctime = git('show', '-s', '--format=%cI', rec['research_commit'])
    out['registration_commit_time'] = ctime
    rc, rtime = git('show', '-s', '--format=%cI', rec['receipt_commit'])
    out['receipt_commit_time'] = rtime
    out['receipt_verified_utc'] = rec['verified_utc']
    for f in ('PROTOCOL.md', 'DEV_GRID.json', 'EXPOSURE_TABLE.md', 'roles/statistical_reviewer/RULES.json',
              'roles/statistical_reviewer/SELECTION_RULE.md', 'DEVGRID_CODE_LOCK.json', 'SOURCE_ADMISSION.json'):
        rc, st = git('status', '--porcelain', '--', str(V.RES / f))
        rc2, lg = git('log', '-1', '--format=%H', '--', str(V.RES / f))
        out.setdefault('committed_clean', {})[f] = bool(lg) and not st
    if not all(out['committed_clean'].values()):
        problems.append('a registration file is not committed or is dirty')
    if not out['code_lock_files_unchanged']:
        problems.append('reader-grid code changed after the lock')
    out['problems'] = problems
    return out


def main():
    rec = {'check': 'VER_SOURCE', 'utc': V.utc(), 'unit_id': V.unit_id()}
    m, rows, adm = check_mirror()
    rec['1a_mirror'] = m
    rec['1b_committed_records'] = check_committed_records(rows, adm)
    rec['1c_roles'], units = check_roles()
    rec['1d_devgrid'] = check_devgrid(units['MODEL_SELECT'])
    rec['1e_crossfit'] = check_crossfit(units['MODEL_FIT'])
    rec['1f_locks'] = check_locks()
    ok = (not m['hash_mismatch_vs_admission'] and not m['mirror_differs_from_original_source_file'] and not m['original_source_file_missing']
          and not rec['1b_committed_records']['problems'] and rec['1c_roles']['all_disjoint'] and rec['1c_roles']['all_counts_ok']
          and not rec['1d_devgrid']['problems'] and rec['1e_crossfit']['all_stored_equal'] and not rec['1f_locks']['problems'])
    rec['status'] = 'PASS' if ok else 'FAIL_OR_ATTENTION'
    V.write_record('VER_SOURCE', rec)
    print(json.dumps({'status': rec['status'], 'mirror': {k: (v if not isinstance(v, list) else len(v)) for k, v in m.items()},
                      'b_problems': rec['1b_committed_records']['problems'][:10], 'roles_ok': [rec['1c_roles']['all_disjoint'], rec['1c_roles']['all_counts_ok']],
                      'devgrid_problems': rec['1d_devgrid']['problems'], 'crossfit': {k: v for k, v in rec['1e_crossfit'].items() if k != 'stored'},
                      'locks': rec['1f_locks']}, default=str))


if __name__ == '__main__':
    main()
