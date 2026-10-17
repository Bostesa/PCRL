"""Unit UM_ADMIT: admit the SCM engineering benchmark's artifacts by hash and copy them into this campaign's private tree.

1. Git: the source research branch tip (recorded 2738cc30b), the public package commit (6fc06021d) on the GitHub remote, and
   every file of the public results package compared with the sanitised local file.
2. Private artifacts checked against their own manifests: FIT_SEED files_sha256 and TRAINING_COMPLETE_LOCK (teachers, out-of-fold
   predictions, fine partition, historical arm tables and outputs), SELECTION_LOCK, SERVICE_BANK_MANIFEST, REPORTING_LOCK reader
   scores, and the source RESTORE_RECEIPT (role exports, frozen comparator bank).
3. Copies (append-only, hash-checked): role exports (admitted fields only; no DEV_REPORT truth), the frozen bank, the fit
   artifacts, and the source encoder package record.
4. Runtime routing by hash: the fold-average of the admitted teachers on each output role, KL-nearest admitted fine centroid;
   the historical C0 and all 31 historical arm outputs must be reproduced bitwise from (routing, historical tables).
No label is read: MODEL_FIT, MODEL_SELECT, ATTACK and DEV_REPORT labels and SEX are untouched by this unit.
"""
from __future__ import annotations
import hashlib
import json
import platform
import shutil
import subprocess
import sys

import numpy as np

from scm_bench import common as SB
from scm_bench import engine as E
from scm_bench.teachers import teacher_probs
from acs_replication import trees
from . import common as C

SOURCE_BRANCH = 'research/pcrl-scm-engineering-benchmark-v1'
SOURCE_TIP = '2738cc30bd46ab7801c0de1686135213996ba55d'
PUBLIC_COMMIT = '6fc06021d3ce42e1b281e92702b54bfe1cf384c8'
PUBLIC_DIR = 'publications/pcrl-scm-engineering-benchmark-v1/results'


def git(*a, binary=False):
    r = subprocess.run(['git', '-C', str(C.ROOT), *a], capture_output=True, text=not binary)
    return r.stdout if binary else r.stdout.strip()


def public_package_check():
    from mechanism_design.publish_build import sanitise
    files = git('ls-tree', '-r', '--name-only', PUBLIC_COMMIT, PUBLIC_DIR).splitlines()
    out = {}
    for f in files:
        rel = f[len(PUBLIC_DIR) + 1:]
        pub = git('show', f'{PUBLIC_COMMIT}:{f}', binary=True)
        local = C.SB_RESULTS / rel
        if not local.is_file():
            out[rel] = 'absent_locally'; continue
        raw = local.read_bytes()
        try:
            cand = sanitise(raw.decode()).encode()
        except UnicodeDecodeError:
            cand = raw
        out[rel] = 'identical_after_sanitise' if hashlib.sha256(cand).hexdigest() == hashlib.sha256(pub).hexdigest() else \
            'differs (changed on the research branch after the publication build: ' + git('log', '-1', '--format=%h', f'{PUBLIC_COMMIT}..{SOURCE_TIP}', '--', str(local.relative_to(C.ROOT))) + ')'
    return {'files': len(files), 'status_counts': {s: sum(v.startswith(s) for v in out.values()) for s in ('identical_after_sanitise', 'differs', 'absent_locally')},
            'detail': out}


def main():
    C.PRIV.mkdir(parents=True, exist_ok=True, mode=0o700)
    tip = git('rev-parse', SOURCE_BRANCH)
    remote = git('ls-remote', 'origin', 'refs/heads/publication/pcrl-scm-engineering-benchmark-v1', 'refs/heads/main').splitlines()
    rec = {'schema': 'SCMUM1-source-admission-v1', 'utc': C.now(), 'campaign': C.CAMPAIGN,
           'source': {'research_branch': SOURCE_BRANCH, 'recorded_tip': SOURCE_TIP, 'actual_tip': tip, 'tip_matches': tip == SOURCE_TIP,
                      'public_commit': PUBLIC_COMMIT, 'remote_refs': dict(l.split('\t')[::-1] for l in remote if l.strip()),
                      'public_package': public_package_check()},
           'manifest_checks': {}, 'copies': {}, 'routing': {}, 'not_copied': [
               'DEV_REPORT income/occupation/SEX truth (re-read in place only after this campaign\'s committed REPORTING_LOCK)',
               'any SEALED_ASSESSMENT record (no export exists)', 'source reader models and scores (fresh readers are fitted)'],
           'not_persisted_by_source': 'the five fold SEX-propensity models were never written to disk by the source fit; their '
                                      'out-of-fold MODEL_FIT outputs (PARTITION.npz oof_sex_propensity), which are all the merge '
                                      'engine uses, are admitted by hash instead'}
    rec['source']['publication_remote_matches'] = rec['source']['remote_refs'].get('refs/heads/publication/pcrl-scm-engineering-benchmark-v1') == PUBLIC_COMMIT
    checks = rec['manifest_checks']
    tl = json.loads((C.SB_RESULTS / 'TRAINING_COMPLETE_LOCK.json').read_text())
    sel = json.loads((C.SB_RESULTS / 'SELECTION_LOCK.json').read_text())
    sbm = json.loads((C.SB_RESULTS / 'SERVICE_BANK_MANIFEST.json').read_text())
    rl = json.loads((C.SB_RESULTS / 'REPORTING_LOCK.json').read_text())
    rr = json.loads((C.SB_PRIV / 'RESTORE_RECEIPT.json').read_text())

    def check(name, path, expected, basis):
        got = C.sha(path) if path.is_file() else 'MISSING'
        checks[name] = {'sha256': got, 'expected': expected, 'basis': basis, 'match': got == expected}
        if got != expected:
            raise PermissionError(f'admission hash check failed: {name}')

    def copy(src, rel, expected, basis):
        dst = C.SRCP / rel
        dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if C.sha(src) != expected:
            raise PermissionError(f'source hash differs from {basis}: {rel}')
        if dst.exists():
            if C.sha(dst) != expected:
                raise PermissionError(f'existing admitted copy differs: {rel}')
        else:
            shutil.copyfile(src, dst); dst.chmod(0o600)
        if C.sha(dst) != expected:
            raise PermissionError(f'copy hash differs: {rel}')
        rec['copies'][rel] = {'sha256': expected, 'basis': basis, 'bytes': dst.stat().st_size}

    for s in C.SEEDS:
        fd = C.SB_PRIV / 'runs' / f'FIT_{s}'
        check(f'FIT_{s}/FIT_SEED.json', fd / 'FIT_SEED.json', tl['fit_seed_sha256'][f'S{s}'], 'source TRAINING_COMPLETE_LOCK fit_seed_sha256')
        check(f'FIT_{s}/ARM_OUTPUTS.npz', fd / 'ARM_OUTPUTS.npz', tl['arm_outputs_sha256'][f'S{s}'], 'source TRAINING_COMPLETE_LOCK arm_outputs_sha256')
        check(f'FIT_{s}/ARM_OUTPUTS.npz (selection lock)', fd / 'ARM_OUTPUTS.npz', sel['arm_output_files_sha256'][f'runs/FIT_{s}/ARM_OUTPUTS.npz'],
              'source SELECTION_LOCK arm_output_files_sha256')
        fs = json.loads((fd / 'FIT_SEED.json').read_text())
        for rel, h in fs['files_sha256'].items():
            copy(fd / rel, f'fit/FIT_{s}/{rel}', h, 'source FIT_SEED.json files_sha256 (itself locked by TRAINING_COMPLETE_LOCK)')
        copy(fd / 'FIT_SEED.json', f'fit/FIT_{s}/FIT_SEED.json', tl['fit_seed_sha256'][f'S{s}'], 'source TRAINING_COMPLETE_LOCK fit_seed_sha256')
        check(f'SERVICE_BANK_S{s}.npz', C.SB_PRIV / 'bank' / f'SERVICE_BANK_S{s}.npz', sbm['bank_files_sha256'][f'S{s}'], 'source SERVICE_BANK_MANIFEST')
        rd = C.SB_PRIV / 'runs' / f'READERS_{s}'
        check(f'READERS_{s}/DEV_REPORT_SCORES.npz', rd / 'DEV_REPORT_SCORES.npz', rl['readers'][f'S{s}']['dev_report_scores_sha256'], 'source REPORTING_LOCK')
        check(f'READERS_{s}/READERS_SEED.json', rd / 'READERS_SEED.json', rl['readers'][f'S{s}']['readers_seed_sha256'], 'source REPORTING_LOCK')
    pub = C.public_export_sha()
    for role, fields in C.ADMITTED.items():
        for f in fields:
            rel = f'exports/{role}/{f}.npz'
            copy(C.SB_PRIV / 'source' / rel, rel, pub[role][f], 'public EXTRACTION_SUMMARY.json export_file_sha256 (f4ea199)')
            if rr['files'][rel]['sha256'] != pub[role][f]:
                raise PermissionError(f'source RESTORE_RECEIPT disagrees with the public extraction summary: {rel}')
    bank_man = json.loads((SB.HYB_RESULTS / 'FROZEN_BANK_MANIFEST.json').read_text())['bank_files_sha256']
    for s in C.SEEDS:
        copy(C.SB_PRIV / 'source' / 'bank' / f'FROZEN_BANK_S{s}.npz', f'bank/FROZEN_BANK_S{s}.npz', bank_man[f'S{s}'],
             'public FROZEN_BANK_MANIFEST.json (3a1f509)')
    man = json.loads((SB.SRC_RESULTS / 'MODEL_MANIFEST.json').read_text())
    copy(SB.SRC_PRIV / 'deploy' / 'T_TREE_S0' / 'PACKAGE.json', 'encoder/T_TREE_S0_PACKAGE.json',
         man['deployment_packages']['T_TREE_S0']['package_json_sha256'], 'public MODEL_MANIFEST deployment_packages (frozen 145-column encoder)')
    # ---- runtime routing by hash: reproduce every historical arm output bitwise from (routing, historical tables)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / 'SOURCE_ADMISSION.json').write_text(json.dumps(rec, indent=1, sort_keys=True))   # interim: copies needed by loaders
    for s in C.SEEDS:
        P, T, fs = C.source_fit(s)
        hist, ids = C.source_arm_outputs(s)
        st_n = np.bincount(P['fine_assign_MODEL_FIT'], minlength=P['cent1'].shape[0]).astype(np.float64)
        s1 = np.zeros_like(P['cent1']); np.add.at(s1, P['fine_assign_MODEL_FIT'], P['oof_income'])
        s2 = np.zeros_like(P['cent2']); np.add.at(s2, P['fine_assign_MODEL_FIT'], P['oof_occupation'])
        cent_ok = bool(np.array_equal(s1 / st_n[:, None], P['cent1']) and np.array_equal(s2 / st_n[:, None], P['cent2']))
        teachers = {t: [trees.TreeTeacher(json.loads((C.SRCP / f'fit/FIT_{s}/TEACHERS/fold{f}_{t}.json').read_text())) for f in range(5)] for t in C.TASKS}
        init, rows = {}, {}
        for r in C.OUT_ROLES:
            X = C.load_role(r, ('features',))
            if not (np.array_equal(ids[f'row_id__{r}'], X['row_id']) and np.array_equal(ids[f'unit__{r}'], X['unit'])):
                raise AssertionError(f'historical output row order differs: {r}')
            prob = {}
            for t in C.TASKS:
                acc = 0.0
                for f in range(5):                      # the source fold-average order (fit_scm.run)
                    acc = acc + teacher_probs(teachers[t][f], X['X'], X['feature_names']) / 5.0
                prob[t] = acc
            init[r] = E.kl_assign(prob['income'], prob['occupation'], P['cent1'], P['cent2'])
            rows[f'row_id__{r}'] = X['row_id']; rows[f'unit__{r}'] = X['unit']
        arms = sorted({k.split('__')[0] for k in T})
        repro = {}
        for arm in arms:
            tab = {k: T[f'{arm}__{k}'] for k in ('v1', 'v2', 'cell_of_initial', 'group_of_initial')}
            repro[arm] = all(np.array_equal(E.release(tab, init[r])[t], hist[arm][r][t]) for r in C.OUT_ROLES for t in C.TASKS)
        rel = f'routing/ROUTING_S{s}.npz'; dst = C.SRCP / rel; dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if dst.exists():
            raise PermissionError('routing file exists (append-only)')
        with open(dst, 'xb') as fh:
            np.savez(fh, **{f'init__{r}': init[r] for r in C.OUT_ROLES}, **rows)
        dst.chmod(0o600)
        rec['copies'][rel] = {'sha256': C.sha(dst), 'basis': 'computed here from admitted teachers and centroids; reproduces every historical arm output bitwise',
                              'bytes': dst.stat().st_size}
        rec['routing'][f'S{s}'] = {'centroids_equal_pooled_oof_cell_means': cent_ok, 'historical_arms_checked': len(arms),
                                   'historical_arms_reproduced_bitwise_all_4_roles': sum(repro.values()), 'per_arm': repro,
                                   'occupied_fine_cells': int(P['cent1'].shape[0]),
                                   'source_routing_diagnostics': fs['diagnostics']}
        if not (cent_ok and all(repro.values())):
            raise AssertionError(f'routing admission failed for seed {s}')
    rec['environment'] = {'python': sys.version.split()[0], 'numpy': np.__version__, 'platform': platform.platform(),
                          'scipy': __import__('scipy').__version__, 'sklearn': __import__('sklearn').__version__}
    rec['status'] = 'ADMITTED' if (rec['source']['tip_matches'] and rec['source']['publication_remote_matches']
                                   and all(v['match'] for v in checks.values())) else 'ADMISSION_CHECK_FAILED'
    (C.RESULTS / 'SOURCE_ADMISSION.json').write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({'status': rec['status'], 'tip_matches': rec['source']['tip_matches'], 'publication_remote_matches': rec['source']['publication_remote_matches'],
                      'public_package': rec['source']['public_package']['status_counts'], 'manifest_checks': len(checks), 'copies': len(rec['copies']),
                      'routing': {k: {kk: v[kk] for kk in ('centroids_equal_pooled_oof_cell_means', 'historical_arms_checked', 'historical_arms_reproduced_bitwise_all_4_roles')}
                                  for k, v in rec['routing'].items()}}, indent=1))
    if rec['status'] != 'ADMITTED':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
