"""Unit ADMISSION: restore the frozen source artifacts read-only and admit them. No model is fitted.

1. Pin: the listed source documents in the inherited results folder equal the published source f4ea199 byte for byte.
2. Restore: copy exactly the needed frozen files from the source private root into .private/acs_frozen_hybrids/source. Each is
   checked against its published sha256 (TRAINING_COMPLETE_LOCK.frozen_files_sha256, REPORTING_LOCK.reader_files_sha256,
   EXTRACTION_SUMMARY.export_file_sha256, MODEL_MANIFEST, SELECTION_LOCK.output_file_sha256). DEV_REPORT truth is not restored.
3. Admit: recompute every stored output used here from its frozen model on the admitted role features (row alignment,
   preprocessing, class order, probability definition): T_TREE, LOCAL_TREE, JOINT_TREE and TASK_REFIT from the teacher JSON plus
   leaf-score corrections; NEURAL_RAW rho1 E2 from its state; U0 and Ucal from U0 and its temperatures. Raw permitted fields:
   DEV_REPORT raw_inputs re-encoded with the frozen encoder equal the admitted features.
4. MODEL_SELECT point feasibility (the source's eligibility rule) is recomputed for every bank service from the already-exposed
   MODEL_SELECT task labels; no other label is read.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import common as C

PINNED_DOCS = ('SCIENCE_LOCK.json', 'MODEL_MANIFEST.json', 'SERVICE_TABLE.csv', 'PRIMARY_ENDPOINTS.csv', 'SELECTION_LOCK.json',
               'LOCK_AMENDMENTS.md', 'SEALED_ASSESSMENT_STATUS.md', 'INDEPENDENT_VERIFICATION.json', 'REPORTING_LOCK.json',
               'DEPLOYMENT_RESULTS.json', 'DEV_REPORT_RESULTS.json', 'POPULATION_AND_SCHEMA.md', 'SOURCE_ADMISSION.json',
               'EXTRACTION_SUMMARY.json', 'QUICKSTART.md', 'TRAINING_COMPLETE_LOCK.json', 'SUBGROUP_TABLE.csv')
RUN_FILES = ('STAGE_A_{s}/TREE_TEACHER_income.json', 'STAGE_A_{s}/TREE_TEACHER_occupation.json', 'STAGE_A_{s}/STAGE_A_OUTPUTS.npz',
             'STAGE_A_{s}/U0_MODEL.pt', 'STAGE_A_{s}/STAGE_A_SEED.json',
             'STAGE_B_{s}/LOCAL_TREE__E20.npz', 'STAGE_B_{s}/JOINT_TREE__E20.npz', 'STAGE_B_{s}/TASK_REFIT__E20.npz',
             'STAGE_B_{s}/STAGE_B_OUTPUTS_E20.npz', 'STAGE_B_{s}/STAGE_B_SEED.json',
             'NEURAL_RAW_{s}/NEURAL_RAW__rho1.0__E2.pt', 'NEURAL_RAW_{s}/NEURAL_RAW_OUTPUTS.npz', 'NEURAL_RAW_{s}/NEURAL_RAW_SEED.json')
READER_FILES = ('READERS_{s}/DEV_REPORT_SCORES.npz', 'READERS_{s}/READERS_SEED.json')
EXPORT_FIELDS = {'ATTACK_FIT': ('features', 'sex'), 'ATTACK_VALIDATION': ('features', 'sex'), 'DEV_REPORT': ('features', 'raw_inputs'),
                 'MODEL_SELECT': ('features', 'income', 'occupation')}


def jload(p):
    return json.loads(Path(p).read_text())


def pin_documents():
    out = {}
    for f in PINNED_DOCS:
        pub = subprocess.run(['git', '-C', str(C.ROOT), 'show', f'{C.PUBLISHED_SOURCE_COMMIT}:{C.PUBLISHED_SOURCE_PATH}/results/{f}'],
                             capture_output=True, check=True).stdout
        import hashlib
        out[f] = {'published_sha256': hashlib.sha256(pub).hexdigest(), 'inherited_sha256': C.sha(C.SRC_RESULTS / f)}
        out[f]['identical'] = out[f]['published_sha256'] == out[f]['inherited_sha256']
    return out


def restore():
    tl = jload(C.SRC_RESULTS / 'TRAINING_COMPLETE_LOCK.json')['frozen_files_sha256']
    rl = jload(C.SRC_RESULTS / 'REPORTING_LOCK.json')['reader_files_sha256']
    ex = jload(C.SRC_RESULTS / 'EXTRACTION_SUMMARY.json')['export_file_sha256']
    mm = jload(C.SRC_RESULTS / 'MODEL_MANIFEST.json')['models']
    sel = jload(C.SRC_RESULTS / 'SELECTION_LOCK.json')['output_file_sha256']
    plan = {}
    for s in C.SEEDS:
        for f in RUN_FILES:
            rel = f.format(s=s); plan[f'runs/{rel}'] = {'expected': tl[rel], 'pinned_by': 'TRAINING_COMPLETE_LOCK.frozen_files_sha256'}
        for f in READER_FILES:
            rel = f.format(s=s)
            if rel in rl:
                plan[f'runs/{rel}'] = {'expected': rl[rel], 'pinned_by': 'REPORTING_LOCK.reader_files_sha256'}
            else:
                plan[f'runs/{rel}'] = {'expected': None, 'pinned_by': 'not pinned (receipt file; content cross-checked against REPORTING_LOCK winners)'}
    rc_src = jload(C.SRC_PRIV / 'extraction' / 'EXTRACTION_RECEIPT.json')
    for role, fields in EXPORT_FIELDS.items():
        for fld in fields:
            item = rc_src['files'][role][fld]
            rel = str(Path(item['path']).relative_to('.private/acs_replication'))
            pub = ex[role][fld] if isinstance(ex[role], dict) else None
            if pub != item['sha256']:
                raise PermissionError(f'extraction receipt and public EXTRACTION_SUMMARY disagree for {role}/{fld}')
            plan[rel] = {'expected': pub, 'pinned_by': 'EXTRACTION_SUMMARY.export_file_sha256'}
    plan['extraction/EXTRACTION_RECEIPT.json'] = {'expected': None, 'pinned_by': 'private receipt; every export hash in it equals the public EXTRACTION_SUMMARY (checked)'}
    files = {}
    for rel, info in sorted(plan.items()):
        src = C.SRC_PRIV / rel; dst = C.SRCP / rel
        h = C.sha(src)
        if info['expected'] is not None and h != info['expected']:
            raise PermissionError(f'source file differs from its published hash: {rel}')
        dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if dst.exists():
            raise PermissionError(f'restore target exists: {rel}')
        shutil.copy2(src, dst); dst.chmod(0o600)
        if C.sha(dst) != h:
            raise PermissionError(f'copy differs: {rel}')
        files[rel] = {'sha256': h, 'bytes': dst.stat().st_size, 'pinned_by': info['pinned_by'], 'matches_published': info['expected'] is not None}
    cross = {}
    for s in C.SEEDS:
        for lab, rel in (('LOCAL_TREE_E20', f'runs/STAGE_B_{s}/LOCAL_TREE__E20.npz'), ('JOINT_TREE_E20', f'runs/STAGE_B_{s}/JOINT_TREE__E20.npz'),
                         ('TASK_REFIT_E20', f'runs/STAGE_B_{s}/TASK_REFIT__E20.npz'), ('NEURAL_RAW_rho1.0_E2', f'runs/NEURAL_RAW_{s}/NEURAL_RAW__rho1.0__E2.pt'),
                         ('U0', f'runs/STAGE_A_{s}/U0_MODEL.pt')):
            cross[f'{lab}_S{s}'] = files[rel]['sha256'] == mm[f'{lab}_S{s}']['sha256']
        for t in C.TASKS:
            cross[f'T_TREE_{t}_S{s}'] = files[f'runs/STAGE_A_{s}/TREE_TEACHER_{t}.json']['sha256'] == mm[f'T_TREE_{t}_S{s}']['sha256']
        for fam in ('STAGE_A', 'STAGE_B', 'NEURAL_RAW'):
            rel = f'runs/{fam}_{s}/' + {'STAGE_A': 'STAGE_A_OUTPUTS.npz', 'STAGE_B': 'STAGE_B_OUTPUTS_E20.npz', 'NEURAL_RAW': 'NEURAL_RAW_OUTPUTS.npz'}[fam]
            cross[f'{fam}_outputs_S{s}_vs_SELECTION_LOCK'] = files[rel]['sha256'] == sel[rel[len('runs/'):]] if rel[len('runs/'):] in sel else None
    if not all(v for v in cross.values() if v is not None):
        raise PermissionError(f'MODEL_MANIFEST / SELECTION_LOCK cross-check failed: {cross}')
    rec = {'schema': 'ACSH1-restore-v1', 'utc': C.now(), 'source_private_root': 'source worktree .private/acs_replication (read-only)',
           'files': files, 'model_manifest_and_selection_lock_cross_checks': cross,
           'not_restored': ['DEV_REPORT_TRUTH (read in place only after this campaign\'s REPORTING_LOCK)', 'raw census file', 'SEALED_ASSESSMENT (no export exists)']}
    (C.PRIV / 'RESTORE_RECEIPT.json').write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


def admit():
    import torch
    from acs_replication import deploy, neural, trees
    torch.set_num_threads(1)
    rc = C.receipt(); enc = rc['encoding']
    names = tuple(enc['feature_names'])
    roles = {r: C.load_role(r, ('features',)) for r in C.BANK_ROLES}
    checks = {}
    # Raw permitted fields -> frozen encoder -> admitted features (DEV_REPORT).
    raw = C.load_role('DEV_REPORT', ('features', 'raw_inputs'))['raw_inputs']
    rows = [{c: raw[c][i] for c in deploy.PERMITTED} for i in range(len(raw['row_id']))]
    Xr = deploy.encode_raw(rows, enc)
    checks['raw_permitted_fields'] = {'fields': list(deploy.PERMITTED), 'forbidden_fields_absent_from_raw_export': not (set(raw) & {'SEX', 'sex', 'PWGTP'}),
                                      'row_order_equal': bool(np.array_equal(raw['row_id'], roles['DEV_REPORT']['row_id'])),
                                      'encoded_equals_admitted_features': bool(np.array_equal(Xr, roles['DEV_REPORT']['X'])),
                                      'feature_names_equal_encoder': roles['DEV_REPORT']['feature_names'] == names}
    recompute = {}
    for s in C.SEEDS:
        outs, rowkeys = C.source_outputs(s)
        for fam, rk in rowkeys.items():
            for r in C.BANK_ROLES:
                if f'row_id__{r}' in rk:
                    checks.setdefault('stored_row_ids_equal_exports', {})[f'S{s}|{fam}|{r}'] = bool(
                        np.array_equal(rk[f'row_id__{r}'], roles[r]['row_id']) and np.array_equal(rk[f'unit__{r}'], roles[r]['unit']))
        teachers = {t: trees.TreeTeacher(jload(C.SRCP / f'runs/STAGE_A_{s}/TREE_TEACHER_{t}.json')) for t in C.TASKS}
        thetas = {}
        for lab in ('LOCAL_TREE', 'JOINT_TREE', 'TASK_REFIT'):
            with np.load(C.SRCP / f'runs/STAGE_B_{s}/{lab}__E20.npz') as z:
                if sorted(z.files) != ['theta_income', 'theta_occupation']:
                    raise PermissionError(f'unexpected checkpoint fields: {lab}')
                thetas[lab] = {t: z[f'theta_{t}'].copy() for t in C.TASKS}
        nstate = torch.load(C.SRCP / f'runs/NEURAL_RAW_{s}/NEURAL_RAW__rho1.0__E2.pt', map_location='cpu', weights_only=True)
        ustate = torch.load(C.SRCP / f'runs/STAGE_A_{s}/U0_MODEL.pt', map_location='cpu', weights_only=True)
        for r in C.BANK_ROLES:
            X = roles[r]['X']
            for t in C.TASKS:
                leaves = teachers[t].route(X, names)
                comp = {'T_TREE': trees.release(t, teachers[t].raw_from_leaves(leaves, None))['probability']}
                for lab in thetas:
                    comp[lab] = trees.release(t, teachers[t].raw_from_leaves(leaves, thetas[lab][t]))['probability']
                m = neural.network(C.K[t], X.shape[1]); m.load_state_dict(nstate[t]); m.eval()
                comp['NEURAL_RAW_rho1.0_E2'] = neural.probabilities(m, X)
                u = neural.network(C.K[t], X.shape[1]); u.load_state_dict(ustate[t]); u.eval()
                comp['U0'] = neural.probabilities(u, X)
                comp['Ucal'] = neural.tempered_probabilities(neural.logits(u, X), float(ustate['temperatures'][t]))
                for svc, p in comp.items():
                    st = outs[svc][r][t]
                    recompute[f'S{s}|{svc}|{r}|{t}'] = {'max_abs_diff': float(np.max(np.abs(p - st))), 'bitwise': bool(np.array_equal(p, st)),
                                                       'decisions_equal': bool(np.array_equal(p.argmax(1), st.argmax(1))),
                                                       'shape': list(st.shape), 'classes': int(st.shape[1])}
    worst = max(v['max_abs_diff'] for v in recompute.values())
    checks['recompute_from_frozen_models'] = {'entries': len(recompute), 'bitwise': sum(v['bitwise'] for v in recompute.values()),
                                              'max_abs_diff': worst, 'all_decisions_equal': all(v['decisions_equal'] for v in recompute.values()),
                                              'tolerance': 1e-12, 'pass': bool(worst <= 1e-12 and all(v['decisions_equal'] for v in recompute.values()))}
    checks['class_order'] = {'income_classes': 2, 'occupation_classes': 5,
                             'all_shapes_match': all(v['classes'] == C.K[k.split('|')[3]] for k, v in recompute.items())}
    return checks, recompute


def model_select_feasibility():
    """Source eligibility rule: all 24 MODEL_SELECT point clauses (2 tasks x U0/Ucal x 3 metrics x 2 seeds)."""
    ms = C.load_role('MODEL_SELECT', ('features', 'income', 'occupation'))
    out = {}
    for s in C.SEEDS:
        outs, _ = C.source_outputs(s, roles=('MODEL_SELECT',))
        bank = {h: {'income': outs[a]['MODEL_SELECT']['income'], 'occupation': outs[b]['MODEL_SELECT']['occupation']} for h, (a, b) in C.SOURCE_OF.items()}
        for svc, probs in bank.items():
            for t in C.TASKS:
                v = C.task_vectors(ms[t], probs[t], C.K[t])
                for ref in C.REFERENCES:
                    rv = C.task_vectors(ms[t], outs[ref]['MODEL_SELECT'][t], C.K[t])
                    d = {'accuracy_loss': rv['accuracy'].mean() - v['accuracy'].mean(), 'nll_excess': v['nll'].mean() - rv['nll'].mean(),
                         'brier_excess': v['brier'].mean() - rv['brier'].mean()}
                    for metric, cap in C.CAPS.items():
                        out.setdefault(svc, []).append({'seed': s, 'task': t, 'reference': ref, 'metric': metric, 'value': float(d[metric]),
                                                        'cap': cap, 'point_pass': bool(d[metric] <= cap)})
    sel = jload(C.SRC_RESULTS / 'SELECTION_LOCK.json')['fixed_services']
    summary = {svc: {'point_feasible_24': all(c['point_pass'] for c in cl), 'n_clauses': len(cl),
                     'failed_clauses': [f"s{c['seed']}|{c['task']}|{c['reference']}|{c['metric']}" for c in cl if not c['point_pass']]}
               for svc, cl in out.items()}
    agree = {'H_LL_vs_SELECTION_LOCK_LOCAL_TREE': summary['H_LL']['failed_clauses'] == sel['LOCAL_TREE']['failed_clauses'],
             'H_JJ_vs_SELECTION_LOCK_JOINT_TREE': sorted(summary['H_JJ']['failed_clauses']) == sorted(sel['JOINT_TREE']['failed_clauses']),
             'TASK_REFIT_vs_SELECTION_LOCK': summary['TASK_REFIT']['failed_clauses'] == sel['TASK_REFIT']['failed_clauses'],
             'T_TREE_vs_SELECTION_LOCK': summary['T_TREE']['failed_clauses'] == sel['T_TREE']['failed_clauses']}
    return summary, out, agree


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    pins = pin_documents()
    if not all(v['identical'] for v in pins.values()):
        raise PermissionError('inherited source documents differ from the published source')
    rec = restore()
    checks, recompute = admit()
    ms_summary, ms_clauses, ms_agree = model_select_feasibility()
    src_ver = jload(C.SRC_RESULTS / 'INDEPENDENT_VERIFICATION.json')
    res = {'schema': 'ACSH1-source-admission-v1', 'utc': C.now(), 'status': None,
           'published_source': {'commit': C.PUBLISHED_SOURCE_COMMIT, 'path': C.PUBLISHED_SOURCE_PATH,
                                'url': f'https://github.com/Bostesa/PCRL/tree/{C.PUBLISHED_SOURCE_COMMIT}/{C.PUBLISHED_SOURCE_PATH}',
                                'documents': pins},
           'restored_files': {k: {kk: v[kk] for kk in ('sha256', 'pinned_by', 'matches_published')} for k, v in rec['files'].items()},
           'model_manifest_and_selection_lock_cross_checks': rec['model_manifest_and_selection_lock_cross_checks'],
           'admission_checks': checks, 'model_select_point_feasibility': ms_summary, 'model_select_inheritance_agreement': ms_agree,
           'source_issues_carried_forward': {
               'D1': ('Transport-annotation failure, not a numerical failure: the source\'s locked records said the transported files differ from '
                      'their sources only on ACS-CHANGE-marked lines; that wording was false (unmarked but declared edits). The code was '
                      'byte-identical to the SCIENCE_LOCK versions. The source verification stays 10/11 PASS (check 1 FAIL); it is not '
                      'restated as 11/11.'),
               'source_verification_counts': {k: src_ver.get(k) for k in ('pass_count', 'fail_count', 'other_count')},
               'D2': ('The source extractor computed sealed-household features, SEX, weights and both task labels transiently in memory before '
                      'role separation and discarded them unwritten (counts, a key-set hash and one data-quality count only). The holdout is '
                      'therefore not described as never accessed. This campaign does not open, regenerate or hash the sealed role, does not '
                      'rerun the extractor and does not read the raw census file.')},
           'exploratory_status': 'EXPLORATORY DEVELOPMENT FOLLOW-UP of the already-opened ACS DEV_REPORT results; not confirmation, not an independent replication, not a rescue of the source claim; source labels unchanged'}
    ok = (all(v['identical'] for v in pins.values()) and checks['recompute_from_frozen_models']['pass']
          and checks['raw_permitted_fields']['encoded_equals_admitted_features'] and checks['raw_permitted_fields']['row_order_equal']
          and all(checks.get('stored_row_ids_equal_exports', {}).values()) and all(ms_agree.values()) and checks['class_order']['all_shapes_match'])
    res['status'] = 'ADMITTED' if ok else 'ADMISSION_FAILED'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'SOURCE_ADMISSION.json').write_text(json.dumps(res, indent=2, sort_keys=True))
    (C.PRIV / 'admission').mkdir(parents=True, exist_ok=True, mode=0o700)
    (C.PRIV / 'admission' / 'RECOMPUTE_DETAIL.json').write_text(json.dumps({'recompute': recompute, 'model_select_clauses': ms_clauses}, indent=1, sort_keys=True))
    print(json.dumps({'status': res['status'], 'recompute': checks['recompute_from_frozen_models'], 'raw': checks['raw_permitted_fields'],
                      'ms_feasible': {k: v['point_feasible_24'] for k, v in ms_summary.items()}, 'ms_agree': ms_agree}, indent=1))
    if not ok:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
