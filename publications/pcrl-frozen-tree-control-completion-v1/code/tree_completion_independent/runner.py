"""Runner: build unit configs, enqueue them on the shared queue, drive it, summarize receipts.

Usage (from the worktree):
  python -m tree_completion_independent.runner enqueue <STAGE> <UID>
  python -m tree_completion_independent.runner drive [N]
  python -m tree_completion_independent.runner status
  python -m tree_completion_independent.runner summary
No numerical work happens here: only hashing for source registration, JSON metadata and receipt aggregation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / '.private/tree_completion/root'))
import coord  # noqa: E402
from coord import CFG, OLDP, OLDRES, PRIV, READERS, RES, REX, SA, SPG, item, sha  # noqa: E402

OUT = RES / 'independent'
INN = REX / 'INNER_SELECTION'
SCI = RES / 'SCIENCE_AUTHORIZATION.json'
SNAP = Path('~/PCRL_eval_cache_private/tc_v1/backups/SNAPSHOT_V1')
SNAP_SHA = '46ffc6145a3aa664514cd97ed17aeb9515b94e9815cd7edc7d31a1a10c6c5861'
NEW = ('T_REFIT_E20', 'RAW_J_NEQ', 'T_NEURAL', 'FARE_TC', 'FARE_TC_Z')
STAGE_MODULE = {'CONTROLS': 'stage_controls', 'WIRES': 'stage_wires', 'NOMINATION': 'stage_nomination', 'POWER': 'stage_power',
                'ACCOUNTING': 'stage_accounting', 'RESTORE': 'stage_restore'}
WALL = {'CONTROLS': 2400, 'WIRES': 900, 'NOMINATION': 1800, 'POWER': 2400, 'ACCOUNTING': 900, 'RESTORE': 2400}


def code():
    return {p.name: item(p) for p in sorted((W / 'tree_completion_independent').glob('*.py'))}


def banks():
    return {'old': {s: {'receipt': item(READERS[int(s)] / 'FROZEN_COMPLETE.json'), 'scores': item(READERS[int(s)] / 'INNER_SCORES.npz')} for s in ('0', '1')},
            'new': {s: {'receipt': item(PRIV / f'readers/S{s}/FROZEN_COMPLETE.json'), 'scores': item(PRIV / f'readers/S{s}/INNER_SCORES.npz')} for s in ('0', '1')}}


def truth():
    return {k: item(INN / f'{k}.npz') for k in ('sex', 'income', 'occupation')}


def refs():
    return {s: item(INN / f'references_s{s}.npz') for s in ('0', '1')}


def locks():
    """reporting/selection locks follow the queue_config convention (path + '<name>_sha256'), so guard() re-verifies them."""
    rl, sl = RES / 'REPORTING_LOCK.json', RES / 'COMPLETION_SELECTION_LOCK.json'
    return {'reporting_lock': str(rl), 'reporting_lock_sha256': sha(rl), 'selection_lock': str(sl), 'selection_lock_sha256': sha(sl),
            'completion_lock': item(RES / 'COMPLETION_LOCK.json'), 'manifest': item(RES / 'FROZEN_CANDIDATE_MANIFEST.json')}


def new_rel(svc):
    return {s: item(PRIV / f'readers/S{s}/INNER_RELEASE__{svc}.npz') for s in ('0', '1')}


def old_rel(lab):
    return {s: item(READERS[int(s)] / f'INNER_RELEASE__{lab}.npz') for s in ('0', '1')}


def cfg_controls():
    rel = {'L_TREE': old_rel('L_TREE'), **{svc: new_rel(svc) for svc in NEW}, 'T_TREE_old': old_rel('T_TREE'), 'T_TREE_new': new_rel('T_TREE')}
    csvs = ('CONTROL_COMPLETION_RESULTS', 'COMPLETION_UTILITY_BOUNDS', 'COMPLETION_SERVICE_TABLE')
    return {'inner_truth': truth(), 'inner_features': item(INN / 'features.npz'), 'inner_references': refs(), 'banks': banks(), 'releases': rel,
            'report': item(PRIV / 'report_public/COMPLETION_REPORT_V2.json'), 'report_results_copy': item(RES / 'COMPLETION_REPORT.json'),
            'report_csv': {c: item(PRIV / f'report_V2/{c}.csv') for c in csvs}, 'results_csv': {c: item(RES / f'{c}.csv') for c in csvs},
            'tc_report_v2_config': item(CFG / 'TC_REPORT_V2.json'), 'bootstrap': {'seed': 815_2026_11, 'draws': 4096}, 'K': 11,
            **locks(), 'code': code()}


def cfg_wires():
    olds = ('L_TREE', 'T_TREE', 'T_REFIT', 'J_TREE', 'RAW_PROJ', 'N_EQ', 'H_JOINT_CODE', 'H_SEQ21_CODE', 'H_DPC_JOINT', 'FARE', 'FARE_Z1')
    return {'inner_features': item(INN / 'features.npz'),
            'neural': {s: {'RAW_J_NEQ': item(PRIV / f'training/S{s}/NEURAL/RAW_J__rho1.0__E2.pt'),
                           'T_NEURAL': item(PRIV / f'training/S{s}/NEURAL/T_NEURAL__rhonone__E2.pt')} for s in ('0', '1')},
            'neural_record': {s: item(PRIV / f'training/S{s}/NEURAL/NEURAL_RECORD.json') for s in ('0', '1')},
            'fare_release': {s: {'FARE_TC': item(PRIV / f'training/S{s}/FARE/FARE_RELEASE__F1.npz'),
                                 'FARE_TC_Z': item(PRIV / f'training/S{s}/FARE/FARE_RELEASE__Z1.npz')} for s in ('0', '1')},
            'fare_record': {s: item(PRIV / f'training/S{s}/FARE/FARE_RECORD.json') for s in ('0', '1')},
            'new_releases': {svc: new_rel(svc) for svc in NEW + ('T_TREE',)}, 'old_releases': {lab: old_rel(lab) for lab in olds},
            'receipts': {s: item(PRIV / f'readers/S{s}/FROZEN_COMPLETE.json') for s in ('0', '1')}, **locks(), 'code': code()}


def cfg_nomination():
    return {'fit2_tables': {s: item(PRIV / f'fit2_public/FIT2_S{s}.json') for s in ('0', '1')},
            'fit2_results_copy': {s: item(RES / f'COMPLETION_FIT2_TABLE_S{s}.json') for s in ('0', '1')},
            'selection': item(PRIV / 'selection_public/SELECTION.json'), 'selection_detail': item(RES / 'COMPLETION_SELECTION_DETAIL.json'),
            'fit2_features': item(SPG / 'VALIDATION/features.npz'), 'fit2_sex': item(OLDP / 'exports/V3/FIT2/sex.npz'),
            'fit2_income': item(SPG / 'VALIDATION/income.npz'), 'fit2_occupation': item(SPG / 'VALIDATION/occupation.npz'),
            'neural': {s: {'RAW_J': item(PRIV / f'training/S{s}/NEURAL/RAW_J__rho1.0__E2.pt')} for s in ('0', '1')},
            'neq': {s: item(SA[int(s)] / 'NEQ_MODEL.pt') for s in ('0', '1')},
            'fare': {s: {'F1': item(PRIV / f'training/S{s}/FARE/FARE_RELEASE__F1.npz')} for s in ('0', '1')},
            **locks(), 'code': code()}


def cfg_power():
    P = RES / 'power'
    return {'feasibility_inputs': item(P / 'FEASIBILITY_INPUTS.json'), 'feasibility_inputs_private': item(PRIV / 'report_public/FEASIBILITY_INPUTS.json'),
            'feasibility_supplement': item(P / 'FEASIBILITY_SUPPLEMENT.json'), 'power_meta': item(P / 'POWER_META.json'),
            'joint_power': item(P / 'JOINT_POWER_TABLES.csv'), 'privacy_only': item(P / 'PRIVACY_ONLY_AND_BEYOND_CAP.csv'),
            'phi_private': item(PRIV / 'feasibility/PHI_PER_GROUP.npz'), 'inner_truth': truth(), 'inner_references': refs(), 'banks': banks(),
            'candidate_release': old_rel('L_TREE'), 'bootstrap_seed': 31_2026_7, 'draws': 200000, 'sim_seed': 271828182,
            **locks(), 'code': code()}


def cfg_accounting():
    roles = {'UPDATE': item(SPG / 'UPDATE/features.npz'), 'GUARD': item(SPG / 'GUARD/features.npz'), 'FIT2': item(SPG / 'VALIDATION/features.npz'),
             'ATT': item(REX / 'ATTACK_FIT_NEW/features.npz'), 'INNER': item(INN / 'features.npz')}
    c = {'roles': roles, 'inner_release': old_rel('L_TREE'), 'scores': {b: {s: v[s]['scores'] for s in ('0', '1')} for b, v in banks().items()},
         'report': item(PRIV / 'report_public/COMPLETION_REPORT_V2.json'), 'feasibility_inputs': item(RES / 'power/FEASIBILITY_INPUTS.json'),
         'power_meta': item(RES / 'power/POWER_META.json'), 'exposure_matrix': item(RES / 'COMPARATOR_EXPOSURE_MATRIX.csv'),
         'population_csv': item(RES / 'POPULATION_COMPATIBILITY_AND_USAGE.csv'), 'population_audit': item(RES / 'POPULATION_AUDIT.md'),
         'old_inner_report': item(OLDRES / 'INNER_REPORT.json'), 'old_selection_lock': item(OLDRES / 'SELECTION_LOCK.json'),
         'fare_record': item(PRIV / 'training/S0/FARE/FARE_RECORD.json'),
         'fit2_tables': {s: item(PRIV / f'fit2_public/FIT2_S{s}.json') for s in ('0', '1')}, **locks(), 'code': code()}
    if (OUT / 'STAGE_CONTROLS.json').exists():
        c['stage_controls'] = item(OUT / 'STAGE_CONTROLS.json')
    return c


def cfg_restore(dest):
    man = json.loads((SNAP / 'SNAPSHOT_MANIFEST.json').read_text())
    return {'snapshot_root': str(SNAP), 'manifest': item(SNAP / 'SNAPSHOT_MANIFEST.json'), 'announced_manifest_sha256': SNAP_SHA,
            'files': {e['path']: item(SNAP / e['path']) for e in man}, 'restore_dir': dest,
            'frozen_manifest': item(RES / 'FROZEN_CANDIDATE_MANIFEST.json'), 'reporting_lock': str(RES / 'REPORTING_LOCK.json'),
            'reporting_lock_sha256': sha(RES / 'REPORTING_LOCK.json'), 'python': coord.PY, 'code': code()}


def _item_dicts(x, acc):
    if isinstance(x, dict):
        if set(x) == {'path', 'sha256'}:
            acc.append(x)
        else:
            for v in x.values():
                _item_dicts(v, acc)
    return acc


def items_in(x, acc):
    if isinstance(x, dict):
        if set(x) == {'path', 'sha256'}:
            acc.append(x['path'])
        else:
            for v in x.values():
                items_in(v, acc)
    elif isinstance(x, list):
        for v in x:
            items_in(v, acc)
    return acc


def enqueue(stage, uid, dest=None):
    if not uid.startswith('TCV_'):
        raise SystemExit('unit ids must start with TCV_')
    cfg = {'CONTROLS': cfg_controls, 'WIRES': cfg_wires, 'NOMINATION': cfg_nomination, 'POWER': cfg_power,
           'ACCOUNTING': cfg_accounting}.get(stage, lambda: cfg_restore(dest))()
    cfg['stage'] = stage
    files = items_in(cfg, [])
    files += [cfg[k] for k in ('reporting_lock', 'selection_lock') if k in cfg]
    files = sorted(set(files))
    out = coord.unit(uid, 'verification', f'tree_completion_independent.{STAGE_MODULE[stage]}', cfg, extra_sources=files,
                     outputs=[str(OUT / f'STAGE_{stage}.json')], wall=WALL[stage], rss=2, amendment=sha(SCI))
    print(out[-400:])


def summary():
    stages = ['CONTROLS', 'WIRES', 'NOMINATION', 'POWER', 'ACCOUNTING', 'RESTORE']
    recs = {}
    for s in stages:
        p = OUT / f'STAGE_{s}.json'
        recs[s] = json.loads(p.read_text()) if p.exists() else None
    st = coord.status()
    out = {'schema': 'TCV1-independent-verification-v1', 'campaign': 'pcrl_frozen_tree_control_completion_v1', 'role': 'INDEPENDENT_VERIFICATION',
           'stages': {s: ({'status': r['status'], 'n_checks': r['n_checks'], 'n_failed': r['n_failed'], 'failed_checks': r['failed_checks'],
                           'queue_unit_id': r['queue_unit_id'], 'queue_status': st['units'].get(r['queue_unit_id'], {}).get('status'),
                           'receipt': f'STAGE_{s}.json', 'receipt_sha256': sha(OUT / f'STAGE_{s}.json')} if r else {'status': 'PENDING'})
                      for s, r in recs.items()},
           'imports': recs['CONTROLS']['imports'] if recs.get('CONTROLS') else None,
           'package_sha256': {k: v['sha256'] for k, v in code().items()},
           'attempt_history': {u: v.get('status') for u, v in st['units'].items() if u.startswith('TCV_')},
           'attempt_notes': ['TCV_CONTROLS_V1 failed before any computation: verifier config used item dicts for reporting/selection '
                             'locks where queue_config expects path + <name>_sha256 (verifier engineering).',
                             'TCV_CONTROLS_V2 completed with 5 failed checks, all verifier-side: (i) winner-rule replay read the '
                             'receipt candidates in sorted-key order instead of the frozen slate order, so exact ATT-validation ties '
                             '(OUT_MLP = LOG_LR on income) broke differently; (ii) the shuffled-label control was scored against true '
                             'SEX instead of group-shuffled INNER SEX. Both fixed in V3; no endpoint, interval, status or label changed.'],
           'headline': {}}
    if recs.get('CONTROLS'):
        n = recs['CONTROLS']['numbers']
        out['headline']['CONTROLS'] = {k: n.get(k) for k in ('family_point_max_abs_diff', 'family_points_bitwise_equal', 'family_interval_end_max_abs_diff',
                                                             'utility_point_max_abs_diff', 'utility_interval_end_max_abs_diff', 'labels')}
    if recs.get('WIRES'):
        out['headline']['WIRES'] = recs['WIRES']['numbers'].get('neural_recompute')
    if recs.get('NOMINATION'):
        out['headline']['NOMINATION'] = {'proxy_max_abs_diff_by_scaling': recs['NOMINATION']['numbers'].get('proxy_max_abs_diff_by_scaling')}
    if recs.get('POWER'):
        n = recs['POWER']['numbers']
        out['headline']['POWER'] = {'targets': {k: {f: v.get(f) for f in ('N80', 'N90', 'published_N80', 'published_N90', 'rel_diff_N80', 'rel_diff_N90')}
                                                for k, v in n['power_targets'].items()}, 'worst_rel_diff': n.get('power_worst_rel_diff'),
                                    'binding_variances': n.get('binding_variances')}
    if recs.get('ACCOUNTING'):
        out['headline']['ACCOUNTING'] = recs['ACCOUNTING']['numbers']
    if recs.get('RESTORE'):
        out['headline']['RESTORE'] = {'deployment': recs['RESTORE']['numbers'].get('deployment'), 'reader_restore': recs['RESTORE']['numbers'].get('reader_restore'),
                                      'restore_dir': recs['RESTORE']['numbers'].get('restore_dir')}
    for s, r in recs.items():
        if not r:
            continue
        c = json.loads((CFG / f"{r['queue_unit_id']}.json").read_text())
        ins = {p: h for p, h in ((x['path'], x['sha256']) for x in _item_dicts(c, []) if 'tree_completion_independent' not in x['path'])}
        for k in ('reporting_lock', 'selection_lock'):
            if k in c:
                ins[c[k]] = c[k + '_sha256']
        out['stages'][s]['unit_config_sha256'] = sha(CFG / f"{r['queue_unit_id']}.json")
        if s == 'RESTORE':
            out['stages'][s]['inputs'] = {'snapshot_manifest_sha256': c['manifest']['sha256'], 'snapshot_files_registered': len(c['files'])}
        else:
            out['stages'][s]['inputs_sha256'] = dict(sorted(ins.items()))
    cur = out['package_sha256']
    out['numerical_modules_unchanged_since_stage_runs'] = all(
        all(cur.get(k) == v for k, v in r['code_sha256'].items() if k != 'runner.py') for r in recs.values() if r)
    done = [v['status'] for v in out['stages'].values()]
    out['status'] = 'PASS' if all(d == 'PASS' for d in done) else ('FAIL' if 'FAIL' in done else 'INCOMPLETE')
    p = OUT / 'INDEPENDENT_VERIFICATION.json'
    p.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v['status'] for k, v in out['stages'].items()}), out['status'])


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('enqueue'); e.add_argument('stage'); e.add_argument('uid'); e.add_argument('--dest')
    d = sub.add_parser('drive'); d.add_argument('n', nargs='?', type=int, default=1)
    sub.add_parser('status'); sub.add_parser('summary')
    a = ap.parse_args()
    if a.cmd == 'enqueue':
        enqueue(a.stage, a.uid, a.dest)
    elif a.cmd == 'drive':
        print(coord.drive(a.n))
    elif a.cmd == 'status':
        s = coord.status()
        print(json.dumps({u: v.get('status') for u, v in s['units'].items() if u.startswith('TCV_') or v.get('status') in ('queued', 'running')}, indent=1))
        print('capacity', s.get('capacity_now'), 'cpu_remaining', round(s['cpu_remaining_s']), 'wall_remaining', round(s['wall_remaining_s']))
    else:
        summary()


if __name__ == '__main__':
    main()
