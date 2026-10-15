"""Runner: build independent-verification unit configs, enqueue them on the shared queue, drive, summarize.

Usage (from the worktree):
  python -m task_headroom_independent.runner enqueue <UID> <STAGE>[,<STAGE>...] [--drive]
  python -m task_headroom_independent.runner summary
No numerical work happens here (only hashing for registration and JSON aggregation).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
PRIV = W / '.private/task_headroom'
RES = W / 'results/pcrl_task_headroom_protected_outputs_v1'
OUT = RES / 'independent'
sys.path.insert(0, str(PRIV / 'root'))
import coord  # noqa: E402

STAGE_A_DIRS = {0: PRIV / 'stage_a/S0_V2', 1: PRIV / 'stage_a/S1'}
FAMILIES = ('J_TREE', 'L_TREE', 'L_STRONG', 'S12_TREE', 'S21_TREE', 'T_REFIT')
B_ROSTER = ['RAW_PROJ', 'N_EQ', 'T_TREE', 'H_JOINT_CODE', 'H_SEQ21_CODE', 'H_DPC_JOINT', 'FARE']
PKG = sorted(str(p) for p in (W / 'task_headroom_independent').glob('*.py'))


def item(p):
    return coord.item(Path(p))


def base_cfg():
    adm = json.loads((PRIV / 'root/configs/THP_ADMISSION_V3.json').read_text())
    roles = {r: {f: dict(v) for f, v in fields.items()} for r, fields in adm['roles'].items()}
    roles['FIT2']['sex'] = item(PRIV / 'exports/V3/FIT2/sex.npz')
    stage_a = {}
    for s, d in STAGE_A_DIRS.items():
        names = ['STAGE_A_SEED.json', 'NEQ_MODEL.pt', 'NEQ_OUTPUTS.npz'] + [
            f'{k}_{t}_d{dp}.{ext}' for t in ('income', 'occupation') for dp in (4, 6)
            for k, ext in (('TREE_TEACHER', 'json'), ('CATBOOST', 'cbm'), ('TREE_OUTPUTS', 'npz'))]
        stage_a[str(s)] = {n: item(d / n) for n in names}
    return {'roles': roles, 'admission_config': item(PRIV / 'root/configs/THP_ADMISSION_V3.json'),
            'admission_receipt': item(RES / 'ADMISSION.json'), 'sci_lock': item(RES / 'SCIENCE_LOCK.json'),
            'stage_b_lock': item(RES / 'STAGE_B_LOCK.json'), 'stage_a_report': item(RES / 'STAGE_A_REPORT.json'),
            'stage_a': stage_a, 'raw_proj_stored': adm['raw_proj_stored'], 'historical': adm['historical'],
            'bootstrap_seed': 20261015, 'teacher_depths': [4, 6], 'B_roster': B_ROSTER}


def training_cfg():
    cks, recs = {}, {}
    for s in (0, 1):
        cks[str(s)] = {}; recs[str(s)] = {}
        for fam in FAMILIES:
            d = PRIV / f'training/S{s}/{fam}'
            if not (d / 'FAMILY_RECORD.json').exists():
                raise SystemExit(f'training incomplete: S{s} {fam}')
            recs[str(s)][fam] = item(d / 'FAMILY_RECORD.json')
            for p in sorted(d.glob(f'{fam}__rho*__*.npz')):
                _, rho, lab = p.stem.split('__')
                cks[str(s)][f'{fam}|{rho[3:]}|{lab}'] = item(p)
    return {'checkpoints': cks, 'family_records': recs, 'families': list(FAMILIES)}


def selection_cfg():
    out = {'fit2_tables': {str(s): item(PRIV / f'fit2_public/FIT2_S{s}.json') for s in (0, 1)},
           'selection': item(PRIV / 'selection_public/SELECTION.json'), 'proxy_scope': 'feasible'}
    if (RES / 'SELECTION_LOCK.json').exists():
        out['sel_lock'] = item(RES / 'SELECTION_LOCK.json')
    return out


def readers_cfg(labels=None):
    out = {'readers': {}}
    sel = json.loads((PRIV / 'selection_public/SELECTION.json').read_text())
    labels = labels or list(sel['attack_roster'])
    out['service_labels'] = labels
    for s in (0, 1):
        d = PRIV / f'readers/S{s}'
        out['readers'][str(s)] = {'receipt': item(d / 'FROZEN_COMPLETE.json'), 'scores': item(d / 'INNER_SCORES.npz'),
                                  'releases': {lab: item(d / f'INNER_RELEASE__{lab}.npz') for lab in labels}}
    primary = sel.get('primary_family')
    refit = [x for x in [primary, 'J_TREE', 'RAW_PROJ', 'T_TREE'] if x]
    out['refit_services'] = list(dict.fromkeys(refit))
    return out


def collect(obj, acc):
    if isinstance(obj, dict):
        if 'path' in obj and 'sha256' in obj and isinstance(obj['path'], str):
            acc.add(obj['path'])
        for v in obj.values():
            collect(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            collect(v, acc)
    return acc


def latest_amendment():
    st = json.loads(coord.q('status'))
    latest = None
    for a in st.get('amendments', []):
        if 'verification' in a['stages']:
            latest = a['sha256']
    return latest


def enqueue(uid, stages, wall=3600, extra=None):
    assert uid.startswith('THPV_')
    cfg = base_cfg()
    need_training = any(s in stages for s in ('COUNTERFACTUALS', 'FIT2', 'WIRES', 'READERS', 'INNER'))
    if need_training:
        cfg.update(training_cfg())
    if any(s in stages for s in ('FIT2', 'WIRES', 'READERS', 'INNER')):
        cfg.update(selection_cfg())
    if any(s in stages for s in ('WIRES', 'READERS', 'INNER')):
        cfg.update(readers_cfg())
    if extra:
        cfg.update(extra)
    cfg['stages'] = list(stages)
    srcs = sorted(collect(cfg, set()) | set(PKG))
    return coord.unit(uid, 'verification', 'task_headroom_independent.worker', cfg, extra_sources=srcs, wall=wall, rss=2,
                      amendment=latest_amendment(), out_bytes=1 << 26)


def summary():
    stages = {}
    total_cpu = 0.0
    for p in sorted(OUT.glob('STAGE_*.json')):
        r = json.loads(p.read_text())
        stages[r['stage']] = {'status': r['status'], 'n_checks': r['n_checks'], 'failed_checks': r['failed_checks'],
                              'queue_unit_id': r['queue_unit_id'], 'cpu_s': r['cpu_s'],
                              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        total_cpu += r['cpu_s']
    st = json.loads(coord.q('status'))
    units = {k: {'status': v.get('status'), 'charged_cpu_s': v.get('charged_cpu_s')} for k, v in st['units'].items() if k.startswith('THPV_')}
    order = ['SOURCE', 'TEACHER', 'STAGE_A', 'COUNTERFACTUALS', 'FIT2', 'WIRES', 'READERS', 'INNER']
    res = {'schema': 'THPV-independent-verification-v1', 'campaign': 'pcrl_task_headroom_protected_outputs_v1',
           'role': 'INDEPENDENT_VERIFICATION', 'stages': stages,
           'pending_stages': [s for s in order if s not in stages],
           'overall': 'PASS' if stages and all(v['status'] == 'PASS' for v in stages.values()) else 'FAIL',
           'complete': all(s in stages for s in order), 'stage_cpu_s_total': total_cpu, 'queue_units': units,
           'queue_charged_cpu_s_total': sum((u['charged_cpu_s'] or 0) for u in units.values()),
           'independence': 'own router/metrics/AUC/bootstrap/readers/rules in task_headroom_independent; only task_headroom.evaluation_queue.queue_config imported'}
    (OUT / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(res, indent=2, sort_keys=True))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('enqueue'); e.add_argument('uid'); e.add_argument('stages'); e.add_argument('--wall', type=int, default=3600)
    e.add_argument('--drive', action='store_true'); e.add_argument('--extra', default=None)
    sub.add_parser('summary'); sub.add_parser('drive')
    a = ap.parse_args()
    if a.cmd == 'enqueue':
        print(enqueue(a.uid, a.stages.split(','), a.wall, json.loads(a.extra) if a.extra else None))
        if a.drive:
            print(coord.drive(1))
    elif a.cmd == 'drive':
        print(coord.drive(1))
    else:
        print(json.dumps(summary(), indent=1))
