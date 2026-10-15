"""Write verification unit configs (metadata only: paths and SHA256 copied from campaign configs) and enqueue via coord."""
from __future__ import annotations
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
PRIV = W / '.private/two_recipient_task_wire_adapter'
ROOT = PRIV / 'root'
sys.path.insert(0, str(ROOT))
import coord  # noqa: E402

RES = W / 'results/pcrl_two_recipient_task_wire_adapter_v1'
OUT = PRIV / 'independent'
PKG = sorted(str(p) for p in (W / 'two_recipient_adapter_independent').glob('*.py'))
C = {n: json.loads((ROOT / 'configs' / f'{n}.json').read_text()) for n in
     ('TRA_FIT2_UTILITY_V1', 'TRA_INNER_REPORT_V1', 'TRA_READ_S0_J_BOTH_V1', 'TRA_TRAIN_S0_V1', 'TRA_TRAIN_S1_V1', 'TRA_ADMISSION_V1')}
FIT = C['TRA_FIT2_UTILITY_V1']; REP = C['TRA_INNER_REPORT_V1']; RD = C['TRA_READ_S0_J_BOTH_V1']
SVC = json.loads((ROOT / 'ATTACK_SERVICES_PRIVATE.json').read_text())


def it(p):
    return coord.item(p)


def paths(obj, acc):
    if isinstance(obj, dict):
        if 'path' in obj and 'sha256' in obj and isinstance(obj['path'], str):
            acc.add(obj['path'])
        for v in obj.values():
            paths(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            paths(v, acc)
    return acc


def seeds_base():
    return {s: {'checkpoint': FIT['checkpoints'][s], 'geometry': {k: FIT['geometry'][s][k] for k in ('path', 'sha256')},
                'adapter_binding': FIT['adapter_bindings'][s]} for s in ('0', '1')}


def banks():
    out = {'0': {}, '1': {}}
    for k, b in REP['banks'].items():
        name, s = k.rsplit('__S', 1)
        out[s][name] = b
    return out


def enqueue(uid, module, cfg, wall=1800, inner=False):
    cfg = dict(cfg, output=str(OUT / f'{uid}.json'))
    src = sorted(paths(cfg, set()) | set(PKG))
    return coord.unit(uid, 'verification', module, cfg, extra_sources=src, wall=wall, rss=2)


def stage_a(uid='TRAV_A_V1'):
    cfg = {'operation': 'independent_stage_A', 'training_complete_lock_item': it(RES / 'TRAINING_COMPLETE_LOCK.json'),
           'selection_lock_item': it(RES / 'SELECTION_LOCK.json'), 'inner_features': {k: RD['inner_features'][k] for k in ('path', 'sha256')},
           'fit2_features': FIT['validation']['features'], 'seeds': seeds_base(), 'checkpoint_paths': FIT['checkpoint_paths'],
           'services': SVC, 'banks': banks()}
    return enqueue(uid, 'two_recipient_adapter_independent.stage_a', cfg, wall=1800)


def train_seed(s):
    t = C[f'TRA_TRAIN_S{s}_V1']
    return {'roles': t['roles']}, t


def stage_b(uid='TRAV_B_V1'):
    seeds = seeds_base()
    for s in ('0', '1'):
        seeds[s]['train_config'] = it(ROOT / 'configs' / f'TRA_TRAIN_S{s}_V1.json')
        seeds[s]['roles_inline'] = C[f'TRA_TRAIN_S{s}_V1']['roles']   # listed so every role export is a bound source
        arms = {}
        for arm in ('T_BOTH', 'L_BOTH', 'J_BOTH', 'S12', 'S21', 'I_ONLY'):
            d = PRIV / f'training/S{s}/{arm}'
            arms[arm] = {'summary': it(d / 'SUMMARY.json'), 'budget_trace': it(d / 'BUDGET_TRACE.json'), 'state_trace': it(d / 'STATE_TRACE.npz'),
                         'update_log': it(d / 'UPDATE_LOG.jsonl'), 'checkpoints': FIT['checkpoint_paths'][s][arm]}
        seeds[s]['arms'] = arms
    cfg = {'operation': 'independent_stage_B', 'seeds': seeds}
    return enqueue(uid, 'two_recipient_adapter_independent.stage_b', cfg, wall=3600)


def stage_c(uid='TRAV_C_V1'):
    seeds = seeds_base()
    for s in ('0', '1'):
        d = PRIV / f'training/S{s}/J_BOTH'
        t = C[f'TRA_TRAIN_S{s}_V1']
        seeds[s].update(train_config=it(ROOT / 'configs' / f'TRA_TRAIN_S{s}_V1.json'), roles_inline=t['roles'], role_split=t['role_split'],
                        counterfactual_states=it(d / 'COUNTERFACTUAL_STATES.pt'), summary=it(d / 'SUMMARY.json'),
                        state_trace=it(d / 'STATE_TRACE.npz'), update_log=it(d / 'UPDATE_LOG.jsonl'))
    sel = json.loads((RES / 'SELECTION_LOCK.json').read_text())
    cfg = {'operation': 'independent_stage_C', 'seeds': seeds, 'admission': C['TRA_TRAIN_S0_V1']['admission'],
           'nominated_label': sel['primary_family']['candidate_label'], 'selection_lock_item': it(RES / 'SELECTION_LOCK.json')}
    return enqueue(uid, 'two_recipient_adapter_independent.stage_c', cfg, wall=3600)


def stage_d(uid='TRAV_D_V1'):
    seeds = seeds_base()
    for s in ('0', '1'):
        seeds[s]['o_previous'] = FIT['o_previous'][s]['occupation']['checkpoint']
    cfg = {'operation': 'independent_stage_D', 'seeds': seeds, 'validation': FIT['validation'], 'checkpoint_paths': FIT['checkpoint_paths'],
           'fit2_utility': it(RES / 'FIT2_UTILITY.json'), 'selection_lock_item': it(RES / 'SELECTION_LOCK.json')}
    return enqueue(uid, 'two_recipient_adapter_independent.stage_d', cfg, wall=1800)


def stage_e(uid='TRAV_E_V1'):
    cfg = {'operation': 'independent_stage_E', 'inner_truth': REP['inner_truth'], 'inner_references': REP['inner_references'],
           'banks': {k: {'release': v['release'], 'scores': v['scores']} for k, v in REP['banks'].items()},
           'inner_report': it(RES / 'INNER_REPORT.json'), 'selection_lock_item': it(RES / 'SELECTION_LOCK.json'),
           'reporting_lock_item': it(RES / 'REPORTING_LOCK.json')}
    return enqueue(uid, 'two_recipient_adapter_independent.stage_e', cfg, wall=3600, inner=True)


def stage_f(plan, uid='TRAV_F_V1'):
    seeds = seeds_base()
    bk = banks()
    for s in ('0', '1'):
        services = {}
        for svc in ('J_BOTH', 'RAW_PROJ', 'L_BOTH'):
            ck = None if svc == 'RAW_PROJ' else FIT['checkpoint_paths'][s][svc]['E10']
            services[svc] = {'checkpoint': ck, 'arm': svc, 'label': 'E10', 'receipt': bk[s][svc]['receipt'], 'scores': bk[s][svc]['scores'],
                             'readers': plan[s][svc]['readers'], 'reasons': plan[s][svc]['reasons']}
        seeds[s]['services'] = services
    cfg = {'operation': 'independent_stage_F', 'seeds': seeds, 'att_features': {k: RD['att_features'][k] for k in ('path', 'sha256')},
           'att_sex': {k: RD['att_sex'][k] for k in ('path', 'sha256')}, 'inner_features': {k: RD['inner_features'][k] for k in ('path', 'sha256')}}
    return enqueue(uid, 'two_recipient_adapter_independent.stage_f', cfg, wall=3600)
