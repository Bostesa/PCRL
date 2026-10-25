"""Verifier check 2 (DRF1): the decisive MODEL_SELECT task metrics, recomputed with separate code.

    python3 -m drf.jobqueue run --unit DRF_VER_C2_V<k> --stage verify --rss-gib 2 -- \
        python3 results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/check2_task_metrics.py

What is recomputed (verifier metric code only: vlib.py; MODEL_SELECT labels through drf.common.load('MODEL_SELECT', ..., 'verify',
...); saved release arrays through drf.release.load / frc.release.load; frozen U0 / Ucal / reference arrays through
ohn.devrun.reference_release (row order checked through ohn.common.frozen_bank), W through tsz.evaluate's frozen loaders):
  A. the 14 conditions per version (12 clauses vs U0_S<v> / Ucal_S<v> + 2 within-SEX occupation guards vs U0) of every DR
     configuration (v1 and R1 readings), the twin TWB_C64, SUCC and the other frozen references; values, household-clustered SEs and
     point passes against the evaluation records (tol 1e-10); G1 (all 28) per configuration; the twin's clause values also against
     FRC1's published utility ledger (6 decimals).
  B. integrity of the evaluated arrays: release file hash = RELEASE.json = the evaluation record; my wire hashes = the record's.
  C. every paired cost recorded in the DR evaluation records (total NLL and the 10 per-task metrics; mean and household SE).
  D. the reading plans' MODEL_SELECT distinct-input counts (three views) and exact-alias relations, recomputed from the wires.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

CAMPAIGN = 'pcrl_direct_recovery_frontier_v1'
RES = REPO / 'results' / CAMPAIGN
PRIV = REPO / '.private' / CAMPAIGN
RUNS = PRIV / 'runs'
OUT = RES / 'roles' / 'verifier' / 'out'
DR_CONFIGS = ('DR_AUC', 'DR_MI', 'DR_CTRL_LO', 'DR_CTRL_HI', 'DR_AUC_R1', 'DR_MI_R1', 'DR_CTRL_LO_R1', 'DR_CTRL_HI_R1',
              'DR_AUC_R2', 'DR_MI_R2', 'DR_CTRL_LO_R2', 'DR_CTRL_HI_R2')     # R2 added 04:23Z (only configurations with records are read)
FRC_LEDGER = REPO / 'results' / 'pcrl_finite_routing_comparison_v1' / 'dev' / 'UTILITY_LEDGER.csv'
TOL = 1e-10


class Loader:
    def __init__(self):
        from drf import release as RL
        from frc import release as FRL
        from ohn import devrun as OD
        from tsz import evaluate as TE
        self.RL, self.FRL, self.OD, self.TE = RL, FRL, OD, TE
        self.cache = {}

    def release(self, spec, v):
        key = (spec, v)
        if key in self.cache:
            return self.cache[key]
        rec = None
        if spec.startswith('DR/'):
            arr, rec = self.RL.load('DR', spec[3:], v, roles=('MODEL_SELECT',))
            rel = arr['MODEL_SELECT']
        elif spec == 'FRC:FA/TWB_C64':
            arr, rec = self.FRL.load('FA', 'TWB_C64', v, roles=('MODEL_SELECT',))
            rel = arr['MODEL_SELECT']
        elif spec.startswith('REF:'):
            rel = self.OD.reference_release(spec[4:], v, 'MODEL_SELECT')
        elif spec == 'W':
            rel = {'income': self.TE.h3_income(v), 'occupation': self.TE.source_a1(v)}
        else:
            raise KeyError(spec)
        rel = {t: np.asarray(rel[t], dtype=np.float64) for t in ('income', 'occupation')}
        self.cache[key] = (rel, rec)
        return rel, rec


def compare_utility(prefix, mine, rec_u, items):
    rc = {(c['task'], c['reference'], c['metric']): c for c in rec_u.get('clauses', [])}
    for c in mine['clauses']:
        r = rc.get((c['task'], c['reference'], c['metric']))
        nm = f'{prefix}:{c["task"]}|{c["reference"]}|{c["metric"]}'
        if r is None:
            items.append({'name': nm + ':record_missing', 'result': 'FAIL'})
            continue
        items.append(V.cmp_num(nm + ':value', c['value'], r.get('value'), TOL))
        items.append(V.cmp_num(nm + ':se', c['se'], r.get('se'), TOL))
        items.append(V.cmp_eq(nm + ':point_pass', c['point_pass'], r.get('point_pass')))
        items.append(V.cmp_num(nm + ':cap', c['cap'], r.get('cap'), 0.0))
    rg = {g['group']: g for g in rec_u.get('guards', [])}
    for g in mine['guards']:
        r = rg.get(g['group'])
        nm = f'{prefix}:guard|{g["group"]}'
        if r is None:
            items.append({'name': nm + ':record_missing', 'result': 'FAIL'})
            continue
        items.append(V.cmp_num(nm + ':value', g['value'], r.get('value'), TOL))
        items.append(V.cmp_num(nm + ':se', g['se'], r.get('se'), TOL))
        items.append(V.cmp_eq(nm + ':point_pass', g['point_pass'], r.get('point_pass')))
        items.append(V.cmp_num(nm + ':cap', g['cap'], r.get('cap'), 0.0))
    items.append(V.cmp_eq(f'{prefix}:counts_12_clauses_2_guards', [len(rec_u.get('clauses', [])), len(rec_u.get('guards', []))], [12, 2]))


def failed_list(mine):
    return ([f"{c['task']}|{c['reference']}|{c['metric']}" for c in mine['clauses'] if not c['point_pass']]
            + [f"guard|{g['group']}" for g in mine['guards'] if not g['point_pass']])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--versions', type=int, nargs='+', default=[0, 1])
    a = ap.parse_args()
    from drf import common as C                       # data access only
    data = C.load('MODEL_SELECT', ('income', 'occupation', 'sex'), 'verify',
                  'independent verifier check 2: the 28 original usefulness conditions, paired task costs and the reading plans\' '
                  'distinct-input and alias records of the DR releases (v1 and R1 readings), the twin, SUCC and the frozen references, '
                  'recomputed with separate metric code')
    res = run(data, a.versions)
    print(json.dumps(V.jsonable(res), indent=1))
    return 0                                          # failed checks are recorded in the output, never raised


def run(data, versions):
    t0 = time.process_time()
    from drf import common as C                       # data access only
    from ohn import common as OC                      # frozen-bank row order only

    class _A:
        pass
    a = _A()
    a.versions = list(versions)
    unit_id = os.environ.get('DRF_UNIT_ID')
    y = {'income': np.asarray(data['income'], dtype=np.int64), 'occupation': np.asarray(data['occupation'], dtype=np.int64)}
    sex = np.asarray(data['sex'], dtype=np.int64)
    unit = np.asarray(data['unit'])
    row_id = np.asarray(data['row_id'])
    L = Loader()
    items = []
    info = {'unit_id': unit_id, 'utc_start': C.now(), 'people': int(len(sex)), 'households': int(len(np.unique(unit))),
            'sex_counts': np.bincount(sex, minlength=2).tolist()}
    refs_rec = json.loads((RUNS / 'REFERENCES' / 'EVAL_MS.json').read_text())
    twin_rec = json.loads((RUNS / 'SOURCE_FRC' / 'FA' / 'TWB_C64' / 'EVAL_MS.json').read_text())
    dr_rec = {cid: json.loads((RUNS / 'DR' / cid / 'EVAL_MS.json').read_text()) for cid in DR_CONFIGS
              if (RUNS / 'DR' / cid / 'EVAL_MS.json').is_file()}
    info['records'] = {'references': V.sha256_file(RUNS / 'REFERENCES' / 'EVAL_MS.json'),
                       'twin': V.sha256_file(RUNS / 'SOURCE_FRC' / 'FA' / 'TWB_C64' / 'EVAL_MS.json'),
                       **{cid: V.sha256_file(RUNS / 'DR' / cid / 'EVAL_MS.json') for cid in dr_rec}}
    subjects = {cid: f'DR/{cid}' for cid in DR_CONFIGS if cid in dr_rec}
    subjects['TWB_C64'] = 'FRC:FA/TWB_C64'
    for name in refs_rec['references']:
        subjects[name] = name
    g1, per = {}, {}
    led = [r for r in csv.DictReader(open(FRC_LEDGER)) if r['config_id'] == 'TWB_C64']
    for v in a.versions:
        refs = {}
        bank, ids = OC.frozen_bank(v, roles=('MODEL_SELECT',))
        items.append(V.cmp_eq(f'A:V{v}:frozen_bank_MODEL_SELECT_row_order', bool(np.array_equal(ids['row_id__MODEL_SELECT'], row_id)), True))
        for r in ('U0', 'Ucal'):
            refs[r], _ = L.release(f'REF:{r}', v)
            items.append(V.cmp_eq(f'A:V{v}:{r}_reference_release_equals_frozen_bank',
                                  bool(all(np.array_equal(refs[r][t], bank[r]['MODEL_SELECT'][t]) for t in ('income', 'occupation'))),
                                  True))
            items.append(V.cmp_eq(f'A:V{v}:{r}_valid_probabilities', bool(V.valid_probs(refs[r]['income'], 2) and
                                                                         V.valid_probs(refs[r]['occupation'], 5)), True))
        for sid, spec in subjects.items():
            if sid in dr_rec:
                rec_v = dr_rec[sid]['versions'].get(str(v), {})
            elif sid == 'TWB_C64':
                rec_v = twin_rec['versions'].get(str(v), {})
            else:
                rec_v = refs_rec['references'][sid].get(str(v), {})
            if spec.startswith('DR/') and not (RUNS / 'DR' / spec[3:] / f'V{v}' / 'RELEASE.json').is_file():
                items.append({'name': f'A:{sid}:V{v}:release_missing', 'result': 'FAIL'})
                continue
            rel, relrec = L.release(spec, v)
            items.append(V.cmp_eq(f'A:{sid}:V{v}:valid_probabilities', bool(V.valid_probs(rel['income'], 2) and
                                                                           V.valid_probs(rel['occupation'], 5)), True))
            mine = V.conditions14(rel, refs, y, sex, unit)
            u = rec_v.get('utility', {})
            compare_utility(f'A:{sid}:V{v}', mine, u, items)
            items.append(V.cmp_eq(f'A:{sid}:V{v}:all_14_point_pass', mine['all_14_pass'], rec_v.get('point', {}).get('point_pass_12_plus_2')))
            items.append(V.cmp_eq(f'A:{sid}:V{v}:failed_list', sorted(failed_list(mine)), sorted(rec_v.get('point', {}).get('failed', []))))
            wires = {t: V.wire_hash(rel[t]) for t in ('income', 'occupation')}
            rw = rec_v.get('wire_sha256_eval') or rec_v.get('wire_sha256')
            items.append(V.cmp_eq(f'B:{sid}:V{v}:wire_hashes_equal_evaluated', wires, rw))
            if relrec is not None:
                d = (RUNS / 'DR' / spec[3:] / f'V{v}') if spec.startswith('DR/') else (
                    REPO / '.private' / 'pcrl_finite_routing_comparison_v1' / 'runs' / 'FA' / 'TWB_C64' / f'V{v}')
                items.append(V.cmp_eq(f'B:{sid}:V{v}:release_file_sha256_equals_record', V.sha256_file(d / 'RELEASE.npz'),
                                      relrec.get('release_sha256')))
                items.append(V.cmp_eq(f'B:{sid}:V{v}:evaluated_release_sha256', rec_v.get('release_sha256'), relrec.get('release_sha256')))
            per.setdefault(sid, {})[str(v)] = {
                'all_14_pass': mine['all_14_pass'], 'failed': failed_list(mine),
                'clauses': {f"{c['task']}|{c['reference']}|{c['metric']}": [c['value'], c['se']] for c in mine['clauses']},
                'guards': {g['group']: [g['value'], g['se']] for g in mine['guards']},
                'max_value_over_cap': max(x['value'] / x['cap'] for x in mine['clauses'] + mine['guards']),
                'distinct_inputs': V.distinct_counts(rel)}
            if sid == 'TWB_C64':
                row = next((r for r in led if int(r['version']) == v), None)
                for c in mine['clauses']:
                    key = f"{c['task']}|{c['reference']}|{c['metric']}"
                    items.append(V.cmp_num(f'A:TWB_C64:V{v}:{key}:frc1_published_ledger_6dp', c['value'], float(row[key]), 5.0001e-7))
                for g in mine['guards']:
                    items.append(V.cmp_num(f'A:TWB_C64:V{v}:guard|{g["group"]}:frc1_published_ledger_6dp', g['value'],
                                           float(row[f'guard|{g["group"]}']), 5.0001e-7))
        # ---- C: paired costs recorded in the DR evaluation records (and the twin's)
        for sid, rec in list(dr_rec.items()) + [('TWB_C64', twin_rec)]:
            rec_v = rec['versions'].get(str(v), {})
            if not rec_v or rec_v.get('missing'):
                continue
            a_rel, _ = L.release(subjects[sid], v)
            for spec, pc in rec_v.get('paired_costs', {}).items():
                if pc.get('missing'):
                    items.append(V.cmp_eq(f'C:{sid}:V{v}:vs_{spec}:recorded_missing',
                                          (RUNS / 'DR' / spec[3:] / f'V{v}' / 'RELEASE.json').is_file() if spec.startswith('DR/') else True,
                                          False))
                    continue
                b_rel, _ = L.release(spec, v)
                mine = V.paired_costs(a_rel, b_rel, y, sex, unit)
                for key, val in mine.items():
                    r = pc['costs'].get(key, {})
                    items.append(V.cmp_num(f'C:{sid}:V{v}:vs_{spec}:{key}:value', val['value'], r.get('value'), TOL))
                    items.append(V.cmp_num(f'C:{sid}:V{v}:vs_{spec}:{key}:se', val['se'], r.get('se'), TOL))
                    items.append(V.cmp_num(f'C:{sid}:V{v}:vs_{spec}:{key}:se_mc', 0.0, r.get('se_mc'), 0.0))
                per.setdefault(sid, {}).setdefault(str(v), {}).setdefault('paired_total_nll', {})[spec] = [
                    mine['total|nll']['value'], mine['total|nll']['se']]
    # ---- G1 per configuration (all 28 = 14 x both versions)
    for sid, by in per.items():
        vs = [by.get(str(v), {}).get('all_14_pass') for v in a.versions]
        g1[sid] = ('INCOMPLETE' if any(x is None for x in vs) else 'G1_PASS_ALL_28' if all(vs) else 'G1_FAIL')
    # ---- D: reading plans (distinct inputs, exact-alias relations) from the MODEL_SELECT wires
    plans = {}
    for pid in ('PLAN_DRF_V1', 'PLAN_DRF_R1', 'PLAN_DRF_R2'):
        p = RES / 'claims' / f'{pid}.json'
        if not p.is_file():
            continue
        plan = json.loads(p.read_text())
        plans[pid] = {'sha256': V.sha256_file(p), 'cand': plan.get('cand'), 'obj_mi': plan.get('obj_mi', {}).get('config_id'),
                      'paths': plan.get('paths'), 'reader_stage': plan.get('reader_stage')}
        specs = dict(plan.get('compare_specs', {}))
        specs.setdefault(plan.get('twin', {}).get('id', 'TWB_C64'), 'FRC:FA/TWB_C64')
        for key, rec in plan.get('distinct_inputs_model_select', {}).items():
            name, vv = key.split('|V')
            spec = specs.get(name, f'DR/{name}')
            rel, _ = L.release(spec, int(vv))
            items.append(V.cmp_eq(f'D:{pid}:distinct_inputs:{key}', V.distinct_counts(rel), rec))
        cand, obj = plan.get('cand'), plan.get('obj_mi', {}).get('config_id')
        twin = plan.get('twin', {}).get('id', 'TWB_C64')
        for vv, rel_rec in plan.get('alias_records', {}).items():
            if not str(vv).isdigit():
                continue
            vv = int(vv)

            def eq(x, y_):
                ra, _ = L.release(specs.get(x, f'DR/{x}'), vv)
                rb, _ = L.release(specs.get(y_, f'DR/{y_}'), vv)
                return bool(all(np.array_equal(ra[t], rb[t]) for t in ('income', 'occupation')))
            mine = {'cand_equals_twin': eq(cand, twin), 'cand_equals_obj_mi': eq(cand, obj), 'obj_mi_equals_twin': eq(obj, twin)}
            for c in plan.get('paths', {}).get('CTRL', []):
                if (RUNS / 'DR' / c / f'V{vv}' / 'RELEASE.json').is_file():
                    mine[f'{c}_equals_twin'] = eq(c, twin)
                    mine[f'{c}_equals_cand'] = eq(c, cand)
            for k_, val in rel_rec.items():
                items.append(V.cmp_eq(f'D:{pid}:alias:V{vv}:{k_}', mine.get(k_), val))
        alias_all = plan.get('alias_records', {})
        for k_ in ('cand_alias_of_twin', 'cand_alias_of_obj_mi'):
            if k_ in alias_all:
                key = 'cand_equals_twin' if k_ == 'cand_alias_of_twin' else 'cand_equals_obj_mi'
                both = all(alias_all.get(str(vv), {}).get(key) for vv in a.versions)
                items.append(V.cmp_eq(f'D:{pid}:alias:{k_}_both_versions', bool(both), alias_all[k_]))
    groups = {}
    for it in items:
        groups.setdefault(it['name'].split(':', 1)[0], []).append(it)
    out = {'schema': 'DRF1-verifier-check2-v1', 'info': info, 'g1': g1, 'per_subject': per, 'plans': plans,
           'summary': {g: V.summarize(x) for g, x in groups.items()}, 'overall': V.summarize(items), 'items': items,
           'cpu_s': round(time.process_time() - t0, 1), 'utc_end': C.now()}
    OUT.mkdir(parents=True, exist_ok=True)
    V.dump(OUT / f'{unit_id or "LOCAL"}_check2.json', out)
    return {'overall': out['overall'], 'summary': out['summary'], 'g1': g1, 'cpu_s': out['cpu_s']}


if __name__ == '__main__':
    sys.exit(main())
