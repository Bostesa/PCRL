"""Verifier check 4 (DRF1): reader refits of a bounded sample of winning cells, and the grid's SCORE AUCs / household-bootstrap
cell SEs of the decisive comparisons recomputed from the cached reader scores with separate code.

    python3 -m drf.jobqueue run --unit DRF_VER_C4_V<k> --stage verify --rss-gib 2 -- \
        <repository>/.venv/bin/python results/pcrl_direct_recovery_frontier_v1/roles/verifier/code/check4_refits.py

(The locked grid interpreter is used so that the refits run with the library versions of DEVGRID_CODE_LOCK.json.)
Inputs: MODEL_SELECT SEX through drf.common.load('MODEL_SELECT', ('sex',), 'verify', ...); release arrays through the frozen loaders
(drf.release, frc.release, ohn.devrun.reference_release); the grid's GRID.json / BOOT.npz and its reader cache (scores only).
My own code (vlib.py): the household reader splits (SHA-256 order, round robin into tenths), the reader seeds, the cache keys, the
SCORE AUC, the household-bootstrap weights and weighted AUCs, the paired cell SEs, the orientation rule and the aggregates. The
admitted recipe code (ffv.readers / ffv.features, scm_bench.readers, and task_conditioned_representation.attacks for the source
slate's constructors, all pinned by DEVGRID_CODE_LOCK.json) is used only to refit the sampled readers.
  A. splits and seeds against READER_SPLITS.json; member wires against the grid's recorded wire hashes;
  B. per registered cell, member of the decisive set and view: the winner's cached SCORE scores -> my SCORE AUC against the record,
     my 400 bootstrap AUC draws against BOOT.npz, and the paired cell SEs of DR_AUC_R1 against each decisive comparator;
  C. the aggregate labels recomputed with my cell SEs (pair label, target label, recipient gates);
  D. refits of 6 cells chosen by a fixed rule: fit on FIT, orient on VALIDATION, score SCORE; SCORE / VALIDATION AUCs against the
     record and the refit scores against the cached scores.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import hashlib
import json
import sys
import time
import warnings
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
OUT = RES / 'roles' / 'verifier' / 'out'
GRID_VERSION = 'DRF1-devgrid-v1'
MODEL_SEED_BASE = 2026101205               # task_conditioned_representation.attacks.MODEL_SEED_BASE (RULES.md 9: 2026102305 = base + 1100)
SEED_OFFSET = 1100
BOOT_SEED = 2026102500
REP_INDEX = {'R1': 0, 'R2': 1, 'R3': 2, 'R4': 3}
DECISIVE = ('DR_AUC_R1', 'DR_MI_R1', 'DR_CTRL_LO_R1', 'DR_CTRL_HI_R1', 'TWB_C64', 'SUCC', 'NEURAL_RAW', 'H_LJ')
SLATE = {'OUT_LR': 'out', 'OUT_MLP': 'out', 'OUT_HGB': 'out', 'LOG_LR': 'log', 'LOG_MLP': 'log', 'DEC_LR': 'dec', 'CONF_HGB': 'conf'}
NEW8 = ('ET5', 'ET25', 'HGB15', 'HGB31', 'MLP64', 'MLP6432', 'CATI01', 'CATI1')
# the refit sample, fixed before any refit: (member, view, cell)
SAMPLE = (('DR_AUC_R1', 'pair', 'S0R1k0'), ('DR_AUC_R1', 'pair', 'S1R2k1'), ('DR_CTRL_HI_R1', 'pair', 'S0R2k0'),
          ('DR_MI_R1', 'pair', 'S1R1k1'), ('NEURAL_RAW', 'pair', 'S0R2k1'), ('DR_AUC_R1', 'occupation', 'S1R1k0'))


def arr_sha(a):
    z = np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes()).hexdigest()


def view_hash(view, inc, occ):
    h = hashlib.sha256(view.encode())
    for x in ((inc,) if view == 'income' else (occ,) if view == 'occupation' else (inc, occ)):
        h.update(np.ascontiguousarray(np.asarray(x, dtype=np.float64)).tobytes())
    return h.hexdigest()


def cache_path(rep, rs, wh, sex_sha):
    key = hashlib.sha256(f'{GRID_VERSION}|{rep}|{rs}|{1.0}|{wh}|{sex_sha}'.encode()).hexdigest()
    return PRIV / 'runs' / 'devgrid_cache' / f'{key}.npz'


def reader_seed(v, k):
    return MODEL_SEED_BASE + SEED_OFFSET + 10 * int(v) + int(k)


def winner_source(view, winner):
    """(view whose readers produced the winner, recipe): pair candidates are pair readers or ignore-one readers."""
    if view == 'pair' and winner.startswith('IGNORE_occupation__'):
        return 'income', winner.split('__', 1)[1]
    if view == 'pair' and winner.startswith('IGNORE_income__'):
        return 'occupation', winner.split('__', 1)[1]
    return view, winner


def load_members(spec, v):
    from drf import release as RL
    from frc import release as FRL
    from ohn import devrun as OD
    out = {}
    for name in DECISIVE:
        m = spec['members'][name]
        if m['type'] == 'drf':
            arr, _ = RL.load(m['family'], m['config'], v, roles=('MODEL_SELECT',))
            rel = arr['MODEL_SELECT']
        elif m['type'] == 'frc':
            arr, _ = FRL.load(m['family'], m['config'], v, roles=('MODEL_SELECT',))
            rel = arr['MODEL_SELECT']
        elif m['type'] == 'reference':
            rel = OD.reference_release(m['name'], v, 'MODEL_SELECT')
        else:
            raise KeyError(m)
        out[name] = {t: np.ascontiguousarray(np.asarray(rel[t], dtype=np.float64)) for t in ('income', 'occupation')}
    return out


def refit(name, view, inc, occ, sex, fit, val, rs):
    """One recipe refitted with the admitted recipe code; my orientation rule (VALIDATION AUC >= 1/2 keeps the sign)."""
    if name == 'LOOKUP':
        from scm_bench import readers as SR
        keys = SR.lookup_keys(inc) if view == 'income' else SR.lookup_keys(occ) if view == 'occupation' else SR.lookup_keys(inc, occ)
        m = SR.fit_lookup([keys[i] for i in np.flatnonzero(fit)], sex[fit])
        raw = SR.predict_lookup(m, keys)
    elif name in SLATE:
        from scm_bench import readers as SR
        from task_conditioned_representation import attacks as TA
        a, b = SR.kinds(inc, 2)[SLATE[name]], SR.kinds(occ, 5)[SLATE[name]]
        X = a if view == 'income' else b if view == 'occupation' else np.column_stack((a, b))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = TA.fit_reader(X[fit], sex[fit], name, rs)
        raw = TA.predict_reader(m, X)
    elif name in NEW8:
        from ffv import readers as FR
        tf, _, _ = FR.design_for(name, view, inc[fit], occ[fit])
        X = tf(inc, occ)
        m, _ = FR.fit_new(name, X[fit], sex[fit], rs)
        raw = FR.predict_new(m, X)
    else:
        raise KeyError(name)
    sign = 1 if V.auc(sex[val], raw[val]) >= 0.5 else -1
    s = raw if sign == 1 else 1.0 - raw
    return s, sign


def configure(plan_name):
    """The reading's decisive members and refit sample from its committed plan (generalized 04:24Z for the R2 reading; the R1
    sample is the one fixed and run in DRF_VER_C4_V1). R2 rule (fixed before any R2 output): the R1 pattern with the reading's own
    ids, the reference refit moved to S1R2k0 so that it is not the R1 refit of the same NEURAL_RAW cell."""
    global DECISIVE, SAMPLE, CAND
    plan = json.loads((RES / 'claims' / f'{plan_name}.json').read_text())
    CAND = plan['cand']
    obj = plan['obj_mi']['config_id']
    ctrl = list(plan['paths'].get('CTRL') or [])
    DECISIVE = tuple([CAND, obj] + ctrl + [plan['twin']['id'], 'SUCC', 'NEURAL_RAW', 'H_LJ'])
    if plan['plan_id'] != 'DRF_R1':
        hi = next(c for c in ctrl if 'CTRL_HI' in c)
        SAMPLE = ((CAND, 'pair', 'S0R1k0'), (CAND, 'pair', 'S1R2k1'), (hi, 'pair', 'S0R2k0'), (obj, 'pair', 'S1R1k1'),
                  ('NEURAL_RAW', 'pair', 'S1R2k0'), (CAND, 'occupation', 'S1R1k0'))
    return plan


CAND = 'DR_AUC_R1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='DRF1_R1')
    ap.add_argument('--plan', default='PLAN_DRF_R1')
    a = ap.parse_args()
    t0 = time.process_time()
    plan = configure(a.plan)
    from drf import common as C                       # data access only
    data = C.load('MODEL_SELECT', ('sex',), 'verify',
                  'independent verifier check 4: refits of sampled winning reader cells and the SCORE AUCs and household-bootstrap '
                  f'cell SEs of the decisive comparisons recomputed from the cached reader scores (grid {a.bank}, reading '
                  f'{plan["plan_id"]})')
    unit_id = os.environ.get('DRF_UNIT_ID')
    sex = np.asarray(data['sex'])
    unit = np.asarray(data['unit'])
    sex_sha = arr_sha(sex)
    items, info = [], {'unit_id': unit_id, 'utc_start': C.now(), 'sex_dtype': str(sex.dtype), 'interpreter': sys.executable,
                       'numpy': np.__version__}
    import scipy
    import sklearn
    info['scipy'], info['sklearn'] = scipy.__version__, sklearn.__version__
    man = json.loads((RES / 'READER_SPLITS.json').read_text())
    spec = json.loads((RES / 'grid_specs' / f'{a.bank}.json').read_text())
    # ---- A: splits, seeds
    roles = {}
    for rep in ('R1', 'R2', 'R3', 'R4'):
        part = V.household_parts(unit, 10, f'DRF1_devgrid_{rep}')
        r = {'FIT': part <= 4, 'VALIDATION': (part >= 5) & (part <= 6), 'SCORE': part >= 7}
        roles[rep] = r
        for role, mask in r.items():
            hh = np.unique(unit[mask])
            rec = man['repetitions'][rep][role]
            items.append(V.cmp_eq(f'A:split_{rep}_{role}', [int(mask.sum()), int(len(hh)),
                                                          hashlib.sha256(','.join(map(str, hh)).encode()).hexdigest()],
                                  [rec['people'], rec['households'], rec['household_set_sha256']]))
    forbidden = {MODEL_SEED_BASE + o + 10 * v + k for o in range(300, 1001, 100) for v in (0, 1) for k in range(10)}
    for v in (0, 1):
        for k in range(4):
            rs = reader_seed(v, k)
            items.append(V.cmp_eq(f'A:reader_seed_V{v}k{k}', rs, man['reader_seeds'][f'V{v}k{k}']))
            items.append(V.cmp_eq(f'A:reader_seed_V{v}k{k}_not_forbidden', rs in forbidden, False))
    grids, boots, members = {}, {}, {}
    for v in (0, 1):
        gp = PRIV / 'runs' / 'GRID' / a.bank / f'S{v}'
        grids[v] = json.loads((gp / 'GRID.json').read_text())
        with np.load(gp / 'BOOT.npz') as z:
            boots[v] = {k: z[k].copy() for k in z.files}
        members[v] = load_members(spec, v)
        for name, rel in members[v].items():
            rec = grids[v]['members'][name].get('wire_sha256', {})
            items.append(V.cmp_eq(f'A:V{v}:{name}:member_wires_equal_grid_record',
                                  {t: V.wire_hash(rel[t]) for t in ('income', 'occupation')}, rec))
    # ---- B: SCORE AUCs, bootstrap draws and cell SEs from the cached reader scores
    draws, recomputed = {}, {}
    for cell in V.REGISTERED:
        v, r, k = V.parse_cell(cell)
        rep = f'R{r}'
        sc = roles[rep]['SCORE']
        y = sex[sc]
        W = V.household_bootstrap_weights(unit[sc], BOOT_SEED + 100 * v + REP_INDEX[rep], B=400)
        rs = reader_seed(v, k)
        for name in DECISIVE:
            rel = members[v][name]
            for view in ('pair', 'income', 'occupation'):
                e = grids[v]['cells'][cell][name][view]
                src_view, recipe = winner_source(view, e['winner'])
                p = cache_path(rep, rs, view_hash(src_view, rel['income'], rel['occupation']), sex_sha)
                nm = f'B:{cell}:{name}:{view}'
                if not p.is_file():
                    items.append({'name': nm + ':cache_entry_missing', 'result': 'FAIL'})
                    continue
                with np.load(p, allow_pickle=False) as z:
                    s = z[f'score__{recipe}'].copy()
                    inf = json.loads(str(z['info']))
                items.append(V.cmp_eq(nm + ':cached_scores_length_equals_SCORE', int(len(s)), int(sc.sum())))
                my_auc = V.auc(y, s)
                items.append(V.cmp_num(nm + ':score_auc', my_auc, e['score_auc'], 1e-12))
                items.append(V.cmp_num(nm + ':val_auc_cache_vs_record', inf[recipe]['val_auc'], e['val_auc'], 0.0))
                d = V.weighted_auc_batch(y, s, W)
                draws[(cell, name, view)] = d
                bk = f'{cell}__{name}__{view}'
                if bk in boots[v]:
                    items.append(V.cmp_num(nm + ':bootstrap_draws_max_abs_diff', float(np.abs(d - boots[v][bk]).max()), 0.0, 1e-12))
                else:
                    items.append({'name': nm + ':boot_record_missing', 'result': 'FAIL'})
        for comp in DECISIVE[1:]:
            for view in ('pair', 'income', 'occupation'):
                if (cell, CAND, view) not in draws or (cell, comp, view) not in draws:
                    continue
                db = (draws[(cell, comp, view)] - draws[(cell, CAND, view)]) if view == 'pair' else (
                    draws[(cell, CAND, view)] - draws[(cell, comp, view)])
                se = float(np.std(db, ddof=1))
                rec = grids[v]['paired'][CAND][comp][view]['cell_se'][cell]
                items.append(V.cmp_num(f'B:{cell}:{CAND}_vs_{comp}:{view}:cell_se', se, rec, 1e-12))
                recomputed.setdefault(comp, {}).setdefault(view, {})[cell] = se
    # ---- C: aggregate labels with my cell SEs (cell values from the per-cell SCORE AUCs)
    labels = {}
    for comp in DECISIVE[1:]:
        cells = {}
        for view in ('pair', 'income', 'occupation'):
            cells[view] = {}
            for cell in V.REGISTERED:
                v = V.parse_cell(cell)[0]
                ca = grids[v]['cells'][cell][CAND][view]['score_auc']
                cb = grids[v]['cells'][cell][comp][view]['score_auc']
                cells[view][cell] = (cb - ca) if view == 'pair' else (ca - cb)
        if comp not in recomputed or any(len(recomputed[comp].get(vw, {})) != 8 for vw in ('pair', 'income', 'occupation')):
            labels[comp] = 'INCOMPLETE'
            continue
        ag = V.aggregate(cells['pair'], recomputed[comp]['pair'])
        ri = V.recipient(cells['income'], recomputed[comp]['income'])
        ro = V.recipient(cells['occupation'], recomputed[comp]['occupation'])
        labels[comp] = {'pair_label': V.benefit_label(ag), 'target_label': V.target_label(ag), 'x': ag['x'], 'se_agg': ag['se_agg'],
                        'version_means': ag['version_means'], 'income_gate': ri['gate'], 'occupation_gate': ro['gate'],
                        'income_label': ri['label'], 'occupation_label': ro['label']}
    # ---- D: refits
    refits = []
    for name, view, cell in SAMPLE:
        v, r, k = V.parse_cell(cell)
        rep = f'R{r}'
        rr = roles[rep]
        e = grids[v]['cells'][cell][name][view]
        src_view, recipe = winner_source(view, e['winner'])
        rel = members[v][name]
        rs = reader_seed(v, k)
        t1 = time.process_time()
        s, sign = refit(recipe, src_view, rel['income'], rel['occupation'], sex, rr['FIT'], rr['VALIDATION'], rs)
        cpu = time.process_time() - t1
        my_score = V.auc(sex[rr['SCORE']], s[rr['SCORE']])
        my_val = V.auc(sex[rr['VALIDATION']], s[rr['VALIDATION']])
        p = cache_path(rep, rs, view_hash(src_view, rel['income'], rel['occupation']), sex_sha)
        nm = f'D:refit:{name}:{view}:{cell}:{e["winner"]}'
        cached, inf = None, {}
        if p.is_file():
            with np.load(p, allow_pickle=False) as z:
                cached = z[f'score__{recipe}'].copy()
                inf = json.loads(str(z['info']))
        else:
            items.append({'name': nm + ':cache_entry_missing', 'result': 'FAIL'})
        items.append(V.cmp_num(nm + ':score_auc', my_score, e['score_auc'], 1e-9))
        items.append(V.cmp_num(nm + ':val_auc', my_val, e['val_auc'], 1e-9))
        items.append(V.cmp_eq(nm + ':orientation', sign, inf.get(recipe, {}).get('orientation')))
        diff = float(np.abs(s[rr['SCORE']] - cached).max()) if cached is not None else None
        refits.append({'member': name, 'view': view, 'cell': cell, 'winner': e['winner'], 'recipe': recipe, 'refit_view': src_view,
                       'reader_seed': rs, 'score_auc_refit': my_score, 'score_auc_record': e['score_auc'], 'val_auc_refit': my_val,
                       'val_auc_record': e['val_auc'], 'orientation': sign, 'max_abs_score_diff_vs_cache': diff,
                       'bitwise_equal_to_cache': (bool(np.array_equal(s[rr['SCORE']], cached)) if cached is not None else None),
                       'cpu_s': round(cpu, 2)})
    groups = {}
    for it in items:
        groups.setdefault(it['name'].split(':', 1)[0], []).append(it)
    out = {'schema': 'DRF1-verifier-check4-v1', 'info': info, 'refits': refits, 'labels_with_my_cell_se': labels,
           'summary': {g: V.summarize(x) for g, x in groups.items()}, 'overall': V.summarize(items), 'items': items,
           'cpu_s': round(time.process_time() - t0, 1), 'utc_end': C.now()}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = '' if plan['plan_id'] == 'DRF_R1' else f'_{plan["plan_id"]}'
    V.dump(OUT / f'{unit_id or "LOCAL"}_check4{tag}.json', out)
    print(json.dumps(V.jsonable({'overall': out['overall'], 'summary': out['summary'], 'refits': refits, 'labels': labels,
                                 'cpu_s': out['cpu_s']}), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
