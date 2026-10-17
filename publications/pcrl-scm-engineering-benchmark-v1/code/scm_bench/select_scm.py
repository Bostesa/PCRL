"""Selection unit (MODEL_SELECT only; PROTOCOL.md section 7). Writes SELECTION.json (to be committed as SELECTION_LOCK.json).

Utility: the 24 held-out clauses per arm and seed against the frozen U0/Ucal MODEL_SELECT outputs. Subgroup status reported,
not gating. Privacy proxy: MODEL_SELECT split into two household halves (fixed seed, same split for every arm); OUT_LR, OUT_HGB
and LOOKUP on the pair wire, fitted on one half and scored on the other in both directions; orientation +1 by construction
(every reader estimates P(female)); reader proxy = mean of the two directions; arm proxy = max over the three readers.
Nomination follows the frozen rules. No ATTACK, DEV_REPORT or SEALED role is read.
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from acs_replication import trees
from . import common as C
from .readers import kinds, lookup_keys, fit_lookup, predict_lookup
from .selection import clauses

BETAS = (0.5, 1.0, 1.5); GAMMAS = (0.5, 1.0, 2.0); N_SHUF = 5


def proxy(fit_out, sex, unit, seed):
    from task_conditioned_representation.attacks import fit_reader, predict_reader, MODEL_SEED_BASE
    from task_conditioned_representation.inference import auc
    groups = np.unique(unit); perm = np.random.default_rng(trees.seed_for('SCM1proxysplit')).permutation(len(groups))
    halfA = np.isin(unit, groups[perm[:len(groups) // 2]]); halves = (halfA, ~halfA)
    a, b = kinds(fit_out['income'], 2)['out'], kinds(fit_out['occupation'], 5)['out']; Xp = np.column_stack((a, b))
    keys = np.array(lookup_keys(fit_out['income'], fit_out['occupation']), dtype=object)
    res = {}
    for name in ('OUT_LR', 'OUT_HGB', 'LOOKUP'):
        aucs = []
        for tr, te in ((halves[0], halves[1]), (halves[1], halves[0])):
            if name == 'LOOKUP':
                m = fit_lookup(list(keys[tr]), sex[tr]); s = predict_lookup(m, list(keys[te]))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    m = fit_reader(Xp[tr], sex[tr], name, MODEL_SEED_BASE + seed)
                s = predict_reader(m, Xp[te])
            aucs.append(float(auc(sex[te], s)))
        res[name] = {'directions': aucs, 'mean': float(np.mean(aucs))}
    res['arm_proxy'] = max(v['mean'] for v in res.values())
    return res


def run(out_path):
    for name in ('ANALYSIS_LOCK.json', 'TRAINING_COMPLETE_LOCK.json'):
        if not C.committed(C.RESULTS / name):
            raise PermissionError(f'selection requires the committed {name}')
    tl = json.loads((C.RESULTS / 'TRAINING_COMPLETE_LOCK.json').read_text())
    S = C.load_role('MODEL_SELECT', ('features', 'income', 'occupation', 'sex'))
    labels = {t: S[t] for t in C.TASKS}
    per = {}
    arms_all = None
    cells = {}
    for s in C.SEEDS:
        fp = C.PRIV / 'runs' / f'FIT_{s}' / 'FIT_SEED.json'
        if C.sha(fp) != tl['fit_seed_sha256'][f'S{s}']:
            raise PermissionError('FIT_SEED differs from TRAINING_COMPLETE_LOCK')
        cells[s] = {arm: r['final_cells'] for arm, r in json.loads(fp.read_text())['arms'].items()}
    for s in C.SEEDS:
        p = C.PRIV / 'runs' / f'FIT_{s}' / 'ARM_OUTPUTS.npz'
        if C.sha(p) != tl['arm_outputs_sha256'][f'S{s}']:
            raise PermissionError('arm outputs differ from TRAINING_COMPLETE_LOCK')
        with np.load(p) as z:
            if not np.array_equal(z['row_id__MODEL_SELECT'], S['row_id']):
                raise AssertionError('MODEL_SELECT row order differs')
            outs = {}
            for key in z.files:
                parts = key.split('__')
                if len(parts) == 3 and parts[1] == 'MODEL_SELECT':
                    outs.setdefault(parts[0], {})[parts[2]] = np.asarray(z[key], dtype=np.float64)
        bank, ids = C.frozen_bank(s, roles=('MODEL_SELECT',))
        if not np.array_equal(ids['row_id__MODEL_SELECT'], S['row_id']):
            raise AssertionError('frozen bank MODEL_SELECT row order differs')
        refs = {r: {t: bank[r]['MODEL_SELECT'][t] for t in C.TASKS} for r in ('U0', 'Ucal')}
        arms_all = sorted(outs)
        per[s] = {}
        for arm in arms_all:
            cl = clauses(labels, outs[arm], refs, s)
            own = C.task_vectors(labels['occupation'], outs[arm]['occupation'], 5)['brier']
            u0 = C.task_vectors(labels['occupation'], refs['U0']['occupation'], 5)['brier']
            sub = {f'sex{g}': float((own - u0)[S['sex'] == g].mean()) for g in (0, 1)}
            per[s][arm] = {'clauses': cl, 'point_feasible_24': all(c['point_pass'] for c in cl),
                           'max_normalized': max(c['normalized'] for c in cl), 'subgroup_occupation_brier_excess_vs_U0': sub,
                           'subgroup_pass_0.005': {g: v <= 0.005 for g, v in sub.items()}, 'proxy': proxy(outs[arm], S['sex'], S['unit'], s)}
        for name in ('NEURAL_RAW', 'H_LJ'):   # descriptive proxies for the frozen comparators, same rule
            o = {t: bank[name]['MODEL_SELECT'][t] for t in C.TASKS}
            per[s][f'FROZEN_{name}'] = {'clauses': clauses(labels, o, refs, s), 'proxy': proxy(o, S['sex'], S['unit'], s)}
    def elig(arm):
        return all(per[s][arm]['point_feasible_24'] for s in C.SEEDS)
    def mean_proxy(arm):
        return float(np.mean([per[s][arm]['proxy']['arm_proxy'] for s in C.SEEDS]))
    def worst(arm):
        return max(per[s][arm]['max_normalized'] for s in C.SEEDS)
    scm = [f'SCM_b{b}' for b in BETAS]
    eligible = [a for a in scm if elig(a)]
    if eligible:
        nominee = min(eligible, key=lambda a: (mean_proxy(a), scm.index(a))); status = 'NOMINEE_SELECTED'
    else:
        nominee = min(scm, key=lambda a: (worst(a), scm.index(a))); status = 'INELIGIBLE_DIAGNOSTIC_FALLBACK'
    b = nominee.split('_b')[1]
    twin = f'TWIN_b{b}'
    def matched(arm):   # reached the nominee's final cell count on both seeds
        return all(cells[s][arm] == cells[s][nominee] for s in C.SEEDS)
    ib = [f'IBSI_b{b}_g{g}' for g in GAMMAS]; ib_el = [a for a in ib if elig(a)]
    ibsi = min(ib_el, key=lambda a: (mean_proxy(a), ib.index(a))) if ib_el else min(ib, key=lambda a: (worst(a), ib.index(a)))
    sh = [f'SHUF_b{b}_r{r}' for r in range(N_SHUF)]; sh_el = [a for a in sh if elig(a) and matched(a)]
    shuf = min(sh_el, key=lambda a: (mean_proxy(a), sh.index(a))) if sh_el else None
    sel = {'schema': 'SCM1-selection-v1', 'utc': C.now(), 'status': status,
           'nominee': {'arm': nominee, 'model_select_point_feasible_24': elig(nominee), 'mean_proxy': mean_proxy(nominee)},
           'twin': {'arm': twin, 'model_select_point_feasible_24': elig(twin), 'mean_proxy': mean_proxy(twin)},
           'ibsi': {'arm': ibsi, 'model_select_point_feasible_24': elig(ibsi), 'mean_proxy': mean_proxy(ibsi), 'eligible_gammas': ib_el},
           'shuf': ({'arm': shuf, 'model_select_point_feasible_24': True, 'mean_proxy': mean_proxy(shuf), 'eligible': sh_el} if shuf else
                    {'arm': None, 'eligible': [], 'note': 'no eligible shuffled control at the nominee beta: the shuffled comparison is a coverage gap'}),
           'bank_arm_map': {'SCM': nominee, 'TWIN': twin, 'IBSI': ibsi, **({'SHUF': shuf} if shuf else {}), 'C0': 'C0'},
           'eligible_scm_settings': eligible,
           'final_cells': {str(s): cells[s] for s in C.SEEDS},
           'twin_matched_capacity': matched(twin), 'ibsi_matched_capacity': matched(ibsi),
           'shuffled_matched_capacity': {a: matched(a) for a in sh},
           'per_seed': {str(s): {arm: {k: v for k, v in r.items()} for arm, r in per[s].items()} for s in C.SEEDS},
           'arm_output_files_sha256': {f'runs/FIT_{s}/ARM_OUTPUTS.npz': tl['arm_outputs_sha256'][f'S{s}'] for s in C.SEEDS},
           'falsifier_F1_check': {s: per[s][twin]['proxy']['arm_proxy'] - per[s][nominee]['proxy']['arm_proxy'] for s in C.SEEDS},
           'DEV_REPORT_labels_opened': False, 'ATTACK_labels_used': False}
    Path(out_path).write_text(json.dumps(sel, indent=1, sort_keys=True, default=float))
    return sel


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args(); r = run(a.out)
    print(json.dumps({k: r[k] for k in ('status', 'nominee', 'twin', 'ibsi', 'shuf', 'eligible_scm_settings', 'falsifier_F1_check')}, indent=1, default=float))
