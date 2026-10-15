"""Per-group influence functions on INNER (already used) for confirmation planning.

Base statistics (both seeds; the two seeds share the same people, so their dependence is retained):
  U|s|task|ref|metric  -- the 24 L_TREE utility clause statistics (mean per-person loss differences);
  AUC|service|s|view   -- coalition/recipient SEX-recovery AUC of each frozen winner reader.
Influence functions: mean statistic d: IF_i = d_i - mean(d). AUC (two-class U-statistic, Hajek projection):
positive i: (V_i - AUC)/pi1 with V_i = P(s_i > S_neg) + 0.5 P(=); negative j: (W_j - AUC)/pi0. Rows are
summed within duplicate groups (each group has one SEX value) and rescaled so that
stat - truth ~= mean over groups of phi_g, i.e. Var(stat) ~= Var(phi)/G for G independent groups.
Writes private per-group phi (never published) and public class-split second moments, point estimates,
and a validation of the influence-function SEs against the original 2048-draw bootstrap SEs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

TASKS = ('income', 'occupation')
VIEWS = ('income', 'occupation', 'pair')


def auc_if(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    pos, neg = s[y == 1], s[y == 0]; n1, n0 = len(pos), len(neg); n = len(y)
    sn = np.sort(neg); sp = np.sort(pos)
    lt = np.searchsorted(sn, s, side='left'); le = np.searchsorted(sn, s, side='right')
    V = (lt + 0.5 * (le - lt)) / n0                       # for positives: P(neg < s) + .5 P(neg = s)
    gt_l = np.searchsorted(sp, s, side='right'); ge_l = np.searchsorted(sp, s, side='left')
    Wv = ((n1 - gt_l) + 0.5 * (gt_l - ge_l)) / n1          # for negatives: P(pos > s) + .5 P(pos = s)
    A = float(V[y == 1].mean())
    pi1, pi0 = n1 / n, n0 / n
    IF = np.where(y == 1, (V - A) / pi1, (Wv - A) / pi0)
    return A, IF


def run(spec_path):
    from .evaluation_queue import queue_config
    from task_headroom.data import load_npz, task_vectors, K
    from task_headroom.inference import group_weights, weighted_mean
    from task_conditioned_representation.inference import auc
    active, cfg, verify, guard = queue_config(spec_path, 'verification', allow_inner_truth=True)
    rl = json.loads(verify(cfg['reporting_lock'], cfg['reporting_lock_sha256']).read_text())
    if rl.get('status') != 'ALL_READERS_FROZEN':
        raise PermissionError('completion reporting lock not frozen')
    truth = {k: load_npz(cfg['inner_truth'][k], verify, ('row_id', 'unit', k)) for k in ('income', 'occupation', 'sex')}
    row_id, unit = truth['sex']['row_id'], truth['sex']['unit']
    y = {k: truth[k][k].astype(np.int64) for k in truth}
    n = len(row_id); groups, inv = np.unique(unit, return_inverse=True); G = len(groups)
    gsex = np.zeros(G, dtype=np.int64); gsex[inv] = y['sex']
    names, points, rows_if = [], [], []
    refs = {}
    for s in (0, 1):
        r = load_npz(cfg['inner_references'][str(s)], verify, ('row_id', 'unit', 'U0_1', 'U0_2', 'Ucal_1', 'Ucal_2', 'd_1', 'd_2'))
        refs[s] = {'U0': [r['U0_1'], r['U0_2']], 'Ucal': [r['Ucal_1'], r['Ucal_2']]}
    cand = cfg['candidate_release']
    for s in (0, 1):
        with np.load(verify(cand[str(s)]['path'], cand[str(s)]['sha256'])) as z:
            P = [z['p1'].copy(), z['p2'].copy()]
        for i, t in enumerate(TASKS):
            a = task_vectors(y[t], P[i], K[t])
            for ref in ('U0', 'Ucal'):
                r = task_vectors(y[t], refs[s][ref][i], K[t])
                for metric, d in (('accuracy_loss', r['accuracy'] - a['accuracy']), ('nll_excess', a['nll'] - r['nll']), ('brier_excess', a['brier'] - r['brier'])):
                    names.append(f'U|{s}|{t}|{ref}|{metric}'); points.append(float(d.mean())); rows_if.append(d - d.mean())
    banks = {}
    for bank in ('old', 'new'):
        for s in (0, 1):
            b = cfg['banks'][bank][str(s)]
            receipt = json.loads(verify(b['receipt']['path'], b['receipt']['sha256']).read_text())
            with np.load(verify(b['scores']['path'], b['scores']['sha256'])) as z:
                scores = {k: z[k].copy() for k in z.files if k not in ('row_id', 'unit')}
            banks[(bank, s)] = (receipt, scores)
    for lab, where in cfg['services'].items():
        for s in (0, 1):
            receipt, scores = banks[(where['bank'], s)]
            for v in VIEWS:
                win = receipt['services'][where['bank_label']]['winner'][v]['reader']
                sc = scores[f"{where['bank_label']}__{v}__{win}"]
                A, IF = auc_if(y['sex'], sc)
                if abs(A - auc(y['sex'], sc)) > 1e-12:
                    raise AssertionError('AUC influence point differs from the reference AUC')
                names.append(f'AUC|{lab}|{s}|{v}'); points.append(A); rows_if.append(IF)
    R = np.vstack(rows_if)                                        # (M, n)
    psi = np.zeros((R.shape[0], G)); np.add.at(psi.T, inv, R.T)    # group sums
    phi = psi * (G / n)                                           # stat - truth ~= mean_g phi_g
    pi1 = float(gsex.mean())
    Mpos = (phi[:, gsex == 1] @ phi[:, gsex == 1].T) / max(1, (gsex == 1).sum())
    Mneg = (phi[:, gsex == 0] @ phi[:, gsex == 0].T) / max(1, (gsex == 0).sum())
    cov = np.cov(phi, ddof=1)
    # Validation against the original bootstrap: SE of L_TREE vs RAW_PROJ coalition benefit (mean over seeds).
    W, _ = group_weights(unit)
    def idx(nm): return names.index(nm)
    a = np.zeros(len(names))
    for s in (0, 1):
        a[idx(f'AUC|RAW_PROJ|{s}|pair')] += 0.5; a[idx(f'AUC|L_TREE|{s}|pair')] -= 0.5
    se_if = float(np.sqrt(a @ cov @ a / G))
    from task_conditioned_representation.inference import bootstrap_auc
    dr = []
    for s in (0, 1):
        for lab, sgn in (('RAW_PROJ', 0.5), ('L_TREE', -0.5)):
            where = cfg['services'][lab]; receipt, scores = banks[(where['bank'], s)]
            win = receipt['services'][where['bank_label']]['winner']['pair']['reader']
            dr.append(sgn * bootstrap_auc(y['sex'], scores[f"{where['bank_label']}__pair__{win}"], W))
    se_boot = float(np.std(np.sum(dr, axis=0), ddof=1))
    u = names.index('U|1|occupation|U0|brier_excess')
    d_boot = None
    priv = Path(cfg['private_output']); priv.mkdir(parents=True, exist_ok=False, mode=0o700)
    np.savez(priv / 'PHI_PER_GROUP.npz', phi=phi, group_sex=gsex, names=np.array(names))
    out = {'schema': 'TC1-feasibility-inputs-v1', 'queue_unit_id': active['unit_id'], 'names': names, 'points': points, 'G': G, 'n_rows': n,
           'sex_prevalence_pi1': pi1, 'cov_phi': cov.tolist(), 'second_moment_pos': Mpos.tolist(), 'second_moment_neg': Mneg.tolist(),
           'validation': {'benefit_L_TREE_vs_RAW_PROJ_se_influence': se_if, 'benefit_L_TREE_vs_RAW_PROJ_se_bootstrap_2048': se_boot,
                          'ratio': se_if / se_boot, 'occupation_brier_s1_U0_point': points[u], 'occupation_brier_s1_U0_se_influence': float(np.sqrt(cov[u, u] / G))},
           'definition': __doc__, 'person_level_data_published': False}
    with Path(cfg['output']).open('x') as f:
        json.dump(out, f)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); run(p.parse_args().spec)
