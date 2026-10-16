"""Engineering smoke test on SYNTHETIC data only (no ACS value is read): interfaces and shapes of the transported
components with the ACS dimensions (income K=2, occupation K=5, critic widths 3/6/9). Shortened loops."""
from __future__ import annotations
import copy
import json
import time

import numpy as np


def main():
    t0 = time.time()
    rng = np.random.default_rng(7)
    n, d = 3000, 24
    X = rng.normal(size=(n, d)).astype(np.float32)
    sex = (X[:, 0] + rng.normal(size=n) > 0).astype(np.int64)
    inc = (X[:, 1] + 0.5 * X[:, 0] + rng.normal(size=n) > 0.5).astype(np.int64)
    occ = np.digitize(X[:, 2] + 0.3 * X[:, 0] + rng.normal(scale=0.5, size=n), [-1, -0.3, 0.3, 1]).astype(np.int64)
    unit = np.repeat(np.arange(n // 2), 2)
    names = tuple(f'f{i}' for i in range(d))
    from acs_replication import trees, leafcorr as lc, neural, neural_raw as nr
    from acs_replication.data import internal_split
    fit_rows, val_rows, moved = internal_split(unit, {'income': inc, 'occupation': occ}, 0)
    out = {'internal_split': [len(fit_rows), len(val_rows), len(moved)]}
    teachers = []
    for task, y in (('income', inc), ('occupation', occ)):
        m = trees.make_classifier(task, 4, 60, 1); m.fit(X[fit_rows], y[fit_rows], eval_set=(X[val_rows], y[val_rows]))
        t = trees.from_catboost(m, task, 4, names)
        par = trees.verify_against_catboost(t, m, X, names)
        assert par['routing_identical'] and par['decision_identical'] and par['raw_max_abs_diff'] <= 1e-9 and par['probability_max_abs_diff'] <= 1e-12, par
        assert trees.TreeTeacher(json.loads(json.dumps(t.payload()))).sha() == t.sha()
        teachers.append(t); out[f'teacher_{task}'] = {'T': t.T, 'dim': t.dim, 'parity': par}
    leaves = [t.route(X, names) for t in teachers]
    rec = [lc.Recipient(task, teachers[i], leaves[i], teachers[i].raw_from_leaves(leaves[i])) for i, task in enumerate(('income', 'occupation'))]
    probe = np.arange(512); ref = np.arange(256)
    tr = lc.Trainer(0, rec, [inc, occ], np.arange(n), sex, probe, ref)
    critics = tr.warm(); calib = tr.calibrate(critics)
    out['calibration_lambda_J_rho1'] = calib['coefficients']['1.0']['lambda_J']
    saved = {}
    old = copy.deepcopy(lc.RECIPE)
    lc.RECIPE['epochs_per_active_recipient'] = 2; lc.RECIPE['checkpoint_epochs'] = [0, 2]
    try:
        for arm in ('T_REFIT', 'L_TREE', 'J_TREE'):
            tr = lc.Trainer(0, rec, [inc, occ], np.arange(n), sex, probe, ref)
            res = lc.run_arm(tr, arm, 1.0, calib['coefficients'], lambda label, th, info, arm=arm: saved.setdefault(arm, {}).__setitem__(label, th))
            out[f'{arm}_updates'] = res['updates_per_recipient']
            th = saved[arm]['E2']
            p = teachers[1].probabilities_from_raw(teachers[1].raw_from_leaves(leaves[1], th[1]))
            assert p.shape == (n, 5) and np.allclose(p.sum(1), 1)
        z0 = saved['L_TREE']['E0']
        assert all(np.all(a == 0) for a in z0)
    finally:
        lc.RECIPE.clear(); lc.RECIPE.update(old)
    models = []
    for task, y, k in (('income', inc, 2), ('occupation', occ, 5)):
        m, info = neural.fit(X, y, k, fit_rows, val_rows, 3)
        tval, tinfo = neural.fit_temperature(neural.logits(m, X[val_rows]), y[val_rows])
        out[f'U0_{task}'] = {'best_epoch': info['best_epoch'], 'temperature': tval}
        models.append(m)
    recn = [nr.NeuralRecipient(t, m, X) for t, m in zip(('income', 'occupation'), models)]
    ntr = nr.NeuralTrainer(0, recn, [inc, occ], np.arange(n), sex, probe, ref)
    cr = ntr.warm(); nc = ntr.calibrate(cr)
    oldn = copy.deepcopy(nr.RECIPE); nr.RECIPE['epochs'] = 1; nr.RECIPE['checkpoint_epochs'] = [0, 1]
    try:
        res = nr.run_arm(ntr, 'RAW_J', 3.0, nc['coefficients'], lambda *a: None)
        out['neural_raw_steps'] = res['steps']
    finally:
        nr.RECIPE.clear(); nr.RECIPE.update(oldn)
    out['seconds'] = round(time.time() - t0, 1)
    print(json.dumps(out, default=str))


if __name__ == '__main__':
    main()
