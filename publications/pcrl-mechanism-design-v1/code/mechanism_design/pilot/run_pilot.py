"""Entry point of the prepared MMP pilot.

    python -m mechanism_design.pilot.run_pilot --source synthetic --out <file>     # rehearsal on synthetic stand-ins (this campaign)
    python -m mechanism_design.pilot.run_pilot --source real --out <file>          # refused here: needs PILOT_AUTHORIZATION.json

The synthetic rehearsal exercises every stage with invented data and stand-in comparators (U0: logistic regression;
Ucal: its CALIBRATION temperature; NEURAL_RAW stand-in: a task-only MLP). Its numbers carry no scientific meaning.
`--rehearsal-continue-after-gates` lets the rehearsal run Stage 1 and Stage 2 even when a Stage-0 gate fails, so the
whole execution path is tested; it is refused for the real source, where any gate failure stops the pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYLIB = ROOT / '.private' / 'mechanism_design' / 'pylib'
if PYLIB.is_dir() and str(PYLIB) not in sys.path:
    sys.path.insert(0, str(PYLIB))

import numpy as np

from mechanism_design import no_real_data
from mechanism_design.pilot import config, pipeline, sources


def protocol_sha256():
    return hashlib.sha256(json.dumps(config.PILOT, sort_keys=True).encode()).hexdigest()


def synthetic_references(src, seeds):
    """U0 stand-in (logistic regression on MODEL_FIT), Ucal (one CALIBRATION temperature per task), NEURAL stand-in (task-only MLP)."""
    from scipy.optimize import minimize_scalar
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    mf = src.load_role('MODEL_FIT', ('features', 'income', 'occupation'))
    cal = src.load_role('CALIBRATION', ('features', 'income', 'occupation'))
    models = {}
    for s in seeds:
        for task in pipeline.TASKS:
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(mf['X'], mf[task])
            z = lr.decision_function(cal['X']); z = np.column_stack((-z / 2, z / 2)) if z.ndim == 1 else z
            def nll(t, z=z, y=cal[task]):
                p = np.exp((z - z.max(1, keepdims=True)) / t); p /= p.sum(1, keepdims=True)
                return -np.mean(np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)))
            temp = float(minimize_scalar(nll, bounds=(0.05, 20.0), method='bounded', options={'xatol': 1e-6, 'maxiter': 500}).x)
            nn = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=1000 + s).fit(mf['X'], mf[task])
            models[(s, task)] = (lr, temp, nn)
    cache = {}

    def probs(s, role, which):
        key = (s, role, which)
        if key not in cache:
            X = src.load_role(role)['X']; out = {}
            for task in pipeline.TASKS:
                lr, temp, nn = models[(s, task)]
                if which == 'NEURAL_STANDIN':
                    out[task] = nn.predict_proba(X)
                else:
                    z = lr.decision_function(X); z = np.column_stack((-z / 2, z / 2)) if z.ndim == 1 else z
                    t = 1.0 if which == 'U0' else temp
                    p = np.exp((z - z.max(1, keepdims=True)) / t); out[task] = p / p.sum(1, keepdims=True)
            cache[key] = out
        return cache[key]

    references = lambda s, role: {'U0': probs(s, role, 'U0'), 'Ucal': probs(s, role, 'Ucal')}
    comparators = lambda s, role: probs(s, role, 'NEURAL_STANDIN')
    return references, comparators


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if k not in ('dev_scores', 'weights', 'truth')}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return '<array omitted>'
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=('synthetic', 'real'), required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale', type=float, default=0.1)
    ap.add_argument('--max-trees', type=int, default=800)
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--rehearsal-continue-after-gates', action='store_true')
    a = ap.parse_args(argv)
    no_real_data.install()
    seeds = [int(x) for x in a.seeds.split(',')]
    t0 = time.time()
    if a.source == 'real':
        if a.rehearsal_continue_after_gates:
            raise PermissionError('gate override is refused for the real source')
        src = sources.RealSource(protocol_sha256())        # raises PermissionError in this campaign
        raise SystemExit('unreachable in this campaign')
    src = sources.SyntheticSource(scale=a.scale, seed=20261016)
    references, comparators = synthetic_references(src, seeds)
    out = {'banner': 'SYNTHETIC STAND-INS ONLY: invented data and stand-in comparators; numbers are not evidence about ACS or MMP',
           'protocol_sha256': protocol_sha256(), 'role_sizes': src.sizes, 'max_trees': a.max_trees, 'seeds': seeds}
    s0, fitted = pipeline.stage0(src, seeds, a.max_trees, references)
    out['stage0'] = s0
    if not s0['continue_to_stage1'] and not a.rehearsal_continue_after_gates:
        out['decision'] = 'NOT_SUPPORTED_AT_GATE'
    else:
        out['stage0_gate_override'] = not s0['continue_to_stage1']
        s1 = pipeline.stage1(src, seeds, fitted, comparators)
        out['stage1'] = s1
        s2 = pipeline.stage2(src, seeds, s1, references)
        s2['utility_and_subgroup'] = pipeline.dev_utility(src, seeds, fitted, references, s2['weights'], s2['critical_clause'],
                                                          s2['critical_family'], s2['truth'])
        out['stage2'] = s2
        out['decision'] = 'REHEARSAL_COMPLETED (synthetic; no scientific decision)'
    out['wall_s'] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(jsonable(out), indent=1))
    print(json.dumps({'decision': out['decision'], 'gates': s0['gates'], 'wall_s': out['wall_s']}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
