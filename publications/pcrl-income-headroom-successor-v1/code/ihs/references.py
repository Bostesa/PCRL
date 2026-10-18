"""Units IHS_REFERENCES_*: MODEL_SELECT development evaluation of the frozen historical releases (no fit of any release).

For one seed: the full utility vector with household-clustered paired SEs and margins (ihs.evaluate.utility) for every release
of the reference set, and the fixed 2 x 2 development reader-proxy grid for the releases named by --proxy. Writes the private
runs/REF/S<s>/EVAL_MS_REFERENCES.json and one ledger row per release (counts_toward_36 = 0, kind = reference). MODEL_SELECT
labels and SEX are read once per call through the logged loader.
"""
from __future__ import annotations
import argparse
import json
import time

from . import common as C
from . import devrun as D
from . import evaluate as EV
from . import ledger as L

PROXY_DEFAULT = ('NEURAL_RAW', 'H_LJ', 'RETIRED_B3CTL', 'B2TWIN', 'B3_TASKONLY', 'T_TREE', 'TASK_REFIT', 'H_JJ', 'H_LL', 'PUB_IBSI')


def run(seed, proxy_names=PROXY_DEFAULT):
    t0 = time.process_time()
    S, refs, _ = EV.ms_context(seed, purpose=f'reference development evaluation S{seed} (utility SEs, reader-proxy grid)')
    outs, rid = D.reference_outs(seed)
    if not (rid == S['row_id']).all():
        raise AssertionError('reference row order differs from MODEL_SELECT')
    out_path = C.RUNS / 'REF' / f'S{seed}' / 'EVAL_MS_REFERENCES.json'
    res = json.loads(out_path.read_text()) if out_path.is_file() else {}
    for name in D.REFERENCE_SET:
        t1 = time.process_time()
        m = EV.measure(seed, outs[name], S, refs, with_proxy=name in proxy_names)
        m['cpu_s'] = time.process_time() - t1
        res[name] = m
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
        print(name, round(m['cpu_s'], 1), flush=True)
    for name in D.REFERENCE_SET:
        m = res[name]
        cmp = D.compare(m, res) if 'proxy' in m else None
        L.append({'family': 'REF', 'version': 'frozen', 'config_id': name, 'seed': seed, 'kind': 'reference', 'counts_toward_36': 0,
                  'n_model_fits': 0, 'code_sha256': L.code_sha(__file__, EV.__file__, D.__file__),
                  'hypothesis': 'historical release (no fit)', 'roles_read': 'MODEL_SELECT (features, labels, SEX: development evaluation, logged)',
                  'cpu_s': round(m['cpu_s'], 1), 'decision': 'REFERENCE', **L.summarize(m, cmp)})
    print(json.dumps({'seed': seed, 'cpu_s': time.process_time() - t0}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--proxy', action='append', default=None)
    a = ap.parse_args()
    run(a.seed, tuple(a.proxy) if a.proxy else PROXY_DEFAULT)
