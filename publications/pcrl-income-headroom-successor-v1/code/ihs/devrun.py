"""Common release format, the reference set and the development evaluation entry point used by every family.

Release file (private): runs/<FAMILY>/<config_id>/S<seed>/RELEASE.npz with keys '<ROLE>__income' (n, 2) and
'<ROLE>__occupation' (n, 5), float64 probability vectors in the admitted row order of each role, for every role the mechanism
computes from the six permitted raw fields (at least MODEL_SELECT; a promoted finalist needs ATTACK_FIT, ATTACK_VALIDATION and
DEV_REPORT too). Decisions are always the argmax. RELEASE.json beside it records the mechanism, inputs used, roles read, the
individual model-fit count and fit CPU.

    python -m ihs.devrun score --family H --version H1 --config H1_X --seed 0 --counts 1 --kind fit \
        --hypothesis "..." --change "..." --training "..." --selection "..." [--notes "..."] [--code path]

writes runs/<FAMILY>/<config>/S<seed>/EVAL_MS.json and appends one EXPERIMENT_LEDGER.csv row. MODEL_SELECT labels and SEX are
read (and logged) only here, for development evaluation.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import evaluate as EV
from . import ledger as L

REFERENCE_SET = ('U0', 'Ucal', 'T_TREE', 'TASK_REFIT', 'NEURAL_RAW', 'H_LJ', 'H_JJ', 'H_LL', 'RETIRED_B3CTL', 'B3_TASKONLY',
                 'B2TWIN', 'PUB_IBSI')
ELIGIBLE_REFERENCES = ('NEURAL_RAW', 'H_LJ')
AMD_RUN_OF = {'RETIRED_B3CTL': 'B3CTL_UNCOUPLED_MIX50_C6A', 'B3_TASKONLY': 'B3CTL_TASKONLY_TR_C6A', 'B2TWIN': 'B2CTL_UNCOUPLED_C6A'}


def release_dir(family, config_id, seed):
    return C.RUNS / family / config_id / f'S{seed}'


def save_release(family, config_id, seed, outs_by_role, record):
    d = release_dir(family, config_id, seed)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    arrays = {}
    for role, outs in outs_by_role.items():
        if role not in C.ROLES6:
            raise ValueError(role)
        for t in C.TASKS:
            p = np.ascontiguousarray(np.asarray(outs[t], dtype=np.float64))
            if p.ndim != 2 or p.shape[1] != C.K[t] or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
                raise ValueError(f'{role} {t}: not a valid probability-vector release')
            arrays[f'{role}__{t}'] = p
    with open(d / 'RELEASE.npz', 'wb') as f:
        np.savez(f, **arrays)
    rec = {**record, 'family': family, 'config_id': config_id, 'seed': seed, 'utc': C.now(), 'roles': sorted(outs_by_role),
           'release_sha256': C.sha(d / 'RELEASE.npz'), 'wire_sha256': {k: C.wire_sha(v) for k, v in arrays.items()},
           'unit_id': os.environ.get(C.UNIT_ENV)}
    (d / 'RELEASE.json').write_text(json.dumps(rec, indent=1, sort_keys=True, default=float) + '\n')
    return d


def load_release(family, config_id, seed, roles=None):
    d = release_dir(family, config_id, seed)
    rec = json.loads((d / 'RELEASE.json').read_text())
    if C.sha(d / 'RELEASE.npz') != rec['release_sha256']:
        raise PermissionError('release file differs from its record')
    out = {}
    with np.load(d / 'RELEASE.npz') as z:
        for k in z.files:
            role, t = k.split('__')
            if roles is None or role in roles:
                out.setdefault(role, {})[t] = z[k].copy()
    return out, rec


def reference_outs(seed, role='MODEL_SELECT'):
    """Frozen historical releases on one role (no fit): frozen-hybrids bank, the source campaign's development runs (retired
    B3CTL finalist and its controls) and the source final bank's PUB_IBSI."""
    bank, ids = C.frozen_bank(seed, roles=(role,))
    out = {n: {t: bank[n][role][t] for t in C.TASKS} for n in ('U0', 'Ucal', 'T_TREE', 'TASK_REFIT', 'NEURAL_RAW', 'H_LJ', 'H_JJ', 'H_LL')}
    for name, cfg in AMD_RUN_OF.items():
        out[name] = C.amd_release(cfg, seed, roles=(role,))[role]
    sb, sids = C.amd_service_bank(seed, roles=(role,))
    out['PUB_IBSI'] = {t: sb['PUB_IBSI'][role][t] for t in C.TASKS}
    rid = ids[f'row_id__{role}']
    if f'row_id__{role}' in sids and not np.array_equal(sids[f'row_id__{role}'], rid):
        raise AssertionError('service bank row order differs')
    return out, rid


def reference_eval(seed):
    """Cached MODEL_SELECT evaluation of the reference set (written by unit IHS_REFERENCES_V1)."""
    p = C.RUNS / 'REF' / f'S{seed}' / 'EVAL_MS_REFERENCES.json'
    return json.loads(p.read_text())


def compare(m, refs_eval):
    return {r: EV.benefits(m['proxy'], refs_eval[r]['proxy']) for r in ELIGIBLE_REFERENCES if r in refs_eval}


def evaluate(family, version, config_id, seed, counts, kind, hypothesis, change, training, selection, notes='', code_paths=()):
    t0 = time.process_time()
    outs, rec = load_release(family, config_id, seed, roles=('MODEL_SELECT',))
    S, refs, _ = EV.ms_context(seed, purpose=f'development evaluation {family}/{config_id}/S{seed}')
    m = EV.measure(seed, outs['MODEL_SELECT'], S, refs)
    refs_eval = reference_eval(seed)
    cmp = compare(m, refs_eval)
    m['comparisons'] = cmp
    reserve = None
    try:
        from . import reserve as R
        reserve = R.screen(m, seed)
        m['reserve'] = reserve
    except ImportError:
        pass
    d = release_dir(family, config_id, seed)
    (d / 'EVAL_MS.json').write_text(json.dumps(m, indent=1, sort_keys=True, default=float) + '\n')
    paths = [Path(__file__), Path(EV.__file__)] + [Path(p) for p in code_paths]
    row = {'family': family, 'version': version, 'config_id': config_id, 'seed': seed, 'kind': kind,
           'counts_toward_36': int(counts), 'n_model_fits': rec.get('n_model_fits', ''), 'code_sha256': L.code_sha(*paths),
           'unit_id': os.environ.get(C.UNIT_ENV, ''), 'hypothesis': hypothesis, 'change_from_previous': change,
           'roles_read': rec.get('roles_read', '') + ' | MODEL_SELECT (features, labels, SEX: development evaluation, logged)',
           'training_effort': training, 'selection_effort': selection,
           'cpu_s': round(float(rec.get('fit_cpu_s', 0.0)) + time.process_time() - t0, 1), 'notes': notes,
           **L.summarize(m, cmp, reserve)}
    n = L.append(row)
    return m, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('score')
    for a in ('--family', '--version', '--config', '--hypothesis', '--change', '--training', '--selection', '--kind'):
        e.add_argument(a, required=True)
    e.add_argument('--seed', type=int, required=True); e.add_argument('--counts', type=int, required=True)
    e.add_argument('--notes', default=''); e.add_argument('--code', action='append', default=[])
    a = ap.parse_args(argv)
    m, n = evaluate(a.family, a.version, a.config, a.seed, a.counts, a.kind, a.hypothesis, a.change, a.training, a.selection, a.notes, a.code)
    print(json.dumps({'ledger_entry': n, **L.summarize(m, m.get('comparisons'), m.get('reserve'))}, default=float))


if __name__ == '__main__':
    main()
