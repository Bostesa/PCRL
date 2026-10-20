"""Release storage and the global configuration budget of the PCRL new-release-mechanisms campaign.

One saved release = one configuration (family, config_id, version). The campaign allows at most 60 new seed- / training-repetition-
specific configurations, fitted matched controls included (C.CONFIG_LIMIT); `save` counts them in the append-only private ledger
.private/<campaign>/runs/CONFIG_LEDGER.jsonl and refuses the 61st. Internal cross-fitting, critic, calibration and reader fits are
recorded separately in the record (n_*_fits) and are not configurations.

    runs/<FAMILY>/<config_id>/V<v>/RELEASE.npz   '<ROLE>__income' (n, 2) and '<ROLE>__occupation' (n, 5) float64 probability vectors
                                                 for ALL six roles in the admitted row order (computed from the six permitted fields
                                                 only; decisions are the argmax). A randomized release stores a (R, n, k) stack of R
                                                 independent one-draw-per-person emitted vectors per role and task instead.
    runs/<FAMILY>/<config_id>/V<v>/RELEASE.json  the record (below) + hashes.

Required record fields:
    mechanism_card     path of the committed card under results/<campaign>/cards/ that governs this fit (checked: committed, unchanged)
    kind               'twin' | 'candidate' | 'control' | 'ablation' | 'perturbation' | 'reference_adaptation'
    description        one line: what this configuration is
    n_model_fits, n_calibrator_fits, n_crossfit_fits, n_critic_fits    integers (separate counts; not configurations)
    roles_read         which roles' labels were read (e.g. 'MODEL_FIT: income, occupation, sex; CALIBRATION: occupation')
    exposure           which people's labels reached which fitted component (one sentence)
    training_people, training_households   actual training-set size of the deployed release parameters
    fit_cpu_s          process CPU seconds of the fit
    randomized         bool; if True also 'draws' (R), 'seed_handling' (how each emitted draw is seeded) and 'noise_coupling'
                       (how the income and occupation draws of one person are coupled)
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import common as C

KINDS = ('twin', 'candidate', 'control', 'ablation', 'perturbation', 'reference_adaptation')
REQUIRED = ('mechanism_card', 'kind', 'description', 'n_model_fits', 'n_calibrator_fits', 'n_crossfit_fits', 'n_critic_fits',
            'roles_read', 'exposure', 'training_people', 'training_households', 'fit_cpu_s', 'randomized')
LEDGER = C.RUNS / 'CONFIG_LEDGER.jsonl'


def rdir(family, config_id, version):
    if not family or '/' in family or not config_id or '/' in config_id:
        raise ValueError('family and config_id must be plain names')
    return C.RUNS / family / config_id / f'V{int(version)}'


def configurations():
    if not LEDGER.is_file():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def _check_prob(p, k, what, randomized):
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    want = 3 if randomized else 2
    if p.ndim != want or p.shape[-1] != k or not np.isfinite(p).all() or (p < 0).any() or \
            not np.allclose(p.sum(-1), 1, atol=1e-9, rtol=0):
        raise ValueError(f'{what}: not a valid {"(R, n, k)" if randomized else "(n, k)"} probability-vector release with k = {k}')
    return p


def save(family, config_id, version, arrays, record):
    """Write RELEASE.npz / RELEASE.json for one configuration. Refuses overwrites and the 61st configuration."""
    if int(version) not in C.VERSIONS:
        raise ValueError(f'version must be one of {C.VERSIONS}')
    missing = [k for k in REQUIRED if k not in record]
    if missing:
        raise ValueError(f'RELEASE.json record needs {missing}')
    if record['kind'] not in KINDS:
        raise ValueError(f'kind must be one of {KINDS}')
    randomized = bool(record['randomized'])
    if randomized and not all(k in record for k in ('draws', 'seed_handling', 'noise_coupling')):
        raise ValueError('a randomized release record needs draws, seed_handling and noise_coupling')
    card = C.card_info(record['mechanism_card'])
    d = rdir(family, config_id, version)
    if (d / 'RELEASE.npz').exists():
        raise PermissionError(f'{family}/{config_id}/V{version} already exists (append-only; a revision gets a new config id)')
    used = {(e['family'], e['config_id'], e['version']) for e in configurations()}
    if (family, config_id, int(version)) not in used and len(used) >= C.CONFIG_LIMIT:
        raise PermissionError(f'configuration budget of {C.CONFIG_LIMIT} new release configurations is used')
    missing_roles = set(C.ROLES6) - set(arrays)
    if missing_roles:
        raise ValueError(f'a release covers all six roles; missing {sorted(missing_roles)}')
    out = {}
    for role in C.ROLES6:
        n = len(C.features(role)['row_id'])
        for task in C.TASKS:
            a = _check_prob(arrays[role][task], C.K[task], f'{role} {task}', randomized)
            if a.shape[-2] != n:
                raise ValueError(f'{role} {task}: {a.shape[-2]} rows, expected {n}')
            if randomized and a.shape[0] != int(record['draws']):
                raise ValueError(f'{role} {task}: {a.shape[0]} draws, record says {record["draws"]}')
            out[f'{role}__{task}'] = a
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    file_sha = C.save_npz_x(d / 'RELEASE.npz', **out)
    rec = {**record, **card, 'family': family, 'config_id': config_id, 'version': int(version), 'utc': C.now(),
           'interpreter': sys.executable, 'python': sys.version.split()[0], 'numpy': np.__version__, 'release_sha256': file_sha,
           'wire_sha256': {k: C.arr_sha(v) for k, v in out.items()}, 'unit_id': os.environ.get(C.UNIT_ENV)}
    C.write_json_x(d / 'RELEASE.json', rec)
    LEDGER.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(LEDGER, 'a') as f:
        f.write(json.dumps({'family': family, 'config_id': config_id, 'version': int(version), 'kind': record['kind'],
                            'card': card['card'], 'card_sha256': card['card_sha256'], 'release_sha256': file_sha,
                            'randomized': randomized, 'utc': rec['utc'], 'unit_id': rec['unit_id'],
                            'n_model_fits': record['n_model_fits'], 'n_calibrator_fits': record['n_calibrator_fits'],
                            'n_crossfit_fits': record['n_crossfit_fits'], 'n_critic_fits': record['n_critic_fits'],
                            'configuration_number': len(used | {(family, config_id, int(version))})}, sort_keys=True) + '\n')
    C.unit_stats(configurations_saved=1, n_model_fits=int(record['n_model_fits']), n_crossfit_fits=int(record['n_crossfit_fits']),
                 n_calibrator_fits=int(record['n_calibrator_fits']), n_critic_fits=int(record['n_critic_fits']))
    return d


def load(family, config_id, version, roles=None):
    """({role: {'income': arr, 'occupation': arr}}, record); hash-checked against RELEASE.json. Randomized: (R, n, k) stacks."""
    d = rdir(family, config_id, version)
    rec = json.loads((d / 'RELEASE.json').read_text())
    if C.sha(d / 'RELEASE.npz') != rec['release_sha256']:
        raise PermissionError(f'{family}/{config_id}/V{version}: RELEASE.npz differs from its record')
    out = {}
    with np.load(d / 'RELEASE.npz') as z:
        for k in z.files:
            role, task = k.split('__')
            if roles is None or role in roles:
                out.setdefault(role, {})[task] = z[k].copy()
    return out, rec


def exists(family, config_id, version):
    return (rdir(family, config_id, version) / 'RELEASE.json').is_file()


def selftest():
    """Invented-data checks of the validation helpers (no file written, no real data)."""
    rng = np.random.default_rng(0)
    ok = {}
    p = rng.dirichlet(np.ones(5), 10)
    ok['valid_deterministic'] = _check_prob(p, 5, 'x', False).shape == (10, 5)
    try:
        _check_prob(np.full((3, 5), 0.3), 5, 'bad', False); ok['invalid_refused'] = False
    except ValueError:
        ok['invalid_refused'] = True
    r = rng.dirichlet(np.ones(2), (4, 10))
    ok['valid_randomized'] = _check_prob(r, 2, 'x', True).shape == (4, 10, 2)
    try:
        _check_prob(r[0], 2, 'x', True); ok['randomized_needs_stack'] = False
    except ValueError:
        ok['randomized_needs_stack'] = True
    ok['all_ok'] = all(ok.values())
    return ok


if __name__ == '__main__':
    res = selftest()
    print(json.dumps(res))
    sys.exit(0 if res['all_ok'] else 1)
