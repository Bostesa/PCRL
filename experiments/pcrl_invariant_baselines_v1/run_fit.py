"""Stage 4: fit the repaired interface maps and build releases. Resumable, atomic.

Selection of every checkpoint uses the training objective only. No attacker,
residence, commute, development outcome or transport table is read here -- this
module cannot reach them.

Two real-data validation checks run inside the fit, because they need the frozen
objects: the closed-form original-moment start is asserted **bitwise** against the
predecessor's stored arm (which was itself asserted against the historical
``spectral_L1/L2/C1``), and ``H_A``/``H_B`` parity is asserted on every released pool.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import joblib
import numpy as np

from experiments.pcrl_nonlinear_rank_v1.inputs import (Registry, array_hash, load_pools, read_json,
                                                       sha_file, write_json)
from experiments.pcrl_nonlinear_rank_v1.run_fit import limit_threads, machine_state, peak_rss_mb

from .maps import HISTORICAL_ALIAS, PREDECESSOR_CONTROLS, RANKS, condition_name, fit_seed
from .objective import POLICIES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/pcrl_invariant_baselines_v1'
POOLS = ('representation_fit', 'source_validation', 'downstream_fit', 'downstream_validation',
         'attacker_fit', 'attacker_validation', 'test')

# The predecessor's fitted objects, read-only. Its ``spectral_lin16_*`` maps were
# asserted bitwise against the historical arms in its own VALIDATION.md, so matching
# them chains this study to the historical baseline.
PREDECESSOR_ROOTS = (Path('/Users/nathansamson/PCRL-terminal-a'), Path('/Users/nathansamson/PCRL'))


def predecessor_maps(seed: int):
    for root in PREDECESSOR_ROOTS:
        path = root / f'results/pcrl_nonlinear_rank_v1/seed_{seed}/maps.joblib'
        if path.exists():
            return joblib.load(path), path
    raise FileNotFoundError(f'predecessor maps for seed {seed} not found under {PREDECESSOR_ROOTS}')


def verify_closed_form_replay(seed: int, closed_form: dict) -> dict:
    """Bitwise replay of the original-moment closed form against the stored arm.

    A sign difference is not a numerical failure -- an eigenvector is defined up to
    sign and both studies apply the same canonicalisation -- so an exact match is
    required and anything else is reported with its magnitude rather than tolerated.
    """
    model, path = predecessor_maps(seed)
    checks = []
    for rank in RANKS:
        for policy in POLICIES:
            name = condition_name('original', rank, policy)
            if name not in model.maps:
                checks.append({'condition': name, 'status': 'ABSENT_FROM_PREDECESSOR'})
                continue
            mine = closed_form[f'{rank}_{policy}']
            theirs = model.maps[name]
            checks.append({
                'condition': name,
                'aliases_historical': HISTORICAL_ALIAS.get(name),
                'bitwise_identical': bool(np.array_equal(mine, theirs)),
                'max_abs_difference': float(np.max(abs(mine - theirs))),
                'shape': list(mine.shape),
            })
    return {'source': str(path), 'source_sha256': sha_file(path), 'checks': checks,
            'all_bitwise': all(c.get('bitwise_identical') for c in checks)}


def build_releases(model, seed: int, registry: Registry, out: Path) -> dict:
    """A = [hA | Z], B = hB, AB = [A | B] for every pool and arm, with parity asserted."""
    _, teacher, anchors = load_pools(seed, registry)
    parity, written = [], {}
    for arm in model.arms:
        path = out / f'seed_{seed}' / 'releases' / arm / 'releases.npz'
        if path.exists():
            written[arm] = sha_file(path)
            continue
        cache = {}
        for pool in POOLS:
            t = np.asarray(teacher[pool], dtype=np.float64)
            ha = np.asarray(anchors[f'{pool}/A'], dtype=np.float64)
            hb = np.asarray(anchors[f'{pool}/B'], dtype=np.float64)
            z = model.transform(t, ha, arm)          # reads T and hA only: legal dependence
            wa = np.column_stack((ha, z))
            wab = np.column_stack((wa, hb))
            # H_A keeps its four original coordinates bitwise; H_B keeps its two.
            assert wa.dtype == np.float64 and np.array_equal(wa[:, :4], ha)
            assert np.array_equal(wab[:, -2:], hb) and np.array_equal(wab[:, :wa.shape[1]], wa)
            assert np.isfinite(wa).all() and np.isfinite(wab).all()
            cache[f'wire/A/{pool}'] = wa
            cache[f'wire/B/{pool}'] = hb
            cache[f'wire/AB/{pool}'] = wab
            parity.append({'arm': arm, 'pool': pool, 'rank': int(z.shape[1]),
                           'anchor_A_exact': True, 'anchor_B_exact': True,
                           'A_width': int(wa.shape[1]), 'AB_width': int(wab.shape[1]),
                           'dtype': str(wa.dtype), 'z_hash': array_hash(z)})
        path.parent.mkdir(parents=True, exist_ok=True)
        import tempfile
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.npz')
        os.close(fd)
        np.savez_compressed(tmp, **cache)
        os.replace(tmp, path)                                   # atomic
        written[arm] = sha_file(path)
    return {'parity': parity, 'release_hashes': written}


def run(out: Path = OUT, seeds=(0, 1, 2), chunk: int = 4096) -> dict:
    limit_threads()
    out = Path(out)
    summary = {}
    for seed in seeds:
        marker = out / f'seed_{seed}' / 'fit_complete.json'
        if marker.exists():
            prior = read_json(marker)
            print('FIT_REUSED', seed, prior['unique_new_fits'], flush=True)
            summary[str(seed)] = prior
            continue
        tick = time.perf_counter()
        registry = Registry.new()
        model, diagnostics, closed_form = fit_seed(seed, registry, chunk=chunk)
        replay = verify_closed_form_replay(seed, closed_form)
        if not replay['all_bitwise']:
            raise AssertionError(f'closed-form replay is not bitwise for seed {seed}: '
                                 f'{[c for c in replay["checks"] if not c.get("bitwise_identical")]}')
        dest = out / f'seed_{seed}'
        dest.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, dest / 'maps.joblib')
        write_json(dest / 'fit_diagnostics.json', diagnostics)
        write_json(dest / 'closed_form_replay.json', replay)
        releases = build_releases(model, seed, registry, out)
        write_json(dest / 'anchor_parity.json', releases['parity'])
        record = {
            'seed': seed,
            'runtime_seconds': time.perf_counter() - tick,
            'conditions': sorted(model.arms),
            'condition_count': len(model.arms),
            'unique_new_fits': diagnostics['unique_new_fits'],
            'reused_historical_conditions': diagnostics['reused_historical_conditions'],
            'reused_predecessor_controls': diagnostics['reused_predecessor_controls'],
            'closed_form_replay_bitwise': replay['all_bitwise'],
            'rank_of': {k: int(v) for k, v in model.rank_of.items()},
            'chunk_size': chunk,
            'maps_sha256': sha_file(dest / 'maps.joblib'),
            'diagnostics_sha256': sha_file(dest / 'fit_diagnostics.json'),
            'release_hashes': releases['release_hashes'],
            'inputs': registry.dump(),
            'machine': machine_state(),
            'selection_rule': 'lowest training objective across two deterministic starts and '
                              'both initial points; no outcome, attacker or label is consulted',
        }
        write_json(marker, record)
        summary[str(seed)] = record
        print('FIT_DONE', seed, round(record['runtime_seconds'], 1), 's',
              record['unique_new_fits'], 'new fits, peak RSS',
              round(record['machine']['peak_rss_mb'], 0), 'MB', flush=True)
    write_json(out / 'FIT_SUMMARY.json', {
        'seeds': summary,
        'unique_new_fits_total': sum(v['unique_new_fits'] for v in summary.values()),
        'nominal_repaired_fits': 18,
        'reused_historical_conditions': sorted(HISTORICAL_ALIAS),
        'historical_alias_map': HISTORICAL_ALIAS,
        'reused_predecessor_controls': sorted(PREDECESSOR_CONTROLS),
        'policies': sorted(POLICIES), 'ranks': list(RANKS),
        'peak_rss_mb': peak_rss_mb(),
    })
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--chunk', type=int, default=4096)
    a = p.parse_args()
    run(a.out, tuple(a.seeds), a.chunk)


if __name__ == '__main__':
    main()
