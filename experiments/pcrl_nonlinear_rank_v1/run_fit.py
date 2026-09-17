"""Phase 1: fit the new interface maps and build releases. Resumable, atomic.

Selection of every checkpoint uses the training objective only. No attacker,
residence, commute, development outcome or transport table is read here — this
module cannot reach them.
"""
from __future__ import annotations

import argparse
import os
import resource
import time
from pathlib import Path

import joblib
import numpy as np

from .inputs import (OUT, Registry, array_hash, load_pools, read_json, sha_file, write_json)
from .maps import HISTORICAL_ALIAS, condition_name, fit_seed
from .objective import POLICIES

POOLS = ('representation_fit', 'source_validation', 'downstream_fit', 'downstream_validation',
         'attacker_fit', 'attacker_validation', 'test')


def limit_threads():
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      VECLIB_MAXIMUM_THREADS='1')


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def machine_state() -> dict:
    """Recorded so that a later corruption can be correlated with real pressure."""
    import subprocess
    def run(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:                                    # pragma: no cover
            return f'unavailable: {exc}'
    return {'swapusage': run('sysctl -n vm.swapusage'),
            'free_percentage': run("memory_pressure | grep -i 'free percentage'"),
            'loadavg': list(os.getloadavg()),
            'peak_rss_mb': peak_rss_mb()}


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
            z = model.transform(t, ha, arm)
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
        model, diagnostics = fit_seed(seed, registry, chunk=chunk)
        dest = out / f'seed_{seed}'
        dest.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, dest / 'maps.joblib')
        write_json(dest / 'fit_diagnostics.json', diagnostics)
        releases = build_releases(model, seed, registry, out)
        write_json(dest / 'anchor_parity.json', releases['parity'])
        record = {
            'seed': seed,
            'runtime_seconds': time.perf_counter() - tick,
            'conditions': sorted(model.arms),
            'condition_count': len(model.arms),
            'unique_new_fits': diagnostics['unique_new_fits'],
            'reused_alias_conditions': diagnostics['reused_alias_conditions'],
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
        'nominal_interface_records': 54,
        'unique_new_fits_total': sum(v['unique_new_fits'] for v in summary.values()),
        'reused_historical_conditions': sorted(HISTORICAL_ALIAS),
        'historical_alias_map': HISTORICAL_ALIAS,
        'policies': sorted(POLICIES),
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
