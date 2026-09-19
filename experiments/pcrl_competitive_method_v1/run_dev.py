"""2018 development audits for this study's releases, with exact-identity deduplication.

The audit machinery is the predecessor's `run_dev_2018.evaluate_seed`, unchanged: same
attack families, fitting/selection budgets, masks, subset indices (asserted against the
historical hashes), scorer and matched-exposure scope. This module only

* adds the untouched `A0` and `J` reference releases (`ref_A0`, `ref_J`), whose audits
  must reproduce the historical matched-scope rows EXACTLY (a registered validation);
* deduplicates releases that are **bitwise identical on every pool** — identity is a
  content hash of every wire array, never an inference from similar scores;
* validates every stored probability matrix before a unit is marked complete, with at
  most three bounded retries on unchanged inputs and a quarantine record per failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.pcrl_direct_adversarial_v1 import inputs as dax
from experiments.pcrl_direct_adversarial_v1 import run_dev_2018 as dev

from .common import OUT, limit_threads, read_json, utcnow, write_json_atomic
from .run_fit_e import frozen_channel

MAX_RETRIES = 3


def selected_keys(condition_dir: Path, seed: int) -> set:
    """Every prediction key any registered selection can reach, both splits.

    Selection is recomputed with the predecessor's own `point_from_records` over every
    weighting, budget and scope, so the compact file covers the matched-exposure primary
    scope and every secondary scope the report can ask for.
    """
    import itertools
    from experiments.pcrl_direct_adversarial_v1 import report as rep
    registry = dax.Registry.new()
    prior = {r['target']: r['scores'] for r in
             read_json(registry.resolve(rep.PRIOR_PATH.format(seed=seed)))['raw_metrics']
             if r['condition'] == 'prior'}
    records = read_json(condition_dir / 'metrics.json')['raw_metrics']
    keys = set()
    for weight, budget, scope in itertools.product(rep.WEIGHTS, rep.BUDGETS, rep.SCOPES):
        point = rep.point_from_records(records, prior, rep.MAIN_SPLIT, weight, budget, scope)
        for name, row in point['selected'].items():
            role, target = name.split('/', 1)[0], None
            if role == 'utility':
                target = name.split('/', 1)[1]
                stem = f"utility/{row['view']}/{target}/None/{row['candidate_id']}"
            else:
                target = name.split('/')[-1]
                stem = f"audit/{row['view']}/{target}/{row['audit_budget']}/{row['candidate_id']}"
            keys.update({stem + '/validation', stem + '/test'})
    return keys


def compact(condition_dir: Path, seed: int) -> dict:
    """Replace predictions.npz by the selected-candidate subset; verify, record hashes.

    Disk forces this: a full audit stores ~1000 candidate matrices (~80 MB). Only the
    selected candidates are read by the report and the independent replay. The full
    file is validated by the audit stage BEFORE compaction, its sha256 is kept, and the
    kept arrays are asserted bitwise equal to the originals. Fitted attacker weights
    (`audits/fitted/**/*.pt`) are deleted after completion for the same reason; no
    stage of this study reads them (the 2017 panel refits attackers fresh).
    """
    from .common import sha_file
    path = condition_dir / 'predictions.npz'
    record_path = condition_dir / 'compaction.json'
    if record_path.exists():
        return read_json(record_path)
    original = sha_file(path)
    keys = selected_keys(condition_dir, seed)
    with np.load(path) as store:
        missing = keys - set(store.files)
        if missing:
            raise AssertionError(f'selected keys absent from predictions: {sorted(missing)[:3]}')
        kept = {k: np.array(store[k]) for k in sorted(keys)}
        total = len(store.files)
    tmp = condition_dir / 'predictions.compact.npz'
    np.savez_compressed(tmp, **kept)
    with np.load(tmp) as check, np.load(path) as store:
        if not all(np.array_equal(check[k], store[k]) for k in kept):
            raise AssertionError('compacted predictions differ from the originals')
    tmp.replace(path)
    removed = 0
    for weight in (condition_dir / 'audits').rglob('*.pt'):
        removed += weight.stat().st_size
        weight.unlink()
    record = {'original_predictions_sha256': original, 'original_entries': total,
              'kept_entries': len(kept), 'compact_predictions_sha256': sha_file(path),
              'deleted_attacker_weight_bytes': removed, 'utc': utcnow()}
    write_json_atomic(record_path, record)
    return record


def release_identity(path: Path) -> str:
    data = np.load(path)
    digest = hashlib.sha256()
    for key in sorted(data.files):
        array = np.ascontiguousarray(data[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_references(out: Path, seed: int):
    registry = dax.Registry.new()
    state = dax.load_frozen_state(seed, registry)
    for name, channel in (('ref_A0', state['channel']),
                          ('ref_J', frozen_channel(seed, registry, state, 'J')['channel'])):
        path = out / f'seed_{seed}' / 'releases' / name / 'releases.npz'
        if not path.exists():
            dax.save_release(path, dax.build_wires(channel, state['anchors']))


def audit_seed_unit(out: Path, seed: int, name: str) -> dict:
    """Audit one release with bounded retries and quarantine; no identity handling."""
    registry = dax.Registry.new()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = dev.evaluate_seed(out, seed, [name], registry)[name]
            result['compaction'] = compact(out / f'seed_{seed}' / name, seed)
            return result
        except Exception as exc:
            target = out / f'seed_{seed}' / 'quarantine' / f'{name}_attempt{attempt}'
            target.parent.mkdir(parents=True, exist_ok=True)
            source = out / f'seed_{seed}' / name
            if source.exists() and not source.is_symlink():
                shutil.move(str(source), str(target))
            target.mkdir(parents=True, exist_ok=True)
            write_json_atomic(target / 'quarantine.json', {
                'seed': seed, 'condition': name, 'attempt': attempt, 'utc': utcnow(),
                'error': repr(exc), 'traceback': traceback.format_exc(),
                'inputs_changed': False})
            print('DEV_QUARANTINE', seed, name, attempt, repr(exc)[:200], flush=True)
    return {'failed': True}


def audit_seed(out: Path, seed: int, conditions=None) -> dict:
    root = out / f'seed_{seed}' / 'releases'
    names = conditions or sorted(p.name for p in root.iterdir()
                                 if (p / 'releases.npz').exists())
    dedup_path = out / f'seed_{seed}' / 'audit_identity.json'
    identity = read_json(dedup_path) if dedup_path.exists() else {}
    registry = dax.Registry.new()
    results = {}
    for name in names:
        key = release_identity(root / name / 'releases.npz')
        canonical = next((c for c, k in identity.items()
                          if k['identity'] == key and k.get('audited_as') == c), None)
        if canonical is not None and canonical != name:
            identity[name] = {'identity': key, 'audited_as': canonical, 'duplicate': True}
            write_json_atomic(dedup_path, identity)
            results[name] = {'duplicate_of': canonical}
            print('DEV_DUPLICATE', seed, name, '->', canonical, flush=True)
            continue
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                results[name] = dev.evaluate_seed(out, seed, [name], registry)[name]
                results[name]['compaction'] = compact(out / f'seed_{seed}' / name, seed)
                break
            except Exception as exc:                          # quarantine, bounded retry
                target = out / f'seed_{seed}' / 'quarantine' / f'{name}_attempt{attempt}'
                target.parent.mkdir(parents=True, exist_ok=True)
                source = out / f'seed_{seed}' / name
                if source.exists():
                    shutil.move(str(source), str(target))
                target.mkdir(parents=True, exist_ok=True)
                write_json_atomic(target / 'quarantine.json', {
                    'seed': seed, 'condition': name, 'attempt': attempt, 'utc': utcnow(),
                    'error': repr(exc), 'traceback': traceback.format_exc(),
                    'inputs_changed': False})
                print('DEV_QUARANTINE', seed, name, attempt, repr(exc)[:200], flush=True)
                if attempt == MAX_RETRIES:
                    results[name] = {'failed': True, 'error': repr(exc)}
        if not results[name].get('failed'):
            identity[name] = {'identity': key, 'audited_as': name, 'duplicate': False}
            write_json_atomic(dedup_path, identity)
    return results


def run(out: Path = OUT, seeds=(0, 1, 2), conditions=None, references=True) -> dict:
    limit_threads()
    out = Path(out)
    summary = {}
    for seed in seeds:
        if references:
            write_references(out, seed)
        summary[str(seed)] = audit_seed(out, seed, conditions)
    write_json_atomic(out / 'logs' / f'dev_run_{int(time.time())}.json',
                      {'seeds': summary, 'utc': utcnow()})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--only', nargs='+')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.out, tuple(args.seeds), args.only)


if __name__ == '__main__':
    main()
