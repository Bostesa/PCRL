"""Phase 5: EXPLORATORY CROSS-YEAR DEVELOPMENT on the already-spent 2017 partitions.

Every number this module produces is exploratory. The 2017 study's locked
evaluation is finished and its `final_evaluation` partition is spent, so:

* this is **not** a second confirmation of anything;
* the original frozen 2017 transport result keeps its historical status and is
  neither restated nor overwritten here;
* the 2017 `final_evaluation` rows are used only as **now-exposed development
  evaluation**, and that is stated on every table.

Probes are fitted on the old 2017 fitting partition and selected on the old 2017
validation partition, exactly as the transport study did, so selection still
never sees the evaluation rows. Years are reported separately and rows are never
pooled into a larger training set.

This module deliberately does NOT call ``ev.load_final``: forging or re-using the
transport lock to read a partition whose seal is already spent would misrepresent
the evidence. It reads the partition directly through the plain loader and
labels the result exploratory.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .inputs import DEV_NAME, OUT, Registry, read_json, resolve, sha_file, write_json
from .maps import HISTORICAL_ALIAS
from .run_fit import limit_threads, machine_state

TRANSPORT_NAME = 'redesign_20260917_acs_spectral_transport_v1'


def prepare(out: Path, seed: int, registry: Registry):
    """Frozen 2018 objects with this study's maps substituted for the spectral channel."""
    import joblib
    from experiments import acs_spectral_transport_eval as ev

    # FrozenSeed loads the historical spectral maps from ev.DEV, so point there first.
    # Resolve the FILE, not the directory: this worktree has the tracked result
    # directories of the completed studies but not their ignored contents, so a
    # directory-level match lands on an empty folder.
    ev.DEV = resolve(f'results/{DEV_NAME}/seed_{seed}/maps.joblib').resolve().parents[1]
    ev.LOCAL = resolve('data/acs_spectral_transport/2017_attacker_fit.npz').resolve().parent
    frozen = ev.FrozenSeed(seed)

    model = joblib.load(out / f'seed_{seed}' / 'maps.joblib')
    conditions = tuple(c for c in sorted(model.arms) if c not in HISTORICAL_ALIAS)
    frozen.spectral = model                     # same transform(T, hA, arm) contract
    ev.SPECTRAL = tuple(sorted(model.arms))
    ev.INTERFACES = ('H',) + conditions
    # Frozen-2018 candidate records for the NEW conditions live in this study's output.
    ev.DEV = Path(out).resolve()
    for path in frozen.object_files():
        registry.add(path)
    return ev, frozen, conditions


def link_transport_H(out: Path, seed: int) -> Path:
    """Reuse the transport study's own H unit read-only; H is never refitted."""
    source = resolve(f'results/{TRANSPORT_NAME}/seed_{seed}/H')
    dest = Path(out) / 'exploratory_2017' / f'seed_{seed}' / 'H'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.symlink_to(source)
    return source


def run(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    limit_threads()
    out = Path(out)
    root = out / 'exploratory_2017'
    root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for seed in seeds:
        registry = Registry.new()
        ev, frozen, conditions = prepare(out, seed, registry)
        h_source = link_transport_H(out, seed)
        registry.add(h_source / 'audit_selection.json')
        registry.add(h_source / 'utility_selection.json')

        # 2017 fitting and validation partitions only, for releases and Mode B fitting.
        rel, labels = {}, {}
        for part in ev.FIT_PARTITIONS:
            data = ev._partition(part)
            path = root / f'seed_{seed}' / 'releases_2017' / f'{part}.npz'
            if not path.exists():
                ev.save_releases(path, ev.build_releases(frozen, data['frame']))
            rel[part] = ev.load_releases(path)
            labels[part] = data['labels']
            registry.add(registry.resolve(f'data/acs_spectral_transport/2017_{part}.npz'))

        records = {}
        for condition in conditions:
            marker = root / f'seed_{seed}' / condition / 'fit_complete.json'
            if marker.exists():
                records[condition] = read_json(marker)
                print('X2017_REUSED', seed, condition, flush=True)
                continue
            tick = time.perf_counter()
            result = ev.fit_unit(root, seed, condition, rel, labels, frozen)
            result['runtime_seconds'] = time.perf_counter() - tick
            records[condition] = result
            print('X2017_FIT', seed, condition, round(result['runtime_seconds'], 1), 's', flush=True)

        summary[str(seed)] = {'conditions': records, 'inputs': registry.dump()}
        write_json(root / f'seed_{seed}' / 'exploratory_inputs.json', registry.dump())

    write_json(root / 'EXPLORATORY_2017_FIT.json', {
        'seeds': summary, 'machine': machine_state(),
        'evaluation_status': 'EXPLORATORY CROSS-YEAR DEVELOPMENT',
        'scope_note': ('The 2017 locked evaluation is finished and its final partition is spent. '
                       'These are exploratory cross-year development numbers, not a second '
                       'confirmation. The original frozen 2017 transport result keeps its '
                       'historical status and is not restated or overwritten here.'),
        'selection_note': ('Probes fitted on the 2017 attacker/task fitting partitions and '
                           'selected on the 2017 validation partitions by minimum unweighted '
                           'validation log loss, then candidate ID. Selection never sees the '
                           'evaluation rows.'),
        'lock_note': ('ev.load_final is deliberately not called: the transport lock protects a '
                      'seal that is already spent, and reusing it would misrepresent the '
                      'evidence. The partition is read through the plain loader and every table '
                      'is labelled exploratory.'),
        'year_separation': 'Years are reported separately; rows are never pooled across years.'})
    return summary




# ------------------------------------------------------------------ exploratory scoring
def score(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    """Score the new conditions on the now-exposed 2017 partitions. Mode B only.

    Mode A is not run: it would reuse 2018-fitted attacks on a different year's
    rows, and this study has no 2018 saved observer for the new interfaces, so
    there is no frozen operational probe to transport for them. Claiming one
    would be a false equivalence with the historical interfaces' catch-up scopes.
    """
    limit_threads()
    out = Path(out)
    root = out / 'exploratory_2017'
    summary = {}
    for seed in seeds:
        registry = Registry.new()
        ev, frozen, conditions = prepare(out, seed, registry)
        link_transport_H(out, seed)
        final = ev._partition(ev.FINAL)          # seal already spent; exploratory by declaration
        rows = {}
        for condition in conditions:
            dest = root / f'seed_{seed}' / condition / 'mode_B'
            if (dest / 'complete.json').exists():
                rows[condition] = read_json(dest / 'complete.json')
                print('X2017_SCORE_REUSED', seed, condition, flush=True)
                continue
            tick = time.perf_counter()
            path = root / f'seed_{seed}' / 'releases_2017' / 'final_evaluation.npz'
            if not path.exists():
                ev.save_releases(path, ev.build_releases(frozen, final['frame']))
            release = ev.load_releases(path)
            dest.mkdir(parents=True, exist_ok=True)
            record = ev.read(root / f'seed_{seed}' / condition / 'audit_selection.json')
            util = ev.read(root / f'seed_{seed}' / condition / 'utility_selection.json')
            y, w = final['labels'], final['weights']
            from experiments.acs_transfer_heads import load_candidate, metrics
            from experiments import acs_fixed_predictions_audits as old
            cache = ev.Loaded()
            out_rows = []
            for role, candidates in util['candidates'].items():
                view, target = role.split('/')
                valid = y[target] >= 0
                for cid, meta in candidates.items():
                    p = ev.stable_predict(load_candidate(meta['dir']).predict_proba,
                                          release['wire'][condition][view][valid], meta['dir'])
                    out_rows.append({'role': 'utility', 'view': view, 'target': target,
                                     'candidate_id': cid,
                                     'selected': util['selection'][role] == cid,
                                     'scores': {'test': metrics(y[target][valid], p, 2),
                                                'test_person_weighted': metrics(
                                                    y[target][valid], p, 2, w[valid])}})
            for budget, roles in record['candidates'].items():
                for role, candidates in roles.items():
                    view, target = role.split('/')
                    valid = y[target] >= 0
                    bundle = {'wire': release['wire'][condition][view][valid]}
                    chosen = record['selections'][budget][role]
                    for cid, meta in candidates.items():
                        p = cache.predict(meta, bundle, 'final')
                        out_rows.append({
                            'role': 'audit', 'view': view, 'target': target,
                            'audit_budget': int(budget), 'candidate_id': cid,
                            'transport_origin': meta['transport_origin'], 'space': meta['space'],
                            'family': meta.get('family'),
                            'anchor_ancestor': meta['anchor_ancestor'],
                            'inherited_singleton': meta['inherited_singleton'],
                            'diagnostic_only': meta['diagnostic_only'],
                            'selected_scopes': [s for s, c in chosen.items() if c == cid],
                            'validation_log_loss': record['validation_scores'][budget][role][cid]['log_loss'],
                            'scores': {'test': metrics(y[target][valid], p, old.CLASSES[target]),
                                       'test_person_weighted': metrics(
                                           y[target][valid], p, old.CLASSES[target], w[valid])}})
            write_json(dest / 'metrics.json', {
                'seed': seed, 'condition': condition, 'mode': 'B', 'year': 2017,
                'evaluation_status': 'EXPLORATORY CROSS-YEAR DEVELOPMENT (seal already spent)',
                'raw_metrics': out_rows})
            complete = {'utc': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()),
                        'runtime_seconds': time.perf_counter() - tick, 'rows': len(out_rows),
                        'nondeterminism_events': list(ev.NONDETERMINISM_EVENTS),
                        'metrics_sha256': sha_file(dest / 'metrics.json')}
            write_json(dest / 'complete.json', complete)
            rows[condition] = complete
            print('X2017_SCORE', seed, condition, len(out_rows), 'rows',
                  round(complete['runtime_seconds'], 1), 's', flush=True)
        summary[str(seed)] = rows
    write_json(root / 'EXPLORATORY_2017_SCORES.json', {
        'seeds': summary, 'machine': machine_state(),
        'evaluation_status': 'EXPLORATORY CROSS-YEAR DEVELOPMENT',
        'mode': 'B only (fresh 2017 adaptation); Mode A not claimed for the new interfaces',
        'note': ('Predictions go through ev.stable_predict, which computes each array twice and '
                 'requires agreement, the transport study\'s own corruption guard.')})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--phase', choices=['fit', 'score'], default='fit')
    a = p.parse_args()
    (run if a.phase == 'fit' else score)(a.out, tuple(a.seeds))


if __name__ == '__main__':
    main()
