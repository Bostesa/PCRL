"""Stage 5b: the prespecified stress attack set.

`PROTOCOL.md` §9 fixes the set before any outcome: `J`, the historical `leace_A0`, and
the new width-16 `C1` and `L2` arms at `beta = 0.3` and `1.0`. Each receives **two
fresh audit initialisations** and a **720-epoch continuation** — double the default
catch-up budget — on the four family sensitive roles.

The stress set deliberately contains **both sides**: the competitors `J` and
`leace_A0` and this study's own arms. The larger suite is carried into every
comparison that uses it, on both sides. Strengthening the attack on a competitor alone
would be the obvious way to manufacture a win and is not done.

Selection is on the **attacker-validation** pool only. The test pool is read once, for
scoring, after the selection is written to disk.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments import acs_fixed_predictions_audits as old
from experiments.acs_preservation_audits import fit_extended_auditors
from experiments.acs_transfer_heads import load_candidate, metrics
from experiments.acs_spectral_audits import save_extended_audits
from experiments.run_acs_coalition import TARGETS as COALITION_TARGETS
from experiments.run_acs_residual_spectral import load_labels, wires
from experiments.run_acs_transfer import subset_indices

from experiments.pcrl_nonlinear_rank_v1.inputs import HIST_ROOT

from .freeze import STRESS_SET
from .inputs import OUT, Registry, read_json, resolve, sha_file, write_json
from .report import HISTORICAL_EXTERNAL, INVARIANT_NAME, condition_file
from .run_fit import limit_threads, machine_state

STRESS_ROLES = (('A', 'SEX'), ('A', 'RAC1P'), ('AB', 'SEX'), ('AB', 'RAC1P'))
DEFAULT_EPOCHS = 360
STRESS_EPOCHS = 720
# Two ADDITIONAL fresh initialisations, disjoint from the audit slate's own offsets
# (0 and 10000) and from the catch-up base. Fixed before any stress outcome.
INIT_OFFSETS = (20000, 30000)
# Slot name -> the epoch count that slot actually holds in this stage.
SLOT_EPOCHS = {'nested120': DEFAULT_EPOCHS, 'nested360': STRESS_EPOCHS}
DEV_NAME = 'redesign_20260910_acs_residual_spectral_v1'


def release_path(out: Path, seed: int, condition: str) -> Path:
    local = Path(out) / f'seed_{seed}' / 'releases' / condition / 'releases.npz'
    if local.exists():
        return local
    if condition in HISTORICAL_EXTERNAL:
        return resolve(f'results/{INVARIANT_NAME}/seed_{seed}/releases/{condition}/releases.npz')
    return resolve(f'results/redesign_20260909_acs_fixed_predictions_v1/seed_{seed}'
                   f'/training/{condition}/releases.npz')


def baseline_validation(out: Path, seed: int, condition: str) -> dict:
    """The default-budget validation log losses this interface already has."""
    path = condition_file(out, seed, condition, 'audits/audit_selection.json')
    record = read_json(path)['candidates'][str(DEFAULT_EPOCHS)]
    best = {}
    for role, candidates in record.items():
        scored = [(c['validation_scores']['log_loss'], cid) for cid, c in candidates.items()
                  if not c.get('diagnostic_only') and cid != 'saved_adversary']
        if scored:
            loss, cid = min(scored)
            best[role] = {'validation_log_loss': loss, 'candidate_id': cid}
    return best


def run_seed(out: Path, seed: int, conditions, registry: Registry) -> dict:
    frame, pools, labels, weights = load_labels(HIST_ROOT, seed)
    ai = {t: subset_indices(labels['attacker_fit'][t], 4096, 1240000 + 100 * seed + j)
          for j, t in enumerate(COALITION_TARGETS)}
    results = {}
    for condition in conditions:
        dest = Path(out) / 'stress' / f'seed_{seed}' / condition
        marker = dest / 'stress_complete.json'
        if marker.exists():
            results[condition] = read_json(marker)
            print('STRESS_REUSED', seed, condition, flush=True)
            continue
        tick = time.perf_counter()
        dest.mkdir(parents=True, exist_ok=True)
        path = release_path(out, seed, condition)
        registry.add(path)
        w, _derived = wires(path)
        baseline = baseline_validation(out, seed, condition)

        rows, selection = [], {}
        for view, target in STRESS_ROLES:
            role = f'{view}/{target}'
            classes = old.CLASSES[target]
            index = ai[target]
            valid_v = labels['attacker_validation'][target] >= 0
            valid_t = labels['test'][target] >= 0
            xf = w[view]['attacker_fit'][index]
            yf = labels['attacker_fit'][target][index]
            xv = w[view]['attacker_validation'][valid_v]
            yv = labels['attacker_validation'][target][valid_v]
            xt = w[view]['test'][valid_t]
            yt = labels['test'][target][valid_t]

            candidates = {}
            for offset in INIT_OFFSETS:
                base_seed = old.role_seed(seed, view, target) + offset
                folder = dest / 'fitted' / view / target / f'init{offset}'
                if not (folder / 'unit_complete.json').exists():
                    if folder.exists():
                        folder.rename(folder.with_name(f'incomplete_{time.time_ns()}'))
                    fitted = fit_extended_auditors(xf, yf, xv, yv, classes, base_seed,
                                                   static_candidates=None,
                                                   epochs=STRESS_EPOCHS,
                                                   nested_epochs=DEFAULT_EPOCHS)
                    save_extended_audits(fitted, folder)
                    write_json(folder / 'unit_complete.json', {
                        'complete': True, 'epochs': STRESS_EPOCHS, 'seed': base_seed,
                        'files': {str(p.relative_to(folder)): sha_file(p)
                                  for p in folder.rglob('*') if p.is_file()}})
                for name, digest in read_json(folder / 'unit_complete.json')['files'].items():
                    if sha_file(folder / name) != digest:
                        raise AssertionError(f'stress unit {folder}/{name} changed on disk')
                # `save_extended_audits` writes the two nested budgets under the fixed
                # slot names `nested120` (lower) and `nested360` (upper) whatever the
                # actual epoch counts are. Here the lower slot holds the 360-epoch
                # checkpoint and the upper slot the 720-epoch continuation.
                for tag, epochs in SLOT_EPOCHS.items():
                    directory = folder / tag
                    for cid in ('mlp_0', 'mlp_1'):
                        candidate_dir = directory / cid
                        if not (candidate_dir / 'metadata.json').exists():
                            continue
                        candidate = load_candidate(candidate_dir)
                        identifier = f'stress{offset}__{tag}__{cid}'
                        candidates[identifier] = {
                            'candidate': candidate, 'epochs': epochs, 'offset': offset,
                            'validation_log_loss':
                                candidate.metadata['validation_scores']['log_loss']}

            if not candidates:
                raise AssertionError(f'no stress candidate produced for {condition}/{role}')
            chosen = min(candidates, key=lambda k: (candidates[k]['validation_log_loss'], k))
            selection[role] = {
                'stress_selected': chosen,
                'stress_validation_log_loss': candidates[chosen]['validation_log_loss'],
                'baseline_selected': baseline.get(role, {}).get('candidate_id'),
                'baseline_validation_log_loss':
                    baseline.get(role, {}).get('validation_log_loss'),
                'candidates': {k: {'validation_log_loss': v['validation_log_loss'],
                                   'epochs': v['epochs'], 'init_offset': v['offset']}
                               for k, v in candidates.items()}}

        write_json(dest / 'selection_before_test.json', {
            'seed': seed, 'condition': condition, 'selection': selection,
            'rule': ('minimum unweighted attacker-VALIDATION log loss, then candidate id; the '
                     'test pool has not been read at this point'),
            'test_pool_read': False})

        for view, target in STRESS_ROLES:
            role = f'{view}/{target}'
            valid_t = labels['test'][target] >= 0
            xt = w[view]['test'][valid_t]
            yt = labels['test'][target][valid_t]
            index = ai[target]
            valid_v = labels['attacker_validation'][target] >= 0
            for offset in INIT_OFFSETS:
                folder = dest / 'fitted' / view / target / f'init{offset}'
                for tag in sorted(SLOT_EPOCHS):
                    for cid in ('mlp_0', 'mlp_1'):
                        candidate_dir = folder / tag / cid
                        if not (candidate_dir / 'metadata.json').exists():
                            continue
                        candidate = load_candidate(candidate_dir)
                        prediction = candidate.predict_proba(xt)
                        if not np.isfinite(prediction).all():
                            raise AssertionError('non-finite stress prediction')
                        if not np.allclose(prediction.sum(1), 1.0, atol=1e-6, rtol=0):
                            raise AssertionError('unnormalised stress prediction rows')
                        identifier = f'stress{offset}__{tag}__{cid}'
                        rows.append({
                            'seed': seed, 'condition': condition, 'role': role,
                            'candidate_id': identifier, 'init_offset': offset, 'tag': tag,
                            'selected': selection[role]['stress_selected'] == identifier,
                            'epochs': SLOT_EPOCHS[tag],
                            'validation_log_loss':
                                candidate.metadata['validation_scores']['log_loss'],
                            'scores': {
                                'test': metrics(yt, prediction, old.CLASSES[target]),
                                'test_person_weighted': metrics(
                                    yt, prediction, old.CLASSES[target],
                                    weights['test'][valid_t])}})

        write_json(dest / 'metrics.json', {
            'seed': seed, 'condition': condition, 'year': 2018,
            'evaluation_status': 'DEVELOPMENT EVALUATION, stress attack suite',
            'raw_metrics': rows})
        record = {'seed': seed, 'condition': condition,
                  'runtime_seconds': time.perf_counter() - tick,
                  'roles': len(STRESS_ROLES), 'init_offsets': list(INIT_OFFSETS),
                  'epochs': STRESS_EPOCHS, 'candidates': len(rows),
                  'metrics_sha256': sha_file(dest / 'metrics.json'),
                  'selection_sha256': sha_file(dest / 'selection_before_test.json')}
        write_json(marker, record)
        results[condition] = record
        print('STRESS_DONE', seed, condition, round(record['runtime_seconds'], 1), 's',
              flush=True)
    return results


def run(out: Path = OUT, seeds=(0, 1, 2), conditions=STRESS_SET) -> dict:
    limit_threads()
    out = Path(out)
    summary = {}
    for seed in seeds:
        registry = Registry.new()
        summary[str(seed)] = run_seed(out, seed, conditions, registry)
    write_json(out / 'STRESS_SUMMARY.json', {
        'seeds': summary, 'machine': machine_state(),
        'stress_set': list(conditions), 'roles': [f'{v}/{t}' for v, t in STRESS_ROLES],
        'epochs': STRESS_EPOCHS, 'default_epochs': DEFAULT_EPOCHS,
        'init_offsets': list(INIT_OFFSETS),
        'prespecified': ('The stress set and its roles were fixed in PROTOCOL.md 9 before any '
                         'outcome. It contains BOTH the competitors (J, leace_A0) and this '
                         "study's own arms, so the larger suite is carried into both sides of "
                         'every comparison that uses it.'),
        'selection_note': ('Selection is minimum unweighted attacker-VALIDATION log loss, then '
                           'candidate id, written to disk before the test pool is read.')})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--conditions', nargs='+', default=list(STRESS_SET))
    a = p.parse_args()
    run(a.out, tuple(a.seeds), tuple(a.conditions))


if __name__ == '__main__':
    main()
