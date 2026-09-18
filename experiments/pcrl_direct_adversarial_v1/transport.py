"""Stages 2 and 6: cross-year transport of 2018-fitted interfaces onto 2017.

Two jobs in one module because they share every mechanism:

* **Stage 2, baseline transport completion.** The `leace_A0`, `splince_A0` and
  `optnet16_{L1,L2,C1}` adaptations of `73903b7f` were never evaluated on 2017. This
  completes those **15** interfaces. Their 2018-fitted transformations are reused; no
  new 2017 eraser or encoder is fitted and then called transported.
* **Stage 6, the new 2017 panel.** The declared 54 new interfaces of `PROTOCOL.md` §10.

**Every number produced here is EXPLORATORY CROSS-YEAR DEVELOPMENT.** The 2017 locked
evaluation is finished and its `final_evaluation` partition is spent. The original
frozen 2017 transport result keeps its historical status and is neither restated nor
overwritten. 2017 is **not** a new test merely because these methods had not been
scored there.

### Reconstruction and identity

The invariant study persisted neither the erasure affine maps nor (in its production
run) the OptNet encoders. Both stages are deterministic, so both are reconstructed
here — and **neither is permitted to inherit the old identity until it is proved**. The
proof is the strongest available: the reconstruction rebuilds the 2018 release and the
result must match the stored 2018 `releases.npz` **bitwise**, on all seven pools. A
mismatch is recorded as a **new fit** and is reported under a new name.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from experiments.pcrl_invariant_baselines_v1.erasure_baselines import (
    PROTECTED_SCHEMA, SPLINCE_CONDITION_MAX, _stabilized_leace, apply_affine, joint_onehot)

from .channel import AdversarialChannel, build_channel
from .inputs import (H_A_WIDTH, OUT, POOLS, Registry, array_hash, arrays, load_frozen_state,
                     read_json, representation_labels, resolve, sha_file, standardize, write_json)
from .run_fit import WIDTHS, limit_threads, machine_state

INVARIANT_NAME = 'pcrl_invariant_baselines_v1'
TRANSPORT_NAME = 'redesign_20260917_acs_spectral_transport_v1'
DEV_NAME = 'redesign_20260910_acs_residual_spectral_v1'

HISTORICAL_ERASURE = ('leace_A0', 'splince_A0')
HISTORICAL_OPTNET = ('optnet16_L1', 'optnet16_L2', 'optnet16_C1')


# ------------------------------------------------------------------ identity proofs
def stored_2018_channel(arm: str, seed: int) -> dict:
    """The invariant study's stored 2018 release, per pool, auxiliary columns only."""
    path = resolve(f'results/{INVARIANT_NAME}/seed_{seed}/releases/{arm}/releases.npz')
    data = arrays(path)
    return {'path': str(path), 'sha256': sha_file(path),
            'channel': {pool: np.ascontiguousarray(
                np.asarray(data[f'wire/A/{pool}'], np.float64)[:, H_A_WIDTH:]) for pool in POOLS}}


def reconstruct_erasure(seed: int, registry: Registry) -> dict:
    """Rebuild the 2018 LEACE / SPLINCE affine maps on the frozen A0 channel, and prove them."""
    state = load_frozen_state(seed, registry)
    labels = representation_labels(seed, registry)['protected']
    x = state['channel']['representation_fit']
    z, complete, _coverage = joint_onehot(labels, len(x))

    out = {}
    fitted = _stabilized_leace(x[complete], z)
    out['leace_A0'] = {'projection': fitted['projection'], 'mean': fitted['mean'],
                       'realised_projection_rank': fitted['projection_retained_rank']}

    from pcrl.baselines.splince import fit_splince as splince_fit
    from .erasure import authorised_task_matrix
    y, _ = authorised_task_matrix(seed, len(x))
    concepts = [np.asarray(labels[name])[complete] for name in PROTECTED_SCHEMA]
    projection, mean, info = splince_fit(x[complete], concepts, y[complete],
                                         cond_max=SPLINCE_CONDITION_MAX)
    record = info.to_dict() if hasattr(info, 'to_dict') else dict(info.__dict__)
    if bool(record.get('fallback_to_leace')) or not record.get('feasible', True):
        out['splince_A0'] = None
    else:
        out['splince_A0'] = {'projection': projection, 'mean': mean,
                             'realised_projection_rank': int(
                                 np.linalg.matrix_rank(projection, tol=1e-10))}

    proofs = {}
    for arm in HISTORICAL_ERASURE:
        if out.get(arm) is None:
            proofs[arm] = {'status': 'SCOPED INFEASIBLE', 'identity_proved': False}
            continue
        stored = stored_2018_channel(arm, seed)
        exact, worst = True, 0.0
        for pool in POOLS:
            rebuilt = apply_affine(state['channel'][pool], out[arm]['projection'], out[arm]['mean'])
            same = bool(np.array_equal(rebuilt, stored['channel'][pool]))
            worst = max(worst, float(np.abs(rebuilt - stored['channel'][pool]).max()))
            exact = exact and same
        proofs[arm] = {
            'status': 'IDENTITY PROVED' if exact else 'RECONSTRUCTION MISMATCH -- NEW FIT',
            'identity_proved': exact,
            'bitwise_identical_all_pools': exact,
            'max_abs_difference': worst,
            'stored_2018_release': stored['path'], 'stored_2018_sha256': stored['sha256'],
            'realised_projection_rank': out[arm]['realised_projection_rank'],
            'rule': ('the reconstruction rebuilds the 2018 release and must match it bitwise on '
                     'all seven pools; a mismatch is a NEW fit and may not inherit the old '
                     'identity')}
    return {'maps': out, 'proofs': proofs, 'state': state}


def reconstruct_optnet(out: Path, seed: int) -> dict:
    """Re-run the deterministic OptNet stage at its recorded budget and prove the identity."""
    from experiments.pcrl_invariant_baselines_v1 import optnet_arl

    recorded = read_json(resolve(f'results/{INVARIANT_NAME}/seed_{seed}/optnet_complete.json'))
    budget = int(recorded['budget_steps'])
    dest = Path(out) / 'optnet_reconstruction'
    encoder_path = dest / f'seed_{seed}' / 'optnet_encoders.joblib'
    if not encoder_path.exists():
        optnet_arl.run(out=dest, seeds=(seed,), budget=budget)
    encoders = joblib.load(encoder_path)

    proofs = {}
    for arm in HISTORICAL_OPTNET:
        stored = stored_2018_channel(arm, seed)
        rebuilt_path = dest / f'seed_{seed}' / 'releases' / arm / 'releases.npz'
        rebuilt = arrays(rebuilt_path)
        exact, worst = True, 0.0
        for pool in POOLS:
            value = np.ascontiguousarray(
                np.asarray(rebuilt[f'wire/A/{pool}'], np.float64)[:, H_A_WIDTH:])
            exact = exact and bool(np.array_equal(value, stored['channel'][pool]))
            worst = max(worst, float(np.abs(value - stored['channel'][pool]).max()))
        proofs[arm] = {
            'status': 'IDENTITY PROVED' if exact else 'RECONSTRUCTION MISMATCH -- NEW FIT',
            'identity_proved': exact, 'bitwise_identical_all_pools': exact,
            'max_abs_difference': worst, 'recorded_budget_steps': budget,
            'stored_2018_release': stored['path'], 'stored_2018_sha256': stored['sha256']}
    return {'encoders': encoders, 'proofs': proofs}


# ------------------------------------------------------------------ transport contracts
class TransportModel:
    """One `transform(T, hA, arm)` contract over every interface this study transports."""

    def __init__(self, spectral, a0, channels: dict, affine: dict, encoders: dict):
        self.spectral = spectral
        self.a0 = a0
        self.channels = channels          # arm -> AdversarialChannel on standardised PCA
        self.affine = affine              # arm -> {'base': arm-or-A0, 'projection', 'mean'}
        self.encoders = encoders          # arm -> OptNet encoder on whitened V
        self.maps = {name: None for name in (*channels, *affine, *encoders)}

    def features(self, t, ha):
        return self.spectral.features(t, ha)

    def _base_channel(self, t, base) -> np.ndarray:
        if base == 'A0':
            with torch.no_grad():
                x = standardize(t, self.a0['input_mean'], self.a0['input_scale'])
                return self.a0['mapper'](x).numpy().astype(np.float64)
        return self.channels[base].release(
            standardize(t, self.a0['input_mean'], self.a0['input_scale']))

    def transform(self, t, ha, arm) -> np.ndarray:
        if arm in self.channels:
            return self.channels[arm].release(
                standardize(t, self.a0['input_mean'], self.a0['input_scale']))
        if arm in self.affine:
            spec = self.affine[arm]
            return apply_affine(self._base_channel(t, spec['base']),
                                spec['projection'], spec['mean'])
        if arm in self.encoders:
            with torch.no_grad():
                v = np.ascontiguousarray(self.features(t, ha))
                return self.encoders[arm](torch.from_numpy(v)).numpy().astype(np.float64)
        raise KeyError(arm)

    @property
    def arms(self):
        return tuple(sorted(self.maps))


def load_new_channels(out: Path, seed: int, arms, a0, reference) -> dict:
    channels = {}
    for arm in arms:
        path = Path(out) / f'seed_{seed}' / 'fits' / arm / 'checkpoints.pt'
        if not path.exists():
            continue
        width = 8 if arm.startswith('dax8') else 16
        model = build_channel(width, a0, reference)['model']
        model.load_state_dict(torch.load(path, map_location='cpu',
                                         weights_only=False)['selected_state'])
        channels[arm] = model.eval()
    return channels


def load_new_erasure_affine(out: Path, seed: int, registry: Registry) -> dict:
    """Refit the new erasure maps on the fitted no-protection channels, for transport."""
    from .erasure import fit_leace, fit_splince
    affine = {}
    for width in WIDTHS:
        for method, fitter in (('leace', fit_leace), ('splince', fit_splince)):
            name = f'{method}_dax{width}_none'
            fitted = fitter(out, seed, width, registry)
            if fitted['infeasible']:
                continue
            affine[name] = {'base': f'dax{width}_none', 'projection': fitted['projection'],
                            'mean': fitted['mean']}
    return affine


def prepare(out: Path, seed: int, registry: Registry, new_arms, include_baselines: bool):
    from experiments import acs_spectral_transport_eval as ev

    ev.DEV = resolve(f'results/{DEV_NAME}/seed_{seed}/maps.joblib').resolve().parents[1]
    ev.LOCAL = resolve('data/acs_spectral_transport/2017_attacker_fit.npz').resolve().parent
    frozen = ev.FrozenSeed(seed)

    state = load_frozen_state(seed, registry)
    a0 = state['a0']
    # The new erasure arms transform a no-protection channel, so those channels must be
    # loaded whether or not the caller listed them among the panel arms.
    needed = tuple(dict.fromkeys(tuple(new_arms)
                                 + tuple(f'dax{w}_none' for w in WIDTHS))) if new_arms else ()
    channels = load_new_channels(out, seed, needed, a0, state['channel']['representation_fit'])
    affine = load_new_erasure_affine(out, seed, registry) if new_arms else {}
    affine = {name: spec for name, spec in affine.items() if spec['base'] in channels}
    encoders, proofs = {}, {}

    if include_baselines:
        erasure = reconstruct_erasure(seed, registry)
        proofs.update(erasure['proofs'])
        for arm in HISTORICAL_ERASURE:
            if erasure['maps'].get(arm) and erasure['proofs'][arm]['identity_proved']:
                affine[arm] = {'base': 'A0', 'projection': erasure['maps'][arm]['projection'],
                               'mean': erasure['maps'][arm]['mean']}
        optnet = reconstruct_optnet(out, seed)
        proofs.update(optnet['proofs'])
        for arm in HISTORICAL_OPTNET:
            if optnet['proofs'][arm]['identity_proved']:
                encoders[arm] = optnet['encoders'][arm]

    # Hash the frozen 2018 objects while `ev.DEV` still points at the residual-spectral
    # study; `object_files()` resolves `maps.joblib` through it. Only after that is
    # `ev.DEV` repointed at this study's tree, which is where `fit_unit` looks for each
    # interface's 2018 attack candidates (`<DEV>/seed_N/<arm>/audits/audit_selection.json`).
    for path in frozen.object_files():
        registry.add(path)

    model = TransportModel(frozen.spectral, a0, channels, affine, encoders)
    conditions = tuple(sorted(model.arms))
    frozen.spectral = model
    ev.SPECTRAL = conditions
    ev.INTERFACES = ('H',) + conditions
    ev.DEV = Path(out).resolve()
    return ev, frozen, conditions, proofs


def link_transport_H(out: Path, seed: int) -> Path:
    source = resolve(f'results/{TRANSPORT_NAME}/seed_{seed}/H')
    dest = Path(out) / 'exploratory_2017' / f'seed_{seed}' / 'H'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.symlink_to(source)
    return source


# ------------------------------------------------------------------ runners
def run(out: Path = OUT, seeds=(0, 1, 2), new_arms=(), include_baselines=True,
        phase='fit') -> dict:
    limit_threads()
    torch.set_num_threads(1)
    out = Path(out)
    root = out / 'exploratory_2017'
    root.mkdir(parents=True, exist_ok=True)
    summary, all_proofs = {}, {}

    for seed in seeds:
        registry = Registry.new()
        ev, frozen, conditions, proofs = prepare(out, seed, registry, new_arms, include_baselines)
        all_proofs[str(seed)] = proofs
        link_transport_H(out, seed)

        if phase == 'fit':
            rel, labels = {}, {}
            for part in ev.FIT_PARTITIONS:
                data = ev._partition(part)
                path = root / f'seed_{seed}' / 'releases_2017' / f'{part}.npz'
                if not path.exists():
                    ev.save_releases(path, ev.build_releases(frozen, data['frame']))
                rel[part] = ev.load_releases(path)
                labels[part] = data['labels']
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
                print('X2017_FIT', seed, condition, round(result['runtime_seconds'], 1), 's',
                      flush=True)
            summary[str(seed)] = {'conditions': records, 'inputs': registry.dump()}
        else:
            summary[str(seed)] = score_seed(ev, frozen, root, seed, conditions)

    name = 'EXPLORATORY_2017_FIT.json' if phase == 'fit' else 'EXPLORATORY_2017_SCORES.json'
    write_json(root / name, {
        'seeds': summary, 'machine': machine_state(), 'identity_proofs': all_proofs,
        'evaluation_status': 'EXPLORATORY CROSS-YEAR DEVELOPMENT',
        'mode': 'B only (fresh 2017 adaptation); Mode A is not claimed for any new interface',
        'scope_note': ('The 2017 locked evaluation is finished and its final partition is spent. '
                       'These are exploratory cross-year DEVELOPMENT numbers, not a second '
                       'confirmation. The original frozen 2017 transport result keeps its '
                       'historical status and is not restated or overwritten. 2017 is NOT a new '
                       'test merely because these methods had not been scored there.'),
        'transport_rule': ('2018-fitted transformations are reused on 2017. No new 2017 eraser '
                           'or encoder is fitted and then called transported. Encoders are not '
                           'retrained on 2017.'),
        'selection_note': ('Probes fitted on the 2017 attacker/task fitting partitions and '
                           'selected on the 2017 validation partitions by minimum unweighted '
                           'validation log loss, then candidate ID. Selection never sees the '
                           'evaluation rows.'),
        'year_separation': 'Years are reported separately; rows are never pooled across years.'})
    return summary


def score_seed(ev, frozen, root: Path, seed: int, conditions) -> dict:
    from experiments import acs_fixed_predictions_audits as old
    from experiments.acs_transfer_heads import load_candidate, metrics

    final = ev._partition(ev.FINAL)
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
                        'validation_log_loss':
                            record['validation_scores'][budget][role][cid]['log_loss'],
                        'scores': {'test': metrics(y[target][valid], p, old.CLASSES[target]),
                                   'test_person_weighted': metrics(
                                       y[target][valid], p, old.CLASSES[target], w[valid])}})
        write_json(dest / 'metrics.json', {
            'seed': seed, 'condition': condition, 'mode': 'B', 'year': 2017,
            'evaluation_status': 'EXPLORATORY CROSS-YEAR DEVELOPMENT (seal already spent)',
            'raw_metrics': out_rows})
        complete = {'runtime_seconds': time.perf_counter() - tick, 'rows': len(out_rows),
                    'nondeterminism_events': list(ev.NONDETERMINISM_EVENTS),
                    'metrics_sha256': sha_file(dest / 'metrics.json')}
        write_json(dest / 'complete.json', complete)
        rows[condition] = complete
        print('X2017_SCORE', seed, condition, len(out_rows), 'rows',
              round(complete['runtime_seconds'], 1), 's', flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--phase', choices=['fit', 'score'], default='fit')
    p.add_argument('--new-arms', nargs='*', default=[])
    p.add_argument('--no-baselines', action='store_true')
    a = p.parse_args()
    run(a.out, tuple(a.seeds), tuple(a.new_arms), not a.no_baselines, a.phase)


if __name__ == '__main__':
    main()
