"""Validation aimed at material risks, not at accumulating redundant assertions.

Eleven checks, each targeting a way this study could silently be wrong:

1. **gradient signs** -- on a synthetic fixture where the answer is known, stronger
   recovery must raise the mapper's penalty and the mapper step must reduce it;
2. **nested baseline behaviour** -- the zero correction must recover `p0_j` exactly, so
   `g = 0` at initialisation by construction;
3. **label exclusion** -- the reserved residence and commute keys must be unreachable
   from the representation-fitting label path;
4. **household boundaries** -- the internal folds must partition
   `representation_fit` with no household straddling two folds;
5. **release parity** -- `H_A` and `H_B` bitwise preserved in every released wire;
6. **map serialisation** -- a channel reloaded from its checkpoint must reproduce its
   release bitwise;
7. **resumed-unit identity** -- a reused unit's recorded hash must still match its file;
8. **score aggregation** -- stored endpoint values must be recomputable from the stored
   per-person predictions;
9. **role masks** -- the audited forbidden registry must be the full historical eleven;
10. **simultaneous comparison construction** -- the candidate-wide family must contain
    every contrast searched, and its critical value must exceed the within-contrast one;
11. **identical channel, identical endpoints** -- two arms whose released channel is
    bitwise identical must score exactly the same endpoints under the matched-exposure
    scope. This is the end-to-end check: it fails if anything between the release and
    the score is not a function of the release.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .attackers import RoleEnsemble, ServiceBaseline, masked_cross_entropy, service_width
from .channel import build_channel
from .inputs import (H_A_WIDTH, H_B_WIDTH, OUT, POOLS, ROLE_ORDER, RESERVED_TASKS, Registry,
                     array_hash, arrays, household_folds, load_frozen_state, read_json,
                     representation_labels, sha_file, standardize, write_json)
from .run_fit import limit_threads, machine_state
from .train import POLICIES, policy_penalty


def check_gradient_signs() -> dict:
    """Synthetic fixture: a channel that literally carries the protected label.

    `z = [s + noise]` makes the label recoverable by construction. The attacker must
    find a positive gain; the mapper's penalty must therefore be positive; and one
    mapper step under `U + beta * penalty` must MOVE THE CHANNEL so the penalty falls.
    If the two players were reversed, or if only the service baseline were being
    optimised, this check fails.
    """
    torch.manual_seed(0)
    n, classes = 4096, 2
    s = torch.randint(0, classes, (n,))
    carrier = torch.nn.Linear(4, 6)
    with torch.no_grad():
        carrier.weight.normal_(0, 0.5)
        carrier.bias.zero_()
    x = torch.randn(n, 4)
    x[:, 0] = s.float() * 3.0 + 0.2 * torch.randn(n)      # the protected signal
    ha = torch.rand(n, H_A_WIDTH)
    base = torch.zeros(n, classes)                         # uninformative p0

    attacker = torch.nn.Sequential(torch.nn.Linear(H_A_WIDTH + 6, 64), torch.nn.ReLU(),
                                   torch.nn.Linear(64, classes))
    with torch.no_grad():
        attacker[-1].weight.zero_()
        attacker[-1].bias.zero_()
    with torch.no_grad():
        zero_gain = float(torch.nn.functional.cross_entropy(base, s)
                          - torch.nn.functional.cross_entropy(base + attacker(
                              torch.cat((ha, carrier(x).detach()), 1)), s))

    opt_a = torch.optim.Adam(attacker.parameters(), lr=1e-2)
    for _ in range(400):
        opt_a.zero_grad(set_to_none=True)
        z = carrier(x).detach()
        loss = torch.nn.functional.cross_entropy(base + attacker(torch.cat((ha, z), 1)), s)
        loss.backward()
        opt_a.step()
    for p in attacker.parameters():
        p.requires_grad_(False)

    def penalty_now():
        z = carrier(x)
        gain = (torch.nn.functional.cross_entropy(base, s)
                - torch.nn.functional.cross_entropy(base + attacker(torch.cat((ha, z), 1)), s))
        return torch.clamp(gain, min=0.0)

    with torch.no_grad():
        trained_gain = float(penalty_now())
    opt_m = torch.optim.Adam(carrier.parameters(), lr=1e-2)
    before = trained_gain
    for _ in range(50):
        opt_m.zero_grad(set_to_none=True)
        (1.0 * penalty_now()).backward()
        opt_m.step()
    with torch.no_grad():
        after = float(penalty_now())
    return {'zero_correction_gain': zero_gain,
            'trained_attacker_gain': trained_gain,
            'penalty_after_mapper_steps': after,
            'stronger_recovery_raises_penalty': bool(trained_gain > zero_gain + 1e-6),
            'mapper_step_reduces_penalty': bool(after < before - 1e-6),
            'pass': bool(trained_gain > zero_gain + 1e-6 and after < before - 1e-6)}


def check_nested_baseline() -> dict:
    """The zero correction must reproduce `p0_j` exactly, to floating-point equality."""
    torch.manual_seed(1)
    p0 = ServiceBaseline(H_A_WIDTH, 2, 7).eval()
    x = torch.rand(512, H_A_WIDTH)
    base = p0(x).detach()
    results = {}
    for family in ('linear', 'mlp64', 'mlp64_32'):
        ensemble = RoleEnsemble('A/SEX', 0, (family,), H_A_WIDTH + 16, 3)
        z = torch.randn(512, 16)
        logits = base + ensemble.nets[family](torch.cat((x, z), 1))
        results[family] = {'max_abs_difference': float((logits - base).abs().max()),
                           'exactly_nested': bool(torch.equal(logits, base))}
    return {'families': results,
            'pass': all(v['exactly_nested'] for v in results.values())}


def check_label_exclusion(seed: int, registry: Registry) -> dict:
    labels = representation_labels(seed, registry)
    leaked = [t for t in RESERVED_TASKS if t in labels['source'] or t in labels['protected']]
    return {'reserved_tasks': list(RESERVED_TASKS),
            'keys_returned': sorted(list(labels['source']) + list(labels['protected'])),
            'leaked': leaked, 'pass': not leaked}


def check_household_boundaries(seed: int, registry: Registry) -> dict:
    labels = representation_labels(seed, registry)
    serials = np.asarray(labels['serials']).astype(str)
    folds = household_folds(serials)
    sizes = {k: int(len(v)) for k, v in folds.items()}
    total = sum(sizes.values())
    assignment = {}
    straddling = []
    for name, index in folds.items():
        for household in np.unique(serials[index]):
            if household in assignment and assignment[household] != name:
                straddling.append(household)
            assignment[household] = name
    return {'sizes': sizes, 'total': total, 'rows': int(len(serials)),
            'partitions': bool(total == len(serials)),
            'straddling_households': len(straddling),
            'pass': bool(total == len(serials) and not straddling)}


def check_release_parity(out: Path, seed: int, registry: Registry, limit=None) -> dict:
    state = load_frozen_state(seed, registry)
    anchors = state['anchors']
    root = Path(out) / f'seed_{seed}' / 'releases'
    arms = sorted(p.name for p in root.iterdir() if (p / 'releases.npz').exists())
    if limit:
        arms = arms[:limit]
    failures = []
    for arm in arms:
        data = arrays(root / arm / 'releases.npz')
        for pool in POOLS:
            wa = np.asarray(data[f'wire/A/{pool}'], np.float64)
            wb = np.asarray(data[f'wire/B/{pool}'], np.float64)
            wab = np.asarray(data[f'wire/AB/{pool}'], np.float64)
            ok = (np.array_equal(wa[:, :H_A_WIDTH], anchors[f'{pool}/A'])
                  and np.array_equal(wb, anchors[f'{pool}/B'])
                  and np.array_equal(wab[:, -H_B_WIDTH:], anchors[f'{pool}/B'])
                  and np.array_equal(wab[:, :wa.shape[1]], wa)
                  and wa.dtype == np.float64 and np.isfinite(wa).all())
            if not ok:
                failures.append(f'{arm}/{pool}')
    return {'arms_checked': len(arms), 'pools_per_arm': len(POOLS),
            'failures': failures, 'pass': not failures}


def check_map_serialisation(out: Path, seed: int, registry: Registry, arms=None) -> dict:
    state = load_frozen_state(seed, registry)
    a0 = state['a0']
    root = Path(out) / f'seed_{seed}'
    candidates = arms or [p.name for p in (root / 'fits').iterdir()
                          if (p / 'checkpoints.pt').exists()][:6]
    rows = []
    for arm in candidates:
        width = 8 if arm.startswith('dax8') else 16
        model = build_channel(width, a0, state['channel']['representation_fit'])['model']
        model.load_state_dict(torch.load(root / 'fits' / arm / 'checkpoints.pt',
                                         map_location='cpu',
                                         weights_only=False)['selected_state'])
        model.eval()
        stored = arrays(root / 'releases' / arm / 'releases.npz')
        exact = True
        for pool in POOLS:
            x = standardize(state['pca'][pool], a0['input_mean'], a0['input_scale'])
            value = model.release(x)
            exact = exact and bool(np.array_equal(
                value, np.asarray(stored[f'wire/A/{pool}'], np.float64)[:, H_A_WIDTH:]))
        rows.append({'arm': arm, 'reloaded_release_bitwise_identical': exact})
    return {'arms': rows, 'pass': all(r['reloaded_release_bitwise_identical'] for r in rows)}


def check_resumed_identity(out: Path, seeds) -> dict:
    mismatches = []
    checked = 0
    for seed in seeds:
        root = Path(out) / f'seed_{seed}' / 'fits'
        if not root.exists():
            continue
        for marker in sorted(root.glob('*/fit_complete.json')):
            record = read_json(marker)
            release = (Path(out) / f'seed_{seed}' / 'releases' / record['arm'] / 'releases.npz')
            fit_record = marker.parent / 'fit_record.json'
            checked += 1
            if sha_file(release) != record['release_sha256']:
                mismatches.append(f'{seed}/{record["arm"]}/release')
            if sha_file(fit_record) != record['fit_record_sha256']:
                mismatches.append(f'{seed}/{record["arm"]}/fit_record')
    return {'units_checked': checked, 'mismatches': mismatches, 'pass': not mismatches}


def check_role_masks() -> dict:
    from .freeze import FORBIDDEN_REGISTRY
    from experiments.pcrl_nonlinear_rank_v1.report import FORBIDDEN
    return {'declared': list(FORBIDDEN_REGISTRY), 'scored': list(FORBIDDEN),
            'count': len(FORBIDDEN),
            'pass': set(FORBIDDEN_REGISTRY) == set(FORBIDDEN) and len(FORBIDDEN) == 11}


def check_simultaneous_construction(out: Path) -> dict:
    """The candidate-wide family must cover every searched contrast and dominate."""
    import csv
    from .run_report import contrasts
    path = Path(out) / 'PAIRED_INTERVALS.csv'
    if not path.exists():
        return {'pass': None, 'reason': 'PAIRED_INTERVALS.csv not written yet'}
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    searched = {(s['left'], s['right']) for s in contrasts()}
    present = {(r['left'], r['right']) for r in rows}
    sizes = {int(r['family_size']) for r in rows}
    wider = all(float(r['candidate_wide_high']) >= float(r['adjusted_high']) - 1e-12
                and float(r['candidate_wide_low']) <= float(r['adjusted_low']) + 1e-12
                for r in rows)
    return {'searched_contrasts': len(searched), 'present_contrasts': len(present),
            'contrasts_absent_because_arm_missing': sorted(searched - present),
            'family_size': sorted(sizes), 'rows': len(rows),
            'candidate_wide_contains_within_contrast': wider,
            'pass': bool(present <= searched and len(sizes) == 1 and wider)}


def check_identical_channel_identical_endpoints(out: Path, seeds) -> dict:
    """End-to-end: a bitwise-identical channel must reproduce its comparator's endpoints.

    Several low-beta arms select their unmoved initial checkpoint, so their released
    channel is bitwise identical to the matched-width no-protection continuation. Under
    the MATCHED-EXPOSURE primary scope those arms must score **exactly** the same
    endpoints as that continuation. If they do not, something between the release and
    the score is not a function of the release.
    """
    from experiments.pcrl_nonlinear_rank_v1.inputs import Registry as BaseRegistry
    from .report import FORBIDDEN, MAIN_BUDGET, MAIN_SCOPE, MAIN_SPLIT, WEIGHTS, load_points

    out = Path(out)
    pairs = []
    for seed in seeds:
        releases = out / f'seed_{seed}' / 'releases'
        if not releases.exists():
            continue
        digests = {}
        for path in sorted(releases.iterdir()):
            npz = path / 'releases.npz'
            if not npz.exists():
                continue
            z = np.asarray(arrays(npz)['wire/A/test'], np.float64)[:, H_A_WIDTH:]
            digests.setdefault(array_hash(z), []).append(path.name)
        for group in digests.values():
            if len(group) > 1:
                pairs.append((seed, sorted(group)))
    if not pairs:
        return {'pass': None, 'reason': 'no two arms share a released channel'}

    conditions = sorted({arm for _s, group in pairs for arm in group})
    scored = [c for c in conditions
              if all((out / f'seed_{s}' / c / 'metrics.json').exists() for s in seeds)]
    if len(scored) < 2:
        return {'pass': None, 'reason': 'identical-channel arms are not all scored yet',
                'groups': [{'seed': s, 'arms': g} for s, g in pairs]}
    points, _raw, _prior = load_points(out, seeds, tuple(scored), BaseRegistry.new())
    mismatches, checked = [], 0
    for seed, group in pairs:
        group = [a for a in group if a in scored]
        for other in group[1:]:
            for weight in WEIGHTS:
                a = points[seed, group[0], MAIN_SPLIT, weight, MAIN_BUDGET, MAIN_SCOPE]
                b = points[seed, other, MAIN_SPLIT, weight, MAIN_BUDGET, MAIN_SCOPE]
                checked += 1
                for endpoint in FORBIDDEN:
                    if a['gains'][endpoint] != b['gains'][endpoint]:
                        mismatches.append(f'{seed}/{group[0]} vs {other}/{endpoint}/{weight}')
                for task in a['utility']:
                    if a['utility'][task] != b['utility'][task]:
                        mismatches.append(f'{seed}/{group[0]} vs {other}/utility/{task}/{weight}')
    return {'groups': [{'seed': s, 'arms': g} for s, g in pairs],
            'comparisons': checked, 'scope': MAIN_SCOPE,
            'mismatches': mismatches[:20], 'mismatch_count': len(mismatches),
            'pass': not mismatches}


def check_score_aggregation(out: Path) -> dict:
    """Stored endpoint values must be recomputable from stored per-person predictions."""
    import csv
    path = Path(out) / 'SCORE_REPLAY.csv'
    if not path.exists():
        return {'pass': None, 'reason': 'SCORE_REPLAY.csv not written yet'}
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    worst = max((float(r['abs_difference']) for r in rows), default=0.0)
    return {'checks': len(rows), 'max_abs_difference': worst,
            'pass': bool(rows and worst < 1e-9)}


def run(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    limit_threads()
    out = Path(out)
    tick = time.perf_counter()
    registry = Registry.new()
    report = {
        'gradient_signs': check_gradient_signs(),
        'nested_baseline': check_nested_baseline(),
        'label_exclusion': check_label_exclusion(seeds[0], registry),
        'household_boundaries': check_household_boundaries(seeds[0], registry),
        'release_parity': check_release_parity(out, seeds[0], registry),
        'map_serialisation': check_map_serialisation(out, seeds[0], registry),
        'resumed_unit_identity': check_resumed_identity(out, seeds),
        'role_masks': check_role_masks(),
        'identical_channel_identical_endpoints':
            check_identical_channel_identical_endpoints(out, seeds),
        'score_aggregation': check_score_aggregation(out),
        'simultaneous_construction': check_simultaneous_construction(out),
    }
    checks = dict(report)
    report['all_pass'] = all(v.get('pass') is not False for v in checks.values())
    report['checks_passed'] = sum(1 for v in checks.values() if v.get('pass') is True)
    report['checks_not_applicable'] = sorted(k for k, v in checks.items()
                                             if v.get('pass') is None)
    report['checks_failed'] = sorted(k for k, v in checks.items() if v.get('pass') is False)
    report['runtime_seconds'] = time.perf_counter() - tick
    report['machine'] = machine_state()
    write_json(out / 'VALIDATION.json', report)
    for name, value in checks.items():
        print('VALIDATE', name, value.get('pass'), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    run(a.out, tuple(a.seeds))


if __name__ == '__main__':
    main()
