"""Stage 9: mechanism checks that explain outcomes, from saved artifacts only.

Nothing here is a new search and nothing here selects anything. Every number comes
from artifacts already written by the fit, erasure and audit stages.

**These are observational correlations among diagnostics of a bounded search. No
causal explanation is inferred from them.** Lower measured recovery against a finite
attacker family is not independence, not differential privacy, not a mutual-information
bound and not protection against arbitrary attackers.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch

from collections import defaultdict

from .inputs import (OUT, POOLS, ROLE_ORDER, Registry, array_hash, arrays, read_json, resolve,
                     sha_file, write_json)
from .report import (BETAS, FAMILY_SENSITIVE, MAIN, MAIN_BUDGET, MAIN_SCOPE, MAIN_SPLIT,
                     NEW_ERASURE, NO_PROTECTION, POLICIES, SINGLE, WEIGHTS, WIDTHS,
                     condition_file, main_arm)
from .run_fit import SINGLE_CELLS, limit_threads


def fit_records(out: Path, seeds, arms) -> dict:
    out = Path(out)
    records = {}
    for seed in seeds:
        for arm in arms:
            path = out / f'seed_{seed}' / 'fits' / arm / 'fit_record.json'
            if path.exists():
                records[seed, arm] = read_json(path)
    return records


# ------------------------------------------------------------------ 1. training vs audit gap
def training_versus_auditor(out: Path, seeds, arms, points) -> list:
    """Training-attacker gain against fresh-auditor additional recovery, per role.

    Two different quantities on two different pools. The comparison is descriptive: it
    shows whether the differentiable family the mapper was trained against saw what the
    independent audit slate later saw, or something else.
    """
    rows = []
    records = fit_records(out, seeds, arms)
    for (seed, arm), record in sorted(records.items()):
        selected = {m['step']: m for m in record['monitor_scores']}
        chosen = selected[record['selected_step']]
        for weight in WEIGHTS:
            key = (seed, arm, MAIN_SPLIT, weight, MAIN_BUDGET, MAIN_SCOPE)
            h = points.get((seed, 'H', MAIN_SPLIT, weight, MAIN_BUDGET, MAIN_SCOPE))
            point = points.get(key)
            if point is None or h is None:
                continue
            for role in ROLE_ORDER:
                rows.append({
                    'seed': seed, 'arm': arm, 'weight': weight, 'role': role,
                    'selected_step': record['selected_step'],
                    'training_fresh_probe_gain': chosen['gains'].get(role),
                    'training_active_attacker': (chosen.get('active_attacker') or {}).get(role),
                    'audit_additional_recovery': point['gains'][role] - h['gains'][role],
                    'audit_absolute_recovery': point['gains'][role],
                    'audit_selected_candidate':
                        point['selected']['audit/' + role]['candidate_id'],
                    'note': ('training gain is measured on the internal monitor fold against a '
                             '300-update differentiable probe; audit recovery is measured on the '
                             'test pool against the full independent slate. Different pools, '
                             'different families, not the same quantity')})
    return rows


# ------------------------------------------------------------------ 2. refresh and catch-up
def refresh_effect(out: Path, seeds, arms) -> list:
    """Did refreshing an attacker recover ground the incumbent had lost?

    A large positive recovery means the incumbent had been fooled by the moving channel
    rather than genuinely defeated, which is the failure mode the refresh exists to
    detect.
    """
    rows = []
    for (seed, arm), record in sorted(fit_records(out, seeds, arms).items()):
        for entry in record.get('refreshes', []):
            for role, decision in entry['decisions'].items():
                rows.append({'seed': seed, 'arm': arm, 'step': entry['step'],
                             'family': entry['family'], 'role': role,
                             'incumbent_monitor_ce': decision['incumbent_monitor_ce'],
                             'refreshed_monitor_ce': decision['refreshed_monitor_ce'],
                             'recovery': decision['recovery'], 'kept': decision['kept']})
    return rows


def catchup_effect(points, seeds, conditions) -> list:
    """Budget 120 against budget 360 on the same scope: what the longer attack buys."""
    rows = []
    for seed, condition, weight in itertools.product(seeds, conditions, WEIGHTS):
        short = points.get((seed, condition, MAIN_SPLIT, weight, 120, MAIN_SCOPE))
        long = points.get((seed, condition, MAIN_SPLIT, weight, 360, MAIN_SCOPE))
        if short is None or long is None:
            continue
        for endpoint in FAMILY_SENSITIVE:
            rows.append({'seed': seed, 'condition': condition, 'weight': weight,
                         'endpoint': endpoint,
                         'recovery_budget_120': short['gains'][endpoint],
                         'recovery_budget_360': long['gains'][endpoint],
                         'catchup_gain': long['gains'][endpoint] - short['gains'][endpoint]})
    return rows


# ------------------------------------------------------------------ 3. distortion vs residence
def distortion_versus_residence(out: Path, seeds, arms, points) -> list:
    """Does teacher fidelity predict residence transfer at all?

    The utility objective uses teacher distortion as a proxy for useful capability.
    Whether that proxy tracks the reserved residence task is an open empirical question
    and is answered here descriptively, never assumed.
    """
    rows = []
    records = fit_records(out, seeds, arms)
    for (seed, arm), record in sorted(records.items()):
        chosen = {m['step']: m for m in record['monitor_scores']}[record['selected_step']]
        for weight in WEIGHTS:
            point = points.get((seed, arm, MAIN_SPLIT, weight, MAIN_BUDGET, MAIN_SCOPE))
            h = points.get((seed, 'H', MAIN_SPLIT, weight, MAIN_BUDGET, MAIN_SCOPE))
            if point is None or h is None:
                continue
            rows.append({
                'seed': seed, 'arm': arm, 'weight': weight,
                'teacher_distortion': chosen['distortion'],
                'source_cross_entropy': chosen['source'],
                'monitor_penalty': chosen['penalty'],
                'residence_gain_vs_H':
                    h['utility']['same_residence'] - point['utility']['same_residence'],
                'income_loss': point['utility']['income_binary'],
                'employment_loss': point['utility']['civilian_at_work'],
                'commute_gain_vs_H':
                    h['utility']['commute_over20'] - point['utility']['commute_over20']})
    return rows


def correlations(rows: list, x: str, y: str) -> dict:
    a = np.array([r[x] for r in rows if r[x] is not None and r[y] is not None], float)
    b = np.array([r[y] for r in rows if r[x] is not None and r[y] is not None], float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return {'n': int(len(a)), 'pearson': None, 'spearman': None}
    order_a = np.argsort(np.argsort(a))
    order_b = np.argsort(np.argsort(b))
    return {'n': int(len(a)), 'pearson': float(np.corrcoef(a, b)[0, 1]),
            'spearman': float(np.corrcoef(order_a, order_b)[0, 1]),
            'caveat': ('an observational association across fitted arms; it does not identify a '
                       'mechanism and no causal reading is offered')}


# ------------------------------------------------------------------ 4. beta-zero equivalence
def beta_equivalence(out: Path, seeds) -> list:
    """Is a beta arm effectively identical to its beta = 0 continuation?

    Compared on the released arrays themselves -- output norms, per-pool maximum
    absolute difference, and the principal subspace angle -- not on a downstream score.
    """
    rows = []
    out = Path(out)
    for seed in seeds:
        for width in WIDTHS:
            base_path = out / f'seed_{seed}' / 'releases' / f'dax{width}_none' / 'releases.npz'
            if not base_path.exists():
                continue
            base = arrays(base_path)
            for policy, beta in itertools.product(POLICIES, BETAS):
                arm = main_arm(width, policy, beta)
                path = out / f'seed_{seed}' / 'releases' / arm / 'releases.npz'
                if not path.exists():
                    continue
                other = arrays(path)
                x = np.asarray(base['wire/A/test'], np.float64)[:, 4:]
                y = np.asarray(other['wire/A/test'], np.float64)[:, 4:]
                xc, yc = x - x.mean(0), y - y.mean(0)
                _, _, vx = np.linalg.svd(xc, full_matrices=False)
                _, _, vy = np.linalg.svd(yc, full_matrices=False)
                singular = np.linalg.svd(vx @ vy.T, compute_uv=False)
                rows.append({
                    'seed': seed, 'width': width, 'policy': policy, 'beta': beta, 'arm': arm,
                    'identical_to_no_protection': bool(np.array_equal(x, y)),
                    'max_abs_difference': float(np.abs(x - y).max()),
                    'rms_difference': float(np.sqrt(((x - y) ** 2).mean())),
                    'norm_ratio': float(np.linalg.norm(y) / max(np.linalg.norm(x), 1e-300)),
                    'min_principal_cosine': float(np.clip(singular, -1, 1).min()),
                    'max_principal_angle_degrees': float(
                        np.degrees(np.arccos(np.clip(singular, -1, 1).min())))})
    return rows


# ------------------------------------------------------------------ 5. compression accounting
def compression(out: Path, seeds, conditions) -> list:
    """Ambient width, effective rank and the variance spectrum of every released channel."""
    rows = []
    out = Path(out)
    for seed in seeds:
        for condition in conditions:
            path = out / f'seed_{seed}' / 'releases' / condition / 'releases.npz'
            if not path.exists():
                continue
            z = np.asarray(arrays(path)['wire/A/test'], np.float64)[:, 4:]
            centred = z - z.mean(0)
            singular = np.linalg.svd(centred, compute_uv=False)
            energy = (singular ** 2) / max(float((singular ** 2).sum()), 1e-300)
            entropy = float(-(energy[energy > 0] * np.log(energy[energy > 0])).sum())
            rows.append({'seed': seed, 'condition': condition,
                         'ambient_width': int(z.shape[1]),
                         'numerical_rank_1e-10': int(np.linalg.matrix_rank(centred, tol=1e-10)),
                         'effective_rank_entropy': float(np.exp(entropy)),
                         'components_for_99pct': int(np.searchsorted(np.cumsum(energy), .99) + 1),
                         'top_singular_share': float(energy[0]),
                         'per_coordinate_std_mean': float(z.std(0).mean())})
    return rows


# ------------------------------------------------------------------ 5b. exact fit accounting
def channel_accounting(out: Path, seeds) -> dict:
    """Completed units against genuinely distinct released channels.

    A completed fit whose preregistered selection returned the unmoved initial
    checkpoint is a **completed unit that is not a unique system**. It is counted as
    such, never as a separate fitted interface, and never dropped either. Units that are
    bitwise identical to a HISTORICAL arm are recorded as proved duplicates and reused
    with that proof rather than presented as new.
    """
    from .inputs import H_A_WIDTH
    out = Path(out)
    per_seed, duplicates_of_historical = {}, []
    for seed in seeds:
        root = out / f'seed_{seed}' / 'releases'
        if not root.exists():
            continue
        groups = defaultdict(list)
        for path in sorted(root.iterdir()):
            npz = path / 'releases.npz'
            if not npz.exists():
                continue
            z = np.asarray(arrays(npz)['wire/A/test'], np.float64)[:, H_A_WIDTH:]
            groups[array_hash(z)].append(path.name)
        # Bitwise duplicates of a historical external arm.
        for arm, historical in (('leace_dax16_none', 'leace_A0'),
                                ('splince_dax16_none', 'splince_A0')):
            local = root / arm / 'releases.npz'
            if not local.exists():
                continue
            try:
                reference = resolve(f'results/pcrl_invariant_baselines_v1/seed_{seed}'
                                    f'/releases/{historical}/releases.npz')
            except FileNotFoundError:
                continue
            a = np.asarray(arrays(local)['wire/A/test'], np.float64)[:, H_A_WIDTH:]
            b = np.asarray(arrays(reference)['wire/A/test'], np.float64)[:, H_A_WIDTH:]
            if np.array_equal(a, b):
                duplicates_of_historical.append(
                    {'seed': seed, 'arm': arm, 'identical_to': historical,
                     'max_abs_difference': 0.0,
                     'reason': ('its input channel is the no-protection continuation, whose '
                                'preregistered selection returned the unmoved A0 channel, and '
                                'both erasers are deterministic closed forms of that channel')})
        per_seed[str(seed)] = {
            'released_units': sum(len(v) for v in groups.values()),
            'distinct_channels': len(groups),
            'duplicate_units_within_seed': sum(len(v) - 1 for v in groups.values()),
            'duplicate_groups': [sorted(v) for v in groups.values() if len(v) > 1]}
    total_units = sum(v['released_units'] for v in per_seed.values())
    distinct = sum(v['distinct_channels'] for v in per_seed.values())
    return {
        'per_seed': per_seed,
        'released_units_total': total_units,
        'distinct_channels_total': distinct,
        'duplicate_units_within_seed_total': total_units - distinct,
        'duplicates_of_historical_arms': duplicates_of_historical,
        'distinct_and_new_channels': distinct - len(duplicates_of_historical),
        'note': ('Duplicates here are OUTCOMES of the preregistered checkpoint rule, not '
                 'failed or wasted fits: every unit ran its full budget and its selection '
                 'returned the unmoved initial checkpoint. They are counted as completed units '
                 'that are not unique systems.'),
    }


# ------------------------------------------------------------------ 6. class support
def class_support(out: Path, seeds) -> list:
    """Which protected categories are unsupported in each ACTUAL training fold.

    Absence in a fit split is **not** population absence and is never reported as one.
    """
    from .inputs import household_folds, representation_labels
    rows = []
    registry = Registry.new()
    for seed in seeds:
        labels = representation_labels(seed, registry)
        folds = household_folds(labels['serials'])
        for attribute, size in (('SEX', 2), ('RAC1P', 9), ('public_coverage', 2)):
            y = labels['protected'][attribute]
            for fold, index in folds.items():
                counts = np.bincount(y[index][y[index] >= 0], minlength=size)
                rows.append({
                    'seed': seed, 'attribute': attribute, 'fold': fold,
                    'rows': int(len(index)), 'missing': int((y[index] < 0).sum()),
                    'support': counts.tolist(),
                    'unsupported_classes_in_this_fold':
                        [int(i) for i in np.flatnonzero(counts == 0)],
                    'note': ('unsupported IN THIS FIT FOLD. Population absence is not inferred '
                             'from a fold split')})
    return rows


# ------------------------------------------------------------------ driver
def run(out: Path = OUT, seeds=(0, 1, 2)) -> dict:
    from experiments.pcrl_nonlinear_rank_v1.run_report import csvout
    from .report import available, load_points
    from .run_report import ALL_CONDITIONS

    limit_threads()
    out = Path(out)
    registry = Registry.new()
    conditions = available(out, ALL_CONDITIONS, seeds)
    points, _raw, _prior = load_points(out, seeds, conditions, registry)

    neural = [a for a in MAIN + NO_PROTECTION + SINGLE if a in conditions]
    gap = training_versus_auditor(out, seeds, neural, points)
    refresh = refresh_effect(out, seeds, neural)
    catchup = catchup_effect(points, seeds, conditions)
    trajectory = distortion_versus_residence(out, seeds, neural, points)
    equivalence = beta_equivalence(out, seeds)
    ranks = compression(out, seeds, [c for c in conditions
                                     if (out / f'seed_{seeds[0]}' / 'releases' / c).exists()])
    support = class_support(out, seeds)
    accounting = channel_accounting(out, seeds)

    csvout(out / 'MECH_TRAINING_VS_AUDIT.csv', gap)
    csvout(out / 'MECH_REFRESH.csv', refresh)
    csvout(out / 'MECH_CATCHUP.csv', catchup)
    csvout(out / 'MECH_TRAJECTORY.csv', trajectory)
    csvout(out / 'MECH_BETA_EQUIVALENCE.csv', equivalence)
    csvout(out / 'MECH_COMPRESSION.csv', ranks)
    csvout(out / 'MECH_CLASS_SUPPORT.csv', support)

    summary = {
        'channel_accounting': accounting,
        'training_versus_auditor': {
            'rows': len(gap),
            'mean_training_gain': float(np.mean([r['training_fresh_probe_gain'] for r in gap
                                                 if r['training_fresh_probe_gain'] is not None])),
            'mean_audit_additional_recovery': float(
                np.mean([r['audit_additional_recovery'] for r in gap])),
            'correlation': correlations(
                [r for r in gap if r['training_fresh_probe_gain'] is not None],
                'training_fresh_probe_gain', 'audit_additional_recovery')},
        'refresh': {
            'decisions': len(refresh),
            'refreshed_kept': sum(1 for r in refresh if r['kept'] == 'refreshed'),
            'mean_recovery': float(np.mean([r['recovery'] for r in refresh])) if refresh else None,
            'max_recovery': float(np.max([r['recovery'] for r in refresh])) if refresh else None},
        'catchup': {
            'rows': len(catchup),
            'mean_catchup_gain': float(np.mean([r['catchup_gain'] for r in catchup]))
            if catchup else None},
        'distortion_versus_residence': {
            'rows': len(trajectory),
            'distortion_vs_residence': correlations(trajectory, 'teacher_distortion',
                                                    'residence_gain_vs_H'),
            'source_vs_residence': correlations(trajectory, 'source_cross_entropy',
                                                'residence_gain_vs_H')},
        'beta_equivalence': {
            'rows': len(equivalence),
            'identical_to_no_protection':
                [f"seed{r['seed']}/{r['arm']}" for r in equivalence
                 if r['identical_to_no_protection']],
            'max_abs_difference_min': float(np.min([r['max_abs_difference']
                                                    for r in equivalence]))
            if equivalence else None},
        'compression': {'rows': len(ranks)},
        'class_support': {
            'rows': len(support),
            'folds_with_an_unsupported_class':
                [f"seed{r['seed']}/{r['attribute']}/{r['fold']}" for r in support
                 if r['unsupported_classes_in_this_fold']]},
        'non_claims': (
            'These are observational associations among diagnostics of a bounded search. No '
            'causal explanation is inferred. Lower measured recovery against a finite attack '
            'family is not independence, not differential privacy, not a mutual-information '
            'bound and not protection against arbitrary attackers.'),
    }
    write_json(out / 'MECHANISM.json', summary)
    print('MECHANISM done', flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args()
    run(a.out, tuple(a.seeds))


if __name__ == '__main__':
    main()
