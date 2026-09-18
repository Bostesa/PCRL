"""Stage 1: enumerate every planned unit and hash-lock the protocol before any outcome.

`MATRIX.json` enumerates each unit with its seeds, its dependency hashes, the category
masks it will use, the caches it may reuse and the reason for that reuse. A failed or
duplicate fit is not a completed unique system; a mathematically identical branch is
reused **with proof**, never fitted to fill a count.
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from .attackers import FAMILIES, HELD_OUT_AUDIT_ARCHITECTURES, SINGLE_FAMILY
from .inputs import (OUT, POOLS, ROLE_CLASSES, ROLE_ORDER, Registry, sha_file, write_json)
from .run_fit import REPEAT_SEEDS, SINGLE_CELLS, WIDTHS, arm_plan, beta_tag, main_arm
from .train import BETAS, POLICIES, SLOT_SCHEDULE, TrainingConfig

SEEDS = (0, 1, 2)

HISTORICAL_BASELINES = ('leace_A0', 'splince_A0', 'optnet16_L1', 'optnet16_L2', 'optnet16_C1')
REFERENCES_2018 = ('H', 'E', 'A0', 'L025', 'L20', 'J')

# 2017 panel membership, fixed before any outcome (PROTOCOL.md 10).
PANEL_2017_BETAS = (0.3, 1.0)

FORBIDDEN_REGISTRY = (
    'A/public_coverage', 'A/commute_over20', 'A/SEX', 'A/RAC1P',
    'B/income_binary', 'B/civilian_at_work', 'B/same_residence', 'B/SEX', 'B/RAC1P',
    'AB/SEX', 'AB/RAC1P')

STRESS_SET = ('J', 'leace_A0', 'dax16_C1_b030', 'dax16_C1_b100',
              'dax16_L2_b030', 'dax16_L2_b100')


def erasure_arms() -> list:
    return [f'{method}_dax{width}_none' for width in WIDTHS for method in ('leace', 'splince')]


def panel_2017() -> dict:
    main = [main_arm(w, p, b) for b in PANEL_2017_BETAS for w in WIDTHS for p in POLICIES]
    return {'main': sorted(main), 'no_protection': [f'dax{w}_none' for w in WIDTHS],
            'new_erasure': erasure_arms(), 'historical_baselines': list(HISTORICAL_BASELINES),
            'references': ['H', 'A0', 'J'] }


def dependency_hashes(registry: Registry) -> dict:
    from .inputs import fixed_path
    out = {}
    for seed in SEEDS:
        for name in ('training/A0/final.pt', 'training/A0/releases.npz', 'pca.npz',
                     'anchors.npz', 'split_rows.npz', 'indices.json'):
            path = registry.resolve(fixed_path(seed, name))
            out[f'seed_{seed}/{name}'] = {'path': str(path), 'sha256': sha_file(path)}
    return out


def build_matrix(out: Path = OUT) -> dict:
    plan = arm_plan()
    config = TrainingConfig()
    counts = {block: len(items) * len(SEEDS) for block, items in plan.items()}
    counts['new_erasure'] = len(erasure_arms()) * len(SEEDS)
    total = sum(counts.values())
    matrix = {
        'study': 'pcrl_direct_adversarial_v1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'seeds': list(SEEDS),
        'widths': list(WIDTHS),
        'policies': list(POLICIES),
        'betas': list(BETAS),
        'training_config': config.as_dict(),
        'attacker_families': list(FAMILIES),
        'single_attacker_family': SINGLE_FAMILY,
        'held_out_audit_architectures': list(HELD_OUT_AUDIT_ARCHITECTURES),
        'slot_schedule': {k: [f'{r}|{s}' for r, s in v] for k, v in SLOT_SCHEDULE.items()},
        'trainable_protected_roles': list(ROLE_ORDER),
        'role_classes': ROLE_CLASSES,
        'audited_forbidden_registry': list(FORBIDDEN_REGISTRY),
        'pools': list(POOLS),
        'blocks': {
            'main': {'arms': [a for a, _ in plan['main']], 'per_seed': len(plan['main']),
                     'fits': counts['main'],
                     'description': '2 widths x 3 policies x 4 betas x 3 anchor seeds'},
            'no_protection': {'arms': [a for a, _ in plan['no_protection']],
                              'per_seed': len(plan['no_protection']),
                              'fits': counts['no_protection'],
                              'description': 'beta = 0, same utility objective and budget'},
            'single_attacker': {'arms': [a for a, _ in plan['single_attacker']],
                                'per_seed': len(plan['single_attacker']),
                                'fits': counts['single_attacker'],
                                'description': ('width16 x {L2,C1} x beta {0.3,1.0} x 3 seeds, '
                                                'one MLP family, same initialisation, utility '
                                                'objective, role weights, alternation schedule '
                                                'and refresh opportunities'),
                                'limitation': ('per-attacker optimiser-step exposure is matched; '
                                               'ensemble-wide gradient steps are one third '
                                               'because there are one third as many attackers. '
                                               'Compute equivalence does not imply equal '
                                               'expressive capacity')},
            'new_erasure': {'arms': erasure_arms(), 'per_seed': len(erasure_arms()),
                            'fits': counts['new_erasure'],
                            'description': ('LEACE/SPLINCE refit on each width no-protection '
                                            'channel; the HISTORICAL leace_A0 and splince_A0 '
                                            'are kept and never replaced'),
                            'reuse_reason': ('refit only because the input channel is new; the '
                                             'method, numerical policy and library overrides are '
                                             'the verified wrapper of 73903b7f')},
            'optimizer_repeat': {'arms': [a for a, _ in plan['optimizer_repeat']],
                                 'per_seed': len(plan['optimizer_repeat']),
                                 'fits': counts['optimizer_repeat'],
                                 'description': ('width16 x {L2,C1} x beta {0.3,1.0} x 3 anchor '
                                                 'seeds x 2 additional optimiser seeds'),
                                 'limitation': ('optimisation repetitions CONDITIONAL on the '
                                                'three historical anchors. They are not six or '
                                                'nine independent population seeds and test '
                                                'only stability of the proposed mechanism')},
        },
        'fit_counts': {**counts, 'total_new_mapper_or_transform_fits': total},
        'declared_total': 126,
        'audit_2018': {'interfaces': total,
                       'per_interface': 'full eleven-role registry, budgets 120 and 360',
                       'stress_set': list(STRESS_SET),
                       'stress': '2 fresh audit initialisations and a 720-epoch continuation'},
        'panel_2017': panel_2017(),
        'panel_2017_count': (len(panel_2017()['main']) * 0 + 36 + 6 + 12 + 15),
        'baseline_transport_completion': {
            'interfaces': [f'{name}/seed_{s}' for name in HISTORICAL_BASELINES for s in SEEDS],
            'count': len(HISTORICAL_BASELINES) * len(SEEDS),
            'rule': ('reuse the 2018-fitted transformation on 2017; never fit a new 2017 eraser '
                     'and call it transported. OptNet reconstruction must be verified against '
                     'the recorded 2018 artifacts before it may inherit the old identity')},
        'references_2018': list(REFERENCES_2018),
        'reuse_policy': ('A mathematically identical branch is reused WITH PROOF and is never '
                         'fitted to fill a count. A failed or duplicate fit is not a completed '
                         'unique system. Every reuse records the hash it was validated against.'),
    }
    return matrix


def run(out: Path = OUT) -> dict:
    out = Path(out)
    registry = Registry.new()
    matrix = build_matrix(out)
    matrix['dependency_hashes'] = dependency_hashes(registry)
    write_json(out / 'MATRIX.json', matrix)

    sources = sorted(Path('experiments/pcrl_direct_adversarial_v1').glob('*.py'))
    freeze = {
        'created_utc': matrix['created_utc'],
        'protocol_sha256': sha_file(out / 'PROTOCOL.md'),
        'method_sha256': sha_file(out / 'METHOD.md'),
        'matrix_sha256': sha_file(out / 'MATRIX.json'),
        'source_hashes': {str(p): sha_file(p) for p in sources},
        'dependency_hashes': matrix['dependency_hashes'],
        'declared_before': ('any 2018 or 2017 OUTCOME for the new arms. The only measurements '
                            'taken before this freeze are the training-only timing pilot of '
                            'PROTOCOL.md 7 and the A0 bit-exactness proof of METHOD.md 1, '
                            'neither of which reads an audit, a downstream pool, a test pool, '
                            'residence or commute.'),
    }
    write_json(out / 'PROTOCOL_FREEZE.json', freeze)
    return freeze


def verify(out: Path = OUT) -> dict:
    from .inputs import read_json
    out = Path(out)
    freeze = read_json(out / 'PROTOCOL_FREEZE.json')
    problems = []
    for name, key in (('PROTOCOL.md', 'protocol_sha256'), ('METHOD.md', 'method_sha256'),
                      ('MATRIX.json', 'matrix_sha256')):
        if sha_file(out / name) != freeze[key]:
            problems.append(name)
    return {'ok': not problems, 'changed': problems}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    print(verify(a.out) if a.verify else run(a.out))


if __name__ == '__main__':
    main()
