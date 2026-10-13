"""Metadata-only spec construction; no numerical imports or artifact fitting."""
from __future__ import annotations
from pathlib import Path

BASE_READERS = ('LR_C1','MLP_64x32','HGB_15x200','DA_MLP_64x32',
                'REP_LR_C1','REP_MLP_64x32','REP_HGB_15x200',
                'OUTPUT_LR_C1','OUTPUT_HGB_15x200')
EXTENDED_READERS = tuple(f'{contract}_{model}' for contract in
                         ('LOGITS_OUTPUT','DECISION','CONFIDENCE')
                         for model in ('LR_C1','MLP_64x32','HGB_15x200'))


def reuse_alias(bank, names=BASE_READERS):
    """Private config alias plus all exact sources needed by the root queue.

    The bank is a metadata entry from R1's reused_reader_candidates inventory.
    Admission remains pending until the numerical guard checks actual wires.
    """
    required = {'FROZEN_COMPLETE.json': {'path': bank['freeze'], 'sha256': bank['freeze_sha256']}}
    for name, pin in bank['release_and_score_files'].items():
        if Path(pin['path']).name in ('ATT_RELEASE.npz','INNER_RELEASE.npz','INNER_SCORES.npz'):
            required[name] = {'path': pin['path'], 'sha256': pin['sha256']}
    for filename, pin in bank['model_files'].items():
        if filename.removeprefix('MODEL_').removesuffix('.joblib').split('__')[-1] in names:
            required[filename] = {'path': pin['path'], 'sha256': pin['sha256']}
    return {'directory': str(Path(bank['freeze']).parent), 'names': list(names),
            'expected_bindings': dict(bank['bindings']), 'sources': list(required.values())}


def fit_count(*, include_base=True, include_extended=False, aliases=None, controls=False):
    names = set(BASE_READERS if include_base else ()) | set(EXTENDED_READERS if include_extended else ())
    count = 0
    for view in ('r1','r2','pair'):
        alias = (aliases or {}).get(view, {})
        aliased = set(alias.get('names', names)) if alias else set()
        if not aliased <= names:
            raise ValueError('unregistered alias reader')
        count += len(names-aliased)
    return count+(6 if controls else 0)


def reader_bindings(*, checkpoint_sha256, source_seed, release_variant,
                    manifest_sha256, science_lock_sha256, role_feature_hashes):
    if source_seed not in (0,1) or release_variant not in ('RAW_FULL','RAW_HYBRID','RAW_PROJ','U0_FULL','U0_PROJ','CONSTRAINED','ORDINARY','LOCAL_TWIN','TASK_ONLY'):
        raise ValueError('unregistered release identity')
    return {'checkpoint_sha256': checkpoint_sha256, 'source_seed': source_seed,
            'release_variant': release_variant, 'manifest_sha256': manifest_sha256,
            'science_lock_sha256': science_lock_sha256, 'role_feature_hashes': dict(role_feature_hashes),
            'wire_order': 'z16,centered_own_head_logitsK,own_head_probabilitiesK,numeric_own_head_decision;pair_income_then_occupation',
            'preprocessing': 'inherited FiniteScaler retains all positive variances, training-only moments',
            'split_salt': 'pcrl-method-portfolio-v1|attack-tune|20261008|',
            'model_seed': 2026101205+source_seed}


def freeze_refit_targets(validation_records, *, required_arms=('U0','RAW_JOINT')):
    """Freeze verifier targets from ATT scores before INNER truth access.

    Every arm/seed/view is a mandatory independent fit, even when its strongest
    reader aliases another view. The independent implementer owns actual fits.
    """
    targets = []
    for arm in required_arms:
        for seed in (0,1):
            matches = [record for record in validation_records if record['arm_id'] == arm and record['seed'] == seed]
            if len(matches) != 1:
                raise ValueError('missing or duplicate mandatory independent refit anchor')
            record = matches[0]
            for view in ('r1','r2','pair'):
                primary = record['ATT_validation'][view]['contracts']['complete_view']
                name = primary['strongest_reader']
                targets.append({'arm_id': arm, 'seed': seed, 'view': view, 'reader': name,
                                'freeze_sha256': record['freeze_sha256'],
                                'selected_on': 'ATT validation only', 'INNER_labels_opened': False})
    return {'schema': 'OB1-independent-refit-targets-v1', 'status': 'FROZEN', 'targets': targets}
