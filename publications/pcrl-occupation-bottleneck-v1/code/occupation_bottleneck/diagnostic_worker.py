"""Interpretation-only task association and supported-stratum diagnostics.

Pre-report science trains on FIT folds0/1 and scores FIT fold2. The source
encoder saw all FIT: this is exploratory development, never an unseen cohort.
True task labels never enter a primary attack or a deployed release.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

MIN_SEX_GROUPS_PER_STRATUM = 32


def fit_label_tables(income, occupation, sex):
    import numpy as np
    table = {}; counts = {}
    for view, size, labels in (('r1', 2, income), ('r2', 6, occupation),
                                ('pair', 12, np.asarray(income)*6+occupation)):
        y = np.asarray(sex); labels = np.asarray(labels)
        if labels.shape != y.shape or not np.isin(y, [0, 1]).all() or np.any(labels < 0) or np.any(labels >= size):
            raise ValueError('task-label diagnostic class order differs')
        count = np.zeros((size, 2), dtype=np.int64)
        np.add.at(count, (labels, y), 1)
        table[view] = (count[:, 1]+1)/(count.sum(1)+2)
        counts[view] = count
    return table, counts


def task_association(table, income, occupation, sex, unit):
    import numpy as np
    from task_conditioned_representation.inference import auc
    result = {}; scores = {}
    for view, size, labels in (('r1', 2, income), ('r2', 6, occupation),
                                ('pair', 12, np.asarray(income)*6+occupation)):
        labels = np.asarray(labels)
        scores[view] = np.asarray(table[view])[labels]
        result[view] = {'auc': auc(sex, scores[view]),
                        'score_orientation': 'fixed positive SEX probability from FIT folds0/1 table; no score-row reversal',
                        'rows': len(labels), 'groups': len(np.unique(unit)),
                        'input': 'true task labels; interpretation only, forbidden deployed input'}
    return result, scores


def within_task_strata(scores, sex, unit, income, occupation):
    import numpy as np
    from task_conditioned_representation.inference import auc
    from .readers import READERS
    y = np.asarray(sex); g = np.asarray(unit); result = {}
    for view, size, labels in (('r1', 2, income), ('r2', 6, occupation),
                                ('pair', 12, np.asarray(income)*6+occupation)):
        labels = np.asarray(labels)
        candidates = {name.split('__', 1)[1]: np.asarray(value) for name, value in scores.items()
                      if name.startswith(view+'__') and name.split('__')[-1] in READERS}
        strata = []
        for value in range(size):
            mask = labels == value
            support = [len(np.unique(g[mask & (y == protected)])) for protected in (0, 1)]
            item = {'class_or_pair_id': value, 'rows': int(mask.sum()), 'SEX_group_counts': support,
                    'minimum_groups_each_SEX': MIN_SEX_GROUPS_PER_STRATUM}
            if min(support) < MIN_SEX_GROUPS_PER_STRATUM:
                item.update(status='MISSING_SPARSE_STRATUM', auc=None, reader_auc=None)
            elif not candidates:
                item.update(status='MISSING_FROZEN_READER_SCORES', auc=None, reader_auc=None)
            else:
                values = {name: auc(y[mask], values[mask]) for name, values in candidates.items()}
                item.update(status='ESTIMATED_DESCRIPTIVE', auc=max(values.values()), reader_auc=values,
                            strongest_reader=max(values, key=values.get))
            strata.append(item)
        result[view] = strata
    return result


def development_worker(spec_path):
    from .evaluation_worker import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'admission')
    import numpy as np
    from .source import load_manifest, load_role
    from task_conditioned_representation.attacks import save_npz, write_json
    manifest = load_manifest(verify(cfg['manifest'], cfg['manifest_sha256']), cfg['manifest_sha256'])
    fit = load_role(manifest, cfg.get('seed', 0), 'OSF_DEFENSE_FIT', {'features', 'income', 'occupation', 'sex'})
    training = np.flatnonzero(fit['fold'] != 2); selected = np.flatnonzero(fit['fold'] == 2)
    if not len(training) or not len(selected) or set(fit['unit'][training]) & set(fit['unit'][selected]):
        raise PermissionError('FIT diagnostic train/selection duplicate groups overlap')
    guard()
    table, counts = fit_label_tables(fit['income'][training], fit['occupation'][training], fit['sex'][training])
    association, table_scores = task_association(table, fit['income'][selected], fit['occupation'][selected], fit['sex'][selected], fit['unit'][selected])
    root = Path(cfg['private_output'])
    if '.private' not in root.resolve().parts or root.exists():
        raise PermissionError('fresh private C3 output required')
    root.mkdir(parents=True, mode=0o700)
    table_sha = save_npz(root/'FIT_LABEL_TABLES.npz', **table, **{view+'__counts': value for view, value in counts.items()})
    score_sha = save_npz(root/'FIT_SELECTION_LABEL_SCORES.npz', row_id=fit['row_id'][selected], unit=fit['unit'][selected], **table_scores)
    frozen_banks = []
    for bank in cfg.get('banks', []):
        freeze = json.loads(verify(bank['freeze']['path'], bank['freeze']['sha256']).read_text())
        if freeze.get('status') != 'FROZEN_COMPLETE':
            raise PermissionError('C3 bank not frozen')
        for name, field in (('release', 'FIT_selection_release_sha256'), ('scores', 'FIT_selection_scores_sha256')):
            pin = bank[name]
            verify(pin['path'], pin['sha256'])
            if freeze.get(field) != pin['sha256']:
                raise PermissionError('C3 score/release differs from exact frozen bank')
        with np.load(bank['release']['path'], allow_pickle=False) as z:
            if set(z.files) != {'row_id', 'unit', 'r1', 'r2'} or any(not np.array_equal(z[k], fit[k][selected]) for k in ('row_id', 'unit')):
                raise PermissionError('C3 frozen reader score-row identities differ')
        with np.load(bank['scores']['path'], allow_pickle=False) as z:
            scores = {key: z[key].copy() for key in z.files}
        from .readers import score_envelopes
        frozen_banks.append({'arm_id': bank['arm_id'], 'seed': bank['seed'],
                             'freeze_sha256': bank['freeze']['sha256'],
                             'scores_sha256': bank['scores']['sha256'],
                             'recovery': score_envelopes(scores, fit['sex'][selected]),
                             'within_true_task_strata': within_task_strata(scores, fit['sex'][selected], fit['unit'][selected], fit['income'][selected], fit['occupation'][selected])})
    out = {'schema': 'OB1-task-association-development-v1', 'status': 'FROZEN_COMPLETE',
           'unit_id': active['unit_id'], 'table_sha256': table_sha, 'score_sha256': score_sha,
           'training_role': 'FIT folds0/1', 'scoring_role': 'FIT fold2',
           'training_rows': len(training), 'scoring_rows': len(selected),
           'training_groups': len(np.unique(fit['unit'][training])),
           'scoring_groups': len(np.unique(fit['unit'][selected])),
           'smoothing': 'Laplace1 per binary SEX class per task class/label pair',
           'task_association': association, 'reader_banks': frozen_banks,
           'sparse_support_rule': 'at least32 unique groups of each SEX per stratum, frozen before fits; no pooling',
           'ATT_task_label_exports': 'ABSENT; no all-role export read or created; separate FIT folds supply diagnostic score rows',
           'interpretation_only': True, 'true_labels_absent_primary_inputs': True,
           'source_training_saw_all_FIT': True, 'exploratory_shared_training_history': True,
           'universal_privacy_floor_established': False, 'INNER_labels_opened_or_hashed': False}
    write_json(root/'FROZEN_COMPLETE.json', out); write_json(cfg['public_output'], out, False)
    return out


def reporting_worker(spec_path):
    from .evaluation_worker import queue_config, require_reporting
    active, cfg, verify, guard = queue_config(spec_path, 'attacks')
    seal, lock = require_reporting(cfg, verify)
    import numpy as np
    from .source import load_manifest, load_role
    from .readers import score_envelopes
    from task_conditioned_representation.attacks import write_json, save_npz
    freeze = json.loads(verify(cfg['table_freeze']['path'], cfg['table_freeze']['sha256']).read_text())
    if freeze.get('status') != 'FROZEN_COMPLETE' or freeze.get('training_role') != 'FIT folds0/1':
        raise PermissionError('C3 original development table freeze differs')
    verify(cfg['table']['path'], cfg['table']['sha256'])
    if freeze['table_sha256'] != cfg['table']['sha256']:
        raise PermissionError('C3 table differs from frozen pre-INNER table')
    with np.load(cfg['table']['path'], allow_pickle=False) as z:
        table = {view: z[view].copy() for view in ('r1','r2','pair')}
    manifest = load_manifest(verify(cfg['manifest'], cfg['manifest_sha256']), cfg['manifest_sha256'])
    # All banks and their score/release hashes are admitted before any truth load.
    for bank in cfg.get('banks', []):
        reader = json.loads(verify(bank['freeze']['path'], bank['freeze']['sha256']).read_text())
        if reader.get('status') != 'FROZEN_COMPLETE':
            raise PermissionError('C3 report bank not frozen')
        for name, field in (('release','inner_release_sha256'), ('scores','score_sha256')):
            verify(bank[name]['path'], bank[name]['sha256'])
            if reader[field] != bank[name]['sha256']:
                raise PermissionError('C3 report frozen bank input differs')
    truth = load_role(manifest, cfg.get('seed',0), 'INNER_SELECTION', {'features','income','occupation','sex'}, reporting_seal=seal)
    association, scores = task_association(table, truth['income'], truth['occupation'], truth['sex'], truth['unit'])
    banks = []
    for bank in cfg.get('banks', []):
        with np.load(bank['release']['path'], allow_pickle=False) as z:
            if set(z.files) != {'row_id','unit','r1','r2'} or any(not np.array_equal(z[key], truth[key]) for key in ('row_id','unit')):
                raise PermissionError('C3 reporting bank and truth people differ')
        with np.load(bank['scores']['path'], allow_pickle=False) as z:
            reader_scores = {key: z[key].copy() for key in z.files}
        banks.append({'arm_id': bank['arm_id'], 'seed': bank['seed'],
                      'freeze_sha256': bank['freeze']['sha256'],
                      'recovery': score_envelopes(reader_scores, truth['sex']),
                      'within_true_task_strata': within_task_strata(reader_scores, truth['sex'], truth['unit'], truth['income'], truth['occupation'])})
    root = Path(cfg['private_output'])
    if '.private' not in root.resolve().parts or root.exists():
        raise PermissionError('fresh private C3 reporting output required')
    root.mkdir(parents=True, mode=0o700)
    score_sha = save_npz(root/'INNER_LABEL_TABLE_SCORES.npz', row_id=truth['row_id'], unit=truth['unit'], **scores)
    out = {'schema': 'OB1-frozen-task-association-INNER-v1', 'unit_id': active['unit_id'],
           'task_association': association, 'reader_banks': banks,
           'table_sha256': cfg['table']['sha256'], 'score_sha256': score_sha,
           'reporting_lock_sha256': seal['sha256'], 'scoring_role': 'INNER_SELECTION',
           'training_role': 'FIT folds0/1', 'orientation_or_fit_changed_after_INNER': False,
           'interpretation_only': True, 'universal_privacy_floor_established': False,
           'sparse_support_rule': 'at least32 groups each SEX, no pooling', 'exploratory_assessment': True}
    write_json(cfg['public_output'], out, False)
    return out


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    parser.add_argument('--phase', choices=('development','report'), default='development')
    args = parser.parse_args(); out = development_worker(args.spec) if args.phase == 'development' else reporting_worker(args.spec)
    print(json.dumps({'status': out.get('status','REPORTED'), 'unit_id': out['unit_id']}))

if __name__ == '__main__':
    main()
