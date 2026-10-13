"""Frozen development readers for the occupation bottleneck campaign.

Numerical calls require the coordinator's authenticated queue guard. Importing
this module neither loads a checkpoint nor opens any person array.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import joblib
from task_conditioned_representation.attacks import (
    READERS, MODEL_SEED_BASE, ATT_NULL_SEED, FiniteScaler, attack_split,
    feature_tables as predecessor_tables, fit_reader, predict_reader,
    orientation, orient, group_shuffle, sha_file, save_npz, write_json,
)
from task_conditioned_representation.inference import auc

EXTENDED_READERS = tuple(f'{contract}_{model}' for contract in
                         ('LOGITS_OUTPUT', 'DECISION', 'CONFIDENCE')
                         for model in ('LR_C1', 'MLP_64x32', 'HGB_15x200'))
CONTRACTS = {'LOGITS_OUTPUT': 'logits_output', 'DECISION': 'decision',
             'CONFIDENCE': 'confidence'}


def tables(r1, r2):
    """True labels and identities are never part of a recipient table.

    The predecessor's OUTPUT contract is probabilities + decision. The new
    LOGITS_OUTPUT contract includes centered own logits as explicitly requested.
    CONFIDENCE contains max own probability alone and no decision.
    """
    result = predecessor_tables(r1, r2)
    for view, wire, classes in (('r1', r1, 2), ('r2', r2, 6)):
        wire = np.asarray(wire, dtype=np.float64)
        result[view]['logits_output'] = wire[:, 16:]
        result[view]['decision'] = wire[:, -1:]
        result[view]['confidence'] = wire[:, 16+classes:16+2*classes].max(1, keepdims=True)
    for kind in CONTRACTS.values():
        result['pair'][kind] = np.column_stack((result['r1'][kind], result['r2'][kind]))
    return result


def kind_for(name):
    for prefix, kind in CONTRACTS.items():
        if name.startswith(prefix+'_'):
            return kind
    from task_conditioned_representation.attacks import reader_kind
    return reader_kind(name)


def add_aliases(scores):
    out = {view: dict(records) for view, records in scores.items()}
    for view, other in (('r1', 'r2'), ('r2', 'r1')):
        for name, values in scores[view].items():
            out['pair'][f'IGNORE_{other}__{name}'] = values
    return out


def flat_scores(scores):
    return {f'{view}__{name}': values for view, records in scores.items()
            for name, values in records.items()}


def compatible_reuse(receipt, expected):
    """Metadata admission only. A source checkpoint hash alone is insufficient.

    Every named identity must match; the caller verifies receipt/model/array
    hashes through the new queue before numerical replay. No old queue opens.
    """
    if receipt.get('status') != 'FROZEN_COMPLETE':
        return False
    actual = receipt.get('bindings') or {}
    return all(actual.get(key) == value for key, value in expected.items())


def reused_view_records(directory, view, att_table, inner_table, names, *,
                        queue_guard, expected_bindings=None, saved_inner=False):
    """Replay an exact admitted bank, retaining its ATT-frozen orientation.

    Model deserialization is queue-owned. All paths must already be exact
    sources registered by the coordinator; guard verifies them before entry.
    """
    queue_guard()
    root = Path(directory)
    receipt = json.loads((root/'FROZEN_COMPLETE.json').read_text())
    if expected_bindings is not None and not compatible_reuse(receipt, expected_bindings):
        raise PermissionError('reader checkpoint/wire/role/preprocessing/split/seed identity differs')
    validation = receipt['validation'][view]
    scores, inner_scores, entries = {}, {}, {}
    preserved_inner = None
    if saved_inner and inner_table is not None:
        if sha_file(root/'INNER_SCORES.npz') != receipt['score_sha256']:
            raise PermissionError('reused saved INNER prediction hash differs')
        with np.load(root/'INNER_SCORES.npz', allow_pickle=False) as z:
            preserved_inner = {name: z[view+'__'+name].copy() for name in names}
    for name in names:
        model_path = root/f'MODEL_{view}__{name}.joblib'
        if receipt['model_files'].get(model_path.name) != sha_file(model_path):
            raise PermissionError('reused reader model hash differs')
        record = joblib.load(model_path)
        sign = validation[name]['orientation']
        kind = kind_for(name)
        scores[name] = orient(predict_reader(record, att_table[kind]), sign)
        if inner_table is not None:
            inner_scores[name] = (preserved_inner[name] if preserved_inner is not None
                                  else orient(predict_reader(record, inner_table[kind]), sign))
        entries[name] = {'status': 'REUSED', 'model_sha256': receipt['model_files'][model_path.name],
                         'orientation': sign, 'source_freeze_sha256': sha_file(root/'FROZEN_COMPLETE.json')}
    return scores, inner_scores, entries


def assert_exact_alias(directory, view, att, inner, seed, names):
    """Compare authorized literal wires before any bank alias is accepted."""
    root = Path(directory)
    receipt = json.loads((root/'FROZEN_COMPLETE.json').read_text())
    if receipt.get('source_seed') != seed or receipt.get('model_seed') != MODEL_SEED_BASE+seed:
        raise PermissionError('exact wire alias requires inherited source/model seed')
    if tuple(receipt.get('readers', ())) != tuple(READERS) and receipt.get('schema') != 'OB1-development-reader-freeze-v1':
        raise PermissionError('unknown prior attack preprocessing/slate')
    if sha_file(root/'ATT_RELEASE.npz') != receipt['att_release_sha256']:
        raise PermissionError('alias ATT release hash differs')
    with np.load(root/'ATT_RELEASE.npz', allow_pickle=False) as z:
        if any(not np.array_equal(z[key], att[key]) for key in ('row_id', 'unit', 'sex')):
            raise PermissionError('alias ATT people/SEX differs')
        source_tables = tables(z['r1'], z['r2']); actual_tables = tables(att['r1'], att['r2'])
        if any(not np.array_equal(source_tables[view][kind_for(name)], actual_tables[view][kind_for(name)]) for name in names):
            raise PermissionError('alias ATT literal reader-contract wire differs')
        train, val = attack_split(att['unit'])
        if not np.array_equal(z['train'], train) or not np.array_equal(z['validation'], val):
            raise PermissionError('alias inherited ATT group split differs')
    if inner is not None:
        if sha_file(root/'INNER_RELEASE.npz') != receipt['inner_release_sha256']:
            raise PermissionError('alias INNER label-free release hash differs')
        from zipfile import ZipFile
        with ZipFile(root/'INNER_RELEASE.npz') as archive:
            if {item.filename for item in archive.infolist()} != {'row_id.npy', 'unit.npy', 'r1.npy', 'r2.npy'}:
                raise PermissionError('alias INNER archive must be label-free')
        with np.load(root/'INNER_RELEASE.npz', allow_pickle=False) as z:
            if any(not np.array_equal(z[key], inner[key]) for key in ('row_id', 'unit')):
                raise PermissionError('alias INNER people differ')
            source_tables = tables(z['r1'], z['r2']); actual_tables = tables(inner['r1'], inner['r2'])
            if any(not np.array_equal(source_tables[view][kind_for(name)], actual_tables[view][kind_for(name)]) for name in names):
                raise PermissionError('alias INNER literal reader-contract wire differs')


def fit_development_bank(att, inner=None, *, seed, private_output, queue_guard,
                         bindings, include_base=True, include_extended=False,
                         controls=False, view_aliases=None, diagnostic=None):
    """Fit on inherited ATT train groups; orient on ATT validation only.

    inner, when supplied, must contain features/wires and identities only.
    view_aliases maps a view to an exact prior bank directory plus expected
    bindings and names. A reused frozen income wire receives its existing
    predictions/seed once, including the pair ignore-income envelope.
    """
    queue_guard()
    if set(att) != {'row_id', 'unit', 'r1', 'r2', 'sex'}:
        raise PermissionError('ATT bank schema differs; task labels forbidden in reader inputs')
    if inner is not None and set(inner) != {'row_id', 'unit', 'r1', 'r2'}:
        raise PermissionError('INNER reader inference must be label-free')
    if diagnostic is not None and set(diagnostic) != {'row_id', 'unit', 'r1', 'r2'}:
        raise PermissionError('FIT selection reader inference must be label-free')
    if seed not in (0, 1):
        raise ValueError('source seed must be paired 0/1')
    root = Path(private_output)
    if '.private' not in root.resolve().parts or root.exists():
        raise PermissionError('fresh owner-only private reader output required')
    root.mkdir(parents=True, mode=0o700)
    y = np.asarray(att['sex'])
    train, val = attack_split(att['unit'])
    if y.shape != (4064,) or not np.isin(y, [0, 1]).all() or any(np.unique(y[ix]).size != 2 for ix in (train, val)):
        raise ValueError('ATT binary class/split differs')
    tab = tables(att['r1'], att['r2'])
    itab = None if inner is None else tables(inner['r1'], inner['r2'])
    dtab = None if diagnostic is None else tables(diagnostic['r1'], diagnostic['r2'])
    names = tuple(READERS if include_base else ()) + tuple(EXTENDED_READERS if include_extended else ())
    if not names:
        raise ValueError('no readers registered')
    att_hash = save_npz(root/'ATT_RELEASE.npz', **att, train=train, validation=val)
    inner_hash = None if inner is None else save_npz(root/'INNER_RELEASE.npz', **inner)
    diagnostic_hash = None if diagnostic is None else save_npz(root/'FIT_SELECTION_RELEASE.npz', **diagnostic)
    scores = {v: {} for v in tab}; inner_scores = {v: {} for v in tab}
    diagnostic_scores = {v: {} for v in tab}
    validations, models, records, reused, control_scores, inner_controls = {}, {}, [], {}, {}, {}
    def fit_one(view, name, x, labels, kind):
        queue_guard()
        started = time.perf_counter(); cpu = time.process_time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            record = fit_reader(x, labels, name, MODEL_SEED_BASE+seed)
        path = root/f'MODEL_{view}__{name}.joblib'
        joblib.dump(record, path, compress=3); path.chmod(0o600)
        models[path.name] = sha_file(path)
        records.append({'view': view, 'reader': name, 'input': kind, 'status': 'NEW',
                        'train_rows': len(x), 'columns': x.shape[1],
                        'cpu_s': time.process_time()-cpu, 'wall_s': time.perf_counter()-started,
                        'warnings': [str(w.message) for w in caught]})
        return record
    for view in tab:
        validations[view] = {}; reused[view] = {}
        alias = (view_aliases or {}).get(view)
        alias_names = set()
        if alias:
            alias_names = set(alias.get('names', names))
            assert_exact_alias(alias['directory'], view, att, inner, seed, alias_names)
            if not alias_names <= set(names):
                raise ValueError('unregistered alias readers')
            s, si, entries = reused_view_records(alias['directory'], view, tab[view],
                                                  None if itab is None else itab[view],
                                                  sorted(alias_names), queue_guard=queue_guard,
                                                  expected_bindings=alias['expected_bindings'], saved_inner=True)
            scores[view].update(s); inner_scores[view].update(si); reused[view] = entries
            if dtab is not None:
                _, ds, _ = reused_view_records(alias['directory'], view, tab[view], dtab[view],
                    sorted(alias_names), queue_guard=queue_guard, expected_bindings=alias['expected_bindings'])
                diagnostic_scores[view].update(ds)
            prior = json.loads((Path(alias['directory'])/'FROZEN_COMPLETE.json').read_text())
            validations[view].update({name: prior['validation'][view][name] for name in alias_names})
            for name in alias_names:
                if abs(auc(y[val], scores[view][name][val])-validations[view][name]['fixed_orientation_auc']) > 1e-12:
                    raise PermissionError('replayed ATT alias differs from frozen validation recovery')
        for name in names:
            if name in alias_names:
                continue
            kind = kind_for(name)
            record = fit_one(view, name, tab[view][kind][train], y[train], kind)
            raw_val = predict_reader(record, tab[view][kind][val]); sign = orientation(y[val], raw_val)
            validations[view][name] = {'positive_auc': auc(y[val], raw_val), 'orientation': sign,
                                      'fixed_orientation_auc': auc(y[val], orient(raw_val, sign))}
            scores[view][name] = orient(predict_reader(record, tab[view][kind]), sign)
            if itab is not None:
                inner_scores[view][name] = orient(predict_reader(record, itab[view][kind]), sign)
            if dtab is not None:
                diagnostic_scores[view][name] = orient(predict_reader(record, dtab[view][kind]), sign)
        if controls:
            shuffled = group_shuffle(y[train], att['unit'][train], ATT_NULL_SEED)
            record = fit_one(view, 'SHUFFLED', tab[view]['full'][train], shuffled, 'full')
            control_scores[f'{view}__SHUFFLED'] = predict_reader(record, tab[view]['full'])
            if itab is not None:
                inner_controls[f'{view}__SHUFFLED'] = predict_reader(record, itab[view]['full'])
            record = fit_one(view, 'PLANTED', np.column_stack((tab[view]['full'][train], y[train])), y[train], 'full+control_SEX_bit')
            for bit in (0, 1):
                control_scores[f'{view}__PLANTED_{bit}'] = predict_reader(record, np.column_stack((tab[view]['full'], np.full(len(y), bit))))
                if itab is not None:
                    inner_controls[f'{view}__PLANTED_{bit}'] = predict_reader(record, np.column_stack((itab[view]['full'], np.full(len(inner['unit']), bit))))
    scores = add_aliases(scores); inner_scores = add_aliases(inner_scores)
    diagnostic_scores = add_aliases(diagnostic_scores)
    diagnostic_scores_hash = None if diagnostic is None else save_npz(root/'FIT_SELECTION_SCORES.npz', **flat_scores(diagnostic_scores))
    att_scores_hash = save_npz(root/'ATT_SCORES.npz', **flat_scores(scores), **control_scores)
    inner_scores_hash = None if inner is None else save_npz(root/'INNER_SCORES.npz', **flat_scores(inner_scores), **inner_controls)
    view_report = score_envelopes(flat_scores(scores), y, indices=val)
    control_report = None
    if controls:
        control_report = {}
        null_labels = group_shuffle(y[val], att['unit'][val], ATT_NULL_SEED)
        for view in ('r1', 'r2', 'pair'):
            shuffled_auc = auc(null_labels, control_scores[view+'__SHUFFLED'][val])
            planted = np.where(y[val] == 1, control_scores[view+'__PLANTED_1'][val], control_scores[view+'__PLANTED_0'][val])
            values = {'null_auc': auc(y[val], np.full(len(val), .5)),
                      'shuffled_independent_auc': max(shuffled_auc, 1-shuffled_auc),
                      'planted_auc': auc(y[val], planted)}
            gates = {'null_exact': values['null_auc'] == .5,
                     'shuffle_at_most_0.60': values['shuffled_independent_auc'] <= .60,
                     'planted_at_least_0.99': values['planted_auc'] >= .99}
            control_report[view] = {'controls': values, 'gates': gates, 'passed': all(gates.values())}
    receipt = {'schema': 'OB1-development-reader-freeze-v1', 'status': 'FROZEN_COMPLETE',
               'source_seed': seed, 'bindings': bindings, 'readers': list(names),
               'model_seed': MODEL_SEED_BASE+seed, 'validation': validations,
               'model_files': models, 'reused_readers': reused,
               'att_release_sha256': att_hash, 'att_scores_sha256': att_scores_hash,
               'inner_release_sha256': inner_hash, 'score_sha256': inner_scores_hash,
               'FIT_selection_release_sha256': diagnostic_hash, 'FIT_selection_scores_sha256': diagnostic_scores_hash,
               'fit_rows': len(train), 'validation_rows': len(val),
               'fit_groups': len(np.unique(att['unit'][train])),
               'validation_groups': len(np.unique(att['unit'][val])),
               'numerical_fit_count': len(records), 'accounting': records,
               'ATT_validation': view_report, 'ATT_validation_controls': control_report,
               'real_data_control_fits_requested': controls, 'INNER_labels_opened_or_hashed': False,
               'contracts': {'full': 'z16+centered_logitsK+probabilitiesK+decision',
                             'output': 'legacy probabilitiesK+decision',
                             'logits_output': 'centered_logitsK+probabilitiesK+decision',
                             'decision': 'own numeric decision only',
                             'confidence': 'max own probability only'},
               'primary_envelope': 'all registered base readers plus exact local-ignore aliases',
               'orientation_selection': 'inherited ATT validation only',
               'numpy_version': np.__version__, 'sklearn_version': __import__('sklearn').__version__}
    write_json(root/'FROZEN_COMPLETE.json', receipt)
    return receipt


def score_envelopes(scores, sex, *, indices=None):
    y = np.asarray(sex)
    ix = np.arange(len(y)) if indices is None else np.asarray(indices)
    result = {}
    for view in ('r1', 'r2', 'pair'):
        values = {key.split('__', 1)[1]: auc(y[ix], np.asarray(s)[ix]) for key, s in scores.items()
                  if key.startswith(view+'__') and not key.split('__', 1)[1].startswith(('SHUFFLED', 'PLANTED'))}
        contracts = {}
        for contract in ('primary', 'representation', 'output', 'logits_output', 'decision', 'confidence'):
            selected = {name: value for name, value in values.items()
                        if ((name.split('__')[-1] in READERS) if contract == 'primary'
                            else kind_for(name.split('__')[-1]) == contract)}
            contracts[contract] = None if not selected else {
                'auc': max(selected.values()), 'strongest_reader': max(selected, key=selected.get),
                'reader_count': len(selected)}
        result[view] = {'reader_auc': values, 'contracts': contracts}
    for contract in result['pair']['contracts']:
        pair = result['pair']['contracts'][contract]
        locals_ = [result[v]['contracts'][contract] for v in ('r1', 'r2')]
        if pair is not None and any(local is not None and pair['auc']+1e-12 < local['auc'] for local in locals_):
            raise AssertionError('pair reader omitted an ignore-other-recipient alias')
    return result
