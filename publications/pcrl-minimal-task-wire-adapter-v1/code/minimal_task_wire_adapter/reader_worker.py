"""Fresh complete-view readers on actual transmitted wires (queued, label-free INNER).

The pinned slate is the inherited 18-reader complete-view bank (9 base
task_conditioned_representation.attacks.READERS + 9 extended contracts). Readers
train on inherited ATT train groups with SEX, are oriented on ATT validation
groups only, and score INNER from features alone. INNER labels never enter.

operations:
  r1_bank     income view once per seed (the income wire is identical in every
              service; each service unit verifies the exact alias by hash)
  service     occupation and aligned-pair views for one service/seed
  controls    null (group-permuted SEX) and planted (appended SEX bit) banks on
              the RAW_PROJ wire, three views; diagnostic-only wires
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path


def arr_sha(a):
    import numpy as np
    z = np.ascontiguousarray(a)
    return hashlib.sha256(str(z.dtype).encode() + str(z.shape).encode() + z.tobytes()).hexdigest()


def slate():
    from task_conditioned_representation.attacks import READERS
    from compressed_income_safeguard.reader_specs import EXTENDED_READERS
    names = tuple(READERS) + tuple(EXTENDED_READERS)
    if len(names) != 18 or len(set(names)) != 18:
        raise PermissionError('pinned complete-view slate must have exactly 18 readers')
    return names


def _role(item, verify, fields):
    import numpy as np
    from zipfile import ZipFile
    p = verify(item['path'], item['sha256'])
    with ZipFile(p) as archive:
        if {x.filename for x in archive.infolist()} != {f + '.npy' for f in fields}:
            raise PermissionError('unexpected role export fields')
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in fields}


def build_wires(cfg, verify, service, X, names):
    import json as _json
    from task_conditioned_representation.models import load_checkpoint
    from .wires import prepare_geometry
    from .services import service_forward, load_adapter_checkpoint
    seed = cfg['seed']
    g = cfg['geometry']
    geometry = prepare_geometry(_json.loads(verify(g['path'], g['sha256']).read_text()), expected_sha256=g['binding_sha256'], source_seed=seed)
    c = cfg['checkpoint']
    start, _ = load_checkpoint(verify(c['path'], c['sha256']), c['sha256'])
    kind = service['kind']
    if kind in ('RAW_PROJ', 'RAW_HYBRID'):
        return service_forward(kind, X, names, model=start, geometry=geometry)
    if kind in ('PRIOR_HYBRID', 'OWNPROJ'):
        pm, _ = load_checkpoint(verify(service['checkpoint']['path'], service['checkpoint']['sha256']), service['checkpoint']['sha256'])
        own = None
        if kind == 'OWNPROJ':
            own = _json.loads(verify(service['ownproj']['path'], service['ownproj']['sha256']).read_text())
        return service_forward(kind, X, names, model=pm, geometry=geometry, ownproj=own)
    if kind == 'ADAPTER':
        ab = cfg['adapter_binding']; binding = _json.loads(verify(ab['path'], ab['sha256']).read_text())
        payload = load_adapter_checkpoint(verify(service['checkpoint']['path'], service['checkpoint']['sha256']),
                                          service['checkpoint']['sha256'], seed=seed, source_checkpoint_sha256=c['sha256'],
                                          adapter_binding_sha256=ab['sha256'])
        if payload['arm'] != service['arm'] or payload['epoch'] != service['epoch']:
            raise PermissionError('attacked adapter checkpoint identity differs from SELECTION_LOCK')
        return service_forward('ADAPTER', X, names, model=start, geometry=geometry, scale=float(binding['scale']),
                               adapter_state={'R': payload['R'], 'b': payload['b_res']})
    raise PermissionError('undeclared service kind')


def run(spec_path):
    from .evaluation_queue import queue_config
    active, cfg, verify, guard = queue_config(spec_path, 'attacks')
    sel = verify(cfg['selection_lock'], cfg['selection_lock_sha256'])
    lock = json.loads(sel.read_text())
    if lock.get('status') != 'NOMINATIONS_AND_ATTACK_ROSTER_FROZEN' or lock.get('INNER_truth_accessed') is not False:
        raise PermissionError('reader fits require the frozen SELECTION_LOCK')
    import numpy as np
    import joblib
    from task_conditioned_representation.attacks import (MODEL_SEED_BASE, ATT_NULL_SEED, attack_split, fit_reader,
                                                         predict_reader, orientation, orient, group_shuffle)
    from task_conditioned_representation.inference import auc
    from compressed_income_safeguard.readers import tables, kind_for
    names_slate = slate()
    seed = cfg['seed']
    att = _role(cfg['att_features'], verify, ('row_id', 'unit', 'X', 'feature_names'))
    sex = _role(cfg['att_sex'], verify, ('row_id', 'unit', 'sex'))
    inner = _role(cfg['inner_features'], verify, ('row_id', 'unit', 'X', 'feature_names'))
    if not (np.array_equal(att['row_id'], sex['row_id']) and np.array_equal(att['unit'], sex['unit'])):
        raise PermissionError('ATT SEX pairing differs')
    y = sex['sex'].astype(np.int64)
    train, val = attack_split(att['unit'])
    if y.shape != (4064,) or len(inner['row_id']) != 2235 or any(np.unique(y[ix]).size != 2 for ix in (train, val)):
        raise PermissionError('ATT/INNER rows or SEX classes differ')
    out = Path(cfg['private_output']).resolve()
    if '.private' not in out.parts or out.exists():
        raise PermissionError('fresh private reader directory required')
    out.mkdir(parents=True, mode=0o700)
    op = cfg['operation']; service = cfg['service']
    guard()
    wa = build_wires(cfg, verify, service, att['X'], att['feature_names'])
    wi = build_wires(cfg, verify, service, inner['X'], inner['feature_names'])
    r1a, r2a, r1i, r2i = wa[0]['wire'], wa[1]['wire'], wi[0]['wire'], wi[1]['wire']
    for r in (r1a, r2a, r1i, r2i):
        if not np.isfinite(r).all():
            raise ValueError('nonfinite wire')
    hashes = {'ATT_r1': arr_sha(r1a), 'ATT_r2': arr_sha(r2a), 'INNER_r1': arr_sha(r1i), 'INNER_r2': arr_sha(r2i)}
    if op in ('service', 'controls') and (hashes['ATT_r1'] != cfg['r1_alias']['ATT_r1'] or hashes['INNER_r1'] != cfg['r1_alias']['INNER_r1']):
        raise PermissionError('income wire is not the exact r1 bank alias')
    # Duplicate groups: canonical releases must agree within a group.
    for wire, units in ((r2a, att['unit']), (r2i, inner['unit'])):
        for gid in np.unique(units):
            ix = np.flatnonzero(units == gid)
            if len(ix) > 1 and not np.array_equal(wire[ix], np.repeat(wire[ix[:1]], len(ix), 0)):
                raise PermissionError('duplicate-group releases differ')
    tab = tables(r1a, r2a); itab = tables(r1i, r2i)
    rel = out / 'INNER_RELEASE.npz'
    with rel.open('xb') as f:
        np.savez(f, row_id=inner['row_id'], unit=inner['unit'], r1=r1i, r2=r2i)
    views = {'r1_bank': ('r1',), 'service': ('r2', 'pair'), 'controls': ('r1', 'r2', 'pair')}[op]
    scores = {}; validation = {}; models = {}; accounting = []; control_scores = {}

    def fit_one(x, labels, name, view, tag):
        guard()
        t = time.perf_counter(); cpu = time.process_time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            record = fit_reader(x, labels, name, MODEL_SEED_BASE + seed)
        path = out / f'MODEL_{tag}{view}__{name}.joblib'
        joblib.dump(record, path, compress=3); path.chmod(0o600)
        models[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        accounting.append({'view': view, 'reader': name, 'tag': tag or 'DIRECT', 'train_rows': len(x), 'columns': int(x.shape[1]),
                           'cpu_s': time.process_time() - cpu, 'wall_s': time.perf_counter() - t,
                           'warnings': sorted({str(w.message)[:120] for w in caught})})
        return record
    if op in ('r1_bank', 'service'):
        for view in views:
            validation[view] = {}
            for name in names_slate:
                kind = kind_for(name)
                rec = fit_one(tab[view][kind][train], y[train], name, view, '')
                raw = predict_reader(rec, tab[view][kind][val]); sign = orientation(y[val], raw)
                validation[view][name] = {'positive_auc': auc(y[val], raw), 'orientation': sign,
                                          'fixed_orientation_auc': auc(y[val], orient(raw, sign)), 'contract': kind}
                scores[f'{view}__{name}'] = orient(predict_reader(rec, itab[view][kind]), sign)
    else:
        shuffled = group_shuffle(y[train], att['unit'][train], ATT_NULL_SEED)
        for view in views:
            for name in names_slate:
                kind = kind_for(name)
                rec = fit_one(tab[view][kind][train], shuffled, name, view, 'SHUFFLED_')
                control_scores[f'{view}__SHUFFLED__{name}'] = predict_reader(rec, itab[view][kind])
                planted = np.column_stack((tab[view][kind][train], y[train]))
                rec = fit_one(planted, y[train], name, view, 'PLANTED_')
                for bit in (0, 1):
                    control_scores[f'{view}__PLANTED_{bit}__{name}'] = predict_reader(
                        rec, np.column_stack((itab[view][kind], np.full(len(inner['row_id']), bit))))
                    validation.setdefault(view, {})[name] = {'planted_att_val_auc': auc(y[val], predict_reader(
                        rec, np.column_stack((tab[view][kind][val], y[val]))))}
    sp = out / 'INNER_SCORES.npz'
    with sp.open('xb') as f:
        np.savez(f, row_id=inner['row_id'], unit=inner['unit'], **scores, **control_scores)
    receipt = {'schema': 'MTA1-fresh-reader-freeze-v1', 'status': 'FROZEN_COMPLETE', 'queue_unit_id': active['unit_id'],
               'operation': op, 'seed': seed, 'service_label': cfg['service_label'], 'service': {k: v for k, v in service.items() if k not in ('checkpoint', 'ownproj')},
               'service_checkpoint_sha256': (service.get('checkpoint') or {}).get('sha256'),
               'readers': list(names_slate), 'views': list(views), 'model_seed': MODEL_SEED_BASE + seed,
               'ATT_split': {'train_rows': int(len(train)), 'validation_rows': int(len(val)),
                             'train_groups': int(len(np.unique(att['unit'][train]))), 'validation_groups': int(len(np.unique(att['unit'][val])))},
               'wire_hashes': hashes, 'ATT_validation': validation, 'model_files': models,
               'inner_release_sha256': hashlib.sha256(rel.read_bytes()).hexdigest(),
               'inner_scores_sha256': hashlib.sha256(sp.read_bytes()).hexdigest(),
               'numerical_fit_count': len(accounting), 'accounting': accounting,
               'INNER_labels_opened_or_hashed': False, 'orientation_selection': 'ATT validation groups only',
               'sklearn_version': __import__('sklearn').__version__, 'numpy_version': np.__version__}
    with (out / 'FROZEN_COMPLETE.json').open('x') as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    with Path(cfg['output']).open('x') as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)
