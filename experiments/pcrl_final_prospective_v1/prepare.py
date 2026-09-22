"""Build label-free per-anchor feature caches and pool-gated label caches.

Datasets:
  acs2016   -- the admitted 2016 cohort and its committed household partition.
  emulated  -- 2018 (already used) people resampled by household to the 2016 pool
               sizes. Timing and integration device only; replicated rows are never
               statistical observations and no emulated number supports a claim.
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd
import torch

from .common import ANCHORS, PRIVATE, array_hash, atomic_json, atomic_npz, now, read_json
from .releases import FrozenReleases
from .transport import (CSV_2016, CSV_2018, FIXED, FrozenServices, LABEL_COLUMNS, load_cohort_2016,
                        read_features, read_labels)

POOL_MAP = {'fit': 'fitting', 'attack_val': 'validation/attacker_validation',
            'task_val': 'validation/task_validation', 'final': 'final_evaluation'}
LABEL_KEYS = ('SEX', 'RAC1P', 'same_residence', 'public_coverage', 'civilian_at_work', 'income_binary', 'commute_over20')


def data_dir(dataset):
    return PRIVATE/'data'/dataset


def encoded_arrays(frozen, x, ha):
    e = frozen.encode(x, ha)
    return {'p': e['p'], 'b': e['b'], 'r': e['r'], 'codes_T0': e['codes']['T0'], 'codes_Ttask': e['codes']['Ttask'],
            'codes_Trisk': e['codes']['Trisk'], 'actions17': e['actions'][17], 'actions33': e['actions'][33],
            'global_offsets': e['global_offsets']}


def write_anchor(dataset, anchor, arrays):
    atomic_npz(data_dir(dataset)/f'anchor_{anchor}'/'features.npz', **arrays)
    return {k: array_hash(v) for k, v in arrays.items()}


def build_2016(anchors=ANCHORS):
    frame, index, cohort_meta = load_cohort_2016()
    out = data_dir('acs2016')
    pools = {k: index[v] for k, v in POOL_MAP.items()}
    atomic_npz(out/'cohort.npz', ids=frame.person_id.to_numpy(str), households=frame.household_id.to_numpy(str),
               weights=frame.PWGTP.to_numpy(float), raw_rows=frame._raw_row.to_numpy(),
               **{f'pool_{k}': v for k, v in pools.items()})
    record = {'created_utc': now(), 'dataset': 'acs2016', 'cohort': cohort_meta,
              'pools': {k: {'people': len(v), 'households': int(frame.household_id.iloc[v].nunique())} for k, v in pools.items()},
              'anchors': {}}
    for anchor in anchors:
        torch.set_num_threads(1)
        svc = FrozenServices(anchor)
        t = svc.transform(frame)
        frozen = FrozenReleases(anchor)
        arrays = {'pca': t['pca'], 'x': t['x'], 'ha': t['ha'], 'hb': t['hb'], 'J': t['J'],
                  **encoded_arrays(frozen, t['x'], t['ha'])}
        record['anchors'][anchor] = write_anchor('acs2016', anchor, arrays)
    atomic_json(out/'PREPARED.json', record)
    return record


def labels_2016(pool_names):
    out = data_dir('acs2016')
    with np.load(out/'cohort.npz') as z:
        raw = z['raw_rows']; pools = {k: z[f'pool_{k}'] for k in POOL_MAP}
    records = {}
    for name in pool_names:
        ix = pools[name]
        rows = raw[ix]
        order = np.argsort(rows)
        frame, labels = read_labels(rows[order], pool=POOL_MAP[name], csv=CSV_2016)
        inverse = np.empty_like(order); inverse[order] = np.arange(len(order))
        arrays = {k: np.asarray(labels[k])[inverse] for k in LABEL_KEYS}
        atomic_npz(out/f'labels_{name}.npz', **arrays)
        records[name] = {k: array_hash(v) for k, v in arrays.items()}
    return records


EMULATED_SIZES = {'fit': 39697, 'attack_val': 7911, 'task_val': 8006, 'final': 23684}
SMOKE_SIZES = {'fit': 900, 'attack_val': 400, 'task_val': 400, 'final': 600}


def build_emulated(anchors=ANCHORS, seed=20260922, *, name='emulated', sizes=EMULATED_SIZES):
    """2018 rows resampled by household to the 2016 pool sizes (timing device only)."""
    base0 = FIXED/'seed_0'
    with np.load(base0/'split_rows.npz') as z:
        rows = np.sort(np.concatenate([z[p] for p in z.files]))
    keys = pd.read_csv(CSV_2018, usecols=['SERIALNO', 'SPORDER', 'PWGTP'], dtype={'SERIALNO': str}).iloc[rows]
    households = keys.SERIALNO.to_numpy(str)
    uniq, inverse = np.unique(households, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    person, hh_ids, pool_index, cursor = [], [], {}, 0
    for pool_name, target in sizes.items():
        start = len(person)
        while len(person)-start < target:
            g = int(rng.integers(len(uniq)))
            take = members[g][:target-(len(person)-start)]
            person.extend(take.tolist())
            hh_ids.extend([f'emu|{cursor}|{uniq[g]}']*len(take))
            cursor += 1
        pool_index[pool_name] = np.arange(start, len(person))
    person = np.asarray(person)
    ids = np.array([f'{h}|{keys.SPORDER.iloc[p]}' for h, p in zip(hh_ids, person)])
    out = data_dir(name)
    atomic_npz(out/'cohort.npz', ids=ids, households=np.asarray(hh_ids), weights=keys.PWGTP.to_numpy(float)[person],
               raw_rows=rows[person], **{f'pool_{k}': v for k, v in pool_index.items()})
    frame, labels = read_labels(rows, pool='emulated_2018_used', csv=CSV_2018)
    for pool_name, ix in pool_index.items():
        atomic_npz(out/f'labels_{pool_name}.npz', **{k: np.asarray(labels[k])[person[ix]] for k in LABEL_KEYS})
    import torch as _t
    for anchor in anchors:
        base = FIXED/f'seed_{anchor}'
        with np.load(base/'split_rows.npz') as z, np.load(base/'pca.npz') as pz, np.load(base/'anchors.npz') as az, \
                np.load(base/'training/J/releases.npz') as jz:
            pools = z.files
            r = np.concatenate([z[p] for p in pools]); order = np.argsort(r)
            if not np.array_equal(r[order], rows):
                raise ValueError('Anchor cohorts differ')
            pca = np.concatenate([pz[p] for p in pools])[order]
            ha = np.concatenate([az[p+'/A'] for p in pools])[order]
            hb = np.concatenate([az[p+'/B'] for p in pools])[order]
            j = np.concatenate([jz[f'wire/A/{p}'][:, 4:] for p in pools])[order]
        state = _t.load(base/'training/J/final.pt', map_location='cpu', weights_only=False)['model_state']
        x = np.asarray((np.asarray(pca, np.float64)-state['input_mean'].numpy())/state['input_scale'].numpy(), np.float32)
        frozen = FrozenReleases(anchor)
        arrays = {'pca': pca[person], 'x': x[person], 'ha': ha[person], 'hb': hb[person], 'J': j[person]}
        enc = encoded_arrays(frozen, x, ha)
        arrays.update({k: v[person] for k, v in enc.items()})
        write_anchor(name, anchor, arrays)
    atomic_json(out/'PREPARED.json', {'created_utc': now(), 'dataset': name, 'seed': seed, 'sizes': sizes,
        'rule': '2018 whole households resampled with replacement to 2016 pool sizes; timing/integration only',
        'distinct_source_households': int(len(set(h.split('|')[2] for h in hh_ids)))})


def load_pools(dataset, anchor, pools, *, with_labels=True):
    """Return {pool: {x, ha, hb, J, ids, households, weights, labels, raw_rows}} and encoded arrays."""
    base = data_dir(dataset)
    with np.load(base/'cohort.npz') as z:
        cohort = {k: z[k] for k in z.files}
    with np.load(base/f'anchor_{anchor}'/'features.npz') as z:
        feats = {k: z[k] for k in z.files}
    out, enc = {}, {}
    for pool in pools:
        ix = cohort[f'pool_{pool}']
        d = {'x': feats['x'][ix], 'ha': feats['ha'][ix], 'hb': feats['hb'][ix], 'J': feats['J'][ix],
             'ids': cohort['ids'][ix], 'households': cohort['households'][ix], 'weights': cohort['weights'][ix],
             'raw_rows': cohort['raw_rows'][ix]}
        if with_labels:
            path = base/f'labels_{pool}.npz'
            if pool == 'final' and dataset == 'acs2016':
                from .transport import lock_verified
                if not lock_verified():
                    raise PermissionError('Final labels sealed until the evaluation lock verifies')
            with np.load(path) as z:
                d['labels'] = {k: z[k] for k in z.files}
        out[pool] = d
        enc[pool] = {'p': feats['p'][ix], 'b': feats['b'][ix], 'r': feats['r'][ix],
                     'codes': {'T0': feats['codes_T0'][ix], 'Ttask': feats['codes_Ttask'][ix], 'Trisk': feats['codes_Trisk'][ix]},
                     'actions': {17: feats['actions17'][ix], 33: feats['actions33'][ix]},
                     'global_offsets': feats['global_offsets'][ix]}
    return out, enc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('acs2016', 'emulated', 'smoke'), required=True)
    ap.add_argument('--labels', nargs='*', default=())
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.dataset == 'emulated':
        build_emulated()
    elif a.dataset == 'smoke':
        build_emulated(name='smoke', sizes=SMOKE_SIZES)
    elif a.labels:
        print(labels_2016(a.labels))
    else:
        print(build_2016())
