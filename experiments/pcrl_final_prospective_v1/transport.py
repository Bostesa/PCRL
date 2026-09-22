"""Label-free transport of frozen services, J and release inputs to a new ACS year.

The cohort and household partition are the committed 2016 admission rules
(evidence-paper branch 0d8f4b67, experiments/pcrl_evidence_review_v1/
admit_acs_2016.py), reproduced here verbatim in logic and verified against the
committed manifest's array hashes. Feature transformation never reads a label
column. Labels are read by a separate, pool-gated function; the final partition's
labels require a verified evaluation lock.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from experiments.acs_transfer_data import CovariatePreprocessor, FEATURES, CATEGORY_CODES
from experiments.acs_transfer_heads import load_candidate as load_service_head
from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs, labels_from_frame
from .common import INPUTS, OUT, ROOT, array_hash, read_json, sha

YEAR = 2016
STATE = 6
PARTITION_SALT = 'PCRL-evidence-review-2016-admission-v1'
POOLS = ('fitting', 'validation', 'final_evaluation')
FRACTIONS = (0.50, 0.20, 0.30)
SUBPOOLS = {'fitting': (('attacker_fit', 0.5), ('task_fit', 0.5)),
            'validation': (('attacker_validation', 0.5), ('task_validation', 0.5))}
KEYS = ('SERIALNO', 'SPORDER', 'PWGTP', 'ST', 'RT')
LABEL_COLUMNS = ('SEX', 'RAC1P', 'PUBCOV', 'PINCP', 'ESR', 'MIG', 'JWMNP')
SERVICE_TASKS = ('income_binary', 'civilian_at_work', 'public_coverage')
CSV_2016 = INPUTS/'data/acs_2016/ss16pca.csv'
CSV_2018 = INPUTS/'data/folktables/2018/1-Year/psam_p06.csv'
ADMISSION = INPUTS/'admission/ACS_2016_ADMISSION_MANIFEST.json'
FIXED = INPUTS/'results/redesign_20260909_acs_fixed_predictions_v1'
TRANSFER = INPUTS/'results/redesign_20260907_acs_transfer_v1'
LOCK = OUT/'EVALUATION_LOCK.json'


# ------------------------------------------------------------ cohort/partition
def eligible_cohort(raw):
    """Admission rule: 19<=AGEP<=34, PWGTP>0, identical duplicate keys dropped."""
    raw = raw.copy()
    raw['_raw_row'] = np.arange(len(raw), dtype=np.int64)
    raw['SPORDER'] = pd.to_numeric(raw.SPORDER).astype(int).astype(str)
    eligible = raw.AGEP.between(19, 34) & raw.PWGTP.gt(0)
    f = raw.loc[eligible].copy()
    if f.duplicated(['SERIALNO', 'SPORDER']).any():
        raise ValueError('Duplicate person keys: label-free replay cannot certify the admission dedup rule')
    return f.reset_index(drop=True)


def partition_households(frame):
    groups = sorted(frame.SERIALNO.unique(),
                    key=lambda s: (hashlib.sha256(f'{PARTITION_SALT}|{YEAR}|{STATE:02d}|{s}'.encode()).hexdigest(), s))
    n = len(groups)
    bounds = np.rint(np.cumsum((0.0,)+FRACTIONS)*n).astype(int)
    blocks = {name: groups[bounds[i]:bounds[i+1]] for i, name in enumerate(POOLS)}
    assignment = {}
    for name, members in blocks.items():
        if name in SUBPOOLS:
            sb = np.rint(np.cumsum((0.0,)+tuple(f for _, f in SUBPOOLS[name]))*len(members)).astype(int)
            for i, (sub, _) in enumerate(SUBPOOLS[name]):
                assignment[f'{name}/{sub}'] = members[sb[i]:sb[i+1]]
        assignment[name] = members
    index = {name: np.flatnonzero(frame.SERIALNO.isin(set(members))) for name, members in assignment.items()}
    top = np.concatenate([index[p] for p in POOLS])
    if len(np.unique(top)) != len(frame) or len(top) != len(frame):
        raise ValueError('Household partition is not a partition of the eligible frame')
    return index


def load_cohort_2016():
    """Features and keys only. Verifies the committed admission hashes exactly."""
    manifest = read_json(ADMISSION)
    if sha(CSV_2016) != manifest['provenance']['sha256']['ss16pca.csv']:
        raise ValueError('2016 CSV differs from admitted file')
    raw = pd.read_csv(CSV_2016, usecols=list(dict.fromkeys(FEATURES+KEYS)),
                      dtype={'SERIALNO': str, 'SPORDER': str, 'RT': str})
    if not raw.RT.eq('P').all() or not raw.ST.eq(STATE).all():
        raise ValueError('Wrong record type/state')
    frame = eligible_cohort(raw)
    index = partition_households(frame)
    expected = manifest['array_hashes']
    observed = {'serialno': array_hash(frame.SERIALNO.to_numpy(dtype=str)),
                'sporder': array_hash(frame.SPORDER.to_numpy(dtype=str)),
                'pwgtp': array_hash(frame.PWGTP.to_numpy(float)),
                'raw_row': array_hash(frame._raw_row.to_numpy()),
                **{f'partition/{k}': array_hash(v) for k, v in index.items()}}
    if observed != expected:
        bad = sorted(k for k in set(observed) | set(expected) if observed.get(k) != expected.get(k))
        raise ValueError(f'Admission hash replay failed: {bad}')
    frame['household_id'] = [f'{YEAR}|{s}' for s in frame.SERIALNO]
    frame['person_id'] = [f'{YEAR}|{s}|{o}' for s, o in zip(frame.SERIALNO, frame.SPORDER)]
    if frame.person_id.nunique() != len(frame):
        raise ValueError('Year-qualified person keys are not unique')
    return frame, index, {'verified_admission_hashes': sorted(expected), 'rows': len(frame),
                          'households': int(frame.SERIALNO.nunique())}


def lock_verified():
    """Final-partition labels require a committed evaluation lock whose inputs still hash."""
    if not LOCK.exists():
        return False
    lock = read_json(LOCK)
    for rel, digest in lock['files'].items():
        path = ROOT/rel
        if not path.is_file() or sha(path) != digest:
            raise PermissionError(f'Evaluation lock input changed: {rel}')
    return True


def read_labels(raw_rows, *, pool, csv=CSV_2016):
    """Read label columns for the requested raw rows only (strictly increasing)."""
    if pool.startswith('final') and not lock_verified():
        raise PermissionError('Final-partition labels are sealed until EVALUATION_LOCK.json verifies')
    rows = np.asarray(raw_rows, dtype=np.int64)
    if len(rows) > 1 and np.any(np.diff(rows) <= 0):
        raise ValueError('Rows must be strictly increasing')
    wanted = set((rows+1).tolist())
    f = pd.read_csv(csv, usecols=['SERIALNO', 'SPORDER', *LABEL_COLUMNS], dtype={'SERIALNO': str, 'SPORDER': str},
                    skiprows=lambda i: i != 0 and i not in wanted)
    if len(f) != len(rows):
        raise ValueError('Missing requested label rows')
    return f, labels_from_frame(f)


# ----------------------------------------------------------- frozen services
class FrozenServices:
    """Per-anchor frozen preprocessor, PCA, three service heads and the J mapper."""

    def __init__(self, anchor):
        staged = {r['path']: r['sha256'] for r in read_json(INPUTS/'STAGED_INPUTS.json')['files']}
        def checked(rel):
            path = INPUTS/rel
            if sha(path) != staged[rel]:
                raise ValueError(f'Frozen input changed: {rel}')
            return path
        self.anchor = anchor
        meta = json.loads(checked(f'results/redesign_20260907_acs_transfer_v1/seed_{anchor}/preprocessing.json').read_text())
        pre = CovariatePreprocessor()
        pre.numeric = meta['numeric']; pre.categories = meta['categories']; pre.feature_names = meta['feature_names']
        self.pre = pre
        self.maps = joblib.load(checked(f'results/redesign_20260907_acs_transfer_v1/seed_{anchor}/release_maps.joblib'))
        identity = read_json(ROOT/'results/redesign_20260909_acs_fixed_predictions_v1/PREFIT_IDENTITY.json')
        self.heads = {}
        for task in SERVICE_TASKS:
            base = identity[str(anchor)]['anchors'][task]['path']
            for name in ('metadata.json', 'preprocessing.npz', 'model.pt', 'model.joblib'):
                if f'{base}/{name}' in staged:
                    checked(f'{base}/{name}')
            self.heads[task] = load_service_head(INPUTS/base)
        state = torch.load(checked(f'results/redesign_20260909_acs_fixed_predictions_v1/seed_{anchor}/training/J/final.pt'),
                           map_location='cpu', weights_only=False)['model_state']
        self.mean = state['input_mean'].numpy(); self.scale = state['input_scale'].numpy()
        mapper = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 16))
        mapper.load_state_dict({k[len('branch.mapper.'):]: v for k, v in state.items() if k.startswith('branch.mapper.')})
        self.mapper = mapper.eval().requires_grad_(False)

    def transform(self, frame):
        raw = self.pre.transform(frame)
        pca = self.maps['pca'].transform(raw).astype(np.float32)
        probs = {t: self.heads[t].predict_proba(pca) for t in SERVICE_TASKS}
        ha = np.column_stack([probs['income_binary'], probs['civilian_at_work']])
        hb = probs['public_coverage'].copy()
        x = np.asarray((np.asarray(pca, np.float64)-self.mean)/self.scale, np.float32)
        with torch.no_grad():
            j = self.mapper(torch.from_numpy(x)).numpy()
        RuntimeInputs(x, ha)
        return {'pca': pca, 'x': x, 'ha': ha, 'hb': hb, 'J': j.astype(np.float64)}


def read_features(csv, rows=None):
    cols = list(dict.fromkeys(FEATURES))
    f = pd.read_csv(csv, usecols=cols)
    return f if rows is None else f.iloc[np.asarray(rows)].reset_index(drop=True)


def verify_2018_parity(anchor):
    """Recompute 2018 PCA/H/J from raw rows and compare with the historical arrays."""
    svc = FrozenServices(anchor)
    base = FIXED/f'seed_{anchor}'
    raw = read_features(CSV_2018)
    report = {}
    with np.load(base/'split_rows.npz') as rows, np.load(base/'pca.npz') as pca, np.load(base/'anchors.npz') as anc, \
            np.load(base/'training/J/releases.npz') as jrel:
        for pool in rows.files:
            frame = raw.iloc[rows[pool]].reset_index(drop=True)
            out = svc.transform(frame)
            wire = jrel[f'wire/A/{pool}']
            report[pool] = {
                'people': len(frame),
                'pca_bitwise': bool(out['pca'].dtype == pca[pool].dtype and np.array_equal(out['pca'], pca[pool])),
                'H_A_bitwise': bool(out['ha'].dtype == anc[pool+'/A'].dtype and out['ha'].tobytes() == anc[pool+'/A'].tobytes()),
                'H_B_bitwise': bool(out['hb'].dtype == anc[pool+'/B'].dtype and out['hb'].tobytes() == anc[pool+'/B'].tobytes()),
                'J_max_abs_error_vs_stored': float(np.max(np.abs(out['J']-wire[:, 4:]))),
                'J_bitwise_vs_stored': bool(np.array_equal(out['J'], wire[:, 4:])),
                'pca_max_abs_error': float(np.max(np.abs(out['pca'].astype(float)-pca[pool].astype(float)))),
                'H_max_abs_error': float(max(np.max(np.abs(out['ha']-anc[pool+'/A'])), np.max(np.abs(out['hb']-anc[pool+'/B'])))),
            }
    return report
