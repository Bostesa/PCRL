"""Raw 2018 ACS loader for a fixed, unprotected task-identity transfer pilot.

This deliberately does not import the historical Folktables dataset loader.
Labels use -1 for invalid/missing; masks never enter feature preprocessing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = ('AGEP', 'WKHP')
CATEGORICAL = ('SCHL', 'MAR', 'RELP', 'CIT', 'DIS', 'DEAR', 'DEYE', 'DREM')
FEATURES = NUMERIC + CATEGORICAL
SOURCE_COLUMNS = ('PINCP', 'ESR', 'PUBCOV')
HELDOUT_COLUMNS = ('MIG', 'JWMNP')
AUDIT_COLUMNS = ('SEX', 'RAC1P')
KEY_COLUMNS = ('SERIALNO', 'SPORDER', 'PWGTP')
SOURCE_KEYS = ('income_binary', 'income_bins', 'esr', 'pubcov', 'joint')
POOLS = ('representation_fit', 'source_validation', 'downstream_fit',
         'downstream_validation', 'attacker_fit', 'attacker_validation', 'test')
FRACTIONS = (.35, .10, .15, .10, .10, .10, .10)
CATEGORY_CODES = {'SCHL': range(1,25), 'MAR': range(1,6), 'COW': range(1,10),
                  'RELP': range(18), 'CIT': range(1,6), 'DIS': (1,2),
                  'DEAR': (1,2), 'DEYE': (1,2), 'DREM': (1,2)}
AUDIT_NAMES = {'SEX': ['Male', 'Female'], 'RAC1P': [
    'White alone', 'Black or African American alone', 'American Indian alone',
    'Alaska Native alone', 'American Indian and Alaska Native tribes specified; or unspecified combination',
    'Asian alone', 'Native Hawaiian and Other Pacific Islander alone',
    'Some other race alone', 'Two or more races']}


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def load_cohort(path, cap=30000, sample_seed=1200000):
    cols = list(dict.fromkeys(FEATURES + SOURCE_COLUMNS + HELDOUT_COLUMNS + AUDIT_COLUMNS + KEY_COLUMNS))
    raw = pd.read_csv(path, usecols=cols, dtype={'SERIALNO': str})
    original_n = len(raw)
    raw['_raw_row'] = np.arange(len(raw), dtype=np.int64)
    eligible = raw.AGEP.between(19,34) & raw.PWGTP.gt(0)
    frame = raw.loc[eligible].copy()
    duplicates = frame.duplicated(['SERIALNO', 'SPORDER'], keep=False)
    # Identical repeats can be deduplicated; conflicting person rows are an error.
    for _, group in frame.loc[duplicates].groupby(['SERIALNO', 'SPORDER']):
        if len(group.drop(columns='_raw_row').drop_duplicates()) != 1:
            raise ValueError('Conflicting duplicate person keys')
    before_dedup = len(frame)
    frame = frame.drop_duplicates(['SERIALNO', 'SPORDER'], keep='first')
    removed = before_dedup - len(frame)
    if frame.SERIALNO.isna().any() or frame.SPORDER.isna().any():
        raise ValueError('Missing grouping/person keys')
    sizes = frame.groupby('SERIALNO', sort=True).size()
    order = np.random.default_rng(sample_seed).permutation(sizes.index.to_numpy())
    # Keep a whole-household prefix, stopping before cap. Never resample on labels.
    cumulative = sizes.loc[order].cumsum().to_numpy()
    chosen = order[cumulative <= cap]
    frame = frame[frame.SERIALNO.isin(chosen)].sort_values('_raw_row').reset_index(drop=True)
    metadata = {'raw_rows': original_n, 'cohort_rows_before_dedup': int(eligible.sum()),
                'duplicate_rows_removed': removed,
                'sample_rows': len(frame), 'sample_households': int(frame.SERIALNO.nunique()),
                'cap': cap, 'sample_seed': sample_seed,
                'raw_row_hash': array_hash(frame._raw_row.to_numpy()),
                'sampling': 'random whole-household prefix, shared cohort across split seeds'}
    return frame, metadata


def split_households(frame, seed):
    groups = np.array(sorted(frame.SERIALNO.unique()))
    groups = np.random.default_rng(1210000 + seed).permutation(groups)
    boundaries = np.rint(np.cumsum((0.,) + FRACTIONS) * len(groups)).astype(int)
    result = {name: np.flatnonzero(frame.SERIALNO.isin(groups[boundaries[i]:boundaries[i+1]]))
              for i, name in enumerate(POOLS)}
    assert len(np.unique(np.concatenate(list(result.values())))) == len(frame)
    return result


def _coded(values, codes):
    values = np.asarray(values, dtype=float)
    valid = np.isin(values, list(codes))
    result = np.full(len(values), -1, dtype=np.int64)
    result[valid] = values[valid].astype(int) - 1
    return result


def source_labels(frame, income_edges):
    income = frame.PINCP.to_numpy(float)
    valid = np.isfinite(income) & (income >= -19998) & (income <= 4209995)
    binary = np.full(len(frame), -1, dtype=np.int64)
    bins = binary.copy()
    binary[valid] = (income[valid] > 50000).astype(int)
    bins[valid] = np.searchsorted(income_edges, income[valid], side='right')
    esr, pubcov = _coded(frame.ESR, range(1,7)), _coded(frame.PUBCOV, (1,2))
    joint = np.full(len(frame), -1, dtype=np.int64)
    complete = valid & (esr >= 0) & (pubcov >= 0)
    joint[complete] = 4*binary[complete] + 2*(esr[complete] == 0) + (pubcov[complete] == 0)
    return {'income_binary': binary, 'income_bins': bins, 'esr': esr, 'pubcov': pubcov, 'joint': joint}


def fit_income_edges(frame):
    income = frame.PINCP.to_numpy(float)
    valid = np.isfinite(income) & (income >= -19998) & (income <= 4209995)
    if not valid.any():
        raise ValueError('No fitting income labels')
    # Repeated quantiles merged, not perturbed to manufacture additional bins.
    return np.unique(np.quantile(income[valid], np.arange(1,8)/8))


def heldout_labels(frame):
    mig = _coded(frame.MIG, (1,2,3))
    residence = np.where(mig < 0, -1, (mig == 0).astype(int))
    commute = frame.JWMNP.to_numpy(float)
    valid = np.isfinite(commute) & (commute >= 1) & (commute <= 200) & (commute == np.floor(commute))
    commute_binary = np.full(len(frame), -1, dtype=np.int64)
    commute_binary[valid] = commute[valid] > 20
    return {'same_residence': residence, 'commute_over20': commute_binary}


def audit_labels(frame):
    return {'SEX': _coded(frame.SEX, (1,2)), 'RAC1P': _coded(frame.RAC1P, range(1,10))}


def support(labels, classes):
    return {name: {'valid': int((y >= 0).sum()), 'missing_or_inapplicable': int((y < 0).sum()),
                   'class_counts': np.bincount(y[y >= 0], minlength=classes[name]).tolist()}
            for name,y in labels.items()}


class CovariatePreprocessor:
    """Train-only numeric statistics and discovered categorical levels.

    Known levels, missing/invalid, and unseen-valid are distinct columns. No
    task, eligibility, identifier, survey weight, or audit label is accessed.
    """
    def fit(self, frame):
        self.numeric = {}
        for name in NUMERIC:
            values = self._numeric_values(frame, name)
            median = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.
            filled = np.nan_to_num(values, nan=median)
            scale = float(filled.std())
            self.numeric[name] = [median, float(filled.mean()), max(scale, 1e-8)]
        self.categories = {name: sorted(int(v) for v in frame[name].dropna().unique()
                                       if v in CATEGORY_CODES[name]) for name in CATEGORICAL}
        self.feature_names = [v for name in NUMERIC for v in (name, name + '_missing')]
        for name in CATEGORICAL:
            self.feature_names += [f'{name}={v}' for v in self.categories[name]] + [name+'=missing', name+'=unseen']
        return self

    @staticmethod
    def _numeric_values(frame, name):
        values = frame[name].to_numpy(float).copy()
        lo, hi = (0,99) if name == 'AGEP' else (1,99)
        values[~np.isfinite(values) | (values < lo) | (values > hi)] = np.nan
        return values

    def transform(self, frame):
        arrays = []
        for name in NUMERIC:
            values = self._numeric_values(frame, name)
            median, mean, scale = self.numeric[name]
            arrays += [((np.nan_to_num(values,nan=median)-mean)/scale)[:,None], np.isnan(values)[:,None]]
        for name in CATEGORICAL:
            values = frame[name].to_numpy(float)
            levels = self.categories[name]
            known = {v: j for j,v in enumerate(levels)}
            positions = np.array([known.get(v, len(levels)+1) if v in CATEGORY_CODES[name] else len(levels) for v in values])
            arrays.append(np.eye(len(levels)+2, dtype=np.float32)[positions])
        return np.concatenate(arrays, axis=1).astype(np.float32)

    def metadata(self):
        return {'numeric': self.numeric, 'categories': self.categories, 'feature_names': self.feature_names,
                'numeric_dtype': 'float32', 'fitted_columns': list(FEATURES)}


def require_test_selection(directory, expected_keys):
    path = Path(directory) / 'selection_before_test.json'
    if not path.exists():
        raise RuntimeError('Final test remains sealed: selection not saved')
    saved = json.loads(path.read_text())
    if sorted(saved['head_selections']) != sorted(expected_keys):
        raise RuntimeError('Incomplete selections: final test remains sealed')
    return sha_file(path)
