"""Frozen raw-covariate -> standardized PCA32 deployment adapter.

Only ten named raw covariates and the unchanged H_A service are accepted. No
fitting, label/ID discovery, B access, or historical artifact fallback occurs.
Artifacts are hash-checked before deserialization. The original preprocessor
implementation is pinned byte-for-byte; its fitting method is never invoked.

Exported model parameters belong in the caller's private deployment archive.
The optional emission helper returns H_A and one sampled categorical token;
the probability row remains internal. Persistent wire replay is caller-owned.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from experiments import acs_transfer_data as historical
from .data import RuntimeInputs

RAW_FEATURES = ('AGEP', 'WKHP', 'SCHL', 'MAR', 'RELP', 'CIT', 'DIS', 'DEAR', 'DEYE', 'DREM')
HISTORICAL_SOURCE_SHA256 = '1cb0290cabd83e4efba1fd990c31721f76ff9a847669951942b6aef14d6fa0a5'
TRANSFER = 'results/redesign_20260907_acs_transfer_v1'
FIXED = 'results/redesign_20260909_acs_fixed_predictions_v1'
RAW = 'data/folktables/2018/1-Year/psam_p06.csv'
FEATURE_PARITY_ATOL = 1e-5
PINNED_ARTIFACTS = {
    0: ('be8b60bf72d978d82555919d6136f93c5fc6b3b88b2ad17843df20828ac62d9e',
        'e5a3af3d8982c720369f5757e5c13725654eafe9884a88694ea263453ac105f7',
        '8a824ca61d2cb5142c364232fe840699ecfff09d8913964d7525d3e25ccb952d'),
    1: ('14a6882c9d972ab0e368254443bb3fe16833a70a3adefcc387e857ff5b1ccb76',
        'c5a4de5bf27331c8df2e45c316e68470f465e21fbbda3cfc6ea7a5d74aafca6b',
        'b6f7763b71dfd2a33bc40e10d76d0ab8b3952dd1b10cac74739ebb375e20eeac'),
    2: ('d97002e6da3479994880f86be89df8f42ce13704c7df859431623b97c20aeb81',
        'f801abde520ab432aa72ad1033a057798290816a89b47fc8d23d40ecb8e7a380',
        '086a07294620687306737ba34afc20e83087caf3aee648aaae146d0a3a5c3390'),
}


def _sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def _verify_source():
    if _sha(historical.__file__) != HISTORICAL_SOURCE_SHA256 or tuple(historical.FEATURES) != RAW_FEATURES:
        raise ValueError('Historical raw preprocessing source differs from pinned implementation')


def _json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def _h(ha, n):
    if (not isinstance(ha, np.ndarray) or ha.shape != (n, 4)
            or ha.dtype.kind not in 'fiu' or not np.isfinite(ha).all()):
        raise ValueError('H_A must be an aligned finite numeric ndarray of exactly four coordinates')
    return np.array(ha, copy=True, order='K')


def _raw_frame(frame):
    if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
            or len(frame.columns) != len(RAW_FEATURES) or set(frame.columns) != set(RAW_FEATURES)):
        raise ValueError('Raw input requires exactly the ten permitted covariate columns; no labels, IDs, weights or B')
    if not len(frame):
        raise ValueError('Nonempty raw covariate rows required')
    return frame.loc[:, RAW_FEATURES]


@dataclass
class FrozenInputMap:
    preprocessing: dict
    pca: object
    input_mean: np.ndarray
    input_scale: np.ndarray
    provenance: dict

    def __post_init__(self):
        _verify_source()
        meta = self.preprocessing
        if (meta.get('fitted_columns') != list(RAW_FEATURES)
                or set(meta.get('numeric', {})) != set(historical.NUMERIC)
                or set(meta.get('categories', {})) != set(historical.CATEGORICAL)
                or meta.get('numeric_dtype') != 'float32'):
            raise ValueError('Preprocessing schema differs from permitted historical covariates')
        self.input_mean = np.asarray(self.input_mean).copy()
        self.input_scale = np.asarray(self.input_scale).copy()
        if (self.input_mean.shape != (32,) or self.input_scale.shape != (32,)
                or not np.isfinite(self.input_mean).all() or not np.isfinite(self.input_scale).all()
                or np.any(self.input_scale <= 0)):
            raise ValueError('Invalid frozen PCA32 standardizer')
        components = np.asarray(self.pca.components_)
        if (components.shape != (32, len(meta['feature_names']))
                or not np.isfinite(components).all() or not np.isfinite(self.pca.mean_).all()
                or bool(self.pca.whiten)):
            raise ValueError('Expected finite historical unwhitened PCA32 map')

    @classmethod
    def from_owned_inputs(cls, anchor, *, inputs_root):
        """Read only hash-pinned artifacts from an explicit owned root; never fit."""
        if isinstance(anchor, bool) or anchor not in PINNED_ARTIFACTS:
            raise ValueError('Anchor must be 0,1,2')
        _verify_source()
        root = Path(inputs_root).resolve()
        relatives = [f'{TRANSFER}/seed_{anchor}/preprocessing.json',
                     f'{TRANSFER}/seed_{anchor}/release_maps.joblib',
                     f'{FIXED}/seed_{anchor}/training/J/final.pt']
        paths, hashes = [], {}
        for relative, expected in zip(relatives, PINNED_ARTIFACTS[anchor]):
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Frozen artifact escapes explicit owned-input root')
            actual = _sha(path)
            if actual != expected:
                raise ValueError(f'Frozen input artifact hash mismatch: {relative}')
            paths.append(path); hashes[relative] = actual
        meta = json.loads(paths[0].read_text())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pca = joblib.load(paths[1])['pca']
        import torch
        state = torch.load(paths[2], map_location='cpu', weights_only=False)['model_state']
        provenance = {'anchor': int(anchor), 'source_artifact_sha256': hashes,
                      'historical_preprocessor_source_sha256': HISTORICAL_SOURCE_SHA256,
                      'deserialization_warnings': [str(w.message) for w in caught],
                      'historical_fit_scope': 'all original representation_fit features; no refitting',
                      'operation': 'historical named raw preprocessing float32; saved PCA.transform; '
                                   'checkpoint (PCA.astype(float64)-input_mean)/input_scale; cast float32'}
        return cls(meta, pca, state['input_mean'].numpy(), state['input_scale'].numpy(), provenance)

    def pca_features(self, covariates):
        """Apply the exact frozen preprocessing and saved PCA, without H or labels."""
        frame = _raw_frame(covariates)
        pre = historical.CovariatePreprocessor()
        pre.numeric = copy.deepcopy(self.preprocessing['numeric'])
        pre.categories = copy.deepcopy(self.preprocessing['categories'])
        pre.feature_names = list(self.preprocessing['feature_names'])
        # Missing/invalid numeric and categorical handling is the historical code.
        raw_features = pre.transform(frame)
        pca = np.asarray(self.pca.transform(raw_features))
        if pca.shape != (len(frame), 32) or not np.isfinite(pca).all():
            raise FloatingPointError('Frozen preprocessing/PCA produced nonfinite or malformed output')
        return pca

    def transform(self, covariates, h_a):
        """Return only standardized PCA32 plus a byte-identical copy of supplied H_A."""
        pca = self.pca_features(covariates)
        ha = _h(h_a, len(pca))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            x = np.asarray((pca.astype(np.float64) - self.input_mean) / self.input_scale, np.float32)
        if not np.isfinite(x).all():
            raise FloatingPointError('Frozen input standardization produced nonfinite output')
        return RuntimeInputs(x, ha)

    def export(self, directory):
        """Preserve a standalone private deployment map and a public-safe hash manifest."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        _json(directory / 'preprocessing.json', self.preprocessing)
        joblib.dump(self.pca, directory / 'pca.joblib', compress=3)
        with (directory / 'standardizer.npz').open('xb') as handle:
            np.savez_compressed(handle, input_mean=self.input_mean, input_scale=self.input_scale)
        files = {name: _sha(directory / name) for name in ('preprocessing.json', 'pca.joblib', 'standardizer.npz')}
        manifest = {'schema': 1, 'files': files, 'raw_covariate_columns': list(RAW_FEATURES),
                    'H_A_coordinates': 4, 'output_auxiliary_coordinates': 32,
                    'forbidden_runtime_inputs': ['labels', 'protected attributes', 'IDs', 'survey weights', 'H_B'],
                    'historical_preprocessor_source_sha256': HISTORICAL_SOURCE_SHA256,
                    'provenance': self.provenance, 'refit': False}
        _json(directory / 'MANIFEST.json', manifest)
        return manifest

    @classmethod
    def load_export(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / 'MANIFEST.json').read_text())
        expected = {'preprocessing.json', 'pca.joblib', 'standardizer.npz'}
        if (manifest.get('schema') != 1 or set(manifest['files']) != expected
                or manifest.get('historical_preprocessor_source_sha256') != HISTORICAL_SOURCE_SHA256):
            raise ValueError('Invalid frozen deployment manifest')
        for name, checksum in manifest['files'].items():
            if _sha(directory / name) != checksum:
                raise ValueError(f'Deployment artifact hash mismatch: {name}')
        meta = json.loads((directory / 'preprocessing.json').read_text())
        pca = joblib.load(directory / 'pca.joblib')
        with np.load(directory / 'standardizer.npz', allow_pickle=False) as state:
            mean, scale = state['input_mean'], state['input_scale']
        return cls(meta, pca, mean, scale, manifest['provenance'])


def emit_token(h_a, token_probabilities, *, rng):
    """Sample one wire token; return H_A and integer token only, never a Q row.

    The caller supplies the RNG and owns persistent wire reuse. Repeated calls
    consume new draws; this helper does not implement a repeated-query policy.
    """
    p = np.asarray(token_probabilities, dtype=np.float64)
    if (p.ndim != 2 or not p.shape[1] or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(1), 1., atol=1e-8, rtol=0)
            or not isinstance(rng, np.random.Generator)):
        raise ValueError('Finite normalized token laws and an explicit NumPy Generator required')
    ha = _h(h_a, len(p))
    cumulative = np.cumsum(p / p.sum(1, keepdims=True), axis=1)
    cumulative[:, -1] = 1.
    token = np.sum(rng.random(len(p))[:, None] >= cumulative, axis=1).astype(np.int64)
    ha.flags.writeable = False
    token.flags.writeable = False
    return {'h_a': ha, 'token': token}


def verify_feature_replay(*, inputs_root, report_path, export_root, n_rows=128):
    """Feature-only replay on first fixed RF rows, never labels or test members.

    Reads only the ten raw covariates, RF raw-row indices, RF PCA32 and RF H_A.
    The public report stores counts, hashes and max errors; no row IDs/values.
    Deployment artifacts are exported under the caller's private export_root.
    """
    if n_rows not in (32, 128):
        raise ValueError('Registered lightweight feature check uses 32 or 128 fixed RF rows')
    root = Path(inputs_root).resolve()
    report_path, export_root = Path(report_path), Path(export_root)
    if report_path.exists() or export_root.exists():
        raise FileExistsError('Refusing to overwrite feature replay or deployment artifacts')
    report = {'schema': 1, 'feature_only': True, 'raw_columns_read': list(RAW_FEATURES),
              'pool': 'representation_fit', 'selection': f'first {n_rows} stored RF rows',
              'outcomes_read': False, 'test_members_read': False, 'refit': False,
              'absolute_tolerance': FEATURE_PARITY_ATOL, 'anchors': []}
    for anchor in (0, 1, 2):
        mapper = FrozenInputMap.from_owned_inputs(anchor, inputs_root=root)
        base = root / FIXED / f'seed_{anchor}'
        with np.load(base / 'split_rows.npz', allow_pickle=False) as bundle:
            rows = np.asarray(bundle['representation_fit'][:n_rows], dtype=np.int64)
        if len(rows) != n_rows or np.any(rows < 0) or np.any(np.diff(rows) <= 0):
            raise ValueError('Fixed RF row selection must be complete, unique, and sorted')
        selected = set((rows + 1).tolist())
        frame = pd.read_csv(root / RAW, usecols=list(RAW_FEATURES),
                            skiprows=lambda i: i != 0 and i not in selected)
        if len(frame) != n_rows:
            raise ValueError('Feature-only raw row alignment failed')
        with np.load(base / 'pca.npz', allow_pickle=False) as bundle:
            saved_pca = bundle['representation_fit'][:n_rows]
        with np.load(base / 'anchors.npz', allow_pickle=False) as bundle:
            ha = bundle['representation_fit/A'][:n_rows]
        actual_pca = mapper.pca_features(frame)
        actual = mapper.transform(frame, ha)
        expected_x = np.asarray((saved_pca.astype(np.float64) - mapper.input_mean) / mapper.input_scale, np.float32)
        pca_error = float(np.max(np.abs(actual_pca - saved_pca)))
        x_error = float(np.max(np.abs(actual.x_a - expected_x)))
        h_exact = actual.h_a.dtype == ha.dtype and actual.h_a.tobytes() == ha.tobytes()
        directory = export_root / f'anchor_{anchor}'
        manifest = mapper.export(directory)
        restored = FrozenInputMap.load_export(directory).transform(frame, ha)
        export_exact = np.array_equal(restored.x_a, actual.x_a) and restored.h_a.tobytes() == ha.tobytes()
        accepted = bool(np.isfinite(pca_error) and np.isfinite(x_error)
                        and max(pca_error, x_error) <= FEATURE_PARITY_ATOL and h_exact and export_exact)
        report['anchors'].append({'anchor': anchor, 'rows': n_rows, 'raw_rows_sha256': _array_sha(rows),
            'saved_pca_sha256': _array_sha(saved_pca), 'reconstructed_pca_sha256': _array_sha(actual_pca),
            'saved_standardized_x_sha256': _array_sha(expected_x), 'reconstructed_x_sha256': _array_sha(actual.x_a),
            'pca_max_abs_error': pca_error, 'standardized_x_max_abs_error': x_error,
            'H_A_byte_identical': h_exact, 'export_reload_byte_identical': export_exact,
            'accepted': accepted, 'deployment_manifest': manifest,
            'deployment_manifest_sha256': _sha(directory / 'MANIFEST.json')})
    report['accepted'] = all(row['accepted'] for row in report['anchors'])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _json(report_path, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature-only frozen raw-to-PCA32 integrity replay')
    parser.add_argument('--inputs-root', type=Path, required=True)
    parser.add_argument('--report-path', type=Path, required=True)
    parser.add_argument('--export-root', type=Path, required=True)
    parser.add_argument('--rows', type=int, choices=(32, 128), default=128)
    args = parser.parse_args()
    result = verify_feature_replay(inputs_root=args.inputs_root, report_path=args.report_path,
                                    export_root=args.export_root, n_rows=args.rows)
    print(json.dumps({'accepted': result['accepted'], 'anchors': [
        {k: row[k] for k in ('anchor', 'rows', 'pca_max_abs_error', 'standardized_x_max_abs_error',
                            'H_A_byte_identical', 'export_reload_byte_identical')}
        for row in result['anchors']]}, indent=2))
    if not result['accepted']:
        raise SystemExit(1)
