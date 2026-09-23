"""The default 2018 working copy must never carry outer labels."""
from pathlib import Path

import joblib
import numpy as np
import pytest

from experiments.pcrl_task_aligned_cuts_v1 import data


def _prepared(anchor: int) -> dict:
    pools = {}
    encoded = {}
    for name in data.POOLS:
        n = 3 if name == 'representation_fit' else 1
        pools[name] = {
            'ha': np.zeros((n, 4), dtype=np.float32),
            'hb': np.ones((n, 2), dtype=np.float32),
            'households': np.asarray([f'{anchor}-{name}-{i}' for i in range(n)]),
            'ids': np.arange(n),
            'weights': np.ones(n),
            'labels': {'same_residence': np.zeros(n, dtype=np.int64)},
        }
        encoded[name] = {'codes': {'T0': np.arange(n, dtype=np.int64)}}
    return {'ctx': {'anchor': anchor, 'pools': pools}, 'encoded': encoded,
            'roles': {'teacher_fit': np.array([0]),
                      'teacher_internal_validation': np.array([1]),
                      'mechanism': np.array([2])},
            'encoder': {'frozen_codebook': np.arange(32, dtype=np.float32)}}


def test_sanitized_copy_keeps_inner_inputs_and_fails_closed_on_corruption(tmp_path, monkeypatch):
    cache = tmp_path / 'sanitized'
    monkeypatch.setattr(data, '_sanitized_root', lambda: cache)
    anchors = {}
    for anchor in (0, 1, 2):
        original = tmp_path / f'original_{anchor}.joblib'
        joblib.dump(_prepared(anchor), original)
        digest = data.sha256_file(original)
        anchors[str(anchor)] = {'prepared': {
            'path': str(original), 'sha256': digest,
            'archive': {'member_path': original.name, 'member_sha256': digest}}}
    value = {'anchors': anchors, 'source_commit': 'synthetic'}
    receipt = data.create_sanitized_working_copies(value)
    assert set(receipt['anchors']) == {'0', '1', '2'}
    for anchor in (0, 1, 2):
        working = data.load_prepared(value, anchor)
        assert 'labels' not in working['ctx']['pools']['attacker_validation']
        assert np.array_equal(working['encoded']['representation_fit']['codes']['T0'],
                              np.arange(3))
        original = joblib.load(anchors[str(anchor)]['prepared']['path'])
        assert 'labels' in original['ctx']['pools']['attacker_validation']
    damaged = cache / receipt['anchors']['0']['sanitized_relative_path']
    with damaged.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='working-copy hash mismatch'):
        data.load_prepared(value, 0)
