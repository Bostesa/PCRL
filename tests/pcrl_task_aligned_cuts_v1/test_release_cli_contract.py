"""Public synthetic wire checks; no ACS inputs or deployment secrets."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from experiments.pcrl_task_aligned_cuts_v1 import release_cli


def _fixture(tmp_path):
    public = tmp_path / 'toy'
    release_cli.synthetic(public)
    private = tmp_path / 'private'
    private.mkdir()
    inputs = private / 'inputs.npz'
    shutil.copyfile(public / 'synthetic_inputs.npz', inputs)
    key = private / 'key.bin'
    release_cli.generate_key(key)
    return public, private, inputs, key


def test_emit_rejects_person_label_fields_before_writing_wire(tmp_path):
    public, private, inputs, key = _fixture(tmp_path)
    with np.load(inputs, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays['Y'] = np.array([0, 1, 1])
    bad = private / 'with_labels.npz'
    np.savez(bad, **arrays)
    output = private / 'wire.npz'
    with pytest.raises(ValueError, match='must contain only'):
        release_cli.emit(public / 'channel', public / 'synthetic_encoder.joblib',
                         bad, key, output)
    assert not output.exists()


def test_emit_rejects_world_readable_key_and_encoder_hash_tamper(tmp_path):
    public, private, inputs, key = _fixture(tmp_path)
    key.chmod(0o644)
    output = private / 'wire.npz'
    with pytest.raises(PermissionError, match='only by its owner'):
        release_cli.emit(public / 'channel', public / 'synthetic_encoder.joblib',
                         inputs, key, output)
    assert not output.exists()
    key.chmod(0o600)
    encoder_path = public / 'synthetic_encoder.joblib'
    encoder_path.write_bytes(encoder_path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='hash|SHA|encoder'):
        release_cli.emit(public / 'channel', encoder_path, inputs, key, output)
    assert not output.exists()
