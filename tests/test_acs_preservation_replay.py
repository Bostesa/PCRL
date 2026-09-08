"""Explicit source-amendment identity checks; no scientific artifacts loaded."""
import json

import pytest

from scripts.verify_acs_preservation import check, effective_execution_hashes


def test_amendment_chain_preserves_original_freeze_and_rejects_bad_identity(tmp_path):
    frozen = {'sha256': {'experiments/extension.py': 'original', 'results/study/config.json': 'fixed-config'}}
    original_path = tmp_path / 'protocol_freeze.json'
    original_path.write_text(json.dumps(frozen))
    original_hash = check.sha(original_path)
    expected, amendments = effective_execution_hashes(tmp_path, frozen)
    assert expected == frozen['sha256'] and amendments == {}
    first = {'original_protocol_freeze_sha256': original_hash,
             'source_changes': {'experiments/extension.py': {'before_sha256': 'original', 'after_sha256': 'repair1'}}}
    second = {'original_protocol_freeze_sha256': original_hash,
              'source_changes': {'experiments/extension.py': {'before_sha256': 'repair1', 'after_sha256': 'repair2'}}}
    first_path, second_path = (tmp_path / f'EXECUTION_AMENDMENT_{n:02d}.json' for n in (1, 2))
    first_path.write_text(json.dumps(first))
    second_path.write_text(json.dumps(second))
    expected, amendments = effective_execution_hashes(tmp_path, frozen)
    assert expected['experiments/extension.py'] == 'repair2'
    assert expected['results/study/config.json'] == 'fixed-config'
    assert frozen['sha256']['experiments/extension.py'] == 'original'
    assert check.sha(original_path) == original_hash
    assert amendments == {p.name: check.sha(p) for p in (first_path, second_path)}
    second['source_changes']['experiments/extension.py']['before_sha256'] = 'original'
    second_path.write_text(json.dumps(second))
    with pytest.raises(AssertionError):
        effective_execution_hashes(tmp_path, frozen)
    second['source_changes'] = {'results/study/config.json': {'before_sha256': 'fixed-config', 'after_sha256': 'changed'}}
    second_path.write_text(json.dumps(second))
    with pytest.raises(AssertionError):
        effective_execution_hashes(tmp_path, frozen)
    second['source_changes'] = {}
    second['original_protocol_freeze_sha256'] = 'different-freeze'
    second_path.write_text(json.dumps(second))
    with pytest.raises(AssertionError):
        effective_execution_hashes(tmp_path, frozen)
