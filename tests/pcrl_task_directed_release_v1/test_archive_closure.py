"""Synthetic closure of retry/original versions and public split metadata."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from experiments.pcrl_task_directed_release_v1 import archive, parallel_verify


@pytest.mark.parametrize('relative,marker,member', [
    ('numerical_recovery/anchor_0/q', 'COMPLETE.json', 'map/solution.joblib'),
    ('numerical_originals/anchor_0/q/map', 'ACCEPTED.json', 'solution.joblib'),
    ('numerical_originals/anchor_0/q/audit', 'COMPLETE.json', 'unique_attacker/candidate.joblib'),
])
def test_inventory_requires_preserved_and_staged_receipt_members(tmp_path, relative, marker, member):
    base = tmp_path/'results'/archive.STUDY/'private'/relative
    base.mkdir(parents=True)
    (base/marker).write_text(json.dumps({'artifact_hashes': {member: '0'*64}}))
    with pytest.raises((ValueError, FileNotFoundError), match='artifact|missing|No such'):
        archive.inventory(tmp_path)


def test_original_receipt_resolves_its_preserved_copy_not_replacement(tmp_path):
    out = tmp_path/'results'/archive.STUDY
    original = out/'private/numerical_originals/anchor_0/q/audit'
    active = out/'private/run/anchor_0/audits/q'
    for base, content in ((original, b'original unique model'), (active, b'replacement unique model')):
        base.mkdir(parents=True)
        (base/'candidate.joblib').write_bytes(content)
        (base/'COMPLETE.json').write_text(json.dumps({'artifact_hashes': {
            'candidate.joblib': hashlib.sha256(content).hexdigest()}}))
    records = archive.inventory(tmp_path)
    assert len(records) == 4
    assert len({r['path'] for r in records}) == 4
    (original/'candidate.joblib').write_bytes(b'corrupt original')
    with pytest.raises(ValueError, match='artifact|changed'):
        archive.inventory(tmp_path)


def test_installed_retry_requires_its_registry_and_original_version_closure(tmp_path):
    name = 'Ttask_C_0.0005_a17'
    base = tmp_path/'results'/archive.STUDY/'private/run/anchor_0/maps'/name
    base.mkdir(parents=True)
    (base/'solution.joblib').write_bytes(b'synthetic replacement')
    (base/'ACCEPTED.json').write_text(json.dumps({'anchor': 0, 'configuration': name,
        'artifact_hashes': {'solution.joblib': hashlib.sha256(b'synthetic replacement').hexdigest()},
        'numerical_retry': {'registration_sha256': '0'*64}}))
    with pytest.raises((ValueError, FileNotFoundError), match='registry|NUMERICAL|No such'):
        archive.inventory(tmp_path)


def test_parallel_snapshot_includes_numerical_registry_and_dated_split_disclosure(tmp_path):
    source = Path(parallel_verify.__file__).parent
    owned = tmp_path/'experiments'/archive.STUDY;owned.mkdir(parents=True)
    for path in source.glob('*.py'): shutil.copy2(path, owned/path.name)
    out = tmp_path/'results'/archive.STUDY;out.mkdir(parents=True)
    required = ('SELECTION.json', 'CONTRASTS.json', 'VALIDATION_GRID.json', 'RESOURCE_SCHEDULE.json')
    extra = ('NUMERICAL_RECOVERY.json', 'SPLITS.json', 'DATED_SPLIT_CLARIFICATION.md')
    for name in required+extra: (out/name).write_text('{}\n')
    before = parallel_verify.source_snapshot(tmp_path)
    assert set(extra) <= set(before['frozen_documents'])
    (out/'NUMERICAL_RECOVERY.json').write_text('{"changed":true}\n')
    after = parallel_verify.source_snapshot(tmp_path)
    assert before != after
