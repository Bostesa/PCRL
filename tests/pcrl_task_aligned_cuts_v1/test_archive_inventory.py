"""The final private archive includes executable analysis and skips its own stage."""

import stat

from experiments.pcrl_task_aligned_cuts_v1 import archive


def test_inventory_includes_analysis_and_excludes_stage(tmp_path):
    expected = {
        'experiments/pcrl_task_aligned_cuts_v1/fit.py',
        'analysis/pcrl_task_aligned_cuts_v1/decision.py',
        'results/pcrl_task_aligned_cuts_v1/PROTOCOL.md',
        'tests/pcrl_task_aligned_cuts_v1/test_fixture.py',
    }
    for relative in expected:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    stage = tmp_path / 'results/pcrl_task_aligned_cuts_v1/private/archive_stage/part-0001.tar.zst'
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_bytes(b'not a scientific source')
    records = archive.inventory(tmp_path)
    assert {record['path'] for record in records} == expected
    assert all(record['bytes'] > 0 and len(record['sha256']) == 64 for record in records)


def test_stage_creates_owner_only_archive_parts_and_manifest(tmp_path, monkeypatch):
    source = tmp_path / 'analysis/pcrl_task_aligned_cuts_v1/decision.py'
    source.parent.mkdir(parents=True)
    source.write_text('replay fixture\n')
    monkeypatch.setattr(archive.subprocess, 'check_output',
                        lambda *args, **kwargs: 'f' * 40)
    staging = tmp_path / 'results/pcrl_task_aligned_cuts_v1/private/archive_stage'
    result = archive.stage(tmp_path, staging)
    assert result['files'] == 1 and result['parts'] == 1
    assert stat.S_IMODE(staging.stat().st_mode) == 0o700
    assert {stat.S_IMODE(path.stat().st_mode) for path in staging.iterdir()} == {0o600}
