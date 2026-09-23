"""The final private archive includes executable analysis and skips its own stage."""

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
