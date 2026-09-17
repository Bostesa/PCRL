"""Tests for the sealed-year loader, on synthetic fixtures only.

None of these tests touches the 2016 file, and none of them scores anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.pcrl_evidence_review_v1 import sealed_year_loader as L


@pytest.fixture()
def world(tmp_path):
    root = tmp_path / 'repo'
    (root / 'experiments').mkdir(parents=True)
    (root / 'data').mkdir()
    (root / 'results').mkdir()
    (root / 'experiments' / 'evaluator.py').write_text('# code\n')
    (root / 'data' / 'final.npz').write_bytes(b'\x00' * 64)
    manifest = {'partition_counts': {'final_evaluation': {'rows': 7, 'households': 4},
                                     'fitting': {'rows': 11, 'households': 6}},
                'array_hashes': {'partition/final_evaluation': 'deadbeef',
                                 'partition/fitting': 'feedface'}}
    (root / 'results' / 'manifest.json').write_text(json.dumps(manifest))
    out = tmp_path / 'out'
    out.mkdir()
    inputs = ['experiments/evaluator.py', 'data/final.npz']
    return root, out, inputs, 'results/manifest.json'


def test_open_without_a_lock_is_refused(world):
    root, out, _, _ = world
    with pytest.raises(L.SealError, match='sealed'):
        L.open_final(out, root)


def test_seal_then_open_records_the_access(world):
    root, out, inputs, manifest = world
    lock = L.seal(out, root, inputs, manifest, note='fixture')
    result = L.open_final(out, root, reason='first read')
    assert result['counts'] == {'rows': 7, 'households': 4}
    assert result['lock_digest'] == lock['lock_digest']
    log = json.loads((out / 'FINAL_ACCESS_LOG.json').read_text())
    assert len(log['accesses']) == 1
    assert log['accesses'][0]['time_utc'] > lock['created_utc']


def test_tampering_with_a_locked_data_input_aborts(world):
    root, out, inputs, manifest = world
    L.seal(out, root, inputs, manifest)
    (root / 'data' / 'final.npz').write_bytes(b'\x01' * 64)
    with pytest.raises(L.SealError, match='locked inputs changed'):
        L.open_final(out, root)


def test_a_missing_locked_input_aborts(world):
    root, out, inputs, manifest = world
    L.seal(out, root, inputs, manifest)
    (root / 'data' / 'final.npz').unlink()
    with pytest.raises(L.SealError, match='missing'):
        L.open_final(out, root)


def test_manifest_edits_are_detected(world):
    root, out, inputs, manifest = world
    L.seal(out, root, inputs, manifest)
    (root / 'results' / 'manifest.json').write_text(json.dumps({'partition_counts': {}, 'array_hashes': {}}))
    with pytest.raises(L.SealError, match='manifest changed'):
        L.open_final(out, root)


def test_a_code_only_amendment_is_accepted(world):
    root, out, inputs, manifest = world
    L.seal(out, root, inputs, manifest)
    (root / 'experiments' / 'evaluator.py').write_text('# fixed code\n')
    with pytest.raises(L.SealError):
        L.verify(out, root)
    L.amend(out, 1, {'experiments/evaluator.py': L.sha256_file(root / 'experiments' / 'evaluator.py')},
            reason='implementation defect')
    lock, amendments = L.verify(out, root)
    assert len(amendments) == 1
    assert L.open_final(out, root)['counts']['rows'] == 7


def test_an_amendment_naming_a_data_input_is_rejected(world):
    root, out, inputs, manifest = world
    L.seal(out, root, inputs, manifest)
    with pytest.raises(L.SealError, match='non-code input'):
        L.amend(out, 1, {'data/final.npz': 'x' * 64}, reason='not allowed')


def test_an_amendment_written_out_of_band_is_rejected_at_verify(world):
    """Even if an amendment file is written by hand, verify() must reject it."""
    root, out, inputs, manifest = world
    L.seal(out, root, inputs, manifest)
    (out / 'YEAR_LOCK_AMENDMENT_9.json').write_text(json.dumps(
        {'created_utc': 'x', 'reason': 'hand-written', 'code_hashes': {'data/final.npz': 'y' * 64}}))
    with pytest.raises(L.SealError, match='non-code input'):
        L.verify(out, root)


def test_amendment_for_an_unlocked_path_is_rejected(world):
    root, out, inputs, manifest = world
    L.seal(out, root, inputs, manifest)
    with pytest.raises(L.SealError, match='not in the lock'):
        L.amend(out, 1, {'experiments/other.py': 'z' * 64}, reason='unknown path')


def test_sealing_twice_is_refused(world):
    root, out, inputs, manifest = world
    L.seal(out, root, inputs, manifest)
    with pytest.raises(L.SealError, match='already exists'):
        L.seal(out, root, inputs, manifest)


def test_a_declared_input_that_does_not_exist_cannot_be_sealed(world):
    root, out, inputs, manifest = world
    with pytest.raises(L.SealError, match='missing'):
        L.seal(out, root, inputs + ['data/absent.npz'], manifest)


def test_loader_returns_no_labels(world):
    """The interface hands back counts and hashes, never label arrays."""
    root, out, inputs, manifest = world
    L.seal(out, root, inputs, manifest)
    result = L.open_final(out, root)
    assert set(result) == {'partition', 'counts', 'array_hash', 'lock_digest', 'access_index'}


# --------------------------------------------------------------- spent-year check
STUDY = Path(__file__).resolve().parents[2] / 'results' / 'redesign_20260917_acs_spectral_transport_v1'


@pytest.mark.skipif(not STUDY.is_dir(), reason='published 2017 study directory not present')
def test_loader_round_trips_on_the_already_spent_2017_study(tmp_path):
    """Exercise seal/verify against real published files from the *spent* 2017 study.

    2017's final partition is already used, so nothing is put at risk here. The
    2016 file is not touched, and `open_final` is not called on 2016 anywhere.
    """
    root = Path(__file__).resolve().parents[2]
    rel = 'results/redesign_20260917_acs_spectral_transport_v1'
    inputs = [f'{rel}/PROTOCOL.md', f'{rel}/COMPARISONS.json', f'{rel}/FAMILIES.csv']
    manifest_rel = f'{rel}/INDEPENDENT_VERIFICATION.json'
    out = tmp_path / 'out'
    out.mkdir()

    # the loader needs partition_counts/array_hashes; build a small stand-in manifest
    stand_in = tmp_path / 'manifest.json'
    stand_in.write_text(json.dumps({'partition_counts': {'final_evaluation': {'rows': 15924, 'households': 10701}},
                                    'array_hashes': {'partition/final_evaluation': 'n/a'}}))
    copied = root / 'results' / 'pcrl_evidence_review_v1'
    assert copied.is_dir()

    lock = L.seal(out, root, inputs, manifest_rel)
    assert len(lock['files']) == 3
    L.verify(out, root)                       # unchanged published inputs verify

    # a lock over a file that then changes must abort
    scratch_root = tmp_path / 'scratch'
    (scratch_root / 'experiments').mkdir(parents=True)
    (scratch_root / 'results').mkdir()
    (scratch_root / 'experiments' / 'x.py').write_text('a\n')
    (scratch_root / 'results' / 'm.json').write_text(json.dumps(
        {'partition_counts': {'final_evaluation': {'rows': 1, 'households': 1}},
         'array_hashes': {'partition/final_evaluation': 'h'}}))
    out2 = tmp_path / 'out2'
    out2.mkdir()
    L.seal(out2, scratch_root, ['experiments/x.py'], 'results/m.json')
    (scratch_root / 'experiments' / 'x.py').write_text('b\n')
    with pytest.raises(L.SealError):
        L.open_final(out2, scratch_root)
