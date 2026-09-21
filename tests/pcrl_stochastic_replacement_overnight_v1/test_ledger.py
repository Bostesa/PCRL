"""Ledger, lock and handoff fixtures. These target reconnect-safety, not implementation mirroring."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from experiments.pcrl_stochastic_replacement_overnight_v1 import ledger as L


def test_config_id_is_content_addressed_and_order_insensitive():
    a = L.config_id({'family': 'pca32', 'k': 9, 'delta': 0.002})
    b = L.config_id({'delta': 0.002, 'k': 9, 'family': 'pca32'})
    c = L.config_id({'family': 'pca32', 'k': 17, 'delta': 0.002})
    assert a == b and a != c


def test_editing_a_config_changes_the_id_so_stale_results_cannot_be_reused():
    base = {'family': 'zj', 'k': 9, 'objective': 'SUP', 'delta': 0.0}
    assert L.config_id(base) != L.config_id(dict(base, delta=0.002))
    assert L.config_id(base) != L.config_id(dict(base, objective='LF'))


def test_atomic_write_never_leaves_a_partial_file(tmp_path):
    p = tmp_path / 'deep' / 'x.json'
    L.write_json_atomic(p, {'a': 1})
    assert json.loads(p.read_text()) == {'a': 1}
    assert not list(tmp_path.glob('**/*.tmp*'))


def test_lock_refuses_a_second_live_worker(tmp_path):
    lk = tmp_path / 'run.lock'
    first = L.RunLock(lk, 'fit')
    first.acquire()
    with pytest.raises(RuntimeError, match='already owned'):
        L.RunLock(lk, 'fit').acquire()
    first.release()
    assert not lk.exists()
    L.RunLock(lk, 'fit').acquire()          # free again


def test_lock_breaks_only_a_genuinely_dead_holder(tmp_path):
    lk = tmp_path / 'run.lock'
    dead = subprocess.run([sys.executable, '-c', 'pass'])
    L.write_json_atomic(lk, {'job': 'fit', 'pid': 999999999, 'host': os.uname().nodename,
                             'started_utc': '2026-01-01T00:00:00Z'})
    got = L.RunLock(lk, 'fit').acquire()
    assert got['pid'] == os.getpid()
    assert (tmp_path / 'run.lock.stale').exists()   # the broken record is preserved, not discarded


def test_lock_does_not_break_a_holder_on_another_host(tmp_path):
    lk = tmp_path / 'run.lock'
    L.write_json_atomic(lk, {'job': 'fit', 'pid': 1, 'host': 'some-other-box',
                             'started_utc': '2026-01-01T00:00:00Z'})
    with pytest.raises(RuntimeError, match='already owned'):
        L.RunLock(lk, 'fit').acquire()


def test_status_file_is_not_evidence_of_liveness(tmp_path):
    """The explicit guard against reading an old status report as live process state."""
    lk = tmp_path / 'run.lock'
    L.write_json_atomic(lk, {'job': 'fit', 'pid': 999999999, 'host': os.uname().nodename,
                             'started_utc': '2026-01-01T00:00:00Z'})
    assert L.lock_holder_alive(lk) is False
    live = L.RunLock(tmp_path / 'live.lock', 'fit')
    live.acquire()
    assert L.lock_holder_alive(tmp_path / 'live.lock') is True


def test_ledger_claim_is_idempotent_for_finished_units(tmp_path):
    led = L.Ledger(tmp_path / 'units')
    cfg = {'family': 'pca32', 'k': 9}
    cid = L.config_id(cfg)
    assert led.claim(cid, cfg) is True
    led.put(cid, cfg, 'fitted', objective_value=1.25)
    assert led.claim(cid, cfg) is False          # never recomputed
    assert led.get(cid)['objective_value'] == 1.25


def test_ledger_rejects_an_unknown_state(tmp_path):
    led = L.Ledger(tmp_path / 'units')
    with pytest.raises(ValueError, match='unknown state'):
        led.put('abc', {}, 'finished-ish')


def test_ledger_keeps_state_history(tmp_path):
    led = L.Ledger(tmp_path / 'units')
    cid = 'deadbeef'
    led.put(cid, {}, 'planned')
    led.put(cid, {}, 'fitting')
    led.put(cid, {}, 'fitted')
    rec = led.get(cid)
    assert [h['state'] for h in rec['history']] == ['planned', 'fitting', 'fitted']
    assert rec['state'] == 'fitted'


def test_ledger_counts_every_declared_state(tmp_path):
    led = L.Ledger(tmp_path / 'units')
    for i, s in enumerate(L.STATES):
        led.put(f'{i:016x}', {'i': i}, s)
    counts = led.counts()
    assert set(counts) == set(L.STATES) and all(v == 1 for v in counts.values())


def test_handoff_dir_normalises_to_absolute(tmp_path):
    d = L.handoff_dir(tmp_path)
    assert d.is_absolute() and d.name == 'pcrl_overnight_replacement_v1'


def test_write_status_never_touches_terminal_2(tmp_path):
    t2 = L.handoff_dir(tmp_path) / 'terminal_2'
    t2.mkdir(parents=True)
    theirs = t2 / 'REVIEW.json'
    L.write_json_atomic(theirs, {'finding': 'mine, do not clobber'})
    before = theirs.read_text()
    L.write_status(tmp_path, {'stage': 'fitting', 'commit': 'abc123'})
    assert theirs.read_text() == before
    ours = L.handoff_dir(tmp_path) / 'terminal_1' / 'STATUS.json'
    rec = json.loads(ours.read_text())
    assert rec['stage'] == 'fitting' and rec['study'] == L.STUDY
    assert 'NOT proof' in rec['liveness_note']


def test_read_terminal_2_is_tolerant_of_absence(tmp_path):
    assert L.read_terminal_2(tmp_path) == {'present': False, 'files': []}
