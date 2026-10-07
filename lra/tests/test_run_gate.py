"""Science stages refuse without the pushed, lock-bound ENGINEERING_READY result (prompt section 9; finding 10)."""
import json

import pytest

from lra import lock as LK
from lra import run as R


def _setup(monkeypatch, tmp_path, verdict, bound=True):
    monkeypatch.setattr(R, "PKG", tmp_path)
    if verdict is not None:
        (tmp_path / R.GATE_RESULT).write_text(json.dumps({"verdict": verdict}))
    sha = LK.sha_file(tmp_path / R.GATE_RESULT) if (bound and verdict is not None) else "0" * 64
    monkeypatch.setattr(LK, "latest", lambda: {"name": "SCIENCE_LOCK", "documents_sha256": {R.GATE_RESULT: sha}})
    monkeypatch.setenv("CBP_LOCAL_ONLY", "1")


def test_missing_result_refuses(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, None)
    ok, why = R.engineering_ready()
    assert not ok and "missing" in why


def test_blocked_refuses(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, "ENGINEERING_BLOCKED")
    ok, why = R.engineering_ready()
    assert not ok and "ENGINEERING_BLOCKED" in why


def test_old_gate_met_string_is_not_ready(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, "GATE_MET")
    assert not R.engineering_ready()[0]


def test_unbound_result_refuses(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, "ENGINEERING_READY", bound=False)
    ok, why = R.engineering_ready()
    assert not ok and "not the version bound" in why


def test_bound_ready_passes(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, "ENGINEERING_READY")
    assert R.engineering_ready() == (True, "ENGINEERING_READY")


@pytest.mark.parametrize("stage", list(R.SCIENCE_STAGES))
def test_main_refuses_every_science_stage_when_blocked(monkeypatch, tmp_path, stage):
    _setup(monkeypatch, tmp_path, "ENGINEERING_BLOCKED")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(LK, "verify_lock", lambda *a, **k: {"ok": True, "mismatches": [], "locked_files": {}})
    with pytest.raises(SystemExit, match="require ENGINEERING_READY"):
        R.main(["--lock", str(tmp_path / "SCIENCE_LOCK.json"), "--stage", stage])


def test_correctness_and_admit_are_not_gated():
    assert "correctness" not in R.SCIENCE_STAGES and "admit" not in R.SCIENCE_STAGES
