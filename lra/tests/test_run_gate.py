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
    monkeypatch.setattr(LK, "on_origin", lambda rel: True)


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


def test_unpushed_result_refuses(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, "ENGINEERING_READY")
    monkeypatch.setattr(LK, "on_origin", lambda rel: False)
    ok, why = R.engineering_ready()
    assert not ok and "not on origin" in why


def test_local_only_env_cannot_bypass_stage_push(monkeypatch, tmp_path):
    """Role F R-3: CBP_LOCAL_ONLY never lets a stage run against an unpushed lock."""
    monkeypatch.setenv("CBP_LOCAL_ONLY", "1")
    lat = LK.latest()
    v = LK.verify_lock(LK.PKG / f"{lat['name']}.json", stage="admit")
    assert not v["ok"] and any("CBP_LOCAL_ONLY" in m for m in v["mismatches"])


def test_class_d1_is_a_fixed_map_code():
    """Role F R-1: CLASS|D1 is a registered fixed-map D1 code (27 per seed), audited and in the code bank."""
    assert R.CLASS_D1 in R.d1_fixed_ids() and R.CLASS_D1 in R.code_ids()
    assert len(R.d1_fixed_ids()) == 27 and len(R.code_ids()) == 84 and len(R.d1_jobs()) == 81
    assert R.unit_for(0, R.CLASS_D1) == "dec__s0__U_CLASS_i1o1_D1"
