"""Runner receipts, resume verification and bounded retry on tiny subprocess units."""
from __future__ import annotations

import json

import pytest

from experiments.pcrl_shared_context_release_v1 import runner

WRITE = ("import os,sys,pathlib;o=pathlib.Path(sys.argv[1]);o.mkdir(parents=True);"
         "(o/'out.txt').write_text(os.environ['OMP_NUM_THREADS']+os.environ['OPENBLAS_NUM_THREADS']"
         "+os.environ['MKL_NUM_THREADS']+'|'+(sys.argv[2] if len(sys.argv)>2 else ''))")
FLAKY = ("import os,sys,pathlib;o=pathlib.Path(sys.argv[1]);o.mkdir(parents=True);"
         "(o/'partial.txt').write_text('x');print('attempt',os.environ['PCRL_UNIT_ATTEMPT']);"
         "sys.exit(1 if os.environ['PCRL_UNIT_ATTEMPT']=='1' else 0)")
ALWAYS_FAIL = "import sys;print('boom',file=sys.stderr);sys.exit(7)"
REFUSE = "import sys;sys.exit(3)"


def queue(*units):
    return {"schema": runner.QUEUE_SCHEMA, "units": list(units)}


def u(uid, code, *extra, deps=(), inputs=(), **more):
    return {"id": uid, "argv": ["-c", code, "{out}", *extra], "depends_on": list(deps),
            "inputs": list(inputs), "data_roles": ["audit_fit"], "timeout_seconds": 60, **more}


@pytest.fixture
def units_root(tmp_path):
    return tmp_path / "private" / "units"


@pytest.fixture(autouse=True)
def frozen_tree(monkeypatch):
    """Other agents edit the package concurrently; pin the tree hash per test."""
    value = runner.code_tree_sha256()
    monkeypatch.setattr(runner, "code_tree_sha256", lambda package_dir=None: value)


def test_complete_receipt_threads_and_dependency_placeholder(units_root, tmp_path):
    source = tmp_path / "input.json"
    source.write_text("{}")
    q = queue(u("first", WRITE, inputs=[str(source)]),
              u("second", WRITE, "{unit:first}", deps=["first"]))
    final = runner.Runner(q, units_root, workers=2).run()
    assert final == {"first": "COMPLETE", "second": "COMPLETE"}
    receipt = json.loads((units_root / "_receipts" / "first.json").read_text())
    for key in ("inputs_sha256", "code_commit", "code_tree_sha256", "data_role_hash",
                "outputs_sha256", "input_files_sha256"):
        assert receipt[key]
    assert receipt["outputs_sha256"]["out.txt"] == runner.laws.sha256_file(units_root / "first" / "out.txt")
    assert set(receipt["outputs_sha256"]) == {"out.txt"}  # runner adds nothing to unit dirs
    assert receipt["attempt_logs"] == ".attempts/first/attempt-1"
    assert (units_root / "first" / "out.txt").read_text() == "111|"
    assert (units_root / "second" / "out.txt").read_text().endswith(str(units_root / "first"))
    second = json.loads((units_root / "_receipts" / "second.json").read_text())
    assert second["dependency_receipts_sha256"]["first"] == runner.laws.sha256_file(
        units_root / "_receipts" / "first.json")
    status = json.loads((units_root / "STATUS.json").read_text())
    assert status["counts"] == {"COMPLETE": 2}


def test_resume_reuses_only_after_full_verification(units_root, tmp_path):
    source = tmp_path / "input.json"
    source.write_text("{}")
    q = queue(u("first", WRITE, inputs=[str(source)]), u("second", WRITE, deps=["first"]))
    runner.Runner(q, units_root).run()
    assert runner.Runner(q, units_root).run() == {"first": "REUSED", "second": "REUSED"}
    (units_root / "first" / "out.txt").write_text("tampered")
    final = runner.Runner(q, units_root).run()
    assert final == {"first": "BLOCKED", "second": "BLOCKED"}
    (units_root / "first" / "out.txt").write_text("111|")
    assert runner.Runner(q, units_root).run()["first"] == "REUSED"
    source.write_text('{"changed": true}')
    status_path = units_root / "status2.json"
    final = runner.Runner(q, units_root, status_path=status_path).run()
    assert final["first"] == "BLOCKED"
    assert "inputs_sha256" in json.loads(status_path.read_text())["units"]["first"]["reason"]
    assert (units_root / "_receipts" / "first.json").exists()  # never overwritten


def test_code_or_data_role_change_blocks_reuse(units_root, monkeypatch):
    q = queue(u("first", WRITE))
    runner.Runner(q, units_root).run()
    other = runner.Runner(q, units_root, index_sha256="f" * 64)
    assert other.run() == {"first": "BLOCKED"}
    monkeypatch.setattr(runner, "code_commit", lambda root=None: "0" * 40)
    assert runner.Runner(q, units_root).run() == {"first": "BLOCKED"}


def test_one_identical_retry_preserves_failed_attempt(units_root):
    final = runner.Runner(queue(u("flaky", FLAKY)), units_root).run()
    assert final == {"flaky": "COMPLETE"}
    first = units_root / ".attempts" / "flaky" / "attempt-1"
    assert json.loads((first / "FAILED.json").read_text())["reason"] == "technical: exit 1"
    assert "attempt 1" in (first / "stdout.log").read_text()
    assert (first / "unit_output" / "partial.txt").exists()
    one = json.loads((first / "ATTEMPT.json").read_text())
    two = json.loads((units_root / ".attempts" / "flaky" / "attempt-2" / "ATTEMPT.json").read_text())
    assert one["argv"] == two["argv"] and one["inputs_sha256"] == two["inputs_sha256"]
    assert json.loads((units_root / "_receipts" / "flaky.json").read_text())["attempt"] == 2


def test_retry_is_bounded_and_dependents_block(units_root):
    q = queue(u("bad", ALWAYS_FAIL), u("after", WRITE, deps=["bad"]))
    assert runner.Runner(q, units_root).run() == {"bad": "FAILED", "after": "BLOCKED"}
    attempts = sorted(p.name for p in (units_root / ".attempts" / "bad").iterdir())
    assert attempts == ["attempt-1", "attempt-2"]
    assert "boom" in (units_root / ".attempts" / "bad" / "attempt-2" / "stderr.log").read_text()
    # A later invocation does not grant a third attempt.
    assert runner.Runner(q, units_root).run()["bad"] == "FAILED"
    assert len(list((units_root / ".attempts" / "bad").iterdir())) == 2


def test_deliberate_refusal_is_not_retried(units_root):
    assert runner.Runner(queue(u("no", REFUSE)), units_root).run() == {"no": "REFUSED"}
    assert [p.name for p in (units_root / ".attempts" / "no").iterdir()] == ["attempt-1"]


def test_interrupted_unit_directory_is_preserved_as_attempt(units_root):
    (units_root / "first").mkdir(parents=True)
    (units_root / "first" / "half.bin").write_text("x")
    assert runner.Runner(queue(u("first", WRITE)), units_root).run() == {"first": "COMPLETE"}
    retired = units_root / ".attempts" / "first" / "attempt-1"
    assert "interrupted" in json.loads((retired / "FAILED.json").read_text())["reason"]
    assert (retired / "unit_output" / "half.bin").exists()


def test_release_descriptor_written_before_receipt(units_root):
    unit = u("rel", WRITE, release={"kind": "nested", "release_id": "NM1_U", "anchor": 0})
    runner.Runner(queue(unit), units_root).run()
    descriptor = json.loads((units_root / "rel" / "release.json").read_text())
    assert descriptor["kind"] == "nested" and "out.txt" in descriptor["pins"]
    assert "release.json" in json.loads((units_root / "_receipts" / "rel.json").read_text())["outputs_sha256"]


def test_only_runs_selection_and_verifies_other_receipts(units_root):
    q = queue(u("first", WRITE), u("second", WRITE, deps=["first"]))
    assert runner.Runner(q, units_root, only=["second"]).run() == {"first": "SKIPPED",
                                                                    "second": "BLOCKED"}
    runner.Runner(q, units_root, only=["first"]).run()
    assert runner.Runner(q, units_root, only=["second"]).run() == {"first": "REUSED",
                                                                    "second": "COMPLETE"}
    assert runner.Runner(q, units_root).run() == {"first": "REUSED", "second": "REUSED"}


def test_queue_validation_refuses_outer_roles_and_bad_graphs(units_root):
    with pytest.raises(PermissionError):
        runner.validate_queue(queue({**u("x", WRITE), "data_roles": ["outer_assessment"]}))
    with pytest.raises(ValueError, match="cycle"):
        runner.validate_queue(queue(u("a", WRITE, deps=["b"]), u("b", WRITE, deps=["a"])))
    with pytest.raises(ValueError, match="depends_on"):
        runner.validate_queue(queue(u("a", WRITE, "{unit:b}"), u("b", WRITE)))
    with pytest.raises(ValueError, match="private"):
        runner.Runner(queue(u("a", WRITE)), "/nonexistent/public/units")


def test_plan_panel_dependency_order(tmp_path):
    index = tmp_path / "index.json"
    index.write_text("{}")
    q = runner.plan_panel([0, 1, 2], str(index), tmp_path / "private" / "units",
                          tmp_path / "private" / "plan")
    by = {unit["id"]: unit for unit in q["units"]}
    assert by["decide_k"]["depends_on"] == ["a0_bank", "a1_bank", "a2_bank"]
    for a in "012":
        for v in runner.NM_VARIANTS:
            assert set(by[f"a{a}_{v}"]["depends_on"]) == {f"a{a}_bank", "decide_k"}
        assert "{json:decide_k/DECISION.json:nm4_K}" in by[f"a{a}_NM4_U"]["argv"]
        assert "--nm4-k" not in by[f"a{a}_NM1_U"]["argv"]
        assert by[f"a{a}_DET_SEL1"]["depends_on"] == [f"a{a}_NM1_U"]
        assert by[f"a{a}_DET_SEL4"]["depends_on"] == [f"a{a}_NM4_U"]
        assert "{unit:a%s_NM4_U}" % a in by[f"a{a}_DET_SEL4"]["argv"]
        for v in ("RD_TASK", "RD_PRIV", "ADV_B1", "ADV_B2", "ADV_B1_P", "ADV_B2_P"):
            assert by[f"a{a}_{v}"]["depends_on"] == [f"a{a}_bank"]
        assert by[f"a{a}_ADV_B2_P"]["argv"][-2:] == ["--select", "privacy"]
        assert by[f"a{a}_ADV_B1"]["argv"][-2:] == ["--select", "task"]
        assert "attack:AB/SEX" in by[f"a{a}_POS_AB_SEX"]["argv"]
        assert by[f"a{a}_J_inner"]["depends_on"] == []
        assert "--methods" in by[f"a{a}_J_inner"]["argv"]
        audit = set(by[f"a{a}_inner_audit"]["depends_on"])
        assert {f"a{a}_DET_SEL4", f"a{a}_T32_P", f"a{a}_ADV_B2", f"a{a}_ADV_B1_P"} <= audit
    sources = json.loads((tmp_path / "private" / "plan" / "a0_audit_sources.json").read_text())
    assert {"D17", "Q_HIST", "NM4_P", "DET_SEL1", "RD_PRIV", "ADV_B2", "ADV_B2_P"} <= set(sources)
    with pytest.raises(ValueError, match="barrier"):
        runner.plan_panel([0], str(index), tmp_path / "private" / "u2", tmp_path / "private" / "p2")
    single = runner.plan_panel([0], str(index), tmp_path / "private" / "u3",
                               tmp_path / "private" / "p3", nm4_k=2)
    nm4 = next(u for u in single["units"] if u["id"] == "a0_NM4_P")
    assert nm4["argv"][-2:] == ["--nm4-k", "2"] and "decide_k" not in nm4["depends_on"]


READ_K = ("import sys,pathlib;o=pathlib.Path(sys.argv[1]);o.mkdir();"
          "(o/'COMPLETE.json').write_text('module-owned');(o/'k.txt').write_text(sys.argv[2])")


def test_capture_and_json_placeholder_feed_the_k_decision(units_root, tmp_path):
    decision = tmp_path / "decision_input.json"
    decision.write_text(json.dumps({"nm4_K": 2, "failing_anchors": ["1"]}))
    q = queue({"id": "decide_k", "argv": ["-m", "experiments.pcrl_shared_context_release_v1.runner",
                                          "capture", "--out", "{out}", "--file", "DECISION.json",
                                          "--", "-m", "json.tool", str(decision)],
               "data_roles": []},
              u("nm4", READ_K, "{json:decide_k/DECISION.json:nm4_K}", deps=["decide_k"]))
    with pytest.raises(ValueError):
        runner.capture(units_root / "x", "DECISION.json", ["-c", "print(1)"])  # only `-m module`
    final = runner.Runner(q, units_root).run()
    assert final == {"decide_k": "COMPLETE", "nm4": "COMPLETE"}
    assert json.loads((units_root / "decide_k" / "DECISION.json").read_text())["nm4_K"] == 2
    assert (units_root / "nm4" / "k.txt").read_text() == "2"
    # The module's own COMPLETE.json is untouched and covered by the runner receipt.
    assert (units_root / "nm4" / "COMPLETE.json").read_text() == "module-owned"
    receipt = json.loads((units_root / "_receipts" / "nm4.json").read_text())
    assert set(receipt["outputs_sha256"]) == {"COMPLETE.json", "k.txt"}


def test_tampered_json_source_blocks_dependent(units_root):
    q = queue(u("dec", "import sys,pathlib,json;o=pathlib.Path(sys.argv[1]);o.mkdir();"
                       "(o/'D.json').write_text(json.dumps({'k': 4}))"),
              u("use", READ_K, "{json:dec/D.json:k}", deps=["dec"]))
    runner.Runner(queue(q["units"][0]), units_root).run()
    (units_root / "dec" / "D.json").write_text('{"k": 2}')
    final = runner.Runner(q, units_root).run()
    assert final["dec"] == "BLOCKED" and final["use"] == "BLOCKED"


def test_cross_commit_reuse_requires_explicit_commit_and_same_tree(units_root, monkeypatch):
    q = queue(u("first", WRITE))
    runner.Runner(q, units_root).run()
    old = json.loads((units_root / "_receipts" / "first.json").read_text())["code_commit"]
    monkeypatch.setattr(runner, "code_commit", lambda root=None: "1" * 40)
    assert runner.Runner(q, units_root).run() == {"first": "BLOCKED"}
    assert runner.Runner(q, units_root, reuse_commits=[old]).run() == {"first": "REUSED"}
    monkeypatch.setattr(runner, "code_tree_sha256", lambda package_dir=None: "2" * 64)
    assert runner.Runner(q, units_root, reuse_commits=[old]).run() == {"first": "BLOCKED"}
