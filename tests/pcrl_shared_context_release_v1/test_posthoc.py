"""Exploratory post-hoc family: existing locked outer scores only, own seed, own Bonferroni."""
from __future__ import annotations

import csv
import json

import pytest
from scipy.stats import norm

from experiments.pcrl_adaptive_release_v1 import evaluate
from experiments.pcrl_shared_context_release_v1 import posthoc

from tests.pcrl_shared_context_release_v1.test_assess import setup


def test_posthoc_family_from_existing_outer_scores(tmp_path):
    value, path, sha, outer, _ = setup(tmp_path, aliases={"DET_SEL1": "D17"})
    before = {a: sorted(p.name for p in outer[a].iterdir()) for a in outer}
    out = tmp_path / "posthoc"
    report = posthoc.run(path, sha, outer, out, n_boot=200)
    assert report["status"] == "EXPLORATORY_POST_HOC_NOT_REGISTERED"
    assert report["family_size"] == 40 and report["bootstrap_seed"] == 20260925
    assert report["critical_value_two_sided"] == pytest.approx(norm.isf(.05 / 80))
    assert report["candidate_identical_to_D17_on_all_anchors"]["DET_SEL1"] is True
    assert report["candidate_identical_to_D17_on_all_anchors"]["NM1_P"] is False
    det1 = [r for r in report["rows"] if r["candidate_release"] == "DET_SEL1"]
    assert all(r["estimate"] == 0. for r in det1)
    task = next(r for r in report["rows"] if r["candidate_release"] == "RD_TASK" and r["clause"] == "task")
    assert (task["plus"], task["minus"]) == ("RD_TASK", "D17")
    guard = next(r for r in report["rows"] if r["candidate_release"] == "RD_TASK" and r["clause"] == "guard")
    assert (guard["plus"], guard["minus"]) == ("D17", "RD_TASK")
    nm1p = {r["clause"] for r in report["rows"] if r["candidate_release"] == "NM1_P"}
    assert nm1p == {"task", "target", "guard"}
    with (out / "POSTHOC_EXPLORATORY.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 40 and {r["status"] for r in rows} == {posthoc.STATUS}
    assert {a: sorted(p.name for p in outer[a].iterdir()) for a in outer} == before  # read-only
    with pytest.raises(FileExistsError):
        posthoc.run(path, sha, outer, out, n_boot=200)


def test_posthoc_refuses_when_a_release_was_not_scored(tmp_path):
    value, path, sha, outer, _ = setup(tmp_path)
    report_path = outer[2] / "OUTER_AUDIT.json"
    record = json.loads(report_path.read_text())
    del record["releases"]["DET_SEL4"]
    report_path.write_text(json.dumps(record))
    (outer[2] / "COMPLETE.json").write_text(json.dumps(
        {"schema": "pcrl-sc-outer-audit-complete-v1", "anchor": 2,
         "artifacts": evaluate._inventory(outer[2])}))
    with pytest.raises(FileNotFoundError, match="DET_SEL4@a2"):
        posthoc.run(path, sha, outer, tmp_path / "posthoc", n_boot=50)
    assert not (tmp_path / "posthoc").exists()


def test_posthoc_refuses_other_lock_bytes(tmp_path):
    value, path, sha, outer, _ = setup(tmp_path)
    with pytest.raises(PermissionError):
        posthoc.run(path, "0" * 64, outer, tmp_path / "posthoc", n_boot=50)
