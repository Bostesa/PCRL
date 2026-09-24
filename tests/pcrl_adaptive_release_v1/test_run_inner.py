import hashlib
import json

from experiments.pcrl_adaptive_release_v1 import run_inner


def test_inner_dispatch_pins_canonical_names_and_aliases(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    output = private / "a0_inner"
    index = private / "a0_specs.json"
    input_index = private / "pinned-index.json"
    input_index.write_text("{}")
    seen = {}
    bundle = {"releases": {"A_selected": {"router": "T0",
                                           "channel_artifact_sha256": "a"*64}},
              "aliases": {"A_selected": "A_selected", "B_selected": "A_selected",
                          "A_control_D17": "A_selected"},
              "source_receipts": {"A_complete_sha256": "a"*64},
              "canonical_release_count": 1, "declared_name_count": 3,
              "deduplication": "exact", "outer_labels_accessed": False}
    monkeypatch.setattr(run_inner.release_specs, "build_release_specs",
                        lambda *args, **kwargs: bundle)

    def fake_audit(anchor, releases, input_index, target, *, resume, slate):
        seen.update(anchor=anchor, releases=releases, input_index=input_index,
                    target=target, resume=resume, slate=slate)
        target.mkdir()
        report = {"schema": "pcrl-adaptive-inner-audit-v1", "anchor": anchor,
                  "slate": slate, "outer_pool_opened": False,
                  "index_sha256": hashlib.sha256(input_index.read_bytes()).hexdigest(),
                  "releases": {"A_selected": {"source": {
                      "channel_artifact": {"sha256": "a"*64}}}}}
        (target / "INNER_AUDIT.json").write_text(json.dumps(report))
        receipt = {"schema": "pcrl-adaptive-inner-audit-complete-v1",
                   "anchor": anchor, "release_ids": ["A_selected"],
                   "index_sha256": report["index_sha256"],
                   "artifacts": run_inner.evaluate._inventory(target)}
        (target / "COMPLETE.json").write_text(json.dumps(receipt))
        return report

    monkeypatch.setattr(run_inner.evaluate, "audit_panel", fake_audit)
    result = run_inner.dispatch_inner(0, .001, input_index, "A-center",
                                      output, index, slate="catchup")
    assert seen["releases"] == bundle["releases"]
    assert seen["slate"] == "catchup"
    assert result["schema"] == "pcrl-adaptive-inner-audit-v1"
    saved = json.loads(index.read_text())
    assert saved["aliases"]["A_control_D17"] == "A_selected"
    assert saved["canonical_ids"] == ["A_selected"]
    assert saved["source_receipts"] == bundle["source_receipts"]
    binding = json.loads((private / "a0_specs.binding.json").read_text())
    assert binding["inner_complete_sha256"] == run_inner._sha(output / "COMPLETE.json")
    assert binding["spec_index_sha256"] == run_inner._sha(index)
    monkeypatch.setattr(run_inner.evaluate, "audit_panel",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("completed audit was refit")))
    recovered = run_inner.dispatch_inner(0, .001, input_index, "A-center",
                                         output, index, slate="catchup")
    assert recovered == result


def test_inner_dispatch_refuses_changed_spec_index(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    index = private / "specs.json"
    index.write_text("{}")
    input_index = private / "pinned-index.json"
    input_index.write_text("{}")
    monkeypatch.setattr(run_inner.release_specs, "build_release_specs",
                        lambda *args, **kwargs: {"releases": {"A": {}},
                                                  "aliases": {"A": "A"},
                                                  "source_receipts": {}})
    import pytest
    with pytest.raises(ValueError, match="differs"):
        run_inner.dispatch_inner(0, .001, input_index, "A", private / "panel", index)
