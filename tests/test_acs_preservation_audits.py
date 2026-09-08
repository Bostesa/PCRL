"""Nested audits preserve original trajectories, direct coordinates and state."""
import copy
import inspect
import json

import numpy as np
import pytest
import torch

from experiments.acs_bottleneck_catchup import fit_catchup
from experiments.acs_preservation_audits import (
    EXTENDED_AUDIT_CONFIG, fit_extended_auditors, fit_extended_catchup,
    save_extended_audits,
)
from experiments.acs_protection_audits import AUDIT_BUDGET, fit_primary_auditors
from experiments.acs_transfer_heads import _network, _state_hash, load_candidate


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def data(d=4, n=53, nv=29, classes=2):
    rng = np.random.default_rng(844)
    x, xv = rng.normal(size=(n, d))*3+8, rng.normal(size=(nv, d))*2-4
    x[:, -1], xv[:, -1] = 7, 7
    y, yv = np.arange(n) % classes, np.arange(nv) % classes
    return x, y, xv, yv


def test_fresh_nested_candidates_exactly_replay_original_prefix_and_full_paths():
    arrays = data()
    original_arrays = [a.copy() for a in arrays]
    budget = {"mlp": {"hidden": [8, 4], "epochs": 2, "batch_size": 16, "validation_interval": 1},
              "histgb": {"max_iter": 2}}
    original = fit_primary_auditors(*arrays, 2, 934, budget=budget)
    statics = {k: v for k, v in original["candidates"].items() if not k.startswith("mlp_")}
    static_metadata = {k: copy.deepcopy(v.metadata) for k, v in statics.items()}
    torch_rng = torch.get_rng_state().clone()
    extended = fit_extended_auditors(*arrays, 2, 934, epochs=6, nested_epochs=2,
                                    static_candidates=statics, budget=budget)
    assert torch.equal(torch_rng, torch.get_rng_state())
    full = fit_primary_auditors(*arrays, 2, 934, budget={**budget, "mlp": {**budget["mlp"], "epochs": 6}})
    for name, expected in (("nested120", original), ("nested360", full)):
        for cid in ("mlp_0", "mlp_1"):
            actual, reference = extended[name]["candidates"][cid], expected["candidates"][cid]
            np.testing.assert_array_equal(actual.predict_proba(arrays[2]), reference.predict_proba(arrays[2]))
            np.testing.assert_array_equal(actual.preprocessing.mean, arrays[0].mean(0))
            for field in ("initial_state_hash", "final_state_hash", "selected_state_hash", "schedule_hash",
                          "optimizer_steps", "training_row_exposures", "selected_epoch", "validation_curve"):
                assert actual.metadata[field] == reference.metadata[field], field
            assert actual.metadata["validation_scores"] == reference.metadata["validation_scores"]
    for cid, candidate in statics.items():
        assert extended["nested120"]["candidates"][cid] is candidate
        assert extended["nested360"]["candidates"][cid] is candidate
        assert candidate.metadata == static_metadata[cid]
    for actual, old in zip(arrays, original_arrays):
        np.testing.assert_array_equal(actual, old)
    assert extended["metadata"]["actual_new_optimizer_steps"] == 48
    json.dumps(extended["metadata"], allow_nan=False)
    assert AUDIT_BUDGET["mlp"]["epochs"] == 120
    assert EXTENDED_AUDIT_CONFIG["epochs"] == 360


def test_direct_catchup_has_exact_prefix_full_path_and_unchanged_inherited_state():
    arrays = data(d=16, n=271, classes=9)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(788)
        model = _network(16, [64, 32], 9).eval().requires_grad_(False)
    inherited = {"fit_pool": "representation_fit", "optimizer_steps": 900, "row_exposures": 8000}
    state_hash, torch_rng = _state_hash(model), torch.get_rng_state().clone()
    lower = fit_catchup(model, *arrays, 9, 454, epochs=5, inherited_exposure=inherited)
    upper = fit_catchup(model, *arrays, 9, 454, epochs=10, inherited_exposure=inherited)
    result = fit_extended_catchup(model, *arrays, 9, 454, epochs=10, nested_epochs=5, inherited_exposure=inherited)
    assert _state_hash(model) == state_hash
    assert torch.equal(torch_rng, torch.get_rng_state())
    for key, original in (("nested120", lower), ("nested360", upper)):
        actual, expected = result[key], original["catchup"]
        np.testing.assert_array_equal(actual.predict_proba(arrays[2]), expected.predict_proba(arrays[2]))
        np.testing.assert_array_equal(actual.preprocessing.mean, np.zeros(16))
        np.testing.assert_array_equal(actual.preprocessing.scale, np.ones(16))
        assert actual.preprocessing.fitted is False
        for field in ("initial_state_hash", "final_state_hash", "selected_state_hash", "schedule_hash",
                      "optimizer_steps", "training_row_exposures", "selected_epoch", "validation_curve"):
            assert actual.metadata[field] == expected.metadata[field], field
        assert actual.metadata["optimizer"]["restored_state"] is False
        assert actual.metadata["inherited_exposure"] == inherited
    np.testing.assert_array_equal(result["saved"].predict_proba(arrays[2]), lower["saved"].predict_proba(arrays[2]))
    assert result["metadata"]["initial_fidelity_exact"] is True
    assert result["metadata"]["actual_new_optimizer_steps"] == 20
    json.dumps(result["metadata"], allow_nan=False)


def test_actual_checkpoints_and_selected_candidates_roundtrip_without_overwrite(tmp_path):
    arrays = data()
    result = fit_extended_auditors(*arrays, 2, 733, epochs=4, nested_epochs=2,
        budget={"mlp": {"hidden": [8, 4], "batch_size": 16, "validation_interval": 1}})
    files = save_extended_audits(result, tmp_path/"audit")
    assert len(files) == 4
    for record in files:
        stored = torch.load(record["path"], weights_only=True)
        model = _network(4, [8, 4], 2)
        model.load_state_dict(stored["model_state"])
        assert _state_hash(model) == record["state_hash"]
        assert all(int(v["step"]) == record["optimizer_steps"] for v in stored["optimizer_state"]["state"].values())
        assert stored["schedule_rng_state"]["bit_generator"] == "PCG64"
    for budget in ("nested120", "nested360"):
        for cid, candidate in result[budget]["candidates"].items():
            restored = load_candidate(tmp_path/"audit"/budget/cid)
            np.testing.assert_array_equal(restored.predict_proba(arrays[2]), candidate.predict_proba(arrays[2]))
    with pytest.raises(FileExistsError):
        save_extended_audits(result, tmp_path/"audit")


def test_validation_labels_do_not_change_actual_last_state_or_schedule():
    x, y, xv, yv = data()
    kwargs = dict(epochs=4, nested_epochs=2, budget={"mlp": {"hidden": [8, 4], "validation_interval": 1}})
    first = fit_extended_auditors(x, y, xv, yv, 2, 873, **kwargs)
    second = fit_extended_auditors(x, y, xv, 1-yv, 2, 873, **kwargs)
    for budget in ("nested120", "nested360"):
        for cid in ("mlp_0", "mlp_1"):
            a, b = first[budget]["candidates"][cid].metadata, second[budget]["candidates"][cid].metadata
            for field in ("initial_state_hash", "final_state_hash", "schedule_hash", "optimizer_steps"):
                assert a[field] == b[field]
    for function in (fit_extended_auditors, fit_extended_catchup):
        assert not any("test" in p or "evaluation" in p for p in inspect.signature(function).parameters)


def test_exposed_controls_preserve_full_missing_category_schema_and_fallback():
    y, yv = np.tile([0, 2, 8], 8), np.tile([0, 8], 6)
    result = fit_extended_auditors(np.eye(9)[y], y, np.eye(9)[yv], yv, 9, 948,
        epochs=4, nested_epochs=2, budget={"mlp": {"hidden": [8, 4], "validation_interval": 1}})
    for key in ("nested120", "nested360"):
        for candidate in result[key]["candidates"].values():
            assert candidate.metadata["fit_support"][3] == 0
            assert candidate.metadata["fit_coverage_complete"] is False
            score = candidate.metadata["validation_scores"]
            assert len(score["per_class"]) == 9
            assert score["macro_auroc"] is None
    fallback = fit_extended_auditors(np.eye(2)[np.zeros(12, int)], np.zeros(12, int),
        np.eye(2)[[0, 1]], np.array([0, 1]), 2, 918, epochs=4, nested_epochs=2)
    assert fallback["metadata"]["actual_new_optimizer_steps"] == 0
    assert fallback["nested360"]["metadata"]["fallbacks"] == {"mlp_0": "single_fitting_class", "mlp_1": "single_fitting_class"}


def test_reused_static_inputs_cannot_silently_change():
    arrays = data()
    budget = {"mlp": {"epochs": 1}, "histgb": {"max_iter": 1}}
    old = fit_primary_auditors(*arrays, 2, 289, budget=budget)
    with pytest.raises(AssertionError, match="recipe/input mismatch"):
        fit_extended_auditors(arrays[0]+.1, *arrays[1:], 2, 289, epochs=4, nested_epochs=2,
                             static_candidates={"logistic": old["candidates"]["logistic"]}, budget=budget)


def test_runner_exact_prefix_check_rejects_a_changed_selected_state():
    from experiments.run_acs_preservation_extended import verify_nested
    arrays = data()
    original = fit_primary_auditors(*arrays, 2, 934,
        budget={"mlp": {"epochs": 2, "validation_interval": 1}, "histgb": {"max_iter": 1}})
    extended = fit_extended_auditors(*arrays, 2, 934, epochs=4, nested_epochs=2,
                                    budget={"mlp": {"validation_interval": 1}})
    old, new = original["candidates"]["mlp_0"], extended["nested120"]["candidates"]["mlp_0"]
    assert verify_nested(old, new, arrays[2])["exact_original_prefix"]
    new.metadata["selected_state_hash"] = "changed"
    with pytest.raises(AssertionError, match="Nested120 replay changed"):
        verify_nested(old, new, arrays[2])


def test_extension_all_release_order_and_saved_candidate_exclusion():
    from pathlib import Path
    from types import SimpleNamespace
    from experiments.run_acs_preservation_extended import release_sources, _selection
    cfg = {"continuation_arms": ["C_warmup_only", "D_warmup_only", "C_persistent", "D_persistent"],
           "init_reference_results": "init", "pca16_reference_results": "pca", "reference_results": "rich"}
    sources = release_sources(Path("study"), cfg, 2)
    assert len(sources) == 13
    assert [name for name, _, _ in sources[:4]] == ["beta_0p1_"+arm for arm in cfg["continuation_arms"]]
    assert [name for name, _, _ in sources[-5:]] == ["C_init", "D_init", "PCA16", "B_rich_bank", "C_tree_bank"]
    candidates = {cid: SimpleNamespace(metadata={"family": "mlp", "validation_scores": {"log_loss": loss, "auroc": .5}})
                  for cid, loss in (("mlp_0", .7), ("mlp_1", .6), ("catchup", .4), ("saved_adversary", .1))}
    primary, independent, _, metadata = _selection(candidates)
    assert primary["selected_family"] == "catchup"
    assert independent["selected_family"] == "mlp_1"
    assert metadata["saved_adversary_diagnostic_only"] is True


def test_independent_verifier_checks_actual_adam_rng_and_state(tmp_path):
    from scripts.verify_acs_preservation_extended import checkpoint_evidence
    result = fit_extended_auditors(*data(n=17), 2, 276, epochs=4, nested_epochs=2)
    save_extended_audits(result, tmp_path/"audit")
    path = tmp_path/"audit"/"last_training_mlp_0_epoch4.pt"
    metadata = result["nested360"]["candidates"]["mlp_0"].metadata
    evidence = checkpoint_evidence(path, metadata, 17)
    assert evidence["optimizer_steps"] == 4
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["schedule_rng_state"]["state"]["state"] += 1
    badpath = tmp_path/"changed.pt"
    torch.save(checkpoint, badpath)
    with pytest.raises(AssertionError):
        checkpoint_evidence(badpath, metadata, 17)


def test_historical_pca16_single_release_hash_schema_is_explicit():
    from experiments.run_acs_preservation_extended import release_output_hashes
    hashes = {"representation_fit": "fit_hash", "attacker_validation": "validation_hash"}
    pca = {"release": "PCA16", "dimension": 16, "component_indices": list(range(16)),
           "representation_fitted": False, "output_hashes": hashes}
    assert release_output_hashes(pca, "PCA16") == hashes
    assert release_output_hashes({"output_hashes": {"C_init": hashes}}, "C_init") == hashes
    wrong = {**pca, "component_indices": list(range(1, 17))}
    with pytest.raises(AssertionError):
        release_output_hashes(wrong, "PCA16")
    with pytest.raises(KeyError):
        release_output_hashes(pca, "C_init")
