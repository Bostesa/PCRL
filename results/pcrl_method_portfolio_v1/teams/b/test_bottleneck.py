"""Contract tests for Team B's synthetic-only bottleneck prototype.

These tests exercise release and loss plumbing, not Adult outcomes or a
scientific synthetic law. No real-data loader is imported.
"""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F
import common_eval

from bottleneck import (
    BottleneckPair,
    LossWeights,
    ReleaseAdapter,
    SavedRelease,
    SexCritics,
    complete_view_tensor,
    diagonal_gaussian_kl,
    model_fingerprint,
    purpose_model_fingerprint,
    schema_fingerprint,
    training_terms,
)
from train_bottleneck import FitData, FrozenPurposeView

FEATURE_NAMES = tuple(f"f{i}" for i in range(83))
FORBIDDEN_NAMES = ("SEX", "income_label", "occupation_label")
TEMPERATURES = {"income": 2.0, "occupation": 1.5}


def pair():
    return BottleneckPair(
        feature_names=FEATURE_NAMES,
        forbidden_names=FORBIDDEN_NAMES,
        class_counts={"income": 2, "occupation": 6},
        log_variance_bounds=(-8.0, 4.0),
    )


def _synthetic_calibration(temperatures):
    records, hashes = {}, {}
    for purpose in ("income", "occupation"):
        record = {"schema": "pcrl-portfolio-global-temp-v1", "task": purpose,
                  "alpha": 1.0 / temperatures[purpose]}
        records[purpose] = record
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        hashes[purpose] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return records, hashes


def _secret(seed):
    return hashlib.sha256(f"team-b-synthetic-test-key:{seed}".encode("ascii")).digest()


def adapter(model=None, *, stochastic=True, seed=3, temperatures=TEMPERATURES):
    model = model or pair()
    valid = set(temperatures) == {"income", "occupation"} and all(
        value > 0 for value in temperatures.values()
    )
    records, hashes = _synthetic_calibration(temperatures) if valid else (None, None)
    return ReleaseAdapter(
        model, stochastic=stochastic, sampler_secret=_secret(seed),
        temperatures=temperatures, source_id="synthetic-test-source",
        expected_model_sha256=model_fingerprint(model),
        expected_schema_sha256=schema_fingerprint(model.feature_names),
        calibration_records=records, calibration_record_sha256=hashes,
        raw_mode=not valid,
    )


def test_release_uses_the_transmitted_sample_for_its_head_and_reuses_one_draw():
    """Catches a teacher-score bypass or a second draw for the same recipient."""
    torch.manual_seed(11)
    model = pair()
    builder = adapter(model, seed=12)
    x = np.arange(83, dtype=np.float32) / 83

    first = builder.release("income", "private-row-key", x, columns=FEATURE_NAMES)
    second = builder.release("income", "private-row-key", x, columns=FEATURE_NAMES)
    assert np.array_equal(first["features"], second["features"])
    assert np.array_equal(first["probabilities"], second["probabilities"])
    with torch.no_grad():
        expected_logits = model.heads["income"](
            torch.from_numpy(first["features"]).unsqueeze(0)
        ).squeeze(0).numpy()
    np.testing.assert_allclose(first["logits"], expected_logits, atol=1e-6)
    np.testing.assert_allclose(
        first["probabilities"], torch.softmax(torch.from_numpy(expected_logits) / 2.0, 0).numpy(), atol=1e-6
    )
    assert first["decision"] == int(np.argmax(first["probabilities"]))
    assert set(first) == {"features", "logits", "probabilities", "decision", "metadata"}
    assert "private-row-key" not in repr(first)
    assert first["metadata"]["draws_per_person"] == 1
    assert first["metadata"]["temperature"] == 2.0


def test_release_refuses_extra_columns_and_changed_input_for_cached_person():
    """Catches accepting a forbidden predictor or silently reusing the wrong person."""
    model = pair()
    builder = adapter(model)
    x = np.zeros(83, dtype=np.float32)
    with pytest.raises(ValueError, match="83"):
        builder.release("income", "a", np.zeros(84, dtype=np.float32), columns=FEATURE_NAMES)
    builder.release("income", "a", x, columns=FEATURE_NAMES)
    with pytest.raises(ValueError, match="changed"):
        builder.release("income", "a", np.ones(83, dtype=np.float32), columns=FEATURE_NAMES)


def test_release_refuses_reordered_or_forbidden_columns():
    """Catches schema drift and accidentally including a protected label."""
    model = pair()
    builder = adapter(model)
    x = np.zeros(83, dtype=np.float32)
    reordered = tuple(reversed(FEATURE_NAMES))
    with pytest.raises(ValueError, match="column order"):
        builder.release("income", "a", x, columns=reordered)
    forbidden = FEATURE_NAMES[:-1] + ("SEX",)
    with pytest.raises(ValueError, match="forbidden"):
        BottleneckPair(
            feature_names=forbidden,
            forbidden_names=FORBIDDEN_NAMES,
            class_counts={"income": 2, "occupation": 6},
            log_variance_bounds=(-8.0, 4.0),
        )


def test_recipients_are_local_and_deterministic_ablation_uses_mean():
    """Catches dependence on the other release and a sampler in B-D."""
    torch.manual_seed(13)
    model = pair()
    x = np.linspace(-1, 1, 83, dtype=np.float32)
    deterministic = adapter(model, stochastic=False, seed=4)
    income = deterministic.release("income", "a", x, columns=FEATURE_NAMES)
    deterministic.release("occupation", "other-person", -x, columns=FEATURE_NAMES)
    repeat = adapter(model, stochastic=False, seed=999).release(
        "income", "a", x, columns=FEATURE_NAMES
    )
    np.testing.assert_array_equal(income["features"], repeat["features"])
    with torch.no_grad():
        mu, _ = model.encoders["income"](torch.from_numpy(x).unsqueeze(0))
    np.testing.assert_allclose(income["features"], mu.squeeze(0).numpy(), atol=1e-6)
    assert income["metadata"]["stochastic"] is False


def test_stochastic_draw_is_local_to_person_purpose_and_private_key():
    """Catches release-call order determining another recipient's noise."""
    model = pair()
    x = np.zeros(83, dtype=np.float32)
    first = adapter(model, seed=7).release("income", "private-a", x, columns=FEATURE_NAMES)
    reordered = adapter(model, seed=7)
    reordered.release("occupation", "private-b", x, columns=FEATURE_NAMES)
    again = reordered.release("income", "private-a", x, columns=FEATURE_NAMES)
    np.testing.assert_array_equal(first["features"], again["features"])
    assert "private-a" not in repr(first)


@pytest.mark.parametrize("first_purpose,second_purpose", [
    ("income", "occupation"), ("occupation", "income"),
])
def test_sequential_first_fixed_draw_survives_other_purpose_training(
    first_purpose, second_purpose,
):
    """The final combined model replays stage 1's one draw in either order."""
    stage1 = pair()
    x = np.linspace(-0.5, 0.5, 83, dtype=np.float32)
    data = FitData(
        x=torch.from_numpy(x[None, :]),
        labels={"income": torch.tensor([0]), "occupation": torch.tensor([2])},
        teacher_probabilities={"income": torch.tensor([[0.6, 0.4]]),
                               "occupation": torch.tensor([[0.1, 0.1, 0.5, 0.1, 0.1, 0.1]])},
        sex=torch.tensor([0]), feature_names=FEATURE_NAMES,
        permitted_feature_names=FEATURE_NAMES,
        expected_schema_sha256=schema_fingerprint(FEATURE_NAMES),
        row_ids=("private-a",), group_ids=("group-a",),
        teacher_identity={"income": "Ucal_1", "occupation": "Ucal_2"},
    )
    frozen = FrozenPurposeView.from_model(
        stage1, data, purpose=first_purpose, source_id="synthetic-test-source",
        first_checkpoint_sha256="a" * 64, sampler_plan_sha256="b" * 64,
        sampler_secret=_secret(7),
    )
    final = copy.deepcopy(stage1)
    with torch.no_grad():
        final.heads[second_purpose].weight.add_(0.25)
    assert model_fingerprint(final) != model_fingerprint(stage1)
    assert purpose_model_fingerprint(final, first_purpose) == purpose_model_fingerprint(stage1, first_purpose)
    released = adapter(final, seed=7).release(first_purpose, "private-a", x, columns=FEATURE_NAMES)
    np.testing.assert_array_equal(released["features"], frozen.z[0].numpy())
    np.testing.assert_array_equal(released["logits"], frozen.logits[0].numpy())
    assert released["metadata"]["purpose_model_sha256"] == purpose_model_fingerprint(stage1, first_purpose)
    with torch.no_grad():
        final.heads[first_purpose].weight.add_(0.25)
    assert purpose_model_fingerprint(final, first_purpose) != purpose_model_fingerprint(stage1, first_purpose)


def test_finalized_batch_replays_exact_views_from_private_immutable_file(tmp_path):
    """Catches accidental support for a repeated-query release interface."""
    builder = adapter(seed=4)
    x = np.zeros(83, dtype=np.float32)
    income = builder.release("income", "private-a", x, columns=FEATURE_NAMES)
    occupation = builder.release("occupation", "private-a", x, columns=FEATURE_NAMES)
    saved_path = tmp_path / "private-release.json"
    builder.finalize(saved_path)
    assert saved_path.stat().st_mode & 0o777 == 0o600
    replay = SavedRelease(saved_path)
    np.testing.assert_array_equal(replay.get("income", "private-a")["features"], income["features"])
    np.testing.assert_array_equal(replay.get("occupation", "private-a")["probabilities"], occupation["probabilities"])
    changed = replay.get("income", "private-a")
    changed["features"][0] = 999.0
    changed["metadata"]["log_variance_bounds"][0] = 999.0
    np.testing.assert_array_equal(replay.get("income", "private-a")["features"], income["features"])
    assert replay.get("income", "private-a")["metadata"]["log_variance_bounds"][0] == -8.0
    assert hashlib.sha256(b"team-b-synthetic-test-key:4").hexdigest() not in saved_path.read_text()
    with pytest.raises(FileExistsError):
        adapter(seed=99).finalize(saved_path)
    with pytest.raises(RuntimeError, match="finalized"):
        builder.release("income", "private-a", x, columns=FEATURE_NAMES)


def test_batch_requires_aligned_pair_before_it_can_be_sealed(tmp_path):
    """Catches an apparently paired batch with one recipient silently missing."""
    builder = adapter(seed=8)
    builder.release("income", "private-a", np.zeros(83, dtype=np.float32), columns=FEATURE_NAMES)
    with pytest.raises(ValueError, match="aligned pair"):
        builder.finalize(tmp_path / "incomplete.json")
    builder.release("occupation", "private-a", np.ones(83, dtype=np.float32), columns=FEATURE_NAMES)
    with pytest.raises(ValueError, match="same input"):
        builder.finalize(tmp_path / "mismatched.json")


def test_kl_uses_sum_over_latent_coordinates_and_mean_over_people():
    """Catches averaging away a two-coordinate information penalty."""
    mean = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    log_variance = torch.zeros_like(mean)
    assert diagonal_gaussian_kl(mean, log_variance).item() == pytest.approx(0.5)


def test_complete_view_includes_each_released_feature_and_head_probability():
    """Catches giving the coalition critic only one purpose's representation."""
    outputs = {
        "income": {"z": torch.tensor([[2.0, 3.0]]), "logits": torch.tensor([[1.0, 2.0]]),
                   "probabilities": torch.tensor([[0.25, 0.75]])},
        "occupation": {"z": torch.tensor([[5.0]]), "logits": torch.tensor([[0.0, 1.0, 2.0]]),
                       "probabilities": torch.tensor([[0.1, 0.2, 0.7]])},
    }
    local, coalition = complete_view_tensor(outputs)
    torch.testing.assert_close(local["income"], torch.tensor([[2.0, 3.0, 1.0, 2.0, 0.25, 0.75, 0.0, 1.0]]))
    torch.testing.assert_close(local["occupation"], torch.tensor([[5.0, 0.0, 1.0, 2.0, 0.1, 0.2, 0.7, 0.0, 0.0, 1.0]]))
    torch.testing.assert_close(coalition, torch.tensor([[2.0, 3.0, 1.0, 2.0, 0.25, 0.75, 0.0, 1.0,
                                                         5.0, 0.0, 1.0, 2.0, 0.1, 0.2, 0.7, 0.0, 0.0, 1.0]]))


def test_training_terms_include_supervision_distillation_kl_and_both_privacy_views():
    """Catches silently omitting an objective component or training on mean z."""
    torch.manual_seed(17)
    model = pair()
    critics = SexCritics(latent_dim=16, class_counts=model.class_counts)
    x = torch.randn(4, 83)
    outputs = model.training_outputs(x, stochastic=True, generator=torch.Generator().manual_seed(7))
    labels = {"income": torch.tensor([0, 1, 0, 1]), "occupation": torch.tensor([0, 1, 2, 3])}
    teacher = {name: torch.softmax(result["logits"].detach(), 1) for name, result in outputs.items()}
    terms = training_terms(outputs, critics, labels, teacher, torch.tensor([0, 1, 0, 1]))
    assert set(terms) == {"task_ce", "distillation", "bottleneck_kl", "local_sex_ce", "coalition_sex_ce"}
    assert terms["task_ce"].item() > 0
    assert terms["distillation"].item() == pytest.approx(0.0, abs=1e-6)
    assert terms["bottleneck_kl"].item() >= 0
    assert terms["local_sex_ce"].item() > 0
    assert terms["coalition_sex_ce"].item() > 0
    expected_task_ce = sum(F.cross_entropy(outputs[name]["logits"], labels[name]) for name in labels) / 2
    torch.testing.assert_close(terms["task_ce"], expected_task_ce)
    expected_kl = sum(diagonal_gaussian_kl(outputs[name]["mean"], outputs[name]["log_variance"])
                      for name in labels) / 2
    torch.testing.assert_close(terms["bottleneck_kl"], expected_kl)
    weights = LossWeights(task_ce=1, distillation=1, bottleneck_kl=1, local_sex=1, coalition_sex=1)
    objective = weights.encoder_objective(terms)
    assert objective.item() == pytest.approx(
        terms["task_ce"].item() + terms["distillation"].item() + terms["bottleneck_kl"].item()
        - terms["local_sex_ce"].item() - terms["coalition_sex_ce"].item()
    )


def test_calibration_scalars_are_required_and_positive():
    """Catches silently scoring uncalibrated probabilities."""
    model = pair()
    with pytest.raises(ValueError, match="temperature"):
        adapter(model, temperatures={"income": 1.0})
    with pytest.raises(ValueError, match="temperature"):
        adapter(model, temperatures={"income": 0.0, "occupation": 1.0})


def test_deployed_calibration_uses_s2_floor_law_exactly(tmp_path):
    """A tiny raw class probability is clipped by S2's fitted law on release."""
    model = pair()
    with torch.no_grad():
        model.heads["income"].weight.zero_()
        model.heads["income"].bias.copy_(torch.tensor([0.0, -40.0]))
    builder = adapter(model, stochastic=False)
    release = builder.release(
        "income", "private-a", np.zeros(83, dtype=np.float32), columns=FEATURE_NAMES
    )
    raw = torch.softmax(torch.from_numpy(release["logits"]).to(torch.float64), dim=0).numpy()
    assert raw.min() < 1e-12
    record = _synthetic_calibration(TEMPERATURES)[0]["income"]
    expected = common_eval.apply_global_temperature(raw.reshape(1, -1), record)[0]
    np.testing.assert_array_equal(release["probabilities"], expected)
    assert release["metadata"]["calibration_record_sha256"] == _synthetic_calibration(TEMPERATURES)[1]["income"]
    builder.release("occupation", "private-a", np.zeros(83, dtype=np.float32),
                    columns=FEATURE_NAMES)
    saved_path = tmp_path / "floor-release.json"
    builder.finalize(saved_path)
    np.testing.assert_array_equal(SavedRelease(saved_path).get("income", "private-a")["probabilities"], expected)


def test_unfitted_raw_calibration_batch_cannot_be_finalized(tmp_path):
    model = pair()
    raw = ReleaseAdapter(
        model, stochastic=False, sampler_secret=_secret(4),
        temperatures={"income": 1.0, "occupation": 1.0},
        source_id="synthetic-test-source",
        expected_model_sha256=model_fingerprint(model),
        expected_schema_sha256=schema_fingerprint(model.feature_names),
        raw_mode=True,
    )
    x = np.zeros(83, dtype=np.float32)
    for purpose in ("income", "occupation"):
        raw.release(purpose, "private-a", x, columns=FEATURE_NAMES)
    with pytest.raises(ValueError, match="cannot be finalized"):
        raw.finalize(tmp_path / "raw-release.json")


def test_saved_release_refuses_planted_extra_top_level_wire_field(tmp_path):
    """Private restore must not silently erase a field that an attacker should see."""
    builder = adapter(seed=14)
    x = np.zeros(83, dtype=np.float32)
    for purpose in ("income", "occupation"):
        builder.release(purpose, "private-a", x, columns=FEATURE_NAMES)
    path = tmp_path / "private-release.json"
    builder.finalize(path)
    wrapped = json.loads(path.read_text())
    wrapped["payload"]["records"][0]["view"]["planted_sensitive_clue"] = 1.0
    canonical = json.dumps(wrapped["payload"], sort_keys=True, separators=(",", ":"))
    wrapped["payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path.write_text(json.dumps(wrapped, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError, match="unexpected transmitted view field"):
        SavedRelease(path)


def test_log_variance_is_bounded_and_nonfinite_release_is_refused():
    """Catches Gaussian overflow and publishing a NaN score as a release."""
    model = pair()
    with torch.no_grad():
        model.encoders["income"].log_variance.weight.zero_()
        model.encoders["income"].log_variance.bias.fill_(1000.0)
    _, log_variance = model.encoders["income"](torch.zeros(1, 83))
    assert torch.all(log_variance == 4.0)
    with torch.no_grad():
        model.heads["income"].bias.fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        adapter(model).release("income", "a", np.zeros(83, dtype=np.float32), columns=FEATURE_NAMES)


def test_deployment_rejects_wrong_pinned_model_or_schema():
    """Catches serving a different checkpoint under the same campaign arm."""
    model = pair()
    with pytest.raises(ValueError, match="model identity"):
        ReleaseAdapter(model, stochastic=True, sampler_secret=b"x" * 32, temperatures=TEMPERATURES,
                       source_id="synthetic-test-source", expected_model_sha256="0" * 64,
                       expected_schema_sha256=schema_fingerprint(model.feature_names),
                       calibration_records=_synthetic_calibration(TEMPERATURES)[0],
                       calibration_record_sha256=_synthetic_calibration(TEMPERATURES)[1])
    with pytest.raises(ValueError, match="schema identity"):
        ReleaseAdapter(model, stochastic=True, sampler_secret=b"x" * 32, temperatures=TEMPERATURES,
                       source_id="synthetic-test-source", expected_model_sha256=model_fingerprint(model),
                       expected_schema_sha256="0" * 64,
                       calibration_records=_synthetic_calibration(TEMPERATURES)[0],
                       calibration_record_sha256=_synthetic_calibration(TEMPERATURES)[1])


def test_stochastic_adapter_requires_private_sampler_key():
    """Catches a guessable public integer seed collapsing sampling privacy."""
    model = pair()
    with pytest.raises(ValueError, match="private sampler key"):
        ReleaseAdapter(model, stochastic=True, sampler_secret=b"short", temperatures=TEMPERATURES,
                       source_id="synthetic-test-source", expected_model_sha256=model_fingerprint(model),
                       expected_schema_sha256=schema_fingerprint(model.feature_names),
                       calibration_records=_synthetic_calibration(TEMPERATURES)[0],
                       calibration_record_sha256=_synthetic_calibration(TEMPERATURES)[1])
