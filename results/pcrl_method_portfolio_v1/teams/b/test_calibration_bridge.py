"""Synthetic one-draw calibration wiring against S2's fixed scalar family."""

import numpy as np
import pytest

from bottleneck import ReleaseAdapter, model_fingerprint, schema_fingerprint
from calibration_bridge import _sha_arr, fit_calibration
from test_attack_bridge import _model


def test_fixed_calibration_uses_actual_released_draw(tmp_path):
    common_eval = pytest.importorskip("common_eval")
    model = _model()
    ids = np.arange(1, 9, dtype=np.int64)
    groups = ids + 10
    x = np.zeros((len(ids), 83), dtype=np.float32)
    secret = b"s" * 32
    model_sha = model_fingerprint(model)
    schema_sha = schema_fingerprint(model.feature_names)
    fitted = fit_calibration(
        model, arm_id="B-SYNTHETIC", model_seed=0,
        feature_record={"X": x, "row_id": ids, "unit": groups,
                        "feature_names": np.asarray(model.feature_names)},
        task_label_records={
            "income": {"row_id": ids, "unit": groups,
                       "income": np.array([0, 1] * 4, dtype=np.int64)},
            "occupation": {"row_id": ids, "unit": groups,
                           "occupation": np.array([0, 1, 2, 3, 4, 5, 0, 1], dtype=np.int64)},
        },
        source_id="synthetic-source", stochastic=True, sampler_secret=secret,
        expected_model_sha256=model_sha, expected_schema_sha256=schema_sha,
        admitted_cal_pair_sha256=_sha_arr(np.column_stack((ids, groups))),
        private_directory=tmp_path / ".private" / "cal-s0",
    )
    assert set(fitted.records) == {"income", "occupation"}
    assert fitted.paired_calibration_forward_passes == 2
    assert max(fitted.replay_max_probability_error.values()) <= 1e-6
    raw_adapter = ReleaseAdapter(
        model, stochastic=True, sampler_secret=secret,
        temperatures={"income": 1.0, "occupation": 1.0},
        source_id="synthetic-source", expected_model_sha256=model_sha,
        expected_schema_sha256=schema_sha,
    )
    for purpose in ("income", "occupation"):
        raw = raw_adapter.release(purpose, "1", x[0], columns=model.feature_names)
        final = fitted.adapter.release(purpose, "1", x[0], columns=model.feature_names)
        assert np.array_equal(raw["features"], final["features"])
        expected = common_eval.apply_global_temperature(
            raw["probabilities"][None, :], fitted.records[purpose]
        )[0]
        assert np.allclose(final["probabilities"], expected, atol=1e-6)


def test_calibration_refuses_wrong_admitted_pair(tmp_path):
    pytest.importorskip("common_eval")
    model = _model()
    ids = np.arange(1, 3, dtype=np.int64)
    with pytest.raises(ValueError, match="admitted role"):
        fit_calibration(
            model, arm_id="B-SYNTHETIC", model_seed=0,
            feature_record={"X": np.zeros((2, 83), dtype=np.float32),
                            "row_id": ids, "unit": ids,
                            "feature_names": np.asarray(model.feature_names)},
            task_label_records={
                "income": {"row_id": ids, "unit": ids,
                           "income": np.array([0, 1])},
                "occupation": {"row_id": ids, "unit": ids,
                               "occupation": np.array([0, 1])},
            },
            source_id="synthetic-source", stochastic=False,
            sampler_secret=b"s" * 32,
            expected_model_sha256=model_fingerprint(model),
            expected_schema_sha256=schema_fingerprint(model.feature_names),
            admitted_cal_pair_sha256="0" * 64,
            private_directory=tmp_path / ".private" / "bad",
        )


def test_calibration_rejects_permuted_label_record_and_columns(tmp_path):
    pytest.importorskip("common_eval")
    model = _model()
    ids = np.arange(1, 3, dtype=np.int64)
    features = {"X": np.zeros((2, 83), dtype=np.float32),
                "row_id": ids, "unit": ids,
                "feature_names": np.asarray(model.feature_names)}
    labels = {
        "income": {"row_id": ids[::-1], "unit": ids[::-1],
                   "income": np.array([0, 1])},
        "occupation": {"row_id": ids, "unit": ids,
                       "occupation": np.array([0, 1])},
    }
    kwargs = dict(
        arm_id="B-SYNTHETIC", model_seed=0, feature_record=features,
        task_label_records=labels, source_id="synthetic-source", stochastic=False,
        sampler_secret=b"s" * 32, expected_model_sha256=model_fingerprint(model),
        expected_schema_sha256=schema_fingerprint(model.feature_names),
        admitted_cal_pair_sha256=_sha_arr(np.column_stack((ids, ids))),
        private_directory=tmp_path / ".private" / "bad-join",
    )
    with pytest.raises(ValueError, match="ordered row/group"):
        fit_calibration(model, **kwargs)
    labels["income"]["row_id"] = ids
    labels["income"]["unit"] = ids
    features["feature_names"] = features["feature_names"][::-1]
    with pytest.raises(ValueError, match="ordered schema"):
        fit_calibration(model, **kwargs)
