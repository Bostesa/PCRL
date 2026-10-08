"""Synthetic Team B-to-S2 wire bridge checks; no fitting or Adult data."""

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pytest
import torch
import attack_eval
import common_eval

from attack_bridge import build_attack_view
from bottleneck import (BottleneckPair, model_fingerprint,
                        purpose_model_fingerprint, schema_fingerprint)


@dataclass
class DummyAttackView:
    row_id: np.ndarray
    unit: np.ndarray
    x: np.ndarray
    components: tuple[str, ...]
    wire_components: tuple[str, ...]
    component_columns: tuple[tuple[str, int, int], ...]
    wire_constants: tuple[tuple[str, str], ...]
    x_defense_aware: np.ndarray
    x_metadata: np.ndarray

    def validate(self):
        assert set(self.wire_components) == set(self.components).union(
            name for name, _ in self.wire_constants)
        assert sum(end-start for _, start, end in self.component_columns) == self.x.shape[1]
        assert self.x_defense_aware.shape[0] == self.x.shape[0]
        return self


def _model():
    model = BottleneckPair(
        feature_names=tuple(f"f{i}" for i in range(83)),
        forbidden_names=("SEX", "income_label", "occupation_label"),
        class_counts={"income": 2, "occupation": 6},
        log_variance_bounds=(-8.0, 4.0),
    )
    with torch.no_grad():
        for purpose, classes in (("income", 2), ("occupation", 6)):
            model.heads[purpose].weight.zero_()
            model.heads[purpose].bias.copy_(torch.arange(classes, dtype=torch.float32))
    return model


def _views(flag, model, cal=None):
    cal = cal or _cal()
    out = {}
    for purpose, classes in (("income", 2), ("occupation", 6)):
        logits = model.heads[purpose].bias.detach().numpy().astype(np.float64)
        raw = torch.softmax(torch.from_numpy(logits.astype(np.float32))
                            .to(torch.float64).reshape(1, -1), dim=1).numpy()[0]
        probabilities = common_eval.apply_global_temperature(
            raw[None, :], cal["calibration_records"][purpose])[0]
        out[purpose] = {
            "features": np.arange(16, dtype=np.float64),
            "logits": logits,
            "probabilities": probabilities,
            "decision": int(np.argmax(probabilities)),
            "metadata": {"purpose": purpose, "fallback": flag,
                         "temperature": 1.0 / cal["calibration_records"][purpose]["alpha"],
                         "calibration_family": "global_temperature",
                         "calibration_record_sha256": cal["calibration_record_sha256"][purpose],
                         "model_sha256": model_fingerprint(model),
                         "purpose_model_sha256": purpose_model_fingerprint(model, purpose),
                         "sampler_domain": "TeamBPurposeDrawV1",
                         "schema_sha256": schema_fingerprint(model.feature_names),
                         "source_id": "synthetic-source"},
        }
    return out


def _cal(alpha=1.0):
    records = {purpose: {"schema": "pcrl-portfolio-global-temp-v1",
                         "task": purpose, "alpha": alpha}
               for purpose in ("income", "occupation")}
    hashes = {purpose: hashlib.sha256((json.dumps(record, sort_keys=True,
                                                   separators=(",", ":")) + "\n").encode()).hexdigest()
              for purpose, record in records.items()}
    return {"calibration_records": records, "calibration_record_sha256": hashes}


@pytest.mark.parametrize("view", ["income", "occupation", "pair"])
def test_bridge_binds_every_wire_field_and_public_head(view):
    model = _model()
    rows = [("1", _views(False, model)), ("2", _views(True, model))]
    attack = build_attack_view(
        rows, view=view, row_id=[1, 2], unit=[11, 12],
        public_model=model, expected_source_id="synthetic-source",
        **_cal(),
        attack_view_type=DummyAttackView,
    )
    assert attack.x.shape[0] == 2
    assert attack.x_metadata.shape[1] > 0
    assert attack.x_defense_aware.shape[1] > attack.x.shape[1]
    assert not any("decision_onehot" in name for name in attack.components)
    assert any("metadata.fallback" in name for name in attack.components)
    assert any("model_sha256" in name for name, _ in attack.wire_constants)
    assert not any("private-a" in name for name in attack.components)


def test_bridge_rejects_head_bypass_and_missing_metadata():
    model = _model()
    rows = [("1", _views(False, model))]
    rows[0][1]["income"]["logits"][0] = 100.0
    with pytest.raises(ValueError, match="public head"):
        build_attack_view(rows, view="income", row_id=[1], unit=[11],
                          public_model=model, expected_source_id="synthetic-source",
                          **_cal(),
                          attack_view_type=DummyAttackView)


def test_bridge_rejects_shuffled_people_and_clean_probability_bypass():
    model = _model()
    rows = [("1", _views(False, model)), ("2", _views(False, model))]
    with pytest.raises(ValueError, match="ordered row ID"):
        build_attack_view(rows, view="pair", row_id=[2, 1], unit=[12, 11],
                          public_model=model, expected_source_id="synthetic-source",
                          **_cal(),
                          attack_view_type=DummyAttackView)
    with pytest.raises(ValueError, match="int64"):
        build_attack_view(rows, view="pair", row_id=[1.5, 2.5], unit=[11, 12],
                          public_model=model, expected_source_id="synthetic-source",
                          **_cal(),
                          attack_view_type=DummyAttackView)
    rows[0][1]["income"]["probabilities"] = np.array([0.4, 0.6])
    with pytest.raises(ValueError, match="bypasses"):
        build_attack_view(rows, view="income", row_id=[1, 2], unit=[11, 12],
                          public_model=model, expected_source_id="synthetic-source",
                          **_cal(),
                          attack_view_type=DummyAttackView)


def test_bridge_meets_actual_s2_attack_view_contract():
    model = _model()
    rows = [("1", _views(False, model)), ("2", _views(True, model))]
    actual = build_attack_view(
        rows, view="pair", row_id=[1, 2], unit=[11, 12],
        public_model=model, expected_source_id="synthetic-source",
        **_cal(),
        attack_view_type=attack_eval.AttackView,
    )
    assert isinstance(actual, attack_eval.AttackView)
    assert actual.wire_constants
    assert any("metadata.fallback" in name for name in actual.components)


def test_bridge_replays_s2_floor_law_on_saturated_logits():
    model = _model()
    with torch.no_grad():
        model.heads["income"].bias.copy_(torch.tensor([0.0, -1000.0]))
        model.heads["occupation"].bias.copy_(torch.tensor([0.0] + [-1000.0] * 5))
    cal = _cal(alpha=0.5)
    rows = [("1", _views(False, model, cal)), ("2", _views(True, model, cal))]
    for purpose in ("income", "occupation"):
        assert np.min(rows[0][1][purpose]["probabilities"]) > 0
    actual = build_attack_view(rows, view="pair", row_id=[1, 2], unit=[11, 12],
                               public_model=model, expected_source_id="synthetic-source",
                               **cal, attack_view_type=attack_eval.AttackView)
    assert actual.x.shape[0] == 2
    rows[0][1]["income"]["probabilities"] = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="bypasses"):
        build_attack_view(rows, view="pair", row_id=[1, 2], unit=[11, 12],
                          public_model=model, expected_source_id="synthetic-source",
                          **cal, attack_view_type=attack_eval.AttackView)


def test_bridge_rejects_micro_payload_in_probability_and_logit():
    model = _model()
    cal = _cal()
    rows = [("1", _views(False, model, cal)), ("2", _views(True, model, cal))]
    p = rows[0][1]["income"]["probabilities"].copy()
    rows[0][1]["income"]["probabilities"] = p + np.array([1e-6, -1e-6])
    with pytest.raises(ValueError, match="bypasses"):
        build_attack_view(rows, view="income", row_id=[1, 2], unit=[11, 12],
                          public_model=model, expected_source_id="synthetic-source",
                          **cal, attack_view_type=attack_eval.AttackView)
    rows[0][1]["income"]["probabilities"] = p
    rows[0][1]["income"]["logits"][0] += 1e-6
    with pytest.raises(ValueError, match="public head"):
        build_attack_view(rows, view="income", row_id=[1, 2], unit=[11, 12],
                          public_model=model, expected_source_id="synthetic-source",
                          **cal, attack_view_type=attack_eval.AttackView)
