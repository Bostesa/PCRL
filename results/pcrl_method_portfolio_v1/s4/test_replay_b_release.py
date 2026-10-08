"""Synthetic S4 release reconstruction checks; no Adult files or candidate imports."""

import hashlib
import json

import numpy as np
import pytest
import torch

from replay_b_release import (head_state, load_saved_wire, model_sha,
                              ordered_split, purpose_sha, rebuild_view,
                              replay_encoder, schema_sha)


def _state():
    state = {}
    for purpose, width in (("income", 2), ("occupation", 6)):
        prefix = f"encoders.{purpose}."
        for name, shape in (("body.0.weight", (64, 83)), ("body.0.bias", (64,)),
                            ("body.2.weight", (64, 64)), ("body.2.bias", (64,)),
                            ("mean.weight", (16, 64)), ("mean.bias", (16,)),
                            ("log_variance.weight", (16, 64)),
                            ("log_variance.bias", (16,))):
            state[prefix + name] = torch.zeros(shape, dtype=torch.float32)
        state[prefix + "logvar_min"] = torch.tensor(-8.0)
        state[prefix + "logvar_max"] = torch.tensor(4.0)
        state[f"heads.{purpose}.weight"] = torch.zeros((width, 16), dtype=torch.float32)
        state[f"heads.{purpose}.bias"] = torch.arange(width, dtype=torch.float32)
    return state


def _fixture(tmp_path):
    state = _state()
    names = tuple(f"f{i}" for i in range(83))
    rows = np.arange(166, dtype=np.float32).reshape(2, 83)
    ids = np.array([11, 22], dtype=np.int64)
    source = "synthetic-source"
    model = model_sha(state)
    calibrator = {"alpha": 1.0}
    cal_sha = hashlib.sha256((json.dumps(calibrator, sort_keys=True,
                                        separators=(",", ":")) + "\n").encode()).hexdigest()
    views = []
    for i in range(2):
        pair = {}
        for purpose, width in (("income", 2), ("occupation", 6)):
            q = torch.softmax(state[f"heads.{purpose}.bias"].to(torch.float64), dim=0).numpy()
            pair[purpose] = {
                "features": [0.0] * 16,
                "logits": list(map(float, range(width))),
                "probabilities": q.tolist(),
                "decision": width - 1,
                "metadata": {
                    "purpose": purpose,
                    "stochastic": False,
                    "draws_per_person": 0,
                    "latent_dim": 16,
                    "log_variance_bounds": [-8.0, 4.0],
                    "calibration_family": "global_temperature",
                    "temperature": 1.0,
                    "calibration_record_sha256": cal_sha,
                    "source_id": source,
                    "model_sha256": model,
                    "purpose_model_sha256": purpose_sha(state, purpose, names),
                    "sampler_domain": "TeamBPurposeDrawV1",
                    "schema_sha256": schema_sha(names),
                    "release_contract": "immutable_one_shot_batch",
                    "planted_flag": float(i),
                },
            }
        views.append(pair)
    records = []
    for i, person in enumerate(ids):
        input_sha = hashlib.sha256(rows[i].tobytes()).hexdigest()
        for purpose in ("income", "occupation"):
            records.append({"purpose": purpose, "private_person_key": str(int(person)),
                            "input_sha256": input_sha, "view": views[i][purpose]})
    payload = {"format": "TeamBPrivateReleaseV1", "source_id": source,
               "model_sha256": model, "raw_mode": False,
               "purpose_model_sha256": {p: purpose_sha(state, p, names)
                                        for p in ("income", "occupation")},
               "schema_sha256": schema_sha(names),
               "temperatures": {"income": 1.0, "occupation": 1.0},
               "stochastic": False,
               "calibration_record_sha256": {"income": cal_sha,
                                                  "occupation": cal_sha},
               "records": records}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    path = tmp_path / "release.json"
    path.write_text(json.dumps({"payload": payload,
                                "payload_sha256": hashlib.sha256(canonical).hexdigest()}))
    return state, names, rows, ids, path, source, model, cal_sha


def test_complete_wire_and_protected_head_replay(tmp_path):
    state, names, x, ids, path, source, model, cal_sha = _fixture(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = load_saved_wire(path, ids, x, digest, source, model,
                           {"income": cal_sha, "occupation": cal_sha})
    replay_encoder(rows, x, ids, state, b"k" * 32, source, names)
    cal = {p: {"alpha": 1.0} for p in ("income", "occupation")}
    for view in ("income", "occupation", "pair"):
        rebuilt = rebuild_view(rows, view, head_state(state), cal)
        assert rebuilt["x"].shape[0] == 2
        assert rebuilt["x_defense_aware"].shape[1] > rebuilt["x"].shape[1]
        assert any("planted_flag" in name for name in rebuilt["components"])
    rows[0]["income"]["probabilities"] = [0.9, 0.1]
    with pytest.raises(ValueError, match="probability|decision"):
        rebuild_view(rows, "pair", head_state(state), cal)


def test_att_group_split_is_fixed_without_labels():
    fit, tune, tune_groups = ordered_split(np.arange(4061, dtype=np.int64))
    assert (len(fit), len(tune), len(tune_groups)) == (3046, 1015, 1015)
    assert not np.intersect1d(fit, tune).size
