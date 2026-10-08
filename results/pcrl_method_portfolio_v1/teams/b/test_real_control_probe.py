"""Synthetic checkpoint custody checks for postrelease real-control runner."""

from types import SimpleNamespace

import pytest
import torch

from bottleneck import model_fingerprint
from control_artifacts import file_sha256
from control_plan import stages_for_arm
from real_control_probe import _parent_attempt_matches, _restore_public_model
from test_attack_bridge import _model


@pytest.mark.parametrize("arm", ["B-JVIB", "B-LOCAL", "B-SEQ12", "B-SEQ21"])
def test_restore_final_public_model_from_hashed_checkpoint(tmp_path, arm):
    model = _model()
    attempt = tmp_path / ".private" / "screen"
    (attempt / "fit").mkdir(parents=True)
    if arm == "B-LOCAL":
        name, key, fmt = "B-LOCAL__assembled.pt", "assembled_local_pair", "TeamBAssembledLocalPairV1"
    else:
        final = stages_for_arm(arm, 0, 10)[-1]
        name, key, fmt = (final.stage_id.replace("/", "__") + ".pt",
                          final.stage_id, "TeamBStageCheckpointV1")
    checkpoint = attempt / "fit" / name
    torch.save({"format": fmt, "model": model.state_dict()}, checkpoint)
    # The active source files are checked by their exact byte hash; no private
    # fitting rows or role exports are needed for this restoration check.
    from pathlib import Path
    import real_control_probe as probe
    own = Path(probe.__file__).resolve().parent
    index = {
        "source_code_sha256": {
            "release_adapter": file_sha256(own / "bottleneck.py"),
            "attack_view_builder": file_sha256(own / "attack_bridge.py"),
            "wire_flattener": file_sha256(own / "attack_views.py"),
            "screen_driver": file_sha256(own / "screen_driver.py"),
        },
        "fit_artifact_sha256": {key: file_sha256(checkpoint)},
        "fit_model_sha256": model_fingerprint(model),
    }
    args = SimpleNamespace(arm=arm, seed=0, attempt=attempt)
    config = {"model_initialization_base_seed": 11, "hidden_dim": 64,
              "latent_dim": 16, "log_variance_bounds": [-8.0, 4.0],
              "hyperparameters": {"steps_per_encoder": 10}}
    restored = _restore_public_model(args, config, index, model.feature_names)
    assert model_fingerprint(restored) == model_fingerprint(model)
    index["fit_artifact_sha256"][key] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint bytes changed"):
        _restore_public_model(args, config, index, model.feature_names)


def test_parent_screen_unit_cannot_be_swapped_to_another_attempt(tmp_path):
    first = (tmp_path / ".private" / "first").resolve()
    second = (tmp_path / ".private" / "second").resolve()
    parent = {"spec": {"argv": ["python", "screen_driver.py",
                                "--private-attempt", str(first)]}}
    index = {"screen_unit_id": "B-SCREEN-B-JVIB-S0"}
    assert _parent_attempt_matches(index, parent,
                                   screen_unit_id="B-SCREEN-B-JVIB-S0", attempt=first)
    assert not _parent_attempt_matches(index, parent,
                                       screen_unit_id="B-SCREEN-B-JVIB-S0", attempt=second)
    assert not _parent_attempt_matches(index, parent,
                                       screen_unit_id="B-SCREEN-B-DVIB-S0", attempt=first)
    parent["spec"]["argv"][-1] = ".private/first"
    assert not _parent_attempt_matches(index, parent,
                                       screen_unit_id="B-SCREEN-B-JVIB-S0", attempt=first)
