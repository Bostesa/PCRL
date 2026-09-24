"""Public synthetic fit, one-token wire, and exact 17-token audit smoke."""
import json

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import demo


def test_synthetic_fit_release_exact_audit_and_immutable_resume(tmp_path):
    output = tmp_path / "private" / "synthetic_demo"
    first = demo.run_demo(output)
    assert first["schema"] == "pcrl-adaptive-public-synthetic-demo-v1"
    assert first["fit_A_status"] == "COMPLETE"
    assert first["fit_B_status"] in (
        "SUPPORT_LIMITED_ALIAS_A", "NO_ACCEPTED_SPLIT_ALIAS_A", "COMPLETE")
    assert first["wire_fields"] == ["h_a", "token"]
    assert first["service_byte_equal"] is True
    assert first["same_token_within_session"] is True
    assert first["same_token_after_restart"] is True
    assert first["changed_input_rejected"] is True
    assert first["exact_enumeration_max_abs_difference"] < 1e-12
    assert first["token_count"] == 17
    assert first["n_original_check_people"] > 0
    assert first["independent_acs_claim"] is False
    assert first["outer_assessment_accessed"] is False
    assert set(first["exact_task_loss_nats"]) == {"U", "PWGTP", "balanced"}
    assert all(np.isfinite(value) for value in first["exact_task_loss_nats"].values())
    persisted = json.loads((output / "SYNTHETIC_DEMO.json").read_text())
    assert persisted == first
    assert "person_ids" not in persisted and "release_key" not in persisted
    before = (output / "SYNTHETIC_DEMO.json").read_bytes()
    second = demo.run_demo(output)
    assert second == first
    assert (output / "SYNTHETIC_DEMO.json").read_bytes() == before


def test_synthetic_demo_cli_and_private_output_guard(tmp_path, capsys):
    output = tmp_path / "private" / "cli_demo"
    result = demo.main(["--output-dir", str(output)])
    assert result["schema"] == "pcrl-adaptive-public-synthetic-demo-v1"
    printed = json.loads(capsys.readouterr().out)
    assert printed["service_byte_equal"] is True
    with pytest.raises(ValueError, match="private"):
        demo.run_demo(tmp_path / "public-demo")
