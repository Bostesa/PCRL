"""Data-free complete reader and real-control output accounting."""

from pathlib import Path

from results.pcrl_utility_reference_repair_v1.r2.attack_fit import (
    CORE, RAW, expected_outputs,
)


def test_six_reader_and_two_probe_artifact_inventory():
    output = Path("/tmp/.private/attacks/seed0/pair/income/ATTACK_FIT_INDEX.json")
    files = expected_outputs(output)
    assert len(files) == 16
    assert len(CORE) == 4 and len(RAW) == 2
    assert str(output.parent / "controls/planted_model.joblib") in files
    assert str(output.parent / "core/model__DA_MLP_64x32.joblib") in files
    assert str(output.parent / "ALL_READER_PREDICTIONS.npz") in files
