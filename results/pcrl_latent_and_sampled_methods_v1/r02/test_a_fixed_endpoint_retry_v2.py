"""Data-free validation amendment regression for the versioned A endpoint retry."""

import json

import pytest

from results.pcrl_latent_and_sampled_methods_v1.r02.a_fixed_endpoint_retry_v2 import (
    CAMPAIGN, SOURCE_NAMES_FIXED, UNIT_ID, _analysis, sha,
)


def test_retry_validation_status_and_predecessor_are_exact(tmp_path):
    previous = CAMPAIGN / "AMENDMENT_20261010_A_FIXED_ENDPOINT_R2_RETRY.json"
    document = json.loads(previous.read_text())
    path = tmp_path / "successor.json"

    def check():
        path.write_text(json.dumps(document))
        digest = sha(path)
        state = {"amendments": [{"path": str(path), "sha256": digest,
                                  "stages": ["verification"]}]}
        return _analysis(state, path, digest,
                         document["producer_amendment_sha256"],
                         document["correction_amendment_sha256"])

    with pytest.raises(PermissionError, match="analysis amendment"):
        check()  # The first sealed retry's status cannot authorize new code.
    document["status"] = "LOCALLY_SEALED_A_FIXED_ENDPOINT_RETRY_VALIDATION"
    document["previous_retry_amendment_sha256"] = sha(previous)
    assert check()["authorized_unit_ids"] == [UNIT_ID]
    document["previous_retry_amendment_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="analysis amendment"):
        check()


def test_inert_successor_pins_versioned_worker_and_previous_seal():
    path = CAMPAIGN / "r02/A_FIXED_ENDPOINT_R2_VALIDATION_CORRECTION_PROPOSAL.json"
    proposal = json.loads(path.read_text())
    assert proposal["status"] == "PREFIT_DRAFT_NOT_EFFECTIVE"
    assert proposal["authorized_unit_ids"] == [UNIT_ID]
    assert proposal["previous_retry_amendment_sha256"] == sha(
        CAMPAIGN / "AMENDMENT_20261010_A_FIXED_ENDPOINT_R2_RETRY.json")
    assert proposal["source_sha256"]["r02/a_fixed_endpoint_retry_v2.py"] == sha(
        CAMPAIGN / "r02/a_fixed_endpoint_retry_v2.py")
    assert "r02/a_fixed_endpoint_retry.py" in SOURCE_NAMES_FIXED
    assert "AMENDMENT_20261010_A_FIXED_ENDPOINT_R2_RETRY.json" in SOURCE_NAMES_FIXED
