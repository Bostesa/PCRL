"""Metadata-only regression for all fixed A endpoint reconstruction routes."""

import json

import pytest

from results.pcrl_latent_and_sampled_methods_v1.r02.a_endpoint_driver_retry import (
    _argv_options, reference_producer_path,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.a_fixed_endpoint_retry_v3 import (
    CAMPAIGN, FAILED_R1_RECEIPT_SHA256, QUEUE_STATE, R03_CLOSEOUT,
    R04_FIT_AUDIT, UNIT_ID, SOURCE_NAMES_FIXED, _analysis, classify_fixed_scope,
    preflight_fixed_replay_metadata,
    verify_failed_r1_lineage, sha,
)

PRODUCER_SHA = "526f509f8077f2ee1c412f6d290df939966a6b81f1ae5a87e9639d4bd317271f"
CORRECTION_SHA = "5cfcf570e27d623625db30f2be61727934f1290d7cef9fdb33a8a38fa146d613"


def _actual_metadata():
    state = json.loads(QUEUE_STATE.read_text())
    producer = json.loads((CAMPAIGN /
        "AMENDMENT_20261010_A_PRODUCER_STAGE_FIT_RETRY_001.json").read_text())
    correction = json.loads((CAMPAIGN /
        "AMENDMENT_20261010_A_FIXED_ATTACK_CORRECTION_R1.json").read_text())
    cells, fixed, missing, reports = classify_fixed_scope(
        producer["unit_plan"], state["units"],
        json.loads(R03_CLOSEOUT.read_text()), json.loads(R04_FIT_AUDIT.read_text()),
        PRODUCER_SHA, QUEUE_STATE.parent, correction, CORRECTION_SHA)
    return state, producer, cells, fixed, missing, reports


def test_all_eighteen_saved_attack_metadata_routes_preflight_without_role_arrays():
    state, producer, cells, fixed, missing, reports = _actual_metadata()
    original = {key: cells[key] for key in fixed}
    preflight_fixed_replay_metadata(fixed, original, state["units"],
                                    producer["unit_plan"], PRODUCER_SHA,
                                    CORRECTION_SHA)
    assert (len(fixed), len(missing), len(reports)) == (18, 40, 16)
    assert sum(uid == original[key] for key, uid in fixed.items()) == 6
    assert sum(uid != original[key] for key, uid in fixed.items()) == 12


def test_both_producer_option_names_are_bound_and_wrong_schema_rejected():
    state, _, cells, fixed, _, _ = _actual_metadata()
    for key, uid in fixed.items():
        spec = state["units"][uid]["spec"]
        options = _argv_options(spec["argv"])
        assert sha(reference_producer_path(spec, options)) == PRODUCER_SHA
        wrong = dict(options)
        if uid == cells[key]:
            wrong.pop("producer-amendment-path")
        else:
            wrong.pop("a-amendment-path")
        with pytest.raises(PermissionError, match="producer option"):
            reference_producer_path(spec, wrong)


def test_failed_r1_lineage_records_authorized_label_read_and_no_output():
    state = json.loads(QUEUE_STATE.read_text())
    receipt = (QUEUE_STATE.parent / "units" /
               "R02-A-K6885-FIXED-ENDPOINT-001-R1" / "RECEIPT.json")
    analysis = {
        "failed_r1_queue_receipt_sha256": FAILED_R1_RECEIPT_SHA256,
        "failed_r1_public_receipt_sha256": sha(CAMPAIGN /
            "r03/A_FIXED_ENDPOINT_R1_TECHNICAL_FAILURE.json"),
        "failed_r1_cause_correction_sha256": sha(CAMPAIGN /
            "r03/A_FIXED_ENDPOINT_R1_FAILURE_CAUSE_CORRECTION.json"),
        "previous_retry_validation_amendment_sha256":
            "2ded16e28edcdea55addc0f45f6a096710a67fd2fe23b6e393e43f7ed724176d",
    }
    verify_failed_r1_lineage(state, receipt, analysis)
    analysis["failed_r1_queue_receipt_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="failed R1"):
        verify_failed_r1_lineage(state, receipt, analysis)


def test_inert_r2_proposal_pins_all_reconstruction_sources():
    proposal = json.loads((CAMPAIGN /
        "r02/A_FIXED_ENDPOINT_R3_RECONSTRUCTION_CORRECTION_PROPOSAL.json").read_text())
    assert proposal["status"] == "PREFIT_DRAFT_NOT_EFFECTIVE"
    assert proposal["authorized_unit_ids"] == [UNIT_ID]
    assert proposal["previous_retry_validation_amendment_sha256"] == sha(
        CAMPAIGN / "AMENDMENT_20261010_A_FIXED_ENDPOINT_R2_VALIDATION_CORRECTION.json")
    assert proposal["fixed_family_clause_count"] == 6885
    assert proposal["fixed_attack_instances"] == 18
    for name in SOURCE_NAMES_FIXED:
        assert proposal["source_sha256"][name] == sha(CAMPAIGN / name)
    assert len(proposal["source_sha256"]) == 50


def test_successor_status_is_exact_without_role_access(tmp_path):
    document = json.loads((CAMPAIGN /
        "r02/A_FIXED_ENDPOINT_R3_RECONSTRUCTION_CORRECTION_PROPOSAL.json").read_text())
    path = tmp_path / "successor.json"

    def check():
        path.write_text(json.dumps(document))
        digest = sha(path)
        state = {"amendments": [{"path": str(path), "sha256": digest,
                                  "stages": ["verification"]}]}
        return _analysis(state, path, digest, PRODUCER_SHA, CORRECTION_SHA)

    with pytest.raises(PermissionError, match="analysis amendment"):
        check()
    document["status"] = "LOCALLY_SEALED_A_FIXED_ENDPOINT_RECONSTRUCTION_CORRECTION"
    assert check()["authorized_unit_ids"] == [UNIT_ID]
    document["previous_retry_validation_amendment_sha256"] = "0" * 64
    with pytest.raises(PermissionError, match="analysis amendment"):
        check()
