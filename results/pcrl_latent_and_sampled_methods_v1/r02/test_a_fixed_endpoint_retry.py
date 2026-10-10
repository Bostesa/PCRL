"""Data-free interpreter and immutable retry-identity regression."""

import subprocess
import json

import pytest

from results.pcrl_latent_and_sampled_methods_v1.r02.a_fixed_endpoint_retry import (
    CAMPAIGN, FAILED_RECEIPT_SHA256, UNIT_ID, PRIVATE_OUT, PUBLIC_JSON,
    SOURCE_NAMES_FIXED, WORKTREE, _runtime_source_hashes, _venv_python,
    verify_failed_endpoint_lineage, sha,
)


def test_retry_uses_project_virtualenv_without_resolving_its_python_symlink():
    python = _venv_python()
    assert python == WORKTREE.parents[1] / ".venv/bin/python"
    assert python.resolve() != python
    result = subprocess.run(
        [str(python), "-c", "import scipy, sys; print(sys.prefix, scipy.__version__)"],
        check=True, capture_output=True, text=True)
    assert str(WORKTREE.parents[1] / ".venv") in result.stdout


def test_retry_has_new_unit_outputs_and_source_identity():
    assert UNIT_ID == "R02-A-K6885-FIXED-ENDPOINT-001-R1"
    assert PRIVATE_OUT.name == "a_k6885_fixed_endpoint_r1"
    assert PUBLIC_JSON.name == "A_K6885_FIXED_INFERENCE_R1.json"
    assert "r02/a_fixed_endpoint_retry.py" in SOURCE_NAMES_FIXED


def test_retry_requires_preserved_failed_pre_role_unit_receipt(tmp_path):
    uid = "R02-A-K6885-FIXED-ENDPOINT-001"
    receipt = tmp_path / "RECEIPT.json"
    receipt.write_text(json.dumps({"unit_id": uid, "status": "failed",
                                   "returncode": 1, "output_sha256": {}}))
    import hashlib
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    state = {"units": {uid: {"status": "failed", "returncode": 1,
                              "output_sha256": {}}}}
    verify_failed_endpoint_lineage(state, receipt, digest)
    receipt.write_text("{}")
    with pytest.raises(PermissionError, match="failed endpoint"):
        verify_failed_endpoint_lineage(state, receipt, digest)


def test_inert_retry_proposal_pins_new_source_and_safe_runtime():
    proposal = json.loads((CAMPAIGN / "r02/A_FIXED_ENDPOINT_R2_ANALYSIS_PROPOSAL.json").read_text())
    assert proposal["status"] == "PREFIT_DRAFT_NOT_EFFECTIVE"
    assert proposal["authorized_unit_ids"] == [UNIT_ID]
    assert proposal["original_failed_endpoint_queue_receipt_sha256"] == FAILED_RECEIPT_SHA256
    assert proposal["source_sha256"]["r02/a_fixed_endpoint_retry.py"] == sha(
        CAMPAIGN / "r02/a_fixed_endpoint_retry.py")
    assert proposal["source_sha256"]["r03/A_FIXED_ENDPOINT_TECHNICAL_FAILURE_001.json"] == sha(
        CAMPAIGN / "r03/A_FIXED_ENDPOINT_TECHNICAL_FAILURE_001.json")
    assert proposal["runtime_source_sha256"] == _runtime_source_hashes()
    assert proposal["interpreter_argv0"] == "PROJECT_ROOT/.venv/bin/python"
    assert not any("/Users/" in name for name in proposal["runtime_source_sha256"])
