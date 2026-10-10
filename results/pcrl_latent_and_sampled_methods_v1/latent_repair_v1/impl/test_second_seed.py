"""Data-free dated-amendment gate for the second paired seed."""

import json

import pytest

from .run_second_seed import _amendment_guard
from .service import sha256_file
from . import run_second_seed


def test_second_seed_requires_latest_queue_bound_worker_amendment(tmp_path):
    amendment = tmp_path / "amendment.json"
    payload = {
        "schema": "pcrl-d-followup-amendment-v1",
        "status": "LOCALLY_SEALED_D_FOLLOWUP",
        "effective_seal_sha256": "a" * 64,
        "seed1_tdk_worker_sha256": sha256_file(run_second_seed.__file__),
        "source_seed1_sha256": "b" * 64,
        "role_manifest_sha256": "c" * 64,
    }
    amendment.write_text(json.dumps(payload))
    digest = sha256_file(amendment)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "amendments": [{"path": str(amendment), "sha256": digest,
                        "stages": ["d_followup"]}],
        "units": {"S1": {"spec": {"amendment_sha256": digest}}},
    }))
    _amendment_guard(state, "S1", amendment, digest, "a" * 64, "c" * 64, "b" * 64)
    payload["source_seed1_sha256"] = "d" * 64
    amendment.write_text(json.dumps(payload))
    with pytest.raises(PermissionError):
        _amendment_guard(state, "S1", amendment, digest, "a" * 64, "c" * 64, "b" * 64)
