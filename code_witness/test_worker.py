"""Data-free worker boundaries and private-release array contract."""

import numpy as np
import pytest

from code_witness.worker import release_payload, main


def test_release_payload_contains_only_alignment_and_two_complete_views():
    ids = np.array([2, 7], np.int64)
    groups = np.array([3, 8], np.int64)
    one = {"code": np.zeros((2, 10), np.uint8),
           "probabilities": np.array([[0.8, 0.2], [0.3, 0.7]]),
           "decision": np.array([0, 1], np.int64)}
    two = {"code": np.ones((2, 10), np.uint8),
           "probabilities": np.full((2, 6), 1 / 6),
           "decision": np.zeros(2, np.int64)}
    out = release_payload(ids, groups, one, two, bits=10)
    assert set(out) == {"row_id", "unit", "code0", "prob0", "decision0",
                        "code1", "prob1", "decision1"}
    assert np.array_equal(out["row_id"], ids)
    assert out["code1"].dtype == np.uint8
    with pytest.raises(ValueError):
        release_payload(ids, groups, one, {**two, "code": np.ones((2, 16), np.uint8)}, bits=10)


def test_worker_refuses_direct_run_before_private_reads(tmp_path):
    with pytest.raises(RuntimeError, match="queue"):
        main(["--stage", "bank", "--seed", "0", "--unit-id", "BANK_S0",
              "--role-manifest", str(tmp_path / "forbidden.npz"),
              "--role-manifest-sha256", "a" * 64,
              "--source-root", str(tmp_path / "source"),
              "--source-sha256", "b" * 64,
              "--qpc-root", str(tmp_path / "qpc"),
              "--private-out", str(tmp_path / "out"),
              "--receipt", str(tmp_path / "public.json"),
              "--public-table", str(tmp_path / "table.json"),
              "--public-table-sha256", "c" * 64])
    assert not (tmp_path / "out").exists()
