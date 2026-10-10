"""Worker boundary and private wire-export checks without any Adult role read."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from code_repair.run_conditional import (run, write_release_npz, scientific_arm_id,
                                         _calibration_module)
from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_file


def test_public_arm_ids_include_selected_capacity():
    assert scientific_arm_id("T+", 10) == "TPLUS_B10"
    assert scientific_arm_id("J", 16) == "J_B16"
    with pytest.raises(ValueError):
        scientific_arm_id("J", 12)


def test_pinned_calibration_loads_its_sibling_guard():
    root = Path(__file__).resolve().parents[2]
    base = "results/pcrl_method_portfolio_v1/shared/"
    stage = {"source_files_sha256": {
        base + filename: sha_file(root / (base + filename))
        for filename in ("common_eval.py", "safe_data_guard.py")}}
    assert callable(_calibration_module(stage, root).fit_global_temperature)


def test_worker_rejects_missing_queue_before_opening_any_supplied_file(tmp_path):
    args = Namespace(unit_id="R3-SYNTHETIC-NO-QUEUE", seed=0,
                     stage_lock=str(tmp_path / "no-stage-lock.json"),
                     stage_lock_sha256="0" * 64,
                     gate_receipt=str(tmp_path / "no-gate.json"),
                     gate_receipt_sha256="0" * 64,
                     role_manifest=str(tmp_path / "no-role.json"),
                     source_model=str(tmp_path / "no-source.pt"),
                     t0_manifest=str(tmp_path / "no-t0.json"),
                     private_output=str(tmp_path / "no-output"),
                     public_receipt=str(tmp_path / "no-receipt.json"))
    with pytest.raises(RuntimeError, match="active queue slot environment missing"):
        run(args)


def test_private_release_export_has_exact_hard_wire_fields(tmp_path):
    n, bits = 5, 16
    first = {"code": np.zeros((n, bits), dtype=np.uint8),
             "probabilities": np.tile([0.2, 0.8], (n, 1)),
             "decision": np.ones(n, dtype=np.int64),
             "metadata": {"bits": bits, "purpose": 0}}
    second = {"code": np.ones((n, bits), dtype=np.uint8),
              "probabilities": np.tile([1 / 6] * 6, (n, 1)),
              "decision": np.zeros(n, dtype=np.int64),
              "metadata": {"bits": bits, "purpose": 1}}
    path = tmp_path / "release.npz"
    digest = write_release_npz(path, np.arange(n, dtype=np.int64),
                               np.arange(n, dtype=np.int64), first, second, bits)
    assert len(digest) == 64
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"row_id", "unit", "code0", "code1",
                                      "prob0", "prob1", "decision0", "decision1"}
        assert archive["code0"].dtype == np.uint8
        assert archive["prob1"].shape == (n, 6)
    with pytest.raises(FileExistsError):
        write_release_npz(path, np.arange(n, dtype=np.int64),
                          np.arange(n, dtype=np.int64), first, second, bits)
