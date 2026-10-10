"""Pure U0/Ucal complete-wire adapter; no protected role opened in tests."""

import hashlib
import json
import numpy as np
import pytest
from argparse import Namespace
from pathlib import Path

from results.pcrl_latent_and_sampled_methods_v1.r02.anchor_worker import (
    R01_SOURCE_MAP_SHA256, _verify_queue_admission, run,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.reference_anchor import (
    prepared_anchor_views, wire_batch_anchor,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.protocol import reader_plan, unit_fit_forecast


def _refs(n=5):
    row = np.arange(n, dtype=np.int64)
    p1 = np.tile([.8, .2], (n, 1))
    p2 = np.tile([.7, .1, .1, .05, .03, .02], (n, 1))
    return {"row_id": row, "unit": row.copy(), "U0_1": p1, "U0_2": p2,
            "Ucal_1": p1.copy(), "Ucal_2": p2.copy(),
            "d_1": np.zeros(n, dtype=np.int64), "d_2": np.zeros(n, dtype=np.int64)}


def test_anchor_exposes_only_actual_score_and_decision_wire():
    att, inner = _refs(), _refs()
    inner["row_id"] += 10
    inner["unit"] += 10
    batch = wire_batch_anchor(att, "U0", 1)
    assert set(batch.fields) == {"probabilities", "decision"}
    assert batch.code_fields == ()
    assert batch.fields["probabilities"].values.shape == (5, 2)
    views = prepared_anchor_views(att, inner, "U0")
    assert set(views) == {"r1", "r2", "pair"}
    assert views["pair"].att.x_code_bits is None
    assert views["pair"].att.cell_key is None
    assert "r1.probabilities" in views["pair"].att.components
    plan = reader_plan(family="D", code_bits=False)
    assert "CELL_A1" not in plan.core and "CODE_BIT_LR_C1" not in plan.core
    assert unit_fit_forecast(plan)["mandatory_fits"] == 24


def test_anchor_rejects_missing_fields_bad_decision_and_unaligned_role():
    att = _refs()
    bad = _refs()
    del bad["U0_1"]
    with pytest.raises((KeyError, ValueError)):
        wire_batch_anchor(bad, "U0", 1)
    bad = _refs()
    bad["d_1"] = np.ones(5, dtype=np.int64)
    with pytest.raises(ValueError, match="decision"):
        wire_batch_anchor(bad, "U0", 1)
    bad = _refs()
    bad["row_id"] = np.array([1, 0, 2, 3, 4], dtype=np.int64)
    with pytest.raises(ValueError, match="ordered"):
        prepared_anchor_views(att, bad, "Ucal")


def test_anchor_worker_refuses_direct_run_before_role_or_inner_access():
    with pytest.raises(RuntimeError, match="active queue slot"):
        run(Namespace(stage="d_followup", seed=0, attacker_seed=0,
                      anchor="U0", unit_id="D_U_ANCHOR_U0_S0"))
    with pytest.raises(PermissionError, match="unregistered"):
        run(Namespace(stage="d_followup", seed=0, attacker_seed=7,
                      anchor="U0", unit_id="D_ANCHOR_U0_S0"))
    with pytest.raises(PermissionError, match="unregistered"):
        run(Namespace(stage="d_followup", seed=0, attacker_seed=0,
                      anchor="U0", unit_id="D_ANCHOR_U0_S0"))


def test_anchor_worker_rejects_unregistered_or_unpinned_amendment_before_role_read(
        tmp_path, monkeypatch):
    amendment = tmp_path / "amendment.json"
    amendment.write_text(json.dumps({"schema": "pcrl-d-followup-amendment-v1",
                                     "status": "LOCALLY_SEALED_D_FOLLOWUP",
                                     "effective_seal_sha256": "a" * 64,
                                     "authorized_unit_ids": ["D_U_ANCHOR_U0_S0"],
                                     "anchor_worker_sha256": "0" * 64}))
    digest = hashlib.sha256(amendment.read_bytes()).hexdigest()
    (tmp_path / "state.json").write_text(json.dumps({"amendments": []}))
    monkeypatch.setenv("PCRL_QUEUE_ROOT", str(tmp_path))
    args = Namespace(amendment=str(amendment), amendment_sha256=digest,
                     seal_sha256="a" * 64, unit_id="D_U_ANCHOR_U0_S0",
                     role_manifest_sha256="b" * 64, seed=0)
    kw = {key: tmp_path / key for key in
          ("seal", "role_manifest", "source_map_path", "audit_path",
           "private_out", "public_out")}
    kw["lock"] = {"implementation_source_sha256": {}}
    with pytest.raises(PermissionError, match="latest registered"):
        _verify_queue_admission(args, {"spec_sha256": "c" * 64}, **kw)
    (tmp_path / "state.json").write_text(json.dumps({"amendments": [
        {"path": str(amendment), "sha256": digest, "stages": ["d_followup"]}]}))
    with pytest.raises(PermissionError, match="does not bind"):
        _verify_queue_admission(args, {"spec_sha256": "c" * 64}, **kw)
    # Advance through the amendment gate into the original-lock source-pin
    # branch. This catches missing helper arguments and inadvertent NameError.
    local = Path(__file__).parent
    four = [f"D_U_ANCHOR_{arm.upper()}_S{seed}"
            for seed in (0, 1) for arm in ("U0", "Ucal")]
    amendment.write_text(json.dumps({
        "schema": "pcrl-d-followup-amendment-v1",
        "status": "LOCALLY_SEALED_D_FOLLOWUP",
        "effective_seal_sha256": args.seal_sha256,
        "authorized_unit_ids": four,
        "role_manifest_sha256": args.role_manifest_sha256,
        "r01_source_map_sha256": R01_SOURCE_MAP_SHA256,
        "anchor_worker_sha256": hashlib.sha256((local / "anchor_worker.py").read_bytes()).hexdigest(),
        "reference_anchor_sha256": hashlib.sha256((local / "reference_anchor.py").read_bytes()).hexdigest(),
        "anchor_spec_helper_sha256": hashlib.sha256((local / "prepare_anchor_queue_units.py").read_bytes()).hexdigest(),
    }))
    args.amendment_sha256 = hashlib.sha256(amendment.read_bytes()).hexdigest()
    (tmp_path / "state.json").write_text(json.dumps({"amendments": [
        {"path": str(amendment), "sha256": args.amendment_sha256,
         "stages": ["d_followup"]}]}))
    with pytest.raises(PermissionError, match="original lock-pinned"):
        _verify_queue_admission(args, {"spec_sha256": "c" * 64}, **kw)
