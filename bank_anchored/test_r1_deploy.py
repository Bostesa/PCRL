"""Data-free black-box deployment through a selected hash-bound policy."""

import hashlib
import os

import numpy as np
import pytest

from code_witness.bank_deploy import load_bank_service
from code_witness.test_bank_deploy import synthetic_bank
from dpc.deploy import Refused
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_laws import (
    BoundaryJointPolicy, IdentityPolicy, wire_fixed_bank_and_actions,
)
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_package import save_policy
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import (
    GateWeights, JointPolicy,
)

from bank_anchored.r1_deploy import main
from bank_anchored.r1_runtime import release_pair_from_raw


def _base(service):
    items = []
    for purpose, part in enumerate((service.policy_pair.p1,
                                    service.policy_pair.p2)):
        items.append(wire_fixed_bank_and_actions(
            np.asarray(service.tables[purpose], dtype=np.float64),
            np.asarray(part.token_class, dtype=np.int64),
            np.asarray(part.cell_token, dtype=np.int64),
            np.asarray(part.fine.cell_class, dtype=np.int64)))
    return JointPolicy(items[0][0], items[1][0], items[0][1], items[1][1],
                       GateWeights.zeros(input_width=12))


def _args(s, package, receipt, output, *, purpose=0):
    return ["--purpose", str(purpose), "--seed", "0", "--source-root", str(s["source_root"]),
            "--source-sha256", s["source_sha"], "--qpc-root", str(s["qpc_root"]),
            "--public-table", str(s["table"]), "--public-table-sha256", s["table_sha"],
            "--schema", str(s["schema"]), "--schema-sha256", s["schema_hash"],
            "--policy-dir", str(package), "--policy-receipt-sha256", receipt["receipt_sha256"],
            "--source-binding-sha256", "a" * 64, "--input", str(s["raw"]),
            "--output", str(output)]


def test_selected_identity_cli_emits_only_original_bank_wire(synthetic_bank, tmp_path):
    s = synthetic_bank
    service = load_bank_service(
        source_root=s["source_root"], qpc_root=s["qpc_root"],
        table_path=s["table"], expected_table_sha256=s["table_sha"],
        expected_source_sha256=s["source_sha"],
        expected_schema_sha256=s["schema_hash"], ordered_names=s["names"],
        bank="W_TSTAR", seed=0)
    base = _base(service)
    package = tmp_path / "private_policy"
    package.mkdir(mode=0o700)
    receipt = save_policy(package, IdentityPolicy(base), arm_id="B0",
                          source_seed=0, source_binding_sha256="a" * 64,
                          fit_receipt_sha256="b" * 64,
                          selected_start="identity", selected_update=0)
    output = tmp_path / "wire.npz"
    main(_args(s, package, receipt, output))
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == {"code", "prob", "decision"}
        expected = service.recipient_from_raw(0, s["x"], s["names"])
        assert all(np.array_equal(archive[key], expected[key]) for key in expected)
    with pytest.raises(ValueError, match="cannot read sampling coins"):
        main(_args(s, package, receipt, tmp_path / "identity_with_coin.npz") +
             ["--draws", str(tmp_path / "unused.npz"), "--draws-sha256", "0" * 64])


def test_selected_stochastic_cli_uses_private_pinned_draws(synthetic_bank, tmp_path):
    s = synthetic_bank
    service = load_bank_service(
        source_root=s["source_root"], qpc_root=s["qpc_root"],
        table_path=s["table"], expected_table_sha256=s["table_sha"],
        expected_source_sha256=s["source_sha"],
        expected_schema_sha256=s["schema_hash"], ordered_names=s["names"],
        bank="W_TSTAR", seed=0)
    policy = BoundaryJointPolicy(_base(service), .5)
    package = tmp_path / ".private" / "policy"
    package.mkdir(parents=True, mode=0o700)
    receipt = save_policy(package, policy, arm_id="J", source_seed=0,
                          source_binding_sha256="a" * 64,
                          fit_receipt_sha256="b" * 64,
                          selected_start="identity", selected_update=1)
    draws = tmp_path / ".private" / "draws.npz"
    uniforms = np.random.default_rng(12).random(len(s["x"])).astype(np.float64)
    with draws.open("wb") as stream:
        np.savez_compressed(stream, uniforms=uniforms)
    os.chmod(draws, 0o600)
    digest = hashlib.sha256(draws.read_bytes()).hexdigest()
    out = tmp_path / "stochastic.npz"
    args = _args(s, package, receipt, out, purpose=1)
    with pytest.raises(ValueError, match="pinned private draws"):
        main(args)
    with pytest.raises(PermissionError, match="pinned or owner-only"):
        main(args + ["--draws", str(draws), "--draws-sha256", "0" * 64])
    main(args + ["--draws", str(draws), "--draws-sha256", digest])
    expected = release_pair_from_raw(service, policy, s["x"], s["names"], uniforms)[1]
    with np.load(out, allow_pickle=False) as archive:
        assert set(archive.files) == {"code", "prob", "decision"}
        assert all(np.array_equal(archive[name], expected[name]) for name in expected)


def test_selected_cli_refuses_wrong_source_schema_and_hidden_predictor(
        synthetic_bank, tmp_path):
    s = synthetic_bank
    service = load_bank_service(
        source_root=s["source_root"], qpc_root=s["qpc_root"],
        table_path=s["table"], expected_table_sha256=s["table_sha"],
        expected_source_sha256=s["source_sha"],
        expected_schema_sha256=s["schema_hash"], ordered_names=s["names"],
        bank="W_TSTAR", seed=0)
    package = tmp_path / "private_policy"
    package.mkdir(mode=0o700)
    receipt = save_policy(package, IdentityPolicy(_base(service)), arm_id="B0",
                          source_seed=0, source_binding_sha256="a" * 64,
                          fit_receipt_sha256="b" * 64,
                          selected_start="identity", selected_update=0)
    out = tmp_path / "refused.npz"
    args = _args(s, package, receipt, out)
    changed = args.copy()
    changed[changed.index("--source-sha256") + 1] = "f" * 64
    with pytest.raises(ValueError):
        main(changed)
    changed = args.copy()
    changed[changed.index("--source-binding-sha256") + 1] = "f" * 64
    with pytest.raises(ValueError, match="source/seed binding"):
        main(changed)
    reordered = tmp_path / "reordered.npz"
    np.savez(reordered, X=s["x"][:, ::-1], feature_names=np.asarray(s["names"][::-1]))
    changed = args.copy()
    changed[changed.index("--input") + 1] = str(reordered)
    with pytest.raises(Refused):
        main(changed)
    hidden = tmp_path / "hidden.npz"
    np.savez(hidden, X=s["x"], feature_names=np.asarray(s["names"]),
             SEX=np.zeros(len(s["x"]), dtype=np.int64))
    changed[changed.index("--input") + 1] = str(hidden)
    with pytest.raises(Refused):
        main(changed)
    with pytest.raises(SystemExit):
        main(args + ["--export-teacher"])
    assert not out.exists()
