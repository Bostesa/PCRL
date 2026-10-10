"""Data-free provider-to-recipient deployment check."""

import numpy as np
import pytest

from code_witness.bank_deploy import load_bank_service
from code_witness.test_bank_deploy import synthetic_bank
from results.pcrl_bank_anchored_release_v1.bank_anchored.r2_laws import (
    IdentityPolicy, wire_fixed_bank_and_actions,
)
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.channel import (
    GateWeights, JointPolicy,
)

from bank_anchored.r1_runtime import release_pair_from_raw


class _Identity:
    def __init__(self, pair):
        self.pair = pair

    def sample_public_pair(self, score0, score1, cell0, cell1, uniform):
        assert uniform is None
        return (int(self.pair.p1.cell_token[cell0]),
                int(self.pair.p2.cell_token[cell1]))


def test_raw_provider_identity_matches_existing_bank_adapter(synthetic_bank):
    s = synthetic_bank
    service = load_bank_service(
        source_root=s["source_root"], qpc_root=s["qpc_root"],
        table_path=s["table"], expected_table_sha256=s["table_sha"],
        expected_source_sha256=s["source_sha"],
        expected_schema_sha256=s["schema_hash"], ordered_names=s["names"],
        bank="W_TSTAR", seed=0)
    actual = release_pair_from_raw(service, _Identity(service.policy_pair),
                                   s["x"], s["names"], None)
    for purpose in (0, 1):
        expected = service.recipient_from_raw(purpose, s["x"], s["names"])
        assert set(actual[purpose]) == {"code", "prob", "decision"}
        assert all(np.array_equal(actual[purpose][key], expected[key])
                   for key in expected)


def test_raw_provider_refuses_reordered_schema(synthetic_bank):
    s = synthetic_bank
    service = load_bank_service(
        source_root=s["source_root"], qpc_root=s["qpc_root"],
        table_path=s["table"], expected_table_sha256=s["table_sha"],
        expected_source_sha256=s["source_sha"],
        expected_schema_sha256=s["schema_hash"], ordered_names=s["names"],
        bank="W_TSTAR", seed=0)
    with pytest.raises(ValueError, match="schema"):
        release_pair_from_raw(service, _Identity(service.policy_pair),
                              s["x"][:, ::-1], s["names"][::-1], None)


def test_r2_identity_policy_keeps_original_bank_token_bits(synthetic_bank):
    s = synthetic_bank
    service = load_bank_service(
        source_root=s["source_root"], qpc_root=s["qpc_root"],
        table_path=s["table"], expected_table_sha256=s["table_sha"],
        expected_source_sha256=s["source_sha"],
        expected_schema_sha256=s["schema_hash"], ordered_names=s["names"],
        bank="W_TSTAR", seed=0)
    bank_action = []
    for purpose, part in enumerate((service.policy_pair.p1,
                                    service.policy_pair.p2)):
        bank_action.append(wire_fixed_bank_and_actions(
            np.asarray(service.tables[purpose], dtype=np.float64),
            np.asarray(part.token_class, dtype=np.int64),
            np.asarray(part.cell_token, dtype=np.int64),
            np.asarray(part.fine.cell_class, dtype=np.int64)))
    base = JointPolicy(bank_action[0][0], bank_action[1][0],
                       bank_action[0][1], bank_action[1][1],
                       GateWeights.zeros(input_width=12))
    wire = release_pair_from_raw(service, IdentityPolicy(base),
                                 s["x"], s["names"], None)
    for purpose in (0, 1):
        expected = service.recipient_from_raw(purpose, s["x"], s["names"])
        assert all(np.array_equal(wire[purpose][key], expected[key])
                   for key in expected)
