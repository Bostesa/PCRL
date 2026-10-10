"""Full-wire bypass and code-only reconstruction fixtures."""

import numpy as np
import pytest

from independent_release import verify_a_sampled_wire, verify_d_hard_code_wire


def test_d_code_only_head_and_extra_field_rejection():
    table = np.repeat(np.array([[.25, .75]]), 1024, axis=0)
    table[1] = [.8, .2]
    code = np.zeros((2, 10), dtype=np.uint8)
    code[1, 0] = 1
    wire = {"code": code, "probabilities": table[[0, 1]],
            "decision": np.array([1, 0]), "metadata": {"model_version": "D1"}}
    report = verify_d_hard_code_wire(wire, table, bit_order="little")
    assert report["occupied_states"] == 2
    with pytest.raises(ValueError):
        verify_d_hard_code_wire({**wire, "teacher_score": np.array([.2, .8])}, table,
                                bit_order="little")
    with pytest.raises(ValueError):
        verify_d_hard_code_wire({**wire, "probabilities": table[[0, 0]]}, table,
                                bit_order="little")


def test_a_global_id_and_hidden_action_rejection():
    bank = np.array([[.1, .9], [.8, .2]])
    wire = {"code_id": np.array([0, 1, 0]), "probabilities": bank[[0, 1, 0]],
            "decision": np.array([1, 0, 1]), "metadata": {"model_version": "A1"}}
    assert verify_a_sampled_wire(wire, bank)["occupied_outputs"] == 2
    with pytest.raises(ValueError):
        verify_a_sampled_wire({**wire, "action_index": np.array([0, 0, 1])}, bank)
    with pytest.raises(ValueError):
        verify_a_sampled_wire(wire, bank[[0, 0]])
