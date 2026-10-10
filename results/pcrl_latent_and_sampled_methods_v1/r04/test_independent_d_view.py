"""Complete-view public-wire reconstruction on an unseen-code fixture."""

import numpy as np

from independent_d_view import build_d_view


def _wire(ids, purpose):
    bits = np.asarray([[(number >> bit) & 1 for bit in range(10)] for number in ids], dtype=np.uint8)
    p = np.tile([.7, .3], (len(ids), 1))
    return {"code": bits, "probabilities": p,
            "decision": np.zeros(len(ids), dtype=np.int64),
            "metadata": {"purpose": purpose, "version": "fixture"}}


def test_unknown_code_has_explicit_column_and_pair_bits_are_public():
    att1, att2 = _wire([0, 1], 0), _wire([1, 0], 1)
    inner1, inner2 = _wire([2], 0), _wire([1], 1)
    r1_att, r1_inner = build_d_view(att1, att2, inner1, inner2, "r1")
    assert r1_att["x"].shape[0] == 2
    assert r1_inner["x"][0, 12] == 1  # ten raw bits, two ATT IDs, then unknown
    assert r1_inner["cell_key"].tolist() == ['["2"]']
    pair_att, pair_inner = build_d_view(att1, att2, inner1, inner2, "pair")
    assert pair_inner["cell_key"].tolist() == ['["2","1"]']
    assert pair_att["x_defense_aware"].shape[1] == pair_att["x"].shape[1] + 4 + 100
    assert pair_inner["x_code_bits"].shape == (1, 20)
