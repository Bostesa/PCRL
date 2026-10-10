"""Data-free complete-wire ATT vocabulary and unknown-code replay."""

from __future__ import annotations

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r04.independent_a_law import IndependentAPolicy
from results.pcrl_latent_and_sampled_methods_v1.r04.independent_a_views import (
    compare_frozen_view_arrays, encode_sampled_views)


def test_att_vocabulary_unknown_inner_code_and_pair_bits():
    bank1 = np.array([[.8, .2], [.9, .1]], dtype=np.float64)
    bank2 = np.array([[.7, .3, 0, 0, 0, 0], [.8, .2, 0, 0, 0, 0]],
                     dtype=np.float64)
    policy = IndependentAPolicy(
        {"release_id": "toy", "source_binding_sha256": "a" * 64,
         "mechanism_sha256": "b" * 64,
         "bank_sha256_by_recipient": {"r1": "c" * 64, "r2": "d" * 64}},
        {"bank1_probability": bank1, "bank2_probability": bank2,
         "bank1_decision": np.zeros(2, np.int64),
         "bank2_decision": np.zeros(2, np.int64)}, ((0, 1),), ((0, 1),))
    att0, att1 = np.array([0, 0], np.int64), np.array([0, 0], np.int64)
    inner0, inner1 = np.array([1], np.int64), np.array([1], np.int64)
    views = encode_sampled_views(policy, att_code1=att0, att_code2=att1,
                                 inner_code1=inner0, inner_code2=inner1,
                                 q_table=None)
    r1 = views["r1"]
    columns = {name: (start, end) for name, start, end in
               r1["inner"]["component_columns"]}
    start, end = columns["code_id"]
    assert np.array_equal(r1["inner"]["x"][0, start:end], [0, 1])
    assert np.array_equal(r1["inner"]["x_code_bits"], [[1]])
    assert views["pair"]["inner"]["cell_key"].tolist() == ['["1","1"]']
    saved = {role: {name: r1[role][name] for name in
                    ("x", "x_metadata", "x_code_bits", "cell_key")}
             for role in ("att", "inner")}
    receipt = {"view_schema": {key: r1["att"][key] for key in
                               ("components", "wire_components",
                                "component_columns", "wire_constants")}}
    assert compare_frozen_view_arrays(r1, saved, receipt)["arrays_checked"] == 8
    saved["inner"]["x"] = saved["inner"]["x"].copy()
    saved["inner"]["x"][0, start] = 1
    with pytest.raises(ValueError, match="complete wire"):
        compare_frozen_view_arrays(r1, saved, receipt)
