import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r02.finite_family import (
    finite_margin_names, registered_margin_at,
)


def record():
    row = np.arange(4, dtype=np.int64)
    unit = row.copy()
    p2 = np.array([[.8, .2], [.2, .8], [.8, .2], [.2, .8]])
    p6 = np.eye(6)[[0, 1, 0, 1]] * .9 + .1 / 6
    release = {"row_id": row, "unit": unit, "prob_1": p2, "prob_2": p6,
               "reader_predictions": {v: {"LR_C1": np.array([.2, .8, .3, .7])}
                                      for v in ("r1", "r2", "pair")}}
    truth = {"row_id": row, "unit": unit, "income": np.array([0, 1, 0, 1]),
             "occupation": np.array([0, 1, 0, 1]), "sex": np.array([0, 1, 0, 1])}
    refs = {"row_id": row, "unit": unit,
            "U0_1": p2, "Ucal_1": p2, "U0_2": p6, "Ucal_2": p6}
    return {0: {"truth": truth, "candidate": release,
                "controls": {"T": release}, "references": refs}}


def test_finite_family_enumerates_all_candidate_reference_and_view_clauses():
    registry = {"D_K": ("T", "L", "S12", "S21", "C", "F"),
                "A_JOINT_STOCH": ("TASK_STOCH",)}
    names = finite_margin_names(registry)
    assert len(names) == 30 + 15
    assert names[0] == "D_K:income:U0:ACC"
    assert names[-1] == "A_JOINT_STOCH:TASK_STOCH:pair:BENEFIT"


def test_registered_margin_uses_same_positions_and_refuses_missing_arm():
    seed_records = record()
    registry = {"D_K": ("T",)}
    names, values = registered_margin_at(registry, {"D_K": seed_records}, np.arange(4))
    assert len(names) == len(values) == 15
    assert np.allclose(values[:12], [.01, .01, .005] * 4)
    with pytest.raises(ValueError, match="registry"):
        registered_margin_at({"D_K": ("T",), "D_ALT": ("T",)},
                             {"D_K": seed_records}, np.arange(4))
