import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.r02.d_inference import (
    D_ARMS, d_margin_names, d_margin_at,
)


def toy_seed():
    row = np.arange(4, dtype=np.int64)
    p2 = np.array([[.8, .2], [.2, .8], [.8, .2], [.2, .8]])
    p6 = np.eye(6)[[0, 1, 0, 1]] * .9 + .1 / 6
    release = {"row_id": row, "unit": row, "prob_1": p2, "prob_2": p6,
               "reader_predictions": {v: {"LR_C1": np.array([.2, .8, .3, .7])}
                                      for v in ("r1", "r2", "pair")}}
    truth = {"row_id": row, "unit": row, "income": np.array([0, 1, 0, 1]),
             "occupation": np.array([0, 1, 0, 1]), "sex": np.array([0, 1, 0, 1])}
    return {"truth": truth, "releases": {name: release for name in (*D_ARMS, "U0", "Ucal")}}


def test_exact_d_family_covers_all_567_finite_clauses():
    names = d_margin_names()
    assert len(names) == len(set(names)) == 567
    assert sum(":utility:" in name for name in names) == 252
    assert sum(":arm_pair:" in name for name in names) == 189
    assert sum(":u_anchor:" in name for name in names) == 126
    assert {n.split(":", 1)[0] for n in names} == {"s0", "s1", "mean"}


def test_d_family_reuses_same_people_and_scores_each_seed_and_mean():
    records = {0: toy_seed(), 1: toy_seed()}
    names, values = d_margin_at(records, np.arange(4))
    assert names == d_margin_names()
    assert values.shape == (567,)
    assert np.allclose(values, 0)
    records[1]["releases"]["D_K"] = {**records[1]["releases"]["D_K"],
                                     "row_id": np.array([1, 0, 2, 3], dtype=np.int64)}
    with pytest.raises(ValueError, match="ordered"):
        d_margin_at(records, np.arange(4))
