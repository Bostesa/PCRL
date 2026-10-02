"""Owner-added (2026-10-02): the primary endpoint R02 is produced by fit-attackers, only under authorization,
with the fit-mean denominator, and the score-mean variant is reported separately as a sensitivity."""
import numpy as np
import pytest

from stored_model_eval.guards import FitAuthorization, ScientificFitRefused
from stored_model_eval.pipeline import closed_form_linear, fit_attackers


def _arrays(n=1200, d=8, seed=0, signal=1.0, synthetic=True):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    H = rng.normal(size=(n, d))
    H[:, 0] += signal * (y - 0.5)
    roles = np.array(["attacker_fit"] * 600 + ["attacker_val"] * 200 + ["assessment"] * 400)
    return {"representations": H, "outputs": None, "labels": y, "units": np.arange(n), "roles": roles,
            "record_keys": None, "row_ids": np.arange(n), "_synthetic": synthetic}


def _cfg():
    from stored_model_eval.config import load_protocol
    return load_protocol(None)


def test_closed_form_fitmean_vs_scoremean():
    a = _arrays()
    cfg = _cfg()
    fi = np.where(a["roles"] == "attacker_fit")[0]
    ei = np.where(a["roles"] == "assessment")[0]
    cf = closed_form_linear(a["representations"], a["labels"], fi, ei, cfg)
    assert cf["r02_heldout_r2_fitmean"] > 0.05
    assert cf["r02_heldout_r2_scoremean_sensitivity"] is not None
    assert cf["r02_heldout_r2_fitmean"] != cf["r02_heldout_r2_scoremean_sensitivity"]
    # null signal: held-out R2 may be negative and must not be clamped
    n = _arrays(signal=0.0, seed=3)
    cfn = closed_form_linear(n["representations"], n["labels"], fi, ei, cfg)
    assert cfn["r02_heldout_r2_fitmean"] < 0.02


def test_fit_attackers_refuses_real_inputs_without_flag():
    a = _arrays(synthetic=False)
    with pytest.raises(ScientificFitRefused):
        fit_attackers(a, _cfg(), FitAuthorization(), attackers=("linear",), surfaces=["rep"])
