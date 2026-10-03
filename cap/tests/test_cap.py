"""Fixture tests for the useful-head comparison (synthetic; no private data)."""
import json

import numpy as np
from scipy.stats import norm

from cap import family as F


def test_family_counts_and_z():
    assert F.PRIMARY_SIZE == 19 and F.SECONDARY_SIZE == 33
    assert abs(F.Z_PRIMARY - norm.ppf(1 - 0.05 / 38)) < 1e-12
    assert abs(F.Z_SECONDARY - norm.ppf(1 - 0.05 / 66)) < 1e-12
    assert sum(e["kind"] == "utility" for e in F.PRIMARY) == 9


def test_logprob_head_offset_is_function_of_margin():
    from odx.surfaces import decompose
    rng = np.random.default_rng(0)
    d = rng.normal(0, 3, 1000)
    L = np.stack([-np.logaddexp(0, d), -np.logaddexp(0, -d)], 1)  # log p0, log p1 of an LR head
    D = decompose(L)
    assert np.allclose(D["margin"][:, 0], d)
    assert np.allclose(D["offset"][:, 0], -(np.logaddexp(0, d) + np.logaddexp(0, -d)) / 2)


def test_retention_linear_stat_matches_accuracy_formula():
    from odx.infer import acc_stat
    rng = np.random.default_rng(1)
    cF, cA, cK = (rng.random(500) < p for p in (0.8, 0.82, 0.75))
    W = np.ones((500, 1))
    v = acc_stat(cF - 0.8 * cA - 0.2 * cK)(W)[0]
    assert abs(v - (cF.mean() - 0.8 * cA.mean() - 0.2 * cK.mean())) < 1e-12


def test_resolution_chain(tmp_path, monkeypatch):
    import cap.infer as CI
    run, oar = tmp_path / "run", tmp_path / "oar"
    for d in ("run/units/bank", "run/units/plus", "run/units/comp", "oar/units/src"):
        (tmp_path / d).mkdir(parents=True)
    (run / "units/bank/record.json").write_text(json.dumps({"kind": "bank", "bank_selected": "plus"}))
    (run / "units/plus/record.json").write_text(json.dumps({"plus_selection": {"selected": "ignore_rep", "alias_source": "comp"}}))
    (run / "units/comp/record.json").write_text(json.dumps({}))
    monkeypatch.setattr(CI, "RUN", run)
    monkeypatch.setattr(CI, "OAR", oar)
    assert CI.resolve("bank") == run / "units/comp"
    assert CI.resolution_chain("bank") == ["bank", "plus", "comp"]
