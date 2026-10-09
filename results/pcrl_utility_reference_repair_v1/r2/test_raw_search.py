"""Synthetic finite-bank search inventory without training a real reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from results.pcrl_utility_reference_repair_v1.r2 import raw_search
from results.pcrl_utility_reference_repair_v1.r2.frozen_bank import load_frozen_bank


ROOT = Path(__file__).resolve().parents[3]
MASK_BANK = ROOT / "results/pcrl_capacity_matched_feature_selection_v1/MASK_BANK.json"
MASK_SHA = "a52750191a1e16051367cade59537bc65ed797d69efcb375e0b1cc2e9364722c"


def test_search_fits_each_unique_local_and_union_once(monkeypatch):
    bank = load_frozen_bank(MASK_BANK, expected_sha256=MASK_SHA)
    calls = []

    def fake_reader(x, y, fit, val, columns, seed):
        calls.append(columns)
        return {"selected_auc": 0.7,
                "readers": {name: {"raw_auc": 0.7, "oriented_auc": 0.7}
                            for name in raw_search.READER_IDS},
                "fit_count": 2}

    monkeypatch.setattr(raw_search, "_one_feature_set", fake_reader)
    x = np.zeros((120, 83), dtype=np.float32)
    sex = np.arange(120, dtype=np.int64) % 2
    fit = np.arange(80, dtype=np.int64)
    val = np.arange(80, 120, dtype=np.int64)
    local, unions, receipt = raw_search.fit_fixed_raw_search(
        bank, x, sex, fit, val, 0)
    assert len(calls) == 11
    assert len(local) == 5 and len(unions) == 6
    assert receipt["fitted_reader_count"] == 22
    with pytest.raises(ValueError, match="split or arrays"):
        raw_search.fit_fixed_raw_search(
            bank, x, sex, np.r_[fit[:-1], fit[0]], val, 0)
