"""F attacks see the complete continuous public wire with no code specialist."""

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.latent_repair_v1.controls.f_adapter import (
    prepared_f_views,
    wire_batch_f,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.protocol import reader_plan


def _release(purpose, n):
    k = (2, 6)[purpose]
    return {
        "features": np.arange(n * 16, dtype=np.float32).reshape(n, 16) / 10 + purpose,
        "probabilities": np.full((n, k), 1 / k, dtype=np.float64),
        "decision": np.zeros(n, dtype=np.int64),
        "metadata": {"schema": "latent-unchanged-v1", "purpose": purpose,
                     "latent_dimensions": 16, "version": "F-s0-v1",
                     "source_sha256": "a" * 64},
    }


def test_f_complete_view_has_full_features_and_no_code_cell():
    row = np.arange(5, dtype=np.int64)
    unit = row.copy()
    release = _release(0, 5)
    batch = wire_batch_f(row, unit, release)
    assert batch.code_fields == ()
    assert batch.fields["features"].kind == "numeric"
    assert batch.fields["features"].values.shape == (5, 16)
    assert set(batch.fields) == {"features", "probabilities", "decision",
                                 *(f"metadata.{k}" for k in release["metadata"])}
    prepared = prepared_f_views(row, unit, _release(0, 5), _release(1, 5),
                                row + 10, unit + 10, _release(0, 5), _release(1, 5))
    assert set(prepared) == {"r1", "r2", "pair"}
    assert prepared["pair"].att.x_code_bits is None
    assert prepared["pair"].att.cell_key is None
    assert "r1.features" in prepared["pair"].att.components
    assert "r2.features" in prepared["pair"].att.components
    assert "CELL_A1" not in reader_plan(family="D", code_bits=False).core


def test_f_rejects_changed_metadata_and_truncated_feature():
    row = np.arange(5, dtype=np.int64)
    good = _release(0, 5)
    bad = _release(0, 5)
    bad["features"] = bad["features"][:, :10]
    with pytest.raises(ValueError):
        wire_batch_f(row, row, bad)
    inner = _release(0, 5)
    inner["metadata"]["version"] = "changed"
    with pytest.raises(ValueError):
        prepared_f_views(row, row, good, _release(1, 5),
                         row + 10, row + 10, inner, _release(1, 5))
