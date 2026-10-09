import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from results.pcrl_utility_reference_repair_v1.r3.source_reference import (
    SourceNetwork, apply_inverse_temperature, file_sha,
    fit_inverse_temperature, forward_u0, restore_source,
)


def test_independent_u_forward_and_hashed_restore(tmp_path):
    rng = np.random.default_rng(903)
    x = rng.normal(size=(30, 83)).astype(np.float32)
    y2 = (np.arange(30) % 2).astype(np.int64)
    y6 = (np.arange(30) % 6).astype(np.int64)
    torch.manual_seed(7)
    model = SourceNetwork().eval()
    with torch.no_grad():
        z = [model.enc[i](torch.from_numpy(x)).double().numpy()
             for i in range(2)]
    heads = tuple(make_pipeline(StandardScaler(), LogisticRegression(
        C=0.1, max_iter=3000)).fit(z[i], y)
                  for i, y in enumerate((y2, y6)))
    expected = tuple(head.predict_proba(z[i]) for i, head in enumerate(heads))
    actual = forward_u0(model, heads, x)
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
    diagnostic_chunked = forward_u0(model, heads, x, batch_size=7)
    assert all(np.max(np.abs(a - b)) < 1e-6
               for a, b in zip(diagnostic_chunked, expected))

    import joblib
    model_path = tmp_path / "model.pt"
    head_paths = (tmp_path / "head_0.joblib", tmp_path / "head_1.joblib")
    torch.save(model.state_dict(), model_path)
    for path, head in zip(head_paths, heads):
        joblib.dump(head, path)
    hashes = tuple(file_sha(path) for path in (model_path, *head_paths))
    restored, restored_heads = restore_source(model_path, head_paths, hashes)
    replay = forward_u0(restored, restored_heads, x)
    assert all(np.array_equal(a, b) for a, b in zip(replay, expected))
    with pytest.raises(ValueError, match="SHA"):
        restore_source(model_path, head_paths, ("0" * 64, *hashes[1:]))


def test_independent_temperature_fit_and_decisions():
    p = np.array([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3],
                  [0.45, 0.55]], dtype=np.float64)
    y = np.array([0, 1, 0, 0], dtype=np.int64)
    alpha = fit_inverse_temperature(p, y)
    q = apply_inverse_temperature(p, alpha)
    assert 0.25 <= alpha <= 4
    assert np.array_equal(p.argmax(axis=1), q.argmax(axis=1))
    assert np.array_equal(apply_inverse_temperature(p, 1.0), p)
    assert np.max(np.abs(q.sum(axis=1) - 1)) < 1e-12
