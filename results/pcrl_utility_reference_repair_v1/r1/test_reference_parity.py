"""A forbidden all-role release cannot be opened by safe U restore."""

import json

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from jcv.train import Model
from reference_parity import _sha_file, safe_load_u


def test_safe_restore_ignores_complete_listed_release(tmp_path, monkeypatch):
    root = tmp_path / "rel__s0__U"
    root.mkdir()
    model = Model(83, [2, 6], 0)
    torch.save(model.state_dict(), root / "model.pt")
    rng = np.random.default_rng(19)
    for i, classes in ((0, 2), (1, 6)):
        x = rng.normal(size=(72, 16))
        y = np.arange(72) % classes
        head = LogisticRegression(max_iter=100).fit(x, y)
        joblib.dump(head, root / f"head_{i}.joblib")
    (root / "record.json").write_text(json.dumps({"seed": 0}))
    files = {name: _sha_file(root / name) for name in
             ("model.pt", "head_0.joblib", "head_1.joblib", "record.json")}
    files["release.npz"] = "0" * 64  # Deliberately absent forbidden artifact.
    (root / "COMPLETE.json").write_text(json.dumps({"id": "rel__s0__U", "files": files}))
    original = _sha_file

    def deny_release(path):
        if path.name == "release.npz":
            raise AssertionError("forbidden all-role release was read")
        return original(path)

    monkeypatch.setattr("reference_parity._sha_file", deny_release)
    restored, heads, model_hash = safe_load_u(root, 0)
    assert model_hash == files["model.pt"]
    assert len(heads) == 2
    assert all(torch.equal(model.state_dict()[key], restored.state_dict()[key])
               for key in model.state_dict())
