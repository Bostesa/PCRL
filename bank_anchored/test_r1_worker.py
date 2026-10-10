"""Queue boundary for the R1 raw-bank admission worker."""

import numpy as np
import pytest

from bank_anchored.r1_worker import run


def test_direct_worker_call_cannot_open_any_role_archive(tmp_path, monkeypatch):
    spec = tmp_path / "spec.json"
    spec.write_text("{}\n")
    monkeypatch.setattr(np, "load", lambda *args, **kwargs:
                        pytest.fail("role archive opened before queue check"))
    with pytest.raises(RuntimeError, match="queue"):
        run(spec)
