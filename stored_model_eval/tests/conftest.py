import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")  # keep HistGradientBoosting from oversubscribing small fixtures
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from stored_model_eval.config import load_protocol  # noqa: E402
from stored_model_eval.guards import FitAuthorization  # noqa: E402


@pytest.fixture
def cfg():
    c = load_protocol(None)
    c["bootstrap"]["n_boot"] = 400
    c["support"]["min_class_support"] = 20
    return c


@pytest.fixture
def auth():
    return FitAuthorization.synthetic_only()
