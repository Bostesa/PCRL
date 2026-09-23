"""Public synthetic-only encoder used to exercise the release wire contract."""
from __future__ import annotations

import numpy as np


class SyntheticCodebook:
    def n_states(self, name):
        if name != "T0":
            raise ValueError("synthetic fixture has only T0")
        return 32


class SyntheticEncoder:
    """Untrained deterministic toy code; never a substitute for the ACS encoder."""

    def __init__(self):
        self.code = SyntheticCodebook()

    def encode(self, inputs):
        x = np.asarray(inputs.x_a)
        h = np.asarray(inputs.h_a)
        t = (np.floor((x[:, 0] + 2.) * 7).astype(np.int64)
             + np.floor((h[:, 0] + 2.) * 3).astype(np.int64)) % 32
        return {"codes": {"T0": t}}
