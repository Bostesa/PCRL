"""Exact one-hot public release law for separately fitted deterministic gates.

Import with the shared sampled_channel_v1/impl directory on PYTHONPATH. This
subclass keeps the R12 cached wire adapter while changing the emitted law.
"""

import numpy as np

from channel import JointPolicy, _entropy_margin


class DeterministicJointPolicy(JointPolicy):
    law_kind = "deterministic_argmax_v1"

    def sparse_law(self, score1, score2, cell1: int, cell2: int):
        # Parent performs exact score, fine-cell, class and action validation.
        pairs, _ = self._pairs_and_probabilities(score1, score2, cell1, cell2)
        features = np.r_[score1, score2, _entropy_margin(score1),
                         _entropy_margin(score2)]
        slots = [4 * a + b for a in range(len(self.actions1[cell1]))
                 for b in range(len(self.actions2[cell2]))]
        winner = int(np.argmax(self.gate.logits(features)[slots]))
        i, j = pairs[winner]
        return ((int(i), int(j), 1.0),)
