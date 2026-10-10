"""Tiny release-contract test: trained hard gates stay hard when deployed."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "impl"))
from bank import FixedBank
from channel import GateWeights, JointPolicy
from release import CachedJointRelease, _mechanism_hash
from deterministic_policy import DeterministicJointPolicy


class DeterministicPolicyTest(unittest.TestCase):
    def test_actual_cached_release_is_one_hot_and_distinct_from_softmax(self):
        score1 = np.array([.9, .1], dtype=np.float64)
        score2 = np.array([.8, .2], dtype=np.float64)
        bank1 = FixedBank.from_source(np.array([[.9, .1], [.6, .4]]),
                                      np.arange(2), np.zeros(2, dtype=np.int64))
        bank2 = FixedBank.from_source(np.array([[.8, .2], [.7, .3]]),
                                      np.arange(2), np.zeros(2, dtype=np.int64))
        bias = np.zeros(16)
        bias[4] = 5.0  # fixed padded slot 4 = recipient 1's second code
        gate = GateWeights(np.zeros((8, 16)), np.zeros(16),
                           np.zeros((16, 16)), bias)
        actions1 = (tuple(map(int, bank1.old_to_global)),)
        actions2 = (tuple(map(int, bank2.old_to_global)),)
        hard = DeterministicJointPolicy(bank1, bank2, actions1, actions2, gate)
        soft = JointPolicy(bank1, bank2, actions1, actions2, gate)
        hard_law = hard.sparse_law(score1, score2, 0, 0)
        self.assertEqual(hard_law, ((actions1[0][1], actions2[0][0], 1.0),))
        self.assertEqual(len(soft.sparse_law(score1, score2, 0, 0)), 4)
        self.assertNotEqual(_mechanism_hash(hard), _mechanism_hash(soft))
        with tempfile.TemporaryDirectory() as directory:
            service = CachedJointRelease(hard, cache_path=Path(directory) / "hard.sqlite",
                                         release_id="hard-fixture", source_seed=0,
                                         role="SYNTHETIC", realization=0,
                                         source_binding_sha256="0" * 64,
                                         private_root=b"k" * 32, offline=True)
            r1 = service.recipient(recipient=1, unit="same-person", score1=score1,
                                   score2=score2, cell1=0, cell2=0)
            r2 = service.recipient(recipient=2, unit="same-person", score1=score1,
                                   score2=score2, cell1=0, cell2=0)
            self.assertEqual((r1["code_id"], r2["code_id"]), hard_law[0][:2])
            self.assertEqual(r1["probability"], bank1.probability[r1["code_id"]].tolist())
            self.assertEqual(r2["probability"], bank2.probability[r2["code_id"]].tolist())


if __name__ == "__main__":
    unittest.main()
