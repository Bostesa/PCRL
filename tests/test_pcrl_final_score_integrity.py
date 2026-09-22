"""A completed score cannot be accepted after its predictions change."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np

from experiments.pcrl_final_prospective_v1.audit_panel import unit_dir
from experiments.pcrl_final_prospective_v1.common import array_hash, atomic_json, sha
from experiments.pcrl_final_prospective_v1.score_integrity import check_one


class ScoreIntegrityTest(unittest.TestCase):
    def test_rejects_changed_predictions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            unit = ('Q', 0, 'attack:A/SEX')
            d = unit_dir(root, *unit)
            d.mkdir(parents=True)
            joblib.dump({'selection': 'frozen'}, d/'registry.joblib')
            q = np.array([[[.8, .2]], [[.3, .7]]])
            law = np.ones((2, 1))
            loss = -np.log(np.array([.8, .7]))
            np.savez_compressed(d/'pred_final.npz', probabilities=q, token_probs=law)
            np.savez_compressed(d/'score_final.npz', ids=np.array(['a', 'b']), y=np.array([0, 1]),
                                loss=loss, weights=np.ones(2))
            mean = float(loss.mean())
            atomic_json(d/'SCORED_final.json', {
                'release': 'Q', 'anchor': 0, 'role': unit[2], 'selection': 'frozen', 'rows': 2,
                'score_sha256': sha(d/'score_final.npz'), 'prediction_sha256': sha(d/'pred_final.npz'),
                'loss_hash': array_hash(loss), 'prediction_hash': array_hash(q), 'token_hash': array_hash(law),
                'ce': {'unweighted': mean, 'weighted': mean, 'balanced': mean}})
            check_one(root, unit)
            with (d/'pred_final.npz').open('ab') as output:
                output.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'file hash'):
                check_one(root, unit)


if __name__ == '__main__':
    unittest.main()
