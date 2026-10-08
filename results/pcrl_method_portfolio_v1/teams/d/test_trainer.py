"""Tiny engineering gradient tests; run only through the shared compute queue."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trainer import DTrainConfig, DTrainingData, make_sign_tables, mmd_terms, surrogate_select, train_d


class TrainerGradientTests(unittest.TestCase):
    def test_hard_forward_assignment_has_codebook_gradient_but_frozen_input_does_not(self):
        h = torch.tensor([[0.1, 0.0], [1.9, 0.0]], dtype=torch.float32)
        centroids = torch.nn.Parameter(torch.tensor([[0.0, 0.0], [2.0, 0.0], [9.0, 9.0]]))
        decoded, ids, assignment = surrogate_select(h, centroids)
        loss = ((decoded - h) ** 2).sum()
        loss.backward()
        self.assertEqual(ids.tolist(), [0, 1])
        np.testing.assert_array_equal(assignment.detach().numpy(), [[1, 0, 0], [0, 1, 0]])
        self.assertGreater(float(centroids.grad[:2].abs().sum()), 0.0)
        self.assertIsNone(h.grad)

    def test_paired_mmd_sees_xor_when_local_mmd_is_zero(self):
        # Frozen shared-law L3 clue: U is local 0 and U XOR S is local 1.
        u = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
        v = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        sex = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        y0 = torch.nn.functional.one_hot(u[:, 0].long(), 2).float()
        y1 = torch.nn.functional.one_hot(v[:, 0].long(), 2).float()
        local0, local1, pair = mmd_terms(y0, y1, sex, make_sign_tables(123, 2, 128))
        self.assertLess(float(local0), 1e-12)
        self.assertLess(float(local1), 1e-12)
        self.assertGreater(float(pair), 0.01)

    def test_dependence_gradient_reaches_selected_centroids(self):
        h0 = torch.tensor([[0.0], [0.0], [2.0], [2.0]])
        h1 = torch.tensor([[0.0], [2.0], [0.0], [2.0]])
        sex = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        centroids = torch.nn.Parameter(torch.tensor([[0.1], [1.9]]))
        _, _, y0 = surrogate_select(h0, centroids)
        _, _, y1 = surrogate_select(h1, centroids)
        terms = mmd_terms(y0, y1, sex, make_sign_tables(123, 2, 128))
        (sum(terms) / 3).backward()
        self.assertGreater(float(centroids.grad.abs().sum()), 0.0)

    def test_public_ids_expose_sex_even_with_nearly_equal_decoded_vectors(self):
        centroids = torch.tensor([[0.0], [1e-5]])
        h = centroids.clone()
        z, ids, hard_y = surrogate_select(h, centroids)
        sex = torch.tensor([0, 1])
        local, _, _ = mmd_terms(hard_y, hard_y, sex, make_sign_tables(11, 2, 128))
        self.assertEqual(ids.tolist(), [0, 1])
        self.assertLess(float((z[0] - z[1]).square().sum()), 1e-8)
        self.assertGreater(float(local), 0.1)

    def test_identity_loss_gradient_can_change_hard_code_assignments(self):
        h = torch.tensor([[-0.1], [0.1]])
        sex = torch.tensor([0, 1])
        centroids = torch.nn.Parameter(torch.tensor([[-0.1], [0.1]]))
        _, before, y = surrogate_select(h, centroids)
        local, _, _ = mmd_terms(y, y, sex, make_sign_tables(15, 2, 128))
        local.backward()
        self.assertGreater(float(centroids.grad.abs().sum()), 0.0)
        gap = centroids.detach()[1, 0] - centroids.detach()[0, 0]
        gradient_gap = centroids.grad[1, 0] - centroids.grad[0, 0]
        self.assertGreater(float(gradient_gap), 0.0)
        # A fixed 10% overshoot beyond the center-crossing step changes IDs.
        step = 1.1 * gap / gradient_gap
        moved = centroids.detach() - step * centroids.grad
        _, after, _ = surrogate_select(h, moved)
        self.assertFalse(torch.equal(before, after))

    def test_tiny_training_returns_code_only_deployment_adapter(self):
        # Engineering scale-down of the fixed architecture, not an Adult fit.
        h0 = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.],
                       [2., 0.], [2., 1.], [3., 0.], [3., 1.]], dtype=np.float32)
        h1 = h0[:, ::-1].copy()
        y0 = (h0[:, 0] > 1).astype(np.int64)
        y1 = (h1[:, 0] > 0).astype(np.int64)
        p0 = np.column_stack((1 - (0.1 + 0.8 * y0), 0.1 + 0.8 * y0))
        p1 = np.column_stack((1 - (0.1 + 0.8 * y1), 0.1 + 0.8 * y1))
        data = DTrainingData(h0, h1, y0, y1, p0, p1,
                             np.array([0, 1, 0, 1, 1, 0, 1, 0]),
                             np.arange(8), np.arange(8), role="SYNTHETIC")
        cfg = DTrainConfig(codes=4, hidden=4, epochs=1, batch_size=8, pair_features=16)
        fit = train_d(data, seed=7, protected=True, config=cfg)
        self.assertEqual(len(fit.history), 1)
        self.assertEqual(fit.release.code_length_bits, 2)
        self.assertEqual(fit.fit_code_change["rows"], 8)
        self.assertIn("coalition", fit.fit_code_change["occupancy"])
        out = fit.release.recipient(0, h0)
        self.assertEqual(out["code_ids"].shape, (8, 1))
        self.assertEqual(out["probabilities"].shape, (8, 2))
        standardized = torch.from_numpy(fit.release.normalizer.transform(0, h0).astype(np.float32))
        decoded, direct_ids, _ = surrogate_select(standardized, fit.release.codebook)
        direct_probs = torch.softmax(fit.release.heads[0](decoded), dim=1)
        np.testing.assert_array_equal(out["code_ids"][:, 0], direct_ids.numpy())
        np.testing.assert_array_equal(out["probabilities"], direct_probs.detach().numpy())

    def test_misaligned_purpose_rows_are_rejected_before_fit(self):
        h = np.zeros((4, 2))
        y = np.zeros(4, dtype=np.int64)
        teacher = np.tile([0.9, 0.1], (4, 1))
        with self.assertRaises(ValueError):
            DTrainingData(h, h, y, y, teacher, teacher,
                          np.array([0, 1, 0, 1]),
                          np.arange(4), np.array([1, 0, 2, 3]), role="SYNTHETIC")


if __name__ == "__main__":
    unittest.main()
