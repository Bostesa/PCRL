"""Smoke tests for Steps 1-3: scaffold, abstractions, and models."""

import torch
import pytest

from pcrl.purposes.spec import PurposeSpec, PurposeRegistry
from pcrl.utils.config import PCRLConfig, TrainingConfig, ExperimentConfig, load_config
from pcrl.models.conditioning import (
    FiLMConditioner,
    ConcatConditioner,
    AttentionConditioner,
    build_conditioner,
)
from pcrl.models.encoder import PurposeConditionedEncoder
from pcrl.models.task_head import TaskHead, MultiTaskHead, PurposeTaskHeads
from pcrl.models.auditor import Auditor, AuditorPool, PostHocAuditorSuite


# ── Step 2 tests: PurposeSpec and PurposeRegistry ────────────────────────


class TestPurposeSpec:
    def test_basic_creation(self):
        spec = PurposeSpec(
            name="test",
            allowed_tasks=["income"],
            disallowed_attrs=["race"],
            task_type="classification",
            allowed_task_dims={"income": 2},
            disallowed_attr_dims={"race": 5},
        )
        assert spec.name == "test"
        assert spec.allowed_task_dims["income"] == 2

    def test_requires_allowed_tasks(self):
        with pytest.raises(ValueError, match="at least one allowed task"):
            PurposeSpec(name="bad", allowed_tasks=[], disallowed_attrs=["race"])

    def test_requires_disallowed_attrs(self):
        with pytest.raises(ValueError, match="at least one disallowed attribute"):
            PurposeSpec(name="bad", allowed_tasks=["income"], disallowed_attrs=[])

    def test_invalid_task_type(self):
        with pytest.raises(ValueError):
            PurposeSpec(
                name="bad",
                allowed_tasks=["x"],
                disallowed_attrs=["y"],
                task_type="invalid",
            )


class TestPurposeRegistry:
    def _make_registry(self) -> PurposeRegistry:
        registry = PurposeRegistry()
        registry.register(
            PurposeSpec(
                name="p1",
                allowed_tasks=["t1"],
                disallowed_attrs=["a1", "a2"],
            )
        )
        registry.register(
            PurposeSpec(
                name="p2",
                allowed_tasks=["t2"],
                disallowed_attrs=["a2", "a3"],
            )
        )
        return registry

    def test_register_and_get(self):
        reg = self._make_registry()
        assert len(reg) == 2
        assert reg.get("p1").name == "p1"

    def test_get_missing_raises(self):
        reg = self._make_registry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_duplicate_raises(self):
        reg = self._make_registry()
        with pytest.raises(ValueError, match="already registered"):
            reg.register(
                PurposeSpec(name="p1", allowed_tasks=["t1"], disallowed_attrs=["a1"])
            )

    def test_all_disallowed_for(self):
        reg = self._make_registry()
        assert reg.all_disallowed_for("p1") == {"a1", "a2"}
        assert reg.all_disallowed_for("p2") == {"a2", "a3"}

    def test_contains_and_iter(self):
        reg = self._make_registry()
        assert "p1" in reg
        assert "p3" not in reg
        names = [p.name for p in reg]
        assert names == ["p1", "p2"]


class TestConfigs:
    def test_pcrl_config(self):
        cfg = PCRLConfig(input_dim=20, num_purposes=3)
        assert cfg.hidden_dims == [256, 256]
        assert cfg.repr_dim == 128
        assert cfg.purpose_emb_dim == 64
        assert cfg.num_purposes == 3

    def test_training_config_defaults(self):
        cfg = TrainingConfig()
        assert cfg.lr_task == 1e-3
        assert cfg.lambda_verify == 0.0

    def test_load_yaml_config(self):
        cfg = load_config("configs/adult.yaml")
        assert cfg.model.num_purposes == 3
        assert len(cfg.purposes) == 3
        registry = cfg.build_registry()
        assert len(registry) == 3

    def test_experiment_config_build_registry(self):
        cfg = ExperimentConfig(
            model=PCRLConfig(input_dim=10, num_purposes=1),
            training=TrainingConfig(),
            purposes=[
                PurposeSpec(
                    name="test",
                    allowed_tasks=["t"],
                    disallowed_attrs=["a"],
                )
            ],
        )
        reg = cfg.build_registry()
        assert len(reg) == 1
        assert "test" in reg


# ── Step 3 tests: Conditioning, Encoder, TaskHead, Auditors ─────────────


BATCH_SIZE = 32
FEATURE_DIM = 64
CONDITIONING_DIM = 16
REPR_DIM = 32


class TestConditioners:
    @pytest.mark.parametrize("cond_type", ["film", "concat", "attention"])
    def test_conditioner_shapes(self, cond_type: str):
        cond = build_conditioner(cond_type, FEATURE_DIM, CONDITIONING_DIM)
        h = torch.randn(BATCH_SIZE, FEATURE_DIM)
        emb = torch.randn(BATCH_SIZE, CONDITIONING_DIM)
        out = cond(h, emb)
        assert out.shape == (BATCH_SIZE, FEATURE_DIM)

    def test_film_identity_init(self):
        cond = FiLMConditioner(FEATURE_DIM, CONDITIONING_DIM)
        h = torch.randn(BATCH_SIZE, FEATURE_DIM)
        emb = torch.zeros(BATCH_SIZE, CONDITIONING_DIM)
        out = cond(h, emb)
        # With zero embedding, gamma should be 1 and beta should be 0
        torch.testing.assert_close(out, h)

    def test_different_embeddings_produce_different_outputs(self):
        """After a gradient step, different embeddings must produce different outputs."""
        cond = FiLMConditioner(FEATURE_DIM, CONDITIONING_DIM)
        # Run one training step to break identity init
        h = torch.randn(BATCH_SIZE, FEATURE_DIM)
        emb = torch.randn(BATCH_SIZE, CONDITIONING_DIM)
        out = cond(h, emb)
        loss = out.sum()
        loss.backward()
        with torch.no_grad():
            for p in cond.parameters():
                p -= 0.1 * p.grad
                p.grad.zero_()

        emb1 = torch.randn(BATCH_SIZE, CONDITIONING_DIM)
        emb2 = torch.randn(BATCH_SIZE, CONDITIONING_DIM) + 5.0
        out1 = cond(h, emb1)
        out2 = cond(h, emb2)
        assert not torch.allclose(out1, out2)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown conditioning type"):
            build_conditioner("invalid", FEATURE_DIM, CONDITIONING_DIM)


class TestEncoder:
    @pytest.mark.parametrize("conditioning", ["film", "concat", "attention"])
    def test_encoder_output_shape(self, conditioning: str):
        encoder = PurposeConditionedEncoder(
            input_dim=20,
            hidden_dims=[64, 64],
            repr_dim=REPR_DIM,
            num_purposes=3,
            purpose_emb_dim=CONDITIONING_DIM,
            conditioning=conditioning,
        )
        x = torch.randn(BATCH_SIZE, 20)
        h = encoder(x, 0)
        assert h.shape == (BATCH_SIZE, REPR_DIM)

    def test_different_purposes_produce_different_representations(self):
        """After a gradient step, different purposes must give different representations."""
        encoder = PurposeConditionedEncoder(
            input_dim=20,
            hidden_dims=[64, 64],
            repr_dim=REPR_DIM,
            num_purposes=3,
            purpose_emb_dim=CONDITIONING_DIM,
        )
        # One training step to break FiLM identity init
        x = torch.randn(BATCH_SIZE, 20)
        h = encoder(x, 0)
        loss = h.sum()
        loss.backward()
        with torch.no_grad():
            for p in encoder.parameters():
                if p.grad is not None:
                    p -= 0.1 * p.grad
                    p.grad.zero_()

        encoder.eval()
        h0 = encoder(x, 0)
        h1 = encoder(x, 1)
        assert not torch.allclose(h0, h1, atol=1e-5)

    def test_encode_all_purposes(self):
        encoder = PurposeConditionedEncoder(
            input_dim=20,
            hidden_dims=[64],
            repr_dim=REPR_DIM,
            num_purposes=2,
            purpose_emb_dim=CONDITIONING_DIM,
        )
        x = torch.randn(BATCH_SIZE, 20)
        all_reprs = encoder.encode_all_purposes(x)
        assert len(all_reprs) == 2
        assert all_reprs[0].shape == (BATCH_SIZE, REPR_DIM)

    def test_purpose_idx_as_tensor(self):
        encoder = PurposeConditionedEncoder(
            input_dim=20,
            hidden_dims=[64],
            repr_dim=REPR_DIM,
            num_purposes=3,
            purpose_emb_dim=CONDITIONING_DIM,
        )
        x = torch.randn(BATCH_SIZE, 20)
        idx = torch.zeros(BATCH_SIZE, dtype=torch.long)
        h = encoder(x, idx)
        assert h.shape == (BATCH_SIZE, REPR_DIM)

    def test_get_purpose_embedding(self):
        encoder = PurposeConditionedEncoder(
            input_dim=20,
            hidden_dims=[64],
            repr_dim=REPR_DIM,
            num_purposes=3,
            purpose_emb_dim=CONDITIONING_DIM,
        )
        emb = encoder.get_purpose_embedding(0)
        assert emb.shape == (CONDITIONING_DIM,)


class TestTaskHead:
    def test_classification_head(self):
        head = TaskHead(repr_dim=REPR_DIM, output_dim=5, task_type="classification")
        h = torch.randn(BATCH_SIZE, REPR_DIM)
        logits = head(h)
        assert logits.shape == (BATCH_SIZE, 5)
        preds = head.predict(h)
        assert preds.shape == (BATCH_SIZE,)

    def test_regression_head(self):
        head = TaskHead(repr_dim=REPR_DIM, output_dim=1, task_type="regression")
        h = torch.randn(BATCH_SIZE, REPR_DIM)
        out = head(h)
        assert out.shape == (BATCH_SIZE, 1)
        preds = head.predict(h)
        assert preds.shape == (BATCH_SIZE,)

    def test_purpose_task_heads(self):
        heads = PurposeTaskHeads(
            repr_dim=REPR_DIM,
            purpose_task_specs={
                "p1": {"income": (2, "classification")},
                "p2": {"occupation": (6, "classification")},
            },
        )
        h = torch.randn(BATCH_SIZE, REPR_DIM)
        out = heads(h, "p1")
        assert "income" in out
        assert out["income"].shape == (BATCH_SIZE, 2)


class TestAuditor:
    def test_mlp_auditor(self):
        auditor = Auditor(repr_dim=REPR_DIM, output_dim=3)
        h = torch.randn(BATCH_SIZE, REPR_DIM)
        logits = auditor(h)
        assert logits.shape == (BATCH_SIZE, 3)

    def test_auditor_pool(self):
        pool = AuditorPool(repr_dim=REPR_DIM, output_dim=2, pool_size=3)
        h = torch.randn(BATCH_SIZE, REPR_DIM)
        all_logits = pool(h)
        assert len(all_logits) == 3
        best_acc = pool.get_best_accuracy(h, torch.randint(0, 2, (BATCH_SIZE,)))
        assert 0 <= best_acc <= 1

    def test_posthoc_auditor_suite(self):
        import numpy as np

        suite = PostHocAuditorSuite(random_state=0)
        X = np.random.randn(200, REPR_DIM)
        y = (X[:, 0] > 0).astype(int)

        suite.fit(X[:150], y[:150])
        results = suite.evaluate(X[150:], y[150:])

        assert "logistic_regression" in results
        assert "random_forest" in results
        assert "svm_rbf" in results
        for name, metrics in results.items():
            assert "accuracy" in metrics
            assert "balanced_accuracy" in metrics
            assert 0 <= metrics["accuracy"] <= 1

        best = suite.best_accuracy(X[150:], y[150:])
        assert best > 0.5  # Should do better than random on this easy task

    def test_posthoc_auditor_suite_with_tensors(self):
        suite = PostHocAuditorSuite(random_state=0)
        X = torch.randn(200, REPR_DIM)
        y = (X[:, 0] > 0).long()

        suite.fit(X[:150], y[:150])
        results = suite.evaluate(X[150:], y[150:])
        assert len(results) >= 3


# ── Integration test: full forward pass ──────────────────────────────────


class TestIntegration:
    def test_full_forward_pass(self):
        """End-to-end: encoder -> task_heads + auditors, all three conditioning types."""
        for cond in ["film", "concat", "attention"]:
            encoder = PurposeConditionedEncoder(
                input_dim=20,
                hidden_dims=[64, 64],
                repr_dim=REPR_DIM,
                num_purposes=2,
                purpose_emb_dim=CONDITIONING_DIM,
                conditioning=cond,
            )
            task_head = TaskHead(repr_dim=REPR_DIM, output_dim=2)
            auditor = Auditor(repr_dim=REPR_DIM, output_dim=3)

            x = torch.randn(BATCH_SIZE, 20)
            h = encoder(x, 0)
            task_out = task_head(h)
            audit_out = auditor(h)

            assert task_out.shape == (BATCH_SIZE, 2)
            assert audit_out.shape == (BATCH_SIZE, 3)

    def test_gradients_flow(self):
        """Verify gradients flow through encoder and task head."""
        encoder = PurposeConditionedEncoder(
            input_dim=20,
            hidden_dims=[64],
            repr_dim=REPR_DIM,
            num_purposes=2,
            purpose_emb_dim=CONDITIONING_DIM,
        )
        task_head = TaskHead(repr_dim=REPR_DIM, output_dim=2)

        x = torch.randn(BATCH_SIZE, 20)
        y = torch.randint(0, 2, (BATCH_SIZE,))

        h = encoder(x, 0)
        logits = task_head(h)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()

        # Check gradients exist on encoder params
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_auditor_detached_does_not_update_encoder(self):
        """Auditor trained on detached representations should not affect encoder."""
        encoder = PurposeConditionedEncoder(
            input_dim=20,
            hidden_dims=[64],
            repr_dim=REPR_DIM,
            num_purposes=2,
            purpose_emb_dim=CONDITIONING_DIM,
        )
        auditor = Auditor(repr_dim=REPR_DIM, output_dim=2)

        x = torch.randn(BATCH_SIZE, 20)
        z = torch.randint(0, 2, (BATCH_SIZE,))

        h = encoder(x, 0)
        h_detached = h.detach()
        audit_logits = auditor(h_detached)
        loss = torch.nn.functional.cross_entropy(audit_logits, z)
        loss.backward()

        # Encoder should have no gradients
        for param in encoder.parameters():
            assert param.grad is None
