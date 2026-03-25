"""Tests for compositional purpose algebra (Step 4) and verification certificates (Step 5)."""

import numpy as np
import pytest
import torch

from pcrl.purposes.spec import PurposeSpec, PurposeRegistry
from pcrl.purposes.composition import (
    ComposedPurpose,
    CompositionLoss,
    LearnedComposer,
    compose_and,
    compose_embeddings_additive,
    compose_embeddings_max,
    compose_hierarchy,
    compose_or,
)
from pcrl.purposes.verification import (
    CertificateResult,
    LinearComplianceCertificate,
    NullSpaceCertificate,
)
from pcrl.evaluation.certificates import (
    ComplianceReport,
    EmpiricalAudit,
    LinearAudit,
)
from pcrl.models.encoder import PurposeConditionedEncoder


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def purpose_a() -> PurposeSpec:
    """Purpose with tasks=[income], disallowed=[race, sex]."""
    return PurposeSpec(
        name="income_prediction",
        allowed_tasks=["income"],
        disallowed_attrs=["race", "sex"],
        allowed_task_dims={"income": 2},
        disallowed_attr_dims={"race": 5, "sex": 2},
    )


@pytest.fixture
def purpose_b() -> PurposeSpec:
    """Purpose with tasks=[occupation], disallowed=[sex, age]."""
    return PurposeSpec(
        name="employment_analysis",
        allowed_tasks=["occupation"],
        disallowed_attrs=["sex", "age"],
        allowed_task_dims={"occupation": 6},
        disallowed_attr_dims={"sex": 2, "age": 4},
    )


@pytest.fixture
def purpose_c() -> PurposeSpec:
    """Purpose with tasks=[education], disallowed=[race]."""
    return PurposeSpec(
        name="education_assessment",
        allowed_tasks=["education"],
        disallowed_attrs=["race"],
        allowed_task_dims={"education": 4},
        disallowed_attr_dims={"race": 5},
    )


# ── Step 4: Composition tests ───────────────────────────────────────────


class TestComposedPurposeAND:
    """AND composition: support tasks from both, hide attrs disallowed by either."""

    def test_and_allowed_tasks_union(self, purpose_a, purpose_b):
        composed = compose_and(purpose_a, purpose_b)
        assert set(composed.allowed_tasks) == {"income", "occupation"}

    def test_and_disallowed_attrs_union(self, purpose_a, purpose_b):
        composed = compose_and(purpose_a, purpose_b)
        # race from A, sex from both, age from B → union = {race, sex, age}
        assert set(composed.disallowed_attrs) == {"race", "sex", "age"}

    def test_and_task_dims_merged(self, purpose_a, purpose_b):
        composed = compose_and(purpose_a, purpose_b)
        assert composed.allowed_task_dims["income"] == 2
        assert composed.allowed_task_dims["occupation"] == 6

    def test_and_attr_dims_merged(self, purpose_a, purpose_b):
        composed = compose_and(purpose_a, purpose_b)
        assert composed.disallowed_attr_dims["race"] == 5
        assert composed.disallowed_attr_dims["sex"] == 2
        assert composed.disallowed_attr_dims["age"] == 4

    def test_and_name(self, purpose_a, purpose_b):
        composed = compose_and(purpose_a, purpose_b)
        assert composed.name == "income_prediction_AND_employment_analysis"

    def test_and_with_overlapping_attrs(self, purpose_a, purpose_b):
        """sex is disallowed by both → should appear in AND union."""
        composed = compose_and(purpose_a, purpose_b)
        assert "sex" in composed.disallowed_attrs

    def test_and_with_non_overlapping_attrs(self, purpose_a, purpose_c):
        """A: [race, sex], C: [race] → AND = [race, sex]."""
        composed = compose_and(purpose_a, purpose_c)
        assert set(composed.disallowed_attrs) == {"race", "sex"}

    def test_and_to_purpose_spec(self, purpose_a, purpose_b):
        composed = compose_and(purpose_a, purpose_b)
        spec = composed.to_purpose_spec()
        assert isinstance(spec, PurposeSpec)
        assert set(spec.allowed_tasks) == {"income", "occupation"}
        assert set(spec.disallowed_attrs) == {"race", "sex", "age"}


class TestComposedPurposeOR:
    """OR composition: support tasks from either, hide only attrs disallowed by both."""

    def test_or_allowed_tasks_union(self, purpose_a, purpose_b):
        composed = compose_or(purpose_a, purpose_b)
        assert set(composed.allowed_tasks) == {"income", "occupation"}

    def test_or_disallowed_attrs_intersection(self, purpose_a, purpose_b):
        composed = compose_or(purpose_a, purpose_b)
        # A: [race, sex], B: [sex, age] → intersection = {sex}
        assert set(composed.disallowed_attrs) == {"sex"}

    def test_or_with_no_overlap(self, purpose_a, purpose_c):
        """A: [race, sex], C: [race] → OR intersection = {race}."""
        composed = compose_or(purpose_a, purpose_c)
        assert set(composed.disallowed_attrs) == {"race"}

    def test_or_empty_intersection(self):
        """If disallowed attrs have no overlap, OR hides nothing."""
        p1 = PurposeSpec(name="p1", allowed_tasks=["t1"], disallowed_attrs=["a1"])
        p2 = PurposeSpec(name="p2", allowed_tasks=["t2"], disallowed_attrs=["a2"])
        composed = compose_or(p1, p2)
        assert composed.disallowed_attrs == []

    def test_or_name(self, purpose_a, purpose_b):
        composed = compose_or(purpose_a, purpose_b)
        assert composed.name == "income_prediction_OR_employment_analysis"

    def test_or_attr_dims_only_intersection(self, purpose_a, purpose_b):
        composed = compose_or(purpose_a, purpose_b)
        # Only sex is in the intersection
        assert "sex" in composed.disallowed_attr_dims
        assert "race" not in composed.disallowed_attr_dims
        assert "age" not in composed.disallowed_attr_dims


class TestComposedPurposeHierarchy:
    """Hierarchy: child inherits all parent constraints plus its own."""

    def test_hierarchy_inherits_parent_attrs(self, purpose_a, purpose_c):
        """Child C: [race], Parent A: [race, sex] → child gets [race, sex]."""
        composed = compose_hierarchy(parent=purpose_a, child=purpose_c)
        assert set(composed.disallowed_attrs) == {"race", "sex"}

    def test_hierarchy_merges_tasks(self, purpose_a, purpose_c):
        composed = compose_hierarchy(parent=purpose_a, child=purpose_c)
        assert set(composed.allowed_tasks) == {"income", "education"}

    def test_hierarchy_name(self, purpose_a, purpose_c):
        composed = compose_hierarchy(parent=purpose_a, child=purpose_c)
        assert "inherits" in composed.name


class TestCompositionInvariants:
    """Cross-cutting invariants that must hold for all compositions."""

    def test_and_is_more_restrictive_than_either(self, purpose_a, purpose_b):
        composed = compose_and(purpose_a, purpose_b)
        a_attrs = set(purpose_a.disallowed_attrs)
        b_attrs = set(purpose_b.disallowed_attrs)
        composed_attrs = set(composed.disallowed_attrs)
        assert a_attrs <= composed_attrs
        assert b_attrs <= composed_attrs

    def test_or_is_less_restrictive_than_either(self, purpose_a, purpose_b):
        composed = compose_or(purpose_a, purpose_b)
        a_attrs = set(purpose_a.disallowed_attrs)
        b_attrs = set(purpose_b.disallowed_attrs)
        composed_attrs = set(composed.disallowed_attrs)
        assert composed_attrs <= a_attrs
        assert composed_attrs <= b_attrs

    def test_and_commutative_attrs(self, purpose_a, purpose_b):
        ab = compose_and(purpose_a, purpose_b)
        ba = compose_and(purpose_b, purpose_a)
        assert set(ab.disallowed_attrs) == set(ba.disallowed_attrs)
        assert set(ab.allowed_tasks) == set(ba.allowed_tasks)

    def test_or_commutative_attrs(self, purpose_a, purpose_b):
        ab = compose_or(purpose_a, purpose_b)
        ba = compose_or(purpose_b, purpose_a)
        assert set(ab.disallowed_attrs) == set(ba.disallowed_attrs)
        assert set(ab.allowed_tasks) == set(ba.allowed_tasks)


# ── Step 4: Embedding composition tests ──────────────────────────────────


class TestEmbeddingComposition:
    def test_additive_shape(self):
        emb1 = torch.randn(8, 32)
        emb2 = torch.randn(8, 32)
        result = compose_embeddings_additive(emb1, emb2)
        assert result.shape == (8, 32)
        torch.testing.assert_close(result, emb1 + emb2)

    def test_max_shape(self):
        emb1 = torch.randn(8, 32)
        emb2 = torch.randn(8, 32)
        result = compose_embeddings_max(emb1, emb2)
        assert result.shape == (8, 32)
        torch.testing.assert_close(result, torch.max(emb1, emb2))

    def test_learned_composer(self):
        composer = LearnedComposer(emb_dim=32)
        emb1 = torch.randn(8, 32)
        emb2 = torch.randn(8, 32)
        result = composer(emb1, emb2)
        assert result.shape == (8, 32)

    def test_learned_composer_has_gradients(self):
        composer = LearnedComposer(emb_dim=16)
        emb1 = torch.randn(4, 16, requires_grad=True)
        emb2 = torch.randn(4, 16, requires_grad=True)
        result = composer(emb1, emb2)
        loss = result.sum()
        loss.backward()
        assert emb1.grad is not None
        for p in composer.parameters():
            assert p.grad is not None

    def test_additive_unbatched(self):
        emb1 = torch.randn(32)
        emb2 = torch.randn(32)
        result = compose_embeddings_additive(emb1, emb2)
        assert result.shape == (32,)


class TestEncoderWithComposedEmbedding:
    """Test that encoder.forward_with_embedding works with composed embeddings."""

    def test_forward_with_embedding_shape(self):
        encoder = PurposeConditionedEncoder(
            input_dim=20, hidden_dims=[64], repr_dim=32,
            num_purposes=3, purpose_emb_dim=16,
        )
        x = torch.randn(8, 20)
        emb = torch.randn(8, 16)
        h = encoder.forward_with_embedding(x, emb)
        assert h.shape == (8, 32)

    def test_additive_composition_through_encoder(self):
        encoder = PurposeConditionedEncoder(
            input_dim=20, hidden_dims=[64], repr_dim=32,
            num_purposes=3, purpose_emb_dim=16,
        )
        x = torch.randn(8, 20)
        e0 = encoder.get_purpose_embedding(0).unsqueeze(0).expand(8, -1)
        e1 = encoder.get_purpose_embedding(1).unsqueeze(0).expand(8, -1)
        composed_emb = compose_embeddings_additive(e0, e1)
        h_composed = encoder.forward_with_embedding(x, composed_emb)
        assert h_composed.shape == (8, 32)


class TestCompositionLoss:
    def test_entropy_loss_decreases_for_uniform(self):
        """Uniform logits should give lower loss than peaked logits."""
        loss_fn = CompositionLoss(confusion_type="entropy")

        # Uniform logits → high entropy → low loss (negative entropy)
        uniform_logits = torch.zeros(16, 4)
        loss_uniform = loss_fn({"attr": uniform_logits})

        # Peaked logits → low entropy → high loss
        peaked = torch.zeros(16, 4)
        peaked[:, 0] = 10.0
        loss_peaked = loss_fn({"attr": peaked})

        assert loss_uniform < loss_peaked

    def test_kl_loss_decreases_for_uniform(self):
        loss_fn = CompositionLoss(confusion_type="uniform_kl")

        uniform_logits = torch.zeros(16, 4)
        loss_uniform = loss_fn({"attr": uniform_logits})

        peaked = torch.zeros(16, 4)
        peaked[:, 0] = 10.0
        loss_peaked = loss_fn({"attr": peaked})

        assert loss_uniform < loss_peaked

    def test_empty_logits_returns_zero(self):
        loss_fn = CompositionLoss()
        loss = loss_fn({})
        assert loss.item() == 0.0

    def test_multiple_attrs(self):
        loss_fn = CompositionLoss(confusion_type="entropy")
        logits = {
            "race": torch.randn(16, 5, requires_grad=True),
            "sex": torch.randn(16, 2, requires_grad=True),
        }
        loss = loss_fn(logits)
        assert loss.requires_grad


# ── Step 5: Verification certificate tests ───────────────────────────────


class TestLinearComplianceCertificate:
    def test_certifies_random_representations(self):
        """Random representations uncorrelated with labels should certify."""
        np.random.seed(42)
        # Use many samples relative to dims to avoid spurious correlations
        H = np.random.randn(2000, 16)
        Z = np.random.randint(0, 3, size=2000)
        cert = LinearComplianceCertificate(epsilon=0.05)
        result = cert.check(H, Z)
        assert result.r_squared < 0.05
        assert result.certified

    def test_rejects_correlated_representations(self):
        """Representations perfectly correlated with labels should NOT certify."""
        np.random.seed(42)
        Z = np.random.randint(0, 2, size=200)
        # Build H where first column perfectly encodes Z
        H = np.random.randn(200, 16)
        H[:, 0] = Z.astype(float) * 10.0
        cert = LinearComplianceCertificate(epsilon=0.05)
        result = cert.check(H, Z)
        assert result.r_squared > 0.3
        assert not result.certified

    def test_accepts_torch_tensors(self):
        H = torch.randn(200, 16)
        Z = torch.randint(0, 3, (200,))
        cert = LinearComplianceCertificate(epsilon=0.1)
        result = cert.check(H, Z)
        assert isinstance(result, CertificateResult)
        assert 0 <= result.r_squared <= 1

    def test_r_squared_range(self):
        np.random.seed(0)
        H = np.random.randn(300, 20)
        Z = np.random.randint(0, 5, size=300)
        cert = LinearComplianceCertificate()
        result = cert.check(H, Z)
        assert 0 <= result.r_squared <= 1


class TestNullSpaceCertificate:
    def test_variance_preserved_high_for_random(self):
        """With random labels, projecting out Z should preserve most variance."""
        np.random.seed(42)
        H = np.random.randn(500, 32)
        Z = np.random.randint(0, 3, size=500)
        cert = NullSpaceCertificate(epsilon=0.05)
        result = cert.check(H, Z)
        assert result.variance_preserved is not None
        assert result.variance_preserved > 0.8  # Most variance preserved

    def test_variance_lost_when_correlated(self):
        """When H strongly encodes Z, null-space projection loses variance."""
        np.random.seed(42)
        Z = np.random.randint(0, 2, size=200)
        # H mostly determined by Z
        H = np.outer(Z, np.ones(8)) * 5.0 + np.random.randn(200, 8) * 0.1
        cert = NullSpaceCertificate(epsilon=0.01)
        result = cert.check(H, Z)
        assert result.variance_preserved is not None
        assert result.variance_preserved < 0.5  # Lots of variance lost

    def test_both_r2_and_variance(self):
        np.random.seed(0)
        H = np.random.randn(300, 16)
        Z = np.random.randint(0, 2, size=300)
        cert = NullSpaceCertificate()
        result = cert.check(H, Z)
        assert 0 <= result.r_squared <= 1
        assert 0 <= result.variance_preserved <= 1


class TestLinearAudit:
    def test_audit_returns_both_certificates(self):
        np.random.seed(42)
        H = np.random.randn(300, 16)
        Z = np.random.randint(0, 3, size=300)
        audit = LinearAudit(epsilon=0.05)
        linear_result, null_result = audit.audit(H, Z)
        assert isinstance(linear_result, CertificateResult)
        assert isinstance(null_result, CertificateResult)
        assert null_result.variance_preserved is not None


class TestEmpiricalAudit:
    def test_audit_random(self):
        """Auditors on random data should get near-chance accuracy."""
        np.random.seed(42)
        train_H = np.random.randn(200, 16)
        train_Z = np.random.randint(0, 2, size=200)
        test_H = np.random.randn(100, 16)
        test_Z = np.random.randint(0, 2, size=100)
        audit = EmpiricalAudit(random_state=42)
        best_acc, results = audit.audit(train_H, train_Z, test_H, test_Z)
        assert 0 <= best_acc <= 1
        # Should be near chance (0.5) for random data — allow some slack
        assert best_acc < 0.75

    def test_audit_correlated(self):
        """Auditors on correlated data should get high accuracy."""
        np.random.seed(42)
        n = 300
        Z = np.random.randint(0, 2, size=n)
        H = np.random.randn(n, 8)
        H[:, 0] = Z * 10.0  # Perfect signal in first feature
        audit = EmpiricalAudit(random_state=42)
        best_acc, results = audit.audit(H[:200], Z[:200], H[200:], Z[200:])
        assert best_acc > 0.9


# ── Integration: composition + encoder + verification ────────────────────


class TestCompositionVerificationIntegration:
    """End-to-end test: compose purposes, run encoder, verify certificates."""

    def test_and_composition_end_to_end(self, purpose_a, purpose_b):
        """AND composition through encoder and linear certificate."""
        encoder = PurposeConditionedEncoder(
            input_dim=20, hidden_dims=[64], repr_dim=32,
            num_purposes=3, purpose_emb_dim=16,
        )
        encoder.eval()
        x = torch.randn(200, 20)

        # Compose embeddings
        e0 = encoder.get_purpose_embedding(0).unsqueeze(0).expand(200, -1)
        e1 = encoder.get_purpose_embedding(1).unsqueeze(0).expand(200, -1)
        composed_emb = compose_embeddings_additive(e0, e1)

        with torch.no_grad():
            h_composed = encoder.forward_with_embedding(x, composed_emb)

        # Run certificate on random labels (should pass)
        Z = np.random.randint(0, 3, size=200)
        cert = LinearComplianceCertificate(epsilon=0.1)
        result = cert.check(h_composed.numpy(), Z)
        assert isinstance(result, CertificateResult)

        # Verify composed purpose spec
        composed = compose_and(purpose_a, purpose_b)
        assert set(composed.disallowed_attrs) == {"race", "sex", "age"}

    def test_or_composition_end_to_end(self, purpose_a, purpose_b):
        """OR composition through encoder and certificate check."""
        encoder = PurposeConditionedEncoder(
            input_dim=20, hidden_dims=[64], repr_dim=32,
            num_purposes=3, purpose_emb_dim=16,
        )
        encoder.eval()
        x = torch.randn(200, 20)

        e0 = encoder.get_purpose_embedding(0).unsqueeze(0).expand(200, -1)
        e1 = encoder.get_purpose_embedding(1).unsqueeze(0).expand(200, -1)
        composed_emb = compose_embeddings_max(e0, e1)

        with torch.no_grad():
            h_composed = encoder.forward_with_embedding(x, composed_emb)

        assert h_composed.shape == (200, 32)

        # OR should only hide intersection {sex}
        composed = compose_or(purpose_a, purpose_b)
        assert set(composed.disallowed_attrs) == {"sex"}
