"""Purpose specification and management for PCRL."""

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
from pcrl.purposes.spec import PurposeRegistry, PurposeSpec
from pcrl.purposes.verification import (
    CertificateResult,
    LinearComplianceCertificate,
    NonlinearCertificateResult,
    NonlinearComplianceCertificate,
    NullSpaceCertificate,
    certified_accuracy_bound,
)

__all__ = [
    # Spec
    "PurposeSpec",
    "PurposeRegistry",
    # Composition
    "ComposedPurpose",
    "compose_and",
    "compose_or",
    "compose_hierarchy",
    "compose_embeddings_additive",
    "compose_embeddings_max",
    "LearnedComposer",
    "CompositionLoss",
    # Verification
    "LinearComplianceCertificate",
    "NonlinearComplianceCertificate",
    "NonlinearCertificateResult",
    "NullSpaceCertificate",
    "CertificateResult",
    "certified_accuracy_bound",
]
