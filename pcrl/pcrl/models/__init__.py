"""Model modules for PCRL."""

from pcrl.models.auditor import (
    Auditor,
    AuditorConfig,
    AuditorPool,
    GradientReversalFunction,
    MultiAttributeAuditor,
    PurposeAuditors,
    gradient_reversal,
)
from pcrl.models.encoder import (
    EncoderWithProjection,
    FiLMLayer,
    PurposeConditionedEncoder,
)
from pcrl.models.task_head import (
    MultiTaskHead,
    PurposeTaskHeads,
    TaskHead,
)

__all__ = [
    # Encoder
    "PurposeConditionedEncoder",
    "EncoderWithProjection",
    "FiLMLayer",
    # Task heads
    "TaskHead",
    "MultiTaskHead",
    "PurposeTaskHeads",
    # Auditors
    "Auditor",
    "AuditorConfig",
    "AuditorPool",
    "MultiAttributeAuditor",
    "PurposeAuditors",
    "GradientReversalFunction",
    "gradient_reversal",
]
