"""Evaluation modules for PCRL."""

from pcrl.evaluation.cvr import (
    CVRMetric,
    CVRResult,
    LinearProbe,
    MLPProbe,
    ProbeConfig,
    ProbeSuite,
    compliance_violation_rate,
    evaluate_all_purposes,
    train_probe,
)
from pcrl.evaluation.probes import (
    PCRLEvaluation,
    PostHocProber,
    ProbeResult,
    ProbeResults,
    RepresentationSimilarity,
    TaskPerformance,
    centered_kernel_alignment,
    cosine_similarity_batch,
    evaluate_pcrl,
    run_full_probe_evaluation,
    verify_purpose_differentiation,
)

__all__ = [
    # CVR
    "CVRMetric",
    "CVRResult",
    "evaluate_all_purposes",
    "compliance_violation_rate",
    "train_probe",
    "ProbeSuite",
    "ProbeConfig",
    "LinearProbe",
    "MLPProbe",
    # Probes
    "PostHocProber",
    "ProbeResult",
    "ProbeResults",
    "run_full_probe_evaluation",
    # Evaluation
    "evaluate_pcrl",
    "PCRLEvaluation",
    "TaskPerformance",
    "RepresentationSimilarity",
    # Similarity measures
    "centered_kernel_alignment",
    "cosine_similarity_batch",
    "verify_purpose_differentiation",
]
