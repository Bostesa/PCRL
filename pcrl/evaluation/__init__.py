"""Evaluation modules for PCRL."""

from pcrl.evaluation.certificates import (
    ComplianceReport,
    EmpiricalAudit,
    LinearAudit,
    generate_report,
    print_compliance_table,
)
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
from pcrl.evaluation.mine import (
    MINEResult,
    MINEstimator,
    StatisticsNetwork,
)
from pcrl.evaluation.visualize import (
    plot_certificate_heatmap,
    plot_cvr_comparison,
    plot_purpose_separation,
    plot_task_privacy_tradeoff,
)

__all__ = [
    # Certificates
    "ComplianceReport",
    "LinearAudit",
    "EmpiricalAudit",
    "generate_report",
    "print_compliance_table",
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
    # MINE
    "MINEstimator",
    "MINEResult",
    "StatisticsNetwork",
    # Visualization
    "plot_purpose_separation",
    "plot_cvr_comparison",
    "plot_certificate_heatmap",
    "plot_task_privacy_tradeoff",
]
