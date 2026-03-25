"""Training modules for PCRL."""

from pcrl.training.losses import (
    FairnessPenalty,
    LossBreakdown,
    PCRLLossModule,
    VerificationRegularizer,
    adversarial_loss,
    compute_multi_auditor_loss,
    compute_multi_task_loss,
    pcrl_loss,
    task_loss,
)
from pcrl.training.trainer import (
    EpochMetrics,
    PCRLTrainer,
    TrainerConfig,
    TrainingState,
)

__all__ = [
    # Loss functions
    "task_loss",
    "adversarial_loss",
    "pcrl_loss",
    "compute_multi_task_loss",
    "compute_multi_auditor_loss",
    "LossBreakdown",
    "PCRLLossModule",
    "FairnessPenalty",
    "VerificationRegularizer",
    # Trainer
    "PCRLTrainer",
    "TrainerConfig",
    "TrainingState",
    "EpochMetrics",
]
