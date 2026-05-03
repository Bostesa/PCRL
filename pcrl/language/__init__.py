"""BIOS-medium (text) language extension for PCRL.

All code here is additive. It does not modify any existing tabular code paths,
including the proxy-Lagrangian solver, V2 trainer, evaluation pipeline, or the
diabetes per-class OvR constraint path. Reuses
``pcrl.training.proxy_lagrangian``, ``pcrl.training.losses.VerificationRegularizer``,
and ``pcrl.models.task_head.TaskHead`` by import only.
"""
from .bert_encoder import (
    CLS_DIM,
    LAYER11_OUTPUT_DENSE_NAME,
    BertWithLoRA,
    build_lora_config,
    count_param_breakdown,
)
from .bios_dataset import (
    BIOS_LABELS,
    BIOS_TOP10,
    BIOS_TOP10_IDS,
    build_bios_loaders,
)
from .diagnostics import (
    bio_length_by_gender,
    cls_shape_trace,
    construction_r2,
    describe_modules,
)
from .dual_controllers import EmaCrossCovPIController, FiveSignalMonitor
from .hsic import gender_one_hot, hsic_unbiased_linear, nhsic_linear
from .leace_warmstart import leace_warm_start_bert
from .online_leace import OnlineLeaceRefit
from .tpr_gap import (
    theil_adjusted_r2,
    tpr_gap_summary,
    tpr_gaps_per_occupation,
)

__all__ = [
    "CLS_DIM",
    "LAYER11_OUTPUT_DENSE_NAME",
    "BertWithLoRA",
    "build_lora_config",
    "count_param_breakdown",
    "BIOS_LABELS",
    "BIOS_TOP10",
    "BIOS_TOP10_IDS",
    "build_bios_loaders",
    "bio_length_by_gender",
    "cls_shape_trace",
    "construction_r2",
    "describe_modules",
    "EmaCrossCovPIController",
    "FiveSignalMonitor",
    "gender_one_hot",
    "hsic_unbiased_linear",
    "nhsic_linear",
    "leace_warm_start_bert",
    "OnlineLeaceRefit",
    "theil_adjusted_r2",
    "tpr_gap_summary",
    "tpr_gaps_per_occupation",
]
