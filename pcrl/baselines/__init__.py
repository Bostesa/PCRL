"""Single-purpose fairness baselines (LAFTR, INLP) for comparison vs PCRL.

These are intentionally simple, unconditioned encoders trained on a single
purpose so we can quote first-purpose Framework D numbers (R²_onehot, R²_DA,
task accuracy) against PCRL on the same metric pipeline.
"""

from pcrl.baselines.laftr import LAFTRDiscriminator, train_laftr
from pcrl.baselines.inlp import INLPEncoder, run_inlp

__all__ = ["LAFTRDiscriminator", "train_laftr", "INLPEncoder", "run_inlp"]
