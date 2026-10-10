"""Frozen-latent, hard-bit D repair implementation.

Private row arrays are supplied by admitted callers; this package never loads
mixed-role data on import.
"""

from .model import FixedScaler, HardCodeModel, HardCodeRelease
from .fit import FitData, TrainConfig, FitResult, fit_arm, multiscale_mmd_terms

__all__ = [
    "FixedScaler", "HardCodeModel", "HardCodeRelease", "FitData",
    "TrainConfig", "FitResult", "fit_arm", "multiscale_mmd_terms",
]
