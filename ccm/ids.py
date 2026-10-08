"""Shared paths of the confidence-constrained mechanism sprint (ccm). No data is loaded here.

Private store: PCRL_CCM_PRIVATE_CACHE (default <HOME>/PCRL_eval_cache_private/ccm_v1); tracked files use
<PRIVATE_CACHE>/ccm_v1. The hcal store is read-only input (admitted teachers and frozen calibration records).
"""
from __future__ import annotations

import os
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
REL = "results/pcrl_confidence_constrained_mechanism_v1"
PKG = WT / REL
BRANCH = "research/pcrl-confidence-constrained-mechanism-v1"
STUDY = "pcrl_confidence_constrained_mechanism_v1"
PRIV = Path(os.environ.get("PCRL_CCM_PRIVATE_CACHE") or (Path.home() / "PCRL_eval_cache_private" / "ccm_v1"))
RUN = PRIV / "run"
UNITS = RUN / "units"
HCAL_PRIV = Path.home() / "PCRL_eval_cache_private" / "hcal_v1"
SOURCE_TIP = "1baf5bbdaf59cfa6a664cae07d652a4712bdfdc5"
SOURCE_EVIDENCE = "159537dcb61d030c870ae42e7addff4796651723"
SEEDS = (0, 1, 2)
RECIPIENTS = (1, 2)            # 1 = income (K = 2), 2 = occupation (K = 6)
