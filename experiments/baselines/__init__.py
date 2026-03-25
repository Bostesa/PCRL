"""Baseline implementations for PCRL comparison.

This module contains baseline implementations to compare against PCRL:

1. standard_encoder: Same architecture, no adversarial training.
   - Measures what's recoverable from vanilla representations
   - Upper bound on information leakage

2. separate_models: Train separate encoder per purpose.
   - No purpose token, just independent models
   - Comparison for compute cost and fairness achievable

3. adversarial_fair: Standard adversarial fair representation (LAFTR-style).
   - Fixed sensitive attributes, no purpose switching
   - Shows limitation of static fairness approaches
"""
