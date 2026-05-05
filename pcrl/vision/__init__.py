"""CelebA-medium PCRL pipeline (vision branch).

Isolated from the tabular v2 trainer. Reuses ``pcrl.training.proxy_lagrangian``
and ``pcrl.training.independence.vicreg`` by import only. The locked tabular
files (v2_trainer, proxy_lagrangian, vicreg, evaluation) are not modified.
"""
