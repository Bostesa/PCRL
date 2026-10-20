"""Deployment interface of family FA (frc.deploy contract; PCRL campaign `pcrl_finite_routing_comparison_v1`).

predict(config_id, version, X, feature_names, draw=0, keys=None) -> {'income': (n, 2), 'occupation': (n, 5)} and
model_paths(config_id, version) -> [MODEL.pkl, MODEL.json] are those of frc.fa.route: deterministic, reading only the configuration's
own model files; per-request keys and draws other than 0 are refused.
"""
from .route import model_paths, predict  # noqa: F401

CODE_MODULES = ('frc.fa', 'frc.fa.route', 'nrm.fa', 'nrm.fa.prc', 'ohn.fam_a')
