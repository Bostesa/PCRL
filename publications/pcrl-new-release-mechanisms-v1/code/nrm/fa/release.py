"""Deployment interface of family FA (nrm.deploy contract; PCRL campaign `pcrl_new_release_mechanisms_v1`).

predict(config_id, version, X, feature_names, draw=0, keys=None) -> {'income': (n, 2), 'occupation': (n, 5)} and
model_paths(config_id, version) -> [MODEL.pkl, MODEL.json] are those of nrm.fa.prc: deterministic, reading only the configuration's
own model files; per-request keys and draws other than 0 are refused.
"""
from .prc import model_paths, predict  # noqa: F401

CODE_MODULES = ('nrm.fa', 'nrm.fa.prc', 'ohn.fam_a')
