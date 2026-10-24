"""Replacement family FS (investigator B; reading FS_CPI, START = SUCC). Deployment interface used by sar.deploy (`family_module('FS')`):
    predict(config_id, version, X, feature_names, draw=0, keys=None) -> {'income': (n, 2), 'occupation': (n, 5)} float64
    model_paths(config_id, version) -> the MODEL.pkl / MODEL.json files predict reads (SUCC's frozen files are hash-checked inside)
Every FS_CPI* configuration is deployed by sar.fs.cpi.predict (numpy / scikit-learn only: python3 can run it; training needs the torch
venv). The import is deferred so that `python -m sar.fs.cpi` does not load the module twice."""


def _module(config_id):
    if str(config_id).startswith('FS_CPI'):
        from . import cpi as mod
        return mod
    raise ValueError(f'{config_id}: no FS family module for this configuration')


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    return _module(config_id).predict(config_id, version, X, feature_names, draw=draw, keys=keys)


def model_paths(config_id, version):
    return _module(config_id).model_paths(config_id, version)
