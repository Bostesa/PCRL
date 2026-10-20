"""Family CTL deployment interface (nrm/deploy.py contract), exported from nrm.ctl.run:

    predict(config_id, version, X, feature_names, draw=0, keys=None) -> {'income': (n, 2), 'occupation': (n, 5)} float64
        deterministic: the base family's prediction from admitted features, then the saved CTL_REM remainder flattening; keys and
        any draw other than 0 are refused.
    model_paths(config_id, version) -> [CTL_PARAMS.npz of the configuration] + the base configuration's model paths.
    CODE_MODULES -> nrm.ctl.post, nrm.ctl.run, nrm.ctl.release, the base family module and its CODE_MODULES.
"""
from __future__ import annotations

from .run import code_modules, model_paths, predict  # noqa: F401

FAMILY = 'CTL'


def __getattr__(name):
    if name == 'CODE_MODULES':
        return code_modules()
    raise AttributeError(name)
