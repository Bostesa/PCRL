"""Family FA (Investigator A, campaign pcrl_strong_anchor_release_v1): deployment interface for sar.deploy.

predict(config_id, version, X, feature_names, draw=0, keys=None) -> {'income': (n, 2), 'occupation': (n, 5)} float64 (deterministic)
model_paths(config_id, version) -> the configuration's own model files; frozen_inputs(version) -> SUCC's frozen files read by predict
(imported lazily so that `python -m sar.fa.tfr` runs without a double import).
"""


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    from . import tfr
    return tfr.predict(config_id, version, X, feature_names, draw=draw, keys=keys)


def model_paths(config_id, version):
    from . import tfr
    return tfr.model_paths(config_id, version)


def frozen_inputs(version):
    from . import tfr
    return tfr.frozen_inputs(version)
