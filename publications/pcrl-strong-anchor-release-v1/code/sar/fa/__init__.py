"""Family FA (Investigator A, campaign pcrl_strong_anchor_release_v1): deployment interface for sar.deploy.

predict(config_id, version, X, feature_names, draw=0, keys=None) -> {'income': (n, 2), 'occupation': (n, 5)} float64 (deterministic)
model_paths(config_id, version) -> the configuration's own model files; frozen_inputs(version) -> SUCC's frozen files read by predict
Routing by config id: FA_TFRB_R2* -> sar.fa.tfrb (revision R2); FA_TFRO_R1* -> sar.fa.tfro (revision R1); every other FA id ->
sar.fa.tfr (FA_TFR_v1). Earlier ids keep their modules unchanged. Imported lazily so that `python -m sar.fa.<module>` runs without a
double import.
"""


def _module(config_id):
    cid = str(config_id)
    if cid.startswith('FA_TFRB_R2'):
        from . import tfrb as m
    elif cid.startswith('FA_TFRO_R1'):
        from . import tfro as m
    else:
        from . import tfr as m
    return m


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    return _module(config_id).predict(config_id, version, X, feature_names, draw=draw, keys=keys)


def model_paths(config_id, version):
    return _module(config_id).model_paths(config_id, version)


def frozen_inputs(version):
    from . import tfr
    return tfr.frozen_inputs(version)
