"""Family FB (investigator B; starting release NEURAL_RAW). Deployment interface used by sar.deploy (`family_module('FB')`):
    predict(config_id, version, X, feature_names, draw=0, keys=None) -> {'income': (n, 2), 'occupation': (n, 5)} float64
    model_paths(config_id, version) -> the MODEL.pkl / MODEL.json files predict reads
FB_PFD_v1's configurations (FB_PFD, FB_PFD_TASK, FB_PFD_DQ_*, FB_PFD_RAWJ_*) are deployed by sar.fb.pfd.predict, unchanged; revision
R1's configurations (FB_PFO, FB_PFO_TASK, FB_PFO_DQ_HI, FB_PFO_RAWJ_HI; card FB_PFO_R1) by sar.fb.pfo.predict. Both are deterministic
float64 numpy forward passes of the saved weights (numpy only, so python3 can run them; training needs the torch venv). Imports are
deferred so that `python -m sar.fb.pfd` / `-m sar.fb.pfo` do not load a module twice."""


def _module(config_id):
    if str(config_id).startswith('FB_PFO'):
        from . import pfo as mod
    else:
        from . import pfd as mod
    return mod


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    return _module(config_id).predict(config_id, version, X, feature_names, draw=draw, keys=keys)


def model_paths(config_id, version):
    return _module(config_id).model_paths(config_id, version)
