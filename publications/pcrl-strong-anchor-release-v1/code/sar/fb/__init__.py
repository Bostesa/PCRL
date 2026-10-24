"""Family FB (investigator B; starting release NEURAL_RAW). Deployment interface used by sar.deploy (`family_module('FB')`):
    predict(config_id, version, X, feature_names, draw=0, keys=None) -> {'income': (n, 2), 'occupation': (n, 5)} float64
    model_paths(config_id, version) -> the MODEL.pkl / MODEL.json files predict reads
Every FB configuration (FB_PFD, FB_PFD_RAWJ, FB_PFD_TASK) is deployed by sar.fb.pfd.predict: a deterministic float64 numpy forward pass
of its saved weights (numpy only, so python3 can run it; training needs the torch venv). The import is deferred so that
`python -m sar.fb.pfd` does not load the module twice."""


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    from .pfd import predict as _predict
    return _predict(config_id, version, X, feature_names, draw=draw, keys=keys)


def model_paths(config_id, version):
    from .pfd import model_paths as _model_paths
    return _model_paths(config_id, version)
