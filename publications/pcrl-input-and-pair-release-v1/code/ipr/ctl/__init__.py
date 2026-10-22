"""Shared controls and the shared twin of the PCRL input-and-pair release campaign (family CTL): deployment interface for
ipr.deploy (lazy, so that `python3 -m ipr.ctl.twin` does not import its own module twice)."""
CODE_MODULES = ('ipr.ctl.twin', 'ipr.famkit', 'nrm.fa.prc', 'ohn.fam_a')


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    from .twin import predict as _p
    return _p(config_id, version, X, feature_names, draw=draw, keys=keys)


def model_paths(config_id, version):
    from .twin import model_paths as _m
    return _m(config_id, version)
