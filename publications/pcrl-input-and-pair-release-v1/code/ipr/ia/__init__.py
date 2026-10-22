"""Family IA (Designer A: shared SEX-aware field generalization, SFG) of the PCRL input-and-pair release campaign: deployment interface for
ipr.deploy (lazy, so that `python3 -m ipr.ia.sfg` does not import its own module twice)."""
CODE_MODULES = ('ipr.ia.sfg', 'ipr.famkit', 'ipr.paths', 'ipr.ctlkit', 'ipr.ctl.twin', 'nrm.fa.prc', 'ohn.fam_a')


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    from .sfg import predict as _p
    return _p(config_id, version, X, feature_names, draw=draw, keys=keys)


def model_paths(config_id, version):
    from .sfg import model_paths as _m
    return _m(config_id, version)
