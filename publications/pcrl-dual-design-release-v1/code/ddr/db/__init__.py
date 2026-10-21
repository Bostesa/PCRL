"""Family DB (Designer B) of the PCRL dual-design release campaign: deployment interface for ddr.deploy (lazy, so that
`python3 -m ddr.db.ptc` does not import its own module twice)."""
CODE_MODULES = ('ddr.db.ptc', 'ddr.da.std', 'ddr.famkit', 'nrm.fa.prc', 'frc.fa.route', 'ohn.fam_a')


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    from .ptc import predict as _p
    return _p(config_id, version, X, feature_names, draw=draw, keys=keys)


def model_paths(config_id, version):
    from .ptc import model_paths as _m
    return _m(config_id, version)
