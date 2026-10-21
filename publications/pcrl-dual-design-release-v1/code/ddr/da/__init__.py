"""Family DA (Designer A) of the PCRL dual-design release campaign: deployment interface for ddr.deploy (lazy, so that
`python3 -m ddr.da.std` does not import its own module twice)."""
CODE_MODULES = ('ddr.da.std', 'ddr.famkit', 'nrm.fa.prc', 'ohn.fam_a')


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    from .std import predict as _p
    return _p(config_id, version, X, feature_names, draw=draw, keys=keys)


def model_paths(config_id, version):
    from .std import model_paths as _m
    return _m(config_id, version)
