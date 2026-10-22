"""Family IB (Designer B: coordinated cell-targeted selective flattening, CSF-T) of the PCRL input-and-pair release campaign: deployment interface for
ipr.deploy (lazy, so that `python3 -m ipr.ib.csf` does not import its own module twice)."""
CODE_MODULES = ('ipr.ib.csf', 'ipr.famkit', 'ipr.paths', 'ipr.ctlkit', 'ipr.ctl.twin', 'nrm.fa.prc', 'ohn.fam_a')


def predict(config_id, version, X, feature_names, draw=0, keys=None):
    from .csf import predict as _p
    return _p(config_id, version, X, feature_names, draw=draw, keys=keys)


def model_paths(config_id, version):
    from .csf import model_paths as _m
    return _m(config_id, version)
