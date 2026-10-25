"""Family FAM of the PCRL reader-fit-aware release campaign (`pcrl_reader_fit_aware_release_v1`; cards with prefix RF_; release
directory runs/FAM/). EXPLORATORY DEVELOPMENT on previously used public-use ACS 2019 1-Year PUMS Florida person records; not
confirmation.

Deployment (numpy only, python3): every saved configuration of this family is the pair of U0-architecture networks (d-64-64-16-K)
stored as float32 weights in its MODEL.pkl; `predict` is their float64 forward pass from the six permitted fields' admitted encoding.
No SEX, task label, person identifier, reader state or routing key is an input or an output.

    rfa.fam.net     the deployment forward pass (verbatim copies of sar.fb.pfd.np_forward / layers_of / release_weights)
    rfa.fam.rfit    the reader-fit-aware episode objective (mechanism investigator)
    rfa.fam.fit     the training loops, cross-fitting, the MODEL_FIT selection rule, the control brackets and the saved configurations
"""
from .net import predict, model_paths  # noqa: F401
