"""Challenger controls for pcrl_new_release_mechanisms_v1 (generic, family-independent).

Modules:
  nrm.ctl.post      post-processing paths applied to ANY family's task-only twin vectors:
                    temperature flattening, label smoothing, SEX-blind cell coarsening (aIB merge path),
                    SEX-aware / shuffled-SEX / random-merge grouping, and a random replacement channel
                    without task adaptation.
  nrm.ctl.frontier  utility-recovery frontier comparison of a candidate against a control path
                    without demanding an exact cost match.

Nothing in this package loads data or labels. Callers pass arrays obtained through the logged access layer under
their own registered operation. Every fitting call takes the role names of its arrays and refuses any role other
than MODEL_FIT / CALIBRATION (never MODEL_SELECT or a closed role). Selftests use invented arrays only:
  python3 -m nrm.ctl.post selftest
  python3 -m nrm.ctl.frontier selftest
"""
