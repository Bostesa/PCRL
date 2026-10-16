"""Independent verification of the ACS frozen-hybrids follow-up (results/pcrl_acs_frozen_hybrids_v1).

Own metric, AUC, bootstrap, endpoint, critical-value and decision code. Campaign code is used only where the verification
brief allows it: acs_hybrids.common for hash-checked data loading; task_conditioned_representation.attacks.fit_reader /
predict_reader to refit readers; acs_hybrids.hybrid_deploy.release as the deployment artifact under test;
task_headroom.inference.group_weights only to cross-check the verifier's own household resampler.
"""
