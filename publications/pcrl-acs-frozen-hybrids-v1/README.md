# PCRL: ACS frozen hybrids (exploratory development follow-up)

**Stage: prefit receipt.** This commit is pushed to GitHub after the analysis lock was committed (research commit `88c86cf57`) and before any recovery reader was fit.

The study tests a fixed composition of already-fitted models from the ACS protected-tree replication (source publication `f4ea199`): **H_LJ = LOCAL_TREE income output + JOINT_TREE occupation output**, paired within each model seed. All four combinations (H_LL, H_LJ, H_JL, H_JJ) are reported, alongside the eligible neural control NEURAL_RAW and the matched control TASK_REFIT E20.

This is an exploratory development follow-up of already-opened DEV_REPORT results. It is not confirmation, and the source labels are unchanged. The composition was chosen after its components' results were public, so the error rates are nominal. A public receipt does not make already-exposed data confirmatory.

- `prefit/ANALYSIS_LOCK.json`: the binding protocol. It contains the K = 38 primary and K = 22 secondary rosters, the three claims, the controls, the settled outcomes and the code hashes.
- `prefit/PROTOCOL.md`: a plain-language summary.
- `prefit/SOURCE_ADMISSION.json`, `prefit/FROZEN_BANK_MANIFEST.json`, `prefit/DEPLOYMENT_RESULTS.json`: source pinning, the hashed bank, and the label-free deployment parity check.
- `prefit/STATISTICAL_REVIEW_PRELOCK.md`, `prefit/FORECASTS.json`, `prefit/LOCK_NOTES.md`: the pre-lock review, the registered forecasts, and the process notes (including one timestamp correction).
- `code/acs_hybrids/`: the campaign code.

No people, labels, predictions, weights, packages, absolute paths or credentials are included.
