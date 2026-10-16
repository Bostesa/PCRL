# PCRL: ACS frozen hybrids (exploratory development follow-up)

**Result: `H_LJ_FIXED_COMPOSITION_NOT_SUPPORTED`.** The fixed composition H_LJ joins LOCAL_TREE's income output to JOINT_TREE's occupation output, paired within each model seed.
- **Utility:** it keeps the trees' point utility on all 24 inherited clauses. Simultaneous utility confidence and subgroup suitability are unresolved.
- **vs continued task fitting (TASK_REFIT E20):** it lowers linked-coalition SEX recovery by 0.022 and 0.024 AUC. The clause bounds are 0.012 and 0.014, so the pre-registered 0.020 margin is not established.
- **vs the eligible neural control (NEURAL_RAW ρ1.0 E2):** H_LJ is not superior, and this is the limiting comparison. At the point its coalition AUC is 0.001 and 0.005 higher than the control's.
- **Where the larger reductions come from:** the combinations that use JOINT_TREE's income output, which fails income utility. They are a post-hoc observation, not nominees.
- **Verification:** an independent verifier using separate code passed 10/10 checks.
- **Status:** the fixed hybrid test is closed.

> Exploratory development result on already-opened DEV_REPORT data; the composition was fixed after its components' DEV_REPORT results were public, so error rates are nominal; conditional on the two fitted seed pairs (shared people, not independent samples), the frozen defense models and the fitted readers, in the unweighted DEV_REPORT household superpopulation; not confirmation; source labels unchanged.

**Contents.**
- **`results/START_HERE.md`:** start here; then `results/ADVISOR_BRIEF.md` (the six questions answered plainly).
- **`prefit/`:** the receipts pushed before any reader. The `ANALYSIS_LOCK` came first, in commit `46dc0b8`. The `REPORTING_LOCK` was pushed after readers and before the DEV_REPORT truth re-read, in commit `a5aa204`.
- **`results/`:**
  - all aggregate results: the K = 38 primary and K = 22 secondary endpoint tables, signed distances, and the four-hybrid, service and subgroup tables;
  - the figure;
  - provenance, reader accounting, deployment checks and backup receipts;
  - the independent verification and the claims review.
- **`code/acs_hybrids/`:** the campaign code. **`code/acs_hybrids_independent/`:** the verifier's separate code.

**Source.** Publication `f4ea199` (ACS Florida 2019 protected-tree replication). Its labels are unchanged: `RECIPE_EFFECT_NOT_ESTABLISHED`, `NOT_SUPERIOR_TO_ALL_ELIGIBLE`, `SUBGROUP_UNRESOLVED`. Its verification stays 10/11; D1 was a transport-annotation failure. Its disclosure D2 still applies: the sealed records were parsed transiently in memory. The Florida sealed assessment stayed closed.

**Not included.** People, labels, predictions, reader models, deployment packages, the frozen bank, absolute paths and credentials.
