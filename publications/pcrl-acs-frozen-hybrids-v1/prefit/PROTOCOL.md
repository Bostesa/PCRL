# Protocol: ACS frozen hybrids (exploratory development follow-up)

The binding record is `ANALYSIS_LOCK.json`, committed before any recovery reader was fit. This page summarizes it.

**Status.** This is an exploratory development follow-up of the already-opened ACS DEV_REPORT results. It is not confirmation, not an independent replication, and not a rescue of the source claim. The source labels stay as published: `RECIPE_EFFECT_NOT_ESTABLISHED`, `NOT_SUPERIOR_TO_ALL_ELIGIBLE`, `SUBGROUP_UNRESOLVED`, benchmark `SUPPORTED`. H_LJ was chosen after its components' DEV_REPORT results were public, so every error rate is nominal.

**Question.** Does a fixed composition of existing models do three things:
- keep the point utility expected from its components;
- reduce coalition SEX recovery materially below continued task fitting (TASK_REFIT E20);
- beat the eligible neural control (NEURAL_RAW ρ1.0 E2)?

The composition is H_LJ: LOCAL_TREE's income output plus JOINT_TREE's occupation output, paired within each of the two model seeds. Each question is answered separately, then together.

**What is fixed.**
- **Source.** Publication `f4ea199`, admitted in `SOURCE_ADMISSION.json`. Every stored output was recomputed from its frozen model. Raw permitted fields re-encode to the admitted features.
- **Bank** (`FROZEN_BANK_MANIFEST.json`). The four combinations H_LL, H_LJ, H_JL and H_JJ, plus the controls NEURAL_RAW, TASK_REFIT and T_TREE (the teacher, diagnostic only). Every recipient array is bitwise its source output, each swap changes only the intended recipient, and the bank is hashed.
- **Primary.** H_LJ, never replaced after results.
- **Deployment.** The runnable H_LJ package reproduces its outputs bitwise from raw permitted fields, and was checked before the lock (`DEPLOYMENT_RESULTS.json`).

**Readers.** The readers use the source's seven-reader slate: linear, MLP and boosted-tree readers on probabilities, log-probabilities, decisions and confidence.
- **Data roles.** Readers are trained on ATTACK_FIT with SEX. Orientation and winners are chosen on ATTACK_VALIDATION only. DEV_REPORT is scored from released outputs alone.
- **Seeds.** One reader seed per model seed, the same for every service.
- **Fits.**
  - Marginal readers are fitted once per unique single-recipient wire (70 per seed).
  - Fresh pair readers are fitted for every service (49 per seed).
  - Pair candidates include the 14 ignore-one-recipient aliases.
  - The null and planted-leak controls run on the H_LJ wire.
  - In total there are 125 fits per seed, 250 in all.

**Inference** (inherited from the source).
- **Bootstrap.** One household bootstrap over DEV_REPORT: B = 2,048, source seed 2026101603, shared across every service, endpoint and both seeds. Bounds are Wald-type with t(3954).
- **Decision family.** Three claims, each an intersection-union test at one-sided α/3 = 0.00833, with a family error of 0.025:
  - **A.** H_LJ coalition benefit vs TASK_REFIT exceeds 0.020 on both seeds, with no recipient worsening above 0.010, and all 24 utility clauses pass at the point.
  - **B.** H_LJ coalition recovery is below NEURAL_RAW's on both seeds, with a nested 0.020 target.
  - **C.** The four occupation-Brier subgroup guards.
- **Reporting families.** K = 38 primary simultaneous intervals (t = 3.2148), the only source of failing-direction labels. K = 22 secondary descriptive intervals (t = 3.0541).

**Settled before readers.** These outcomes follow from published source values and are not findings of this campaign:
- H_LJ passes all 24 utility clauses at the point; 15 of 24 pass at their clause bounds.
- Claim C is SUBGROUP_UNRESOLVED.
- The income worsening clauses pass.

Only six primary endpoints carry new information: the two coalition benefits, the two neural-control comparisons and the two increments over H_LL.

**Controls** test the pipeline only.
- Every planted-leak AUC must be at least 0.99, and every shuffled-null AUC must lie in [0.45, 0.55].
- Stage 1 (ATTACK_VALIDATION) runs before truth is read. Stage 2 (DEV_REPORT) runs at the reading.
- A failure invalidates claims A and B. It is never evidence of privacy.

**Prohibitions.**
- No defense-model fitting, calibration, strength search, averaging, routing or new population.
- No reader sees MODEL_SELECT, DEV_REPORT truth or sealed people.
- The Florida sealed assessment stays closed: the extractor is not rerun and the raw census file is not read.

**Resources.** 6 CPU-hours, 6 hours elapsed, $0, two heavy jobs at once, one thread each, 4 GiB combined RSS.
