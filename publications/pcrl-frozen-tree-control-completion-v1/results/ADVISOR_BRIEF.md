# Advisor brief: frozen L_TREE, completed controls, and confirmation feasibility

**Recommendation:** `ORIGINAL_CONFIRMATION_CRITERION_NOT_PRACTICALLY_FEASIBLE` + `NO_COMPATIBLE_UNEXPOSED_POPULATION_IDENTIFIED`.

Neither label authorizes opening new labels. The original labels of the source study are unchanged.

**What the candidate demonstrably achieves (exploratory, on already-used Adult people).**
- **The release:** L_TREE (CatBoost depth-4 teachers plus local leaf-score privacy corrections, ρ=1, epoch 20) releases only probabilities and decisions.
- **Versus the incumbent:** it lowers linked-coalition SEX recovery by 0.0505 [0.0257, 0.0754] relative to RAW_PROJ.
- **Utility:** all 24 original point utility caps pass. 12 occupation bounds remain unresolved.
- **Versus the historical score codes:** 0.014–0.018, unresolved.
- **Frozen and reproducible:** the frozen packages, re-hashed and re-run, reproduce the original releases bitwise. A rehearsal of the confirmation pipeline reproduced all 30 original clause statistics (within 1e-12; 28 bitwise).

**Do the missing controls explain its advantage?** No.
- **Task-only continuation at the same epoch (T_REFIT E20).** Its recovery is indistinguishable from the teacher's (coalition AUC 0.875 / 0.866 vs 0.877 / 0.868; difference 0.0019 [−0.0024, 0.0065], descriptive). L_TREE is 0.085 [0.065, 0.105] lower, so privacy training, not continued task fitting, produces the reduction.
- **Freshly fitted, equally exposed neural RAW-J.** Same people, same SEX access, identical online critics and strength grid. It could protect only slightly before breaking the utility caps: its only feasible point was ρ=1, epoch 2, and even that point passes only 20/24 INNER point caps (0/24 under simultaneous bounds). L_TREE beats it by 0.066 [0.045, 0.088], beyond the 0.020 margin.
- **Official FARE (access-honoring: fair tree on UPDATE only).** It reaches much lower recovery than L_TREE (coalition AUC 0.69; C4 is an established shortfall for L_TREE, −0.091 [−0.121, −0.061]). But it fails utility badly (occupation accuracy 0.444; 4/24 points), so it is reported descriptively and is not counted as beaten.
- **The remaining gap.** The gap to the historical codes is untouched: those codes had more SEX-labelled training data and historical INNER selection, so the comparison does not isolate algorithm quality.

**Corrected interpretations.**
- **Contract share is not identified.** The share of the margin due to the narrowed probability+decision contract is NOT_IDENTIFIED. Earlier wording that "part of the margin reflects the contract" is withdrawn.
- **The joint-vs-local comparison is between selected strategies at different checkpoints**, not a test of the coalition term.
- **H_DPC_JOINT stays a required comparator.** It passes only 16/24 INNER points; that is disclosed, and it is not deleted.

**What claim could a fresh study realistically test?**
- **The original criterion:** ≥0.020 against every strong comparator. It cannot be confirmed at any size if the true code margin is ≤0.020, which is where the observed 0.014–0.018 lies.
- **A clearly narrower, new claim:** original utility caps, plus ≥0.020 over the incumbent, plus superiority over the best code. This is testable in principle. It means "materially better than the incumbent and better than the best code control by some positive margin"; it is not the 0.020 competitive-margin claim. Its choice of strongest code (H_SEQ21_CODE) used exposed INNER outcomes, which must be disclosed if it is ever pre-registered. Choosing it would be a new decision, not a pass of the original.

**Required sample size (claim B).**
- About **317,000 independent evaluation groups** (MC 95% 312k–321k) for 80% joint power (423,000 for 90%), with one pre-registered IUT at one-sided α = 0.025. About 641,000 with the original simultaneous bounds.
- **What drives it:** the seed-1 occupation summed-Brier clause, with slack 0.000775 against the 0.005 cap. Privacy alone needs only about 4,000 groups.
- **Sensitivity:**
  - 0.001 less true Brier excess: about 77,000 groups.
  - 0.0005 more: about 2.5 million.
  - A population with a more balanced SEX mix: not achievable once the minority share exceeds about 0.36, because the excess is concentrated in the minority class.
- **Selection makes this optimistic.** Under a flat prior around the development estimates, the probability that every clause truly passes is about 0.38, and average power is only about 0.31 at 317,000 groups (`power/ASSURANCE_CLAIM_B.json`). For claim A the probability is 0.05.

**What remains unavailable.**
- **A compatible unexposed population.** All 48,842 Adult records are used or excluded. 317,000 exceeds all of Adult. ACS can only host a recipe replication with transported definitions: California 2016–2018 is spent, TX 2018 is reserved by another plan, and NY 2018 is only that plan's conditional fallback.
- **Runnable U0/Ucal reference packages for new people.** These must be bound before any fresh run.
- **A utility-feasible FARE configuration** within the registered grid.

**Exact next executable action.** None that opens labels. The decision is yours:
- **(a)** Close frozen-release confirmation and report L_TREE as exploratory development evidence (manuscript handoff provided).
- **(b)** Authorize a label-blind ACS admission (household counts and a schema crosswalk lock only) for unreserved state-years, as groundwork for a separately pre-registered recipe replication. It would be a new transported benchmark, not confirmation.

No parameter sweep is proposed, and no win is promised.
