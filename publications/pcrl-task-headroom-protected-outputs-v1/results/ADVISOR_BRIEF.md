# Advisor brief: task headroom and protected prediction outputs

**One line:** stronger boosted-tree task models gave real headroom. Local privacy training of their leaf scores spent it to cut coalition SEX recovery by 0.050 AUC against the incumbent, within the original point utility caps. That margin is established. Against the strongest historical score codes the gain (0.014–0.018) is not established, and joint design added nothing.

The label is `POINT_FEASIBLE_NO_COMPETITIVE_CRITERION`. All evidence is exploratory on repeatedly used Adult development people.

**Did the new task learner actually create headroom, and how much?**
Yes.
- **On the internal selection role**, against an equally exposed neural model, CatBoost lowered NLL by 0.027 nats for income [0.022, 0.032] and 0.014 for occupation [0.009, 0.019]. The occupation target of 0.010 was met at the point estimate, not the lower bound.
- **On the reporting role**, against the original U0 reference, the teacher was 0.042 (income) and 0.013 (occupation) nats better.
- **The cost:** the teacher also leaked more than the incumbent (coalition AUC 0.872 vs 0.836).

**Did privacy training spend that headroom effectively?**
For the local arm, mostly yes.
- **Recovery:** L_TREE cut coalition recovery from 0.872 to 0.785, and individual recovery to 0.651 (income) and 0.737 (occupation).
- **Occupation cost:** NLL ended 0.002 above U0. That used essentially all of the occupation headroom.
- **Income cost:** NLL stayed 0.031 below U0.
- **Utility clauses:** all 24 point estimates pass. 12 occupation clauses are unresolved under simultaneous bounds (upper bounds up to 0.032), and none is violated.
- **Stronger pressure:** every family nominee is at ρ=1. Stronger strengths were feasible only at the earliest checkpoint. The ρ=10 epoch-20 diagnostic reached 0.659 but broke occupation utility.

**Did the output become harder for fresh recipients to exploit?**
Yes against the incumbent and the task-only models:

| L_TREE vs | coalition benefit | margin (0.020) cleared? |
|---|---|---|
| RAW_PROJ | 0.0505 [0.026, 0.075] | yes |
| neural reference | 0.080 | yes |
| own teacher | 0.087 | yes |

No recipient got worse against any of them. Against the three historical score codes, the benefits are 0.018, 0.015 and 0.014, all with intervals crossing 0, so this is not established. FARE reads lower recovery (0.684), but it fails utility and is not counted as beaten or as beating.

**Did joint design add anything beyond local and sequential training?**
No.
- The joint arm's coalition recovery was 0.016 higher than its local twin's, unresolved [−0.031, 0.000], and its occupation recovery was 0.036 higher (established).
- Against the stronger local control and the S12 sequential order it was not distinguishable (unresolved).
- Its income recovery excess was unresolved against L_TREE (0.018 [0.004, 0.032]) and S12 (0.019 [0.005, 0.033]).
- The FIT2-selected primary is the local recipe; the coalition comparison with its local twin is unresolved.

**What is established, unresolved, untested or incomplete?**
- **Established:**
  - task headroom;
  - the margin over RAW_PROJ, N_EQ and the teacher;
  - INNER point feasibility;
  - privacy updates move the transmitted outputs.
- **Unresolved:**
  - the gap to the historical codes;
  - occupation utility under simultaneous bounds.
- **Untested:**
  - other attributes;
  - stronger adversaries (model access, repeated queries, insiders);
  - imperfect linkage;
  - another population;
  - reusable feature services;
  - fresh confirmation people.
- **Incomplete:**
  - FARE was not refit on this pool (it was attacked from its stored release);
  - the confidence-constrained mechanism was never fitted;
  - off-device backup is pending.

**What runnable model exists at the end?**
- **Packages:** 8 numpy-only packages (raw83 → probabilities + decision), each bitwise equal to the attacked INNER release and passing 13/13 refusal checks. All 8 were re-run after a same-device restore from the versioned backup (`BACKUP_VERIFICATION.json`); off-device backup is pending.
  - teacher T_TREE;
  - primary L_TREE;
  - strongest matched control S12_TREE;
  - J_TREE;
  - all for seeds 0 and 1.
- **How to run:** see `QUICKSTART.md`.

**Caveats:**
- Intervals are conditional on the fitted models and readers and do not cover the development history.
- Both seeds share the same INNER people. FIT2 is an exposed selection role, and Stage A intervals are unadjusted diagnostics.
- The pair view assumes perfect linkage.
- The historical codes had asymmetric exposure: they were trained with SEX on all of FIT and historically selected on INNER.
- Under this narrowed contract, those codes are also INNER point-feasible and already sit about 0.033 below RAW_PROJ (descriptive). Part of the margin over the incumbent therefore reflects the contract.
- S12/S21 are learned sequential controls, not Taylor et al.'s algorithm.

**Next decision (evidence-based):** keep the recipe and comparator set frozen. The only way to resolve the gap to the historical codes is a pre-registered head-to-head on unexposed people, and authorizing those people is your decision. No nearby sweep is proposed.
