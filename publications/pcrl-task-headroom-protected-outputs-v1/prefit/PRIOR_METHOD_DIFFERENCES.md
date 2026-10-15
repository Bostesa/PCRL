# Prior methods and what this campaign changes

Campaign: task headroom and protected prediction outputs. Role: adversary/methodology. Adopted at
SCIENCE_LOCK, 2026-10-15, before any fit.

All prior studies are exploratory development on repeatedly used Adult people. Lower SEX AUC is better.
"Pair" means the aligned income + occupation coalition view. The 0.020 target is the required
lower-bound pair benefit.

## 1. Prior studies, one entry each

Paths are under `~/PCRL/.worktrees/<worktree>/results/<dir>/`; HEADs as read.

| Study (worktree @ HEAD) | What it did | Outcome |
|---|---|---|
| Task-conditioned representation (`pcrl-task-conditioned-representations-v1` @ b21c4cc5e) | Separate 83→64→64→16 income/occupation encoders with folded heads, continued jointly with a task-label-conditioned kernel matching SEX locally and on aligned pairs, against local, unconditional, ordinary adversarial (RAW_JOINT) and both sequential controls. | All 24 point utility clauses passed, but pair benefit vs task-only was 0.008 [0.0001, 0.016], a SHORTFALL. Worse than RAW_JOINT by 0.005, with a local income violation (+0.063). Recipe stopped; RAW_JOINT retained as the benchmark. |
| Occupation bottleneck (`pcrl-occupation-bottleneck-v1` @ b0243f6a5) | Diagnostic: projected released features onto the useful heads' row space (ordinary compression) and attacked predictions vs features. No defense fitted (pilot NOT_TRIGGERED). | Prediction leakage and extra-feature leakage both supported. Projected RAW kept utility and cut income recovery by 0.062, but its pair benefit of 0.011 [0.0004, 0.021] stays UNRESOLVED against 0.020. This projection is the origin of the **RAW_PROJ** incumbent. |
| Separate-pool utility guard (`pcrl-separate-pool-utility-guard-v1` @ 968c5e8a7; START_HERE + FEASIBILITY_DECISION, no RESEARCH_DECISION) | Occupation privacy updates accepted only if a whole separate GUARD pool (2,043 people) stayed in budget. | COMPLETE_NEGATIVE_FEASIBILITY. The guard held its own pool, but VALIDATION NLL rose +0.012 to +0.016 and Brier +0.006 to +0.008, over the caps. Epoch 0 was selected; privacy NOT_TESTED. |
| Minimal task-wire adapter (`pcrl-minimal-task-wire-adapter-v1` @ 059c5a60e) | A 325-parameter row-space correction to occupation predictions over a frozen trunk; trust/free and local/joint arms. | POINT_FEASIBLE_NO_COMPETITIVE_CRITERION. Utility passed 24/24 point clauses. Occupation-only recovery fell 0.010 (established, but matched by the local arm). Pair benefit vs RAW_PROJ was 0.0049 [−0.0017, 0.0115], an established shortfall. |
| Two-recipient adapter (`pcrl-two-recipient-task-wire-adapter-v1` @ ee67c29ee) | Adapted both wires (65 income + 325 occupation parameters); T/L/J/S12/S21 arms. | POINT_FEASIBLE_NO_COMPETITIVE_CRITERION. J_BOTH pair benefit vs RAW_PROJ was 0.0036 [−0.0025, 0.0097]; all five comparisons are shortfalls. Changing income did not help the coalition. Only 5/24 simultaneous utility clauses resolved, vs 10/24 for RAW_PROJ. |
| Frozen-bank release (`pcrl-bank-anchored-release-v1` @ dda8e450c) | Kept the useful frozen ten-bit W_TSTAR code bank fixed and learned which code to transmit (T/D/P/L/J/S12/S21). | Most arms kept 24/24 point utility (S21 22/24). The best new pair gain was P at 0.006 vs B0. J did not beat P, and the older P* (0.803) stays stronger. No competitive claim. |
| Confidence capacity (`pcrl-confidence-capacity-v1` @ d0c8a45c8) | Finite k-means confidence codes; asked whether convergence or capacity restores confidence, then added privacy weights. | CONFIDENCE_FEASIBILITY_ESTABLISHED: capacity did it (i8o64), convergence did not. Privacy-trained JOINT λ0.1 cut pair AUC by 0.034, but occupation log-loss upper bound was 0.0121 > 0.010, so NOT_ESTABLISHED. No eligible JOINT nominee. |
| Confidence-budgeted privacy (`pcrl-confidence-budgeted-privacy-v1` @ 7f3ec67b2) | Added a confidence headroom rule to the privacy-trained codes. | CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION. The only eligible nominee was the weakest λ (0.01), with pair benefit 0.0022 ≪ 0.020. |
| Decision-preserving compression (`pcrl-decision-preserving-compression-v1` @ 0a7b05a52) | Tokens with m ≤ 8 states per predicted class, a public decoded vector and the exact decision. | EXPERIMENTAL_NO_ADVANTAGE. Decisions were preserved exactly (20.2M checks), but occupation log loss rose +0.020 nats. JOINT fallback leaked 0.018 less than task-only (< 0.02). Only continuous U met the contract. |
| Held-out calibration (`pcrl-heldout-calibration-v1` @ 1baf5bbda) | Refitted only the probability decoder of frozen releases (per-token, global and per-class temperature) on held-out calibration people. | NO_COMPETITIVE_RELEASE_CRITERION_ESTABLISHED. Temperature beat per-token decoding and tied the original decoder; precision failures on P07/P13. Frozen-decoder line closed. |
| Utility-reference repair (`pcrl-utility-reference-repair-v1` @ ce1308a52) | Fixed the exposure-asymmetric U0-vs-F validation by moving to a common INNER role; then ran a five-mask raw-feature selection screen (T/J/S12/S21). | UTILITY_FEASIBILITY_ESTABLISHED at the point gate (simultaneous 10 PASS / 14 UNRESOLVED / 0 VIOLATION). J picked the same masks as task-only; control minus J was −0.0013 vs +0.020. No privacy or joint lead. No U0/Ucal bug was found. |

**Common pattern.** Every line kept point utility, or nearly did. None reached a 0.020
simultaneous pair benefit over the incumbent within the caps. Coalition recovery stayed near
0.83–0.85 for prediction-wire releases. The binding constraint was occupation confidence (NLL/Brier).

## 2. What this campaign changes

1. **Task learner.** The task predictors are gradient-boosted trees (CatBoost; binary Logloss for
   income, 6-class MultiClass for occupation) trained directly on permitted raw83 inputs. All prior
   protected services used neural encoders and heads, frozen code banks or finite codes.
2. **Family contains the useful teacher exactly.** The correction is
   f_r = f_teacher,r + Σ_t δ[r,t,leaf_t(x)] with δ = 0 at initialization, so T_TREE is an exact
   member. Prior adapters were initialized from a teacher too. Here, however, the teacher is a
   different, possibly stronger, learner class. That is where the "task headroom" comes from: if the
   tree teacher beats U0/Ucal, the caps leave room to spend on privacy.
3. **Release narrowed.** Each recipient gets only its own full-precision probability vector plus
   argmax. There are no features, tokens, codes or metadata. This is narrower than the feature
   releases (task-conditioned, bottleneck, mask screen) and different from the token/code releases
   (bank, confidence codes).
4. **Leaf-score corrections.** The privacy parameters are additive per-leaf logit offsets under fixed
   routing. That gives a capacity set by the number of leaves, piecewise-constant in x, and no change
   to tree structure.
5. **No KL trust region.** The minimal adapter's trust arms (J_TRUST/L_TRUST) are dropped. Utility is
   enforced only by task CE during training and by the original caps at selection.
6. **Equal-exposure neural reference.** A neural arm with the same people, labels and roles is used
   for internal selection only. It is not a claimed competitor and not the incumbent.
7. **Arms.**
   - T_TREE: teacher.
   - T_REFIT: task-only leaf training.
   - L_TREE: local penalties.
   - J_TREE: local + coalition.
   - S12_TREE / S21_TREE: sequential.

   This mirrors the T/L/J/S12/S21 roster of the two-recipient adapter, on a new learner.

## 3. What this campaign reuses

- Recipient roles (income R1, occupation R2, aligned pair), SEX labels and the attacker-fit /
  attacker-validation / INNER group roles.
- Utility caps and references: accuracy loss ≤ 0.010, NLL excess ≤ 0.010 nats, summed Brier excess
  ≤ 0.005, against U0 and Ucal, both tasks and both seeds (24 clauses). Point gate plus simultaneous
  bounds are reported separately.
- The online critic implementation (linear and MLP critics against SEX on each recipient's vector
  and on the pair), as in the task-wire adapters.
- The validated reader model implementations (logistic, MLP 64x32, HGB 15x200 from
  `task_conditioned_representation.attacks.fit_reader`), validation orientation, and null, shuffle
  and planted-leak gates. **Changed:** the slate is re-specified for output-only wires, and the
  primary reading is the attacker-validation winner rather than the INNER envelope maximum (the
  envelope is descriptive).
- Paired duplicate-group bootstrap with Bonferroni Student-t multiplicity, conditional on fitted
  models and readers.
- The RAW_PROJ incumbent (RAW_JOINT strength-0.3 epoch-5 own heads, projected; the primary
  comparator in both adapters), narrowed here to probabilities + decision and freshly re-attacked.
  It is also the competitively trained ordinary RAW-J baseline (strength 0.3 was the best RAW-J in
  the strength studies), so RAW_J is recorded as an exact alias of RAW_PROJ, not refitted.
- Stored historical releases, narrowed and freshly re-attacked: JOINT and SEQ-21 i8o64 lambda 0.1
  score codes, decision-preserving JOINT m8 lambda 0.1, and raw83 FARE c1 with its zero-fairness
  control Z1. Their old AUCs are never imported.
- The criterion: pair benefit ≥ 0.020 vs RAW_PROJ and competitive references, with no recipient
  recovery increase > 0.010.

## 4. Has a boosted-tree task service with leaf-score correction already been run? (search result)

**Answer: no.** No boosted-tree *task predictor with leaf-score privacy correction* exists anywhere
in the project. No CatBoost code exists at all. Boosted trees appear only as readers/attackers and
in two non-privacy-trained ACS contexts.

**Search scope.**

- Both git object stores:
  - main `~/PCRL/.git`: 23 worktrees plus the custody copies, 2,263 commits;
  - `~/PCRL/.worktrees/pcrl-method-portfolio-v1/.git`: 26 worktrees, including the
    adapter, bank, bottleneck, task-conditioned, separate-pool and new task-headroom worktrees,
    2,451 commits.
- Every local and remote ref, with `git grep` over `*.py`, `*.md`, `*.toml`, `*.cfg` and
  `requirements*`.
- A filesystem grep of all worktrees, which covers untracked files.

**Findings.**

1. **Commit messages.**
   `git log --all --oneline | grep -i -E "catboost|boost|tree|leaf|gbdt|xgb"` matches only:
   - `0937a76a6 Prepare bounded independent HGB refit worker` (bank-anchored reader refits);
   - `061bc18b8 Verify representative archive restores without original-tree fallback` (file tree).

   The same two hits appear in both stores. Neither is a task service.
2. **CatBoost and LightGBM.** Zero hits in any ref of either store, and zero in untracked worktree
   files. CatBoost is **not installed** in `~/PCRL/.venv` (`ModuleNotFoundError`). xgboost 3.2.0 is
   installed. The new worktree `pcrl-task-headroom-protected-outputs-v1` (branch
   `research/pcrl-task-headroom-protected-outputs-v1`, base ee67c29ee) has only untracked scheduling
   scaffolding (`task_headroom/{admission,scheduler,host_queue,evaluation_queue}.py`) and no model
   code.
3. **XGBoost.** Used only as an auditor/attacker: `pcrl/models/auditor.py`,
   `experiments/run_reconciliation.py`, `run_cross_purpose_*`, `run_leace_baseline.py`,
   `scripts/run_splince_benchmark.py`, `scripts/crosspurp/run_eval.py`,
   `pcrl/evaluation/diverse_auditors.py`. The durable-guarantees paper also used XGBoost as a Tier-1
   attacker.
4. **sklearn HistGradientBoosting, as reader/attacker or search proxy.** It is the
   "boosted-tree reader" in:
   - `task_conditioned_representation/attacks.py`, `conditional_release/attacks.py`,
     `experiments/prediction_release_attackers.py`, `stored_model_eval/recipes.py`
     (GBT attacker family);
   - `bank_anchored/r5_refit_reader.py`, `incumbent_view/r5_reader_refit*.py`, `jcv/audit.py`,
     `rgj/audit.py`;
   - every `results/*/verification/replay_*.py`;
   - the SEX search proxy in `results/pcrl_capacity_matched_feature_selection_v1/m2/M2_METHOD_SPEC.md`
     step 5 and `results/pcrl_utility_reference_repair_v1/r2/raw_search.py` (@81178e149).

   Other uses that are not task services:
   - `results/redesign_20260907_prediction_audit_v1/PROTOCOL.md:58` (@5f162ab37): an HGB *regressor
     probe*.
   - `experiments/pcrl_shared_context_release_v1/policies.py` (@e2f3fba03): an HGB regressor as a
     cost-regression oracle inside the release policy, on ACS.
   - `pcrl_task_directed_release_v1` AMENDMENT_1 (@e5da0ac44): HGB audits on token-expanded rows.
5. **Closest precedent: an HGB task-prediction release without privacy training.** This is the ACS
   transfer benchmark `results/redesign_20260907_acs_transfer_v1/PROTOCOL.md:101` (commit e765ced24,
   2026-09-07, `experiments/acs_transfer_models.py:321-398` `SourceTreeBank` /
   `fit_source_tree_bank`):
   - **C_tree_bank** released source-task probability vectors from independently fitted
     HistGradientBoosting heads on allowed preprocessed **ACS CA-2018** inputs (15/31 leaves, 150
     iterations).
   - It was compared as a competitive prediction bank against learned representations for transfer
     utility, with SEX/RAC1P attribute audits.
   - The protocol states that "No PCRL, LoRA, erasure, adversarial encoder objective, or privacy
     threshold is used" (line 9).
   - `experiments/acs_transfer_heads.py:22-32,407` also offers `histgb` as a downstream head family
     on frozen releases.

   So a boosted-tree task service **has** been released and attacked, on ACS rather than Adult, but
   never privacy-corrected and never with leaf-score parameters.
6. **"Leaf" in code.** It refers to other things:
   - FARE's fair-tree cells (`oar/fare_official.py:298-300,438`);
   - recursive partition children in the adaptive-release state refinement
     (`experiments/pcrl_adaptive_release_v1/refinement.py:291`, @1b80f7968; the split never ran, per
     its NEXT_ACTION);
   - partition admission tests in `dpc/`.

   None is a boosted-ensemble leaf-value correction.
7. **FARE is the nearest tree-based *privacy/fairness* mechanism.** It is a tree-structured encoder
   that releases cells, run as a baseline in combined-output-aware-removal and durable-guarantees. It
   is not a task predictor with leaf-score corrections.

**Implications for novelty wording.** The campaign may say this is the project's first boosted-tree
task service with privacy-trained leaf corrections. It must disclose:

- the ACS C_tree_bank precedent: an HGB task-prediction release, attacked, without privacy training;
- that boosted-tree readers are an established part of the attack slate;
- that FARE is a published tree-based fair-representation method.

The repository search cannot rule out external prior art on leaf-value fairness or privacy
corrections, for example post-hoc leaf reweighting in fair boosting. That needs a separate
literature check before any novelty claim.

## 5. Adversary-role cautions specific to the change

- **Tree releases and tree readers.** The boosted-tree reader is well matched to a
  piecewise-constant release. Keep it in the slate and refit per arm.
- **T_REFIT is the critical control.** Any recovery change from leaf training without the SEX
  penalty must be separated from privacy effects, as TASK_TRUST and T_BOTH were in the adapters.
- **Headroom must be measured, not assumed.** If T_TREE does not beat U0/Ucal on NLL/Brier with
  margin, the campaign is in the same confidence-bound regime as the prior lines.
- **Equal-exposure fairness.** The CatBoost teacher must see the same FIT people as U0 and the
  neural reference. The utility-reference repair showed that asymmetric exposure contaminated an
  earlier comparison.
