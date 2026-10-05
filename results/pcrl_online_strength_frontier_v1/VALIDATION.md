# Validation

## Before nonzero fits

| Check | Result |
|---|---|
| **Data and roles** (`ROLE_MANIFEST.json`) | `provenance/role_check.py` recomputed every role independently (no osf/smf/rgj import; no label read) and matches `osf.data.load()`: role and pool sets, counts and row-id hashes, X bitwise on all 39,170 rows, numeric refit, label sealing. Raw numerics match the pinned raw loader for every kept row. The fitting tensors are byte-identical to smf `NEW_DEFENSE_FIT`. The assessment is 13,936 rows = 5,243 + 3,397 + 3,796 + 1,500, with no group overlap and no group shared between pools. |
| **Certification-pool eligibility** | **ESTABLISHED** by custody from code at the pin, release row sets, head scaler moments, LEACE fit hashes and FARE tree fit-row fingerprints. The verdict is bound to `osf.data.CERT_ELIGIBLE` in every lock (`cert_eligibility_verdict`). |
| **Admission** (`ADMISSION.json`) | 87 smf units and 42 FARE trees admitted, 0 failures. Copies are re-read uncached. Every jcv/pnx/rgj checkpoint is ineligible (trained on the old 19,230 defense rows). |
| **Design review** (`MATH_REVIEW.md`) | train/data: 0 REQUIRED. select/infer: 2 REQUIRED (S1, a missing guard comparator did not invalidate N\*/R\*; S2, the label could not become INCOMPLETE_OR_INVALID), both fixed before SELECTION_AND_AUDIT_LOCK. RECOMMENDED A1–A4 and A7 applied before DATA_AND_ENGINEERING_LOCK. 53 reviewer fixtures; 53/53 injected defects caught. |
| **Tests** | 109 pass (pipeline, runner, late, admission, closeout, audit, math review). |
| **Audit pre-lock** (`AUDIT_PRELOCK_CHECKS.json`, U, inner roles) | Kept rank equals the feature-block rank, and null directions sit ≤ 7.4e-15 below the tolerance. Shuffled-label null AUCs 0.477–0.515 (threshold 0.565). Planted and rotated 1e-6 clues are detected (0.89–0.91) through save/reload. Real-data null exceedances: 0/60 and 0/15. |
| **Precision planning** (`PRECISION_PLANNING.json`) | z = 3.113017. The one-point guard needs SE < 0.00128 at a true loss of 0.006. The planning SE for a RAW-J occupation loss was 0.0017, so the guard was flagged in advance as possibly unresolved. Realized: the descriptive nominees' occupation losses (−1.4 and −2.0 points) are point effects beyond the guard, not precision failures. |
| **Predictions** (`PREDICTIONS.json`) | Committed at `9159c57` (03:28Z), before any real-data fit. |

## Engineering stages (DATA_AND_ENGINEERING_LOCK `0179033`, pushed 03:54:59Z)

| Check | Result |
|---|---|
| Admission | 3 warm starts replayed bitwise with `jcv.train.warm_start` on OSF_DEFENSE_FIT. 15 admitted releases (U, RAW β 0.1/0.3 × J/L × 3 seeds) rebuilt on all osf rows and bitwise equal to the smf releases on every smf row. |
| Parity | 12/12 bitwise: RAW-J/L β 0 and NORM-J/L ρ 0 equal TASK over 4 epochs with critics training. |
| Raw fidelity | 6/6. The 2-epoch osf RAW is bitwise equal to pinned `rgj.train` J-O/L-O (model and critics). Receipts reproduce rgj's logged norms within 7.8e-7. |
| Frozen-minibatch equivalence | Relative L2 error 4.5e-6 to 2.9e-5 (tolerance 1e-4) on all 12 encoder cases. The registered failures fail: a common ρ, and a cap. |
| Instrumented replays | 15/15 bitwise: epoch-20 and epoch-40 models, and RAW epoch-40 critics, equal the admitted checkpoints. |
| Timing | 0.73 s per NORM epoch, so the full bank was chosen. |

**Amendments A1 (`ad5d867`) and A2 (`cbe72dd`).** These are the two allowed engineering amendments. Each was pushed **before** the affected fidelity rerun, and neither changed training.
- **Defect:** the frozen-minibatch equivalence was vacuous (no applicable minibatch).
- **A1 change:** scan epoch 2 for the first minibatch with nonzero proxy gradients.
- **A2 change:** use deterministic bounded-refit critics on the frozen epoch-2 model.
- **Correction of their stated cause** (review A8, found after A2):
  - The receipts were vacuous because an epoch checkpoint pairs θ_e with the floored-ZCA transform of θ_{e−1}. On these rank-deficient views the stale transform amplifies a one-step change by about 10⁴, so the stored critics lose to the constant.
  - It is **not** that the online critics are weak: training receipts show 0% no-gradient steps, and the aligned snapshots θ_{T−1} and θ_T with a recomputed transform show the same.
  - A2's check is algebraically valid; the reviewer reproduced it on real rows with errors 1.2e-6 to 1.4e-5.
  - The vacuous receipts are kept in `<PRIVATE_CACHE>/osf_v1/quarantine_amendment_A1` and `quarantine_amendment_A2`.

## Training, references, inner selection

- **TRAINING_PROTOCOL_LOCK** (`b8b0e9e`, 04:02:33Z): full bank, 48 new fits, 0 nonfinite, 0 rescued.
  - Zero-direction fractions ≤ 3.8% (NORM-L ρ 5, a 0.5); clip fractions ≤ 34% (NORM-J ρ 5, a 2).
  - Every run applied all 2,440 steps.
- **SELECTION_AND_AUDIT_LOCK** (`a72664f`, 04:14:45Z): references, 63 inner audits, selection and 12 tracking units ran after it.
- **Deviation (disclosed).**
  - The audit/baseline owner ran the reference admission, label-free encodes, inner audits/gates and FARE reselection at about 03:32Z, before any lock.
  - All 96 units and 5 run files were quarantined (kept) and rerun under SELECTION_AND_AUDIT_LOCK.
  - The rerun's data files are identical in all 96 units, and the statuses are unchanged (E and F0 INFEASIBLE_CONTROL; F NO_FEASIBLE_NOMINEE; FARE configuration 1).
  - Record differences: the seed-1 alias note (written in parallel), and an added U cross-check (max |diff| 0). See `<PRIVATE_CACHE>/osf_v1/run/reference_rerun_comparison.json`.
  - The early run saw only inner-role reference statistics, which equal smf's published ones. No bank, rule or grid was changed after it.
- **Critic tracking** (`CRITIC_TRACKING.csv`):
  - Aligned online-minus-fresh critic CE gaps on DIAGNOSTIC_CALIB are 0.05–0.07 at every progress point for both raw and normalized runs (0.24 at step 1).
  - The released θ_T read through the stale transform shows 0.41–1.34. That is the A8 artifact, and it disappears with a recomputed transform (0.05–0.08).

## Assessment (EVALUATION_LOCK `3554235`, pushed 04:19:24Z)

- **Opening.** The lock was on origin before the assessment opened. The opening was logged at 04:19Z, and the first outer unit completed at 04:20:23Z.
- **Gate.** `osf.assess` verified the lock commit on origin, the CHAIN code hashes, assessment rows/groups/row-id hash and the fitting-prior hash before unsealing.
- **Scope.** 72 outer units: 24 labels × 3 seeds, the full locked grid plus E, F and F0, scored once.
- **Inference.** B = 1,999 exact-record-group replicates, seed 20261006, all replicates finite.
- **Result.** Label **EXPERIMENTAL_NO_ADVANTAGE** with `complete = true`. All 27 primary clauses are DESCRIPTIVE_ONLY, because N\* and R\* are NO_FEASIBLE_NOMINEE; their numeric decisions are kept in `decision_numeric`.
- **No post-assessment repair** of code, numbers or reports was needed.

## Deployment

- `osf.deploy` rebuilds `rel__s1__RAW-J_b0.3` (the deployable best), `rel__s1__NORM-J_r3_a1` and `rel__s1__U` bitwise from the 83 permitted columns.
- It refuses an 84-column input.

## Backup and restore (`BACKUP_VERIFICATION.json`)

- **Drive.** Not mounted at any check, so status is `LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_PENDING`.
- **Same-device copy.** 2,583 files copied to `<PRIVATE_CACHE>/osf_v1_local_copy_20261005`, all re-read uncached and matching. This was an uncached read, not a physical cold-disk read, and on the same device; it may share storage.
- **Restores from the copy alone** (own forward pass, the copy's heads), bitwise on every array:
  - U, the incumbent RAW-J β 0.3, the N\* fallback (NORM-J ρ 3), the R\* fallback (RAW-J β 0.6) and L\* (NORM-L ρ 3);
  - the incumbent's AUC-selected pair attacker, refitted from the copy, reproduces its saved assessment predictions (max |diff| 0).
- **Off-device custody: PENDING.**
- **Predecessor custody repair: PENDING** (`PREDECESSOR_CUSTODY_REPAIR.json`). The two later smf logs are hashed and copied to a local supplement. The independent smf restore needs the drive, and its exact command (output redirected away from the closed smf results) is recorded.

## Independent verification (`INDEPENDENT_VERIFICATION.json`, `verification/replay_osf.py`)

**Result: 0 FAIL, 0 WARN.**
- Phase 2: 346 PASS and 3 INFO across all nodes, with every top-level check PASS. Run as one process at worktree HEAD `5f4680a`: 447 s wall, peak 1.6 GB.
- Independence: an import guard refused osf, smf, rgj, jcv, pnx, oar and stored_model_eval, including during unpickling. `torch.load` used `weights_only=True`.
- The verifier's corrections to its own code are logged in the JSON (`verifier_corrections`). None changes a study number.

**Reproduced independently:**
- **Roles.** Rebuilt from the rule; every count and row-id/group hash matches.
- **Releases.** All 63 rebuilt with its own forward pass and its own head refits, bit-exact (126/126 heads). The 15 admitted releases equal smf on all 29,030 shared rows. LEACE and FARE releases also check out.
- **Gradient identities.** Checked on all 63 runs' receipts:
  - NORM ratio = ρ s_i on every uncapped nonzero step (max relative deviation about 1e-8);
  - RMS and combined identities; clip factor;
  - 30 critic updates per step; pair R absent in every local step;
  - minibatch fingerprints identical across all configurations of a seed;
  - 8/8 injected receipt corruptions caught.
- **Training replay.**
  - Its own functional engine reproduces the last update θ_{T−1}→θ_T bit-exactly for all 63 runs.
  - Full 40-epoch retraining with its own loop is bit-exact for U (3 seeds) and for seed-0 RAW-J β 0.3, RAW-J β 0.6, NORM-J ρ 3 (a = 1 and a = 0.5) and NORM-L ρ 3.
  - Adding the shadow pair critic to a local arm changes θ_T, so the test does detect a pair gradient.
- **Replays and engineering.** The 12 RAW replays reproduce all 122 rgj logged norm entries per run. Parity 12/12; fidelity 6/6 including both expected failures.
- **Selection.** L\*, C\*, N\* and R\* fallbacks, shortfalls (0.0043, 0.0215), the deployable best and the FARE/reference statuses all match.
- **Assessment order.** Lock pushed 04:19:23Z, first assessment event 04:19:42Z. All 72 outer records cite the lock commit and hash. All 78 unit pins match.
- **Endpoints.** All 88 slots and 578 levels match `inference.json` within 1e-10, using B = 1,999, seed 20261006, 13,929 groups, z = 3.113017 and 3.346065. Conjunctions and the label match.
- **Final attacker restore.** For the incumbent's assessment unit, the HGB pair and DA_MLP local attackers were refitted and reproduce the saved probabilities bit-exactly at all three attacker seeds.
- **Critic tracking.** 1,512 CE values recomputed exactly.
- **Controls.** Real-data shuffled-SEX null AUC 0.505; a planted rotated 1e-6 clue is detected at AUC 1.0.

**Wording the verifier asked for (adopted).**
- The 10 secondary slots involving N\* or R\* (S-Nstar-\*, S-Rstar-\*, S-syn-N\*, S-syn-R\*, S-ll-\*) score the descriptive **fallback** configurations (NORM-J ρ 3, RAW-J β 0.6). Their ABOVE/BELOW labels describe those configurations, not valid nominees.
- On the numbers alone, fallback claims A, B and C would pass 8/9, 7/9 and 8/9 clauses. Clause counts are not partial success, and fallbacks cannot pass a claim.
- The pre-lock reference inner gates read inner-role labels only. They are a disclosed process deviation, not contamination of the assessment.

## Information items

- **RECOMMENDED A6 (not applied).** `osf.data.load(unseal=True)` relies on `osf.assess`'s gate rather than requiring a lock token itself.
- **N1.** The `labels_for` allowlist is documentary for training. The effective guard is the −1 masking at load, which the review verified.
- **FARE trees are seed-invariant.** F and F0 seeds are aliases, not independent replications.
- **Interpolation is post hoc.** The raw-line interpolation in RESEARCH_DECISION.md is a descriptive post hoc comparison, not a registered endpoint.
