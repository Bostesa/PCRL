# Protocol: refreshed critics with per-recipient local feedback (J-G)

**Status.** This is exploratory method development on already-exposed Adult rows.
- Every row used here has been exposed in earlier studies (see `EXPOSURE_LEDGER.md`).
- DEVELOPMENT_ASSESSMENT is a new development partition. It is not fresh confirmation.
- The predecessor's verdict (research/pcrl-penalty-no-erasure-v1 @ b628ff5) stays closed. Its old assessment rows are neither rescored nor read.
- This protocol, the code lock and the family definitions are pushed before any nonzero fit.
- EVALUATION_LOCK.json is pushed before the single assessment.

## 1. Data and roles (`rgj/data.py`, `ROLE_MANIFEST.json`)

**Source.** `adult_jcv.npz` (sha256 `e0d9e54a…`): 83 permitted columns, preprocessing fitted on the old defense_train only. The analysis unit is the exact-record group.

| New role | Source | Rows / groups | Use |
|---|---|---|---|
| DEFENSE_FIT | old defense_train | 19,230 / 19,221 | encoders, training heads, critics, LEACE maps, FARE trees |
| ⤷ CRITIC_FIT | 70% of DEFENSE_FIT groups | 13,543 | critic training (online and refit) |
| ⤷ CRITIC_VAL | 15% | 2,844 | bounded-refit early stopping and attempt choice |
| ⤷ CALIB | 15% | 2,843 | local surrogate budgets and multiplier updates |
| HEAD_VALIDATION | 30% of old defense_val groups (hash seed 20261004) | 1,500 / 1,499 | task-head C selection only |
| DEVELOPMENT_ASSESSMENT | 70% of old defense_val groups | 3,397 / 3,393 | sealed until EVALUATION_LOCK.json is pushed |
| AUDIT_FIT | old attacker_fit | 6,065 | inner and final attacker fitting |
| INNER_SELECTION | old attacker_val | 2,235 | candidate evaluation, nomination, attacker selection |

- **Allocation rule.** A group goes to HEAD_VALIDATION iff `int(sha256("20261004|dev|<unit>")[:16], 16) / 2^64 < 0.30`. The critic subroles use the salt `critic` with cut points 0.70 and 0.85.
- **Dropped rows.** The old assessment (5,243), certification (1,500) and excluded rows (35) are dropped at load. No function receives their inputs, labels or ids, and no prediction is produced for them.
- **Groups and labels.** No group spans two roles. SEX is the primary protected attribute, forbidden to both recipients. Race is a supported-class stress audit.
- **Proxies kept.** The permitted inputs keep the relationship proxy (Husband/Wife ≈ 44% of rows determine SEX) and other proxies. Protection here means protection with those proxies present.

## 2. Model and arms (`rgj/train.py`, `METHOD_CARD.md`)

**Model.** Unchanged from the predecessor:
- two separate encoders 83-64-64-16 (ReLU) and affine training heads;
- the shared task-only warm start (predecessor `warm__s{k}`, hash-verified; fixed 20-epoch Adam on DEFENSE_FIT only);
- SGD 0.05, global-norm clip 5, 20 protection epochs of 76 steps;
- release r_i = g_i(X) with no erasure;
- deployed heads refit as StandardScaler + LR, with C chosen on HEAD_VALIDATION;
- the primary view [r_i, centred logits_i].

**Critic views (review R1-a, before any nonzero fit).** Critics read v_i = [r_i, centred(W_w r_i + b_w)], where (W_w, b_w) is the seed's warm-start training head, held fixed for every critic arm and every stage. The pair view is [v1, v2].
- This carries the same information as r_i, because the logits are affine in r_i.
- With the drifting training head, the logit block left the snapshot's null space, and a block-fixed floored transform amplified it about 3,000×. The reviewer's fixture showed the online critics degrading within a block.
- The fixed head keeps the null space constant and makes transport exact.
- The released views and deployed heads are unchanged: they use the refit deployed head.

**Critic banks.** Every non-task arm trains the same three critic banks (v1, v2, pair), with two kinds each:
- A: 32-unit MLP;
- B: 64-64 MLP.

Initialisations are deterministic and identical across arms. In local arms the pair bank is a shadow bank: trained, but never in the encoder gradient.

**Recovery surrogate and objective.** R_v = (CE_const − min(CE_const, CE_A, CE_B)) / H_fit(S), on the same rows. The objective is:

L_task + β (R_1 + R_2 + R_0)/3 + λ_1 (R_1 − c_1) + λ_2 (R_2 − c_2)

- In joint arms R_0 = R_pair.
- In local arms R_0 = (R_1 + R_2)/2, so each R_i gets weight β/2 and the pair none. The base weights therefore sum to β in both arms.

| Arm | Stage | Base term | Critic schedule | Local multipliers |
|---|---|---|---|---|
| U-B / U | B / C reference | task only | none | none |
| L-O | B | local | online: per-step floored ZCA (inherited) | no |
| L-R | B (frozen reference) | local | refreshed: frozen-snapshot refits | no |
| **J-G** (candidate) | C | joint | refreshed | **yes** |
| L-G (decisive control) | C | local (+ shadow pair bank) | refreshed | yes (identical rule) |
| J-R | C | joint | refreshed | no |
| J-O | C | joint | online | no |
| E | C reference | official LEACE per recipient on U's features | — | — |
| F / F0 | C reference | official FARE per purpose, and its zero-fairness twin | — | — |

**Refreshed critics.**
- Refits happen at protection epochs 0, 4, 8, 12, 16 and 20. Each one freezes the model, computes views on DEFENSE_FIT and fits a new floored ZCA transform on CRITIC_FIT. The transform then stays fixed for the block.
- Per view and kind, two attempts are trained, both on CRITIC_FIT with at most 15 epochs, patience 3, batch 256 and Adam 3e-3:
  - a continued critic, whose first layer is transported so that its function is unchanged;
  - one deterministic fresh restart.
- The attempt with the lower CRITIC_VAL cross-entropy is kept. Both receipts are saved.
- Between refits, each critic takes 5 online Adam steps per encoder step.
- The epoch-20 refit is for diagnosis only.
- This is a bounded refit, not convergence or an exact best response.

**Online critics (L-O, J-O).**
- Before every encoder step, the floored ZCA transform is recomputed on a fixed 4,096-row CRITIC_FIT subset; there is no transport. This is the inherited schedule.
- Stage C critics inherited from L-R receive one transport into the first transform.

**Multipliers (J-G, L-G).**
- λ_i ← clip(λ_i + β(R_i − c_i), 0, 3β) at each refit, using R_i measured on CALIB with the refitted critics. λ_i starts at 0.
- c_i = R_i(frozen L-R; fresh critics; CALIB) + 0.005, in surrogate units, not AUC.

**Grid.**
- β ∈ {0.03, 0.1, 0.3}, checkpoints at protection epochs {5, 10, 15, 20}, seeds 0, 1, 2.
- Stage B data order: rng([seed, 0, epoch]), identical to the task line. Stage C: rng([seed, 1, epoch]).

**Task-only references.** The task line runs 40 epochs from the warm start, with checkpoints every 5.
- U-B is the epoch-20 checkpoint.
- U on seed k is checkpoint 20 + e_LR(k), so its total encoder-update budget matches a Stage C arm. If L-R is TASK_ONLY_ALIAS, e_LR = 20.

**Stage C initialisation.** Every Stage C arm starts from the frozen L-R checkpoint (model, critics, transforms). If L-R is the task-only alias, it starts from U-B with fresh critics, and this is disclosed.

## 3. Pre-fit checks

These are `rgj/tests/test_pipeline.py` (lead), `rgj/tests/test_math_review.py` (math reviewer) and `rgj/tests/test_audit.py` (audit owner), plus the parity stage.

- **β = 0, λ = 0 identity.** Each arm reproduces the task-only continuation bitwise on all seeds, from an identical initialisation.
- **Environment parity.** The task line at epoch 20 equals the predecessor's U bitwise.
- **Stage C receipts.** Each Stage C arm at β = 0 started from the frozen L-R equals the task-only continuation from that same L-R checkpoint (receipt `tc__s{k}`). These are labelled initialisation-specific receipts; they are not U.
- **Retained receipts.** Failing receipts are kept.

## 4. Selection (inner roles only; `rgj/select.py`)

**Utility gates.** For both tasks, on INNER_SELECTION:
- G1: Acc ≥ Acc(ref) − 0.01;
- G2: Acc − const ≥ 0.8 (Acc(ref) − const);
- G3: Acc − const ≥ 0.03.

The reference is U-B in Stage B and U in Stage C. Recovery is the INNER_SELECTION AUC of the AUC-selected inner attacker (LR, MLP 64×64, HGB; fit on AUDIT_FIT; fixed orientation). The coalition bank includes the ignore-recipient candidates.

1. **Freeze L-R, and L-O by the same rule.** Among gate-passing candidates (β × checkpoint, plus the U-B alias), minimise the worse local AUC, then the mean local AUC, then β, then the checkpoint.
   - If the alias is selected, the status is TASK_ONLY_ALIAS.
   - If U-B itself fails G3, the status is NO_VALID_REFERENCE.
   - The freeze is committed and pushed before Stage C is fitted.
2. **Calibrate c_i on the frozen L-R.**
3. **Train Stage C.**
4. **Select the trained controls (L-G, J-R, J-O).** Among candidates that pass the gates and the zero-buffer local guard AUC_i ≤ AUC_i(L-R), minimise the coalition AUC, then β, then the checkpoint. The single-configuration controls U, E, F and F0, and the frozen L-R and L-O, are feasible iff they pass the gates and the same guard. F uses the predecessor's published per-purpose FARE rule, reselected on the new roles.
5. **Choose C\*.** C\* is the lowest coalition AUC among the feasible controls in the order L-G, L-R, L-O, J-R, J-O, U, E, F, F0.
6. **Nominate J-G.** A candidate must pass the gates and satisfy AUC_i(J-G) ≤ min(AUC_i(L-R), AUC_i(L-G) if a nominee, AUC_i(C\*) if present), for i = 1, 2. Among those, minimise the coalition AUC, then β, then the checkpoint.
   - The zero-point-increase buffer is a nomination buffer. The final +0.01 clause is unchanged.
   - If nothing qualifies, the status is NO_FEASIBLE_NOMINEE. The closest candidate is scored descriptively and can never support a claim.

Every candidate considered and every rejection reason is recorded. No assessment value is used anywhere in selection.

## 5. Stages, locks and stops

- **Order:**
  1. CODE_LOCK.json, pushed;
  2. parity;
  3. task line and Stage B;
  4. inner audit;
  5. Stage B freeze, pushed;
  6. calibration and Stage C;
  7. baselines;
  8. inner audit;
  9. Stage C selection;
  10. critic tracking and the whitening diagnostic;
  11. the optional ablation if time allows;
  12. EVALUATION_LOCK.json, pushed;
  13. the single assessment;
  14. inference;
  15. reports, verification, backup and closeout.
- **Later code.** Files written later (baselines, assessment, tracking, whitening, inference, deployment) enter through dated, pushed CODE_LOCK_A<n>.json amendments before the stage that runs them.
- **Grid and repairs.** The grid is never expanded. Completed valid fits are never refitted.
- **Technical retries.** An exception or a corrupt incomplete unit is rerun. Nonfinite training gets one registered half-learning-rate retry, the same for every arm, and the failed receipt is kept. Poor utility or privacy is a result, not a rescue condition.
- **After the assessment opens.** Only numerical or label defects may be repaired, through a dated analysis amendment with the original results preserved.

## 6. Assessment (`rgj/assess.py`, audit owner)

**Final attackers.** For each frozen release and view, the registered final slate is fitted on AUDIT_FIT:
- LR with C ∈ {0.01, …, 100};
- MLPs (64), (128), (64, 64), (128, 128);
- HGB with learning rate {0.05, 0.1} × leaves {15, 31};
- DA, the defense-aware attacker: canonical PCA-whitening on AUDIT_FIT, which restores low-variance directions, followed by an MLP(128, 128). It knows the release structure and the public head. It is not a white-box or repeated-query attacker.
- Finite releases also get the cell-conditional attacker.

**Dual selection on INNER_SELECTION.** The primary attacker is chosen by AUC with a fixed orientation. A proper-loss attacker is chosen separately by cross-entropy, for log-loss reporting. Both are refitted at attacker seeds 0, 1, 2.

**Coalition banks.** Each contains its own pair attackers plus the ignore-recipient-1 and ignore-recipient-2 candidates. All tables are archived. There is no assessment-based choice, no orientation flip and no clamping.

**Secondary audits.**
- Output-only views (probabilities, hard decisions).
- Supported-race audit (≥ 30 rows per class in AUDIT_FIT, INNER_SELECTION and DEVELOPMENT_ASSESSMENT). An absent supported class is "not estimable", never zero leakage.
- Utility: accuracy, balanced accuracy, per-class and minority recall, log loss, Brier and ECE.
- Common LR probe.
- Linear diagnostics: OLS R² on the fitting rows and held out, plus a fixed-penalty ridge R², each with a shuffled null.

**Controls.** Real-data shuffled-label nulls (permutations frozen) and planted leaks, on the inner roles. A failed control invalidates that audit; it is never silently dropped.

**Scored labels per seed:** J-G, L-G, J-R, J-O, L-R, L-O, U, E, F, F0, plus fixed-β units J-G/J-R/J-O at β 0.1, epoch 20 (secondary component contrasts). C\* resolves to one of these.

## 7. Endpoints and inference (`rgj/family.py`, `PRIMARY_FAMILY.json`)

**Primary family.** 18 slots, z = Φ⁻¹(1 − 0.05/36) = 2.991316.
- Claim A compares J-G with L-G; claim B compares J-G with C\*.
- Each claim has nine clauses:
  - coalition improvement: lower bound > 0.02;
  - local increase on each recipient: upper bound < 0.01;
  - accuracy versus U on each task: lower bound > −0.01;
  - 80% retention on each task: lower bound > 0;
  - gain over the constant on each task: lower bound > 0.03.
- Slots P13–P18 alias P04–P09.

**Claim rules.**
- Claim A requires all nine clauses, a valid L-R, and nontrivial J-G and L-G NOMINEEs on all three seeds.
- Claim B requires all nine clauses, a valid L-R, a J-G NOMINEE and an existing C\* on all three seeds.
- The family never shrinks.

**Secondary family (30 slots, z = 3.143980).**
- output-only formats ×6;
- race ×3;
- proper-loss recovery ×3;
- fixed-β local-feedback component J-R − J-G ×3;
- fixed-β refresh component J-O − J-R ×3;
- J-G versus each control on the coalition ×8;
- coalition minus best local ×4.

**Inference.**
- Per-seed statistics are averaged over seeds.
- The bootstrap is a paired multinomial bootstrap over DEVELOPMENT_ASSESSMENT exact-record groups, with B = 1999 and seed 20261004. The same groups are resampled for every arm and seed.
- SE is the bootstrap standard deviation; intervals are point ± z·SE.
- The intervals are conditional on the fitted models. They exclude defense and attacker refit variance and the dataset's reuse history.

**Labels.**
- DEVELOPMENT_ADVANTAGE_ESTABLISHED: the full conjunction passes. Novelty and fresh confirmation remain separate questions.
- EXPERIMENTAL_NO_ADVANTAGE: the work completed without the full conjunction.
- INCOMPLETE_OR_INVALID: a required unit, reference or control prevents a valid comparison.

## 8. Diagnostics (inner roles only)

**Critic tracking (`rgj/critic_track.py`).** For each selected J-G, L-G, J-R and J-O run, aligned snapshots are taken at Stage C step 1 (start), step 761 (midpoint) and the last step. Each snapshot is (θ_{s−1}, the critics after their step-s updates, the transform in use).
- Fresh critics of the same kinds are trained on the same frozen views with the same transform, using CRITIC_FIT and CRITIC_VAL.
- The registered statistic is the mean over kinds of the paired CE(online_j) − CE(fresh_j), on CALIB and on INNER_SELECTION rows. Best-of-bank differences are named separately.
- A smaller gap counts as tracking only if fresh critics are not themselves weaker. It counts as protection only alongside lower independently audited recovery.

**Whitening (`rgj/whiten_diag.py`).** On frozen snapshots, the floored, raw-centred and capped (ridge 1e-4 · max) transforms are compared on:
- singular values and condition numbers;
- critic fit quality on CRITIC_VAL;
- input-gradient norms.

A planted clue of amplitude 1e-6 must remain detectable by a suitably scaled classifier. The main release never has directions projected out.

**Optional ablation (registered now; diagnostic only).** J-G with the capped transform at β = 0.1 on seeds 0–2, run only if mandatory stages finish before the reserve. It is never eligible as a nominee.

## 9. Resources

- Ceilings: 10 h elapsed, 20 CPU-h, at most two heavy workers, < 8 GiB RAM, ≥ 5 GiB free, $0 cloud. The final 2 h (from 12:46Z) are reserved.
- A watchdog stops only this study's processes at 14:46Z.
- Engineering timing (seed 0, β = 0 refreshed run): about 20 s per 20-epoch run.
- Core queue: 21 Stage B runs plus the task line, 36 Stage C runs, about 290 inner audits and 39 outer units. That is about 1.5 CPU-h and well inside the ceilings, so the core is not reduced.
- Working private units stay on the laptop's internal disk (136 GiB free) for reliability during unattended runs. A versioned, verified copy goes to the external drive at closeout.

## 10. Registered predictions (written before any nonzero fit)

1. Refreshed arms close most of the measured critic gap. At the final aligned snapshot, the mean paired gap for J-R and J-G is below J-O's on most seeds. The gap stays positive, because refits are bounded.
2. L-R is a nonzero NOMINEE (not TASK_ONLY_ALIAS) on at least two of three seeds, most often at β = 0.1 or 0.3.
3. The multipliers are mostly small. With η = β and |R − c| of a few hundredths, λ stays below β/2 on most guarded runs, so local feedback changes the trajectory only modestly.
4. Claim A is NOT_ESTABLISHED. The most likely failing clause is the coalition margin (lower bound ≤ 0.02): the joint arm puts only β/3 on the pair, and nomination requires J-G to be locally no worse than L-G. The alternative is J-G having no feasible nominee on some seed.
5. Claim B is NOT_ESTABLISHED, with C\* = L-G or L-R on most seeds.
6. Selected J-G is within 1 accuracy point of U on both tasks, so the utility clauses pass if a nominee exists.
7. The multipliers are weak by design. With 5 training-effective updates and η = β, λ_i ≲ β·Σ(R − c)₊. That is about 0.04β, 0.12β or 0.4β for violations of 0.01, 0.03 or 0.1 per update, against base weights of β/3, and the 3β cap is unreachable (review). So J-G ≈ J-R at matched β, and λ/(β/3) stays below 1 on every run.
8. The zero-buffer nomination guard, min(L-R, L-G, C\*) on noisy inner AUCs, gives J-G NO_FEASIBLE_NOMINEE on at least one seed (review A6). That alone would make both claims NOT_ESTABLISHED.

## 11. Review notes carried into reporting (MATH_REVIEW.md)

| Item | Note |
|---|---|
| R1 (fixed) | Critic views use the fixed warm-start head. |
| R2 (fixed) | A feasible task-only L-R or L-O alias is eligible for C\*. |
| A2 | The calibrated c uses fresh critics, while in-run R at Stage C epoch 0 uses best-of(continued, restart). At the same state the two differ by 0.001–0.006, the same order as the 0.005 margin. Kept as registered and reported. |
| A3 | Online kind-B critics overfit CRITIC_FIT within a block even at β = 0. The tracking gap is therefore reported per kind, and a β = 0 J-R tracking baseline from the same L-R start is added. |
| A4 | The block-end max\|T(V)\| is logged; it grows at β > 0. |
| A5 | Checkpoint critics were last updated on θ_{step−1}. |
| A7 | Under the β/3 normalisation, the new β = 0.3 ≈ the old PN β = 0.1 per term. |
| A8 | Matched base sums do not match the adaptive weights, the gradient magnitudes or the clip share. |
| A9 | The encoder trains on all DEFENSE_FIT rows, including CRITIC_VAL and CALIB, which are held out from critic fitting only. |
| A10 | A rotated 1e-6 clue survives every transform for a re-whitened linear reader, but bounded critics are blind to it under all three. The audit's planted clue is coordinate-aligned only. |

## 12. Limits

- Development evidence on reused rows.
- Attacker-based evidence only, with intervals conditional on fitted models.
- No population, DP, MI or universal nonlinear guarantee.
- Fitted-row linear guardedness is not provided.
- J-G versus L-G changes the coalition weight while holding the base weight sum. The adaptive weights and gradient magnitudes are part of the treatment and are reported.
- A failed recipe does not bound future methods.
