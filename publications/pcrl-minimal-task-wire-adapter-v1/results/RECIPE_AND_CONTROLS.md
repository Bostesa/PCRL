# Recipe and matched controls (frozen in SCIENCE_LOCK.json before any fit)

## Mechanism
The trunk, both folded own heads, the bases B (occupation, rank 5) and Bi (income, rank 1), the class orders and the input preprocessing are all frozen. The only trainable part is a residual in the occupation head's row space:

  q0 = float64(z0) Bᵀ,  q = (q0 + (h64/s) Rᵀ) + b_res,  z_release = q B.

The occupation outputs are the frozen own head applied to z_release. R and b_res start at exact zeros, giving 325 parameters; at zero the service is exactly RAW_PROJ. Income is the unchanged factorized RAW projection. The wires keep widths 21, 29 and 50, and the occupation effective rank is 5. Nothing else is transmitted.

This is a constrained task-head adapter over frozen nonlinear features. It is not a new representation learner, not a proven privacy mechanism, and not joint design of both recipients.

## Arms (both source seeds)
| arm | applied privacy objective | teacher-function trust K = mean KL(p0‖pθ) ≤ 0.001 |
|---|---|---|
| J_TRUST | occupation-local + coalition (pair) critic risk | yes |
| J_FREE | same as J_TRUST | no (K logged only) |
| L_TRUST | occupation-local critic risk | yes |
| L_FREE | same as L_TRUST | no |
| TASK_TRUST | none (zero applied privacy gradient) | yes |

Reference services are RAW_PROJ (the primary comparator) and RAW_HYBRID (an older contract with the full occupation z16; greater disclosure, not a matched control).

## Shared recipe
- **Task objective:** (8180/10223)·CE on the UPDATE minibatch + (2043/10223)·CE on the whole GUARD pool.
- **Updates:** SGD without momentum, lr 0.01, actual displacement ≤ 0.01, batch 256, 32 proposals per epoch, 10 epochs. Checkpoints at epochs 0/1/3/5/10. Paired random streams across arms.
- **Critics:** linear, MLP32 and MLP64, on occupation29 and pair50. Adam 0.003, 80 warm updates, 5 updates per critic before each proposal, trained on UPDATE SEX only. Every arm runs the same critic bank, including TASK_TRUST, from an identical warm state (hash asserted).
- **Risk:** 1 − min(critic CE, H)/H per view.
- **Coefficients:** 3 × median‖g_task‖ / median‖g_priv‖ over 16 warm draws.
  - λ_J = 0.934 (S0) and 0.788 (S1), shared by J_TRUST and J_FREE.
  - λ_L = 1.539 (S0) and 1.086 (S1), shared by L_TRUST and L_FREE.
- **Budgets:** absolute, on the full 10,223-row task pool, anchored at the zero-residual start and never reset: NLL ≤ start + 0.004, summed Brier ≤ start + 0.002, correct ≥ start − 51. TRUST arms also enforce K ≤ 0.001.
- **Step rule:** an exact KKT active-set projection over at most 3 halfspaces plus the norm ball, checked against SLSQP on data-free and degenerate cases. Then a finite recheck at fractions 1, ½, ¼ and ⅛ with exact rollback.

## Training outcome (fitting pool, epoch 10)
| seed | arm | accepted / rejected of 320 | task-pool NLL − start | K |
|---|---|---|---|---|
| 0 | J_TRUST | 291 / 29 | +0.0024 | 0.00099 |
| 0 | J_FREE | 309 / 11 | +0.0039 | 0.00195 |
| 0 | L_TRUST | 309 / 11 | +0.0024 | 0.00098 |
| 0 | L_FREE | 315 / 5 | +0.0039 | 0.00199 |
| 0 | TASK_TRUST | 320 / 0 | −0.0003 | 0.00009 |
| 1 | J_TRUST | 314 / 6 | +0.0014 | 0.00100 |
| 1 | J_FREE | 313 / 7 | +0.0034 | 0.00280 |
| 1 | L_TRUST | 318 / 2 | +0.0014 | 0.00100 |
| 1 | L_FREE | 318 / 2 | +0.0036 | 0.00281 |
| 1 | TASK_TRUST | 320 / 0 | −0.0002 | 0.00003 |

No QP fallback occurred. The critic budgets matched exactly in every arm: (80 + 5×320) × 6 optimizer steps.

## Privacy-bearing movement
- **Counterfactual test.** For the first 8 accepted J_TRUST updates in each seed (updates 1–8, epoch 1), the full update was re-solved at the saved pre-step state with the privacy gradient removed.
- **Result.** All 16 differ in actual displacement beyond the contribution tolerance (1e-10; repeat-proposal noise 0). All 16 also differ in the deployed 512-row probe occupation wire beyond the frozen tolerance.
- **Independent check.** The verifier re-solved all 16 with its own code and matched the stored displacements to 2.2e-18. No budget halfspace was active in these states, so the re-solves exercise only the unconstrained and norm-ball branches.
- **Probe movement at epoch 10 vs start.** The occupation wire changed in every arm:
  - J_TRUST: max logit change 0.21 (S0) and 0.22 (S1), with 21 and 24 of 512 decisions changed.
  - TASK_TRUST: 0.045 and 0.034, with 4 and 5 decisions changed.

## Selection (FIT2, opened once after TRAINING_COMPLETE_LOCK)
The rule takes the latest common epoch > 0 that passes all 24 original POINT clauses (income and occupation × U0/Ucal × accuracy, NLL and Brier × both seeds).
- J_TRUST: epoch 10.
- J_FREE: epoch 10.
- L_TRUST: epoch 10.
- L_FREE: epoch 5. Epoch 10 fails seed-0 occupation NLL vs U0 at 0.0102 against a cap of 0.010.
- TASK_TRUST: epoch 10.
- RAW_PROJ is point-eligible.
- The prior guards fail FIT2 point caps (occupation NLL excess 0.014–0.021). They are kept only as fixed diagnostics.

## Fresh readers
- **Slate.** The inherited 18-reader complete-view slate, trained on ATT train groups, oriented on ATT validation groups, and scored on INNER without labels.
- **Pair view.** It also includes exact ignore-income and ignore-occupation aliases.
- **Income view.** The income wire is identical in every service, so its 18 readers were fitted once per seed and alias-verified by hash in each service unit.
- **Direct fits.** 972 in total:
  - per seed: 18 income + 10 services × 36 (occupation and pair) + 108 control fits;
  - the science lock's planned count of 1,188 mis-stated the control fits as 216 per seed (the actual total is 216 across both seeds).
- **Controls.** Group-permuted-SEX and planted-SEX-bit banks: 2 banks × 18 readers × 3 views = 108 fits per seed, 216 in total. All gates passed:
  - null AUC exactly 0.5;
  - shuffled envelope 0.509–0.527 (cap 0.60);
  - planted envelope 1.000 (floor 0.99).
