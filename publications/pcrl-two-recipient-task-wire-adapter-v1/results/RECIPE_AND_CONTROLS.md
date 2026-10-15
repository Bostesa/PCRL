# Recipe and matched controls

## Mechanism (per recipient i; 0 = income, 1 = occupation)
q_i = float64(z0_i) B_iᵀ + (h64_i / s_i) R_iᵀ + b_i. The released representation is z_i = q_i B_i, and the released centered logits, probabilities and decision come only from that representation through the frozen own head i.
- **Size:** the income adapter is rank 1 with 65 parameters; the occupation adapter is rank 5 with 325. That is 390 trainable parameters in total.
- **Scales:** s = 1.0 for both recipients and both seeds (`max(1, rms h64)` over UPDATE+GUARD).
- **Zero residual:** zero residuals reproduce RAW_PROJ bitwise on every admitted role.
- **Wires:** nominal widths 21/29/50. Nothing else is released: no h64, z16, q0, teacher scores, the other head's outputs, labels, ids or controller state.
- **What this is:** constrained prediction adaptation over frozen features. It is not automatically a novel algorithm or a population privacy guarantee.

## Fixed bank (both seeds, no sweep)
| arm | schedule | privacy objective applied to each active recipient |
|---|---|---|
| T_BOTH | simultaneous | none (task only) |
| L_BOTH | simultaneous | risk(income21) + risk(occupation29); each adapter feels its own view |
| J_BOTH (primary) | simultaneous | risk(income21) + risk(occupation29) + risk(pair50) |
| S12 | income 320, frozen, then occupation 320 | joint objective restricted to the active recipient |
| S21 | occupation, then income | as S12 |
| I_ONLY | income 320; occupation fixed at RAW_PROJ | joint objective restricted to income |

S12 and S21 are learned sequential controls; they are not implementations of Taylor et al.'s finite-law algorithm. **References:** immutable RAW_PROJ, and O_PREVIOUS, an inherited component that is not an equally trained two-adapter control.

**Shared by every arm:**
- **Optimizer:** SGD without momentum, learning rate 0.01, batch size 256.
- **Proposals:** 320 opportunities per active recipient, in 10 epochs per active recipient, with checkpoints 0/1/3/5/10. Each actual displacement is at most 0.01 per recipient.
- **Task objective:** (8180/10223) × UPDATE minibatch CE + (2043/10223) × whole-GUARD CE.
- **Budgets:** each recipient has its own budget, anchored at its zero start and never reset: NLL +0.004, summed Brier +0.002, correct ≥ start − 51, and mean KL(p0‖pθ) ≤ 0.001.
- **Step rule:** exact KKT projection, then the ladder 1, ½, ¼, ⅛ with exact rollback.
- **Critics:** linear, MLP32 and MLP64 on income21, occupation29 and pair50, trained on UPDATE SEX with Adam at 0.003. They warm up for 80 rounds, then get 5 rounds per recipient opportunity (10 per simultaneous macro-step). Totals are 3,280 rounds for the simultaneous and sequential arms and 1,680 for I_ONLY.
- **Coefficients:** fixed per recipient so the initial privacy-gradient norm is 3 × that recipient's task-gradient norm, using the median over 16 warm draws. Equal target ratios do not imply equal force.

| coefficient | seed 0 | seed 1 |
|---|---|---|
| joint, income | 1.537 | 1.315 |
| joint, occupation | 0.740 | 0.749 |
| local, income | 3.111 | 2.228 |
| local, occupation | 1.438 | 1.617 |

## Training outcome (fitting pool only)
- **Acceptance:** every income opportunity was accepted in every arm and seed. Occupation steps rejected per seed (seed 0 / seed 1):

  | arm | rejected occupation steps |
  |---|---|
  | J_BOTH | 7 / 6 |
  | L_BOTH | 7 / 10 |
  | S12 | 20 / 18 |
  | S21 | 14 / 10 |
  | T_BOTH | 0 / 0 |

- **Active budgets:** the income adapter never had an active budget constraint in J_BOTH (final income KL 0.00074 / 0.00036); occupation had 88 and 79 active-budget cases and ended at its KL limit (0.00100 / 0.00098).
- **Privacy-driven movement:** in J_BOTH, each of the first 8 accepted updates per recipient, seed 0 and seed 1, changed the deployed probe wire when its privacy gradient was removed (8/8 each, all in epoch 1, before the nominated epoch 10; independently re-solved, see `INDEPENDENT_VERIFICATION.json`).
- **FIT2 selection:** every arm is point-feasible at its latest checkpoint (E10; P2E10 for the sequential arms). RAW_PROJ and O_PREVIOUS are point-eligible too, so the primary roster is complete.

## Attack slate and accounting
- **Readers:** the inherited 18-reader complete-view slate, trained on ATT train groups with SEX and oriented on ATT validation groups. Views are income21, occupation29 and pair50; the pair envelope includes explicit ignore-income and ignore-occupation readers.
- **Services per seed (10):**
  - the six new arms;
  - RAW_PROJ and O_PREVIOUS;
  - two fixed wire substitutions from J_BOTH E10: HYB_JI_RO (new income, RAW occupation) and HYB_RI_JO (RAW income, new occupation).
- **Fit counts:**
  - 18 readers × 3 views × 10 services × 2 seeds = 1,080 direct fits.
  - Shuffled and planted control banks: 216 fits.
  - Independent refits: see `INDEPENDENT_VERIFICATION.json`.
  - Cap: 1,500.
- **Aliases:** exact wire aliases are recorded in `REPORTING_LOCK.json` but never used to skip a fit.
- **Control gates:** all six seed × view gates passed (shuffled envelope ≤ 0.5270, planted 1.000).
