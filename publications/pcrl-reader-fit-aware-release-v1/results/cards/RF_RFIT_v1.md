# Card RF_RFIT_v1 (family FAM, revision v1)

Coordinator, 2026-10-25, written before any label read by this campaign and committed with the code it governs (`rfa/fam/rfit.py`,
`rfa/fam/fit.py`, `rfa/fam/net.py`, `rfa/fam/__init__.py`) before its first fit. Inputs: the mechanism investigator's hand-back
(`roles/investigator/HANDBACK_1.md`) and the independent challenger's audit (`roles/challenger/HANDBACK_1.md`; verdict PROCEED;
required changes C1-C7 adopted here). EXPLORATORY DEVELOPMENT. Nothing here claims that the operation works, removes information or is
novel.

## 1. Start and scope

- START = NEURAL_RAW_S<v> (rho 1, E2), as the brief proposes.
- Only the occupation network is trained. The income network and released income vector stay START's bitwise. The reason is recorded
  development evidence: SAR1's FB_PFD_v1 diagnosis (the income network absorbed the protection objective and bound first), and
  NEURAL_RAW's pair AUC on the SAR1 grid equals its occupation view (0.6253).
- Deployment inputs and outputs are unchanged: the six fields in, each task's probability vector and decision out.

## 2. Operation (frozen in `RECIPE_RF`, schema RFA1-reader-fit-recipe-v1, and `rfa.fam.fit.RECIPE`)

Per release step: occupation cross-entropy on a 256-row batch of NEURAL_RAW's task stream, plus lambda x S, where S is computed on
one episode.

- **Episode.** Household-separated rows A (about 4,096) and E (about 2,400), drawn from the fit unit's rows. The episodes are identical
  in FULL, STOPFIT and ALTCRIT. The inner size matches the declared recipient's FIT budget, and E matches its SCORE size.
- **Views.** Occupation (4 logit differences) and pair (with START's constant income logit difference).
- **Readers.** Two families, SIMPLE (linear) and RFF (64 frozen random Fourier features), on standardized, soft-bounded features. The
  A-row mean and SD are detached constants.
- **Inner fit.** Ridge-logistic (ridge 1e-2) on A, solved exactly by Newton in float64, with no gradient through the solve itself.
- **Surrogate.** On E, R = (AUC_b - 1/2)^2, where AUC_b is the binormal smooth AUC of the held-out score: invariant to scale and shift,
  and symmetric, so an inverted score is not rewarded. S sums R over the two views, averaging over the two families.
- **Modes.**
  - **FULL** (candidate): the derivative of S through the E outputs (direct path), plus the implicit-function derivative through the
    fitted reader (fit path: -(d/dtheta)[v . grad_w L_A(w*, theta)], v = H^-1 grad_w R, through the A outputs only).
  - **STOPFIT**: the direct path only, with w* detached from its fitting path. Everything else is identical.
  - **ALTCRIT**: an ordinary persistent critic of the same family, features, ridge and surrogate, updated by 10 full-batch Adam steps
    (lr 3e-3) on each episode's A rows. It uses the direct path only.

The computational graph is in the investigator's hand-back, section 4. The completeness of the FULL derivative was checked against
central finite differences on invented data, to relative errors 1.9e-10 (loop pattern) and up to 1.2e-9 (U0-size network). The
independent verifier re-checks it with separate code.

**Choice of recipe values.** All of them were set label-free: the declared recipient budget for n_A and n_E; NEURAL_RAW's critic
schedule for ALTCRIT; and a conventional ridge written before any computation (investigator's hand-back, section 6). No outcome entered
them.

## 3. Operational difference from closed recipes; prior work

- **NEURAL_RAW's own recipe (RAW-J).** Lagging online critics, with a gradient through the released outputs only. Here it is
  continued as the ALT control, and ALTCRIT is the same class with this card's reader family and surrogate.
- **SAR1's closed-form Fisher penalty (FB_PFD / FB_PFO).** An in-sample best response. By the envelope theorem its gradient equals a
  stopped-fit gradient (checked to 1.4e-14 on FB_PFD's own code). STOPFIT is its held-out analogue with other readers.
- **Plain continuation.** CONT (lambda = 0).
- **The ingredient tested.** The fit path. It is non-zero because the held-out outer criterion differs from the in-sample inner loss.
  The challenger's analysis says that in expectation it contains a term from the inner family's fitting bias and a term of order
  p / n_A. Neither targets the population information that a fresh 16-recipe bank measures.
- **Expected size.** On invented data the systematic share was 0.23-0.27 of the mean direct gradient, a turn of 5-9 degrees.
- **Prior work.** The investigator (hand-back section 5) and the challenger (section 3) read the methods of arXiv 2109.05535 and
  arXiv 2304.05135 and placed the operation among published bilevel and implicit-gradient methods. Neither reference establishes
  novelty or suitability. **No novelty is claimed.**

## 4. Controls

| Config | Role | Isolates |
|---|---|---|
| RF_FULL | CAND at S_k* | - |
| RF_TASK | CONT at S_k* (gating) | ordinary further task fitting |
| RF_STOP_LO, RF_STOP_HI | STOPFIT path bracket (gating) | the fit path |
| RF_ALTC_HI | ALTCRIT path, upper bracket point; LO recorded, not saved (gating) | exact fresh refit per episode vs a lagging critic |
| RF_RAWJ_HI | ALT path (RAW-J continued beyond E2), upper bracket point; LO recorded, not saved (gating) | the start's own recipe at comparable task cost |

- **ALTCRIT alias rule.** If the median relative critic lag on the deployed fit is below 1%, ALTCRIT is declared a NEAR_ALIAS of
  STOPFIT and is not counted as an independent control (challenger C3).
- **References.** START and the nine frozen references are read on the same grid.
- **lambda.** Calibrated once per unit on FULL at START (rho 1: lambda = median ||grad CE_occ|| / median ||grad S||, 16 fixed draws)
  and shared by STOPFIT and ALTCRIT.

## 5. Selection on MODEL_FIT only

- **Path.** Checkpoints every 10 steps from START: S10..S400. The RAW-J path runs S90..S480 from the U0 recipe.
- **CAND rule.** k* = the largest S_k such that every S_10..S_k keeps estimate + 2 household-clustered SE within these budgets, on the
  5-fold MODEL_FIT out-of-fold paired spend over the fold start (the U0 recipe):

  | Component | Budget |
  |---|---|
  | occupation NLL | 0.0060 |
  | occupation Brier | 0.0025 |
  | men's occupation Brier | 0.0028 |
  | women's occupation Brier | 0.0028 |
  | occupation accuracy | 0.0060 |

  These are SAR1 FB_PFO_R1's occupation budgets, unchanged. The rule uses no recovery diagnostic. The within-SEX rows are usefulness
  constraints. (Reviewer q2: there is no SEX-recovery selection of the operating point.)
- **Brackets.** STOPFIT, ALTCRIT and ALT are bracketed by the SAR1 rule on MODEL_FIT out-of-fold total-NLL spend over START. The
  residual cost differences at the identical checkpoint S_k* are recorded.
- **F0 gate (challenger C1).** Before any arm is fitted, the gate reads 64 episodes at the deployed START (MODEL_FIT only). The
  statistics are unbiased cross-group estimates over 8 groups of g_dir and g_fit in parameter space.
  - Label: MECHANISM_INERT iff the orthogonal share is < 0.05 and cos(E g_tot, E g_dir) > 0.995.
  - An inert version stops its unit and saves nothing.
  - This is a SEX-based MODEL_FIT diagnostic that decides only whether the family is fitted. It does not choose an operating point.
- **Logged along FULL** (F1): per 40-step epoch, the same statistics.
- **Logged in every episode arm:** exact and binormal held-out AUCs, the held-out orientation sign, the held-out degree-2 Fisher
  monitor, and the per-coordinate A-row SD.

## 6. Readings, falsifiers and predictions

- **Main bank.** R1 and R2, k0 and k1, FIT fraction 1.0.
- **Learning curve.** FIT fraction 0.5: the inherited `RFA1_devgrid_LC` fold rule, folds 0-1 of 4 within FIT. It uses paired SEs from
  the shared bootstrap draws. The 0.25 fraction is not run (reviewer q3 and q4).
- **Second split.** R3 and R4, k0-k3, read at CAND whenever CAND is G1-feasible and the grid is valid.
- **Falsifiers, registered:**

| Falsifier | Signal |
|---|---|
| F0 | MECHANISM_INERT |
| F1 | the orthogonal share below 0.05 before S_k* |
| F2 | no resolved FULL benefit over STOPFIT at matched spend on the fresh bank |
| F3, inner-family over-fit | FULL's descriptive gain over STOPFIT on the per-recipe tree, table and categorical readers (HGB15, ET25, LOOKUP, CATI) less than half its gain on the linear-logit reader LOG_LR |
| F4, learnability limitation | the benefit at the full budget smaller than at the half budget (rules LC) |
| F5, inversion | mostly negative held-out orientation, or an unchanged oriented fresh AUC |
| F6, surrogate exploitation | the binormal AUC falls but the exact AUC does not; or the Fisher-2 monitor rises; or an A-row SD shrinks more than 10x from START |

- **Predictions, registered:**

| Prediction | Probability |
|---|---|
| F0 active | 0.8 |
| k* between S20 and S200 | 0.7 |
| CAND G1-feasible on MODEL_SELECT at the point | 0.7 |
| CAND's point pair benefit over START between +0.003 and +0.020 | 0.6 |
| resolved over START | 0.3 |
| FULL vs STOPFIT at matched spend: tie (NOT_SEPARATED) | 0.75 |
| FULL vs STOPFIT: resolved FULL benefit | 0.10 |
| resolved FULL benefit over ALT | 0.10 |
| D3 | 0.02 |
| D4 | 0.01 |
| F4 triggered given a positive benefit | 0.5 |

- **Failure readings.**
  - MECHANISM_INERT: a design failure on these data.
  - A tie with STOPFIT: the fit path adds nothing beyond the held-out direct path at this budget.
  - A benefit not confirmed on the second split, or one that shrinks at the larger budget: reader- or budget-specific.
  - A benefit only against the weaker controls: partial, under its partial name.

## 7. Specifications

At most 6 new configurations per version, 12 in total for this card:
- RF_FULL;
- RF_TASK;
- RF_STOP_LO and RF_STOP_HI;
- RF_ALTC_HI;
- RF_RAWJ_HI.

A bracket point that does not exist is not saved. The revision (one, with its own isolating control) and follow-through use the
remaining 12 of the campaign's 24.
