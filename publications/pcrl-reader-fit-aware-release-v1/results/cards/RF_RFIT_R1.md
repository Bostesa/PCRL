# Card RF_RFIT_R1 (family FAM, revision R1; ADAPTIVE: designed after the registered v1 reading RFA_v1)

Coordinator, 2026-10-25, written after the v1 MODEL_SELECT decision record and before any R1 fit; committed with the code it governs
(`rfa/fam/rfit_r1.py` by the mechanism investigator, `rfa/fam/fit_r1.py`, and the unchanged v1 modules it calls). This is the one
substantive revision the protocol allows. Its readings carry the ADAPTIVE_ prefix. Nothing here claims that it works or is novel.

## 1. Observed failure (recorded v1 evidence)

From `claims/RFA_v1/DECISIONS.json`, the RFA1_MAIN and RFA1_LC grids and the MODEL_FIT fit records:
- D2 is PARTIAL_START_ONLY:

  | Comparator | Point | Pair benefit (se_agg) |
  |---|---|---|
  | START | - | +0.0127 (0.0059), resolved |
  | CONT | - | +0.0086 (0.0092) |
  | STOPFIT | RF_STOP_HI | +0.0028 (0.0045) |
  | ALTCRIT | RF_ALTC_HI | +0.0005 (0.0010 floor) |
  | ALT | RF_RAWJ_HI | -0.0016 (0.0084) |

  The derivative through the reader fit did not separate from the stopped-fit or the ordinary alternating critic.
- F0 found the fit path active but small: orthogonal share 0.10 and 0.11, a turn of about 6 degrees.
- **F3 triggered.** The pair winners for RF_FULL are categorical-interaction, extra-trees and MLP readers on probabilities and
  decisions (CATI1, ET25, MLP64, IGNORE_income__ET25). Against STOPFIT's bracket point, the gain on these tree, table and
  categorical readers was below half the gain on the linear-logit reader. Against START, the per-recipe pair SCORE AUC means fell:

  | Reader | Fall vs START |
  |---|---|
  | LOG_LR | -0.014 |
  | ET25 | -0.009 |
  | CATI1 | -0.009 |
  | HGB15 | -0.005 |

  The inner families (linear and RFF64 on standardized logit differences) contain neither decision-cell nor binned structure.
- F4 triggered: the benefit against RF_TASK, RF_ALTC_HI and H_LJ is smaller at the full FIT budget than at its half.

## 2. The changed operation (one change)

The inner reader family set becomes {SIMPLE, RFF, SOFTBIN}. SOFTBIN is a bounded, differentiable, table-like ridge-logistic reader. Its
features per view are:
- soft one-hot bins of each standardized, soft-bounded logit-difference coordinate;
- a soft decision vector (a softmax of the logits at a fixed temperature: the wire's argmax decision with a derivative);
- for the pair view, the products of the income and occupation soft decisions (a categorical interaction).

Its frozen values are in `rfa.fam.rfit_r1.RECIPE_RF_R1`, written into the module before its first run, label-free
(`roles/investigator/HANDBACK_2.md`):
- 8 Gaussian soft bins per coordinate, at standard-normal quantile centres, width 0.3;
- soft decisions at temperature 0.25 in standardized units (the decision scale is the detached A-row SD of the recipient's logit
  differences);
- the 10 income x occupation soft-decision products in the pair view;
- a softbin ridge of 1e-3 (SIMPLE and RFF keep 1e-2).

The widths are 38 features for the occupation view and 51 for the pair view. With SIMPLE and RFF only, the module reproduces v1's
gradients bitwise. The complete FULL derivative matches central finite differences to 8.8e-10 in the registered setting.

**Everything else is v1's**: START, the occupation-only loop, episodes and sizes (n_A about 4,096, n_E about 2,400), Newton and
guards, the surrogate (AUC_b - 1/2)^2 (now averaged over three families), the lambda rule (recalibrated by the same rule on R1's FULL
objective), the 10-step path, BUDGET_OCC, the bracket rules and the F0 gate. The inner readers are still fitted on realized MODEL_FIT
SEX labels.

## 3. Prediction and the controls that isolate the change

| Config | Role | Isolates |
|---|---|---|
| RF1_FULL | CAND (ADAPTIVE_R1) at S_k* | - |
| RF1_STOP_LO, RF1_STOP_HI | STOPFIT path bracket (gating) | the fit path within R1 |
| RF1_ALTC_HI | ALTCRIT (gating) | a lagging critic of R1's families |
| RF1_TASK | CONT at S_k* (gating) | - |
| RF1_RAWJ_HI | ALT (gating) | - |

- **Family change.** It is isolated by reading RF1_FULL against the v1 candidate RF_FULL on the same cells. That comparison is
  reported beside D2 with the cost difference, and RF_FULL is a member of R1's banks.
- **Predictions, registered before the R1 fit:**

  | Prediction | Probability |
  |---|---|
  | F0 active, with a larger orthogonal share than v1's | 0.6 |
  | RF1_FULL G1-feasible on MODEL_SELECT at the point | 0.7 |
  | FULL_R1 resolved over STOPFIT_R1 at matched spend | 0.10 |
  | FULL_R1's pair AUC below v1 RF_FULL's at the point | 0.45 |
  | the ET25 / CATI1 / HGB15 pair AUCs fall more than under v1 | 0.45 |
  | D2 met | 0.04 |
  | D3 | 0.03 |
  | D4 | 0.01 |
  | the learning-curve limitation again | 0.6 |

- **Falsifiers.** The card RF_RFIT_v1's F0-F6, unchanged.

## 4. Specifications

At most 6 new configurations per version: RF1_FULL, RF1_TASK, RF1_STOP_LO, RF1_STOP_HI, RF1_ALTC_HI and RF1_RAWJ_HI, 12 in total.
That uses the remaining 12 of the campaign's 24. A TASK or RAW-J point bitwise equal to a v1 configuration is saved again under its R1
id; the plan records the identity, and the grid counts the identical wire once by its alias detection.
