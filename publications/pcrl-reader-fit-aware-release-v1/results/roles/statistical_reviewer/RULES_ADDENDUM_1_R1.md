# RFA1 rules addendum 1: revision R1 (card RF_RFIT_R1) and the beside family-change reading

Statistical reviewer, written 2026-10-25 22:24-22:27Z (`date -u` 22:26:47Z at completion), after the registered v1 decision record
and before any R1 fit.
The frozen rules (`rfa/rules.py` 76baa04a..., `RULES.json` 4e03270c..., `RULES.md` 1548b6de...) are unchanged; they were committed
only in the registration commit 43830ae19. This addendum adds one beside reading and record requirements. It changes no label, gate
or class.

## 1. The frozen rules read R1 unchanged

A record with `family` RFA (or FAM), `revision` 1, `card_committed_before_any_model_select_outcome` False (card first committed
22:23:27Z, after the first MODEL_SELECT read at 21:59:53Z) and R1's own ids (CAND RF1_FULL, CONT RF1_TASK, STOPFIT [RF1_STOP_LO,
RF1_STOP_HI], ALTCRIT [RF1_ALTC_HI], RAWJ [RF1_RAWJ_HI], `lo_not_saved` [ALTCRIT, RAWJ], R1's `f0`, R1's
`altcrit_median_relative_lag`, selection data MODEL_FIT_OOF) is read by `decisions()` as `ADAPTIVE_R1`:
- every label is prefixed `ADAPTIVE_R1__`;
- R1's STOPFIT is its isolating control: the same episodes, the same three families and R1's re-calibrated lambda, with the fit path
  detached;
- `campaign_summary` counts K = 2 D2 looks (registered v1 + ADAPTIVE_R1).

## 2. Record requirements for R1 (assembly, not rules)

1. **Bitwise-equal TASK / RAW-J points.** Keep the R1 ids in `grid`, `costs_vs` and `nolookup`, and record the identity with a v1
   configuration as a plan disclosure. Never record it in `alias_of`: that key maps a path's saved ids to START, CONT or an earlier
   id of the same reading.
2. **Learning-curve validity.** `learning_curve.grid_valid` must be the half-budget (`@0.5`) cells' own `grid_validity`, with the
   same seeds, bootstrap seeds, B = 400 and minimum set. RULES.md 9 requires this; v1's record passed the main bank's validity
   instead (claims review, item 2).
3. **Falsifier detail.** Save `falsifier_detail` (F3's per-recipe gains against STOPFIT's bracket, its tree-table mean and the
   LOG_LR gain; F1, F5 and F6 statistics) beside DECISIONS.json.

## 3. Beside reading FC_R1: the family change RF1_FULL vs RF_FULL (descriptive; never gating)

**Inputs** (R1's bank, R1's registered cells, MODEL_SELECT):
- the grid entry (pair and both views) of RF1_FULL against RF_FULL, a beside member of R1's banks;
- the paired costs of RF1_FULL against RF_FULL (`total|nll` and the 8 secondary metrics).

**Computation** (nrm.rules unchanged):
- c = NR.comparison(entry);
- d = NR._primary(costs) = pooled cost(RF1_FULL) - cost(RF_FULL), with its SE;
- each secondary metric is pooled by NR._pooled and marked resolved-higher iff its value - 2 SE > 0.

**Spend state** (from RF1_FULL's side):

| state | condition |
|---|---|
| `R1_SPENDS_NO_MORE` | d <= 0 |
| `R1_SPENDS_MORE_UNRESOLVED` | 0 < d and d - 2 SE <= 0 |
| `R1_SPENDS_MORE_RESOLVED` | d - 2 SE > 0 |

**Label** (first applicable; every label prefixed `ADAPTIVE_R1__`):

| label | condition |
|---|---|
| `INCOMPLETE` | any input missing |
| `FAMILY_CHANGE_RESOLVED` | AGG_POSITIVE pair benefit, both recipient gates pass, the spend state is not `R1_SPENDS_MORE_RESOLVED`, and no secondary metric is resolved higher |
| `FAMILY_CHANGE_WITH_HIGHER_SPEND` | AGG_POSITIVE and both gates pass, but R1 spends resolved more (total NLL or a secondary metric) |
| `FAMILY_CHANGE_RECIPIENT_GATE_FAILED` | AGG_POSITIVE, but a recipient gate fails |
| `FAMILY_CHANGE_POINT_ONLY` | mean > 0, not resolved |
| `FAMILY_CHANGE_NOT_POSITIVE` | otherwise |

**Always reported:** the pair mean, se_agg and version means; both single views; d; the 8 secondary differences.

**Meaning.** The two releases are compared at their own operating points, each chosen by the same MODEL_FIT rule, so R1 differs
from v1 in its inner family set, its re-calibrated lambda and its k*. FC_R1 is a family-change contrast, never a component gain and
never an attribution to SOFTBIN. It does not enter D2, D3, D4, the learning curve, D2R or the outcome class.

## 4. Wording any R1 sentence must carry

- R1 is ADAPTIVE. It was designed after v1's MODEL_SELECT decision record, and its new family targets the reader recipes that won on
  MODEL_SELECT (CATI1, ET25, the trees; F3). Any R1 reduction of those recipes' recovery on the same SCORE households is adaptively
  targeted: it is evidence about this procedure on reused people, not information removal and not protection against other
  readers.
- Report R1 beside the registered v1 reading, with K = 2 D2 looks and no multiplicity correction.
- A positive R1 result would need households not used here before it could be confirmed.
