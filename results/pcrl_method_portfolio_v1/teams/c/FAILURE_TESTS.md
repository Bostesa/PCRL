# Team C: failure tests and stop conditions (pre-lock)

## Before any real-data fit

1. **Allowed input and labels.** The fit path must open only role-specific exports containing permitted rows and labels. Instrument or independently inspect the full call chain; returning a masked object after transiently reading old assessment arrays fails. Refuse a loader that opens the all-role NPZ to filter after loading. Verify the pinned role counts, group hashes, row order, 83 columns and exclusions. HEAD_VALIDATION and old assessment person-level labels/predictions remain unread.
2. **Mask semantics.** Toggle categorical fields by complete one-hot blocks; tie `education` and `education-num` as one raw-fact atom. SEX, race, income, occupation and task labels are absent from predictors. Reconstruct the transmitted vectors independently from frozen mask, input and head. Confirm that every released task probability depends only on its own selected features; no appended clean teacher output.
3. **Premise.** The mask is one fixed global mask per purpose, learned on permitted fitting rows. Deployment reads neither SEX nor task labels nor other-party private inputs. Check that `relationship` is excluded from the candidate and restricted attribution selectors, as registered before outcomes. Unrestricted task-only and independent controls may retain it at the same recipient budgets. Distinguish full-feature and drop-relationship controls.
4. **Finite-law suite.** Freeze the law definitions and expected assertions before running the selector: feasible null, a removable sensitive distinction, complementary coalition clue, no complementarity, and useful confidence versus decision-only utility. The fixed logistic plus depth-2-tree SEX scorecard must detect its planted XOR/complementary clue; a logistic-only null result there is a failed premise test, not evidence of privacy. Exhaustively enumerate tiny mask pairs where practical; the heuristic is compared with the exact optimum and is never called an exact solver. A candidate win on a synthetic law is capability evidence only.
5. **Capacity.** Validate each arm's recipient and union atom counts and transmitted column counts before evaluation. An unmatched capacity vector cannot establish joint advantage. Ties and infeasible counts use only the predeclared rule, never INNER_SELECTION labels.

## Real-data engineering controls

- **Full-view serialization:** For each recipient and aligned pair, round-trip raw selected values, probability vectors, decisions, mask/feature names, head/calibrator identity and every flag. Independently hash views and compare with deployment output. A score-only serialization fails the feature contract.
- **Shortcut probe:** When relationship is retained (`C_FULL`), the attacker must see Husband/Wife one-hot values. Report their coverage and SEX recovery within those rows. For every selected raw feature, test a raw-only reader. If complete-view AUC is implausibly below a raw-only AUC or below an ignore-one-recipient route, treat it as an attack or serialization failure.
- **Leak controls:** Shuffled SEX must give null recovery by the frozen threshold; a constant reader establishes prevalence; a planted SEX bit in a separate engineering copy must be found by the same attack pipeline. Mask identity is supplied to defense-aware readers even though it is global within an arm. A failed control invalidates the affected attack result.
- **Independent coalition check:** Rebuild the raw union from both transmitted vectors, deduplicate shared atoms, keep aligned people, append both outputs/metadata, and verify neither local view contains the other party's private fields. Report three attack AUCs, including ignore-recipient alternatives. Do not reduce coalition recovery to a difference between local AUCs.
- **Utility:** Report accuracy, mean NLL, multiclass Brier, per-class recall, and confidence against both frozen U0 and Ucal, for both tasks and each paired seed. Fit the same H-GLOBAL-TEMP family for every arm. A point estimate within cap is only promotion-eligible; a crossing simultaneous interval is `UNRESOLVED`.
- **Fresh nonlinear attack:** Fit the frozen nonlinear slate on ATTACK_FIT_NEW and score only on INNER_SELECTION. A gain seen only with a weak linear reader is not a privacy gain. Public mask and model information is included.

## Scientific kill and attribution rules

| Observation | Disposition |
|---|---|
| All privacy-reducing masks violate either task's mean utility cap against U0 or Ucal | Stop Team C real-data development; report utility shortfall. |
| Raw-only or complete-view fresh nonlinear coalition attack removes the apparent privacy gain | Stop the claimed privacy result; keep code/results as development evidence. |
| `C_DROP_REL` matches the candidate at usable utility | Attribute the result to simple removal of a known proxy; no joint selector claim. |
| Either task-only arm, either independent arm, `C_GENERIC` or either sequential order matches the candidate under equal recipient/union capacity | Report the winning simpler/known control; no joint-design lead. |
| Local SEX AUC increases by more than the common 0.010 cap while coalition AUC decreases | Failure of the no-leakage-shift objective. |
| Candidate has lower measured leakage only because it transmits fewer raw facts or scores a different view | Mark comparison invalid; restore matched capacity/view before interpreting. |
| Utility point estimates meet caps but simultaneous intervals cross a boundary | `PRECISION_LIMITED`, eligible for promotion under the frozen rule but not a passed criterion. |
| Any claimed improvement relies on a selected seed, post-outcome budget, different calibrator, or changed attacker bank | Invalid comparison; preserve outputs and record a dated amendment before rerun. |

No item above grants permission to open the old assessment. These are repeatedly used Adult development roles, and any favourable result remains exploratory.
