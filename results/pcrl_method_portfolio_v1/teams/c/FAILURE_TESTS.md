# Team C: failure tests and stop conditions (pre-lock)

## Before any real-data fit

1. **Allowed input and labels.** The fit path must open only role-specific exports containing permitted rows and labels. Instrument or independently inspect the full call chain; returning a masked object after transiently reading old assessment arrays fails. Refuse a loader that opens the all-role NPZ to filter after loading. Verify the pinned role counts, group hashes, row order, 83 columns and exclusions. HEAD_VALIDATION and old assessment person-level labels/predictions remain unread.
2. **Mask semantics.** Toggle categorical fields by complete one-hot blocks; tie `education` and `education-num` as one raw-fact atom. SEX, race, income, occupation and task labels are absent from predictors. Reconstruct the transmitted vectors independently from frozen mask, input and head. Confirm that every released task probability depends only on its own selected features; no appended clean teacher output.
3. **Premise.** The mask is one fixed global mask per purpose, learned on permitted fitting rows. Deployment reads neither SEX nor task labels nor other-party private inputs. Check that `relationship` is excluded from the candidate and restricted attribution selectors, as registered before outcomes. Unrestricted task-only and independent controls may retain it at the same recipient budgets. Distinguish full-feature and drop-relationship controls.
4. **Finite-law suite.** Freeze the law definitions and expected assertions before running the selector: feasible null, a removable sensitive distinction, complementary coalition clue, no complementarity, and useful confidence versus decision-only utility. The fixed logistic plus depth-2-tree SEX scorecard must detect its planted XOR/complementary clue; a logistic-only null result there is a failed premise test, not evidence of privacy. Exhaustively enumerate tiny mask pairs where practical; the heuristic is compared with the exact optimum and is never called an exact solver. A candidate win on a synthetic law is capability evidence only.
5. **Capacity.** Validate each arm's recipient and union atom counts and transmitted column counts before evaluation. An unmatched capacity vector cannot establish joint advantage. Ties and infeasible counts use only the predeclared rule, never INNER_SELECTION labels.

## Real-data engineering controls

- **Full-view serialization:** For each recipient and aligned pair, round-trip raw selected values, probability vectors, decisions, mask/feature names, head/calibrator identity and every flag. Independently hash views and compare with deployment output. A score-only serialization fails the feature contract.
- **Shortcut probe:** When relationship is retained (`C-FULL-FEATURE`), the attacker must see Husband/Wife one-hot values. Report their coverage and SEX recovery within those rows. For every selected raw feature, test a raw-only reader. If complete-view AUC is implausibly below a raw-only AUC or below an ignore-one-recipient route, treat it as an attack or serialization failure.
- **Leak controls:** Shuffled SEX must give null recovery by the frozen threshold; a constant reader establishes prevalence; a planted SEX bit in a separate engineering copy must be found by the same attack pipeline. Mask identity is supplied to defense-aware readers even though it is global within an arm. A failed control invalidates the affected attack result.
- **Independent coalition check:** Rebuild the raw union from both transmitted vectors, deduplicate shared atoms, keep aligned people, append both outputs/metadata, and verify neither local view contains the other party's private fields. Report three attack AUCs, including ignore-recipient alternatives. Do not reduce coalition recovery to a difference between local AUCs.
- **Utility:** Report accuracy, mean NLL, multiclass Brier, per-class recall, and confidence against both frozen U0 and Ucal, for both tasks and each paired seed. Fit the same H-GLOBAL-TEMP family for every arm. A point estimate within cap is only promotion-eligible; a crossing simultaneous interval is `UNRESOLVED`.
- **Fresh nonlinear attack:** Fit the frozen nonlinear slate on ATTACK_FIT_NEW and score only on INNER_SELECTION. A gain seen only with a weak linear reader is not a privacy gain. Public mask and model information is included.

## Scientific kill and attribution rules

| Observation | Disposition |
|---|---|
| All privacy-reducing masks violate either task's mean utility cap against U0 or Ucal | Stop Team C real-data development; report utility shortfall. |
| Raw-only or complete-view fresh nonlinear coalition attack removes the apparent privacy gain | Stop the claimed privacy result; keep code/results as development evidence. |
| `C-DROP-RELATIONSHIP` matches the candidate at usable utility | Attribute the result to simple removal of a known proxy; no joint selector claim. |
| Either task-only arm, either independent arm, a compatible prior control, or either sequential order matches the candidate under equal recipient/union capacity | Report the winning simpler/known control; no joint-design lead. |
| Local SEX AUC increases by more than the common 0.010 cap while coalition AUC decreases | Failure of the no-leakage-shift objective. |
| Candidate has lower measured leakage only because it transmits fewer raw facts or scores a different view | Mark comparison invalid; restore matched capacity/view before interpreting. |
| Utility point estimates meet caps but simultaneous intervals cross a boundary | `PRECISION_LIMITED`, eligible for promotion under the frozen rule but not a passed criterion. |
| Any claimed improvement relies on a selected seed, post-outcome budget, different calibrator, or changed attacker bank | Invalid comparison; preserve outputs and record a dated amendment before rerun. |

## Dated Phase I disposition: 2026-10-08

The queued `C-TEST-SCORECARD-BUILDER-016` fixed XOR clue check failed: the declared depth-two raw SEX tree attained reversible AUC 0.6123299693 against a predeclared >0.90 check. The other seven scorecard tests passed. A greedy depth-two split can miss a clue whose individual coordinates have no marginal SEX signal. The original failed receipt and law remain intact. Team C therefore cannot claim reliable complementary-clue detection or a joint-design advantage from this scorecard. It may proceed only as a transparent, fixed grouped-mask and retraining baseline, with fresh raw-union and complete-view attacks. No post-outcome proxy change or utility-threshold change is admitted under the current candidate identity.

S1's prospective C-SIMPLE register has seven active standard arms (`C-SIMPLE-FULL`, `C-SIMPLE-REL0`, `C-SIMPLE-TASK-SHARED1`, `C-SIMPLE-INDEP1`, `C-SIMPLE-COMMON1`, `C-SIMPLE-SEQ-IO1`, `C-SIMPLE-SEQ-OI1`) plus the two zero-fit failed/skipped prototype slots. The active controls and encoded-capacity rule are in STRONGEST_CONTROL.md. Raw-only and defense-aware fresh attacks remain required; missing them blocks a complete comparison.

## Dated prefit task-only custody correction: 2026-10-08

The active task-only control now fits a separate ten-mask scorecard using only features, income, occupation and group IDs from the admitted defense-fit role. Its internal group split and class-support repair receive only the two task labels. The selector consumes only this card's task deltas. Charge twenty additional task proxy fits per seed, for sixty active scorecard fits per seed including the original privacy card. A test checks the builder's missing SEX argument and the class-repair label keys. If the runner uses task deltas from the privacy scorecard, task-only custody fails and the affected arm is invalid; preserve the original receipts and rerun only under S1's dated amendment.

## Dated prefit worker and view-custody correction: 2026-10-08

The separate task-only OS worker must not request the FIT SEX file or import the privacy scorecard builder. Both Team C fit workers initially read FIT/CAL roles only; opening ATTACK_FIT_NEW or INNER_SELECTION features before the mask, heads, calibrators and packages are frozen fails the role boundary. A postfit forward worker verifies an immutable handoff before reading ATT/INNER features only. S2's later attack job reads their SEX labels. Synthetic sentinels refuse forbidden reads, a second process tests the private fit lock, and complete-wire NPZ/JSON round trips must reproduce every local and coalition AttackView and task probability vector. A missing or changed handoff, role/arm/seed, source hash, package hash, wire field, person order or restored decision blocks the affected C result.

No item above grants permission to open the old assessment. These are repeatedly used Adult development roles, and any favourable result remains exploratory.
