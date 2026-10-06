# Exposure ledger

## Registered exposure statement (verbatim, before any new fit)

"The design is motivated by opened Adult development results, including the completed qpc confidence-capacity study. The fitting, inner-selection and assessment data have all been used historically. This is an exploratory, locked development comparison. Its nominal intervals condition on fitted artifacts and do not account for the adaptive research history. It is not fresh confirmation or a population privacy guarantee."

## What is already known before this study

The qpc study (evidence `9dd06da6b64e558e1c079f76e43982b60b327e63`) published assessment aggregates for every source map reused here:

| Release (qpc assessment means) | Value |
|---|---|
| U continuous, pair AUC | 0.858 |
| DIRECT-TASK i8o64 | Pair 0.849; occupation log-loss excess ≈ 0.0030 |
| FINE-TASK i8o64 | Pair 0.846 |
| JOINT λ 0.1 | Pair 0.813; occupation log-loss excess 0.008072 (upper bound 0.012122) |
| SEQ-12 / SEQ-21 λ 0.1 | Pair 0.816 / 0.816 |
| LOCAL λ 0.1 | Pair 0.834 |
| Decisions alone | Pair 0.739 |

The intermediate-λ design of this study was chosen knowing those numbers. They are acknowledged as motivation and must not be read as untouched evidence.

## Data roles (unchanged)

| Role | Rows | Use here |
|---|---:|---|
| OSF_DEFENSE_FIT | 15,434 | Partitions, prototypes, privacy fitting (SEX on this role only) |
| HEAD_VALIDATION | 1,500 | Historical head selection only; no head is fitted |
| AUDIT_FIT | 6,065 | Attacker fitting |
| INNER_SELECTION | 2,235 | Attacker selection, ordinary and headroom eligibility, comparators, nominees |
| OSF_DEVELOPMENT_ASSESSMENT | 13,936 (13,929 groups) | Masked until the pushed cbp EVALUATION_LOCK; opened once. Historically used: the mask prevents further leakage but does not make the rows fresh |

**Mask scope.** Assessment labels and prior per-person assessment predictions (qpc `outer__*` units) are never copied into the cbp store. They are never read by fitting, selection or debugging. cbp.admit copies no outer unit.

## Populations

- No new state, year or population is opened.
- ACS 2016 is spent, California development history is spent, and Texas 2018 carries a prior reservation.
- No other population (for example New York) is assumed untouched without reconciling both repositories and the private exposure ledgers.
- No confirmation data is spent by this study.

## Historical custody issue (recorded separately; not rewritten)

Earlier committed history on predecessor branches contains an identifying drive-folder name. No history rewrite or force-push is authorised, so it is not removed. New public files use placeholders only.
