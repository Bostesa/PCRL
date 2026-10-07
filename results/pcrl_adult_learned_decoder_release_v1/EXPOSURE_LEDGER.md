# Exposure ledger — lra

## Registered exposure statement (verbatim, prompt §4)

"This study is motivated by opened Adult development results from qpc, cbp and the earlier PCRL lineage, plus the opened
known-law fixture results of lcr. lcr did not fit or assess this Adult method. All real-data roles have been used
historically. This is exploratory development evidence. Nominal intervals condition on fitted artifacts and do not
correct for the adaptive research history. A masked assessment prevents additional selection leakage but does not make
the rows fresh. No confirmation population is opened."

## Roles (unchanged; ROLE_MANIFEST.json)

| Role | Rows | Use in lra |
|---|---|---|
| OSF_DEFENSE_FIT | 15,434 | task labels and SEX for the NEW supervised fitting: D1 decoders, C-TASK, weighted and constrained mappings |
| HEAD_VALIDATION | 1,500 | historical head selection only; not repurposed; no label read |
| AUDIT_FIT | 6,065 | attacker training (labels: SEX) |
| INNER_SELECTION | 2,235 | attacker selection and all nomination (labels: SEX, task labels for utility) |
| OSF_DEVELOPMENT_ASSESSMENT | 13,936 (13,929 exact-record groups) | sealed; unsealed only by `lra.assess` after the pushed EVALUATION_LOCK |

- **Input.** `adult_jcv.npz` (sha256 e0d9e54a…2f12): the 83 permitted columns in pinned order, with the same duplicate
  rules and group assignments. Source proxies are kept. Explicit sensitive labels, true task labels, person IDs and
  role indicators are never inference inputs.

## Material change in label use (stated plainly)

- **Fitting labels.** Task labels and SEX on OSF_DEFENSE_FIT are used for a NEW supervised fitting procedure: learned
  decoder probabilities, and coarse code assignments chosen under explicit utility and information budgets. The source
  codebooks were label-blind.
- **Unchanged.** The fine partitions stay fixed and label-blind. No new encoder, task head or feature extractor is
  fitted.
- **SEX.** SEX influences only the partition search. It never enters the per-token decoder objective and is not a
  deployment input.

## Staged exposure boundaries

- **Admission.** Teacher forward passes on the permitted inputs (inference; no labels), release re-encoding, and hash
  checks.
- **Correctness stage.** Synthetic known-law fixtures only. It starts after CORRECTNESS_LOCK is pushed.
- **First Adult label use.** No Adult label-based decoder solve or mapping fit runs before SCIENCE_LOCK is pushed.
- **Loaders.** The fit-only, audit-only and inner-only loaders never return assessment labels: assessment rows are
  masked to −1, and only `lra.assess` may unseal.
- **Assessment.** No assessment label is read before EVALUATION_LOCK is pushed.

## Cohorts

- No new state or year is opened.
- ACS 2016 is spent, and Texas 2018 has a previous reservation.
- No other cohort is assumed unexposed.

The realized use is appended at closeout.
