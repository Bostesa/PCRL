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

- **Fitting labels.** The source (qpc/cbp) codebooks never read a TRUE TASK label: their task-only maps read no label,
  and their privacy maps read SEX on OSF_DEFENSE_FIT. New here, true task labels on OSF_DEFENSE_FIT enter:
  - every D1 decoder, including the fixed-map calibration controls and CLASS|D1;
  - C-TASK;
  - the weighted controls' task objective;
  - the fitting budgets.
- **Unchanged.** The fine partitions stay fixed and label-blind. No new encoder, task head or feature extractor is
  fitted.
- **SEX.** SEX on OSF_DEFENSE_FIT enters three places: the partition search of every privacy arm and weighted control,
  the local caps I_i(C-TASK), and the fitted and permutation MI diagnostics. It never enters the D1 objective and is
  not a deployment input.
- **In-sample fitting.** The U heads were trained on OSF_DEFENSE_FIT, so D1 calibrates against in-sample teacher
  outputs. The decoders hold fitting-row label counts and stay private.

## Staged exposure boundaries

- **Admission.** Teacher forward passes on the permitted inputs (inference; no labels), release re-encoding, and hash
  checks.
- **Correctness stage.** Synthetic known-law fixtures only. It starts after CORRECTNESS_LOCK is pushed.
- **First Adult label use.** No Adult label-based decoder solve or mapping fit runs before SCIENCE_LOCK is pushed.
- **Loaders.** The fit-only, audit-only and inner-only loaders never return assessment labels: assessment rows are
  masked to −1, and only `lra.assess` may unseal.
- **Assessment.** No assessment label is read before EVALUATION_LOCK is pushed.
- **Predecessor custody.** The predecessor custody chain (cbp → qpc → dpc → osf) DOES unseal
  OSF_DEVELOPMENT_ASSESSMENT, inside osf.closeout.backup. lra.closeout therefore runs it only after this study's
  assessment opening (the lock on origin plus a bound outer__ unit). A proof from code alone is not available.

## Cohorts

- No new state or year is opened.
- ACS 2016 is spent, and Texas 2018 has a previous reservation.
- No other cohort is assumed unexposed.

The realized use is appended at closeout.

## Realized use (appended at closeout, 2026-10-07)

- **OSF_DEFENSE_FIT.** True task labels and SEX were used for the registered new supervised fitting: 81 D1 decoders,
  3 C-TASK, 72 weighted and 15 constrained fits (SCIENCE_LOCK, from 05:22Z).
- **AUDIT_FIT and INNER_SELECTION.** AUDIT_FIT trained attackers; INNER_SELECTION selected (267 inner audits, controls,
  selection).
- **Assessment.** OSF_DEVELOPMENT_ASSESSMENT was unsealed once, by lra.assess, after EVALUATION_LOCK was pushed
  (07:47:30Z): 63 outer units, then inference.
- **HEAD_VALIDATION.** Not repurposed.
- **Correctness stage.** It ran on the synthetic pinned fixture laws only, after CORRECTNESS_LOCK.
- **Custody.** The attacker restore refit on AUDIT_FIT only; no assessment label was read in custody.
