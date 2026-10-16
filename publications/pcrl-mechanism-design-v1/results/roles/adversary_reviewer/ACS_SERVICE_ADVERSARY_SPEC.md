# Adversary specification for the current ACS output service, checked against code

Scope: the Florida ACS 2019 two-task output service, as evaluated by the protected-tree replication (`acs_replication/`,
locked in `results/pcrl_acs_protected_tree_replication_v1/SCIENCE_LOCK.json`) and by the frozen-hybrids follow-up
(`acs_hybrids/`, `results/pcrl_acs_frozen_hybrids_v1/ANALYSIS_LOCK.json`). The reader definitions come from
`task_conditioned_representation/attacks.py`, and the slate was transported from `task_headroom/reader_worker.py`.
Worktree `research/pcrl-mechanism-design-v1` at `54f089d3e`. Every claim below cites `file:line`. I read code and
published aggregates only. No array was opened and nothing was run.

## Specification

- **Population and tasks.** Employed household residents aged 25–64 in the ACS 2019 1-Year PUMS for Florida
  (`acs_replication/schema.py:33-45`). Tasks: income above $50,000 (binary) and occupation group (5 classes)
  (`acs_hybrids/common.py:35-36`). Protected attribute: SEX, which is never a model input (`schema.py:46`).
- **Service.** For a person with permitted inputs x = (AGEP, WKHP, COW, SCHL, MAR, POBP) (`schema.py:55`), a frozen
  service computes `p_inc(x)` (2 probabilities) and `p_occ(x)` (5 probabilities). Recipient r's release is
  `W_r(x) = (p_r(x), argmax p_r(x))` (`acs_hybrids/common.py:66-70`; `acs_replication/registry.py:3-4, 45-47`).
- **Single recipient.** Receives only its own release for each target person.
- **Coalition.** Receives `(W_inc(x), W_occ(x))` for the same person, perfectly linked.
- **Granted knowledge.** Labelled auxiliary releases `{(W(x_i), S_i)}` for the ATTACK_FIT people (8,001 persons in 5,319
  households), and a labelled ATTACK_VALIDATION set (3,000 persons, 1,974 households) for choosing and orienting
  readers. Sizes are from `EXTRACTION_SUMMARY.json`. Both are random household splits of the same cohort as the targets.
- **Target and score.** DEV_REPORT persons (5,943 persons in 3,955 households). The score is the DEV_REPORT AUC for
  SEX of the reader chosen on ATTACK_VALIDATION. Uncertainty comes from a household bootstrap, conditional on the
  frozen models and the fitted readers (`ANALYSIS_LOCK.json:1147`).
- **Claims this access tests** (`SCIENCE_LOCK.json:286`):
  - Recipient worsening, with upper bound ≤ 0.010: single-recipient access.
  - Coalition benefit against TASK_REFIT, with lower bound > 0.020: coalition access.
  - A point-utility gate.

  The published labels stay RECIPE_EFFECT_NOT_ESTABLISHED and, for the hybrid, NOT_SUPPORTED
  (`ANALYSIS_LOCK.json:1133`).

## Bullet-by-bullet verification

| # | Statement | Verdict | Code evidence | Caveats |
|---|---|---|---|---|
| 1 | Each recipient receives its own task probabilities and argmax decision about a person. | **Verified in the evaluation.** | Release contract at `SCIENCE_LOCK.json:896`. The bank stores probability vectors only (`acs_hybrids/common.py:199-214`). Single-recipient views are built from one task's probabilities and argmax (`acs_hybrids/readers.py:35-38, 79-94`). The released object is described at `acs_replication/trees.py:117-120`. | The decision is argmax(p), so it adds no information beyond p. The base deployment `acs_replication/deploy.py:65-89` returns **both** tasks to one caller, so routing each task to its recipient is an operator duty there. The H_LJ package can release a single task (`acs_hybrids/hybrid_deploy.py:72-80`) and tests that each recipient's output can be computed from a package holding only that recipient's component (`:104-117`). |
| 2 | A coalition receives the perfectly linked pair. | **Verified.** | Same-row column concatenation at `acs_hybrids/readers.py:99-101`, after checking that the bank's row and household order equals each role's export (`:55-57`). The locks state the assumption: `SCIENCE_LOCK.json:9` ("perfect linkage of the two recipients assumed") and `ANALYSIS_LOCK.json:2`. | No linkage-error model. No coalition of more than two recipients. |
| 3 | Auxiliary SEX-labelled examples (ATTACK_FIT) are granted by the benchmark. | **Verified; this is an assumption, not a demonstrated capability.** | Readers load ATTACK_FIT and ATTACK_VALIDATION with SEX (`readers.py:52`) and fit on ATTACK_FIT SEX (`:70`). The admitted fields are listed at `common.py:108-109`. Roles are deterministic household hashes over one cohort (`acs_replication/schema.py:63-68, 76-91`), so the auxiliary and target people share one distribution **by construction**, and no household spans two roles. | Availability, size (about 11,000 labelled people), distributional match, and access to the **same frozen service's** releases for the auxiliary people (`bank[svc]['ATTACK_FIT']`, `readers.py:83, 99-101`) are all granted. The experiment does not show that recipients can obtain them in deployment. Readers are **not** given auxiliary task labels (`common.py:108, 116-117`) and do not use auxiliary inputs, so attacks that exploit auxiliary raw features (such as Song and Shmatikov's de-censoring) are untested. |
| 4 | Readers fit on ATTACK_FIT and are selected and oriented on ATTACK_VALIDATION. | **Verified.** | `fit_one` fits on ATTACK_FIT, orients by the sign of the ATTACK_VALIDATION AUC and scores DEV_REPORT (`readers.py:66-76`, using `task_conditioned_representation/attacks.py:87-91, 117-129`). Winners are chosen by maximum ATTACK_VALIDATION AUC (`readers.py:104-115`). Lock entries: `ANALYSIS_LOCK.json:1109, 1119, 1131`. | The reader hyperparameters are fixed (`attacks.py:117-123`; `ANALYSIS_LOCK.json:1098`), so the slate is small and finite. Recovery is a lower bound for this slate, not a certificate. |
| 5 | Readers do not receive target raw inputs, target SEX, true target task labels, model weights, gradients, teacher internals, leaf states or hidden representations. | **Verified as reader inputs. Implementation note below.** | Every reader matrix is built by `kinds()` from probabilities alone (`readers.py:35-38, 83, 99-101`). The bank holds probabilities only (`common.py:199-214`; `registry.py:3-4`). Leaf ids are never transmitted (`trees.py:15`), and hidden-output requests are refused (`deploy.py:5-7, 65-67`; `hybrid_deploy.py:72-74`). The contract excludes "leaf ids, representations, teacher outputs, labels, raw features, ids, weights or flags" (`SCIENCE_LOCK.json:896`). DEV_REPORT truth (SEX, task labels) can be read only through `dev_report_truth` after the committed REPORTING_LOCK (`common.py:139-163`), and only the evaluator calls it (`acs_hybrids/report.py:77-83`), never the reader worker (`readers.py:170`). | The reader worker **loads** the DEV_REPORT feature export, which contains the encoded inputs X, but only to obtain row and household ids and to check row order (`readers.py:53-57`; `common.py:121`). X never reaches a reader. The same holds for the ATTACK_FIT and ATTACK_VALIDATION feature exports. |
| 6 | Household and person identifiers enforce separation and linkage and are not reader features. | **Verified.** | Ids are hashed household and person keys (`acs_replication/extract.py:206-207`). Roles are assigned by household (`schema.py:72-91`). Ids drive the row-order checks (`readers.py:55-57`), the linkage of the pair (`:99-101`), the score files (`:164`) and the household bootstrap (`report.py:83, 101-110`). SERIALNO, SPORDER and PWGTP are forbidden model inputs (`schema.py:56-58, 166`). | — |
| 7 | One fixed release per person; repeated queries, insider access and model-access attacks are untested. | **Verified.** | There is one stored array per seed, service, role and task (`common.py:195-214`). Releases are deterministic: tree routing plus softmax (`trees.py:92-120`), and the neural model runs in evaluation mode (`deploy.py:80-86`). Stochastic channels and averaging are prohibited (`ANALYSIS_LOCK.json:1089`). The threat model excludes "model internals, insiders or interaction" (`SCIENCE_LOCK.json:9`). The reader worker never calls a deployment API (no `deploy` import in `acs_hybrids/readers.py`). | Because the service is deterministic, repeating a query returns the same release. Probing with chosen inputs, white-box or gradient access, insiders, recipients who also hold x or later see the outcome, and recipients that observe many service versions are all untested. |
| 8 | "Defense-aware" means refitting a reader to the actual defended auxiliary releases, not unrestricted adaptive or white-box access. | **Correct in substance, but the term does not appear in the ACS code or locks.** | Every reader is fit from scratch on the defended service's own ATTACK_FIT wire: marginal banks per wire (`readers.py:79-94`) and fresh pair readers per service (`:102-103`). No reader receives θ, the critics, the teacher or the training objective (bullet 5). | The term means different things elsewhere: the AAAI Tier 2 LRT, the cell-conditional attacker (`oar/study.py:180` at `3b5065805`), and the TCR1 transform (`attacks.py:114, 168`). If the manuscript uses "defense-aware" for ACS, define it as "refit to the defended releases". |
| 9 | The coalition reader bank includes readers that ignore either recipient. | **Verified.** | The pair candidates are the 7 pair readers plus 7 `IGNORE_occupation__*` plus 7 `IGNORE_income__*` aliases, 21 in all (`readers.py:109-115`; `ANALYSIS_LOCK.json:1104`). | Selection uses validation, so DEV_REPORT pair AUC is not forced to be at least the best single-recipient AUC. |

## Facts about the defending side, kept separate from the adversary

- The defenses were trained with MODEL_FIT task labels and SEX (`SCIENCE_LOCK.json:4, 724`;
  `acs_replication/leafcorr.py:3-4, 40`).
- They were trained against online critics (linear, MLP32, MLP64) on the same three views: income, occupation and pair
  (`leafcorr.py:10-11, 16-19, 41-42`). LOCAL uses income and occupation risk; JOINT adds pair risk.
- The post-hoc readers are fresh and household-disjoint from MODEL_FIT. They share nothing with the critics.
- Pipeline controls on every view: a shuffled-SEX null and a planted SEX bit (`readers.py:119-135`; acceptance rule in
  `ANALYSIS_LOCK.json` "controls").

## Contradictions with the bullets

None of substance. Two precision notes:
1. The per-recipient split is a property of the evaluation and of the H_LJ package, not of the base deployment API
   (bullet 1).
2. The target feature export is loaded into the reader process but never used (bullet 5).

The term "defense-aware" is absent from the ACS code (bullet 8). The adversary is not new: `SCIENCE_LOCK.json:9` itself
says "auxiliary-data recipient adversaries follow existing literature (not a new adversary)".
