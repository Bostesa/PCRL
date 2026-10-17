# Manuscript and claims handoff (additive; no manuscript or claims branch was changed)

Target branches named for convenience only. These are the manuscript lineage (for example `research/combined-paper-advisor-revision-v1`) and the claims lineage (`research/pcrl-claims-foundation-v1`). Neither was modified by this follow-up. A maintainer who adopts any sentence below should cite this package: `publication/pcrl-scm-utility-matched-v1`, research branch `research/pcrl-scm-utility-matched-v1`.

## May be stated (exploratory engineering evidence on previously used Florida ACS 2019 development records)

- **Decision.** `COMPARISON_CLOSED_INCONCLUSIVE_AT_REGISTERED_TOLERANCE` (claim M `MATCH_DID_NOT_CARRY_OVER`). Qualifiers:
  - DEV utility confidence `CONFIDENCE_UNRESOLVED`;
  - claim C `SUBGROUP_UNRESOLVED`;
  - ORD_NLLM DEV match `MISMATCH_ESTABLISHED` (seed 0) and `EQUIVALENCE_UNRESOLVED` (seed 1), with SCM spending more on both;
  - claim B `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`.

  Claim K is also `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`.
- **Conservative SCM endpoint keeps utility, not the men's guard.**
  - A conservative Shared-Cell Merging endpoint was nominated on MODEL_SELECT (threshold 0.008; 298 / 300 cells).
  - On DEV_REPORT it passes all 24 original utility point clauses.
  - The men's within-SEX occupation Brier excess is 0.0054 / 0.0064, above the 0.005 guard at the point. Its status is unresolved, not established harm.
  - The 24 utility confidence statuses are 3 PASS, 21 UNRESOLVED and 0 violations.
- **Matched-loss question: inconclusive.**
  - The control was matched on MODEL_SELECT occupation NLL within 0.001 nats per seed.
  - SCM's linked-pair SEX recovery read 0.034 / 0.025 AUC lower than that control on DEV_REPORT, with clause bounds above 0.
  - The match did not carry over: on DEV_REPORT, SCM spent +0.0070 / +0.0017 nats more occupation NLL, and also more income NLL (+0.0064 / +0.0035; above zero on seed 0, unresolved on seed 1).
  - The registered label is `MATCH_DID_NOT_CARRY_OVER`. The comparison is therefore not cost-matched, and the advantage is not separated from extra loss spent.
- **Ordinary merging barely moved recovery at the loss levels measured** (descriptive; on DEV_REPORT no ordinary release reached SCM's spend):
  - C0 minus the MODEL_SELECT-matched ordinary control on DEV_REPORT: +0.005 / +0.005, UNRESOLVED;
  - on MODEL_SELECT (cross-fitted proxies), the ordinary proxy stays within 0.645–0.649 from 0 to 0.0145 nats on seed 0 and within 0.643–0.648 from 0 to 0.0125 nats on seed 1.
- **No measured edge for SCM's ordering over the published IBSI objective (descriptive).**
  - Selected at a MODEL_SELECT occupation-NLL match, the published IBSI objective read within about 0.01 AUC of SCM on DEV_REPORT.
  - SCM's pair benefit was +0.006 / −0.010, unresolved on both seeds. This is not equivalence.
  - Its match also did not carry over (+0.0011 / −0.0039 nats). On seed 1 it fails three DEV utility point clauses.
  - Its finer partition can be under-read more than SCM's.
  - The component sequence was not tested, because the matched claim was not established.
- **Measured limitation.** A ±0.001-nat held-out loss match cannot be certified at these sample sizes.
  - Paired DEV_REPORT standard errors are 0.0016–0.0020.
  - The MODEL_SELECT-matched ordinary control's held-out expenditure fell from 0.0083 (MODEL_SELECT) to 0.0043 / 0.0054 (DEV_REPORT), while SCM's moved from 0.0082 / 0.0084 to 0.0113 / 0.0072.
- **Not competitive with the frozen references** (claim B `NOMINEE_FAILS_DEV_SUBGROUP_POINT_GUARD`).
  - Pair benefits against NEURAL_RAW and H_LJ are unresolved except H_LJ seed 0, and all four are below 0.020 at the point.
  - SCM's income recipient reads higher than both references at the point.
  - Its occupation recipient reads higher than H_LJ's, an established violation on seed 0.

## Must not be stated

- That SCM's advantage survives matched occupation loss, or that it was established at matched utility. The match did not carry over.
- That SCM is competitive with, or superior to, NEURAL_RAW or H_LJ.
- That SCM is new, a new algorithm or a joint design. It is an existing-method engineering combination.
- That a point match within 0.001 is equivalence. Equivalence was never established.
- That matching occupation NLL matched the whole utility vector. Income NLL, Brier and the men's burden differ.
- That PUB_IBSI and SCM are equivalent, that PUB_IBSI "accounts for" SCM's difference from ordinary merging, or that the SEX-guided releases' lower recovery is separated from the extra occupation and income loss they spent.
- That the lookup reader is population-optimal, or that any recovery AUC is a privacy guarantee.
- That this changes the source benchmark's labels. They stand (source `HANDOFF.json`):
  - A (recipe vs twin): `RECIPE_EFFECT_NOT_ESTABLISHED`;
  - attribution vs PUB_IBSI and SHUF: `NOT_TESTED_RECIPE_NOT_ESTABLISHED`;
  - B: `NOMINEE_FAILS_DEV_UTILITY_GATE`;
  - C: `SUBGROUP_UNRESOLVED`;
  - D and E: `NOT_CLAIMED`.
- That the result generalises to every shared partition.
- That the missing seed-1 NEURAL_RAW package, or any other missing coverage, is a demographic-inference result or a method win.
- That this justifies opening the sealed assessment.

## Required caveats

- **Previously used records.** The DEV_REPORT and MODEL_SELECT people had been used before. The design was informed by the source's published results, and the registration fixes execution order only.
- **Conditioning.** Intervals are conditional on the fitted releases and readers and exclude the selection and development history. The two seeds share people.
- **Experimental assumptions.** The demographic-labelled auxiliary records (ATTACK_FIT, 8,001), accurate linkage and distributional similarity are explicit assumptions of the controlled benchmark.
- **Reader budget.** Against the MODEL_SELECT-matched ordinary control, SCM keeps 2.3 / 2.8 times as many occupied pair vectors on ATTACK_FIT (297 / 299 against 129 / 105), with a median of 16 / 17 against 53 / 64 people per vector. A fixed reader budget can under-read finer partitions. That favours SCM against ordinary merging and favours PUB_IBSI (median 13) against SCM.
- **Redundancy.** For every grouped release in the bank (the check was not run for NEURAL_RAW or H_LJ), the occupation output identifies the shared cell and the income output is a function of it. A Bayes-optimal reader gains nothing from the linked pair beyond the occupation output for these people. Differences between pair and occupation-only readers are differences between fitted readers.

## Suggested sentence

"In an exploratory follow-up on previously used development records, a conservative shared-cell merging endpoint met all 24 utility clauses at the point (simultaneous confidence unresolved). It exceeded the men's within-sex occupation Brier guard at the point (unresolved, not established harm). Its linked-pair sex recovery read lower than an ordinary-merging control selected for matching held-out occupation log-loss in the selection role. That match did not carry over to the evaluation role, where the merged release spent more occupation and income log-loss, so the comparison is inconclusive at the registered 0.001-nat tolerance, a tolerance these sample sizes cannot certify. The fixed reader budget may under-read the finer merged partition, and the release was not competitive with the frozen neural and tree references."
