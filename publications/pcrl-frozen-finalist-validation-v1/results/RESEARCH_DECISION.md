# Decision: frozen-finalist validation of `B3CTL_UNCOUPLED_MIX50_C6A`

**Scope.** Statistical machine-learning research on public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, in the existing development roles only. The service predicts:
- an income class: 2 classes, PINCP × ADJINC / 10^6 above $50,000;
- an occupation group: 5 classes, OCCP mapped to the five civilian broad groups.

It uses AGEP, WKHP, COW, SCHL, MAR and POBP. Each recipient receives its own probability vector and argmax decision. SEX is the dataset's recorded demographic column, which separate evaluation readers try to infer from the released predictions. SEX is never a service input.

**What this campaign did.** It changed nothing in the candidate. It validated the frozen finalist of the source campaign: research branch `research/pcrl-autonomous-method-development-v1` at `9e2b6c567`, publication `5901f7c73`.

**Status of the evidence.** Development robustness evidence on already-used records. It is not fresh confirmation, and it does not erase the source's adaptive selection history (`EXPOSURE_AND_SELECTION_AUDIT.md`).
- **DEV_REPORT had been read 10 times across the lineage before this campaign:** 5 evaluations and 5 verifier replays.
- **This campaign adds two reads.** The first is one logged coordinator re-read at 22:27:16Z, after the registration receipt `55b017918` and the reporting receipt `8993a4bdd` were published. The second is one separately logged independent-verifier replay at 22:50:37Z (unit VER_DEVREPORT_V1).
- **Seed 0 and seed 1 are not replications.** They are two fits of each release, evaluated on the same DEV_REPORT people with shared bootstrap weights. Agreement across seeds is robustness to the release seed.

## The five distinctions

**1. Reproduced source result: yes, exactly.**
- The admitted bank, packages, readers and records match their recorded hashes: 414 copies, no problems.
- Recomputed from the admitted stored reader scores, the ORIGINAL scorecard reproduces every source endpoint (70 of 70; maximum point and SE difference 0.0) and every source claim label:
  - E is `DEV_POINT_ELIGIBLE`;
  - B is `SUPERIOR_TO_ALL_ELIGIBLE`, with the 0.020 target not established;
  - M is `MECHANISM_ATTRIBUTION_ESTABLISHED`;
  - J is `JOINT_ADVANTAGE_NOT_ESTABLISHED`.
- The source winners refitted with the source code are bitwise identical (8 of 8). The recomputed source-definition controls are bitwise identical.
- **Deployment.** All 7 existing packages redeploy bitwise as full-role batches, through the wrapper API, the CSV path and the underlying function: FINALIST and TASKONLY on both seeds, H_LJ on both seeds, and NEURAL_RAW on seed 0. They also do so from the source's same-device backup alone. No NEURAL_RAW seed-1 package exists: that is a coverage gap, and its output is admitted by hash only.

**2. Reader-robust exploratory result: partly. The pair benefits survived this larger bank against the frozen fitted readers; the frozen claim-B conjunction did not.**

The enhanced bank was registered, reviewed and locked before fitting. It adds 8 recipes (ExtraTrees, histogram boosting, MLPs and categorical-interaction readers) to the source's 8, with access matched to the source and selection on ATTACK_VALIDATION only.

Pair benefits against the eligible references (reference AUC − finalist AUC, seed 0 / seed 1):

| Reference | ENHANCED point | Clause bound | Source (ORIGINAL) point | Source clause bound |
|---|---|---|---|---|
| NEURAL_RAW | +0.027 / +0.022 | 0.0135 / 0.0069 | +0.027 / +0.0199 | 0.0135 / 0.0059 |
| H_LJ | +0.028 / +0.026 | 0.0191 / 0.0140 | +0.028 / +0.025 | 0.0191 / 0.0129 |

All four ENHANCED pair-benefit bounds stay above 0, and none exceeds 0.020. These hold against these frozen fitted readers. With household-bootstrap-trained readers, the benefit shrank in 3 of 4 seed × reference cells; individual refits ranged from −0.001 to +0.031 (`CONFIRMATION_FEASIBILITY.md`). The benefit is reader-relative.

**The conjunction fails** on the seed-1 income recipient non-worsening clause against H_LJ:
- the clause is UNRESOLVED: point +0.0076 (finalist income-view AUC minus H_LJ income-view AUC), clause bound 0.0235, against a limit of 0.010;
- the H_LJ income-view winner changed on an ATTACK_VALIDATION near tie (HGB15 0.5523 against LOG_MLP 0.5517) and read 0.011 lower on DEV_REPORT (0.549 → 0.538; marginal 95% interval for the change −0.024 to +0.001);
- the finalist's income reader did not change.

Under the frozen interpretation rule:
- **B** is `READER_SENSITIVE`, side `COMPARATOR_READ_LOWER`. The frozen side rule was written for a failing benefit; it is applied here, unchanged, to the failing recipient clause, using the same paired reader-class changes (verifier item V-D10).
  - "Not added finalist exposure" describes only the change between banks: the finalist's income reader stayed the same.
  - Under the ENHANCED readers, the finalist's seed-1 income view reads +0.0076 above H_LJ's, and the data allow up to +0.0235.
  - Selection instability is not one-sided. The finalist's own seed-1 income winner sits on an exact ATTACK_VALIDATION tie, broken by the source-first rule. H_LJ's seed-0 income winner also changed (−0.0085).
- **M** is `READER_SENSITIVE` through step 2, which is an alias of claim B's H_LJ keys (side `COMPARATOR_READ_LOWER`); its ENHANCED label is `TASK_ONLY_ATTRIBUTION_ONLY`. Step 1, against the task-only control (+0.051 / +0.048), `SURVIVED_THIS_READER_BANK`.
- **J** and the 0.020 target are `NOT_ESTABLISHED_EITHER_BANK`.

**Descriptive support** (never used for selection). With the same recipe applied to both releases, the finalist's pair reads lower than H_LJ's and NEURAL_RAW's for 14 of the 15 recipes other than LOOKUP, on each seed and against each reference. The matched benefits range from +0.005 to +0.034 (`READER_MATCHED_BENEFITS.csv`).
- The exception is CONF_HGB, which sees only the maximum probability: −0.015 to −0.002.
- LOOKUP is degenerate on these mostly unique pairs and reads about 0.

Across reader recipes the pair advantage is fairly stable at this sample size, for these fitted readers. The registered conjunction is not.

**3. Unresolved utility or subgroup condition: yes.**
- In both scorecards, every utility (24), guard (4) and recipient (8) condition holds at the point.
- Simultaneous utility confidence is `CONFIDENCE_UNRESOLVED`: 25 of 28 unresolved.
- The binding clauses sit at the edge:

  | Clause | Value | Cap |
  |---|---|---|
  | income Brier excess vs Ucal | 0.00492 / 0.00494 | 0.005 (slack 0.04 / 0.03 SE) |
  | seed-1 income accuracy loss | 0.00959 | 0.010 (slack 0.11 SE) |

- The occupation-side guards are inherited from H_LJ: men 0.0032 / 0.0040 against 0.005.

**4. Confirmation scenario: none realistic at this release's margins** (`CONFIRMATION_FEASIBILITY.md`). V2 incorporates the statistical reviewer's corrections. No claim label or n80 value changed; some feasibility status labels did, as listed in `PROTOCOL_DEVIATIONS.md` item 9. The specification was frozen before the re-read, and the corrections were applied after the lock (`PROTOCOL_DEVIATIONS.md` item 9).
- **The required sample size is not identified by these data.** It depends almost entirely on the unknown true income utility slack, which sits 0.03–0.11 SE from its caps. For 80% joint power, with the less favourable scorecard governing:

  | True slack of the binding utility clauses (assumed) | Source scorecard (D1) | Confirmatory (D2) |
  |---|---|---|
  | 0.002, the largest on the registered grid | about 3.2×10^5 households, set by the ENHANCED recipient clause against H_LJ (governs) [ORIGINAL about 4×10^4] | about 4.0×10^5 [ORIGINAL 3.2×10^5] |
  | the plug-in | about 4.6×10^6 households | about 6×10^7, more than there are households in Florida |
  | 0 or below | never | never |

- **Floor from the recipient clause.** Whatever the utility headroom, the ENHANCED seed-1 income recipient clause against H_LJ needs about 318,000 households on its own.
- **Power at the size that exists.** The only Florida 2019 people who could test this release number about 17,500 households. At that size, joint power for D1 is about 0.01 under the governing ENHANCED scorecard (about 0.25 under ORIGINAL), and D2 is about 0.
- **Assurance**, a flat-prior heuristic averaging over development uncertainty, stays at or below about 0.10 (ENHANCED) or 0.16 (ORIGINAL) at every n.
- **Same-size replication.** A replication with 3,955 households would pass the source scorecard with probability about 0.003 (ENHANCED) or 0.115 (ORIGINAL).
- **What binds.** At the plug-in slack, income utility binds. The four pair-benefit clauses pass at DEV's size, but the ENHANCED recipient clause above does not; it sets the ENHANCED requirement whenever the true utility slack is 0.002 or more.
- **The 0.020 target** is not reliably establishable at 0.020 as n grows under the ORIGINAL plug-in or under shrinkage scenarios. That is conditional on those scenarios; the target is not shown to be false.
- **Refit spread.** A rough measured reader-refit spread (benefit SD 0.003–0.011; direction of bias unknown) is of the same order as the conditional SEs and is not covered by the intervals.

**5. Populations whose eligibility or exposure remains unknown** (`POPULATION_INVENTORY.md`; no population was chosen or opened).
- **Florida 2019, not yet used by any model or reader:** the sealed 20% (9,853 households) and the capped-out development households (7,657) are the only people that could test the frozen release itself. Both carry the replication's disclosure D2 (in-memory extraction): they are new people, not untouched people.
- **Other state-years:** every one would be a transport. Their usage status is NO_USE_FOUND_IN_SEARCHED_RECORDS or EXPOSURE_UNKNOWN, with counts unknown.
- **TX 2018** is reserved by another programme.
- **CA 2014, 2016, 2017 and 2018** are used. CA 2014 is a new finding.
- Not searched: S3, other machines and the backup drive.

## Output structure

`OUTPUT_MAPPING_CHECK.md` corrects the source's "essentially one-to-one" description:
- **Within each role:** the observed income codes are a function of the observed occupation vectors.
- **Across roles, seed 1:** two occupation vectors each carry two income codes (levels 2 and 3), which straddle the income decision.
- **Over the valid input domain:** the occupation output is many-to-one (43–48% of encoded inputs share a routing cell). The income code is not a function of the occupation vector: 689 and 1,610 invented counterexample pairs were found, confirmed through the frozen package.
- **Consequence.** The source's "no added information" reading holds only as a near-vacuous fact about the observed records within each role. There, 93–97% of occupation vectors occur once, and only 2 (seed 0) and 13 (seed 1) within-role vectors are shared by distinct inputs. It is not evidence about information content and cannot be extended to future records.

## Recommendation

**Retire `B3CTL_UNCOUPLED_MIX50_C6A` as a competitive candidate. Retain the evidence and the runnable artifact.** This is option 3 of the four in the brief.

It was conditional on the independent verification recording no unresolved discrepancy in a primary endpoint, label or feasibility number; otherwise the call would have been option 4, with the exact failing units. **That condition is met** (`INDEPENDENT_VERIFICATION.json`):
- checks 1–8 pass, with every endpoint, label, refit and power value reproduced by separate code;
- check 9 found documentation items only, all corrected (`PROTOCOL_DEVIATIONS.md` item 14).

**Why not "preserve and prepare a prospective study" (option 1).**
- The release's own income utility margins make the joint registered conditions practically unconfirmable. At the plug-in slack, millions of households would be needed, while the only people who could test the frozen release number about 17,500 households.
- The data do not show that the binding slack is positive. A larger true slack would need fewer households, but never fewer than about 318,000 under the governing ENHANCED scorecard, because of its recipient clause. A slack of zero or below would never reach 80%. The required size is therefore not identified, and no realistic design was found.
- A fresh population does not fix inadequate utility headroom, and no novelty is claimed.

**Why not "preserve as a reader-sensitive engineering result without confirmation yet" (option 2).**
- The reader sensitivity found here is real under the frozen rule, but it is narrow. It comes from one comparator income-view clause on an ATTACK_VALIDATION near tie, while the pair benefits held against the frozen fitted readers.
- "Without confirmation yet" implies a realistic path to confirmation. The feasibility analysis shows none at these margins.
- Option 3 already retains the evidence and the runnable artifact.

**Why validation is not incomplete (option 4).**
- Every registered unit ran.
- Every lock was committed and published before the stage it governed.
- The feasibility numbers were corrected after the lock, through a disclosed V2 (`PROTOCOL_DEVIATIONS.md` item 9). V2 changed no claim label and no n80 or n90 value. It did change feasibility status labels: 6 assurance cells, and the governing-scorecard assignment in 8 cells (verifier item V-D13).
- The independent verifier reproduced every endpoint, label, refit and power value with separate code, in checks 1–8, and logged its own replay (`INDEPENDENT_VERIFICATION.json`).

**What remains useful.**
- The 7 existing frozen packages deploy reproducibly, including from the source's same-device backup.
- The descriptive finding is consistent across recipes: a coarse, decision-aligned income code beside the JOINT_TREE occupation output lowered pair recovery relative to H_LJ and NEURAL_RAW on these records, against these fitted readers.
- Any successor would be a new candidate with real income utility headroom, developed under a new authorization. It is not a reselection of this one.

## Not claimed

- No arbitrary-reader protection, no privacy guarantee, and no population-optimal statement from lookup tables.
- No information-removal claim.
- No confirmation, no novelty, and no fairness claim beyond the registered guard.
- An UNRESOLVED outcome is not equivalence.
- Deterministic outputs are not assumed injective; the domain analysis shows they are not.
