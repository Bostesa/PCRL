# Manuscript and claims handoff

No manuscript branch was edited and no message was sent. This file is for whoever next edits the manuscript or the claims register. It gives wording this campaign supports, with the required qualifiers, and lists what it does not support.

## Supported, with the exact qualifiers

**1. Reusable wording.**

> This is exploratory development evidence on previously used records: public-use 2019 American Community Survey (ACS) 1-Year PUMS person records for Florida (U.S. Census Bureau), covering housing-unit residents aged 25–64 who are civilian employed.
>
> A service releases two probability vectors for each person, each with its argmax decision: an income-class vector (PINCP × ADJINC / 10^6 > $50,000) and an occupation-group vector (OCCP mapped to five civilian broad groups). Both are computed only from AGEP, WKHP, COW, SCHL, MAR and POBP.
>
> The release under test pairs the protected-tree replication's stored, pair-aware JOINT_TREE occupation output (bitwise H_LJ's) with a six-level code of the equal-weight average of the replication's stored LOCAL_TREE and JOINT_TREE income log-odds, calibrated on the CALIBRATION role.
>
> On one logged DEV_REPORT evaluation (5,943 people, 3,955 households), this recombined release met the registered clause-level competitiveness conjunction against both eligible references (NEURAL_RAW and H_LJ), on two model seeds fitted to the same people. The declared readers (an 8-reader slate fitted on an 8,001-person auxiliary sample) recover the recorded SEX column from its linked pair less well than from NEURAL_RAW's or H_LJ's.

Results (seed 0 / seed 1):

| Comparison | Pair benefit (points) | Clause bounds |
|---|---|---|
| vs NEURAL_RAW | +0.027 / +0.0199 | 0.0135 / 0.0059 |
| vs H_LJ | +0.028 / +0.025 | 0.0191 / 0.0129 |

The clause bounds are positive, below the original 0.020 target: no bound exceeds 0.020, and seed 1 against NEURAL_RAW is a point shortfall.

Qualifiers that must accompany any use of this wording:
- **Edge utility.** Every utility (24/24), within-SEX guard (4/4) and recipient (8/8) condition holds at the point.
  - Income Brier excess against Ucal (the replication's CALIBRATION-temperature neural reference) is 0.0049 / 0.0049, against a 0.005 cap.
  - Seed-1 income accuracy loss is 0.0096, against a 0.010 cap.
  - Both references meet all 12 DEV utility clauses at the point.
- **Unresolved confidence.** Simultaneous confidence is unresolved (3 PASS, 25 UNRESOLVED of 28). A point pass is not established population preservation.
- **Adaptive, edge-seeking selection** (`EDGE_SELECTED`).
  - The finalist was control-designated and was promoted under the statistical reviewer's adjudication, which superseded an unregistered interim exclusion.
  - The claim conjunctions were written after selection but before any final reader fit.
  - Intervals are conditional on the fitted releases and readers and exclude the adaptive search.
  - The two seeds are the same people.
- **Replay.** The occupation side replays a published output (REPLAY_OF_PUBLISHED; the occupation comparison against H_LJ is IDENTICAL_RELEASE). It was chosen with knowledge of its published DEV results.
- **No added information.** On every observed role the income output adds no distinct split of people beyond the occupation output. The differences are in what the declared readers extract, not in the information carried.

**2. The coarse income code.**
- **What it changes.** H_LJ has the same occupation output and a continuous LOCAL_TREE income output. Replacing that income output with the six-level calibrated code of the equal LOCAL_TREE / JOINT_TREE mix lowers the declared readers' recovery of SEX from the linked pair. The reduction is positive at the clause level on both seeds and below the original 0.020 target.
- **Two parts not separated.** The code changes two things at once: it coarsens, and it takes half of the statistic from the pair-aware JOINT_TREE income. These are not separated:
  - the LOCAL-only twin's difference from H_LJ is descriptive only;
  - claim J is not established;
  - no continuous-mixture control was run.
- **Cost.** The finalist spends +0.0083 / +0.0076 nats more income NLL than H_LJ (ABOVE_ZERO).
- **Nature.** It is a post-processing recombination of inherited components.

**3. Negative development findings** (MODEL_SELECT):
- **Utility-constrained adversarial leaf correction (family A).**
  - No configuration met the occupation recipient bar at a feasible men's guard: A2 read 0.597 / 0.598 against bars of 0.582 / 0.595.
  - An internal holdout drawn from the teacher's own fitting people does not predict held-out cost.
- **Coupling the income output to the occupation output (family B, B1–B2).**
  - A simple calibrated map of the occupation vector did not carry income utility: income NLL excess +0.053 nats against U0 (MODEL_SELECT, seed 0).
  - With a 6-level income code added (B2), the coupled release and its uncoupled twin differed by 0.002 in the MODEL_SELECT pair proxy, and the reader classes disagreed on the sign.
  - Coupling was not the active ingredient.
- **SEX-mixture marginalization (Pope & Sydnor 2011 style; family C).** The release is the population-share mixture over SEX of SEX-aware CatBoost task models.
  - Against the exact SEX-blind teacher, every pair and occupation proxy reader went down, by a small amount: occupation recovery fell 0.022 / 0.011, to about NEURAL_RAW's level (0.621 / 0.629), still above the occupation bar.
  - A matched-cost temperature control moved the readers by at most ±0.0025.
  - The outputs stay essentially one-to-one on the observed inputs. This is lower reader recovery, not information removal.
  - The cost falls on men in male-dominated occupation groups: seed-1 men's guard 0.0064 > 0.005.

## Not supported; do not write

- Material (≥ 0.020) superiority, established.
- Any novelty claim, or a claim that this campaign designed the privacy-trained components.
- A joint-design advantage (claim J is not established).
- Information-removal or "added information" language for the pair. On every observed role the income output adds no distinct split of people beyond the occupation output.
- Any protection or guarantee statement beyond the declared readers.
- Equivalence from unresolved results.
- Independence of the two seeds.
- Any statement about the sealed assessment, which stayed closed and unopened.
- Any coupling-based "log m" information bound. For m ≥ 2 it is vacuous.

## Where the evidence is

- `RESEARCH_DECISION.md` (claims in sections 1–9, with conjunctions)
- `report/DEV_REPORT_RESULTS.json`
- `PRIMARY_ENDPOINTS.csv` and `SECONDARY_ENDPOINTS.csv`
- `INDEPENDENT_VERIFICATION.json`
- `roles/claims_reviewer/CLAIMS_REVIEW.md` and its disposition

The public package is on `publication/pcrl-autonomous-method-development-v1`. The final commit is recorded in `PUBLICATION_RECORD.json`, kept on the research branch.
