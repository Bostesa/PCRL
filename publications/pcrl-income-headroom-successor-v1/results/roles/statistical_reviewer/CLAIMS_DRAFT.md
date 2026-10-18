# Claims draft: conjunctions for a possible finalist (written before any new fit)

**Purpose.** Every conclusion the campaign could draw about a finalist is fixed here, with generic method roles, before any
configuration is fitted. No conjunction, threshold or label may be added or changed after the winner is known. FINALIST_LOCK
only fills the roles with release hashes and copies these definitions. If no finalist is nominated, none of these claims is
evaluated and DEV_REPORT labels are not read.

**Data and scope.** Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, development roles only. The
service predicts income class (2 classes) and occupation group (5 classes) from AGEP, WKHP, COW, SCHL, MAR and POBP. Each
recipient receives only its own probability vector and argmax decision. SEX is the dataset's recorded demographic column;
readers estimate it from the released vectors: each output alone and the linked pair. All final endpoints are on DEV_REPORT
(5,943 people, 3,955 households). DEV_REPORT has been read 12 times in the lineage; this campaign would add one coordinator
evaluation and one verifier replay. **The evidence is development robustness on already-used records, not confirmation.**
Seeds 0 and 1 are two fits evaluated on the same people, not replications.

## 1. Roles

| Role | Meaning | Status in claims |
|---|---|---|
| SUCCESSOR (S) | the nominated finalist, both seed packages | subject of every claim |
| MATCHED_TASK_TWIN (TW) | S's matched task-only control: same input access, task-training exposure, capacity and optimization effort, with the recovery-targeting component removed. Declared in S's mechanism card before its first fit; fitted on both seeds | mechanism attribution (claim M) |
| ELIGIBLE_REFERENCES (REF) | NEURAL_RAW and H_LJ, the frozen historical releases. NEURAL_RAW has no seed-1 package; its seed-1 output is admitted by hash only | superiority and recipient non-worsening (claims P, N) |
| MATCHED_COMPONENT_CONTROL (CC) | any control that isolates one component of S, declared at FINALIST_LOCK (for example a fixed-occupation control if S changes the occupation output) | claim C, only if declared; otherwise `NOT_CLAIMED` |
| RETIRED_B3CTL | the retired B3CTL_UNCOUPLED_MIX50_C6A | diagnostic only, no claim |
| PUB_IBSI | the published IBSI objective as adapted in the SCM follow-up | descriptive comparator, no claim (`COMPARATOR_INELIGIBLE`: it fails the MODEL_SELECT seed-1 men's guard and the occupation recipient screen against H_LJ) |
| B2TWIN | the existing 6-level LOCAL income code; the nomination bar (c3) | descriptive comparator, no claim |

The final bank holds at most 8 effective releases per seed, with exact aliases deduplicated: S, NEURAL_RAW, H_LJ, TW, CC (if
declared), RETIRED_B3CTL, B2TWIN, PUB_IBSI.

## 2. Reader standard and inference

- **Final standard: the validation's ENHANCED bank** (`results/pcrl_frozen_finalist_validation_v1/READER_BANK_LOCK.json`):
  the source 8 recipes plus ET5, ET25, HGB15, HGB31, MLP64, MLP6432, CATI01 and CATI1, with single-view alternatives for
  every pair reader. Readers are fitted on ATTACK_FIT, then selected and oriented on ATTACK_VALIDATION by the lock's rules:
  source-first order, exact ties to the earlier candidate, and an AV near-tie flag below 0.010. Every label below is decided
  on the ENHANCED bank.
- **ORIGINAL bank** (source 8): reported beside every reader endpoint for historical comparison. If the two banks disagree on
  any key of a conjunction, the label gets the qualifier `READER_SENSITIVE`, with the key and side under the validation's
  paired reader-class change rule. It never switches to the more favourable bank.
- **Controls (CV).** Planted-SEX and shuffled-label controls stay inside the lock's bands on every view and seed: planted
  >= 0.95, |shuffled - 0.5| <= 0.05, on ATTACK_VALIDATION and DEV_REPORT.
- **Inference.** Household bootstrap, B = 2048, with Wald intervals and df = 3954. The weights are shared across releases,
  endpoints and seeds. I recommend reusing the lineage's registered weights seed 2026101705, so that the references'
  published endpoints replay exactly as a verifier check. Results are conditional on the fitted releases and readers and are
  not ACS design-based. They do not cover the adaptive development search.
- **Multiplicity** (the source's conventions):
  - One-sided alpha = 0.025, split equally across the alpha-spending claims: P and M (n_claims = 2), plus C if a CC is
    declared (n_claims = 3).
  - Clause critical value c1 = t(1 - 0.025 / n_claims): 2.2422567 (n = 2) or 2.3949991 (n = 3).
  - Family statuses use c_P = t(1 - 0.05 / (2 K_P)), with K_P = 2 x (12 utility + 2 subgroup + 2 REF benefit
    + 4 REF recipient + 3 TW + 3 if CC) = 46 (3.2694077) or 52 (3.3040310).
  - E, R, N (inside P), T (nested), O and D spend no alpha.

Shorthand: `BEN_X(s, m)`: L1 = point - c1 SE of the pair benefit AUC_pair(X) - AUC_pair(S) is > m. `NW_X(s, v)`: U1 = point +
c1 SE of AUC_v(S) - AUC_v(X) is <= 0.010, for single view v in {income, occupation}. `AV_X`: X is in the final bank on both
seeds.

## 3. Claims (each a separate conclusion)

### E: development point eligibility (gate; no alpha)

- `UG`: all 24 utility points <= cap. `SG`: all 4 within-SEX occupation guard points <= 0.005. `RG`: all 8 recipient points
  against REF <= 0.010. `E_POINT` = UG ∧ SG ∧ RG.
- Label `DEV_POINT_ELIGIBLE` iff E_POINT. Otherwise `DEV_POINT_INELIGIBLE`, with failed keys in the order UG, SG, RG.
- Always reported:
  - the MODEL_SELECT nomination labels;
  - the DEV utility confidence over the 28 utility and guard endpoints at c_P: `CONFIDENCE_FEASIBLE`,
    `ESTABLISHED_UTILITY_VIOLATION` or `CONFIDENCE_UNRESOLVED (n)`;
  - the recipient confidence counts.
- A point pass is not established population preservation.

### R: utility reserve (no alpha)

- **R1, MODEL_SELECT** (from nomination): `RESERVE_MET` on both seeds, by construction for a finalist; `min_income_ratio`
  reported.
- **R2, DEV_REPORT replication.** For each of the 12 income clauses: margin_DEV >= 1.7267108 x max(SE_DEV(S), SE_DEV(H_LJ)),
  with bootstrap SEs. The factor 2 x sqrt(3955 / 5306) is the registered MODEL_SELECT requirement re-expressed at DEV's
  household scale (`UTILITY_RESERVE_RULE.md`).
  - `RESERVE_REPLICATED`: all 12 hold.
  - `RESERVE_ERODED`: all 12 margins >= 0, some below the requirement.
  - `CAP_EXCEEDED_AT_POINT`: some margin < 0; E fails too.
- **R3, income confidence.** `INCOME_CONFIDENCE_RESOLVED` if all 12 income upper bounds at c_P are <= cap. Otherwise
  `INCOME_CONFIDENCE_UNRESOLVED (n)`, or `ESTABLISHED_INCOME_VIOLATION` if a lower bound exceeds a cap.
- **R4, planning (descriptive).** Each income margin in SE units projected to the 17,510 same-population households
  (SE_DEV x sqrt(3955 / 17510)). This is planning arithmetic, not a claim.
- Wording: "On DEV_REPORT, the successor's income margins did / did not remain at the registered reserve." Never "confirmed
  to meet the caps".

### P: pair superiority over each eligible reference (alpha / n_claims; intersection-union)

- `SUPERIOR_TO_ALL_ELIGIBLE` = CV ∧ E_POINT ∧ ⋀_{X in REF, s} BEN_X(s, 0) ∧ ⋀_{X in REF, s, v} NW_X(s, v).
- Per-reference legs (`vs NEURAL_RAW`, `vs H_LJ`) are condition values, not separate conclusions.
- Failure label: the first failing item in the order `CONTROL_FAILURE`, `FAILS_DEV_UTILITY_GATE`,
  `FAILS_DEV_SUBGROUP_POINT_GUARD`, `FAILS_DEV_RECIPIENT_POINT_GATE`, `NOT_SUPERIOR_TO_ALL_ELIGIBLE`. The full condition
  vector is always printed.
- Wording: "On DEV_REPORT, on both seeds, the declared readers recover SEX from the successor's linked pair less well than
  from NEURAL_RAW's and H_LJ's." It is evidence against the declared readers and auxiliary budget only.

### N: recipient non-worsening (inside P; also reported on its own)

- For each X in REF, seed and single view: `NW_X(s, v)`.
- `NON_WORSENING_ESTABLISHED`: all 8 U1 <= 0.010. `NON_WORSENING_UNRESOLVED`: all 8 points <= 0.010, some U1 > 0.010.
  `NON_WORSENING_VIOLATED_AT_POINT`: some point > 0.010.
- Always printed: the maximum point, the AV near-tie flags of both winners, and whether either winner changed between the
  ORIGINAL and ENHANCED banks. The retired candidate failed exactly here: seed-1 income vs H_LJ, +0.0076, bound 0.0235.
- An unresolved clause is not equivalence.

### M: mechanism attribution against the matched task-only twin (alpha / n_claims; strict fixed sequence)

- **Step 1.** `M_TW` = CV ∧ E_POINT ∧ AV_TW ∧ ⋀_s BEN_TW(s, 0) ∧ ⋀_{s, v} NW_TW(s, v).
- **Step 2** (only if step 1 holds): P's conjunction, `SUPERIOR_TO_ALL_ELIGIBLE`.
- **Labels:**
  - `MECHANISM_ATTRIBUTION_ESTABLISHED`: steps 1 and 2.
  - `TASK_ONLY_ATTRIBUTION_ONLY`: step 1 only; step 2's failure category is printed.
  - `NOT_ESTABLISHED`: step 1 fails; step 2 is `NOT_TESTED_SEQUENCE_STOPPED`.
- **Mandatory qualifiers:**
  - the S - TW income and occupation NLL and Brier differences (the mechanism's utility cost);
  - TW's own MODEL_SELECT eligibility and reserve, and TW's pair benefit against REF;
  - for a control-designated finalist, the statement that the finalist was designated a control.
  An advantage over an ineligible twin is a mechanism contrast, not a competitiveness result.
- Wording: "the declared readers recover SEX from S's linked pair less well than from its matched task-only twin's". Never
  "the mechanism removes SEX information".

### C: component attribution (alpha / n_claims; only if a CC is declared at FINALIST_LOCK, otherwise `NOT_CLAIMED`)

- `COMPONENT_ATTRIBUTION_ESTABLISHED` = CV ∧ E_POINT ∧ AV_CC ∧ ⋀_s BEN_CC(s, 0) ∧ ⋀_{s, v} NW_CC(s, v). Otherwise
  `NOT_ESTABLISHED`.
- Mandatory qualifier: the S - CC income and occupation NLL and Brier differences.

### T: the 0.020 material-improvement target (nested in P; no alpha)

- `MATERIAL_TARGET_ESTABLISHED`: P ∧ ⋀_{X in REF, s} BEN_X(s, 0.020), using the same bounds.
- `TARGET_POINT_ONLY`: all 4 points > 0.020, but some bound <= 0.020.
- `TARGET_NOT_ESTABLISHED`: otherwise.
- Always worded as "positive" versus "beyond the original 0.020 target". A target that is not established is not shown to be
  false.

### O: do the inherited occupation constraints still bind? (no alpha; always reported)

- **If S's occupation output is bitwise the inherited H_LJ / JOINT_TREE output** on DEV_REPORT on both seeds (wire hashes),
  its 12 occupation clauses and 4 guards are `REPLAY_OF_PUBLISHED` values.
  - Label `INHERITED_CONSTRAINTS_BIND` if any of the 16 has margin < 1.7267108 x SE_DEV (the R2 rule); otherwise
    `INHERITED_CONSTRAINTS_HAVE_RESERVE`.
  - From the published validation values the label is already known to be `INHERITED_CONSTRAINTS_BIND`. Six conditions sit
    below the requirement (margins in DEV SE units, `INHERITED_OCCUPATION_MARGINS.json`):
    - men's guard: 0.862 / 0.550;
    - occupation NLL vs U0: 0.871 / 1.534;
    - occupation NLL vs Ucal: 0.398 / 1.476.
  - Even projected to 17,510 households, the men's guard stays below 2 SE (1.81 / 1.16 with DEV SEs; 0.58 / 1.13 with
    MODEL_SELECT SEs), and so do occupation NLL vs U0 and vs Ucal on seed 0 (1.83 and 0.84, with DEV SEs).
  - Wording: "The successor changes only the income output. The inherited occupation constraints are unchanged and still bind;
    no income-side mechanism can relieve them."
- **If S changes the occupation output:** `OCCUPATION_CONSTRAINTS_RELIEVED` if all 16 meet the R2 rule;
  `OCCUPATION_IMPROVED_BUT_BINDING` if none is worse than the inherited margin but some are below the rule;
  `OCCUPATION_WORSENED` otherwise. The fixed-occupation control is then the CC.

### D: diagnostics and descriptive comparisons (no alpha; marginal 95% intervals only; no status)

- S - RETIRED_B3CTL: pair AUC difference and income NLL and Brier margins (diagnostic).
- S - PUB_IBSI: pair AUC difference (descriptive; `COMPARATOR_INELIGIBLE`).
- S - B2TWIN: pair AUC difference and income NLL and Brier differences. The label `GAIN_OVER_B2TWIN_DESCRIPTIVE` applies if
  the marginal lower bound is > 0 on both seeds.
- Reader-matched descriptive benefits: the same recipe on S and on each comparator, as in the validation's
  `READER_MATCHED_BENEFITS.csv`.

## 4. Composite statements

- **`COMPETITIVE_SUCCESSOR`** = E (`DEV_POINT_ELIGIBLE`) ∧ R2 (`RESERVE_REPLICATED`) ∧ P (`SUPERIOR_TO_ALL_ELIGIBLE`). It is
  always printed together with O's label, R3's label and any `READER_SENSITIVE` qualifier. Without R2 it is at most
  `SUPERIOR_WITHOUT_RESERVE`: the retired candidate's profile.
- **Not claimed:** novelty; joint-design advantage (unless declared at FINALIST_LOCK with its own conjunction in the source's
  J form); arbitrary-reader protection; any privacy guarantee; information removal; fairness beyond the registered guard;
  population optimality of LOOKUP.
- **Further limits.** Deterministic outputs are not assumed injective; observed mostly unique vectors do not show information
  equality; and counterexamples to a global output mapping do not quantify extra SEX information.
