# Claims draft of the overnight occupation-headroom campaign (generic roles; written before any outcome)

**Purpose.** Every conclusion the campaign could draw is fixed here with generic method roles before any configuration of this
campaign is fitted or scored. No conjunction, threshold, label, endpoint, sign, reference set or multiplicity constant may be
added or changed after an outcome is seen; FINALIST_LOCK only fills the roles with release hashes and copies these definitions.
Machine-readable constants: `RULES.json` (`final_stage`, `comparators`, `separate_labels`). Development rules:
`SELECTION_RULE.md`.

**Data and scope.** Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, development roles only. The
service predicts an income class (2 classes) and an occupation group (5 classes) from AGEP, WKHP, COW, SCHL, MAR and POBP; each
recipient receives only its own probability vector and argmax decision. SEX is the dataset's recorded demographic column;
readers estimate it from the released vectors (each output alone and the linked pair). Final endpoints are on DEV_REPORT (5,943
people, 3,955 households), which has been read 14 times in the lineage and is **not fresh**. **The evidence is development
robustness on already-used records, never confirmation.** Seeds 0 and 1 are two fits on the same people, not replications. Every
recovery statement is evidence against the declared readers and auxiliary budget only.

## 1. Roles

| Role | Meaning | Used in |
|---|---|---|
| S | the nominated finalist (both seed packages) | every claim |
| REQ | the required comparators, frozen: NEURAL_RAW, H_LJ, B2TWIN, RETIRED_B3CTL, SUCC (H3 income + JOINT_TREE occupation), TWJ (HC_STRONG income + JOINT_TREE occupation) | P, N, T (subset) |
| TREF | the original target references NEURAL_RAW and H_LJ | T |
| CC | SUCC as the matched-income component: same income as S when S keeps H3, inherited JOINT_TREE occupation | C |
| MTC | S's matched task-only component: S's own task-only occupation output (its stage-1 witness) with S's income, fitted and evaluated on both seeds; declared in FINALIST_LOCK from S's record. `NOT_APPLICABLE` when S is itself task-only; for an income-branch S, the same occupation with H3 income is reported as a second, descriptive component | M |
| TTC | T_TREE and TASK_REFIT, the equally exposed task-only occupation controls | W (utility), D |
| PUB_IBSI | point-ineligible published comparator | D only |

Final bank: S, the 6 REQ, MTC (if applicable), deduplicated by exact wire hash. REQ's DEV values are published for the ENHANCED
and ORIGINAL banks and replay exactly with the same readers and weights (a verifier check).

## 2. Reader standard and inference (frozen before any final reader fit)

- **ENHANCED bank decides**: the validated slate (source 8 + ET5, ET25, HGB15, HGB31, MLP64, MLP6432, CATI01, CATI1) with
  ignore-one alternatives for the pair; fitted on ATTACK_FIT (8,001), selected and oriented on ATTACK_VALIDATION (3,000; exact ties
  to the earlier candidate; near tie < 0.010 flagged), scored on DEV_REPORT. **ORIGINAL bank** (source 8) beside it; any key of a
  conjunction that differs between the banks gives the qualifier `READER_SENSITIVE` (with key and side); the label never switches
  to the more favourable bank.
- **Controls (CV):** planted-SEX AUC >= 0.95 and |shuffled-label AUC - 0.5| <= 0.05 on ATTACK_VALIDATION and DEV_REPORT, every
  view and seed. CV failing makes every reader-based claim `CONTROL_FAILURE`.
- **Inference:** household bootstrap on DEV_REPORT, B = 2048, weights seed 2026101705 shared across releases, endpoints and seeds;
  Wald intervals with df 3954. Conditional on the fitted releases and readers; not ACS design-based; does not cover the adaptive
  search.
- **Reader-refit sensitivity beside every interval:** 4 matched household-bootstrap refits of ATTACK_FIT per winning recipe and
  release (resampling shared across releases). For each BEN and NW key: the 4 refit values. Qualifier **`REFIT_SENSITIVE`** on a
  claim when any refit benefit of a BEN key in its conjunction is <= 0 or any refit recipient difference of an NW key exceeds
  0.010; otherwise `REFIT_STABLE`. A `REFIT_SENSITIVE` claim is never described as robust.
- **Multiplicity:** one-sided alpha 0.025 split equally over the three alpha-spending claims **P, C, M** (n_claims = 3, fixed
  even if M is `NOT_APPLICABLE`): clause critical value **c1 = t_3954(1 - 0.025/3) = 2.3949991**; intersection-union inside each
  conjunction (no further adjustment). Family statuses at **c_P = t_3954(1 - 0.05/(2 x 70)) = 3.3867024**, K_P = 2 seeds x (12
  utility + 2 subgroup + 6 REQ benefit + 12 REQ recipient + 1 MTC benefit + 2 MTC recipient) = 70. E, R, N (inside P), T (nested
  in P), O and D spend no alpha.

**Endpoints (per seed s; signs fixed):**
- `UTILITY|S|<task>|<ref>|<metric>|s` (12): no_excess against the cap.
- `SUBGROUP|S|sex<g>|s` (2): no_excess against 0.005.
- `BEN|<X>|s` (X in REQ and MTC): benefit = AUC_pair(X) - AUC_pair(S); positive favours S.
- `NW|<X>|<view>|s` (X in REQ and MTC; view income, occupation): AUC_view(S) - AUC_view(X), no_excess against 0.010.
- `UTILITY|H_LJ|...|s` and `SUBGROUP|H_LJ|...|s`: H_LJ's DEV SEs (R2 floor; descriptive).

Shorthand: `BEN_X(s, m)`: L1 = point - c1 x SE > m. `NW_X(s, v)`: U1 = point + c1 x SE <= 0.010. `AV_X`: X is in the final bank
on both seeds.

## 3. Pre-registered consequences already fixed by the published source record

These follow from the published DEV_REPORT values, because the income utility clauses depend only on the income output and the
income-view readers only on the income output; they are stated now so that nobody reads them later as findings:
1. **If S keeps the H3 income**, its 12 DEV income clauses equal the published H3 values: R2's income part is
   `RESERVE_ERODED` (seed-1 accuracy loss 0.0070671, 0.83 x se_used against 1.7267), so **`COMPETITIVE_SUCCESSOR` is unattainable**
   for an H3-anchored finalist.
2. **If S keeps the H3 income**, its income-view recipient clause against H_LJ on seed 1 under ENHANCED equals the published
   +0.0061 with clause bound 0.0217 at c1 = 2.395 (the same c1 as here) > 0.010, provided the final reader protocol replicates the
   source's (same slate, seeds and weights). **`NW_H_LJ(1, income)` then fails and P cannot be established under ENHANCED**; the
   ORIGINAL bank's value (-0.0070, bound 0.0013) passes, so P's failure would carry `READER_SENSITIVE`.
3. Against SUCC, an H3-anchored S has an identical income view: `NW_SUCC(s, income)` has point 0 and SE 0 and passes.
4. The REQ comparators' DEV pair AUCs are published (ENHANCED, seeds 0 / 1): NEURAL_RAW 0.6400 / 0.6386, H_LJ 0.6414 / 0.6428,
   B2TWIN 0.6314 / 0.6278, **RETIRED_B3CTL 0.6130 / 0.6168**, SUCC 0.6272 / 0.6263, TWJ 0.6557 / 0.6578. P's binding comparator is
   therefore expected to be RETIRED_B3CTL.
For an H3-anchored finalist the open questions on DEV_REPORT are claims C, M, O, the benefit legs of P and T, and N for the
occupation view.

## 4. Claims (each a separate conclusion)

### W: useful task-only component (development only; no alpha; label 1)

- `USEFUL_TASK_ONLY_COMPONENT` iff S's matched task-only occupation output (or S itself, if task-only) was a `SELECTED_WITNESS`
  in stage 1 (complete-release point feasibility on both seeds and R1 on all 16 occupation-side conditions on MODEL_SELECT).
- Always printed: its occupation clauses and guards on DEV_REPORT with SEs, its differences against TTC, and whether it beats
  TTC's occupation-side minimum ratio. Wording: "the stronger occupation learner keeps the original occupation limits with
  development reserve". It says nothing about SEX recovery.

### E: development point eligibility (gate; no alpha)

- `UG`: all 24 utility points <= cap. `SG`: all 4 guard points <= 0.005. `RG`: all 24 recipient points against REQ <= 0.010.
  `E_POINT` = UG ∧ SG ∧ RG. Labels `DEV_POINT_ELIGIBLE` / `DEV_POINT_INELIGIBLE` (failed keys in the order UG, SG, RG).
- Always reported: the MODEL_SELECT nomination labels; utility confidence over the 28 utility and guard endpoints at c_P
  (`CONFIDENCE_FEASIBLE`, `ESTABLISHED_UTILITY_VIOLATION`, `CONFIDENCE_UNRESOLVED (n)`); recipient confidence counts. A point
  pass is not established population preservation.

### R: whole-release reserve (no alpha)

- **R1, MODEL_SELECT:** `RESERVE_MET` on all 28 conditions, both seeds (by construction for a finalist), with the leximin vector.
- **R2, DEV_REPORT replication, all 28 conditions:** margin_DEV >= 1.7267108 x max(SE_DEV(S), SE_DEV(H_LJ)) (2 x sqrt(3955 /
  5306): the MODEL_SELECT requirement at DEV's household scale). `RESERVE_REPLICATED` (all 28), `RESERVE_ERODED` (all margins >= 0,
  some below), `CAP_EXCEEDED_AT_POINT`. Its income and occupation parts are also printed separately.
- **R3, confidence:** `UTILITY_CONFIDENCE_RESOLVED` if all 28 upper bounds at c_P <= cap; else `UNRESOLVED (n of 28)` or
  `ESTABLISHED_VIOLATION`.
- **R4, planning (descriptive):** every margin in SE units projected to the 17,510 same-population households.
- Wording: "On DEV_REPORT the release's margins did / did not remain at the registered reserve." Never "confirmed".

### P: positive superiority over all required comparators (alpha / 3; intersection-union; label 3)

- `SUPERIOR_TO_ALL_REQUIRED` = CV ∧ E_POINT ∧ ⋀_{X in REQ, s} BEN_X(s, 0) ∧ ⋀_{X in REQ, s, v} NW_X(s, v).
- Per-comparator legs are condition values, not separate conclusions. Failure label: first of `CONTROL_FAILURE`,
  `FAILS_DEV_UTILITY_GATE`, `FAILS_DEV_SUBGROUP_POINT_GUARD`, `FAILS_DEV_RECIPIENT_POINT_GATE`, `NOT_SUPERIOR_TO_ALL_REQUIRED`;
  the full condition vector is always printed, with `READER_SENSITIVE` and `REFIT_SENSITIVE` qualifiers.
- Wording: "On DEV_REPORT, on both seeds, the declared readers recover SEX from S's linked pair less well than from every required
  comparator's, without worsening either single output by more than 0.010."

### N: recipient non-worsening (inside P; reported alone too)

- `NON_WORSENING_ESTABLISHED`: all 24 REQ clauses U1 <= 0.010. `NON_WORSENING_UNRESOLVED`: all points <= 0.010, some U1 > 0.010.
  `NON_WORSENING_VIOLATED_AT_POINT` otherwise. The income-view and occupation-view parts are printed separately (section 3, item
  2). An unresolved clause is not equivalence.

### C: coalition-aware occupation correction against the matched-income component (alpha / 3; label 2)

- `OCCUPATION_COMPONENT_EFFECT_ESTABLISHED` = CV ∧ E_POINT ∧ ⋀_s BEN_SUCC(s, 0) ∧ ⋀_{s, v} NW_SUCC(s, v). Otherwise
  `NOT_ESTABLISHED`.
- Mandatory qualifiers: S - SUCC occupation accuracy, NLL and Brier differences and the men's and women's guard differences (the
  utility change bought with the new occupation output); for an income-branch S, the statement that the income also differs, so
  C is a whole-release contrast, not an occupation-only one.
- Wording: "holding the income output fixed, the declared readers recover SEX from the linked pair less well with S's occupation
  output than with the inherited JOINT_TREE occupation output". It is incremental, reader-relative protection, not superiority.

### M: incremental protection against the matched task-only component (alpha / 3; strict fixed sequence; label 2)

- Step 1: `M_MTC` = CV ∧ E_POINT ∧ AV_MTC ∧ ⋀_s BEN_MTC(s, 0) ∧ ⋀_{s, v} NW_MTC(s, v). Step 2 (only if step 1 holds): P.
- Labels: `MECHANISM_ATTRIBUTION_ESTABLISHED` (both), `TASK_ONLY_ATTRIBUTION_ONLY` (step 1 only; step 2's failure category
  printed), `NOT_ESTABLISHED` (step 1 fails; step 2 `NOT_TESTED_SEQUENCE_STOPPED`), `NOT_APPLICABLE` (S task-only).
- Mandatory qualifiers: S - MTC occupation NLL, Brier and guard differences (the correction's utility cost); MTC's own
  MODEL_SELECT labels and pair benefits against REQ; `CONTROL_DESIGNATED_FINALIST` if applicable. An advantage over a component
  that is not itself competitive is a mechanism contrast, not a competitiveness result. Wording: "the declared readers recover SEX
  from S's linked pair less well than from its matched task-only component's"; never "the correction removes SEX information".

### T: the original 0.020 material-improvement criterion (nested in P; no alpha; label 4)

- `MATERIAL_TARGET_ESTABLISHED`: P ∧ ⋀_{X in TREF, s} BEN_X(s, 0.020). `TARGET_POINT_ONLY`: all 4 TREF points > 0.020 but some
  L1 <= 0.020. `TARGET_NOT_ESTABLISHED` otherwise. The 0.020 differences against the other REQ members are printed descriptively.
- Kept separate from P: "positive" is never written as "beyond the 0.020 target", and a target not established is not shown false.

### O: occupation constraints (no alpha; always reported)

- On DEV_REPORT, the 12 occupation clauses and 4 guards of S against the R2 rule: `OCCUPATION_CONSTRAINTS_RELIEVED` (all 16
  meet it), `OCCUPATION_IMPROVED_BUT_BINDING` (none has a smaller margin than the inherited JOINT_TREE output's published margin,
  some below the rule), `OCCUPATION_WORSENED` (otherwise). The inherited output's published binding conditions (men's guard 0.86 /
  0.55 DEV SE; occupation NLL vs Ucal 0.40 / 1.48, vs U0 0.87 / 1.53) are printed beside S's.

### J and V: joint design and novelty (labels 5 and 6)

- **J:** `NOT_CLAIMABLE`. Only the occupation output is optimised in this campaign; any pair term is a coalition-aware occupation
  correction, not a two-output joint design. No outcome can change this label.
- **V:** `NOT_CLAIMED`. No algorithmic novelty is claimed; known ingredients (stronger task learners, calibration, critic-aware
  corrections) are named as such.

### D: diagnostics (no alpha; marginal 95% intervals; no status)

- S - PUB_IBSI and S - TTC pair differences; reader-matched benefits (same recipe on S and each comparator); the refit roster;
  learning-curve diagnostics from the development grid; the MODEL_SELECT-to-DEV shift of every utility condition.

## 5. Composite statements

- **`COMPETITIVE_SUCCESSOR`** = E (`DEV_POINT_ELIGIBLE`) ∧ R2 (`RESERVE_REPLICATED`, all 28) ∧ P (`SUPERIOR_TO_ALL_REQUIRED`),
  printed with O, R3, `READER_SENSITIVE` and `REFIT_SENSITIVE`. With P but not R2: `SUPERIOR_WITHOUT_RESERVE`. Otherwise
  `NOT_COMPETITIVE`. For an H3-anchored finalist the composite is `NOT_COMPETITIVE` by section 3.
- **Ladder of labels, never merged:** W (useful task-only component), C and M (incremental reader-relative protection), P
  (positive superiority over all required), T (the original 0.020 criterion), J (joint-design advantage), V (novelty). T, J and V
  never follow from W, C, M or P, and a partial pass never becomes a competitive claim.
- **Not claimed:** arbitrary-reader protection; any privacy guarantee; information removal; fairness beyond the registered
  guards; population optimality of any reader. Deterministic outputs are not assumed injective, mostly unique vectors do not show
  information equality, and an unresolved clause is not equivalence.
