# Exact-reference cases

Campaign `pcrl_finite_solver_adequacy_v1`. The exact-reference investigator wrote and ran these cases with its own code
(`roles/exact_reference/code/fsa_exact.py`). The numbers are in `tables/EXACT_REFERENCE_TABLES.md` and the investigator's records
(`RESULTS_V1.json`, `NARROW_V1/V2.json`, `ENUM_V1/V2.json`, `STEP5_V1.json`, `route_e_input_SV1.json`). No real data is involved.

## Design (registered before any run)

- **Registration.** `roles/exact_reference/LAWS.md` and `laws.json` were committed at 02:42:16Z, before any numerical experiment on
  them. The revised solver's code was committed at 02:46:39Z, and the first solver run on a law started at 02:49:12Z.
- **The laws.** Five laws P(X, S, Y_income, Y_occupation), each with 4-8 support states and explicit probabilities. P(x) is a
  multiple of 1/N, so each law maps exactly to N replicated rows, and the code's π-weighted plug-in objective equals the population
  objective. Capacity: 2 income × 3 occupation cells. Strengths: the registered grid 0.25-8. κ = 1, shrinkage 20. Twin seeds 0 and 1
  give identical results in every cell, so the 60 registered cells are 30 distinct ones, and duplicates count once below.
- **Utility limits**, relative to the full-posterior release: accuracy loss <= 0.010, NLL excess <= 0.010, summed Brier excess <=
  0.005, and the per-SEX occupation Brier guards <= 0.005.
- **Recovery.** Exact Bayes AUC of S from the emitted values (pair, income and occupation views), with identical vectors merged and
  ties counted as 1/2.

| Law | What it tests | Feasible comparator |
|---|---|---|
| A_null | S independent of X: no release can lower recovery | every release has AUC 0.5; no benefit is possible or required |
| B_collide | four task-score collision pairs (identical task posteriors, different P(S \| x)) | witness split, pair AUC 0.5065, feasible; the best task-score-only partition is 0.5795 |
| C_distinct | all task posteriors distinct, S correlated with task scores | decision-only release 0.6182 |
| D_witness | two cheap competing merges, of which only one removes SEX information | witness 0.5828 and decision-only release 0.5803, both feasible; the twin is 0.6882 |
| E_distinct_N200 | C's population law on 200 rows (shrinkage 10 times stronger) | as C |

## Findings

1. **The code is what the card says.** Objective within 1.2e-13, routing with 0 mismatches, refresh within 1e-14. Lloyd, chain and
   protected path are reproduced bitwise, and "never worse than the twin at the same strength" holds in all 60 path states. No
   implementation defect.
2. **Narrow reference (exact for one proposal step).** For each refreshed book, every tilt breakpoint on the support was enumerated,
   and the exact objective was evaluated on every resulting deployable partition.
   - When the original accepted a step, its first-improvement choice was the best in its class (gap 0).
   - It missed improving windows that contain no power of two:
     - law D, (0.002717, 0.002866) at every strength (smallest tilt tried λ/64 >= 0.0039): a feasible partition with pair AUC 0.5803
       against 0.6882;
     - law E, (0.287, 0.466) at strengths 0.5 and 1.
3. **Broader enumeration, reported as a relaxation.** Every assignment of the support states to the 2 × 3 cells was enumerated
   (1,296-1,679,616 partitions), with the rule's codewords and tables recomputed for each.
   - Realizability was certified only for a self-consistent fixed point of the rule (checked by running the rule), or for a partition
     a solver reached.
   - The same-class reference is the best certified point. It is not a proven class optimum: in law E at strengths 0.5-4, SV1 ends
     0.0003-0.041 below it.
   - In C and E at strength 0.5 and above, the unconstrained objective optimum is infeasible (pair AUC 0.5214 / 0.5338, occupation
     accuracy loss up by 0.033 / 0.036).
4. **The revised solver (STEP 5; all 60 registered cells, 30 distinct; route E input).**

   | Law | SV1 against original (distinct cells) | Usefulness (rule 3.4) |
   |---|---|---|
   | A_null | 6 equal | none possible |
   | B_collide | 6 equal (both reach the witness split, 0.5065) | none needed: already at the feasible optimum |
   | C_distinct | equal at 0.25; objective-only gains at 0.5-1; at 2-8 it moves to an infeasible partition (0.5214) | not useful |
   | D_witness | all 6 improve: pair AUC 0.6882 to 0.5803, every task loss within +0.0005 | useful (recovery), all 6 (12 registered cells) |
   | E_distinct_N200 | equal at 0.25; at 0.5-1 pair AUC 0.6182 to 0.5979, but occupation accuracy loss up by 0.0060; infeasible at 2-8 | not useful |

   - SV1 is never worse than the original (0 of 60 cells).
   - SV1 is not optimal. It stays above the best same-class reference in 10 of 30 distinct cells:
     - law C at 0.25 (+0.0004) and 1 (+0.0101);
     - law D at all six strengths (+0.00006 to +0.00015);
     - law E at 0.25 (+0.0004) and 8 (+0.0102).
   - SV1_first, the control with the same candidates per step, equals SV1 in every cell. On these laws the gain comes from the finer
     candidate set, not the selection rule.
   - Route E label: `E_ADVANCES`, superseded by route S (`SOLVER_COMPARISONS.md`). Law D is the only law with a useful cell, and it
     was built as a witness. Route E therefore shows that SV1 can find such a gain when one exists, not that such gains are common.
5. **Surrogate against recovery.**
   - In every law the feasible partition with the lowest plug-in information and the feasible partition with the lowest exact AUC
     are the same partition (`RESULTS_V1.json`, enum `mi_feasible` / `auc_feasible`).
   - In all 343 recorded states the information on cell ids equals the information on released values, so merging identical
     vectors never changed it.
   - These laws therefore do not show an information/AUC conflict. The objective/utility conflict (point 3) is the one they show.
6. **Propensity against HQ.**
   - A genuine expressivity gain exists only in B. There the split separates equal-task-score states: 0.5065, against 0.5795 for the
     best feasible task-score-only partition (`RESULTS_V1.json`, B enum `auc_feasible_hq`).
   - In C, D and E the task scores identify the states, so no expressivity claim is made.
   - In E the original solver's HQ path reaches a partition with pair AUC 0.5979 at strength 0.5, with occupation accuracy loss
     +0.006, while its PRC path stays at 0.6182 (`RESULTS_V1.json`, law E `hq_path`). SV1's PRC path reaches the same pair AUC
     (STEP 5), so the HQ lead is a search artifact of the original solver.

## Scope

- Exact for the proposal step (narrow reference) and for the enumerated relaxations, on these five small laws only.
- It is not a global optimality statement for the alternating optimizer, nor a statement about other capacities or real data.
- The laws were written by the team. Law D was built to contain a feasible gain, so route S, measured on the source, governs the
  decision.
