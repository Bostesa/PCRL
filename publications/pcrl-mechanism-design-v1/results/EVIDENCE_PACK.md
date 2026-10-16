# Evidence pack (published aggregates only)

Every number below comes from a published aggregate table. Nothing comes from a person-level array.

**Sources:**
- `results/pcrl_acs_protected_tree_replication_v1/SERVICE_TABLE.csv` (source study, public `f4ea199`);
- `results/pcrl_acs_protected_tree_replication_v1/SELECTION_DETAIL.json`;
- `results/pcrl_acs_frozen_hybrids_v1/SERVICE_TABLE.csv` and `PRIMARY_ENDPOINTS.csv` (frozen hybrids, public `3a1f509`).

The computation summary is unit `COORD_EVIDENCE_SUMMARY_V1` (`mechanism_design/evidence/summarize_published.py`).

**Conventions.** All values are DEV_REPORT, seed 0 / seed 1. Recovery is the AUC for SEX of the reader selected on ATTACK_VALIDATION. These are fitted readers, so the AUCs are lower estimates, not population floors.

## A. Recovery and utility of every fitted ACS service

| Service | Role | Income recipient AUC | Occupation recipient AUC | Linked pair AUC | Pair minus best single | Point utility (24 clauses) |
|---|---|---|---|---|---|---|
| T_TREE | teacher | 0.569 / 0.568 | 0.650 / 0.649 | 0.664 / 0.664 | 0.013 / 0.016 | pass |
| TASK_REFIT | task-only twin | 0.572 / 0.567 | 0.650 / 0.645 | 0.664 / 0.666 | 0.014 / 0.022 | pass |
| U0 | neural reference | 0.560 / 0.567 | 0.641 / 0.638 | 0.654 / 0.649 | 0.013 / 0.011 | reference |
| LOCAL_TREE | local adversarial leaf correction | 0.558 / 0.549 | 0.582 / 0.584 | 0.657 / 0.649 | 0.074 / 0.065 | pass |
| JOINT_TREE | local + pair term (diagnostic) | 0.535 / 0.537 | 0.560 / 0.583 | 0.603 / 0.606 | 0.043 / 0.023 | **fails** 2 / 4 income clauses |
| NEURAL_RAW ρ1 E2 | eligible neural control | 0.546 / 0.551 | 0.622 / 0.623 | 0.640 / 0.638 | 0.018 / 0.015 | pass |
| H_LJ | LOCAL income + JOINT occupation | 0.558 / 0.549 | 0.560 / 0.583 | 0.641 / 0.643 | 0.081 / 0.060 | pass |
| H_JL | JOINT income + LOCAL occupation | 0.535 / 0.537 | 0.582 / 0.584 | 0.625 / 0.625 | 0.042 / 0.042 | fails income |
| FAGTB λ0.03 | published tree baseline | 0.552 / 0.558 | 0.635 / 0.642 | 0.648 / 0.656 | 0.013 / 0.014 | fails 12 / 12 |
| FARE F1 | official FARE | 0.643 / 0.639 | 0.596 / 0.599 | 0.640 / 0.633 | −0.004 / −0.006 | fails 12 / 12 |
| FARE Z1 | FARE's no-fairness twin | 0.646 / 0.646 | 0.658 / 0.660 | 0.667 / 0.666 | 0.008 / 0.007 | fails 12 / 12 |

"Pair minus best single" is a descriptive difference between fitted-reader AUCs, computed from unrounded table values. It is not an information measure and not a synergy theorem.

**Task metrics on DEV_REPORT**, seed 0. Brier is summed over classes.

| Service | Income acc / NLL / Brier | Occupation acc / NLL / Brier |
|---|---|---|
| U0 | 0.7252 / 0.5433 / 0.3641 | 0.4844 / 1.2328 / 0.6243 |
| T_TREE | 0.7251 / 0.5357 / 0.3592 | 0.4920 / 1.2213 / 0.6196 |
| TASK_REFIT | 0.7256 / 0.5361 / 0.3594 | 0.4934 / 1.2204 / 0.6193 |
| LOCAL_TREE | 0.7239 / 0.5380 / 0.3609 | 0.4846 / 1.2335 / 0.6238 |
| JOINT_TREE | 0.7166 / 0.5486 / 0.3695 | 0.4891 / 1.2397 / 0.6256 |
| NEURAL_RAW | 0.7252 / 0.5455 / 0.3661 | 0.4811 / 1.2328 / 0.6241 |

**Reading the task metrics.**
- **Teacher headroom over U0** (seed 0 / seed 1, MODEL_SELECT; source `RESEARCH_DECISION.md` §2):
  - income NLL 0.0066 / 0.0068, Brier 0.0041 / 0.0045;
  - occupation NLL 0.0121 / 0.0082, Brier 0.0051 / 0.0036.
- **Budget available from the teacher's starting point.** A tree method can spend about the headroom plus the cap: roughly 0.009 summed Brier on income and roughly 0.02 nats NLL on occupation. NEURAL_RAW starts from U0 itself, so it has only the cap.
- **Occupation is the dominant single-recipient leak for task-only models:** 0.650 against 0.57 for income.

## B. The neural control's grid (MODEL_SELECT feasibility; source `SELECTION_DETAIL.json`)

| Configuration | Feasible | Max clause / cap | Failed clauses (task \| metric) | MODEL_SELECT pair proxy AUC (OUT_LR) |
|---|---|---|---|---|
| ρ1 E2 | **yes** (the only one) | 0.570 | — | 0.625 |
| ρ1 E5 | no | 1.059 | income \| Brier (seed 1 only, 2 clauses) | 0.607 |
| ρ1 E10 | no | 2.060 | income acc/NLL/Brier, occupation NLL | 0.582 |
| ρ1 E20 | no | 2.212 | income acc/NLL/Brier, occupation NLL/Brier | 0.570 |
| ρ3 E2 | no | 1.482 | income acc/Brier | 0.617 |
| ρ3 E5 … E20 | no | 3.08–4.73 | income and occupation | 0.585 … 0.533 |
| ρ10 E2 … E20 | no | 2.67–6.23 | income and occupation | 0.613 … 0.523 |

- Every infeasible configuration fails at least one income clause.
- Failed-clause counts across the grid:
  - income summed Brier 41;
  - income accuracy 32;
  - income NLL 32;
  - occupation NLL 28;
  - occupation Brier 20;
  - occupation accuracy 6.
- **The nearest infeasible point,** ρ1 E5, lowers the recovery proxy by 0.019 relative to ρ1 E2. It exceeds the income Brier cap by 6% on one seed.

## C. How the 0.020 claims are decided (unchanged), and what they would require

**The rules** (from the source lock):
- **Recipe effect:** coalition benefit vs the task-only twin must have a one-sided α/3 lower bound > 0.020 on both seeds.
- **Competitive:** benefit vs every eligible comparator must have a lower bound > 0, with nested material superiority at 0.020.

**What they would require** (descriptive planning at DEV_REPORT size):
- **Paired coalition-difference SEs:** 0.0045 / 0.0039 vs TASK_REFIT and 0.0051 / 0.0050 vs NEURAL_RAW. The α/3 critical value is 2.395.
- **Material superiority over NEURAL_RAW, established:** point benefit ≈ 0.032, i.e. pair AUC ≈ 0.608 / 0.606.
- **Plain superiority:** point benefit ≈ 0.012, i.e. pair AUC ≈ 0.628 / 0.626.
- **Feasibility of those targets:** the only fitted releases near the first target are JOINT_TREE (0.603 / 0.606) and H_JJ, which is the same service. Both fail income utility.

## D. Frozen-hybrid decomposition (descriptive, post-hoc)

Coalition AUC drops from LOCAL_TREE when JOINT outputs are swapped in:

| Swap | Seed 0 / seed 1 |
|---|---|
| Occupation only (H_LJ) | 0.015 / 0.006 |
| Income only (H_JL) | 0.032 / 0.024 |
| Both (JOINT_TREE) | 0.053 / 0.042 |

The interaction term is unresolved. The JOINT income output, which fails income utility, carries the larger part.

## E. Status labels already decided (not reopened)

**Source study:**
- `RECIPE_EFFECT_NOT_ESTABLISHED`
- `NOT_SUPERIOR_TO_ALL_ELIGIBLE` (complete; eligible set {NEURAL_RAW})
- `SUBGROUP_UNRESOLVED`
- benchmark `SUPPORTED`

**Frozen hybrids:** `H_LJ_FIXED_COMPOSITION_NOT_SUPPORTED`.

**Disclosures:**
- **D1:** source verification 10/11 (a transport-annotation failure).
- **D2:** sealed records were parsed in memory during extraction. The sealed role stays closed and is never described as "never accessed".
