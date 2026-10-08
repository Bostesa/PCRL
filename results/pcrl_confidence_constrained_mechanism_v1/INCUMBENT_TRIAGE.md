# Incumbent triage — precision versus method contribution (Stage A)

**Inputs.** Committed aggregates only: hcal tip 1baf5bb, evidence 159537d (PRIMARY_ENDPOINTS.csv, SELECTION.json,
RESEARCH_DECISION.md, CALIBRATION_GENERALIZATION.csv). No attacker was refit, no assessment row or label was read,
and nothing was rescored.

## 1. Which clauses remain unresolved

The incumbent is hcal P\*: the frozen JOINT λ0.1 partition with the held-out class-temperature decoder. T\* is
FINE-TASK with a global temperature, and the calibrated reference is U with a global temperature. The original
conjunction has 11 clauses (P01–P11), and the calibration-matched conjunction adds 4 more (P12–P15).

| Slot | Point | SE | Family interval (z = 3.065383) | Cap | Status |
|---|---|---|---|---|---|
| P07: occupation LL, P\* − U0 | 0.007537 | 0.001463 | [0.003051, 0.012023] | < 0.010 | NOT_ESTABLISHED_PRECISION |
| P13: occupation LL, P\* − Ucal\* | 0.009183 | 0.001054 | [0.005953, 0.012413] | < 0.010 | NOT_ESTABLISHED_PRECISION |

- Every other clause passes. This includes P01, the pair benefit: 0.031920 [0.027237, 0.036604].
- Neither unresolved clause is an established violation; both points lie below the cap.
- The conjunctions are NOT established. "10 of 11" is not partial success of a conjunction.

## 2. What is old information repeated

- **The privacy effect is not new.** P\*'s partition is the frozen JOINT λ0.1 map that lra nominated with its mean
  decoder (and qpc before it). It was assessed on the same reused rows and attacked with an enlarged but overlapping
  reader bank.
  - A decoder change on unchanged tokens discloses the same information (hcal COMPLETE_INTERFACE_EQUIVALENCE.json).
  - The 0.032 pair benefit therefore repeats the old map's advantage on the same rows. It is not an independent
    reproduction.
- **The genuinely new hcal evidence is about calibration.**
  - A shared temperature beat per-token fitting.
  - It tied the original mean decoder.
  - Held-out per-token fitting helped income only.

## 3. What more evaluation precision would and would not resolve

**Conditional arithmetic only.** n_new ≈ n_old · (z·SE_old / (cap − point))², with n_old = 13,929 exact-record groups
and z = 3.065383. This is not 3× that many observations: the three model seeds share the same people.

| Slot | Upper-bound threshold alone | 90% normal-approximation power |
|---|---|---|
| P07 | ≈ 46,180 independent comparable groups | ≈ 92,864 |
| P13 | ≈ 217,834 (rounded inputs give 217,835) | ≈ 438,050 |

**These are not powered designs.** They assume:
- unchanged effects, comparable loss distributions and fixed fitted models;
- SE scaling as 1/√n.

They do not account for data transport, retraining, the full selection history or all-clause joint power. A formal
prospective plan would register its multiplicity rule in advance; it would not revise hcal retrospectively.

**What precision could resolve.** Only whether the true occupation log-loss excesses of this frozen release lie below
the caps.
- Because P13's point sits 0.0008 below its cap, even a large sample could resolve it either way.
- A larger sample can help a true near-cap effect. It cannot manufacture a better algorithm.

**What precision would leave unresolved** (obstacle B, method contribution):

1. **Sequential control.** The INNER_SELECTION comparison (inner data only; not a locked assessment comparison; no
   confidence interval):

   | Release | Mean inner pair AUC | Mean summed NLL |
   |---|---|---|
   | JOINT λ0.1 H-CLASS-TEMP | 0.8170955 | 1.5995780 |
   | SEQ-12 λ0.1 H-GLOBAL-TEMP | 0.8197813 | 1.5971155 |

   The joint recovery advantage is 0.0026858 inner AUC, with worse summed task loss. Joint design has not shown a
   useful advantage over a strong sequential design. Sequential arms are mandatory controls.
2. **Novelty.** The incumbent is an existing partition plus established temperature scaling. Neither is a new
   algorithm.
3. **Mechanism.** No study has yet shown a mechanism that changes the disclosed information and beats strong matched
   controls under a confidence contract.

## 4. Compatible truly unexposed population (inventory only; nothing acquired)

| Population | Status |
|---|---|
| Adult (UCI) | All usable rows of both Adult files are allocated to the five roles and repeatedly exposed. hcal PROVENANCE_REPORT.md: OSF_DEFENSE_FIT comes from `adult.data` and AUDIT_FIT from `adult.test`. No unused Adult population exists. |
| ACS / folktables | Different schema and deployment contract: not the 83 permitted Adult columns. 2016, 2017 and 2018 are spent; Texas 2018 is reserved; New York is not authorised. The Adult teacher must not be assumed to transfer to ACS. |
| Other | No same-schema, same-contract cohort is available locally. |

**Conclusion.** No compatible truly unexposed population with the same schema and deployment contract is available.
A confirmation of the incumbent would need a new population and a new deployment contract, which this handoff does
not authorise.

## 5. Triage verdict

| Obstacle | Assessment |
|---|---|
| A (precision) | Only a new, larger, unexposed comparable population could resolve it. None exists locally. |
| B (method contribution) | Precision cannot resolve it at all. |

**Consequence for this sprint.** The next study must change the disclosed information under a stated confidence
contract, and must carry the sequential and task-only controls from the start. Hence the feasibility-first design of
the confidence-constrained mechanism.
