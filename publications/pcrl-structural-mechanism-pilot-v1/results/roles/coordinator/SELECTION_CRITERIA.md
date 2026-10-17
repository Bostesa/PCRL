# Selection criteria (fixed before either proposal was revealed)

Written by the coordinator after launching the two designers and before reading either proposal. At most one candidate family is selected. A candidate must pass every hard criterion S1 to S5.

| # | Criterion | Kind |
|---|---|---|
| S1 | **Contract.** Each recipient receives only its own probability vector and argmax, computed deterministically from the six permitted raw fields. No identifier, cell label, extra score, seed or state is transmitted. Deployment refuses SEX. | hard |
| S2 | **Substantive structure.** The mechanism changes which input distinctions the release keeps or merges. It is not a privacy weight, wider adapter, refreshed critic, calibration sweep, frozen-output mixture, composition of frozen outputs, or a renamed earlier recipe (MMP, RCCS, leaf-score correction, codes on frozen outputs). | hard |
| S3 | **Distinct ingredient.** The closest-work reviewer finds a substantive algorithmic difference from the closest published method (FARE, FAGTB, the privacy funnel, Learning Fair Representations, discrimination-aware splitting, adversarial representations, optimized pre-processing) in objective, partition construction, optimisation, task heads or release contract, and the difference plausibly changes what is released. A proposal equivalent to an existing method can serve as a faithful baseline but not as a candidate. | hard |
| S4 | **Rationale.** It names the distinctions merged, why they matter less for the permitted tasks than for SEX inference, why the neural comparator may retain them, and a falsifier fixed in advance. Its mathematics is consistent with `PROBLEM_SPECIFICATION.md` section 9 and `CORRECTIONS.md` section 1 (no injectivity or impossibility claims). | hard |
| S5 | **Feasibility.** A task-only twin that can plausibly meet the 24 point caps on MODEL_SELECT exists, the candidate and twin fit within about 2 CPU-hours for two seeds with at most three predefined settings per seed, and a synthetic or finite-law check of the intended behaviour can be written. | hard |
| S6 | **Masking safeguard.** Apparent success cannot come only from fooling the mechanism's own critic: protection should come from merging or removing distinctions that fresh readers would otherwise use. | ranking |

**Rules.**
1. If no proposal passes S1 to S5: `NO_CANDIDATE_SURVIVED`. The campaign then finishes the service and threat model and the decision record, and runs no pilot.
2. If both pass: select the one with fewer SERIOUS objections from the adversary and closest-work reviewers; on a tie, the lower implementation risk and cost.
3. After implementation: if the mechanism fails its own synthetic check (it does not merge when it claims to, repeated inputs in a cell give different outputs, or apparent protection comes from label leakage, an omitted wire or a useless task): `NOT_PILOT_READY`, no real fit of the candidate.
4. If the task-only twin fails the 24 point caps on MODEL_SELECT: `CAPACITY_OR_LEARNER_FAILURE`, reported as such and not as a privacy failure; no privacy training.
5. Joint (two-recipient) optimisation is accepted only with a concrete computational or statistical justification; otherwise the local version is the candidate.
6. Nothing in these criteria lowers a utility cap, drops a confidence requirement or redefines a comparator.
