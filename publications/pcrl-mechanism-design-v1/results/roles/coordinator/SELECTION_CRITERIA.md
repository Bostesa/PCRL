# Selection criteria (written before any proposal was read)

The coordinator selects **at most one** proposal. The selection happens only after three written assessments of both proposals exist: the adversary reviewer's, the prior-work reviewer's and the mathematical verifier's. A vote does not resolve a mathematical objection.

## Eligibility: every item must hold

| # | Criterion | What fails it |
|---|---|---|
| S1 | **Contract.** The proposal uses the primary contract, or declares a change exactly and evaluates every comparator under the same changed contract | An unstated contract change; or comparing a richer contract against comparators held to the poorer one |
| S2 | **Specific removable information.** It names what information is removed or changed, and gives a quantitative, evidence-backed reason why the eligible neural control (NEURAL_RAW ρ1 E2) still carries that information at its selected configuration | A generic "better optimiser" or "stronger critic" argument with no evidence about the neural control |
| S3 | **Distinct from the closed lanes.** It is not a closed lane under a new name. If it revisits a closed family, it names the missing mathematical or statistical ingredient and gives a test that isolates that ingredient | A new coefficient, a checkpoint, a larger adapter or a renamed penalty |
| S4 | **Mathematical soundness.** The verifier's verdict is not `NOT_SOUND`, and no FATAL objection from any reviewer is left unresolved | A false inequality, an ill-posed objective, or a joint-versus-sequential expressiveness claim |
| S5 | **Feasibility.** The target improvement is mathematically possible under the binding caps and is not contradicted by published evidence. The proposal states how large an improvement is plausible and why | A target that every published fact makes impossible, such as beating a decision-only comparator that is already at zero leakage, or relying on an output that fails utility |
| S6 | **Robustness to its own critic.** The protection claim is judged by fresh defense-aware readers, and the rationale does not depend on fooling the training critic | A rationale that only holds against the training critic |
| S7 | **Falsifiability.** A stated observation, measurable in the data-free prototype and in the prepared pilot, would falsify the explanation | No falsifier, or one that cannot be measured |
| S8 | **Testability under this campaign's limits.** Exact or certified finite-law checks are possible, and a real-data pilot can be prepared from admitted manifests and existing code paths, with SEALED_ASSESSMENT closed and no DEV_REPORT label reads | Needs sealed data, new populations or unavailable private artifacts for its core test |

## What informs the label but does not decide eligibility

- **Prior-work classification:** established technique, engineering combination, or potentially new algorithmic ingredient. A useful engineering combination can be worth testing, but it must be labelled as such. No priority claim is made without evidence.
- **Whether a joint component is claimed.** If it is, the proposal needs a complementary-clue case with local controls and controls for both sequential orders.

## Decision rule

1. If exactly one proposal is eligible, select it.
2. If both are eligible, select the one whose S2 and S5 evidence is stronger and whose unresolved SERIOUS objections are fewer. Record why the other was rejected.
3. If neither is eligible, the decision is `NO_DEFENSIBLE_NEW_MECHANISM_IDENTIFIED`. The adversary revision, the closed-lane ledger, the prior-work review and the advisor brief are still completed. Prototype restoration and pilot preparation are then marked not applicable.
4. **After selection.** A selected proposal still has to pass its data-free prototype. If the prototype contradicts the mechanism's rationale, that result is reported and the candidate is not described as ready for a real-data pilot.
