# Stage B mechanism decision table (roles A, B, D; registered before FEASIBILITY_LOCK, before any real-array geometry)

This is a design decision, not a parameter search. At most one mechanism may go to a prototype, and only after
Stage D's registered go rule (PROTOCOL.md §5) holds. The other two are not tried on Adult.

**Sources.**
- PRIOR_WORK_AND_NOVELTY.md (role D, primary sources).
- MATH_REVIEW.md (role B).
- SOURCE_INDEX.json (closed studies with full commit SHAs).

| | (1) Confidence-constrained score channel (SELECTED for Stage D feasibility) | (2) Retrain purpose heads jointly with an explicit confidence objective | (3) Stochastic channel with the same permitted inputs and the same pointwise utility constraint |
|---|---|---|---|
| What it changes | Which distinctions of the frozen calibrated teacher Ucal are disclosed. The release is a many-to-one token whose representative q satisfies G for every member, plus the disclosed fallback. | The teacher itself: new purpose heads and encoder trained with a privacy term and a confidence penalty. | The randomness of the release: a channel W(t \| p) whose every supported output satisfies G for its input. |
| Confidence | GUARANTEED pointwise for every input and every label relative to Ucal (G: NLL d = 0.005, Brier b = 0.0025, strict class), by certification of every member. Unseen inputs are guaranteed through the fallback. | MEASURED only: an average-loss penalty on fitting rows. No per-input or held-out guarantee, and it is not relative to a frozen reference. | GUARANTEED pointwise only if EVERY supported output satisfies G (a mean guarantee is not pointwise). It is then the same geometry as (1): every output pins p to a TV cell of about 0.005 (TOY_LAWS_CONSTRUCTION.md; MATH_REVIEW.md R5, R9). |
| Prior work already covering it | Hard-distortion privacy (Liao et al.); D∞/KL-bounded quantisation and the information bottleneck; argmax-preserving confidence release (MemGuard, MAD); greedy privacy-funnel merging. At most an application plus a small adaptation (certified bins under a capacity cap). | Fair/invariant representation learning (FARE, LAFTR, adversarial heads). Closed in this repository: joint complete-view method NEGATIVE; refreshed guarded joint, strength-matched feedback, online strength frontier and no-erasure penalty all EXPERIMENTAL_NO_ADVANTAGE (SOURCE_INDEX.json). | LP privacy mechanisms (Cai, Zhang and Khalili; Rassouli and Gündüz); collusion-aware sequential release (Taylor et al.). The repository's earlier stochastic-channel study (research/pcrl-stochastic-channel-v1, deb6992ea868f8854e2462f3302d8b7ef63d7287) tested a different channel, an ACS residence quantizer, and closed negative at its capacity gate. It does not test this contract. |
| Utility premise checkable before privacy? | YES: label-free geometry F1–F5 on Ucal (PROTOCOL.md §5). | NO: confidence is known only after fitting. | Partly: the support geometry is that of (1). The LP adds only mixing inside admissible cells. |
| Strongest matched control | Task-only minimal admissible cover at the same capacity, guard and fallback; sequential 1→2 and 2→1; the LP stochastic-local arm; the PF greedy merge restricted to G. | hcal P* (JOINT λ0.1 + class temperature) and the strong sequential partition. Both are already measured, and both fail the occupation log-loss bound (UB 0.0120 / 0.0124). | Arm (1) with the same supports (a deterministic partition is an LP vertex). |
| Materially different from a closed recipe? | Yes, in the contract: a pointwise all-label guard relative to a frozen calibrated reference, not an average-loss allowance. Whether it leaves room to operate is exactly the Stage D question. | No: it re-enters the closed joint-training line under a renamed objective. | Not as a first step: its admissible support is (1)'s, so it adds nothing until (1) shows room. |
| Decision | Investigate first (Stage D geometry + finite-law oracles). A prototype only if the go rule holds. | Not investigated in this sprint. | Carried only as an exact oracle arm on the finite laws (stochastic local LP). Not fitted on Adult. |

**Recorded expectation before any real-array run.**
- Under G every admissible output pins p_k to an interval of width about 0.005 (MATH_REVIEW.md R5).
- Covering one whole income decision class needs at least 147 bins, against the registered 8.
- If P(SEX | p) is L-Lipschitz in TV, every admissible release is within L·0.005 Bayes accuracy of releasing p (R6).
- The prompt's question, "does the required constraint make it essentially an identity release?", is therefore
  expected to be answered yes. The registered F1–F4 metrics decide it on the frozen teachers; this expectation does
  not decide it.
