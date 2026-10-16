# Selection decision

**Decision: designer 1's proposal is selected for prototyping, conditionally.** It is "SEX-mixture marginalisation" (MMP), at proposal hash `2b7273ae…ff368`.

Designer 2's proposal is rejected. It is "release-conditional counter-shift" (RCCS), at hash `83653f2d…09f21`.

The decision was made by the coordinator at 2026-10-16T06:04Z (decision-log entry; recorded in commit 998726d62 before the prototype ran) under the criteria fixed before either proposal existed (`roles/coordinator/SELECTION_CRITERIA.md`, sha256 logged at 04:56Z). These assessments of both proposals were in hand:
- the mathematical verifier, `roles/math_verifier/MATH_ASSESSMENT.md`;
- the adversary and methodology reviewer, `roles/adversary_reviewer/PROPOSAL_THREAT_ASSESSMENT.md`;
- the closest-prior-work reviewer, `roles/prior_work_reviewer/PROPOSAL_COMPARISON.md`;
- both designers' cross-critiques.

No vote was taken. Each objection below was resolved, or carried forward, on its merits.

**What "selected" means here.** MMP is the one idea from this campaign that is worth a cheap, gated real-data test. It is **not** evidence that a competitive algorithm is near. Its designer puts the chance of establishing the 0.020 margin over NEURAL_RAW at no more than 0.07. The verifier's verdict was `SOUND_WITH_FIXES`, and four fixes are still required (see "Conditions" below). If the data-free prototype contradicts the rationale, the candidate will not be called pilot-ready.

## Criterion by criterion

| Criterion | MMP | RCCS |
|---|---|---|
| S1 contract | Primary contract, deterministic, no person's SEX read. Comparators are under the same contract. **Pass** | Primary contract. Deployment computes an explicit SEX proxy π̂(x) and steers occupation with it. **Pass**, with a disclosure duty |
| S2 removable information and evidence the neural control keeps it | It removes the inferred-SEX term Δ = (π − π̄)(q¹ − q⁰). NEURAL_RAW E2 is a small move from U0, whose target contains Δ in full; it keeps 91% / 93% of U0's above-chance pair AUC. Δ's share of the learnable signal is **unknown**; gate F1 measures it. **Pass (indirect)** | The unspent occupation budget is shown. The "90% occupation-given-income" reading is non-specific: it is 0.81–0.92 for every service, including task-only ones. **Weak pass** |
| S3 distinct from closed lanes | No closed lane used SEX-aware fitting with marginalisation. The nearest lane, feature masks, also deletes within-SEX task value. **Pass** | Overlaps substantially with the small-adapter lane (income frozen, low-rank occupation shift) and with the coalition- and conditional-constraint lanes (OWN-03, RCSR, conditional budgets). Its distinct elements are a propensity driver and a closed-form ridge objective. **Marginal** |
| S4 mathematical soundness | All identities exact. 3 SERIOUS objections, 0 FATAL | Linear-information identities correct. Supporting computations do not hold as stated. 5 SERIOUS objections, 0 FATAL |
| S5 feasibility | Utility price is a closed form; feasibility is genuinely open and contradicted by no published number. The size of any recovery benefit is unknown | The prototype's feasible case is **infeasible under the binding three clauses**: the first reduction costs 3.75 pp of accuracy. With consistent accounting the Gaussian support for the size of the benefit largely disappears. **Fail as presented** |
| S6 robustness to its own critic | No critic. Its removal is defined by the data law, not by any reader class. **Pass** | The criterion is an exact linear and quadratic critic, and the map *injects* a SEX-propensity residual, cancelled only for that reader class. In its own law, Bayes pair recovery rises from 0.615 to 0.644. Designer 1's exact check: a categorical driver lets an optimal reader recover 21% to over 100% of the cancelled signal. **Fail** |
| S7 falsifiability | F1 (unbiased MODEL_SELECT proxy), F2, F3. **Pass** | Falsifiers 1 and 3 use a proxy inside its own criterion class; the nonlinear gate (0.03) is looser than the margin at stake. **Weak** |
| S8 testability | Exact finite laws plus continuous laws; pilot code paths exist (CatBoost registry, reader slate). **Pass** | Testable, but the decisive test requires the nonlinear readers it does not target |
| Prior-work class (label only) | An **established technique** (Žliobaitė & Custers 2016 / Žliobaitė 2017; Pearl's adjustment formula), applied to a new endpoint. No new algorithmic ingredient | Mostly established ingredients. One combination (propensity-driven conditional shift) was not found in the sources read; unproven, and no priority claim |

## Why RCCS is rejected

RCCS fails S6, and its S5 support does not hold.

**S6.** Its protection is defined against the linear and quadratic readers it optimises against. It works by adding SEX-predictive content to the occupation output and cancelling it for that reader class. That is the failure pattern this repository has already documented for linear erasure: LEACE passes its own check while nonlinear recovery barely moves. The verifier's synthetic test of the literal map (`V_RCCS_MAP_V1`) is recorded in its file. It does not change this conclusion, because the rationale itself depends on the reader class.

**S5.** Three supporting computations fail:
- its finite "feasible case" violates the binding clauses;
- its Gaussian model is internally inconsistent, and the replay offered as validation fails under consistent accounting;
- its reading of the published AUCs is model-dependent and non-specific.

**Coalition insight is preserved.** The coalition point RCCS raised remains valid as a description: per-recipient protection can enlarge the pair's contrast signal, so the target with income fixed is zero *partial* covariance. That description informs the pilot's single-recipient reporting.

**The proposed combination is not adopted.** Designer 2's proposed combination (MMP on income plus RCCS on occupation) is not a candidate. It was formed after the reveal, and it inherits RCCS's reader-class dependence.

## Why MMP is selected, and on what conditions

**Why.** MMP is the only proposal that:
1. names a specific, well-defined component to delete;
2. deletes it without optimising against any reader;
3. prices the deletion exactly;
4. can be killed cheaply by a gate its own construction cannot bias.

**Conditions.** These are the verifier's required fixes, and the coordinator adopts them before any prototype run:
1. **Add the laws that could contradict the rationale.**
   - A finite law **without pure proxies**: every SEX-predictive field has within-SEX task effects. There MMP should have no Bayes effect.
   - A finite **coalition required-disclosure** law, where the two tasks jointly need a SEX-revealing combination.
   - A **fitted-reader masking comparison**: a local, critic-style linear protection that hides each recipient's own SEX signal, set against MMP and the twin. This tests MMP §3.6, "deletion versus masking".
2. **Register within-SEX (subgroup) predictions.** Women are expected to bear about 1.26× the population occupation Brier price.
3. **Correct the NLL approximation.** The χ² denominator is the mixture m, not q⁰.
4. **Optional:** report the Brier-optimal constant weight w\* beside π̄. This is a utility-only change; the nominee stays at π̄ as registered.

**Carried forward unresolved:**
- Δ's share of the learnable signal (S1-MMP), which gate F1 measures;
- subgroup feasibility (S3-MMP), a separate status field;
- M10: MMP can *raise* recovery relative to its twin when the SEX-blind posterior pools cells by cancellation. This is why F1 is essential rather than a formality.

## Post-prototype status (added 2026-10-16T06:22Z; the decision above is unchanged as a record)

The selected candidate failed the part of its pre-registered prototype that a real-data benefit would need.
- **The learnability leg failed** (`prototype/PROTOTYPE_REPORT.md`).
- **The verifier's registered post-hoc probe** showed MMP's effect on coalition recovery has no definite sign without merging structure: −0.033 in one draw, +0.018 / +0.019 in two.

**Final status: `NO_DEFENSIBLE_NEW_MECHANISM_IDENTIFIED` for a real-data pilot.** MMP is reported as a design-stage selection that did not survive its prototype. It is not reported as a mechanism ready to test.
