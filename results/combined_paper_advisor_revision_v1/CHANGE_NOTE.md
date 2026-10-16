# Change note: advisor-ready revision of the combined empirical paper

- **Revised source:** `papers/combined_empirical_v2/`.
- **Admitted source:** `papers/combined_empirical_v1/` at `3b5065805`, unchanged in this branch.
- **Focused diff:** `MANUSCRIPT_DIFF.diff`.
- **Source choice:** `SOURCE_RESOLUTION.md` explains why this version was revised and records which newer version was
  not available.

## What changed in the story

- **The abstract is selective.** It states the problem (checks cover a particular object and attacker class), the
  question (what a recipient recovers from everything it receives, alone or linked, and how useful the release stays),
  the approach and four results. It ends with the evidence status. The loose ranges "0.81–0.86 / 0.82–0.87" are
  replaced by the table's exact values, and "often releases" (unsourced) is gone.
- **The introduction explains before it reports.** The order is:
  1. protection promises and their scope;
  2. what a recipient may receive, as an illustrative scenario that is labelled illustrative;
  3. the research question and the threat model;
  4. the approach;
  5. the findings, organised as three outcomes kept apart (own test fails; test holds but recovery occurs outside its
     scope; recovery through an uncovered surface) and the role of utility;
  6. what the findings ask of future methods, the contribution and the evidence status.
- **Internal framing removed.** Gone are "four bodies of work in our repositories", the internal checkpoint label, the
  study code names, "units", "nominee" and "arm". The four sources of evidence are now Experiments 1–4, defined by
  content in Section 5. The unmerged-branch remark is reduced to the disclosure that the encoders' public code still
  exposes the retired bound, which the paper does not use.
- **Each result says why it matters.** Section 6 opens each subsection with the question it answers and ends with what
  the result does and does not support. Examples:
  - LEACE's result is described as the scope of a linear guarantee, not a violation;
  - low decision-level recovery on the frozen heads is tied to their low usefulness;
  - the offset result is described as a property of particular heads.
- **The "strongest favourable and adverse" labels are gone.** In the Discussion they are replaced by an ordinary account
  of the comparison that favours a fairness-term defense (FARE beyond a capacity-matched tree) and the three results
  that limit it:
  - non-inferiority not established;
  - cell dependence;
  - the output bypass.
- **New closing paragraph:** "What the findings ask of future methods". It gives evaluation requirements, not claims
  that methods fail.

## What changed in the adversary definition

The verified adversary patch v2 was applied first (commit `133d2c740`). Its content is kept. A new Section 3, "Threat
model and access", replaces the single paragraph and adds:

- **Why this access.**
  - Authorized recipients necessarily possess their delivered information.
  - Linkage is granted, with perfect linkage as the attacker-favourable limit. Pooling is assumed, not claimed to be
    unavoidable.
  - Labelled auxiliary data model a recipient that knows the attribute for some people. Availability, size and
    population match are assumptions.
  - A stronger attacker does not by itself justify more access.
- **The attackers' data budget,** with per-experiment labelled-sample sizes. The labelled sample is a requirement of
  these supervised attackers, not of every attack. "Defense-aware" means refitting only. Knowledge of the algorithm and
  objective is distinguished from access to fitted parameters.
- **What recovery measures.** It covers removal versus attacker limits, why deterministic does not mean reversible,
  information removal within a probability-output contract, and the fact that posterior preservation is not
  established.
- **A per-experiment access table:** what is received, the labelled sample, fit/select/score, defense knowledge,
  linkage, queries and what is untested.
- **The out-of-scope threat models, listed once.** Several of them involve no breach.
- **Related work:** the Stadler et al. sentence is corrected to the theorem's scope (every attribute, constructed
  violating attribute), and the label-only reference is described as descriptive, not a floor.

## Numbers

No number was added to or removed from the scientific record. A token-level comparison of the two `main.tex` files finds
no new scientific value. The only values absent from the revision are the abstract's loose AUC ranges and the
attacker-unit count, which was dropped together with the word "unit". The role sizes in Table 3 come from committed
metadata (see `CLAIM_SOURCE_CHECK.md`).

## Not changed

- The data, fits, endpoints, intervals, decisions and figures.
- The evidence cutoff.
- The utility targets.
- The pre-registration disclosure.
- The post-hoc labels.

The later ACS studies and mechanism proposals were not added. Possible extensions are listed separately in
`EVIDENCE_EXTENSIONS.md`.
