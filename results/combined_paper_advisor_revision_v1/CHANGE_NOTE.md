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

The numeric record is unchanged, apart from one correction and the reporting of existing values.

- **Corrected: "12 of 42" becomes "11 of 42"** (abstract-level claim, introduction, Section 6.1, Table 1,
  Discussion).
  - In the committed native-check table (`results/combined_matched_removal_benchmark_v1/NATIVE_AND_HELDOUT_CHECKS.csv`,
    `70f978ffc`), 11 of the 42 historical rows read "above tau (historical check fails)" and 31 read "passes". There is
    no borderline value: the largest passing value is 0.0438 and the smallest failing value is 0.0546.
  - The source decision file (`RESEARCH_DECISION.md` §1) says "12 of 42", but its own itemized list names 11
    combinations. The manuscript and the earlier claim ledger inherited the "12". The historical files are left
    unchanged; this note and `CLAIM_SOURCE_CHECK.md` record the correction.
- **Reported from Experiment 1's committed record** (`RECOVERY.csv`, `70f978ffc`): the exploratory Gaussian
  likelihood-ratio attackers on the noise treatment.
  - Adult: 0.546 and 0.563, against 0.528 for the slate.
  - HMDA: 0.526 and 0.575, against 0.513.
  - These attackers know the noise scale; one also uses clean auxiliary representations. They were omitted from the
    admitted manuscript, which said that no attacker used defense knowledge beyond refitting.
- **Reported from committed metadata:** the labelled-sample sizes per experiment (Table 3), the FARE certification
  sizes, and the selection rules for σ\* and FARE's configuration.
- **Rounding corrected against stored values** (claims reviewer, `review/CLAIM_SOURCE_INDEX.md` D2–D6):
  - untreated useful-head score: 0.774 → **0.773** (stored 0.773489; the paper already printed 0.773 for the same value
    elsewhere);
  - Table 1, HMDA LEACE features in Experiment 2: 0.864 → **0.863**;
  - zero-fairness tree, HMDA: 0.607 → **0.608**;
  - a per-seed offset value: "−0.000" → **−0.0003**;
  - compute: "about 0.2 CPU-hours" → **about 0.25** (`COST_AND_STORAGE.md`).
- **Further stored values now reported:**
  - the unadjusted 90% upper ends of the held-out ρ₁² (0.013 and 0.033), so that "near zero" no longer rests on point
    estimates alone;
  - the complete-release contrast R(F0) − R(F) = 0.096 [0.076, 0.116] (secondary endpoint of Experiment 4), so that the
    two cells in Section 6.6 are compared on the same surface;
  - the size of the owner's `defense_fit` (24,127 and 63,704 records);
  - Experiment 1's level bars (0.55 AUC, 0.05, one point).
- **Replaced and removed values.** The abstract's loose AUC ranges (0.81–0.86 / 0.82–0.87) are replaced by Table 1's
  exact values. The count of attacker units is dropped together with the word "unit".
- **Check.** A token-level comparison of the two `main.tex` files found no other new or missing scientific value (see
  `CLAIM_SOURCE_CHECK.md`).

## Disclosures added after the claims review

- **Three post-run changes,** stated in Section 5:
  - the FARE certification duplicate guard (without it, every certificate was unavailable);
  - the easier-task scoring repair (found by the replay; the lower bound moved from 0.0001 to 0.0019, and the decision
    is unchanged);
  - reporting repairs that changed no estimand, selection, role or decision.
- **The replay's scope:** it recomputed every registered endpoint, but not the post hoc diagnostics, the two separately
  recomputed descriptive values or the Figure 2 intervals.
- **Interval types:** the unadjusted 90% intervals used in Section 6.1 and the marginal intervals of Figure 2 are named
  as such.
- **Citation scope:**
  - Stadler et al. (every attribute, with the label granted to the adversary; finite discrete data);
  - "could not" is attributed to Elazar and Goldberg, not to Song and Shmatikov;
  - AttriGuard's attacker reads public data;
  - Li et al.'s guarantee is for linear least-squares reconstruction;
  - Tian et al.'s result is one attack on image benchmarks;
  - LEACE's guarantee is stated for the fitted moments;
  - FARE's certificate holds with high probability;
  - Jayaraman and Evans concern training-record inference.
- **Bibliography:** `tian2025fairnessprivacy` now cites IEEE TDSC 22(5):4654–4670 (DOI 10.1109/TDSC.2025.3551157);
  `jia2018attriguard` gains pages 513–529 and its publisher.
- **Figures:** the Figure 1 caption explains that Experiments 1–3 release the stored frozen head's output, and the
  bypass label is clarified. The Figure 3 caption maps "employment" to the occupation-group task.

## Not changed

- The data, fits, endpoints, intervals, decisions and figures.
- The evidence cutoff.
- The utility targets.
- The pre-registration disclosure.
- The post-hoc labels.

The later ACS studies and mechanism proposals were not added. Possible extensions are listed separately in
`EVIDENCE_EXTENSIONS.md`.
