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

## Changes from the adversary review and the independent reader

**Adversary review** (`review/ADVERSARY_REVIEW_DRAFT.md`: 7 must-fix, 19 should-fix).
- **The exploratory noise-aware attackers are described exactly.**
  - Which one exceeds the access: the second, fitted on clean auxiliary representations.
  - Neither is selected on validation.
  - They ran on the features-only surface only.
  - Neither is an insider, repeated-query or white-box attack.
- **Noise was evaluated only on the two primary settings,** not on all 14 pairs.
- **Utility claims are stated per experiment.**
- **The introduction no longer says that "where promises could be checked, they held"** (it contradicted the 11 of 42).
- **The held-out LEACE statement is a measurement on one split,** not a guarantee.
- **The access is described as a choice,** not a consequence.
- **Stronger attackers within the same access make each figure a lower estimate,** while more access is a separate
  threat model.
- **Table 3 coverage is completed.**
- **The brief and the scope proposal are corrected to match.**

**Independent reader** (`review/INDEPENDENT_READER.md`: 9 major, 38 minor, 7 layout).
- **New description of the data and models:**
  - UCI Adult and HMDA 2023 California;
  - the six purposes and their 14 pairs;
  - the training and test splits;
  - supported classes;
  - the collapsed encoder.
- **One name per head and output:** frozen head, clean output, own head, probe. Release and output contracts are
  defined.
- **HMDA's FARE release is disclosed as constant on seed 1.** HMDA's task gain is only 0.6–2.3 points, so the one-point
  admissibility rule admits a single-cell release.
- **What the intervals cover is stated:** the mean over the three stored encoders, without retraining variance. The
  family sizes and α of Experiments 1–2 are given. "Registered" and "not established" are defined.
- **The bypass and the features-plus-own-output results are framed as expected by construction.** The contribution now
  names what the paper adds.
- **The breakdown of the 4 correct and 6 wrong predictions is given.**
- **The offset subsection now comes before the decisions subsection,** so the argument runs from decisions to useful
  heads.
- **The Discussion is trimmed of restated numbers.** It now explains the score–decision gap as what scores reveal beyond
  their thresholded decisions.
- **Slogans and repeated disclaimers are replaced,** and hard sentences are rewritten.
- **Layout:**
  - all three tables had floated past the references; float placement is fixed;
  - the Figure 1 bypass label no longer collides with the recipient box;
  - the Figure 3 caption states that no intervals are shown and that the axes are truncated.

## Changes from the claims reviewer's second pass

`review/CLAIMS_REVIEW_DRAFT.md` (4 must-fix, 12 should-fix, 13 notes; 32 of the 35 phase-1 discrepancies resolved).
- **Post-run changes.** All are now listed:
  - two lock amendments before the affected fits;
  - two changes after results;
  - a report-script fix;
  - reporting repairs that belonged to Experiment 2.
- **The measured gap is a within-head comparison.** The zero-fairness tree's score recovers 0.648 and its own decisions
  0.546, so the gap is no longer explained through a different head's decisions.
- **`defense_fit` is the training split minus 18 / 43 records that also occur in the test split** (24,145 → 24,127;
  63,747 → 63,704).
- **"Weak frozen heads" is limited to the primary settings.** Three other heads gain 0.37–0.69 over the constant.
- **Terminology:**
  - "development evidence" (all results) is kept apart from "exploratory" (outside registered families);
  - the scopes of the own head and the probe are stated per experiment;
  - the HMDA gain is the probe's.
- **Coverage:** noise ran on four secondary pairs, not reported; the cell-conditional match claim is limited to
  Experiment 2; the unsupported claims that the pair was "conservative" are removed.
- **Citations and sources:**
  - LAFTR's guarantee is conditional on an optimal adversary;
  - output-surface numbers are labelled by experiment;
  - the easier task's non-inferiority interval is added;
  - the 0.096 interval type is stated.
- **Records:**
  - the brief now gives both noise-aware attackers' values;
  - the scope proposal cites the access source, the worsening upper bound and the per-experiment targets;
  - the claim-to-source table has corrected rows, renumbered rows 1–51 and the reviewer's verdict.

## Not changed

- The data, fits, endpoints, intervals, decisions and figures.
- The evidence cutoff.
- The utility targets.
- The pre-registration disclosure.
- The post-hoc labels.

The later ACS studies and mechanism proposals were not added. Possible extensions are listed separately in
`EVIDENCE_EXTENSIONS.md`.
