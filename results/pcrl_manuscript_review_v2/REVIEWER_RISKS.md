# REVIEWER_RISKS — writing fixes separated from scientific gaps

Extends `results/pcrl_evidence_review_v1/REVIEWER_RISKS.md`, which covered Study 1 only and remains
valid for it. Three tiers, and no acceptance probability, because that would not be a measurement.

## Tier 1 — repaired in this integration (writing, scope and counting)

| Risk | What a reviewer would have said | Repair |
|---|---|---|
| "8 of 8 sensitive endpoints better under both weightings" | "Your own forest plot shows two grey intervals" | §7.2 and Table 4 report 14 of 16 better, 0 worse, 2 unresolved, and separately report the registered rule firing in 4 of 4 cells. Withdrawn as item 11. |
| "No endpoint significantly worse" for the nonlinear penalty | "Your own CSV has `significantly_worse = True` for `nlr8_L1` on A/race" | §8.3 and Table 10 give per-factor better/worse/unresolved counts; Appendix A lists every significant cell. Withdrawn as item 12. |
| "63 % of optimisation was wasted, 37 % bought the protection" | "That is not what a rotation search measures" | §8.5 reports an attained share under a bounded invariant search and drops the causal partition and the prediction. Withdrawn as item 13. |
| "Not an optimisation artifact" | "Three passing checks do not certify a nonconvex optimum" | §8.7 states exactly which three failures are excluded and that suboptimality remains live. Withdrawn as item 14. |
| "+0.0098 person-weighted" for `nlr8-C1` vs J | "Your Table 11 says +0.01210" | §8.4 now reports +0.0121; the generated table was always right. Withdrawn as item 17, and it enlarges the deficit. |
| Nuisance weakness read as a property of the interface | "A weak model is not an absence of information" | §8.6 scopes it to the fitted nuisances and contrasts 0.03–0.66 % on sex with 8.1–8.9 % on an authorised attribute. Withdrawn as item 15. |
| `rank(U) = 32` presented as settling the sign pattern | "Where is the inertia argument?" | §8.1 rests the conclusion on the measured counts and says so. |
| Stale cross-terminal status ("still running", "2016 never admitted") | "Which is it?" | Removed; Table 1 and `REVIEW_INDEX.md` carry the current status. Item 16. |
| Three evidence roles for two survey years | "Is your 2017 result confirmatory or not?" | Table 1, the abstract, every figure caption and §10 use the same four labels; Figure 7 shows both 2017 roles side by side. |
| Residence as an "unseen task" | "You have been optimising against residence all along" | §6 separates *labels excluded from fitting* (structural) from *task chosen during development* (a researcher decision). |

## Tier 2 — answerable now from existing evidence

| Risk | Answer available |
|---|---|
| "Your bootstrap treats three seed predictions as three people" | It does not. Study 1 averages seeds before resampling, one resample count per person; Study 2 draws cohort households once per replicate and applies them to all seeds, precisely because the 2018 seed pools overlap. Both deviations are recorded, not silent. |
| "The simultaneous family was chosen after seeing results" | `COMPARISONS.json` is hashed in the lock, committed and pushed before the first of ten final reads. |
| "Negative increments mean your attacker is weak" | Both directions are reported: 22 of 429 negative cells, and the routed-H control showing the selected attack losing to H's own in 8 of 312. The H attack stays executable, so a negative increment removes nothing. |
| "Adding frozen attackers looks like scope-hacking" | All four scopes and both budgets are reported; the primary scope was pre-declared; `SCOPE_COMPARISON.csv` shows the change is a baseline effect. |
| "Only three seeds" | Stated as a limitation, and per-seed values are plotted for every F1 endpoint. Study 2's attribution table shows most cells are simply unresolved at three seeds rather than claiming null effects. |
| "Race conclusions rest on unsupported classes" | The comparable-category diagnostic reproduces the F1 race differences to the fourth decimal on supported classes only. |
| "Why believe the numbers?" | Independent scorer (1.1e-15), independent selection re-derivation (0 of 924), independent bootstrap, cross-interpreter agreement, two cross-process prediction replays (0 mismatches in 59,874 and 27,540), corrected lock recheck over 17,639 inputs. |
| "Amendment 3 is missing from your verification artifact" | Correct, and `CORRECTED_INDEPENDENT_VERIFICATION.json` covers all three: 0 changed, 0 missing, 0 invalid, and 0 final reads after amendment 3. |
| "Study 2 evaluated a broken objective" | Stated in §4.2 and again wherever its numbers appear. The defect was diagnosed after fitting and **not** repaired in the evaluated maps. |

## Tier 3 — genuine gaps needing new data, new methods or authorised compute

| Gap | Why existing evidence cannot close it | What would |
|---|---|---|
| **No external method baseline.** SARL, K-TOpt, U-FaTE, OptNet-ARL, LEACE, SPLINCE discussed, never run. | Nothing here measures them, so no competitiveness claim is possible in either direction and `J` is not a benchmark. | The six adaptations in `CONTRIBUTION_ASSESSMENT.md` §6 with the two corrections in `RELATED_WORK_SCOPE.md` §3 (channel-only erasure; no residence labels), each at a declared width, under identical attackers. Substantial compute; not authorised here. |
| **The refined objective's defect is unrepaired in the evaluated maps.** | Every Study-2 number is a number for the defective objective. | Apply the `√2` off-diagonal weighting, pass the outcome-free rotation gate (< 0.10) before reading any outcome, and refit. A gate pass is not a reason to expect competitiveness. |
| **The surrogate-mismatch diagnosis is still not isolated.** | Study 2 varied penalty form and rank. The feature map, the nuisance, the normalisation and the utility surrogate remain unvaried. | The remaining cells of the 2×2×… in `METHOD_REVIEW.md` §8. |
| **The confirmation set is spent.** 2017 used, 2018 development. | No reanalysis creates a fresh evaluation. | ACS 2016 is admitted and unscored — and must not be spent on the strength of a development result. |
| **Attack strength is a floor.** | Any recovery number is what the tested families achieved. | A stronger adaptive attacker, or a certificate; this mechanism has none by construction. |
| **Retraining variability unmeasured.** | Three fixed systems; both bootstraps are over people. | Refitting the mechanism many times, never budgeted. |
| **Census design variance unmeasured.** | PWGTP replicate weights are not in the stored evidence. | Replicate-weight variance on a re-prepared extract. |
| **Cross-year person distinctness unprovable.** | Public identifiers do not support it. | Nothing public does; it stays a limitation. |
| **A/B asymmetry.** Only recipient A receives a channel. | The design never instantiates two channels. | A genuinely multi-channel study — a different paper. |

## Presentation risks that are not scientific errors

* **The paper now carries one positive confirmatory result and two negative method results.** A
  reviewer who reads §7.2 alone will take it as a success; a reviewer who reads §8 alone will take
  the whole line of work as a failure. Both readings are wrong and the abstract is ordered to
  prevent them. That ordering should not be softened in either direction.
* **Seventeen withdrawn or narrowed claims is a lot**, and invites the reading that the underlying
  work was careless. The opposite is nearer the truth: every published *table* was right, all 90
  Study-1 endpoints reproduce to 1e-15, both prediction replays are clean, and sixteen of the
  seventeen are wording, scope or counting. The seventeenth is a figure mis-typed into a prose
  summary whose own generated table carried the right value — and correcting it makes the result
  *worse* for the candidate, not better. §9's preamble says so and should keep saying so.
* **The withholding-matching analysis (§7.6) is the most novel analysis in the paper and is
  post-hoc.** It is labelled so in the caption, the section and the ledger. It must not migrate
  into the abstract as a registered finding.
* **Study 2's most transferable result is a defect in our own construction**, not a mechanism win.
  Presented as a method insight it would be an overclaim; presented as a diagnostic it is useful to
  anyone building a penalty that is not a trace form.
