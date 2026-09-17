# Reviewer risk assessment

Three tiers: what this revision already repaired, what existing evidence can still answer if a
reviewer asks, and what would need new data or new methods. No acceptance probability or readiness
percentage is given, because neither would be a measurement.

## Tier 1 — already repaired in this revision

| Risk | What a reviewer would have said | Repair |
|---|---|---|
| The `±0.001` band reads as an equivalence claim | "Your interval reaches 0.0019; you cannot call that a band" | §7.3 and Table 2 state the rule is a one-sided point criterion and report that 0/4 comparisons support non-inferiority at the margin. Listed in §8. |
| "Fails on local race" understates F3 | "It fails on everything" | §7.5 states all four sensitive endpoints under both weightings, plus both source probes. |
| A pass/fail inversion in the frozen source-probe sentence | "Your own table says the opposite" | Table 5 and its caption report *passing* seeds explicitly. |
| "Two lock amendments" with three files present | "Which is it?" | §9 reports three, with times and the fact that the third post-dates every final read. |
| "Several times more" without a scale | "Six times or two-and-a-half times?" | §7.4 gives both and quotes the paired difference instead. |
| "20/20 development signs" | "Over seeds or over seed means?" | §7.8 and Figure 4: 20/20 seed means, 13/20 in every seed. |
| "Resolution, not direction" | "Point estimates moved 15-fold" | §7.8 reports the 0.054×–14.8× range and declines to attribute the change to sample size. |
| Family-level negative inference | "Two arms is not a family" | Withdrawn in §8 item 8, with the trace-form argument. |
| Residence capability reported only against H | "What about a real representation?" | §7.1 and Table 5 add the half-headroom failure. |
| Rank-rule misattribution | "Eigenvalue-sign selection is SARL 2019" | §4.1 attributes it correctly to three prior papers. |

## Tier 2 — answerable from existing evidence if raised

| Risk | Answer available now |
|---|---|
| "Your bootstrap treats three seed predictions as three people" | It does not: seeds are averaged *before* resampling and a person receives one resample count. Independently reconstructed; `STATISTICAL_REANALYSIS.md` §0. |
| "Weighted intervals ignore the weight denominator inside replicates" | Ratio estimators are recomputed per replicate; verified against the study to 4.4e-17 on standard errors. |
| "The simultaneous adjustment is over a family you chose after seeing results" | Families are in `COMPARISONS.json`, hashed in the lock, committed and pushed before the first final read (10 accesses, all after the lock timestamp). |
| "Negative increments mean your attacker is weak" | Both are reported: 22/429 negative cells, and the routed-H control shows the selected attack losing to H's own attack in 8/312 cells. The H attack remains executable. |
| "Adding frozen attackers looks like p-hacking the scope" | All four scopes and both budgets are reported; the primary scope was pre-declared; `SCOPE_COMPARISON.csv` shows the change is a baseline effect. |
| "You should have used a p-grid point that matches utility" | §7.6 does exactly that, in closed form and in both directions, and labels it post-hoc. |
| "Only three seeds" | Stated as a limitation; per-seed values are shown for every endpoint, and Figure 4 shows where they disagree. |
| "Your race conclusions rest on unsupported classes" | The comparable-category diagnostic reproduces the F1 race differences to the fourth decimal on supported classes only. |
| "Why should I believe the numbers?" | Independent scorer, independent selection re-derivation, independent bootstrap, independent lock recheck; all reported in §11 with agreement to 1e-15. |

## Tier 3 — genuine gaps requiring new data or new methods

| Gap | Why existing evidence cannot close it | What would |
|---|---|---|
| **No external method baseline.** SARL, K-TOpt, U-FaTE, OptNet-ARL and LEACE/SPLINCE are discussed but never run. | Nothing in this study measures them. The paper therefore cannot claim competitiveness with any of them — and does not. | The six adaptations specified in `CONTRIBUTION_ASSESSMENT.md` §6, each at a declared width, under identical attackers. A serious cost; not authorised here. |
| **The surrogate-mismatch diagnosis is not isolated.** Five candidate causes remain unseparated. | The design varies neither the feature map, nor the nuisance, nor the rank, nor the utility surrogate. | The 2×2 in `METHOD_REVIEW.md` §8 item 3 (Terminal A is running the penalty × rank half of it). |
| **The confirmation set is spent.** 2017 is used; 2018 is development. | No amount of reanalysis creates a fresh evaluation. | 2016 is now admitted with a prospective partition and a lock-enforcing loader (`DATA_2016_ADMISSION.md`), unscored. |
| **Attack strength is a floor.** | Any measured recovery is what the tested families achieved. | A stronger adaptive attacker, or a certificate — and the mechanism has no certificate by construction. |
| **Retraining variability is unmeasured.** | Three fixed systems; the bootstrap is over people. | Refitting the mechanism many times, which was never budgeted. |
| **Census design variance is unmeasured.** | PWGTP replicate weights are not in the stored evidence. | Replicate-weight variance estimation on a re-prepared extract. |
| **Cross-year person distinctness is unprovable.** | Public identifiers do not support it. | Nothing public does; it must remain a stated limitation. |
| **A/B asymmetry.** Only recipient A receives a channel. | The design never instantiates two channels. | A genuinely multi-channel study, which is a different paper. |

## Presentation risks that are not scientific errors

* The paper is a negative method result with a positive mechanism finding inside it. A reviewer
  skimming §7.2 alone will read it as a success. The abstract and §1 lead with the negative verdict
  deliberately; that ordering should not be softened.
* The withholding-matching result (§7.6) is the most novel analysis in the paper and is exploratory.
  It is labelled post-hoc in the table caption, the section text and the ledger. It must not migrate
  into the abstract as a registered finding.
* Ten withdrawn claims in §8 is unusual and invites the reading that the original work was careless.
  The opposite is nearer the truth: the study's own tables were right in every case, all 90 endpoints
  reproduce to 1e-15, and nine of the ten are wording. Section 8's preamble should keep saying so.
