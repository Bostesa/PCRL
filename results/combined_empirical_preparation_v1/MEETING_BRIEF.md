# Meeting brief: empirical analysis first (preparation for Dr. Yus)

Prepared 2026-10-01. This is a preparation phase only.

- No new model fits, cloud jobs or new data were used, and new compute spend was $0.
- Every number below is either recounted from stored results or produced by small synthetic fixtures.
- Pins:
  - PCRL `origin/main` 55e4cb1d1;
  - durable-guarantees 956f5c88;
  - other refs in BOTH_REPOS_INDEX.json.

## How the pieces connect

- **The attribute-removal analysis** (durable-guarantees, "Outputs Leak What They Use") asks whether a
  method's protection survives a realistic recipient.
- **PCRL** asks how recipients with different prediction purposes can receive useful information.
  - Its encoder paper used per-purpose adapters with linear erasure.
  - Its later ACS studies used finite-token releases.
- **The revised empirical analysis** would check protection, utility and permitted combinations under
  one consistent contract.
- **A future algorithm** should be motivated only by a gap that this analysis actually demonstrates.

The three lineages are separate work, and no code path combines them:

| Lineage | Repository | Venue status |
|---|---|---|
| Encoder | PCRL | NeurIPS 2026, withdrawn 21 Aug |
| Attribute-removal analysis | durable-guarantees | AAAI-27, not advanced at Phase 1 on 24 Sep |
| ACS release studies | PCRL | SaTML '27, withdrawn 26 Sep |

## 1. What each repository contributes

**durable-guarantees**
- A reusable attacker battery.
- A re-audit of 67 linear-certified configurations: 18 were approved by the certificate, and 15 of those
  collapse under XGB/MLP. The README's "21" is wrong.
- A seven-method gauntlet using official code where it exists.
- A label-coupling cost relation (r = 0.80, n = 27).
- Repeated-query and DP side studies.

**PCRL**
- The scorers the analysis depends on: one-hot R², dominant axis, and the unsupported-class fix.
- Per-recipient and joint LEACE machinery, and a cross-purpose concatenation audit.
- Sound coalition-attack slates with nested references (ACS lineage).
- A large set of cautious, closed-negative release studies.
- durable-guarantees imports PCRL code and checkpoints, so PCRL is the dependency root.

**Shared fits and data.** Some results that appear in both repositories come from the *same* fits:
- the Adult and HMDA checkpoints;
- the CelebA network;
- the California 2018 ACS data.

They must be counted once.

## 2. What is already known

The literature review is at ANALYSIS_POSITIONING §1. It covers 39 papers, including 8 reviewer-cited
or mandatory papers read in full.

Already established elsewhere:
- Linear tests miss nonlinear recovery (Ravfogel et al. 2022, both papers).
- Training-time adversaries are fooled while fresh attackers succeed.
- Recoverability is not causal use (Elazar et al. 2021).
- Representations overlearn and can be de-censored (Song & Shmatikov 2020).
- Label-only leakage is measurable before training, and universal least privilege is impossible
  (Stadler et al. 2024).
- Model outputs leak more than labels at matched accuracy (Wang et al. ICCV 2019). Neither manuscript
  cites this.
- Certificates have been checked against held-out classifiers (FARE; FRG, NeurIPS 2025).
- Averaging undoes isolated-subspace noise (FairNVT, TMLR 2026; the earlier version was at an ICLR
  workshop, not the main conference).
- Coalition budgets exist for finite alphabets (Taylor et al. 2026).

**Our analyses mostly reproduce these.** Specific overclaims to remove:
- AAAI "we contribute its measurement" of label-only leakage.
- Elazar et al. cited for the wrong point.

## 3. Is an additional empirical question worth pursuing?

**Recommendation: narrow it, and pilot before committing.** This is a judgement call for discussion.

**Primary question.** Across method families, how often does each method's *own* stated test agree with
recovery by a defense-aware recipient? The conditions:
- at matched **held-out** utility;
- including the task output, measured against a label-only reference;
- under a declared finite sensitive set.

**Secondary question.** Does label coupling predict the **held-out** utility cost of reaching a fixed
protection level? This is the most specific candidate contribution. It is also fragile: the existing
r = 0.80 uses in-sample utility and cells chosen after earlier results.

**Multi-recipient combination** belongs as a scenario, not the headline. Our own evidence weakens it:
- In the encoder lineage, single-recipient leakage dominates. In 17 of 33 cells, a recipient that is
  not allowed an attribute already recovers it from its own view.
- In the ACS lineage, the combination effects are small, near a null-token floor, and on spent data.

**Reconsider the whole direction if the one-cell pilot shows either of these:**
- the cost relation vanishes under held-out utility; or
- method tests already agree with defense-aware recovery once utility is held out.

## 4. Most consequential methodology fixes

Ordered by impact. Details are in METHODOLOGY_IMPROVEMENTS.md.

1. **The "informed / Tier 2" attacker is mislabelled. [verified]**
   - It is fit on *clean* representations of a training population and scores one noised release.
   - It never sees the target's clean vector and never averages repeated queries, contrary to the
     README and paper.
   - Repeated queries were never tested on the full-rank channel marked as passing.
   - Fix: name every attacker by its access, A1–A5.
2. **Utility is in-sample while attackers are held out. [source + recount]**
   - Adult clean lift is 0.185 in-sample vs 0.069 held-out.
   - PCRL's "within 1pp of the unconstrained backbone" holds for only 2 of 7 tasks.
   - PCRL tuned on its test splits.
   - Fix: one held-out, linkage-grouped utility contract for every arm.
3. **The output surface is claimed but not gated.**
   - The audit and gauntlet verdicts use the representation only.
   - Fix: gate on both surfaces, against label-only and clean-output references.
4. **The worst-class metric misses class contrasts. [verified, fixtures F1–F3]**
   - "Dominant axis" is the maximum single-class R². It misses class contrasts by up to (K−1)×. At
     K = 10 it reads 0.047 (passing) while a contrast reaches 0.364.
   - The certificate on public main is computed in float32. It understates R² and certifies 0 where
     the truth is 0.578 in a stress case.
   - Absent and constant classes are scored as 1.0 or dropped.
   - Fix: top canonical correlation, per-class support, and "not estimable" reporting.
5. **Thresholds were never registered.**
   - The 0.55 bar was set after an early result. Register a grid of thresholds, selected on validation.
6. **Matched comparisons don't match.** Corrected numbers are in the next section.
7. **Several baselines are not what they are called.**
   - The in-house "LEACE" is not LEACE; it under-erases.
   - "R-LACE" is an INLP loop.
   - The protected attribute is a model input everywhere.
   - Erasers are fit on the rows that are then attacked. This is exploitable: Johansson 2024 inverts
     projection erasure fitted on the released rows by anti-clustering.
8. **Architecture claims don't match the code.**
   - The tabular "pretrained backbone" is a frozen, randomly initialised network.
   - vCLUB was active although the paper says it was off.
   - The CelebA vision check uses PCRL's own network, which treats Young as disallowed.
   - Splits ignore identity, household and patient linkage.
9. **Public PCRL main still ships the refuted R²→accuracy guarantee.**
   - The 20-row counterexample reproduces.
   - A fix branch exists but is unmerged. Nothing was merged in this phase.

### Numbers that change

| Claim | Stated | Corrected |
|---|---|---|
| Full-rank channel at the isolate channel's protection | keeps 1.9% / 38.6% | about 55.5% / 56.6% (frontier matching instead of nearest-match) |
| AAAI easy-cell utility | 49.7% | 41.0% (the registered 5-seed result) |
| PCRL strict compliance, final checkpoint vs best-validation checkpoint | 56/60 | 54/60 |
| PCRL "7/60 cleanly compliant" | 7/60 | 5/60 on a single checkpoint |
| Rebuttal cross-purpose result | 8/33 | 19/33 under the absolute criterion (8/33 was a different model plus a post-hoc criterion) |

## 5. Methods for the first revised comparison

The full catalog is METHOD_CATALOG.csv (51 methods); the shortlist is in `notes/methods/PRIORITY_LIST.md`.

**Core, CPU, all pinned code:**
- Anchors: release the task only; withhold; a label-only predictor.
- LEACE (`concept-erasure` 0.2.4), per recipient and over the union of attributes.
- Official SPLINCE.
- Official INLP.
- The isotropic noise channel.
- Official FARE. FARE has no licence, so run and cite only.

**Extended:**
- A DANN-style adversary, with an official-LAFTR check on binary cells.
- VFAE.
- Obliviator. It never reached its own stopping rule, so report it as "did not reach own criterion".
- The PCRL erase layer, with a no-LEACE ablation. That ablation was registered but never run.

**Missing implementations:**
- No collusion-aware learned method exists.
- No official code for VFAE, Gitiaux & Rangwala, or FairNVT. Gitiaux & Rangwala and FairNVT are
  missing baselines for the noise channels.
- RLACE has never actually been run.
- Binary-only methods are N/A on multiclass cells, never scored as 0.

## 6. Decisions needed from Dr. Yus

1. **Direction.** Accept the narrowed question (§3), or reconsider the contribution?
2. **Policy.** Approve the declared sensitive set and the recipient/coalition table (protocol §2):
   - Is the auditor allowed to share with the underwriter?
   - Is the protected attribute excluded from model inputs as the primary regime?
3. **Thresholds and margins.** Approve the threshold grid, minimum utility lift, and the support and
   margin values marked [DECIDE].
4. **Confirmation population.** Choose a population that has never been opened, such as ACS 2018 Texas
   or New York. It must be chosen by an outcome-blind rule written before download.
5. **Pilot.** Authorise the one-cell pilot (about 1 CPU-hour, on spent development data) before any
   benchmark.
6. **One paper or two.** Defer until the outline (ONE_PAPER_OUTLINE.md) and the evidence map have been
   reviewed.
7. **Public main.** Whether to merge the existing retirement branch, which would remove the refuted
   guarantee from public main.

## 7. Verified, proposed, missing

**Verified (independent fixtures plus recounts):**
- F1–F10, all in INDEPENDENT_CHECKS.json.
- Recounts:
  - durable-guarantees: 30 recounts, 21 match and 9 flagged;
  - PCRL encoder: recounted with checkpoint-dependence;
  - ACS: 39 of 39 match.
- Nested coalition slates in the ACS evidence: correct.

**Proposed (not agreed):**
- Access tags A1–A5.
- The utility contract, policy table, threshold grid, inference rules and pilot.
- The CelebA identity-disjoint split.
- The accuracy-guarantee retirement.

**Missing:**
- **Official review text.** It is not available locally or by email, and OpenReview requires a login.
  This means:
  - the two references from the reviewer identified in the meeting summary are unresolved;
  - the reviewer's requested threshold range is unknown;
  - attribution of every concern rests on the meeting summary.
- The meeting transcript file.
- The final NeurIPS source (.tex).
- The source of the SaTML PDF.
- durable-guarantees' raw score arrays. They exist only on the relocated external drive, which is not
  mounted.
- The S3 archive check. AWS credentials have expired.
- `PCRL_Joint_Purpose_Algorithm_Execution_Prompt.md`. It is not on this machine; no job from it is
  running locally. Cloud status could not be checked.

**Data is spent.**
- California ACS 2016, 2017 and 2018 are all used up.
- The Adult, HMDA and Diabetes test splits were used for tuning.
- All re-scoring on these is development evidence only (DATA_EXPOSURE_LEDGER.csv).
