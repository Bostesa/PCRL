# Advisor brief: does input-side generalization or coordinated SEX-blind flattening reduce SEX inference beyond matched controls?

Campaign `pcrl_input_and_pair_release_v1`, 2026-10-22 (start 01:19:06Z). Florida ACS 2019 development cohort (admitted public-use PUMS
records).
- **Data used.** MODEL_FIT for fitting and MODEL_SELECT (8,001 people) for every evaluation. CALIBRATION was not used. All other roles'
  labels, the reserved populations and the sealed assessment stayed closed.
- **Release.** Six input fields. An income recipient and an occupation recipient, each receiving only its own probability vector and
  decision.
- **Recovery.** The AUC of classifiers ("readers") that infer the recorded SEX attribute from what the recipients receive, from one
  output or from the linked pair. Development evidence on adaptively used records, for the declared readers only.

**Result in one line:** no method is established.
- **Utility:** every candidate met all 28 original utility conditions at the point. Point feasibility is not confidence preservation.
- **Matched controls:** no reading reduced recovery beyond its matched controls under the registered rule (D2: 0 of 2 registered and 0
  of 1 adaptive readings, 3 looks).
- **References and criterion:** none is competitive with the strong references or meets the 0.020 material criterion.
- **Where the leakage sits:** the twin's pair recovery (AUC 0.6480) is carried almost wholly by its occupation view (0.6443). Both
  families move that view by less than 0.01. When the occupation view alone was cut, the linked pair hardly moved, consistent (as point
  estimates) with recovery through the income vector.

The statistical reviewer's required headline is quoted verbatim in `RESEARCH_DECISION.md`.

## The designs

- **IA, shared SEX-aware field generalization (SFG; Designer A).**
  - A fixed menu of coarser levels of the six raw fields (age bands, hours bands, class-of-worker and schooling merges, marital status,
    place of birth) is ordered by SEX information removed per nat of task cost.
  - Both task models are refitted on the coarsened record, and both recipients see it.
  - Strength is chosen on MODEL_FIT under budget S3.
- **IB, coordinated keyed selective flattening (CSF; Designer B, redirected by the challenger).**
  - A keyed share pi of people receive, in both outputs, the mean vector of their decision cell; the decisions are kept exactly.
  - The operation is SEX-blind by construction.
  - The designer's SEX-targeted version (CSF-T) became the targeting ablation, and independent per-recipient keys (IND) the
    coordination ablation.
- **The challenger.**
  - **Not equivalent:** it judged the two designs to be different operations (different released function classes, SEX information and
    stage), so there was no redirect on equivalence grounds.
  - **Why IB was redirected:** its targeting layer was a diluted form of the closed HQ recipe, and the protocol reason Designer B gave
    for targeting no longer held.
  - **Required for IA:** a predeclared budget (S3), stricter on income than the designer's; the permuted-SEX ladder as IA's only gating
    SEX-independent control; and the generic release-only tilt as IA's declared release-only control.
  - **A prediction before any fit:** both families are information removal by merging, at about 0.3-0.7 pair AUC per nat against about
    2.4 for the closed tilts, so neither can reach 0.020 inside the utility budget. The readings are consistent with that prediction,
    which was qualitative, not a test.
  - **Its replacement idea**, a cross-recipient keyed swap (FS_SWAP), was specified by Designer B and not run: in its own plug-in world
    it is no better than occupation-only flattening.
- **IA's revision, IA_OSG (adaptive).**
  - **Why:** after the registered reading, the coordinator diagnosed that the shared ladder was stopped by S3's income components at
    MAR1 (F1) and that the leakage sits in the occupation view (F3). The card's F2 (leakage moved to the income view) is not
    supported by the records.
  - **What it does:** IA_OSG generalizes the occupation recipient's input only, with the twin's income output unchanged. Its matched
    controls are restricted to the occupation output.
  - **Numbering:** it fills IA's first revision slot, hence ADAPTIVE_R1. It is not the slack-weighted ladder that IA_SFG_v1 declared as
    its R1, whose trigger failed.

## The seven questions

**1. Task feasibility.** Met.
- **Each candidate** meets all 28 original utility conditions at the point.
  - **Clauses (12 per version):** accuracy loss <= 0.010, NLL excess <= 0.010 nats and summed Brier excess <= 0.005, against U0 and Ucal
    for both tasks.
  - **Guards (2 per version):** the men's and women's occupation summed-Brier guards (excess vs U0 <= 0.005).
  - In all, 24 clauses and 4 guards over the two versions.
- **IA_SFG** costs almost nothing: +0.0004 / +0.0001 nats over the twin.
- **IB_CSF** costs +0.0099 / +0.0108 nats with no accuracy loss.
- **IA_OSG** costs +0.0041 / +0.0120 occupation nats, with income identical to the twin.
- **The reserve heuristic**, reported separately: met by IA_SFG and IB_CSF. IA_OSG misses it in version 1 (ratio 1.39 < 2), where the
  men's occupation Brier guard binds.

**2. Improvement beyond matched controls (D2).** Not met.
- **The count.** 0 of 2 registered and 0 of 1 adaptive readings, 3 D2 looks in total. One AGG_POSITIVE comparison passes a true null in
  about 0.07-2% of replicates under six moderate NRM1 known laws, and about 8% when one version or split dominates. With 3 looks, the
  chance that at least one passes by chance is at most 3 times the per-look rate. A D2 look is an intersection of its gating
  comparisons: four for IA's looks (TWIN, SBC, SPEND, RO) and two for IB's SEX-blind look (TWIN, SPEND).
- **Notation.** Pair benefit = comparator AUC - candidate AUC, with se_agg, for the declared readers on the 8 registered cells.
- **IA_REG (SFG), NOT_SEPARATED:**
  - vs the twin: +0.0016 (0.0058);
  - vs its permuted-SEX bracket: +0.0033 (0.0056);
  - vs its spending bracket: +0.0006 (0.0053);
  - vs its release-only tilt bracket: -0.0009 (0.0058).
  - The candidate is one marital-status merge (version 0), plus five-year age bands (version 1): a near-identity release.
  - Its higher income-view AUC (0.5600 vs the twin's 0.5498) rests on one repetition in which the selected reader switched. The
    verifier's fixed LR and HGB readers in that repetition, the data-rich readers and the MODEL_FIT readers show no comparable
    difference, so this is not evidence that it leaks more through income.
- **IB_REG (CSF), NOT_SEPARATED:** +0.0069 (0.0057) vs the twin and +0.0007 (0.0081) vs SEX-blind quantization (its SPEND bracket,
  which spends 0.0037 nats more). Its gain over the twin is not separated from that spending control. No SEX-targeting claim is possible
  for a SEX-blind operation.
- **IA_R1 (OSG; adaptive; never IA's result), ADAPTIVE_R1__NOT_SEPARATED:**
  - **Occupation view:** 0.6351 (twin 0.6443), but the pair only +0.0017 (0.0062) vs the twin.
  - **SBC:** +0.0020 (0.0058).
  - **SPEND:** no bracket. Pooled over versions its MODEL_SELECT spend exceeds both occupation-only quantization points (by 0.0018 and
    0.0043 nats). Its version-1 spend grew from 0.0070 to 0.0120 nats, while the upper point's fell from 0.0104 to 0.0082. Both points are
    at the point at least as hard to read.
  - **RO:** it is not separated from any of its controls, and at the point it is more recoverable than its occupation-restricted
    release-only tilt (-0.0072, se 0.0064, against a bracket point that fails the utility limits in one version; -0.0051 against the
    feasible lower point; none resolved).

**3. Robustness to classifier refitting.** Matched reader refits on new household splits and reader seeds of the same people were
genuinely effective.
- **The family readings** are IA_OSG's (IA's strongest feasible release since the revision, by 0.0001: a tie in substance and a
  winner's-curse selection) and IB_CSF's. IA_SFG's is an extra reading of the registered candidate.
- **All three are `ROBUSTNESS_NOT_CONFIRMED`.** Refit-cell benefits:
  - vs the twin: IA_OSG +0.0028, IB_CSF +0.0061, IA_SFG +0.0018;
  - IA_OSG vs its occupation-restricted release-only tilt: -0.0057 (against IA_OSG_RO_HI, which fails G1 in version 1);
  - IB_CSF and IA_SFG vs their spending brackets: -0.0006 and +0.0018.
- **Component ablations** (descriptive, registered cells; none resolved):
  - IA targeting off, +0.0008 vs IA_SFG;
  - IB targeting on, -0.0002;
  - IB coordination off, +0.0016;
  - IA_OSG targeting off, +0.0042.

**4. Competitiveness against strong references (D3).** Not competitive.
- Six of the nine references are less recoverable (lower pair AUC 0.596-0.636) than every candidate.
- All three candidates are more recoverable than NEURAL_RAW and H_LJ by 0.016-0.028, resolved in five of the six comparisons.
- The previous campaign's strongest release, DDR_PTC_R1 (0.624), is resolved less recoverable than IA_SFG and IA_OSG and not resolved
  against IB_CSF. It is the dual-design campaign's adaptive revision, chosen with a winner's curse on the same people and never separated
  from its own release-only control.

**5. The material criterion (D4).** Not met. The criterion is a resolved 0.020 over NEURAL_RAW and H_LJ. Every candidate is more
recoverable than both at the point, resolved in five of the six comparisons (best -0.0158 / -0.0225, IB_CSF; the first is not
resolved). The differences against the twin, +0.0016 to +0.0069 and none resolved, are a smaller result
against a weaker baseline.

**6. Joint-design evidence (D5).**
- **IB:** its card declared before the fit that both sequential orders equal the coordinated design by construction
  (NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL; rebuilt bitwise by the verifier). The coordination contrast (against independent
  per-recipient keys, IND) is +0.0016 (se 0.0066), not resolved; coordination effects up to about 0.015 are not excluded.
- **IA:** not declared joint.
- No joint-design advantage is claimed.

**7. Novelty (D6).** Not claimed. Field generalization and suppression, the privacy funnel, microaggregation and coordinated
suppression are published; no literature search was run.

## What would change the picture (untested)

- **A pair-aware or link-breaking design.** IA_OSG's point estimates are consistent with the linked pair recovering an occupation-only
  reduction through the income vector, but nothing tested that mechanism. FS_SWAP, the only design aimed at the pair link, was not run.
- **A wider budget than S3.** Within S3 the income input cannot be coarsened past its first informative step, so a pair-aware
  input-side design needs a different budget. That is why no second IA revision was run.
- **IA's substituted release-only path.** It was not built (the declared generic-tilt fallback), so IA's RO comparisons are not an
  information bound.
- **More precision.** At 2 SE the data exclude pair benefits over the twin above about 0.013-0.018, and effects below about 0.01 were not
  resolvable. More cells on these people would be another adaptive look; a confirmation population was not authorized.
- **Other ideas and readers.** The challenger's input-PRAM idea was not run, and no reader beyond the declared bank and the three
  data-rich recipes was tried.

## Resources

- **Elapsed:** 4.58 h of the 8 research hours.
- **CPU:** 4.282 of 12 CPU-hours (3.986 in 91 queue units, one of which failed; 0.296 in agent and coordinator charges).
- **Paid compute:** $0.
- **Configurations:** 39 of 48. **Agents:** never more than six live at once, including the coordinator.
- **Access:** 47 label reads, all after the registration receipt; CALIBRATION and the closed roles' labels were never read.
- **Verification:** the independent verifier replayed every decision record exactly, in two reports.
- **Custody:** same-device custody was verified from the backup alone (narrower, compressed scope for disk space). Off-device custody
  is pending (drive not attached).
- **Publication:** the publication commit is recorded in `PUBLICATION_RECEIPT.json`.
