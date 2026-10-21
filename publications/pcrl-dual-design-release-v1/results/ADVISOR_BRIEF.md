# Advisor brief: does a materially different release reduce SEX inference beyond matched controls?

Campaign `pcrl_dual_design_release_v1`, 2026-10-21 (start 16:37:33Z). Florida ACS 2019 development cohort (admitted public-use PUMS
records).
- **Data used.** MODEL_FIT for fitting and MODEL_SELECT (8,001 people) for every evaluation. CALIBRATION was not used. All other
  roles' labels, the reserved populations and the sealed assessment stayed closed.
- **Release.** Six input fields. An income recipient and an occupation recipient, each receiving only its own probability vector
  and decision.
- **Recovery.** The AUC of classifiers ("readers") that infer the recorded SEX attribute from what the recipients receive, from one
  output or from the linked pair. Development evidence on adaptively used records, for the declared readers only.

**Result in one line:** no method is established.
- Every release met all 28 original utility conditions at the point (12 clauses and 2 guards in each of its two version-specific
  configurations; all 48 configurations pass). Point feasibility is not confidence preservation.
- Neither family reduced recovery beyond its matched controls under the registered rule.
- Neither is competitive with the strongest references (D3 NOT_SUPERIOR in every reading) or meets the 0.020 material criterion (D4
  NOT_MET). The largest point values, DB_PTC_R1's, are +0.0110 (se 0.0135) over NEURAL_RAW and -0.0058 (se 0.0151) over H_LJ.
- Both families test one idea family. STD and PTC each subtract the propensity-proportional part of the released task posterior, and
  the revisions stay inside it, so this joint negative closes one idea, not two.

The statistical reviewer's required headline is quoted verbatim in `RESEARCH_DECISION.md`.

## The designs

- **DA, SEX-standardized task posteriors (Designer A).** SEX-aware task models are averaged over both SEX values with fixed weights,
  so the person's SEX propensity no longer enters through the mixture weights. The release is still a function of the six fields,
  which predict SEX. On MODEL_FIT its income log-probabilities kept about two thirds of the twin's linear propensity dependence
  (R^2 0.066 vs 0.100). Revision R1 (IPW-STD) instead trains SEX-blind models with inverse-propensity weights.
- **DB, propensity-tilt cancellation (Designer B).** Each recipient's log-probabilities are tilted against a cross-fitted SEX
  propensity, with closed-form weights that cancel the propensity's covariance with the released scores. The release writes the
  propensity into the vector and is injective on its support (a design property stated in its card; not tested here). Revision R1
  gives each recipient its own strength (0.35 / 0.65).
- **The challenger.** It found the two designs to be different operations within one idea family: both subtract the
  propensity-proportional part of the posterior. So neither designer was redirected, and a joint negative closes one idea, not two.
  Before any fit it required three things:
  - ZTILT, the same tilt driven only by what can be estimated from the twin's release, as a saved DB control with a reading declared
    in advance;
  - data-rich readers;
  - a fairer SEX-independent fallback target for both families' SBC paths.

  On its advice, adopted in a stronger form, the SPEND path became SEX-blind dithered quantization.

## The seven questions

**1. Task feasibility.** Met. Every release meets all 28 original utility conditions at the point, and all 48 version-specific
configurations pass their 12 clauses and 2 guards (accuracy loss
<= 0.010, NLL excess <= 0.010 nats, summed Brier excess <= 0.005; four occupation subgroup guards). The reserve heuristic is reported
separately. The DB revision DB_PTC_R1 misses the two-SE reserve in version 1 (ratio 1.87 < 2), as do its ablation and several
controls, so it is near the limits.

**2. Improvement beyond matched controls (D2).** Not met.
- **The count.** 0 of 2 registered and 0 of 2 adaptive readings, 4 D2 looks in total. One AGG_POSITIVE comparison passes a true null
  in about 0.07-2% of replicates under six moderate NRM1 known laws, and about 8% when one version or split dominates. With 4 looks,
  the chance that at least one passes by chance is at most 4 times the per-look rate.
- **Notation.** Pair benefit = comparator AUC - candidate AUC, with se_agg, for the declared readers on the 8 registered cells.
- **Registered DA (STD), PARTIAL_SBC_FRONTIER_ONLY:**
  - vs the twin: +0.0175 (0.0090), not resolved;
  - vs the SEX-blind spending control: +0.0137 (0.0174), not separated;
  - vs the SBC control: +0.0273 (0.0104), resolved but weak, because those controls are more recoverable than the twin at the
    point.
- **Registered DB (PTC), NOT_SEPARATED:** +0.0164, +0.0173 and +0.0170 against the twin and both controls, none resolved. It is not
  separated from ZTILT either.
- **DB revision R1 (DB_PTC_R1), ADAPTIVE_R1__PARTIAL_TWIN+SPEND_FRONTIER_ONLY.** Selected afterwards, on the same registered cells, as
  DB's strongest feasible release (winner's curse):
  - vs the twin: +0.0239 (0.0101), resolved;
  - vs the SPEND bracket: +0.0245 (0.0122), passing the SPEND frontier comparison with a lower 2-SE bound of only +0.0001;
  - vs the SBC bracket: +0.0265 (0.0139), not separated (lower bound -0.0013).
- **DA revision R1:** ADAPTIVE_R1__NOT_SEPARATED, +0.0072 vs the twin. It was run outside its declared trigger, after the registered
  DA outcome.

**3. Robustness to classifier refitting.** Matched reader refits were run on new household splits and reader seeds of the same people.
They were genuinely effective.
- **DA_STD (DA's family D2R):** `ROBUSTNESS_NOT_CONFIRMED`. On the refit cells it is +0.018 vs the twin and +0.010 vs the spending
  control.
- **DB_PTC (registered; an extra reading):** not confirmed, +0.013 and +0.010.
- **DB_PTC_R1 (DB's family D2R):** `ROBUSTNESS_CONFIRMED` on the refit cells against the twin (+0.0219, se 0.0103), its SBC control
  (+0.0231, se 0.0091) and its spending control (+0.0189, se 0.0091). This is descriptive: D2 is unchanged, the same people are
  used, the refit estimates are below the registered ones, and two of the three clear the 2-SE threshold by about 0.001.
- **DB_PTC_R1 against its release-only control DB_ZTILT_R1.** Read in the same breath: +0.0128 (se 0.0081) on the registered cells and
  +0.0066 on the refit cells, neither resolved. DB_ZTILT_R1 is a tilt whose amplitude is estimated from the twin's released vectors
  alone. By the card's declared reading, the gain is not separated from release-only information, which is the failure that closed
  the earlier propensity-routing line. That control was intended to match the release's spend but overspent (0.0169 vs 0.0099
  nats). This is disclosed and does not change the reading.
- **Component ablations** (descriptive; no ablation difference against the twin is resolved):
  - with the person's own propensity as the SEX weight (DA_COMP), the release is resolved more recoverable than DA_STD;
  - the occupation-only ablations keep about 60% of DA_STD's point benefit and about 40% of DB_PTC's and DB_PTC_R1's.

**4. Competitiveness against strong references (D3).** Not competitive. Four inherited references have a lower pair AUC (0.600-0.622)
than every release here (DB_PTC_R1 0.627; DA_STD 0.633). DB_PTC_R1 is resolved worse than RETIRED_B3CTL (by 0.027); the other three
differences are not resolved.

**5. The material criterion (D4).** Not met. The criterion is a resolved 0.020 over NEURAL_RAW and H_LJ; the best is +0.011 and
-0.006. The differences against the twin, +0.007 to +0.024 (resolved only for the adaptive DB_PTC_R1), are a smaller result against
a weaker baseline.

**6. Joint-design evidence (D5).** Not assessed. Separately designed recipients and both sequential orders were not built, and no
joint-design advantage is claimed. They were not planned: the protocol registered D5 as not assessed without them, and the
configuration budget allocated none. DB's revision differs from DB_PTC only in its occupation strength. Its pair difference from DB_PTC
is +0.0075 (se 0.0120), not resolved.

**7. Novelty (D6).** Not claimed. The devices have published antecedents:
- STD: Pope and Sydnor 2011 and Pearl's adjustment formula;
- PTC: AttriGuard, LEACE and multiaccuracy-style proxies;
- IPW-STD: inverse-propensity reweighing.

## What would change the picture (untested)

- **A confirmation population.** This was not authorized here, and any new reading on these people would be a fifth adaptive look.
  It would carry a registered reading of DB_PTC_R1 against its SBC bracket and against a release-only ZTILT point that is
  spend-matched along its split law. That needs a finer ZTILT scale grid: the registered one jumped from 0.90x / 0.75x to
  2.08x / 1.73x of the release's spend. A finer SBC grid would not help, because the SBC bracket is already within 0.0008 nats of
  the release.
- **More precision.** The effects sit at the grid's resolution: realized se_agg was 0.009-0.018, against the 0.005 the protocol planned
  for.
- **A different family.** Coarser occupation outputs, as the strongest references use, are a different family. So is any operation
  that does not subtract the propensity-proportional part.
- **Joint-design controls.**

## Resources

- **Elapsed:** 3.60 h of the 8 research hours.
- **CPU:** 3.077 of 12 CPU-hours (2.809 in 84 queue units, one of which failed; 0.268 in agent and coordinator charges).
- **Paid compute:** $0.
- **Configurations:** 48 of 48. **Agents:** at most six live at once, including the coordinator.
- **Access:** 63 label reads, all after the registration receipt; CALIBRATION and the closed roles' labels were never read.
- **Verification:** the independent verifier replayed every decision record exactly.
- **Custody:** same-device custody was verified from the backup alone; off-device custody is pending (drive not attached).
- **Publication:** the publication commit is recorded in `PUBLICATION_RECEIPT.json`.
