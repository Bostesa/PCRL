# Research decision: an input-side family and a joint pair-release family (campaign `pcrl_input_and_pair_release_v1`)

Florida ACS 2019 development cohort (admitted public-use PUMS records), 2026-10-22 (start 01:19:06Z).
- **Roles:** MODEL_FIT for every fit; MODEL_SELECT (8,001 people) for every evaluation. CALIBRATION was not used. The other roles'
  labels, the reserved populations and the sealed assessment stayed closed.
- **Scope:** development evidence on adaptively used MODEL_SELECT records, for the declared readers only. Lower recovery for these
  readers is not protection against every reader.

## Headline

The statistical reviewer's required headline, quoted verbatim from `roles/statistical_reviewer/CLOSING_STATEMENT.md` section 2:

> This is development evidence on previously and adaptively used MODEL_SELECT records of the admitted ACS 2019 Florida cohort (8,001
> people), for the declared readers only: it is not confirmation, and lower recovery for these readers is not protection against every
> reader. Two independently designed release families, an input-side SEX-ordered field generalization (IA_SFG) and a SEX-blind
> coordinated keyed collapse of both outputs to decision-cell means (IB_CSF), met the original utility limits at the point, but neither
> reduced the readers' recovery of the recorded SEX attribute beyond its matched controls: D2 was met in 0 of 2 registered family
> readings and in 0 of 1 non-registered (ADAPTIVE_R1) reading, 3 D2 looks in total; one AGG_POSITIVE comparison passes a true null in
> about 0.07-2% of replicates under six moderate NRM1 known laws and about 8% when one version or split dominates, and with 3 looks the
> chance that at least one passes by chance is at most 3 times the per-look rate. Registered readings (linked-pair benefit = comparator
> AUC - candidate AUC, mean (se_agg) over the 8 registered cells): IA_SFG generalized almost nothing inside its utility budget (one
> marital-status merge, plus five-year age bands in one version) and is NOT_SEPARATED: +0.0016 (0.0058) over the equally trained twin,
> and +0.0033 (0.0056), +0.0006 (0.0053) and -0.0009 (0.0058) over its SEX-independent-target, SEX-blind-spending and generic
> release-only-tilt brackets. IB_CSF is NOT_SEPARATED: +0.0069 (0.0057) over the twin and +0.0007 (0.0081) over its SEX-blind
> spending bracket; as a SEX-blind operation it admits no SEX-targeting claim. Both are TASK_ONLY_FEASIBLE_NO_PROTECTION, and neither
> registered release's benefits are confirmed on refit cells (D2R ROBUSTNESS_NOT_CONFIRMED with genuinely effective refits). Beside
> them, never as a family result: IA's adaptive revision IA_OSG (occupation-only generalization, the income recipient keeping the
> twin's output) is ADAPTIVE_R1__NOT_SEPARATED; it lowered the occupation view (0.6351 against 0.6443) but not the linked pair (+0.0017
> (0.0062) over the twin), and at the point it is more recoverable than its occupation-restricted release-only tilt (-0.0072 (0.0064),
> unresolved, against a bracket point that fails the utility limits in one version). No reading is competitive with the nine inherited
> references (D3 NOT_SUPERIOR) and none meets the 0.020 material criterion (D4 NOT_MET): the candidates are 0.016-0.028 more
> recoverable than NEURAL_RAW and H_LJ. The +0.0016 to +0.0069 differences against the twin are smaller, unresolved results against a
> weaker baseline; at the 2-SE level the data exclude pair benefits over the twin above about 0.013 (IA_SFG), 0.018 (IB_CSF) and 0.014
> (IA_OSG). No release is a method, candidate or component; joint design is not claimed (IB's sequential orders equal its joint design
> by construction) and novelty is not claimed.

## Decisions (frozen IPR1 rules; `claims/<reading>/DECISIONS.json`)

| Decision | IA_REG: shared SEX-aware field generalization (SFG) | IB_REG: coordinated keyed selective flattening (CSF; SEX-blind) | IA_R1: occupation-side generalization (OSG; ADAPTIVE_R1, never IA's result) |
|---|---|---|---|
| D1 task feasibility | TASK_FEASIBLE (reserve met) | TASK_FEASIBLE (reserve met) | ADAPTIVE_R1__TASK_FEASIBLE; reserve met in version 0 only (1.39 in version 1) |
| D2 beyond matched controls | NOT_SEPARATED | NOT_SEPARATED (SEX-blind route: TWIN and SPEND gate; SBC and RO degenerate by construction) | ADAPTIVE_R1__NOT_SEPARATED (flags NO_BRACKET_SPEND, CONTROL_GRID_COARSE_SBC / _SPEND) |
| D2R refit robustness (descriptive) | ROBUSTNESS_NOT_CONFIRMED (refits genuinely effective; an extra reading of the registered candidate) | ROBUSTNESS_NOT_CONFIRMED (refits genuinely effective; IB's family D2R) | ROBUSTNESS_NOT_CONFIRMED (refits genuinely effective; IA's family D2R, since IA_OSG became IA's strongest release; D2R labels carry no adaptive prefix) |
| D3 strong references | NOT_SUPERIOR | NOT_SUPERIOR | ADAPTIVE_R1__NOT_SUPERIOR |
| D4 material criterion | NOT_MET | NOT_MET | ADAPTIVE_R1__NOT_MET |
| D5 joint design | NOT_APPLICABLE_NOT_JOINT | NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL (declared before the fit) | ADAPTIVE_R1__NOT_APPLICABLE_NOT_JOINT |
| D6 novelty | NOT_CLAIMED | NOT_CLAIMED | ADAPTIVE_R1__NOT_CLAIMED |
| Outcome class | TASK_ONLY_FEASIBLE_NO_PROTECTION | TASK_ONLY_FEASIBLE_NO_PROTECTION | ADAPTIVE_R1__TASK_ONLY_FEASIBLE_NO_PROTECTION |

**Scope** (`D2.scope` of each reading).
- **IA_REG and IA_R1:** the RO controls are the generic release-only tilt of the twin, not the operation itself; the same operation
  driven only by the twin's release was not built.
- **IB_REG:** SEX-blind. SBC and RO are degenerate by construction, and D2 was assessed against the twin and the SEX-blind spending path
  only. No SEX-targeting claim and no claim about SEX information beyond the twin's release is possible. SEX entered IB only through
  the MODEL_FIT strength guards, which fixed pi* together with income NLL, as a utility constraint, not an input of the operation.

**The enhanced-winner flags.** The flags READER_SENSITIVE_ENHANCED_* (IA_REG: _IA_RO_HI / _IA_SPD_HI; IB_REG: _IB_SPD_HI; IA_R1:
_IA_OSG_RO_HI) are reported only. In every targeting entry of the three readings the enhanced (TABLE_CV) winner equals the non-table
winner, so these flags repeat the targeting failures and are not a separate reader-sensitivity result.

**Notation.**
- **Pair benefit** = comparator AUC - candidate AUC on the 8 registered cells (positive favours the candidate), with se_agg.
  "Resolved" = mean - 2 se_agg > 0 with both version means > 0; "point only" = positive but not resolved.
- **Registered-cell AUC** (winner SCORE AUC; pair / income / occupation):

  | Release | Pair | Income | Occupation |
  |---|---|---|---|
  | shared twin CTL_TWIN | 0.6480 | 0.5498 | 0.6443 |
  | IA_SFG | 0.6464 | 0.5600 | 0.6408 |
  | IB_CSF | 0.6411 | 0.5525 | 0.6364 |
  | IA_OSG | 0.6463 | 0.5498 (the twin's) | 0.6351 |

  The twin's pair recovery is carried almost wholly by its occupation view. IA_SFG's income-view figure is a reader-selection
  sensitivity, not a resolved difference (question 2).

## The questions of the brief

**1. Task feasibility.** Met for all three candidates. Every evaluated configuration passes G1 except one control.
- **Utility conditions:** each candidate passes all 28 original conditions at the point (12 clauses and 2 guards in each version).
- **IA_SFG** barely moves utility: MODEL_SELECT paired total NLL over the twin +0.0004 / +0.0001 nats. Its reserve ratios are
  4.67 / 4.27.
- **IB_CSF** spends +0.0099 / +0.0108 nats with exactly zero accuracy loss (decisions kept by construction). Reserve 2.97 / 2.30.
- **IA_OSG** spends occupation NLL +0.0041 (se 0.0010) / +0.0120 (0.0017). Its income output is the twin's bitwise, so its income cost
  is exactly 0. It is point-feasible in both versions but misses the two-SE reserve in version 1 (ratio 1.39, bound by the men's
  occupation Brier guard), so it carries
  POINT_FEASIBLE_NOT_RESERVE_QUALIFIED. Point feasibility is not confidence preservation.
- **The one infeasible control:** IA_OSG_RO_HI (the upper point of the revision's RO path) fails G1 in version 1. Its men's occupation
  Brier guard is at 1.005 of the cap. The rules gate G1 only for the candidate and the twin.
- **Reserve below 2 elsewhere:** the IB targeting ablation IB_CSFT (1.62, version 0), and several controls of the revision.
- **The one-version control point IA_SBC_LO.**
  - The frozen rules pool each path point over both versions, and a point recorded in one version only makes D2 INCOMPLETE (the
    reviewer's invented-data check). So leaving IA_SBC_LO out of the path was the only way to obtain a reading.
  - Leaving it out removes only the cheaper-point check (c3) in version 0. SBC_FRONTIER fails at its bracket with or without it, and
    the verifier's numbers show it would not be resolved less recoverable (O1).
  - As a saved configuration without utility evaluation or reader cells it breaches PROTOCOL 4's equal-evaluation clause (disclosed,
    deviation 2). No label depends on it.

**2. Improvement beyond matched controls (D2).** Not met in any reading: 0 of 2 registered, 0 of 1 adaptive.
- **IA_REG (SFG).** The S3 budget stopped the SEX-ordered ladder at its first informative step. The candidate is marital status reduced
  to married / not married (version 0), plus five-year age bands (version 1).
  - **The comparisons:**
    - vs the twin: +0.0016 (0.0058), point only;
    - vs its SEX-independent-target bracket: +0.0033 (0.0056);
    - vs its SEX-blind spending bracket: +0.0006 (0.0053);
    - vs its release-only tilt bracket: -0.0009 (0.0058).
    - None is resolved.
  - **Control geometry.** The candidate spends almost nothing, so its SPEND and RO brackets are near-identity points (DQ step 0.177;
    tilt 0.17 / 0.21). Its SBC bracket spends 0.0050 nats more than the candidate on MODEL_SELECT, because the permuted ladder's first
    step that reaches the candidate's spend + 2 SE is a whole field step. This favours the control in the SBC comparison and changes
    no label.
  - **The income view.** IA_SFG's income-view winner AUC is 0.5600 against the twin's 0.5498, and its income recipient gate failed at
    the point against the twin, its SBC bracket and its release-only bracket.
    - **What it rests on:** one repetition's cells (S1R1: +0.038 against the twin), in which the validation-selected income reader
      switched.
    - **What the other readers show:**
      - in that repetition the verifier's fixed LR and HGB readers give +0.001 and +0.0002, while its own validation-chosen reader
        reproduces the switch (+0.034, verifier O4);
      - the data-rich income readers give -0.0010 / +0.0003 in the two versions;
      - the descriptive MODEL_FIT income readers give 0.5579 / 0.5560, against the twin's 0.5581 / 0.5571;
      - the SEX-blind dithered quantization of the twin (IA_SPD_HI), which removes information, reads 0.5580 on the same grid.
    - **Reading:** this is a reader-selection sensitivity of the weak income view, not evidence that the refitted income model leaks
      more. No income-view statement is resolved.
- **IB_REG (CSF).**
  - vs the twin: +0.0069 (0.0057), point only, and version-1-dominated (version means +0.0037 / +0.0102);
  - vs its SEX-blind spending bracket: +0.0007 (0.0081), not separated. The bracket spends 0.0037 nats more than IB_CSF on MODEL_SELECT
    (version 0: 0.0064; version 1: 0.0010); the interpolated point, reported only, is +0.0030 (se 0.0072).
  - As a SEX-blind operation, no SEX-targeting claim is possible whatever the outcome.
- **IA_R1 (OSG; adaptive; never IA's result).** The occupation recipient's input is generalized while the income recipient keeps the
  twin's output. The candidate is marital status suppressed (version 0) and ten-year age bands, three hours bands and marital status
  suppressed (version 1).
  - **Occupation view:** 0.6351 against the twin's 0.6443.
  - **Linked pair:** +0.0017 (0.0062) against the twin and +0.0001 (0.0061) against IA_SFG.
    - That is consistent with the pair reader recovering the occupation reduction through the unchanged income vector, the risk the
      card named (FAILURE_MAP row 8 and failure note 2).
    - This is a reading of point estimates, not a resolved effect.
  - **SBC:** +0.0020 (0.0058) against its SEX-independent-target bracket. That bracket spends 0.0061 nats more than the candidate (the
    grid is coarse in version 0).
  - **SPEND:** no bracket on MODEL_SELECT.
    - Pooled over versions, IA_OSG's MODEL_SELECT spend exceeds both occupation-only quantization points: by 0.0018 (se 0.0018) and
      0.0043 (se 0.0017). In version 0 it is 0.00003 nats below IA_OSG_SPD_HI.
    - Its version-1 spend grew from 0.0070 (MODEL_FIT) to 0.0120 nats, while IA_OSG_SPD_HI's fell from 0.0104 to 0.0082.
    - Both points are, at the point, at least as hard to read (-0.0008, se 0.0058, each).
  - **RO:** IA_OSG is not separated from any of its controls. At the point it is more recoverable than its occupation-restricted
    release-only tilt:
    - -0.0072 (0.0064) against the upper bracket IA_OSG_RO_HI, which fails G1 in version 1 (men's occupation Brier guard at 1.005 of
      its cap; the rules gate G1 only for the candidate and the twin);
    - -0.0051 (0.0060) against the cheaper, task-feasible IA_OSG_RO_LO;
    - -0.0057 (0.0066) on the refit cells.
    - None is resolved.

**3. Robustness to classifier refitting (D2R).** Matched reader refits on new household splits (R3, R4) and new reader seeds (k2, k3)
of the same 8,001 people were genuinely effective in all three readings. None is confirmed (descriptive; D2 unchanged).
- **Which reading counts:** under RULES.md 8 the family D2R readings are IA_R1_D2R (IA's strongest feasible nonzero release is IA_OSG)
  and IB_REG_D2R. IA_REG_D2R was computed while IA_SFG was IA's only operation configuration; it is an extra, descriptive reading of
  the registered candidate.
- **The strongest-release choice:** IA_OSG was selected by 0.0001 against se_agg about 0.006, a tie in substance. The rule's tie
  tolerance is 1e-12, so the choice is mechanical and carries the winner's curse.
- **IA_OSG (IA's family D2R)**, refit cells:
  - vs the twin +0.0028 (0.0059);
  - vs its SBC bracket -0.0010 (0.0081);
  - vs its occupation-restricted release-only tilt -0.0057 (0.0066), against IA_OSG_RO_HI, which fails G1 in version 1;
  - vs NEURAL_RAW -0.0175 (0.0105), H_LJ -0.0261 (0.0093).
  - Its SPEND path has no bracket, so the D2R reading holds no SPEND comparator. Descriptively, beside it
    (`claims/IA_R1_SPEND_REFIT.json`): +0.0019 (0.0060) against IA_OSG_SPD_HI and +0.0009 (0.0056) against IA_OSG_SPD_LO on the refit
    cells, ROBUSTNESS_NOT_CONFIRMED.
- **IB_CSF (IB's family D2R)**, refit cells:
  - vs the twin +0.0061 (0.0056);
  - vs its SPEND bracket -0.0006 (0.0069);
  - vs NEURAL_RAW -0.0142 (0.0112), H_LJ -0.0227 (0.0090).
- **IA_SFG (extra reading)**, refit cells:
  - vs the twin +0.0018 (0.0055);
  - vs its SBC bracket +0.0032 (0.0061), SPEND bracket +0.0018 (0.0044), RO bracket +0.0011 (0.0058);
  - vs NEURAL_RAW -0.0185 (0.0102), H_LJ -0.0270 (0.0088).
- **The ablations** (IA_BLIND, IA_OSG_BLIND; IB_CSFT, IB_IND) are not confirmed either.
- **Seed effectiveness.** "Genuinely effective" is the refit-level check. Per comparator, the reader-seed qualifier reads
  READER_SEEDS_PARTLY_EFFECTIVE, with n_eff of 13-19 of the 24 robustness cells for the decisive comparators: seed-invariant recipes on
  few-valued views count once.

**4. Competitiveness against strong references (D3).** Not competitive in any reading.
- **The nine references:**
  - **Less recoverable than every candidate:** six references, RETIRED_B3CTL 0.5964, B2TWIN 0.6061, SUCC 0.6074, H_LJ 0.6186,
    NEURAL_RAW 0.6253 and TWJ 0.6359.
  - **Between the candidates:** W (0.6459) and TT100 (0.6454) are below both IA candidates (0.6464, 0.6463) but above IB_CSF (0.6411).
  - **Above all three:** TR100 (0.6479).
  - **None is passed** in any reading (resolved benefit with the recipient gates).
- **NEURAL_RAW and H_LJ:** all three candidates are more recoverable than both, by 0.016-0.028. This is resolved in five of the six
  comparisons; IB_CSF against NEURAL_RAW (-0.0158, se 0.0095) is not resolved.
- **The previous campaign's strongest release**, DDR_PTC_R1 (pair 0.6239, beside):
  - it is resolved less recoverable than IA_SFG (-0.0225, se 0.0073) and IA_OSG (-0.0224, se 0.0093);
  - against IB_CSF the difference is -0.0172 (0.0090), not resolved (upper 2-SE bound +0.0009);
  - DB_PTC_R1 is the dual-design campaign's adaptive revision, chosen with a winner's curse on the same people and never separated from
    its own release-only control.

**5. The material criterion (D4).** Not met. The criterion is a resolved 0.020 against NEURAL_RAW and H_LJ. The registered-cell
benefits are:

| Candidate | vs NEURAL_RAW | vs H_LJ |
|---|---|---|
| IA_SFG | -0.0211 (0.0085) | -0.0278 (0.0097) |
| IB_CSF | -0.0158 (0.0095) | -0.0225 (0.0098) |
| IA_OSG | -0.0210 (0.0087) | -0.0277 (0.0101) |

The benefits against the twin (+0.0016, +0.0069, +0.0017) are smaller, unresolved results against a weaker baseline.

**6. Joint-design evidence (D5).** None is claimed.
- **IA:** not declared jointly designed. Both recipients see the same generalized record, but no joint claim was registered. IA_OSG
  changes one output only.
- **IB:** its card declared, before its fit, that both sequential orders equal the joint design by construction.
  - A sequential second stage that sees the first stage's release and the shared key reproduces the coordinated collapse bitwise; the
    verifier rebuilt both orders bitwise (561/561 and 605/605 flattened people).
  - So D5 is NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL. No advantage of joint over sequential design can exist here; this is not a
    negative finding about coordination.
- **Coordination contrast:** CSF against independent per-recipient keys (IND) is +0.0016 (se 0.0066), not resolved. Coordination
  effects up to about 0.015 are not excluded.

**7. Novelty (D6).** Not claimed. Known antecedents:
- field generalization and suppression (Samarati and Sweeney 1998; Sweeney 2002; LeFevre et al. 2006);
- the privacy funnel (Makhdoumi et al. 2014);
- local suppression and microaggregation (Willenborg and de Waal 2001; Hundepool et al. 2012);
- coordinated suppression (Cox 1980);
- per-model feature restriction.

No literature search was run in this campaign.

## Ablations and beside comparisons (descriptive, registered cells)

- **IA targeting off** (BLIND: the same menu ordered by task spend only): +0.0008 (0.0060) against IA_SFG. IA_BLIND is bitwise IA_SFG in
  version 1 (the two ladders coincide on their first two steps), so this contrast rests on version 0 (+0.0017).
- **IA_OSG targeting off** (IA_OSG_BLIND): IA_OSG is +0.0042 (0.0063) against it. IA_OSG_BLIND is -0.0025 (0.0070) against the twin
  (at the point its pair is more recoverable than the twin's, while its occupation view drops as much as IA_OSG's).
- **IB targeting on** (CSF-T, rates allocated by a release-level SEX propensity): -0.0002 (0.0079) against IB_CSF.
- **IB coordination off** (IND): +0.0016 (0.0066) against IB_CSF.
- **Ablations against the twin:** IA_BLIND +0.0008 (0.0069), IB_CSFT +0.0071 (0.0057), IB_IND +0.0053 (0.0044).
- **IA_OSG against IA_SFG:** +0.0001 (0.0061). Restricting the generalization to the occupation recipient made no detectable
  difference on the pair.

## What was not run, and why

- **The replacement family** (the challenger's cross-recipient keyed swap, FS_SWAP).
  - Designer B specified it, and it was not run. The decision came at 02:45Z, before any MODEL_SELECT read, on Designer B's label-free
    plug-in: in that world it is no better than occupation-only flattening or CSF at equal spend (`roles/coordinator/FS_SWAP_DECISION.md`).
  - It is a planning decision, not a result, and no D2 look was spent on it.
  - It was the only design aimed at the pair link itself, so the super-additivity failure remains untested here.
  - No other replacement family was identified.
- **The revisions declared in the cards.** None was run (coordinator log, 02:47Z; MODEL_FIT records):
  - IA_SFG_v1 R1 (slack-weighted ladder): its trigger failed in version 0. Two occupation components of the MODEL_FIT budget keep less
    than 50% of their slack at the next rung (occupation accuracy 37%, women's occupation Brier 45%);
  - IA_SFG_v1 R2 (finer splits): its trigger held in both versions (the next rung overshoots its binding component by 95% / 69%). But
    the binding step, marital-status suppression, has no predeclared half beyond MAR2, which the menu already contains, so it could not
    change the release;
  - IB_CSF_v1 R1: its trigger needs an income component to bind while every occupation component keeps at least 40% slack. At
    pi = 0.10 income NLL (1.22 / 1.001 of S3) and the men's occupation guard (1.26 / 1.17) both fail, and in version 0 the women's
    guard too (1.18);
  - IB_CSF_v1 R2: pi* = 0.075 > 0.025.
- **IA's revision that was run.**
  - **Numbering:** IA_OSG fills IA's first revision slot (PROTOCOL 7 allows two), hence the tag ADAPTIVE_R1. It is not the
    slack-weighted ladder that IA_SFG_v1 declared as its R1.
  - **Timing:** it was designed after, and from, the registered MODEL_SELECT reading, so it is adaptive as a whole. Its own card was
    committed before its fit (04:11:44Z).
  - **Diagnosis:** it rests on F1 (MODEL_FIT: income binds the shared ladder at MAR1) and F3 (the twin's pair recovery sits in its
    occupation view). The card's F2 (leakage moved to the income view) is not supported by these records (question 2).
- **No second IA revision.** PROTOCOL 7 allowed one more, and 9 configurations remained (enough for a revision with upper bracket points
  only). The coordinator judged that, within budget S3, the menu offers no revision addressing the observed failures:
  - a shared or income-side ladder is held back by S3's income components at its first informative step (F1: the rung with marital
    status suppressed fails S3 on income accuracy, 1.95 / 1.69 of the budget, and income NLL, 1.07 / 1.29);
  - an occupation-only ladder was IA_OSG, whose occupation reduction the linked pair did not follow;
  - so a pair-aware variant would need the income input coarsened further than S3 allows. That is a different budget, not a revision
    (listed under untested questions).
- **No IB revision.** Its declared triggers did not hold.
  - The coordinator's further reasons are allocation judgments that use MODEL_SELECT outcomes, not evidence that a revision would fail.
  - Those reasons: the targeting and coordination ablations showed no resolved gain, and the constraints binding at pi = 0.10 (income
    NLL and the occupation guards) would bind any deeper collapse.
- **Configurations:** 39 of the 48 were used.

## Favourable and adverse evidence

**Favourable** (none of it is a protection claim):
- **Task feasibility.** Every candidate is task-feasible at the point. IA_SFG and IB_CSF are reserve-qualified in both versions; IB_CSF
  keeps every decision exactly.
- **Consistent direction for IB_CSF.** Its point benefit over the twin keeps its sign under refitting (+0.0069 registered, +0.0061 on
  the refit cells), and its recipient gate against the twin passes. It is never resolved, and IB_CSF's gain over the twin is not
  separated from SEX-blind dithered quantization (+0.0007, se 0.0081, against its SPEND bracket, which spends 0.0037 nats more).
- **IA_OSG's occupation effect was predicted.**
  - The drop in its occupation view (0.6443 to 0.6351) is the size the label-free plug-in and the card predicted (registered: 0.003-0.012).
  - The data-rich LOG_LR occupation reader (the only data-rich recipe on that view) agrees: occupation-view benefit over the twin
    +0.011 / +0.017 in the two versions.
- **The machinery checked out.**
  - The verifier replayed every registered decision exactly, and its addendum replayed IA_R1, the three D2R readings and the
    strongest-release choice exactly (`VERIFICATION.md`).
  - Deployment parity holds bitwise for 10 packages.
  - The matched refits were genuinely effective in all three D2R readings.
  - Every card and bank was committed before the fit or grid it governs.
- **The registered bets.**
  - **What held:** the decision-level predictions held in all three readings (D2 NOT_SEPARATED, D3 NOT_SUPERIOR, D4 NOT_MET), and
    every observed pair benefit fell inside its registered interval.
  - **The misses:**
    - IB's strength: pi* 0.075, against a predicted 0.10-0.15 at probability 0.6. Its predicted binding component (the men's guard)
      bound only jointly with income NLL.
    - IA_SFG's "k* = 1 in both versions" (0.5): version 1 took two steps.
    - IA_SFG's substituted-ladder equality (0.8): it held in version 1 only (deviation 3).
    - IA_SFG's predicted RECIPIENT_GATE_FAILED_VS_RO occurred through the one-repetition income-view sensitivity, not through the
      predicted mechanism.
    - IA_OSG's SPEND comparison ended without a bracket (predicted at 0.05).
  - **Also recorded:** IA_OSG_R1's "k* includes WK3 or a COW step in at least one version" (0.35) occurred (version 1 includes WK3).
  - **The protocol's expectations** (the statistical reviewer's section 4.9):
    - P(CAND G1-feasible) 0.8: all three were feasible;
    - P(D2 met for at least one family) 0.08: met in none;
    - P(D3) 0.03 and P(D4) 0.01: neither met;
    - the most likely outcome (0.005-0.02 over the twin, not separated from a control) fits IB (+0.0069), while IA (+0.0016) fell below
      that range.

**Adverse:**
- **No D2 look met:** 3 looks. Every difference against the twin is point only and at most 0.007 (+0.0016, +0.0069, +0.0017).
  - **Precision:** realized se_agg for the decisive comparisons was 0.0053-0.0081, below the planned 0.009-0.018 because the candidates
    are close to the twin, so the per-comparison 2-SE threshold was about 0.011-0.016.
  - **What is excluded:** at the 2-SE level the data exclude pair benefits over the twin above about 0.013 (IA_SFG), 0.018 (IB_CSF)
    and 0.014 (IA_OSG).
  - **Power:** in a normal approximation, a true benefit of 0.020 against every gating control would have passed all of D2's
    comparisons with probability about 0.83 for IA's four and 0.66 for IB's two (the reviewer's invented-data calculation). Effects
    below about 0.01 were not resolvable.
- **The leakage sits in the occupation view.**
  - The twin's pair recovery is carried by its occupation view, and neither family moves it by more than 0.01 inside the budget.
  - When the revision cut it alone, the linked pair hardly moved (pair +0.0017 against an occupation-view drop of 0.0092). That is
    consistent with recovery through the income vector, as point estimates.
- **IB's gain is not separated from spending.** +0.0007 (0.0081) against SEX-blind quantization (its SPEND bracket, which spends 0.0037
  nats more). The targeting (CSF-T,
  -0.0002) and coordination (IND, +0.0016) contrasts are not resolved.
- **IA's registered candidate barely generalizes.** One marital-status merge in version 0, plus five-year age bands in version 1; a
  near-identity release (MODEL_SELECT spend +0.0004 / +0.0001 nats).
- **IA_OSG is not separated from any of its controls.**
  - At the point it is more recoverable than its occupation-restricted release-only tilt (-0.0072 against a bracket point that fails G1
    in version 1; -0.0051 against the feasible lower point; -0.0057 on the refit cells; none resolved).
  - It outspends both occupation-only quantization points on MODEL_SELECT, which are at the point at least as hard to read.
  - Its costs grew from MODEL_FIT to MODEL_SELECT (version 1: occupation NLL 0.0070 to 0.0120 nats; men's occupation Brier 0.0026 to
    0.0049). The men's occupation Brier guard binds its reserve in version 1 (ratio 1.39).
- **References.**
  - Six references are less recoverable than every candidate.
  - All three candidates are more recoverable than NEURAL_RAW and H_LJ, by 0.016-0.028, resolved in five of the six comparisons (not
    IB_CSF against NEURAL_RAW).
  - The previous campaign's DDR_PTC_R1 is resolved less recoverable than both IA candidates (not resolved against IB_CSF).
- **The challenger's prediction.** The readings are consistent with its prediction before any fit (information removal by merging runs
  at about 0.3-0.7 pair AUC per nat, against about 2.4 for the closed tilts, so 0.020 is out of reach inside the budget). It was a
  qualitative prediction, not a test.
- **Adaptivity.** IA_OSG was chosen after a MODEL_SELECT outcome, and its strongest-release selection is a winner's-curse selection on
  the same cells. Everything here is development evidence on adaptively used records.

## Untested questions

- **A pair-aware or link-breaking design.** IA_OSG's point estimates are consistent with the linked pair recovering an occupation-only
  reduction through the income vector, but no link-breaking operation was tested. Not run:
  - FS_SWAP, the only design aimed at the pair link;
  - coarsening both inputs with recipient-specific ladders priced on the pair;
  - IA_SFG_v1's declared slack-weighted ladder (R1, whose trigger failed in version 0).
- **A wider budget than S3.** The original utility limits leave more occupation headroom than S3. Selections there would sit nearer the
  limits, and the reserve heuristic shows how little slack several releases already have.
- **The substituted release-only path for IA.**
  - **Not built:** the card declared the GENERIC_ZTILT fallback before the fit, because the twin's released pair identifies the
    six-field record for about 93% of MODEL_FIT people.
  - **Premise held in one version only:** the descriptive substituted ladder equals SFG's up to k* in version 1 but starts with a
    different step in version 0 (AGE5 against MAR2) (deviation 3).
  - **Consequence:** IA's RO comparisons answer only the generic-tilt question and are not an information bound.
- **More precision.** More cells on these people would be another adaptive look, and a confirmation population was not authorized.
  Effects below about 0.01 were not resolvable here.
- **The challenger's other idea.** Within-field randomized response on the inputs (input PRAM, challenger finding 20) was not specified
  or run.
- **Other readers.** Readers beyond the declared bank and the three data-rich recipes were not tried.

## Resources, verification, custody and publication

**Resources** (`RESOURCE_ACCOUNTING.json`, `EXPERIMENT_LEDGER.csv`, `CONFIG_LEDGER.csv`; queue closed at 05:53:53Z):
- **Elapsed:** 4.58 h of the 8 research hours (start 01:19:06Z), with closeout after that.
- **CPU:** 4.282 CPU-h of 12 charged.
  - 3.986 CPU-h in 91 queue units: 90 done, and 1 failed (the first reader-custody attempt, deviation 10). By stage: readers 3.35,
    fits 0.55, custody 0.03, verify 0.02, deployment 0.02, claims 0.01 and evaluation 0.005 CPU-h.
  - 0.296 CPU-h in 9 out-of-queue charges by the designers, the challenger, the statistical reviewer, the verifier, the claims reviewer
    and the coordinator, all upper estimates.
- **Paid compute:** $0. At most two heavy jobs at a time, one thread each.
- **Configurations:** 39 of 48.
  - Registered: 23 (the shared twin 2, shared control points 8, IA 7, IB 6).
  - IA's revision: 16.
- **Agents:** two designers, the challenger, the statistical reviewer, the verifier and the claims reviewer, with the coordinator. Never
  more than six were live at once.

**Access** (`ACCESS_RECEIPT.json`):
- **47 label reads,** all after the registration receipt was verified (02:14:15Z):
  - 6 fitting reads on MODEL_FIT under committed cards (IA_SFG_v1, IB_CSF_v1 and IA_OSG_R1, 2 each);
  - 23 MODEL_SELECT utility reads;
  - 13 MODEL_SELECT reader reads (grids 8, reader custody 3, data-rich readers 2);
  - 5 verifier reads.
- **Not read:** CALIBRATION, the ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels, the sealed assessment and the capped-out records.
- **Checks:** no fitting read precedes its card's commit, and the DEV_REPORT lineage log is unchanged.

**Verification** (`VERIFICATION.md`):
- An independent verifier replayed every decision, robustness and strongest-release record and every grid bank exactly, with its own
  code, in two reports (checks 1.1-5.4 and A1-A8). Every check passes.
- Two closing reviews were applied: the statistical reviewer's (headline and corrections C1-C19) and the claims review (R1-R23).

**Custody** (`custody/BACKUP_VERIFICATION.json`):
- **Same-device backup:** `pcrl_input_and_pair_release_v1_20261022T055206Z` was verified from the backup alone at 05:53:22Z (all ok):
  - 14 packaged models re-run bitwise;
  - every package re-deploys with all refusals;
  - 5 custody readers reproduce their scores bitwise;
  - 39 releases, 201 admitted copies and 2,368 reader-cache entries re-hash.
- **Its scope** is narrower and compressed for disk space (deviation 11).
- **A first attempt failed** on packages pinned to pre-revision code. They were rebuilt, and the first attempt is recorded (deviation 12).
- **Off-device custody is pending.** The owner's drive was not attached at 04:37:17Z or 05:53:32Z. When it is, run
  `python3 -m ipr.custody offdevice <worktrees>/pcrl-input-and-pair-release-v1-backup/pcrl_input_and_pair_release_v1_20261022T055206Z <external drive>`.

**Publication:**
- Sanitized code and aggregate records are on the branch `publication/pcrl-input-and-pair-release-v1`, as a child of the registration
  receipt e909fb8c0, under `publications/pcrl-input-and-pair-release-v1/`.
- The commit SHA and its remote verification are in `PUBLICATION_RECEIPT.json`, written in the research copy after publishing.
- The research branch `research/pcrl-input-and-pair-release-v1` stays local. Earlier branches, manuscripts and results are untouched.
