# Handoff to the manuscript owner and the claims owner

Campaign `pcrl_dual_design_release_v1` (2026-10-21). This file is a repository handoff; no message was sent to anyone.

## What can be said

Use the statistical reviewer's required headline in `RESEARCH_DECISION.md` exactly. In short:

- **The design.** Two independently designed release families were each compared against an equally trained task-only twin, two
  matched control paths, component ablations and the nine inherited references, under rules frozen before any label was read. The
  families were SEX-standardized task posteriors (STD) and propensity-tilt cancellation (PTC).
- **One idea family.** STD and PTC each subtract the propensity-proportional part of the released task posterior, and the revisions
  stay inside it. This joint negative closes one idea, not two.
- **Utility.** Every release met all 28 original utility conditions at the point (12 clauses and 2 guards in each of its two
  version-specific configurations; all 48 configurations pass). Point feasibility is not confidence preservation.
- **D2 not met.** Neither family met the registered criterion for reducing SEX recovery beyond matched controls: 0 of 2 registered
  and 0 of 2 adaptive readings, 4 D2 looks in total. One AGG_POSITIVE comparison passes a true null in about 0.07-2% of replicates
  under six moderate NRM1 known laws, and about 8% when one version or split dominates. With 4 looks, the chance that at least one
  passes by chance is at most 4 times the per-look rate.
- **References.** Neither family is competitive with the strongest references, and neither meets the 0.020 material criterion.
- **DB's adaptive revision, DB_PTC_R1** (recipient-split strengths). It was selected afterwards, on the same registered cells, as DB's
  strongest feasible release (winner's curse; ADAPTIVE_R1; never DB's result).
  - Registered cells: its linked-pair AUC is below the twin's by 0.0239 (se 0.0101; lower bound +0.0037). It passes the SPEND
    frontier comparison with a lower bound of +0.0001. It is not separated from its SEX-blind-target control (+0.0265, se 0.0139).
  - Refit cells: on new household splits and reader seeds of the same people, its benefits over the twin and both bracketing controls
    stay resolved. This is descriptive: D2 is unchanged, and it is not a replication.
  - Release-only control: in the same breath, by the reading declared before that control's fit, it is not separated from
    DB_ZTILT_R1, a tilt whose amplitude is estimated from the twin's released vectors alone (+0.0128, se 0.0081; refit cells +0.0066,
    se 0.0095). So its gain is not separated from release-only information. That control overspent (0.0169 vs 0.0099 nats), which
    is disclosed.

## What must not be said

**Overall result:**
- Not "a method that reduces SEX inference beyond matched controls", and not "a component" or "a candidate". D2 was not met in any
  reading, and no partial outcome is a claim.
- Not "information removal" for any release. None is a post-processing of the twin's release, and PTC writes the propensity into the
  released vector.
- Not "protection" or "privacy" without qualification. The results concern the declared readers and three data-rich recipes only.

**DB_PTC_R1:**
- Not that it "needs (or uses) propensity information beyond the release": it is not separated from DB_ZTILT_R1. Equally, not that
  its gain is "fully explained by" or "reproduced by" release-only information: the pair comparison is unresolved, with a positive
  point estimate in both versions, and the control overspent.
- Not that it separates from, beats or dominates its SBC control. Not that it "meets D2 on the refit cells" or "after refitting", and
  not that the refit "resolves" the SBC miss. D2R never changes D2.
- Not that the SBC miss "is only a power problem" or "would separate with more data". A non-separation is evidence neither of equality
  nor of a hidden effect.
- Not "replicated", "confirmed" in the confirmation sense, or "independent evidence". Not "robust" for DB_PTC_R1 in any form other
  than the D2R label (robustness of its recovery comparisons to classifier refitting, descriptive): the release carries
  POINT_FEASIBLE_NOT_RESERVE_QUALIFIED, which RULES.md 6 says is never called robust. The refit cells are new
  splits and seeds of the same people, the release was selected on the registered cells, and it is not reserve-qualified in version 1.
- Not "stable across reader seeds". In one block (version 0, split R1) its benefit over the twin is +0.034 with seeds k0 / k1 and
  +0.003 with k2 / k3.
- Not "DB's result" or "the family's result". The registered DB reading is DB_PTC, NOT_SEPARATED.

**References, criterion, design and novelty:**
- Not "competitive with" or "better than" the inherited references. Four references have lower pair AUC (0.600-0.622).
- Not "meets the material criterion", and not "a 0.02 reduction". The differences against the twin are +0.007 to +0.024, resolved
  only for the adaptive DB_PTC_R1. D4 compares with NEURAL_RAW and H_LJ, where the best is +0.011 / -0.006.
- Not "a joint design" (D5 not assessed) and not "novel" (D6 not claimed). Both devices have published antecedents.

**DA:**
- Not "IPW-STD fixes standardization". The MODEL_FIT diagnostic gain did not carry over to MODEL_SELECT (+0.0072 vs the twin;
  -0.0103 against DA_STD, not resolved).
- Not "fixed SEX weights are essential". The DA_COMP ablation is descriptive and not resolved against the twin.

## Records to cite

- **Decisions.** `claims/{DA_REG,DB_REG,DB_R1,DA_R1}/DECISIONS.json` and `claims/CAMPAIGN_SUMMARY.json`. Rules:
  `roles/statistical_reviewer/RULES.md`, frozen before any outcome. Closing statement: `roles/statistical_reviewer/CLOSING_STATEMENT.md`.
- **Robustness and ablations.** `claims/*_D2R/D2R.json` and `claims/STRONGEST.json`. The family D2Rs are DA_REG_D2R and DB_R1_D2R.
- **Beside comparisons.** `claims/BESIDE_*.json` and `claims/ZTILT_R1_READING.json`.
- **Numbers.** `TABLES.md` / `TABLES.json`, generated by `ddr.report` from the records.
- **Verification.** `VERIFICATION.md` and `roles/verifier/` (both reports).
- **Claims review.** `roles/engineering_review/REVIEW_CLAIMS.md`.
- **Cards and deviations.** `cards/` and `PROTOCOL_DEVIATIONS.md` (deviations 1-13).

## Open items

- **Off-device custody is pending.** The owner's drive (YOTUO) was not attached at 18:44Z or 19:47Z. Same-device custody was verified
  from the backup alone (`custody/BACKUP_VERIFICATION.json`). When the drive is attached, run
  `python3 -m ddr.custody offdevice <worktrees>/pcrl-dual-design-release-v1-backup/pcrl_dual_design_release_v1_20261021T194459Z <external drive>`.
  A check right after copying is warm; a cold re-check needs an ordinary eject and re-attach first. Nothing is mounted, unmounted or
  forced by the tool.
- **Untested** (see `RESEARCH_DECISION.md`):
  - confirmation on a new population (not authorized here), with a release-only ZTILT point spend-matched along DB_PTC_R1's split
    law;
  - joint-design controls;
  - families outside the propensity-subtraction idea;
  - readers beyond the declared bank and the three data-rich recipes.
- **No configuration budget remained** (48 of 48). DB's declared R2 was not triggered on either reading. DA's declared R2 was not
  triggered either: lambda* = 1.0 in the registered selection, and in R1 the binding condition was the women's occupation Brier
  budget clause (MODEL_FIT), not an accuracy clause.
