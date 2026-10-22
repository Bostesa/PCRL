# Handoff to the manuscript owner and the claims owner

Campaign `pcrl_input_and_pair_release_v1` (2026-10-22). This file is a repository handoff; no message was sent to anyone.

## What can be said

Use the statistical reviewer's required headline in `RESEARCH_DECISION.md` exactly. In short:

- **The design.** Two independently designed release families were each compared, under rules frozen before any label was read,
  against an equally trained task-only twin, matched control paths, component ablations and the nine inherited references:
  - input-side shared SEX-aware field generalization (SFG);
  - a SEX-blind coordinated keyed collapse of both outputs to decision-cell means (CSF).
- **Utility.** Every candidate met all 28 original utility conditions at the point. Point feasibility is not confidence preservation.
- **D2 not met.** No reading reduced SEX recovery beyond its matched controls: 0 of 2 registered and 0 of 1 adaptive readings, 3 D2
  looks in total; one AGG_POSITIVE comparison passes a true null in about 0.07-2% of replicates under six moderate NRM1 known laws and
  about 8% when one version or split dominates, and with 3 looks the chance that at least one passes by chance is at most 3 times the
  per-look rate.
- **Where the leakage sits.** The twin's pair recovery is carried by its occupation view, and both families move that view by less
  than 0.01 inside the utility budget.
- **The pair did not follow the occupation view.** IA's adaptive revision lowered the occupation view alone at the point (0.6351
  against 0.6443), but the linked pair hardly moved (+0.0017, se 0.0062, against the twin). As point estimates this is consistent with
  recovery through the unchanged income vector; it is not a resolved effect.
- **References and criterion.** Neither family is competitive with the strongest references, and none meets the 0.020 material
  criterion.

## What must not be said

**Overall result:**
- Not "a method that reduces SEX inference beyond matched controls", and not "a component" or "a candidate". D2 was not met in any
  reading.
- Not "reduces SEX inference", "lowers recovery" or "protects" for any release. Every difference against the twin is point only
  (+0.0016, +0.0069, +0.0017; none resolved).
- Not "protection" or "privacy" without qualification. The results concern the declared readers and three data-rich recipes only.

**IB (CSF):**
- Not "SEX-targeted" or "uses SEX information". The operation is SEX-blind by construction.
- Not that its gain "beats spending", and equally not that it "is spending" or "equals spending". It is not separated from SEX-blind
  dithered quantization (+0.0007, se 0.0081, against its SPEND bracket, which spends 0.0037 nats more).
- Not that "coordination helps", and not that it "buys nothing". The coordination contrast is +0.0016 (0.0066), not resolved;
  coordination effects up to about 0.015 are not excluded.
- Not "a joint-design advantage". The card declares IB jointly designed, but D5 is NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL: both
  sequential orders equal the coordinated design by construction, so no advantage over sequential design can exist.

**IA (SFG) and its revision (OSG):**
- Not "input-side generalization removes SEX information from the release":
  - the registered candidate is one marital-status merge (plus five-year age bands in version 1), a near-identity release;
  - its RO comparison is against the generic release-only tilt (the declared fallback), not an information bound.
- Not "leakage moved to the income view" or "the refitted income model leaks more".
  - IA_SFG's higher income-view AUC (0.5600 vs 0.5498) rests on one repetition in which the validation-selected reader switched.
  - The verifier's fixed LR and HGB readers in that repetition, the data-rich readers and the MODEL_FIT readers show no comparable
    difference; the verifier's own validation-chosen reader reproduces the switch (+0.034, verifier O4).
  - IA_OSG's card failure F2 is therefore not supported; its diagnosis rests on F1 and F3.
- Not "IA_OSG is IA's result" or "IA's strongest release reduces recovery":
  - IA_OSG is an adaptive revision;
  - its selection as IA's strongest release beat IA_SFG by 0.0001 on the same cells (winner's curse);
  - it is not reserve-qualified in version 1;
  - it is not separated from any of its controls, and at the point it is more recoverable than its occupation-restricted release-only
    tilt: -0.0072 (se 0.0064) against a bracket point that fails G1 in version 1, -0.0051 against the feasible lower point; none
    resolved.
- Not that occupation-only generalization "reduces pair leakage". The occupation view fell, but the pair did not.
- Not that the linked pair "recovers through the income vector" as a finding. That is a reading of point estimates, consistent with
  FAILURE_MAP row 8 and failure note 2, not a resolved effect.

**References, criterion, design and novelty:**
- Not "competitive with" or "better than" the inherited references.
- Not "meets the material criterion" and not "a 0.02 reduction". The best results against NEURAL_RAW and H_LJ are -0.0158 and
  -0.0225 (IB_CSF).
- Not "the data show no effect". At 2 SE they exclude pair benefits over the twin above about 0.013 (IA_SFG), 0.018 (IB_CSF) and 0.014
  (IA_OSG); effects below about 0.01 were not resolvable.
- Not "novel" (D6 not claimed).

## Records to cite

- **Decisions:** `claims/{IA_REG,IB_REG,IA_R1}/DECISIONS.json` and `claims/CAMPAIGN_SUMMARY.json`.
- **Rules:** `roles/statistical_reviewer/RULES.md`, frozen before any outcome. Closing statement:
  `roles/statistical_reviewer/CLOSING_STATEMENT.md`.
- **Robustness:** all ROBUSTNESS_NOT_CONFIRMED.
  - The family D2R readings (RULES.md 8) are `claims/IA_R1_D2R/D2R.json` (IA's strongest release IA_OSG) and
    `claims/IB_REG_D2R/D2R.json`.
  - `claims/IA_REG_D2R/D2R.json` is an extra reading of the registered candidate.
  - See also `claims/STRONGEST.json`.
- **Beside comparisons:** `claims/BESIDE_IPR1_MAIN.json`, `claims/BESIDE_IPR1_R1.json`.
- **Numbers:** `TABLES.md` / `TABLES.json`, generated by `ipr.report` from the records.
- **Verification:** `VERIFICATION.md` and `roles/verifier/`. **Claims review:** `roles/engineering_review/REVIEW_CLAIMS.md`.
- **Cards and deviations:** `cards/` and `PROTOCOL_DEVIATIONS.md`.

## Open items

- **Off-device custody is pending.** The owner's drive (YOTUO) was not attached at 04:37:17Z or 05:53:32Z.
  - Same-device custody was verified from the backup alone (`custody/BACKUP_VERIFICATION.json`; narrower, compressed scope for disk
    space, deviation 11).
  - When the drive is attached, run
    `python3 -m ipr.custody offdevice <worktrees>/pcrl-input-and-pair-release-v1-backup/pcrl_input_and_pair_release_v1_20261022T055206Z <external drive>`.
  - Nothing is mounted, unmounted or forced by the tool.
  - The occupation-headroom runs excluded from this backup are in that campaign's own backup.
- **Disk space.** The volume was 99-100% full at closeout (about 3.4 GiB free after the backup).
- **Untested** (see `RESEARCH_DECISION.md`):
  - a pair-aware or link-breaking design (FS_SWAP, the only one aimed at the pair link, was not run);
  - a budget wider than S3;
  - IA's substituted release-only path;
  - more precision on new people (a confirmation population was not authorized);
  - the challenger's input-PRAM idea;
  - readers beyond the declared bank.
- **No configuration slot was needed beyond 39 of 48.** No second IA revision and no IB revision were run (reasons in
  `RESEARCH_DECISION.md`).
