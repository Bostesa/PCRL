# Verification (campaign `pcrl_input_and_pair_release_v1`)

An independent verifier agent checked the campaign's decisive numbers twice with its own code. For recomputation it imported none of
`ipr.evaluate`, `ipr.devreaders`, `ipr.rules`, `ipr.claims` and `nrm.rules`.
- **The registered readings** (IA_REG, IB_REG), at 02:51-03:23Z: `roles/verifier/VERIFIER_REPORT.md`, checks 1.1-5.4.
- **The post-registration readings**, at 04:53-05:08Z: `roles/verifier/VERIFIER_ADDENDUM_REPORT.md`, checks A1-A8. These cover:
  - the three D2R readings;
  - IA's revision IA_R1;
  - the strongest-release choice;
  - the campaign summary.

Both reports are saved verbatim: the verifier was instructed to return its reports rather than write them.
- **Code and records:** its code is in `roles/verifier/code/` and its records in `roles/verifier/records/`.
- **Labels:** every label read ran inside a queue unit (operation `verify`; 5 reads in all).
- **CPU:** about 49 CPU-s for the first report and 53 CPU-s for the addendum (81.8 CPU-s of it in queue units).

## Result

- **Every check passes**, in both reports, and every recorded label reproduces exactly.
- **Replays** (differences 0, except utility values at most 3.5e-16 and reserve ratios at most 3e-14):
  - release, model and card hashes;
  - the deployment function's recomputation of the MODEL_SELECT arrays of all 39 configurations, bitwise;
  - the income output of every IA_OSG* configuration, equal to the twin's in every role;
  - 616 + 308 utility values and their SEs;
  - 190 paired-cost records;
  - winner checks, SCORE AUCs (each with its 400 bootstrap draws) and NOLOOKUP AUCs: 312 cells in the registered bank and 2,568 in the
    later banks;
  - 336 + 2,544 paired cells and SEs;
  - every D label, the D2R per-comparator statistics and the strongest-release values.
- **The one approximation.** The verifier's own LR and 15-leaf HGB readers (first report, check 3.5) are an independent approximation,
  not a replay. They are consistent with the grid within its uncertainty (the verifier's verdict, check 3.5).
  - The LR and validation-chosen readers agree in sign with the grid for all 11 comparisons.
  - The HGB reader has the opposite sign for three point-only comparisons (IA vs TWIN, IA vs IA_SPD_HI, IB vs IB_SPD_HI), each within
    about 1.2 se_agg.
- **Declarations checked:**
  - IB's `operation_sex_blind`: both releases rebuilt bitwise from SEX-free inputs;
  - IB's D5 collapse: both sequential orders rebuilt bitwise, 561/561 and 605/605 flattened people;
  - IA's declared RO fallback;
  - IA_OSG's payloads and its occupation-only controls.

## Observations the verifier raised, and what was done

**First report:**
1. **O1, IA_SBC_LO outside the SBC path** (saved in version 0 only): deviation 2. The verifier's own numbers show it could only be a
   cheaper point, never Q*. The statistical reviewer's check shows that a one-version point makes D2 INCOMPLETE under the frozen rules,
   so leaving it out was the only way to obtain a reading. No label depends on it.
2. **O2, IA_BLIND version 1 is bitwise IA_SFG version 1** (the two ladders share their first two steps). Stated beside the ablation in
   `RESEARCH_DECISION.md`: that contrast rests on version 0.
3. **O3, card wording.** IB_CSF_v1 section 1 says non-flattened people get the twin's vectors "bitwise". That holds for income;
   occupation vectors differ by up to 2.2e-16 (clip and renormalize), with identical decisions. Wording only; recorded here.
4. **O4, IA's income-gate failure rests on one repetition** (S1R1, a reader-selection sensitivity). It changes no label. It does
   qualify the income-view interpretation: the claim that leakage moved to the income view is not supported (`RESEARCH_DECISION.md`,
   question 2; the statistical reviewer's C2).
5. **O5, three label-free verifier scripts ran outside the queue** (0.034 CPU-s): deviation 7, charged at closeout.

**Addendum:**
1. **O1, IA_OSG_RO_HI fails G1 in version 1** (men's occupation guard at 1.0046 of the cap), and IA_OSG_SBC_HI misses the reserve.
   Descriptive: the rules gate G1 only for the candidate and the twin. Stated beside the RO comparison wherever it is reported.
2. **O2, the occupation-only SPEND path does not bracket on MODEL_SELECT** (NO_BRACKET_SPEND): a limit of the path, not evidence of
   equality.
3. **O3, the revision's SBC path is coarse:** the COARSE flags are reported, never gating.
4. **O4, the code drift is disclosed:** deviation 6. The only within-unit drift is the verifier's own files. The `ipr/` change between
   units was the revision commit 221a91f21.
5. **O5, the superseded early D2R records were not checked** (deviation 9). Only the reruns, which are the readings in use, were
   checked.

## Not checked by the verifier

- **The MODEL_FIT computations themselves.** The verifier re-derived the ladders, rungs, selections and brackets from the recorded
  statistics, but did not recompute the fits.
- **Records it did not replay:**
  - the grid entries of the seven references other than NEURAL_RAW and H_LJ in D3 (arithmetic on the recorded entries only);
  - the enhanced (TABLE_CV) winners behind the reported-only READER_SENSITIVE_ENHANCED flags;
  - the reader-budget flags.
- **Descriptive records:** the beside records, the data-rich readers and the custody records (reader custody and the backup).
- **Deployment packages:** these were checked by the campaign, not by the verifier:
  - `deploy/DEPLOYMENT_CHECK_V2.json` checks all 14 packages against the final code;
  - the earlier checks are `deploy/DEPLOYMENT_CHECK.json` (the first 10, rebuilt after the revision changed their code; deviation 12)
    and `deploy/DEPLOYMENT_CHECK_R1.json`;
  - the restore check run from the backup alone (`custody/BACKUP_VERIFICATION.json`, all ok) re-deployed all 14 bitwise.

  The verifier recomputed the deployment function's arrays for all 39 configurations, which is not a package check.
