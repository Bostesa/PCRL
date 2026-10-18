# Errata (append-only; times UTC)

Record errors found after a record was written. The original record is never edited; each entry states what is wrong, what is
right, the evidence, and the effect on results.

## E1. P70A witness record: fold tags (verifier C2.1 FAIL, reported 20:35Z; written 20:37Z, committed with this file)
- **Record.** `runs/TRACKF/P70A/WITNESS/WITNESS.json` (unit CTS_P70A_WITNESS_V1, written 20:21:08Z) states
  `outer_fold_tag` = `CTS1_cf_P70A` and `inner_fold_tag` = `CTS1_inner_P70A`.
- **Correct values.** `CTS1_cf_P70` and `CTS1_inner_P70`.
- **Evidence.**
  - `cts.trackf.fit_witness` deals both fold sets with the tags of the `same_rows_as` partition (`base`, which is P70 for P70A;
    commit 5002fed58, amendment 9). This is what the amendment registers ("the same rows / folds").
  - The function writes the two record fields from the partition id instead (`f'CTS1_cf_{P}'`), so they name tags that were never
    used.
  - The verifier found the saved P70A outer fold array identical to P70's (C2.1 V5, `same_outer_folds_as_twin` True).
  - The verifier also found that the recorded tag does not regenerate the saved folds.
- **Effect on results.** None. The P70A fit, its MODEL_SELECT evaluation and the paired P70A-P70 differences used P70's folds, as
  registered; only the two label fields are wrong. P70 records its own tags correctly. D0B uses a different witness function with
  explicit tags (`CTS1_cf_D0`) and is not affected.
- **What is not changed.**
  - WITNESS.json is not edited: its hash belongs to the verified artifact set.
  - The code line is left unchanged; no further same-rows witness fit is planned, and the verifier's C2.1 FAIL stays in its
    history as a record defect.

## E2. Findings note F11: Track R training share (claims review A56; recorded 21:33Z)
- **Record.** `roles/coordinator/FINDINGS_NOTES.md` F11 says the frozen H3 package's income margin "does not survive a refit of its
  recipe on 75% of MODEL_FIT".
- **Correct value.** The R partitions' FIT sets are 75% of the non-panel pool: 10,523 and 10,506 people, about 53% of MODEL_FIT.
  CALIBRATION is added for the income lineage. The GUARD and CHECK households are excluded.
- **Effect on results.** None on any number. The income values in F11 are correct. The wording now says "on about half of
  MODEL_FIT": one refit per partition at about half H3's training size, so training size and recipe are not separated.
- **What is not changed.** F11 is not edited; this entry corrects it.
