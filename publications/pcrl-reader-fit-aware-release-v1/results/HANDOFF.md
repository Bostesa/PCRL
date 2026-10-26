# Handoff (campaign `pcrl_reader_fit_aware_release_v1`, RFA1)

Coordinator, 2026-10-25. The resource, access and custody figures come from the queue ledger and the custody records
(`RESOURCE_ACCOUNTING.json` and `ACCESS_RECEIPT.json`, written at 23:34Z after the queue closed; `custody/`). Work after the queue
closed (verifier checks, publication) is charged to the ledger and summarized in `PUBLICATION_RECEIPT.json` and the coordinator log.

## State

- **Branch.** Research branch `research/pcrl-reader-fit-aware-release-v1` (local; base 913eb5c91 = the DRF1 research tip). Worktree
  `.claude/worktrees/pcrl-reader-fit-aware-release-v1`.
- **Registration.** Commit 43830ae19. The sanitized receipt 34dc42b51 on `publication/pcrl-reader-fit-aware-release-v1` was pushed
  and verified at 21:53:27Z, before any label read. The final publication is a child commit of the receipt (`PUBLICATION_RECEIPT.json`).
- **Readings.**
  - RFA_v1 (registered): `claims/RFA_v1_final/` governs. The earlier records `claims/RFA_v1/` and `claims/RFA_v1_rebuilt/` are kept.
  - RFA_R1 (ADAPTIVE_R1): `claims/RFA_R1_final/` governs. The earlier record `claims/RFA_R1/` is kept.
  - Second splits: `claims/RFA_v1_D2R/` and `claims/RFA_R1_D2R/`, both ROBUSTNESS_NOT_CONFIRMED.
  - The final records attach the second splits, with labels unchanged (deviation 10).
- **Budget.** All 24 configurations are used: 12 v1 (RF_*) and 12 R1 (RF1_*), under `.private/pcrl_reader_fit_aware_release_v1/runs/FAM/`.
  Any continuation needs a new campaign.
- **Deployment packages.** Six packages under `.private/pcrl_reader_fit_aware_release_v1/deploy/` (RF_FULL, RF_STOP_HI and RF1_FULL,
  versions 0 and 1). Report: `deploy/DEPLOYMENT_CHECK_MAIN.json`.

## What a successor needs to know

- **No evidence that the fit-aware derivative helps.** It is implemented and verified (`rfa/fam/rfit.py`, `rfa/fam/rfit_r1.py`;
  finite differences to about 1e-9, checked independently by the verifier). F0 found it active on MODEL_FIT (orthogonal share
  0.10-0.11). In neither reading was the candidate separated from its stopped-fit control or its lagging critic:
  - v1 development bounds: about 0.012 for STOPFIT and 0.0025 for ALTCRIT;
  - R1 upper bounds: +0.0068 and +0.0060.

  The registered benefit over NEURAL_RAW was not confirmed on the second split. The one resolved difference is descriptive and post
  hoc: at equal step count the controls spent 0.0002-0.0006 nats more task NLL. Resuming the derivative as a protection lever on this
  start is not supported by this evidence.
- **The binding facts.**
  - On MODEL_FIT, k\* was stopped by the women's occupation Brier and the occupation NLL components of BUDGET_OCC, within 20-70
    steps.
  - On MODEL_SELECT, the tightest condition at the selected points is the version-1 men's occupation guard, at 0.885 / 0.915 of its
    cap.
  - The occupation output is about 0.03 AUC more recoverable than the tree-based references' occupation output.
- **Known record defects and deviations** (PROTOCOL_DEVIATIONS.md 1-13). None changes a label:
  1. the fit-record inner-fit accounting (the ALTCRIT count);
  2. reader refusals abort a unit (ReaderRefused not caught);
  3. the fold records' log-label cadence;
  4. coordinator time stamps written ahead of the clock, corrected;
  5. the interrupted card draft;
  6. the first v1 decision record, rebuilt;
  7. the income wire's 6.8e-8 non-identity with the frozen START wire;
  8. a cosmetic episode count;
  9. the family-change field;
  10. the final records carrying the second splits;
  11. the unprefixed R1 second-split label and the per-recipe falls computed from rounded means;
  12. the SE range 3.1-6.1, the CSV line endings in the publication, a custody record annotated after its unit, and the backup gap;
  13. one reviewed leak-scan hit: a bare temporary-path prefix in the verifier's own scan code.

## Resources (at the queue close, 23:34Z)

- **Time.** Started 20:58:41Z; 2.59 h elapsed at closeout (limit 6 h).
- **CPU.** 2.50 CPU-h charged (limit 8), including failures and 477 CPU-s of agent and coordinator work outside the queue. By stage:
  - readers 6,967 s;
  - fit 1,083 s;
  - custody 234 s;
  - verify 112 s;
  - deployment check 104 s;
  - evaluate 9 s;
  - claims 2 s.
- **Jobs.** At most two heavy jobs, one numerical thread each; no paid or cloud compute. 46 units done and 2 failed:
  - RFA_VER_USEFULNESS_v1, which stopped on the verifier's F1 check, by design (deviation 7);
  - RFA_VER_W4_R1ROB_v1, a verifier wave-4 unit (see the verifier's hand-back 4).

  No unit was left without an end, and no campaign process is left running.
- **Configurations.** 24 of 24: 4 candidates, 4 task twins and 16 controls.
- **After the close.** Two late charges went to the ledger at about 00:40Z, both label-free and outside the queue:
  - the verifier's waves 4-5, 18 CPU-s;
  - the coordinator's closeout and publication work, 45 CPU-s.

  The ledger total is then 9,052 CPU-s (2.514 CPU-h) at 3.68 h elapsed. RESOURCE_ACCOUNTING.json stays as written at the close.

## Data access (ACCESS_RECEIPT.json)

- **Label reads.** 40 in total, all inside queue units and after the registration receipt (verified 21:53:27Z):
  - MODEL_FIT, by the two fit entry points (2 each), under committed cards;
  - MODEL_SELECT, by the benchmark (13), the readers (16) and the verifier (7).
- **Never read.** No fitting entry point read MODEL_SELECT. No CALIBRATION, ATTACK or DEV_REPORT label was read, and neither were the
  sealed assessment or the capped-out records. The lineage DEV_REPORT log is unchanged since the campaign-start snapshot.

## Custody

- **Final backup.** `<worktrees>/pcrl-reader-fit-aware-release-v1-backup/pcrl_reader_fit_aware_release_v1_20261025T232718Z`, at tip
  42ef29832, 846,815,076 bytes. Its restore from the backup alone passed every check in both interpreters
  (`custody/BACKUP_VERIFICATION.json`).
- **Off-device copy.** On the verified drive YOTUO (volume UUID 3D74932B-...), at
  `PCRL_private_backups/pcrl_reader_fit_aware_release_v1/`. The manifest equals the source's, and the restore from the drive copy
  passed. This check is **WARM** only (`custody/OFFDEVICE_COPY.json`).
  - The drive left /Volumes at about 23:28Z, with no write in progress, and macOS remounted it by itself by 23:30:36Z. Nothing was
    mounted, unmounted or forced.
  - The 24 existing campaign folders on the drive were not touched.
- **Gap.** The commits after the backup tip are in git and in the publication, but not in the backup. They hold the closeout receipts,
  the custody records, the verifier's wave-4 and later records, the corrected per-recipe record and the publication receipt. The
  private-tree changes made after 23:27:18Z are also not in the backup:
  - the verifier's two wave-4 queue units, with their label-access entries and ledger lines;
  - the off-device unit's folder;
  - the queue close.

  The drive was absent again when the verifier looked at 00:25Z and 00:33Z. The copy had been made and checked at 23:31-23:33Z. The
  copy's recorded size, 846,808,188 bytes, is the backup's less its RESTORE_CHECK.json (6,888 bytes). The copy carries its own restore
  check record instead, written on the drive and not counted in that size.

## Owner actions

- A COLD re-check of the off-device copy after an ordinary eject and reconnect (the copy is checked WARM only; see `custody/`).
- The verifier's and reviewers' records are in `roles/`. No further owner decision is pending for this campaign.
