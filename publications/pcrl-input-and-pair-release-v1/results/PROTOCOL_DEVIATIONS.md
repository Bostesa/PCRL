# Protocol deviations and disclosures (campaign `pcrl_input_and_pair_release_v1`)

Coordinator. Each item is dated from a `date -u` read at or before the event. None changes a registered rule or a registration file.

1. **A module briefly present in the package during the fit units (02:41-02:43Z).** `ipr/roles.py` (the role members of the shared
   path points) was written into `ipr/` while the IA / IB fit units ran and moved out about two minutes later; no unit imported it, and
   the units' end-of-run code digests equal their start digests (`code_changed_during_run` empty for every fit unit). Its final
   version was restored after the fits (commit 1a55d1c91), before any MODEL_SELECT read.
2. **A control saved in one version only.** IA_SBC_LO exists in version 0 only: in version 1 the permuted-SEX path's first rung
   already spends more than CAND + 2 SE, so its lower bracket point is TWIN. By the plan writer's both-versions membership rule
   (inherited from the dual-design plan writer) it is not a member of the IA reading's SBC path or of the banks; the saved
   configuration counts toward the 48.
   - **Why it had to be left out** (statistical reviewer, C8): the frozen rules pool each path point over both versions, and a point
     recorded in one version only makes D2 INCOMPLETE (the reviewer's invented-data check). So leaving IA_SBC_LO out of the path was
     the only way to obtain a reading.
   - **What it costs:** it removes only the cheaper-point check (c3) in version 0. SBC_FRONTIER fails at its bracket with or without
     it, and the verifier's numbers show it would not be resolved less recoverable (verifier O1).
   - **The breach:** as a saved configuration without utility evaluation or reader cells, it breaches PROTOCOL 4's equal-evaluation
     clause. No label depends on it.
3. **IA's RO path is the declared GENERIC_ZTILT fallback**, not the registered default SUBSTITUTED path (PROTOCOL_AMENDMENT_1 item 5;
   card IA_SFG_v1 section 7; allowed by RULES.md 3.2). The substituted ladder was computed descriptively in the fit unit: it equals
   SFG's rung k* in version 1 but not in version 0 (`ia/FIT_V0.json` `substituted_ladder`), which the card's prediction (0.8 for equality
   up to k*) did not anticipate. The D2 scope sentence is the fallback's.
4. **Robustness lanes withdrawn before start (02:54-02:55Z).** IPR_GRID_ROB_V0 / V1 were queued while utility-evaluation units were
   still waiting; their FIFO tickets would have delayed the evaluations by about an hour, so both lanes were stopped before either unit
   started (no `unit_start`, no CPU; their tickets were removed by the queue's dead-process rule; the unit folders keep only
   CODE_AT_START.json). They are re-queued later as IPR_GRID_ROB2_V0 / V1 with the same bank IPR1_ROB.
5. **Cosmetic print in the fit units.** The fit commands' final one-line summary prints `"sbc_target": null`; the fit records carry
   the SBC construction correctly under `sbc_construction` (PERMUTED in both IA versions).
6. **Verifier code written while campaign units ran or waited (02:58-03:20Z).** `roles/verifier/code/` is part of the queue's hashed
   code manifest (so that the verifier's own label-reading units are pinned). The verifier wrote its six files from 02:58:00Z to
   03:07:08Z and `v6_seeds.py` at 03:20:24Z. So the queue records:
   - IPR_GRID_MAIN_V0 / V1 (02:50-03:11Z): `code_changed_during_run` = the six files;
   - IPR_EVAL_IA_V1 (queued 02:50:34Z, launched 03:09:54Z): a launch-time drift record (`CODE_AT_LAUNCH.json`) naming the same six files;
   - IPR_DEPLOY_STORE_CTL_TWIN_V0_V1 (queued 03:14Z, launched 04:04:46Z): a launch-time drift record naming `v6_seeds.py` (found by the
     verifier's addendum, check A8);
   - IPR_GRID_ROB2_V0 / V1 (03:11:54Z to 04:07:29Z / 04:04:45Z): `code_changed_during_run` = `v6_seeds.py`;
   - the verifier's own units IPR_VER_ACCESS_V1 / IPR_VER_INTEGRITY_V1: launch drift of its own files.

   No file of `ipr/` or of any other campaign or source package changed during or before any of these units. The grid, evaluation
   and claims code does not import the verifier's code.
7. **Verifier work outside the queue** (verifier observation O5). Both slots were held by the robustness grids, so three label-free
   scripts ran outside the queue between 03:19 and 03:21Z:
   - the decision arithmetic (v4);
   - an access re-check (v5);
   - a seed check (v6).

   They took 0.034 CPU-s in all and are charged with the verifier's out-of-queue work at closeout. Its queued unit
   IPR_VER_DECISIONS_V1 was withdrawn before it started (no `unit_start`).
8. **IA's revision R1 is not the R1 declared in IA_SFG_v1 section 8.**
   - Neither declared IA revision was run. Their triggers were checked on MODEL_FIT records at 02:47Z, as the coordinator log
     records.
   - The revision run, IA_OSG, was diagnosed from the registered reading after its MODEL_SELECT outcome, so it is adaptive as a whole
     (PROTOCOL 7).
   - It is governed by its own card (`cards/IA_OSG_R1.md`), committed with its code before its fit.
   - That card restricts all three gating control paths to the occupation output, the only output the revision changes. Under
     PROTOCOL 7 a revision's card states its controls. The restriction is a choice made after the registered outcome, not a registered
     construction.
   - **Numbering** (statistical reviewer, C9): IA_OSG fills IA's first revision slot (PROTOCOL 7 allows two), hence the tag ADAPTIVE_R1.
     The declared R1's MODEL_FIT trigger failed in version 0: two occupation components of the MODEL_FIT budget keep less than 50% of
     their slack at the next rung (occupation accuracy 37%, women's occupation Brier 45%; the claims review's R4b corrects the
     reviewer's "45% against 50% slack"). The declared R2's trigger held, but it could not change the release.
   - **Diagnosis:** IA_OSG's diagnosis rests on its card's F1 (MODEL_FIT: income binds the shared ladder at MAR1) and F3 (the twin's
     pair recovery sits in its occupation view).
   - **F2 not supported** (statistical reviewer, C2). The card's F2, IA_SFG's income view becoming more recoverable, rests on one
     repetition's cells (S1R1) in which the validation-selected income reader switched. The verifier's independent readers, the
     data-rich readers and the MODEL_FIT readers show no difference. The card is not edited.
9. **The robustness readings first ran before their grid was complete (04:04:50-04:04:59Z).**
   - **What happened:** the D2R lane (IPR_D2R_IA_REG_V1, IPR_D2R_IB_REG_V1) was queued while IPR_GRID_ROB2_V0 still ran. Its units took
     the slot freed by IPR_GRID_ROB2_V1 and read IPR1_ROB with version 0 missing.
   - **No wrong label:** the rules returned D2R NOT_RUN (incomplete robustness cells) for both readings.
   - **What was done:** the two records were moved, unchanged, to `claims/superseded/<reading>_D2R_EARLY/D2R.json`; their SHA-256 and
     writing units are recorded in `claims/superseded/SUPERSEDED.json`. The readings were rerun into `claims/<reading>_D2R` after the
     version-0 grid ended (IPR_D2R2_IA_REG_V1 / IPR_D2R2_IB_REG_V1, 04:07:58-04:08:03Z).
   - **Not read as results:** the early records are never read as D2R results.
   - **Why the early records claim effective refits** (statistical reviewer, C17): their refit-effectiveness field reads
     REFIT_GENUINELY_EFFECTIVE although version 0's robustness cells were missing. The frozen check requires every seed-dependent recipe
     to be checked, not every version x repetition block (the reviewer's invented-data check). The reruns contain every block.
10. **The first reader-custody attempt failed (04:51:04-04:51:05Z).**
    - **What happened:** IPR_READER_CUSTODY_MAIN_V1 asked for the registered pair-view winners of CTL_TWIN, IA_SFG and IB_CSF in cell
      S0R1k0 of IPR1_MAIN. Two of them are the composite recipe `IGNORE_income__CATI1`, which the custody refitter cannot rebuild, so
      the unit failed after 1.2 CPU-s and its lane stopped before its second unit.
    - **The rerun:** READER_CUSTODY2 used cells whose winners are base recipes (IPR1_MAIN S0R2k0: CATI01 for all three; IPR1_R1 S1R1k0:
      ET25 for IA_OSG's pair and occupation views). Both units are all_ok: every refitted reader reproduces the grid's scores bitwise
      (`custody/READER_CUSTODY_*.json`).
    - **Records kept:** the failed unit stays in the queue ledger and counts toward CPU. This is the same failure as the dual-design
      campaign's deviation 10; the cells were not checked for composite winners beforehand.
11. **A narrower, compressed same-device backup (05:45Z).**
    - **Why:** the volume had about 4.4 GiB free (99% used). A backup of the inherited custody scope (about 2.5 GB) plus the restore
      check's full extraction (another 2.5 GB) would not fit, and could have filled the disk.
    - **What changed in `ipr/custody.py`** (committed before the custody unit):
      - the private archive is gzip-compressed (`private.tar.gz`; bsdtar detects the compression when extracting);
      - the occupation-headroom campaign's run tree (785 MB) is left out of the private scope.
    - **What stays in scope:** the admitted mirror and the frozen reference bank this campaign reads (under the income-headroom tree's
      `admitted/`, 194 MB), its admitted logs, the dual-design beside releases and the finite-routing / new-release-mechanisms caches.
      The restore check reads nothing from the excluded tree.
    - **What it costs:** the excluded runs keep their own same-device backup (`pcrl-occupation-headroom-overnight-v1-backup`). This
      backup alone restores and checks this campaign; re-running the reader grids with every reference would also need that backup.
    - **Not done:** no other campaign's data or backup was moved or deleted to free space.
12. **Stale deployment packages and a failed first backup (05:47-05:52Z).**
    - **The failure:** the first same-device backup (unit IPR_CUSTODY_V1, 05:47:03-05:48:04Z) passed its manifest re-hash and bundle
      verification, but its restore check failed.
    - **The cause:**
      - the ten packages of `lanes/DEPLOY_V1.json`, built at 04:05Z, pin the hashes of their code dependencies;
      - commit 221a91f21 (04:11:44Z) changed four of those files afterwards (`ipr/common.py` in a comment, and `ipr/ia/sfg.py`,
        `ipr/paths.py` and `ipr/ia/__init__.py` for the revision);
      - so `ipr.deploy` refused the packages from the restored code (`package dependency hash differs: CODE::ipr.common`), and the
        python3 pass stopped before it re-ran the packaged models.
      - The other pass re-hashed 39 releases, 201 admitted copies and 2,368 reader-cache entries, all ok.
    - **The attempt's record:** `custody/BACKUP_ATTEMPT1_FAILED.json`. Its backup folder (1.14 GB, created by this campaign at 05:47Z)
      was then removed for disk space.
    - **The repair:**
      - the ten stale package folders were moved unchanged to the private `deploy_stale_v1/`;
      - they were rebuilt against the current code (`lanes/DEPLOY_V2.json`, units IPR_DEPLOY_BUILD2_*);
      - all fourteen packages were checked again (`deploy/DEPLOYMENT_CHECK_V2.json`: all bitwise, all refusals, no label read).
      - The backup was then retaken (unit IPR_CUSTODY_V2).
    - **Why the earlier check passed:** `deploy/DEPLOYMENT_CHECK.json` (04:05Z) checked the ten packages with the code they pinned, so it
      stays a correct record of that state. `deploy/DEPLOYMENT_CHECK_V2.json` is the check against the campaign's final code.
