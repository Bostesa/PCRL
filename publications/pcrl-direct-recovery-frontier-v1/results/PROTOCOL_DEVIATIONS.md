# Protocol deviations (campaign `pcrl_direct_recovery_frontier_v1`; append-only; each item with its reason and timing)

1. **A reader-grid selftest ran without the one-thread environment** (coordinator; before registration; invented data only). The
   command `python -m drf.devreaders selftest` (torch interpreter) was run outside the queue without OMP / BLAS / torch thread limits,
   ending before 02:53:57Z (the `date -u` of the next command) after 1 min 46 s of wall time, with 363.7 s user + 720.8 s system CPU at about ten cores. This breaks the
   rule of one numerical thread per heavy job (no other heavy job was running). The whole 1,085 CPU-s is charged
   (`OUT_OF_QUEUE_DEVGRID_SELFTEST_MULTITHREADED`, charged 02:53:57Z); the charge note's time range ("about 02:53-02:55Z") was an
   estimate written before `date -u` was read, and the times stated here supersede it. Every later out-of-queue command exports the
   thread limits; numerical units run through `drf.jobqueue`, which sets them. No result depends on it.

2. **Files changed while fit units ran** (coordinator and verifier; recorded 2026-10-25 after 03:35:46Z by `date -u`). The queue's
   end-of-unit code manifest lists files that changed during DRF_FIT_V0_V1 / DRF_FIT_V1_V1 (`drf/drclaims.py`, `drf/drplan.py`:
   created by the coordinator) and DRF_FIT_R1_V0_V1 / DRF_FIT_R1_V1_V1 (also `drf/deploy.py` and the verifier's new files under
   `roles/verifier/code/`). None of these files is imported by `drf.dr fit` (the unit's own modules `drf.dr`, `drf.solver`,
   `drf.finite_law`, `drf.common`, `drf.release` and the frozen sources were unchanged and committed at launch, as the launch digests
   record). Later units are launched only after the code they import is committed.

3. **Item 2 extended: later units also ran beside uncommitted or changing files** (coordinator; recorded 2026-10-25T04:21:47Z by
   `date -u`). The statement in item 2 that later units are launched only after the code they import is committed held, but the
   queue's manifests also list files that were uncommitted at launch or changed during these later units:
   - DRF_GRID_DRF_R1_V0_V1 / V1_V1 (03:34-03:45Z): changed during the run `drf/custody.py`, `drf/frontier.py`,
     `drf/restore_check.py` and verifier files.
   - DRF_OPT_V0_V1 / V1_V1 (03:45-03:54Z): `drf/frontier.py` and verifier files uncommitted at launch; `drf/frontier.py` and verifier
     files changed during the run.
   - DRF_CLAIMS_DRF_V1_V1 / DRF_R1_V1 (03:55Z): verifier files uncommitted at launch.
   - DRF_FIT_R2_V0_V1 / V1_V1 (03:59-04:12Z): verifier files uncommitted at launch; `drf/closeout.py`, `drf/drtables.py` and verifier
     files changed during the run.
   - DRF_GRID_DRF1_R1_ROB_V0_V1 / V1_V1 (from 04:12Z): verifier files uncommitted at launch; `drf/frontier.py` edited during the run
     (adding the R2 rows).
   Checked by import inspection at 04:21Z: `drf.dr` loads `drf.common`, `drf.solver`, `drf.finite_law`, `drf.release`, `nrm.fa.prc`
   and `ohn.fam_a`; `drf.devreaders` loads `drf.common`, `drf.readerextra` and admitted source modules (`ffv`, `ohn`, `scm_bench`,
   `task_conditioned_representation`, `tsz.evaluate`); `drf.drclaims` loads `drf.claims`, `drf.common`, `drf.dr`, `drf.rules`,
   `drf.solver`, `nrm.rules` and `tsz.rules`. No unit imports any listed file, and verifier code is imported only by verifier units
   (whose own disclosures are in `roles/verifier/VERIFICATION_REPORT.md`). No result depends on it. The verifier's files were
   committed in ffb524dcc (04:19:12Z) with the hashes its report lists.

4. **A cosmetic defect in the frozen rules code repeats SUCC in the D2R records** (found by the verifier, hand-back 2, and by the
   coordinator; cause named by the statistical reviewer, `roles/statistical_reviewer/CLAIMS_REVIEW.md` C7; recorded
   2026-10-25T05:06:42Z by `date -u`). `drf.rules.d2r` lists START (SUCC) and then the nine references, which include SUCC again, so
   `claims/DRF_R1/D2R.json` and `claims/DRF_R2/D2R.json` repeat SUCC in `beside`, `sign_changes`, `recipient_failures` and the
   headline caveat text. The family label uses the decisive set and the per-comparator entries are keyed once, so no label, number
   or set is affected (the verifier's independent D2R matches both records, 434 of 434 items each). The frozen rules are not
   patched; the documents name each comparator once.

5. **Item 3 extended to the units after 04:12Z** (coordinator; recorded 2026-10-25T05:06:42Z by `date -u`). Of the non-verifier
   code, only three files were edited while a unit ran, and none was imported by it:
   - `drf/frontier.py` during the R1 robustness grids;
   - `drf/restore_check.py` during the R2 grids (also uncommitted when DRF_VER_C1_V3 started);
   - the new `drf/deploy_validity.py` during DRF_VER_C6_V1 and DRF_DEPLOY_CHECK_R2_V1.
   The entry points `drf.devreaders` and `drf.deploy`, and the verifier's checks, do not load these files. The verifier's own files
   were uncommitted or changing during most units; they are imported only by verifier units, whose disclosures are in
   `roles/verifier/VERIFICATION_REPORT.md`. The queue records every path and hash.
