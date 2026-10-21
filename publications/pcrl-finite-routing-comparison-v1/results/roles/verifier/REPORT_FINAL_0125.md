# Independent verifier, final report (saved verbatim with attribution)

Author: the independent verifier of campaign `pcrl_finite_routing_comparison_v1` (separate code under `roles/verifier/code/`).
Returned at about 01:26Z on 2026-10-21 (its own header: written 01:25:06Z; work from 23:44:46Z). Saved verbatim by the coordinator
(indentation removed). Its last commit before this report: 1415576ee.

---

**Independent verifier, final report — campaign pcrl_finite_routing_comparison_v1 (written 01:25:06Z on 2026-10-21; work from 23:44:46Z)**

**Bottom line.** Every decisive record I could replay agrees with my own code and data paths. That covers T0–T3, the qualification decision, the protected-stage tables, the registered and robustness grids, the D1–D6 decisions under ER1, the reader-budget flags, the stability record, and the same-device backup. There is no numerical FAIL. The three procedural FAILs are one cause: the append-only card amendment, your deviation 7.

**Totals.** CHECKS.json holds 115 append-only checks; earlier entries are re-hashed before every write.

| Status | Numerical | Procedural |
|---|---|---|
| PASS | 96 | 66 |
| FAIL | 0 | 3 |
| ATTENTION | 0 | 4 |
| NOT_APPLICABLE | 17 | 42 |
| NOT_RUN | 2 | 0 |

**V8: backup pcrl_finite_routing_comparison_v1_20261021T010846Z (my own fresh restore; all steps label-free, all through the queue)**
1. **Files, bundle, tip (FRC_VER_BACKUP_V1, 01:23:33Z): PASS.**
   - My own SHA-256 of TIP_COMMIT, branch_since_base.bundle, code.tar and private.tar equals MANIFEST.sha256, which lists exactly these four files.
   - `git bundle verify` reports "is okay".
   - The bundle head 628937bb71b5f9a774f1cc0836ad436bc0a4bd6b points to refs/heads/research/pcrl-finite-routing-comparison-v1, and TIP_COMMIT equals that tip.
2. **Fresh extraction (FRC_VER_BACKUP_EXTRACT_V1, 01:23:40Z): PASS.** 6,505 files (1.92 GB) went into a new directory under my scratchpad, not the worktree or the backup folder.
3. **Deployment replay (FRC_VER_BACKUP_DEPLOY_V1, 01:23:45Z): PASS.**
   - Every module came from the restored root: frc.common, frc.fa.route, ohn.fam_a, acs_replication.deploy.
   - For FA/PRC_S2, FA/TWB_C64 and FA/HQ_S2, both versions, 5,943 DEV_REPORT people each:
     - the restored raw inputs, encoded by each package's own encoder, equal the restored admitted features bitwise;
     - my own replay (the saved teachers' decision functions, then my own softmax/clip, then my own protected/HQ/twin routing) equals both the stored DEV_REPORT wires (runs/STORED) and RELEASE.npz, bitwise, all 6 of 6.
   - The minimum routing gap is 1.5e-7 or more, so the routing is not arithmetic-order sensitive.
   - Refusal probe: the restored predict refused per-request keys, draw 1, and reordered feature columns.
4. **Reader custody (FRC_VER_BACKUP_CUSTODY_V1, torch venv Python 3.13.7, 01:23:48Z): PASS.**
   - All 10 restored pickles re-load.
   - Each pickled predict function resolves to restored code (ffv.readers, scm_bench.readers, task_conditioned_representation.attacks).
   - predict(model, X_score) equals score_raw bitwise for all 10, with 2,391 SCORE rows each.
   - The pickle hashes equal the custody records.
5. **Optional probe (FRC_VER_PROBE_V1): PASS.** In both versions, PROBE_STRENGTH's strength-8 partition hash equals my own routing of the saved PRC_S2 book (PRC@8.0) on the cached out-of-fold statistics (V0 4257345f…, V1 6dca8dfb…).
6. **Cleanup (FRC_VER_BACKUP_CLEANUP_V1): PASS.** My restore directory is deleted and confirmed absent.

This is a same-device check. Off-device custody remains your item.

**Earlier results, all with my own code**
- **V1 (receipt, admission, access).**
  - The receipt is re-verified with ls-remote, and the source admission is re-hashed.
  - The final access audit covers 35 reads: none before the receipt and none from a closed role.
  - No label read was logged while RULES.md/RULES.json differed from the receipt (6d6cd0829 at 00:32:13Z until the restoration adbed977f at 00:53:08Z; deviation 9).
  - R11 shows no unit ran an entry package different from its launch HEAD (81 units).
- **V2 (T2).** TWB B1 tables match bitwise. The source A decoders, recomputed from labels, match bitwise. All six roles share the partition. My deployment replay matches bitwise.
- **V3 (T1).** Every NEST number reproduces (largest difference 4.3e-19), and the GATE reproduction holds. Q-PE is PE_ERROR_SUPPORTED at 16×64 and 24×128.
- **V4 (T3/E1).** Every evaluation record reproduces (values ≤ 1.7e-16, SEs ≤ 9e-19). QUAL, the capacity walk and D1 = REPAIRED_TASK_ONLY_RELEASE reproduce from my own numbers.
- **V5.**
  - P1 selection, brackets and F4 reproduce from the recorded numbers.
  - B1 tables of all 18 protected-stage configuration-versions match bitwise on their own out-of-fold partitions.
  - Exact alias: PRC_S1 ≡ PRC_S2 in V1 (deviation 3).
  - The R1 UNT bracket replays.
- **V6 / V6b.**
  - 4,800 grid paired cells match exactly.
  - ER1 validity: 129 seed-dependent equalities, 85 on the minimum set, all explained.
  - MAIN_ER1 and R1_ER1 reproduce in full: D2 NO_REDUCTION_BEYOND_ANY_CONTROL, D3 NOT_SUPERIOR, D4 NOT_MET, outcome ENGINEERING_REPAIR.
  - The reader-budget flags reproduce.
- **V7.**
  - 2,592 robustness-grid cells match exactly.
  - Everything in CLASSIFIER_STABILITY.json matches: robustness NOT_CONFIRMED against every matched control and target reference, the ablation labels, the split hashes and overlaps, and REFIT_NOT_GENUINELY_EFFECTIVE (240 strict minimum-set equalities).

**Every non-PASS check**
- **Procedural FAIL ×3 (V1.ACCESS_LOG__V2/__V3/__V4).** Rule R6, "card unchanged after use", fires on the 4 T1 reads. V1B (ATTENTION) shows the change is append-only: Addendum A1, 23:56:09Z; the logged bytes are recoverable from git. This is deviation 7; I did not widen the rule.
- **Procedural ATTENTION ×4.**
  - V1B, above.
  - V1.UNIT_CODE_AT_LAUNCH__V3/__V4: units launched while unrelated files were uncommitted (deviation 8).
  - V5C: the PRC_S1/PRC_S2 V1 exact alias (deviation 3).
- **Numerical NOT_RUN ×2.**
  - V5A P1 person-level replay: P1 saved no per-person outputs and I did not refit, so the spends, SEs, nested AUCs and F4 bootstrap SE were checked only for internal consistency.
  - V7.STABILITY__V1: the stability record did not exist yet. It was superseded by __V2, which passed.

**My disclosures**
- **Failed unit FRC_VER_UTILITY_V2 (00:10:30Z).** My own bug: a comparator recorded as missing. It read MODEL_SELECT labels once (logged), wrote nothing, and was rerun as V3.
- **Selftest defects in my own invented cases.** Three were fixed before any real run (v3, v4, v5d). Commit 3b7bfa8a1 carried a failing v5d selftest because a pipe hid the exit code; the next commit fixed it. No acceptance rule changed.
- **Stopped processes, before any unit_start.**
  - FRC_VER_UNT_V1, while still waiting (it left a unit directory only; rerun as FRC_VER_UNT_V2).
  - My chained V6 wait loop at 00:42:31Z (V6 was resubmitted directly).
- **Out-of-queue label-free looks.** Two, later repeated as queued checks: the launch-code scan and the PRC_S1/PRC_S2 comparison.
- **Admitted loaders used.** frc.common.features/load; ohn.fam_a.compact_po; ohn.devrun.reference_release; ohn.common.frozen_bank; tsz.evaluate; the package encoder acs_replication.deploy.encode_raw (in V8); the saved models' own decision functions; the custody pickles' own predict.
- **Not taken from elsewhere.** No frc/nrm numerical function was imported for any verified quantity.
- **Label reads.** 4, all operation `verify` inside queue units after the receipt; none since 00:25:31Z.
- All times come from `date -u` or the ledger; none was rounded ahead.

**Not covered**
- Reader SCORE AUCs and bootstrap draws from raw scores (only fingerprints are saved).
- Winner-recipe selection; nrm reader_stability; the FRC1_LC grid.
- Off-device custody.

**CPU.** 26 verifier units, one failed, used 45.04 CPU-s; I charged a further 17 CPU-s. Total 62.04 CPU-s, about 0.017 CPU-h. The campaign total was 1.732 CPU-h at 01:25:06Z.

**Files.** Everything is under `results/pcrl_finite_routing_comparison_v1/roles/verifier/`:
- code/: vlib, v1_access, v1_admission, v1b, v2, v3, v4, v5, v5b, v5c, v5d, v6, v6b, v7, v8, v8b;
- CHECKS.json;
- records/, with history/.

Last commit 1415576ee. Only my own files were committed; the untracked empty lock file is intentional.
