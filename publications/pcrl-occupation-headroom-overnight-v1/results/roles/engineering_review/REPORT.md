# Engineering and deployment review (final at 07:38Z; saved by the coordinator)

**Disclosure.** The engineering and deployment reviewer (a separate agent) returned this report as the text of its final hand-back;
the coordinator saved it here unchanged in substance. Machine record: `CHECKS.json` (21 findings with status, gate tests,
access-log counts, deployment results, provenance table, proposed changes); code and the proposed patches: `code/`. Coordinator
actions on the findings are listed at the end.

Role: engineering and deployment reviewer. I read no person-level labels: every unit I ran was label-free, and the 12 negative gate
tests raise before any data access. I committed and pushed nothing and edited nothing under ohn/.

## Bottom line

- **Gates hold.** No label read precedes the verified public receipt. ATTACK SEX and DEV_REPORT truth were never read. 12 negative
  gate tests were all refused before any data access, and wrote no log line. One gap: the MODEL_SELECT label paths are enforced only
  by logging, not by caller (E12).
- **Deployment works.** Every package I tested deploys bitwise from the six raw fields, under both interpreters, at all three entry
  points (`release`, `release_csv`, `underlying`). Only probability and decision are exported. Every missing, extra, reordered,
  target (PINCP, OCCP) or SEX field and column is refused (42 cases per package). Either interpreter alone can serve every complete
  release.
- **One HIGH defect, which decides an outcome if left alone:** the nomination deployment gate (E1). You have acknowledged it.
- **MEDIUM defects:** E2 tables (fixed in code); E3/E4 package records (partly fixed in 5fb9d494a; tested patch provided); E9 code
  provenance; E13 receipt write-once on failure.
- **No recorded number is affected** by any defect I found.

## Findings (severity; location; result affected?; status)

**HIGH**
- **E1 — nomination deployment gate.** `ohn/nominate.py:40-52,:99`; `ohn/rules.py:457-460`. `_deploy_ok` only reads
  `results/<campaign>/**/DEPLOYMENT_CHECKS*.json`, needs both seeds, and none existed. So every candidate becomes NOT_ADMISSIBLE, and
  `six_raw_fields` is just set equal to the parity result. Result: none recorded yet, but it would decide the nomination. Status:
  acknowledged (records for both seeds of each candidate, including INC@COMPLETE).

**MEDIUM**
- **E2 — tables.** `ohn/tables.py:82` skipped the INC complete releases. Fixed in 7b80b1d5d; UTILITY_FRONTIER.csv still needs
  regenerating.
- **E3 — PACKAGE.json records** (`ohn/deploy.py@5fb9d494a:66-103`). No interpreter, numpy/sklearn/scipy versions, CPU or platform.
  Only the family module is hashed; the release path also loads `ohn/{common,deploy,devrun}`, `ihs/{common,bases,fam_h}`,
  `acs_replication/{deploy,schema,trees}` and `scm_bench/common` (measured by the runtime audit). Falls short of PROTOCOL section 7
  ("recorded CPU, dependencies and source hashes").
- **E4 — manifest completeness.** 5fb9d494a added the PROT witness run dir and module, and INC's Step-1, witness and `fam_a`. Still
  unlisted at runtime: `runs/B/BASES/<learner>/S<s>/{BASE.json,MODEL.pkl}` for every chain that contains B (H3+B, PROT with the B
  witness, INC); INC's mirror income teacher and JOINT/LOCAL E20 files; both SOURCE_ADMISSION.json records (read by `admitted()`).
  The source-anchor states added for COMPLETE packages are not read at runtime (harmless). Runtime integrity still holds through each
  module's own hash chain; a manifest-driven restore would miss these files.
- **E9 — reproducibility record.** `fam_b.py` and `fam_c.py` were not in the registration commit and were first committed after their
  first fits. B records no code hash anywhere. Four recorded C hashes are absent from git, covering seven releases. PROT's OCC.json
  has no code hash, and xgboost/torch versions are not recorded. Mitigation: every release replays bitwise from features with the
  committed HEAD code. Status for PROT: `protect.py` committed at 06b0ca47a (07:21:30Z); deviation 14 covers the uncommitted-run units.
- **E13 — receipt verification, future reporting stage** (`ohn/receipt.py:37-54`). `verify` writes the private record even when
  verification fails, then refuses any re-verification (write-once): one failed attempt permanently blocks `dev_report_truth` unless a
  private file is deleted by hand. It also never compares the published blob ids with the local files (I checked: all five
  registration blobs are byte-identical). Fix: write the record only on success, log failures separately, compare blob ids.

**LOW**
- **E5 — VERIFY.json is hashed in manifests** (`ohn/deploy.py@5fb9d494a:41-42` excludes only `EVAL_*`). Any `verify` run after a
  package build breaks the package (fail-closed); with 5fb9d494a this also covers the witness's VERIFY.json. The same staleness hit
  one of my builds: `protect.py` was edited at 07:08:47Z, 3 s after I built a PROT package.
- **E6 — leftover raw-input CSVs** (`ohn/deploy.py@5fb9d494a:197-199`). `check()` leaves the full DEV_REPORT raw-input CSV (5,943
  people, six fields, no labels) in a system `mkdtemp()` directory. Two copies remained, both in owner-only (0700) directories
  (OHN_DEPLOY_TEST_V1, 05:02Z; verifier, 07:05Z). I removed only my own copy.
- **E7 — check() coverage gaps.** `check()` never tests CSV-path refusals or PINCP. My tests cover both.
- **E8 — value-level inputs.** None, empty or unseen categorical values are silently encoded as `__OTHER__` (source encoder,
  `acs_replication/deploy.py:39-41`). A None numeric raises TypeError rather than the documented exception types.
- **E10 — code hash read at record time, not import time** (`ohn/fam_a.py:522`, `ohn/fam_inc.py:261`, PROT VERIFY). Example: INC
  STEP1 S1 records bb7b3290 but executed 670a8cf6; it crashed in the old print (deviation 8).
- **E11 — wrong record text for the selected witness A1_PO_16 (and A1_PO_8).** OCC.json `operation` / `inputs_used`
  (`ohn/fam_a.py:485-486,:499`) says "admitted 145-column coding", but the release uses the six-column `compact_po` coding; says T*
  comes from the "OHN1_crossfit partition", but it uses `OHN_A_cv_S<seed>`; says "used as is" for the inputs. Needs an erratum.
- **E12 — MODEL_SELECT reads outside the registered paths.** At 07:37:51Z, 62 of 153 MODEL_SELECT label reads came through other
  scripts: 40 verifier `VER_*`, 19 reader-robustness `OHN_RR_*`, 3 `OHN_DECOMP_*`. Deviation 9 lists two of them. All are logged with
  unit ids and purposes.
- **E14 — queue robustness** (`ohn/jobqueue.py:132-140,150-157,172-174,184-188`). With psutil present (it is in `.venv`),
  `max_wall_s` is never enforced; all wrappers so far ran under Homebrew python3, which has no psutil. Refusals are not logged.
  Unit-id uniqueness is checked only against finished units. A killed waiter whose PID is reused can block the FIFO head. None of
  this happened: the ledger shows no orphans, no duplicate starts, and no RSS overrun.
- **E15 — `ohn/devrun.py:71-78,:206`.** `save_occupation` writes OCC.npz before validating the record. EVAL_MS files are overwritten
  on a re-score. No configuration has been scored twice (142 ledger rows checked).
- **E21 — parameter files not anchored to the release record.** The B, C and PROT `predict` functions don't check STATE.json /
  PARAMS.json against the record. PROT writes THETA/CODEBOOK and PARAMS.json after `save_occupation`
  (`protect.py@651d57b7f:645-651,700-705`). OCC.json pins none of them, including the release `scale`.

**INFO**
- **E16** — the guard SE ratio is +inf when SE is 0 (`ihs/evaluate.py:94`). Inert, because `rules.py:146` floors the SE.
- **E17** — witness task-only status is a `roles_read` string test (`witness.py:38`). It is correct for all current records.
- **E18** — the grid lock does not cover `ohn/common.py`. My replay reproduces DEV_GRID.json exactly.
- **E19** — PROT `fit` accepts a 1e-9 tolerance for witness parity instead of bitwise; parity was bitwise in practice.
- **E20** — any edit to the five receipt-pinned files closes all label reads permanently. The ATTACK gate checks only that a
  FINALIST_LOCK is committed.

## Deployment check results

- **Requested set** (OHN_ENG_DEPLOY_CHECK_V1, python3, the coordinator's `check` unchanged): H3+SOURCE:JOINT (11 dependencies),
  H3+B/B2_SINGLE (17), H3+A/A1_PO_16 (23), all seed 0. All bitwise at API, CSV and underlying, decisions equal; exported keys exactly
  `{decision, probability}`; all 11 refusals hold; the batch-of-7 differences are 0.
- **Extended checks** (OHN_ENG_DEPLOY_REVIEW_PY314_V2 and _VENV_V1; those three plus INC:INC_H3_PO@COMPLETE S0): 42 refusal cases per
  package (17 API, 12 CSV, 13 underlying), all PermissionError or ValueError in both interpreters. Output contract and dtypes
  correct. Reordering all 5,943 DEV_REPORT rows, 25 single-row calls and duplicated rows all give bitwise-identical outputs: each
  output depends only on its own row. All six roles replay bitwise from features. Runtime audit: no label-bearing file opened, no
  subprocess, no socket. One API release of all DEV_REPORT rows takes 1.1–1.6 CPU-s.
- **PROT and INC** (OHN_ENG_PROT_INC_DEPLOY_PY314_V1 / _VENV_V1, build and check in one process, deploy.py 5fb9d494a): 8 packages
  (P2_PAIR_B80 S0/S1, P2A_PAIR_B80 S0/S1 (theta arms, B and A witnesses), P2_COMP_B80 S0 and P2A_COMP_B80 S1 (codebook arms),
  INC_H3_PO@COMPLETE S1, INC_TW_PO@COMPLETE S0). `all_bitwise` and `all_refused` true under both interpreters. `protect.predict` uses
  numpy only (torch only for training); the witness `predict` is bitwise across interpreters; all-six-role replays of P2_PAIR_B80
  and P2A_PAIR_B80 S0 (committed e4c3dc3f, whose `predict` equals 06b0ca47a's) are bitwise in both interpreters.
- **Interpreter needs.** Every deployment path uses only numpy and scikit-learn. The sklearn 1.8.0 pickles load and predict bitwise
  under Homebrew 3.14.3/numpy 2.4.4 and `.venv` 3.13.7/numpy 2.4.2 (both Accelerate BLAS, Apple M4 Pro). Family C (fitted under
  `.venv`) also replays bitwise under python3 (C2_XF_D1 S0/S1, C3_XC_D2 S0). Training needs `.venv` for C (xgboost) and PROT (torch).
  The reader grid is pinned to `.venv` (its lock records numpy 2.4.2).
- **Proposed fix:** `code/proposed_ohn_deploy.patch` (re-based on 5fb9d494a, 68 changed lines): the B base-learner branch and
  recursion through the witness chain; INC's three mirror files; exclusion of `VERIFY*`; `REPO::` entries for the governance files;
  `environment_at_build` and `code_closure_sha256`, reported by `check`; the CSV under `.private/` with cleanup; the PINCP and CSV
  refusal cases. Tested in OHN_ENG_DEPLOY_PATCH_TEST_V2 on six package types: all bitwise, 18/18 refusals, zero unlisted runtime
  files, zero unrecorded code (only the empty `__init__.py` files). Optional `code/proposed_ohn_fam_b.patch` (4 lines) makes
  `fam_b.predict` check STATE.json; demonstrated on copies: the current code silently accepts a modified STATE.json (max |Δp| 0.186
  against the release); the patched code refuses it.

## Reproducibility record

- **A:** Homebrew python3 3.14.3, numpy 2.4.4, sklearn 1.8.0, scipy 1.17.1; code hash taken at the end of the fit. A0: e787ac9e =
  b1fd591d0. A1 S0: 3ab1cd74 = cb89a8b51. A1 S1 records 8be2baeb, not in git; reconstructed exactly as cb89a8b51 plus the unused D6–D8
  diagnostic block, so the fit code is the committed one. VERIFY at 06:07Z with 57ce5130 (= HEAD) is bitwise.
- **B:** python3; sklearn version only in BASE.json; no code hash recorded anywhere. B1 S0 fits (05:43Z) and their base learners
  (from 05:13Z) predate the first commit of `fam_b.py` (2f785ec18, 05:45:46Z). The B2 fits ran between 2f785ec18 and 5c9c6cef5; that
  diff only adds B3 and record text, so the single/pool paths are unchanged. `verifyall` at 06:21Z (HEAD) is bitwise for all 10
  releases.
- **C:** `.venv` 3.13.7, numpy 2.4.2; xgboost 3.2.0 not recorded. Code hashes not in git: e89393f2 (C1_RB_D1, C1_RB_D2), 4c18d641
  (CC_CONT_D4), 9631a62b (C2_XF_D2 S0), 81137716 (C2_XF_D1 S0/S1, CC_XF_CONT_D4). In git: ed663529 (CC_VSCAL) and dc749253 (C3 ×4).
  `verifyall` at 06:23Z with 71b0bfad (= HEAD) is bitwise for all 12.
- **PROT:** `.venv`, torch 2.10.0. Code hash only in VERIFY.json, taken at the end of the fit: P1 = 721a64c15; P2/P2A = 651d57b7f
  (committed after the first P2 start, deviation 10). The P2A S1 units ran an uncommitted file (deviation 14).
- **INC:** RELEASE/VERIFY records bb7b3290 = HEAD. STEP1 S1's hash is misattributed (E10).
- **Scorer and grid:** the scoring path is unchanged since registration (the devrun change was additive; `evaluate.py` unchanged).
  All 96 stored evaluations' wire hashes equal their saved releases; one anchor income wire per anchor and seed. The grid lock and
  partition replay are exact.

## Gate integrity facts

Receipt 435fe44e5 was verified at 05:11:09Z. The first MODEL_SELECT label read was at 05:11:30Z, the first training read at
05:11:36Z. All reads are logged with a unit id. Task-only families never read SEX in fits; their 28 SEX reads are all scorer
diagnostics (deviation 4). No ATTACK or DEV_REPORT truth log exists, and `dev_report_truth` is never called. No campaign code uses a
non-ohn label loader or opens label exports directly.

## Required fixes, in order (reviewer's list)

1. Before nominating, run E1 (save `DEPLOYMENT_CHECKS_<name>.json` records for both seeds of each candidate).
2. Regenerate UTILITY_FRONTIER.csv.
3. Apply the deploy patch (or equivalent), then build candidate packages after the last verify of each configuration and of its
   witness chain.
4. Write errata or deviation entries for E11, E10, E12 (the full unit list), and the provenance statement for B and the seven C
   releases (E9).
5. Fix E13 before any reporting receipt.
6. Optional: the `fam_b` patch; import-time code hashing; E14, E15, E21.

## Compute

9 queue units, all done, 206.1 CPU-s in total. OHN_ENG_DEPLOY_REVIEW_PY314_V1 was withdrawn with SIGINT while still waiting for a
slot; it has no ledger record (its PROT package had gone stale). Out-of-queue work charged as ENGINEERING_REVIEW_AGENT: 60 CPU-s.
The campaign total after the charge was 4.613 CPU-h.

---

## Coordinator actions on these findings (07:40-07:50Z)

- E1: packages built and checked for both seeds of every candidate through `ohn.deploy_batch` (batch 1, 34 packages, unit
  OHN_DEPLOY_BATCH1_V1, all bitwise, all refused; batch 2 with the patch below), reports under `results/.../deploy/`.
- E2: fixed (7b80b1d5d); tables regenerated at closeout.
- E3 / E4 / E5 / E6 / E7: the proposed deploy patch applied unchanged (f4d6ad65f) and the packages rebuilt after every
  configuration's last verify (batch 2; batch 1 kept under runs/DEPLOY_BATCH1 as a record). The three leftover DEV_REPORT raw-input
  CSV copies in the system temp directory (05:02Z deploy test, 07:05Z verifier, 07:30Z batch 1) were inspected and removed.
- E9 / E10 / E11 / E12: recorded in PROTOCOL_DEVIATIONS.md (items 17-19).
- E13: not fixed; no reporting stage runs in this campaign (no finalist). Recorded as a known defect for any successor.
- Optional fam_b patch: not applied. Inside a package, the B run directory (including STATE.json) is already hash-checked by the
  manifest before `predict` runs; the patch would only matter for direct, unpackaged calls.
- E14 / E15 / E21: recorded; none occurred.
