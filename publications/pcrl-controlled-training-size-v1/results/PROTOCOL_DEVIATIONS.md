# Protocol deviations, amendments and disclosures

Every departure from PROTOCOL.md, every dated amendment (made before the stage it affects), every failed attempt and every timing
disclosure is listed here in the order it happened, with the `date -u` time at which the entry was written. Earlier entries are never
edited; corrections are new entries.

## Registration-time disclosures (written before any new real-label computation)

1. **Worktree directory.** The campaign branch research/pcrl-controlled-training-size-v1 was created (23:53:57Z) from the
   correction-transfer research tip 507024649 inside the worktree directory `.claude/worktrees/pcrl-correction-transfer-stability-v1`,
   because this session is bound to that directory. The previous branch is unchanged; only the directory name is inherited.
2. **Known source margins.** The coordinator and the statistical reviewer had read the published source records of the full-data
   A1_PO_16 package (WITNESS_SELECTION.json of the occupation-headroom campaign: it met the full reserve, minimum occupation-side ratio
   4.30) and the correction-transfer training-size diagnostic before this registration. The trigger and the strength-rule constants
   were therefore set with that knowledge (RULES.md states which constants depend on published source values).
3. **Stage B epoch order at S50 (control recipe adaptation, decided before any fit).** The replication's Stage B loop draws each
   epoch's 40 batches of 256 rows from one permutation of the training rows, which assumes at least 10,240 people. Every S50 subset
   has fewer (9,934-10,046), so the unchanged loop would leave a short last batch (S50_O2: 62 rows; S50_O3: 30) or, for S50_O0 /
   S50_O1, a short batch and one EMPTY batch per epoch (a step with no task gradient). For subsets below 10,240 people the control
   extends the same permutation cyclically to 10,240 positions, so every update has exactly 256 rows and every subset person is used
   once before any is reused; for 10,240 or more people the unchanged loop runs (bitwise equal; task-refit investigator's selftest).
   The update count (20 x 40 x 256) is unchanged, so smaller subsets are passed over more often (recorded per fit).

## After the registration receipt (verified 00:20:07Z)

4. **Two selftest label reads outside the queue (written 00:22Z).** The size-fit selftest contained a probe asserting that a
   MODEL_FIT label read is refused before the registration. It was run 6 s (task-refit investigator, 00:20:13Z) and 20 s
   (coordinator, 00:20:27Z) after the receipt verification, when the access layer correctly released labels, so the probe read
   MODEL_FIT occupation of S100 (20,000 people) through `tsz.common.load` (operation 'fit', entry point tsz.sizefit:selftest, logged,
   unit id null). Both returned arrays were discarded at once: nothing was fitted, computed or written except the two access-log
   lines. The probe was removed (tsz/sizefit.py frozen 00:21:51Z, SHA-256 de5d9cb4...) and a module flag now makes the real loader
   refuse during selftests. Both reads are counted in ACCESS_RECEIPT.json.
5. **Dated code amendments from the engineering review (written 00:41Z, before the size analysis and before any selection).**
   (a) tsz/evaluate.py (ENG1-01, applied 00:39Z): the size table refuses to read MODEL_SELECT until every registered cell is fitted
   or recorded missing, and a missing cell makes its labels INCOMPLETE instead of being dropped (implements RULES.md sections 2 and
   3.3). (b) tsz/rules.py `interpretation` (ENG1-05, applied 00:40Z): the IR4 reading is also reported when the primary size endpoint
   is unresolved under another margin pattern, as RULES.md 3.6 states; no label changes. Neither change touches a fit, a strength rule
   or a reader. (c) Disclosure (ENG1-10): PAIR's training critic reads the frozen H3 income, fitted by its source on all MODEL_FIT, so
   in the nested estimator it is the one upstream input not refitted per outer fold; it never enters a held-out cost. Other findings:
   adjudication A2.
6. **Failed coordinator deployment checks before the trigger (written 00:41Z).** TSZ_DEPLOYCHECK_W100_V1 failed (a local variable
   shadowed the check function in tsz.deploy's CLI) and TSZ_DEPLOYCHECK_W100_V2 failed (the packages pinned the hash of the
   pre-fix tsz/deploy.py). Neither compared any output. The packages were rebuilt as W100_V*_B2 with code pins narrowed to the modules
   each package kind executes; TSZ_DEPLOYCHECK_W100_V3 passed. The superseded W100_V0 / W100_V1 package directories are kept.
7. **Benchmark units ran uncommitted code (written 00:58Z; found by the independent verifier, check V6.PROCESS.CODE_PROVENANCE__V1
   FAIL).** Five label-free compute-benchmark units of the correction investigator (TSZ_BENCH_CORR_ROUTING_V1 00:14:46Z,
   _CORRECT_LOC_V1 00:14:54Z, _CORRECT_PAIR_V1 00:15:22Z, _WITNESS_V1 00:15:50Z, _DIRECTION_PAIR_V1 00:17:47Z) ran tsz/nested.py
   before any campaign code was committed (first commit b3a5343f4 at 00:19:41Z), a literal departure from PROTOCOL section 12. They
   used invented labels on label-free features, before the registration receipt, and fed only the compute-budget estimate
   (COMPUTE_BENCHMARK.json); no outcome depends on them. Every production unit after the registration ran committed modules
   (verifier V6.PROCESS.ORDER__V1 PASS). Code edits to modules a running unit did not execute are listed by the queue and the
   verifier, not failed.
8. **Inherited queue record defect (written 00:58Z; verifier).** tsz.jobqueue (unchanged from cts.jobqueue) strips git's porcelain
   output before slicing each line, so when the first line starts with a space the first character of that path is lost in the
   ledger's `uncommitted_code_paths` (e.g. 'sz/evaluate.py'). The code digest and git head recorded at launch are unaffected. Not
   fixed during the run (the queue module serves running lanes); the launch audit reads paths from the digests, not these strings.
9. **A time stamp written ahead of the clock (written 01:33:53Z).** The coordinator's adjudication request to the statistical reviewer
   about the reader-seed check was headed "01:40Z", an estimate; `date -u` read 01:33:28Z right after it was sent. A correction was
   sent at once, before any ruling. Rule kept: every time is a `date -u` reading.
10. **Reader-seed check outcome (written 01:33:53Z; ruling pending).** READER_SEED_CHECK.json (unit TSZ_SEEDCHECK_COND_V1) returned
   READER_SEEDS_NOT_EFFECTIVE: the seed-dependent ExtraTrees recipes ET5 / ET25 gave bitwise identical SCORE scores under the two
   reader seeds in 16 of 336 checks, all on the income view of the inherited references B2TWIN and RETIRED_B3CTL, whose income
   outputs take only 6 distinct vectors on MODEL_SELECT (trees grown to purity on one 6-valued input are seed-invariant). On the
   witness and every selected release, and on every other view and member, all six seed-dependent recipes differ. The check covered
   every bank member (a superset of the registered minimum, W100_v and every selected release). Under RULES.md section 9 no stability
   label is computed until the statistical reviewer rules on the scope of the check; the ruling, its timing (after the check outcome,
   before any stability label) and its classification (clarification or amendment) are recorded in adjudication A3.
11. **Verifier reader labels computed before the seed check (written 01:34:24Z).** The independent verifier's reader replay (unit
   VER_READERS_V1, 01:32:47-01:33:09Z) computed its own stability, target and recipient-gate labels for all bank pairs before
   READER_SEED_CHECK.json existed and before the coordinator's instruction to wait reached it. They are stored, unreported, in the
   verifier's append-only CHECKS.json (V5.READERS.LABELS__V1) and records/V5_READERS__V1.json. The coordinator and the statistical
   reviewer were told not to read them; the reviewer rules on the seed-check scope from the seed-check record alone and states so.
12. **Correction of item 11's timing (written 01:38:12Z; raised by the statistical reviewer).** Item 11 says the verifier's labels
   were computed "before READER_SEED_CHECK.json existed". That is wrong: the seed-check record carries utc 01:31:29Z, and the
   verifier's unit VER_READERS_V1 ran from 01:32:59Z to 01:33:09Z by the queue ledger (the verifier reported 01:32:47Z as its own start).
   The labels were computed after the seed-check record was written, before the coordinator had examined it or told the verifier to
   wait (about 01:33Z). Item 11 is otherwise unchanged; the verifier's labels stay unread until the claims are computed.
13. **Adjudication A-SC1 applied (written 01:38:12Z).** The statistical reviewer ruled at 01:36:00Z (ADJUDICATION_SEEDCHECK.md): an
   AMENDMENT of RULES.md section 9 item 2's scope, made after the seed-check outcome and before any reader-stability label was computed
   or read by the coordinator or the reviewer. Grid invalidation is limited to unexplained equalities; the minimum set (W100_v, LOC_SEL,
   PAIR_SEL) must show no equality; each other equality must be explained by another seed-dependent recipe differing in the same cell,
   the same recipe differing on another view, and a label-free count of the view's distinct inputs (readers/WIRE_CARDINALITY.json).
   Both outcomes are kept: literal READER_SEEDS_NOT_EFFECTIVE, amended READER_SEEDS_EFFECTIVE (if the amended check passes). No
   stability formula changes; a non-gating flag COARSE_REFERENCE_INCOME_WIRE marks the affected comparisons. tsz/claims.py takes grid
   validity from A-SC1 (code change committed before its run) and tsz/seedcheck.py gains a label-free `cardinality` command.
14. **Custody dry run failed once (written 01:46:34Z).** TSZ_RESTORE_DRYRUN_V1 failed on the superseded W100_V0 / W100_V1 packages
   (item 6), which pin a pre-freeze hash of tsz/correct.py. They were moved, unchanged, to `.private/.../deploy/superseded/` so that the
   restore check covers only the 11 final packages; TSZ_RESTORE_DRYRUN_V2 passed (all_ok), and so did the restore check from backup B1
   (TSZ_CUSTODY_B1_V1). No external drive was attached, so off-device custody is PENDING (BACKUP_VERIFICATION.json); nothing was
   mounted or forced.
