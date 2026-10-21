# Protocol deviations (campaign `pcrl_finite_routing_comparison_v1`)

Departures from the registered PROTOCOL.md, the card and the rules, in the order they occurred. Each entry: what, when (`date -u`),
why, and its effect on any result. Changes adjudicated before registration are in roles/coordinator/ADJUDICATION_PREREG.md, not here.

(None at registration, 23:47Z.)

1. **T1 ran before the engineering review returned (disclosure).** The T1 units FRC_NEST_* ran 23:48:50-23:49:41Z; the engineering
   review (started 23:40Z, before registration) was returned at about 23:52Z. Its source-pin finding was then fixed. Effect: none on
   any value. Every read-only source artefact's SHA-256 at 23:55:16Z equals its admission pin recorded at 23:19Z
   (`roles/coordinator/SOURCE_PIN_CHECK.json`), and T1 reproduced the GATE records exactly and the saved decoders bitwise.
2. **A vacuous identity in the T1 records (disclosure).** The field `A_and_B1_count_the_same_people` in fa/NEST_*.json (and the
   corresponding selftest check) is an identity that cannot fail (engineering review finding 8). It is kept in the append-only
   records, is not evidence, and later records carry `oof_partition_sha256` instead.
3. **One configuration spent on an exact alias (00:02Z).** PRC_S1 in version 1 (selected strength 2.0) and PRC_S2 in version 1
   (8.0) transmit identical vectors on all six roles: the full-pool chain reaches the same partition at both strengths, while the
   alias check of P2 compares the selected strengths, not the wires. PRC_S1 V1 is counted in the 48-configuration budget, is not a
   reader-bank member, and is reported as an exact alias of PRC_S2 V1. PRC_S1 does not exist in version 0 (NOT_FEASIBLE).
4. **Version-specific controls are not pooled path points (00:09Z).** HQ_BR exists in version 1 only (version 0:
   `CONTROL_DID_NOT_REACH_SPEND`), and PRC_S1 in version 1 only. The registered comparisons pool both versions, so these are not
   bank members; the pooled HQ path is [HQ_S2] (HQ_S2 spends more than PRC_S2 on MODEL_SELECT in both versions). Recorded in the
   claims plans' notes.
5. **Revision R1 (card FA_FRC_R1, committed 00:06:27Z before its fit and before any MODEL_SELECT read of a protected release).**
   Not a deviation (PROTOCOL section 11 allows two revisions); listed here because it changes the untargeted control path of D2. The
   registered reading ([SHUF_S2]) and the R1 reading ([SHUF_S2, UNT_BR]) are both reported; the registered label is never
   overwritten.
6. **Post-hoc engineering repair ER1 of the grid-validity rule (00:27Z).** The first claims run on FRC1_MAIN (00:25:50Z) returned
   NOT_COMPUTED_GRID_INVALID for D2-D4 ('85 seed-dependent equalities on the minimum set'). All 85 are ET5 / ET25 on the income view
   (16 distinct vectors) or occupation view (64; once 32) of the finite releases; for every one, another seed-dependent recipe
   differs on the same wire, view, version and repetition, the same recipe differs on the pair view, and the distinct-input count is
   recorded. Mechanism: an extra-trees forest on a few-valued view isolates every distinct value in its own leaf and reduces to a
   seed-invariant frequency table. The inherited NRM1 rule explains such equalities only off the minimum set. ER1, written by the
   statistical reviewer after the validity failure and before any D2-D4 label, comparison value or AUC of this grid was read, applies
   the same explanation to minimum-set equalities and reports them with the n_eff they imply. Effect: decisions are computed under
   ER1; the first claims records (NOT_COMPUTED_GRID_INVALID) are kept.
7. **Card FA_FRC_v1 amended after it governed fitting reads (verifier V1.ACCESS_LOG, procedural FAIL; adjudicated 00:31Z).**
   Addendum A1 was appended at 23:56:09Z (b538fa7a4), after the T1 reads (23:48-23:49Z) that logged the card's earlier SHA-256.
   The change is append-only: the logged bytes (research commit 98ede167e) are an exact prefix of the current card and stay in git;
   no operation, rule, budget or control changed. A separate card file would have avoided this; later revisions use new card files
   (FA_FRC_R1.md). The FAIL is kept; the T1 values are unaffected (they reproduce the GATE records exactly).
8. **Units launched with uncommitted files (verifier V1.UNIT_CODE_AT_LAUNCH, ATTENTION).** 23 units, including the P2 fits at
   00:01:16Z, started while files outside their import path were uncommitted (frc/claims.py, frc/evaluate.py, documents). No unit's
   imported package differed from its launch HEAD (the queue records the code digest at launch and at end).
9. **ER1 was written into two registration files, and the access gate then refused every label read (00:32:13-00:52:31Z).** The
   statistical reviewer recorded ER1 inside `roles/statistical_reviewer/RULES.md` and `RULES.json`, two of the eight files of the
   published registration receipt (commit 6d6cd0829, 00:32:13Z). `frc.common.registered` allows a label read only while those files
   equal the receipt, so from that commit on every label read was refused: units FRC_GRID_LC_S0_V1, FRC_GRID_LC_S1_V1,
   FRC_READER_CUSTODY_S0_V1 and FRC_READER_CUSTODY_S1_V1 failed at their first label read (00:49:20-00:49:39Z, `PermissionError: no
   label read before the committed registration and its verified public receipt`). The access log has no read after 00:30Z, and
   the FRC1_ROB grid had read its labels at 00:27:43Z, before the edit. Fix (coordinator, 00:52:31Z): the reviewer's ER1 text moved
   verbatim to `roles/statistical_reviewer/RULES_ER1.md` / `RULES_ER1.json`, and the two files restored to the receipt's bytes
   (SHA-256 2b7e0be8... / a0dc0fc7...). A script first checked that the post-ER1 files are exactly the registered files plus the
   moved text. No rule changed: ER1 lives in `frc/rules.py` (untouched), `frc.test_rules` passes 18/18 after the move, and
   `frc.rules.load_rules` reads the registered RULES.json. The four failed units are kept in the ledger, and their reruns use new unit
   ids (`_V2`). As with deviation 7, registered files are immutable and amendments go in separate dated files. The claims records
   under ER1 (`claims/MAIN_ER1`, `claims/R1_ER1`, computed at 00:32:21-00:32:23Z) cite `rules_sha256` e2a6af23..., the post-ER1
   RULES.json kept at commit 6d6cd0829, not the restored file. ER1 adds no constant (`load_rules` checks only the registered
   constants), so the same run today would cite a0dc0fc7... with identical labels. The statistical reviewer confirmed the move
   changed no rule (`roles/statistical_reviewer/CLOSING_STATEMENT.md`, deviation 9 check).
