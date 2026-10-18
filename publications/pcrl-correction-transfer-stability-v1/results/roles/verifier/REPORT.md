# Independent verification report (verifier agent; saved by the coordinator)

**Attribution.** The independent verifier wrote this text and returned it at the end of its assignment (19:22-21:00Z). The tool
refuses report files from a subagent, so the verifier could not write this file itself. The coordinator saved its returned text
verbatim at 21:03Z. This is not coordinator analysis. The verifier's own records are CHECKS.json, records/CHECK_CHANGES.json,
records/NOTES.jsonl, records/V_*.json, records/history/ and code/ in this folder. The coordinator's dispositions are in
`roles/coordinator/ADJUDICATIONS.md` (A1, A3, A4, A5) and PROTOCOL_DEVIATIONS items 16-17.

---

## Independent verification report: CTS1 PCRL correction transfer (2026-10-18, 19:22–21:00Z)

**Bottom line:** 263 checks, no FAIL that changes a registered outcome. Two genuine findings remain: a post-freeze change to frozen
rule code (F3 V2), and the P70A fold-tag record defect already covered by ERRATA E1. Every other number I replayed matches. Change 4
still needs your adjudication.

| Kind | PASS | FAIL | ATTENTION | NOT_RUN |
|---|---|---|---|---|
| Numerical replay | 212 | 6 | 17 | 0 |
| Procedural compliance | 21 | 2 | 5 | 0 |

Of the 8 FAILs, 3 were bugs in my own checks; each was re-run and passes (C5 D_FRONTIER reader refits V1, C2.2 V3, C6.3 R_COMP0
V1). Changes 1–3 were accepted (A1, A3); change 4 is pending. The other 5 are the two genuine findings below.

### Genuine FAILs
1. **F3_FROZEN_RULES_AND_RECIPE__V2 (procedural): frozen rule code changed after the freeze.**
   - `cts/rules.py` and `cts/questiona.py`, both hashed in REPAIR_FREEZE.json, were changed by commit eece7dbdb at 20:34:05Z
     (deviation 10). The freeze was committed at 20:23:59Z.
   - The question A runs with the frozen code had already read the panels: C0 20:25:19–20:27:53Z, C1 20:28:02–20:29:05Z.
   - The corrected runs then replaced them as the primary labels (20:35:13–20:35:43Z).
   - The correction was motivated by D-stage evidence and the RULES.md text, and the statistical reviewer adjudicated it. It made
     results more conservative: R1 REFIT_LOC full and R1 REFIT_PAIR_TR full went from PREDICTION_IMPROVED to NOT_IMPROVED. The
     claim ladder is unchanged.
   - Deviation 10 should say explicitly that the primary R0/R1 labels come from code changed after the freeze and after the panels
     were read with the frozen code (PROTOCOL section 8).
2. **C2.1 V4/V5/V6 (numerical): P70A record defect.** P70A's WITNESS.json records fold tags `CTS1_cf_P70A` / `CTS1_inner_P70A`, but
   the saved folds are P70's, as registered. ERRATA E1 has the correct tags, and my separate item C2.11 confirms they regenerate the
   saved folds. Kept as FAIL per A3.

### Checks 1–2: pins, roles, exclusion, model identity (PASS apart from the P70A defect)
- **Partitions:** all 18 household sets in PARTITIONS.json reproduce exactly from SHA-256 of household ids, including C0, C1, POOL
  and every D/R role. Overlaps match, roles are disjoint, and the panels touch no FIT or GUARD set.
- **Source admission:** all 28 used release hashes match, a 30-file H3-mirror sample matches, and the registration files and
  receipt are unchanged and precede the first label read.
- **Masks and dictionaries:** OOF masks, row ids and folds equal my reconstruction. DICT.json follows from FIT rows only (and
  differs from a pool or all-MODEL_FIT dictionary). Nested DICT_k equal FIT minus fold k.
- **Bitwise refits on my FIT-only rows:**
  - D0/D1/R0/R1 witness: deployed model plus 5 fold models. A FIT+GUARD control differs by 0.23 / 0.16.
  - D0/D1/R0/R1 REFIT routing reproduce exactly.
  - D0 nested variant: fold witnesses, the fold-0 base and the fold-0 routing teacher.
  - Track R income teacher: routing identical, raw within 1.2e-14.
  - Track R Step-1 HGB: bitwise.
- **Access log:** every line matches my row reconstruction. Panel reads come after the freeze commit; P70/P70A reads come after
  amendment commit 5002fed58.
- **Identity:** 248 artifact hashes, 245 unique. The three aliases are expected deterministic ones, counted once: D0B's routing
  equals D0's, and each R Track R occupation teacher equals its Track F routing.
- **Order:** on D0 and D1, all directions were written before the first GUARD or CHECK read.

### Checks 3–4: costs, rules, zero-correction accounting (PASS)
- All Track F ladders on D0, D1, R0 and R1 (7 estimators, 11 strengths; pre- and post-freeze for R) match within 6.7e-18 on
  statistics and 8.7e-19 on SEs.
- Witness baselines match within 2.1e-16. Men's-guard reserve slack: D0 +0.000493, D1 −0.000779, R0 +0.000142, R1 −0.000306.
- OLD-B80/B50, SIMPLE and UNCERTAINTY selections (full and ens) are reproduced, as are realized CHECK/MS costs, crossing flags and
  outcome_complete.
- Identity selections are recorded as identity with |cost| ≤ 1e-12.
- Gates G1–G3 for all 32 recipes and the NO_RECIPE outcome are reproduced.
- **Question A:** all 34 corrected (REG) labels replay with my implementation of the registered scheme, with identical replicates
  (q05 within 6e-15).
- Compression controls (R0) and Track R income clauses match exactly.

### Checks 5–6: grids, screens, banks (PASS)
- **Grids** (4 banks, all seeds): all paired cells and SEs recomputed exactly. Release arrays equal my recomposition within 4.4e-16,
  including the COMP and H3R/HCR members.
- **Reader refits:** OUT_LR and LOG_LR readers refitted from scratch equal the cached readers exactly.
- **Screens:** the old per-cell gate and AGG1 are reported side by side for 1273 records with stability labels, all matching at
  1e-12. They disagree on 64 records. The one-seed R_COMP0 bank correctly carries no screen labels.
- **Banks:** utility values match within 3.4e-16, and minimum reserve ratios and binding conditions match.
- **Screen study L1:** my replication is within 2.3 Monte Carlo SEs of the recorded rates. At mu = 0 the gate passes 0.19 and AGG1
  passes 0.074.

### Check 7: restored deployment (PASS)
- Replayed R0__REFIT_PAIR@s1.0, R1__W and R0__ANCHOR; all 40 dependency hashes match.
- My encoder equals the admitted encoding, and my own raw-field pipeline reproduces the stored DEV_REPORT occupation wire within
  3.3e-16 (bitwise for W).
- The api entry point returns the stored wires bitwise (income equals the admitted H3 release) and exports only probability and
  decision.
- Refusals hold at all three entry points: api (SEX field, hidden output), csv (SEX column, reordered columns) and underlying (SEX
  column, reordered columns).

### ATTENTION items (no registered outcome changes)
- **D1/R1 identity outcomes** (10 checks): they carry reserve_crossing = True because the men's reserve slack is negative. This
  explains all 16 D1 G2 failures; G1 fails there anyway.
- **Question A bootstrap** (6 checks): the original scheme understated the SE by 7–25% and flipped 3 labels. Resolved by deviation
  10.
- **Amendment dating** (P2; you already disclosed it as deviation 8).
- **New provenance item** (P3 V2, not covered by deviation 7): CTS_R0_DIR_REFIT_LOC_V1 launched at 20:12:55Z, 30 s before its
  trackf.py change was committed. The code it ran equals commit 5002fed58.
- **COMP matching:** the utilization path over K is non-monotone, so "MATCHED" codes reach only 6–55% of the target utilization
  (s0.2: 0.23 against 3.80). For s0.1, TARGET_NOT_REACHED rests on K = 2048 alone.
- **RESEARCH_DECISION wording:**
  - "Panel strengths mostly overspent the reserve" is 16 of 22 on C0, but all 13 ens selections cross, and so does the uncorrected
    ensemble e_0 against W (slack use 4.0). For the full deployment it is only 3 of 9.
  - "NOT_IMPROVED on all four partitions" is true for REFIT_PAIR only; FROZEN_PAIR on D1 ens is PREDICTION_IMPROVED, so please name
    it in the scope.
- C2.1 V3 had verifier errors and is superseded by change 2.

### Check changes (CHECK_CHANGES.json)
1. My grid reader-refit check fed its own recomposed array instead of the stored bytes. Accepted (A1).
2. My partition reconstruction was missing the amended partitions P70/P70A. Accepted (A3).
3. A misleading P70A fold item of my own; the C2.1 FAIL is kept. Accepted (A3).
4. My screen check computed gate and AGG1 on a one-seed bank, where RULES.md section 10 defines no screen. **Pending your
   adjudication.**

None of these widens a criterion.

### Not covered
- The torch-trained Stage B and Track F directions were not refitted; their exclusion rests on the access log, code reading and the
  delta_oof / fold-correction consistency.
- The H3 income lineage was not rebuilt (only compared against the admitted arrays), and laws L0b and L2 of the screen study were
  not replayed.
- The source campaign's other checkout was not read.

### Resources
50 queue units, none failed. 301 CPU-s in the queue plus 32 CPU-s charged for light out-of-queue work: about 0.09 CPU-h in total.

Files are in `results/pcrl_correction_transfer_stability_v1/roles/verifier/`:
- CHECKS.json
- records/CHECK_CHANGES.json
- records/NOTES.jsonl
- records/V_*.json (one per run)
- records/history/ (64 CHECKS snapshots plus CHECK_CHANGES snapshots)
- code/: vlib.py and 18 v_*.py scripts

---

## Addendum (verifier follow-up on adjudication A5, returned at about 21:11Z; saved verbatim by the coordinator at 21:12Z)

## Verifier follow-up on the F3 V2 ruling (A5): condition 1 holds, and the R1 label shifts come from the bootstrap scheme, not heavy tails

### 1. Condition 1: the question-A finding used only D-stage inputs (P4_QA_CORRECTION_TRIGGER_INPUTS__V2: PASS)

**The send time in my message header is wrong.** I wrote "20:29Z" as an estimate. The message actually went out between 20:27:28Z
(the last record written before it) and 20:28:04Z (my next clock reading). Please use those bounds in the deviation timeline
instead of "20:29Z".

**Code basis.** I first noticed the per-statistic resampling by reading `cts.rules.prediction_comparison`, before any question-A
output existed.

**Queue units behind the finding (all D only):**

| Unit | Start | End (UTC) |
|---|---|---|
| VER_QA_D0_REFIT_PAIR_FULL_OOF_V1 | 20:19:19 | 20:19:20 |
| VER_QA_D0_ALL_V2 | 20:24:04 | 20:24:11 |
| VER_QA_D1_ALL_V2 | 20:24:11 | 20:24:18 |
| VER_QA_BOOT_D0_V1 | 20:25:44 | 20:26:49 |
| VER_QA_BOOT_D1_V1 | 20:26:51 | 20:27:28 |

**Label reads of those units.** There were 15 reads, all operation `verify` on MODEL_FIT, partitions D0/D1, roles
FIT/GUARD/CHECK, between 20:19:19Z and 20:26:52Z. None touched a repair panel.

**Inputs.** The finding used 60 campaign files from D0/D1 only, all listed by SHA-256 in `records/V_TRIGGER__V2.json`:
- the superseded QUESTION_A records;
- the SELECT baselines;
- WITNESS.npz and OOF.npz;
- TRAIN.npz and DELTA.npz of the directions;
- the NESTED npz files.

None was modified after 20:22:43Z, so the hashes are the bytes that were read. The 20 verifier records that carried the finding
(V_QUESTIONA D0/D1 V1/V2, V_QA_BOOT D0/D1 V1) are listed with their hashes and the code hashes at run time.

**R reads before sending.** No R0/R1 output and no panel read entered the finding. My only R-partition reads before sending were
the R0/R1 FIT and GUARD reads of the witness and routing refit checks (20:21:04-20:21:50Z), which were not used by the finding. My
first panel read was VER_COSTS_R0_V1 at 20:28:37Z, after the message.

**Disclosure for the ruling (P4b, ATTENTION).**
- Between my message and the correction commit (eece7dbdb, 20:34:05Z), my units read C0 (20:28:38Z, 20:28:56Z) and C1 (20:33:26Z,
  20:33:29Z, 20:33:50Z).
- The R0 question-A replays were written to CHECKS.json at 20:29:04Z; they showed no scheme dependence.
- The R1 replays wrote the two scheme-dependent R1 labels into CHECKS.json at 20:33:36Z, 29 s before the commit.
- My message about R1 went out after 20:34:34Z, i.e. after the commit.
- I cannot know whether anyone read CHECKS.json in those 29 s.

### 2. Variance detail for the two scheme-dependent R1 units (C3.QA.R1.VARIANCE_DETAIL__V3, descriptive)

My code uses the registered tag and B = 2000 and replays both recorded schemes exactly.

| Unit | D | Registered: SE / q05 | Superseded per-statistic: SE / q05 |
|---|---|---|---|
| REFIT_LOC full OOF | 0.5420 | 0.2502 / −0.0203 | 0.1869 / +0.1053 |
| REFIT_PAIR_TR full OOF | 0.4842 | 0.2403 / −0.0198 | 0.1912 / +0.0502 |

**The q05 shift is explained by the SE change, not by heavy tails.**
- The per-statistic SE is 25% and 20% smaller. In q05 units that is z × ΔSE = 0.104 and 0.081, against observed shifts of 0.126
  and 0.070.
- Both bootstrap distributions sit well below D: the bootstrap mean is 0.405 and 0.367 under either scheme. This is the upward
  bias of the absolute errors, which is larger for the smaller GUARD and CHECK sets. So q05 is about 0.14 below the normal
  approximation D − z·SE under both schemes.
- No NLL increment is near the 1e-12 floor. The largest absolute NLL increment is 1.39 nats, and 0-19 persons exceed 1 nat, out of
  2,997-10,506 per set.

**Household influence (linearized D).**
- The top 10 households carry 5.7% and 4.9% of the variance.
- The linearized SE (0.249 and 0.243) matches the registered bootstrap SE.
- The top households are CHECK households for LOC (women's Brier and NLL) and mostly GUARD households for PAIR_TR (accuracy loss);
  they contain 1-4 people each.

Dropping the most influential households leaves the registered label at NOT_IMPROVED in every case:

| Unit | Drop top 1: D / q05 | Drop top 5: D / q05 | Drop top 10: D / q05 |
|---|---|---|---|
| REFIT_LOC | 0.511 / −0.038 | 0.507 / −0.046 | 0.534 / −0.023 |
| REFIT_PAIR_TR | 0.461 / −0.057 | 0.462 / −0.061 | 0.412 / −0.077 |

### Records and housekeeping

- CHECKS.json now has 269 checks:

  | Kind | PASS | FAIL | ATTENTION |
  |---|---|---|---|
  | Numerical replay | 212 | 6 | 19 |
  | Procedural compliance | 23 | 2 | 7 |

- **New check change 5, pending your adjudication.** The evidence list of P4 V1 wrongly included the later REG replay records; the
  status did not depend on it. Corrected in V2 with an unchanged criterion. Change 4 is also still pending.
- **Failed unit kept.** VER_QA_INFLUENCE_R1_V1 failed because my script asked for partition role 'OOF' instead of FIT. The access
  layer refused it before any label was read; it is noted in NOTES.jsonl and V2/V3 are the reruns.
- Verifier CPU in the queue is now 332 s, plus 32 s charged for out-of-queue work.

Files are in `results/pcrl_correction_transfer_stability_v1/roles/verifier/`:
- CHECKS.json
- records/V_TRIGGER__V1.json, records/V_TRIGGER__V2.json
- records/V_QA_INFLUENCE__R1__V2.json, records/V_QA_INFLUENCE__R1__V3.json
- records/CHECK_CHANGES.json
- records/NOTES.jsonl
- code/v_trigger.py, code/v_qa_influence.py
