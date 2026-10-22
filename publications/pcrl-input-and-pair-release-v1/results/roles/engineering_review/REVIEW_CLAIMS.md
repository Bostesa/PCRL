<!-- Saved verbatim by the coordinator from the engineering and claims reviewer's hand-back (agent aaaebc018362b7c8f, started about
04:55Z with roles/coordinator/CLAIMS_REVIEW_BRIEF.md), received 2026-10-22 shortly after 05:30Z. The reviewer was asked to return its
report rather than write a file; the text below is unedited apart from the harness's two-space line indentation, which is removed. -->

# Claims review (engineering and claims reviewer, campaign `pcrl_input_and_pair_release_v1`)

Written 2026-10-22 between 04:55:38Z and 05:30:19Z (both times read with `date -u`). It follows `roles/coordinator/CLAIMS_REVIEW_BRIEF.md`. The author is the engineering and claims reviewer, an agent. The report is returned as text and was not written to the repository.

**What was reviewed.** The documents changed while I worked, so the review covers two states.
- **First pass:** the drafts committed at 5c497a958 (04:53:48Z).
- **Re-based pass:** the uncommitted working-tree rewrite that applies the statistical reviewer's closing statement (files modified 05:15-05:17Z). SHA-256 prefixes at 05:18:24Z, unchanged at 05:29:41Z:
  - START_HERE (SH) 3b8588c63aac
  - RESEARCH_DECISION (RD) 88555be53f86
  - ADVISOR_BRIEF (AB) 6c65ab1db194
  - HANDOFF (HO) 59e756f3f006
  - VERIFICATION (VER) 0eb7cc431d87
  - PROTOCOL_DEVIATIONS (DEV) 8052f19f83ad
  - QUICKSTART (QS) 52ece4304c09 and TABLES are unchanged since the commit.
- Line numbers below refer to these working-tree files.

**Records read:**
- Everything the brief lists that exists: `claims/*` (DECISIONS, D2R, STRONGEST, BESIDE_*, CAMPAIGN_SUMMARY, PLAN_*, superseded/), `paths/SHARED_ROLES.json`, `ia/`, `ib/`, `rich/`, `deploy/DEPLOYMENT_CHECK.json`, `dev/UTILITY_LEDGER.csv`, the cards, PROTOCOL, AMENDMENT_1, DEVIATIONS, FS_SWAP_DECISION and `roles/challenger/`.
- Both verifier reports and the verifier's records (V3_GRID G/R, the addendum's A4_GRID), and `CLOSING_STATEMENT.md` (CS; it appeared at 05:13Z).
- `custody/READER_CUSTODY_*.json`, `grid_specs/`, `lanes/`, and `git log` / `git ls-tree` of the research and publication branches.
- Private files, metadata and aggregates only: the queue ledger, `runs/CONFIG_LEDGER.jsonl`, `access/LABEL_ACCESS.jsonl` (metadata only), `START_UTC.json`, `REGISTRATION_RECEIPT_VERIFIED.json`, and the aggregate `EVAL_MS.json` of IA_OSG, IA_SFG and IB_CSF.

## 0. Summary

- **Labels, decisive numbers and tables.** Every D-label, outcome class and campaign count matches its record, and so do the headline (verbatim from CS section 2) and nearly every number.
  - The tables regenerate identically with QUICKSTART's command (section 5).
  - The brief's own command is stale: it omits the revision bank.
- **The rewrite.** It applied CS C1-C19 and fixed most first-pass problems (section 7).
- **Still open: 23 required corrections (R1-R23).** The most important:
  - **R1:** HANDOFF's "What can be said" asserts the recovery-through-income mechanism that its own "What must not be said" forbids.
  - **R2:** "every other reader shows no difference" is contradicted by the verifier's own validation-chosen reader.
  - **R3:** two refit-cell numbers have no record.
  - **R4:** "None was triggered" is false for IA_SFG_v1 R2, whose trigger held.
  - **R5:** IA_OSG's missed reserve is attributed to occupation NLL, but the men's occupation Brier guard binds it.
  - **R7:** VERIFICATION says the independent readers agree in direction "for every comparison", but the HGB reader has the opposite sign in three.
- **Strength.** No claim is stronger than the rules allow, apart from R1, R2, R11 and the advisory wording in A1, A5 and A7.

## 1. Required corrections

**R1. HANDOFF states as sayable what it later forbids.**
- **Quote:** HO 18-19: "**Recovery through the pair.** IA's adaptive revision cut the occupation view alone (0.6351 against 0.6443), but the linked pair recovered it through the unchanged income vector (+0.0017, se 0.0062, against the twin)."
- **Records:**
  - HO 57-58 in the same file: "Not that the linked pair "recovers through the income vector" as a finding. That is a reading of point estimates...".
  - CS 4.2.
  - `claims/IA_R1/DECISIONS.json`: the TWIN comparison is AGG_POSITIVE_POINT_ONLY. No record tests the mechanism.
- **Corrected:** "**The pair did not follow the occupation view.** IA's adaptive revision lowered the occupation view alone at the point (0.6351 against 0.6443), but the linked pair hardly moved (+0.0017, se 0.0062, against the twin). As point estimates this is consistent with recovery through the unchanged income vector; it is not a resolved effect."

**R2. "Every other reader" overstates the evidence on IA_SFG's income view.**
- **Quotes:**
  - RD 107: "**Every other reader disagrees:**"
  - SH 94-95: "...rests on one repetition in which the selected reader switched; every other reader shows no difference."
  - AB 78-79: "Every other reader shows no difference, so this is not evidence that it leaks more through income."
  - HO 47: "The verifier's, the data-rich and the MODEL_FIT readers show no difference."
- **Records:**
  - VERIFIER_REPORT O4: "my validation-chosen reader gives +0.034 because the chosen reader switches".
  - `roles/verifier/records/V3_GRID__IPR_VER_GRID_R_V1.json`, IA_SFG − CTL_TWIN on the income view:
    - LR: mean +0.0003 (S1R1 +0.0011);
    - HGB: mean +0.0029 (S0R1 +0.0099, S1R2 +0.0063, S1R1 +0.0002);
    - validation-chosen: mean +0.0085 (S1R1 +0.0337).
  - CS C2 names only the fixed LR and HGB readers in the S1R1 cell. "Every other reader" is the documents' generalization.
- **Corrected:**
  - RD 107 becomes "**What the other readers show:**", with the first sub-bullet: "in that repetition the verifier's fixed LR and HGB readers give +0.001 and +0.0002, while its own validation-chosen reader reproduces the switch (+0.034, verifier O4)".
  - SH 95 and AB 78-79: "...the verifier's fixed LR and HGB readers in that repetition, the data-rich readers and the MODEL_FIT readers show no comparable difference."
  - HO 47 likewise.
  - Optional support: in the same repetition IB_CSF's income view is +0.021 against the twin in both cells (`V3_GRID__IPR_VER_GRID_G_V1.json`). That is consistent with a weak twin income winner there.

**R3. Two refit-cell numbers have no record.**
- **Quote:** RD 151-152: "Its SPEND path has no bracket. Descriptively, against its two occupation-only quantization points: +0.0019 (0.0060) and +0.0009 (0.0056)."
- **Records:**
  - `claims/IA_R1_D2R/D2R.json` per_comparator holds only TWIN, IA_OSG_SBC_HI, IA_OSG_RO_HI, NEURAL_RAW and H_LJ.
  - `PLAN_IA_R1_D2R.json` has bracket_ids SPEND null.
  - No committed JSON under `results/` holds these values. I searched all 97 campaign JSON records for a mean/se pair matching either number.
  - TABLES.json has the registered cells only: −0.0008 against each point.
  - The figures appear to come from an unsaved scratch computation (`osg_spend_rob.py` in the shared scratchpad, 04:52Z).
- **Corrected:** delete the sentence. Alternatively, save the computation as a descriptive record written by a committed script, cite it, and say which value is IA_OSG_SPD_HI and which IA_OSG_SPD_LO.

**R4. The declared IA revisions.**
- **(a) Quote:** RD 229-232: "**The revisions declared in the cards.** None was triggered (coordinator log, 02:47Z; MODEL_FIT records): ... IA_SFG_v1 R2 (finer splits): the binding step has no further predeclared half, so it cannot change the release;"
- **(a) Records:**
  - COORDINATOR_LOG 02:47Z: "IA R2's trigger holds in both versions (the next rung overshoots its binding component by 95% / 69%)".
  - Card IA_OSG_R1, lines 15-16: "its trigger holds in both versions".
  - `ia/FIT_V0.json` / `FIT_V1.json` selection path: income accuracy at the next rung is 1.953 / 1.692 of S3.
- **(a) Corrected:** "None was run (...): ... IA_SFG_v1 R2 (finer splits): its trigger held in both versions (the next rung overshoots its binding component by 95% / 69%), but the binding step, marital-status suppression, has no predeclared half beyond MAR2, which the menu already contains, so it could not change the release;"
- **(b) Quotes:**
  - RD 230-231: "its trigger failed in version 0 (the women's occupation guard keeps 45% of its slack against the 50% required)"
  - DEV 60: "(45% against 50% slack)" (CS C9's wording).
- **(b) Record:** `ia/FIT_V0.json` selection.path rung 2 (estimate + 2 SE over S3):
  - occupation accuracy 0.629, so 37% slack;
  - women's occupation Brier 0.547, so 45% slack.
  - Card IA_OSG_R1 F1 itself gives "0.63".
  - The component is the MODEL_FIT S3 budget component, not the G1 guard.
- **(b) Corrected:** "its trigger failed in version 0: two occupation components of the MODEL_FIT budget keep less than 50% of their slack at the next rung (occupation accuracy 37%, women's occupation Brier 45%)". Make the same change in DEV 60.

**R5. IA_OSG's missed reserve is misattributed.**
- **Quote:** RD 300-301: "Its utility cost grew from MODEL_FIT to MODEL_SELECT (version 1: 0.0070 to 0.0120 occupation nats), so it misses the reserve in version 1."
- **Records:**
  - IA_OSG `EVAL_MS.json` reserve_rules: binding `s1|guard|sex0`, version-1 ratio 1.394.
  - Verifier A3.4: "V1 men's-guard ratio 1.394 < 2".
  - The paired men's occupation Brier cost against the twin grew from 0.0026 (MODEL_FIT; `ia/FIT_R1_V1.json` saved.IA_OSG occ_brier_men, se 0.0007) to 0.0049 (MODEL_SELECT; EVAL_MS, se 0.0011).
- **Corrected:** "Its costs grew from MODEL_FIT to MODEL_SELECT (version 1: occupation NLL 0.0070 to 0.0120 nats; men's occupation Brier 0.0026 to 0.0049), and the men's occupation Brier guard binds its reserve in version 1 (ratio 1.39)." The same wording is needed in CS 1.4 / C5 (section 7).

**R6. The IA_OSG SPEND explanation is imprecise.**
- **Quotes:**
  - RD 129-130: "IA_OSG spent more than both occupation-only quantization points because its spend grew from MODEL_FIT to MODEL_SELECT (version 1: 0.0070 to 0.0120 occupation nats)."
  - AB 85-86: "Its spend grew from MODEL_FIT to MODEL_SELECT, so it spent more than both occupation-only quantization points..."
- **Records:**
  - `claims/IA_R1/DECISIONS.json` SPEND_FRONTIER primary_cost_differences:
    - vs IA_OSG_SPD_HI: +0.00185 (se 0.00184; V0 −0.00003, V1 +0.00373);
    - vs IA_OSG_SPD_LO: +0.00435 (se 0.00166).
  - IA_OSG_SPD_HI's version-1 spend fell from 0.0104 on MODEL_FIT (`ia/FIT_R1_V1.json` fineness_inputs SPEND upper) to 0.0082 on MODEL_SELECT (TABLES section 8).
- **Corrected:** "Pooled over versions, IA_OSG's MODEL_SELECT spend exceeds both points (by 0.0018, se 0.0018, and 0.0043, se 0.0017; in version 0 it is 0.00003 nats below IA_OSG_SPD_HI). Its version-1 spend grew from 0.0070 (MODEL_FIT) to 0.0120 nats, while IA_OSG_SPD_HI's fell from 0.0104 to 0.0082." Shorten the same way in AB.

**R7. VERIFICATION overstates the independent-reader agreement.**
- **Quote:** VER 30-31: "They agree in direction and size with the grid for every comparison."
- **Records:**
  - VERIFIER_REPORT check 3.5 table.
  - `V3_GRID__IPR_VER_GRID_R_V1.json`: the HGB reader's sign is opposite to the grid's for IA vs TWIN (−0.0005 against +0.0016), IA vs IA_SPD_HI (−0.0055 against +0.0006) and IB vs IB_SPD_HI (−0.0092 against +0.0007).
- **Corrected:** "They are consistent with the grid within its uncertainty (the verifier's verdict, check 3.5). The LR and validation-chosen readers agree in sign with the grid for all 11 comparisons. The HGB reader has the opposite sign for three point-only comparisons (IA vs TWIN, IA vs IA_SPD_HI, IB vs IB_SPD_HI), each within about 1.2 se_agg."

**R8. VERIFICATION cites a backup check that does not exist yet.**
- **Quote:** VER 73-74: "**Deployment packages:** checked by `deploy/DEPLOYMENT_CHECK.json` and by the restore check run from the backup (`custody/BACKUP_VERIFICATION.json`)."
- **Record:** `custody/` holds only the two READER_CUSTODY files. SH 79 marks the backup record [PENDING].
- **Corrected:** "...checked by `deploy/DEPLOYMENT_CHECK.json`; the restore check from the backup is [PENDING: `custody/BACKUP_VERIFICATION.json`]."

**R9. "Stated beside the RO comparison wherever it is reported" is not yet true.**
- **Quote:** VER 55.
- **Records:** RD 149 ("its occupation-restricted release-only tilt -0.0057 (0.0066)") and AB 97 ("IA_OSG vs its occupation-restricted release-only tilt: -0.0057") omit the G1 failure of IA_OSG_RO_HI (verifier addendum O1).
- **Corrected:** add "(against IA_OSG_RO_HI, which fails G1 in version 1)" at RD 149 and AB 97.

**R10. ADVISOR_BRIEF miscounts the guards.**
- **Quote:** AB 59-60: "...summed Brier excess <= 0.005, and four occupation subgroup guards, in each version."
- **Records:**
  - PROTOCOL 3: 24 clauses plus "the men's and women's occupation summed-Brier excess vs U0 <= 0.005 (4 guards)"; G1 is all 28.
  - `dev/UTILITY_LEDGER.csv`, column point_pass_12_plus_2.
  - RD 72 and SH 23 already say "12 clauses and 2 guards in each version".
- **Corrected:** "...accuracy loss <= 0.010, NLL excess <= 0.010 nats and summed Brier excess <= 0.005 against U0 and Ucal for both tasks (12 clauses), and the men's and women's occupation summed-Brier guards (excess vs U0 <= 0.005), in each version: 24 clauses and 4 guards in all."

**R11. "Worse than both" is not resolved for one comparison.**
- **Quote:** AB 112-113: "every candidate is worse than both (best -0.0158 / -0.0225, IB_CSF)."
- **Record:** `claims/IB_REG/DECISIONS.json` / TABLES: IB_CSF vs NEURAL_RAW is −0.0158 (se 0.0095), with upper bound +0.0032, so it is not resolved. AB 107 itself says "resolved in five of the six".
- **Corrected:** "every candidate is more recoverable than both at the point, resolved in five of the six comparisons (best -0.0158 / -0.0225, IB_CSF; the first is not resolved)."

**R12. Mis-rounded standard errors.**
- **Quote:** RD 76: "**IA_OSG** spends occupation NLL +0.0041 (se 0.0011) / +0.0120 (0.0018)."
- **Record:** IA_OSG `EVAL_MS.json` paired costs against CTL/CTL_TWIN, occupation|nll: 0.0040596 (se 0.0010470) and 0.0119529 (se 0.0017477). The verifier's A3.2 replays them exactly.
- **Corrected:** "+0.0041 (se 0.0010) / +0.0120 (0.0017)".

**R13. A mis-rounded bound.**
- **Quote:** RD 174: "(upper 2-SE bound +0.0008)".
- **Record:** `claims/BESIDE_IPR1_MAIN.json` IB_CSF__vs__DDR_PTC_R1: mean −0.0171893 and se_agg 0.0090270, so the upper bound is +0.000865.
- **Corrected:** "(upper 2-SE bound +0.0009)".

**R14. Campaign-level counts without the frozen statement.**
- **Quotes:**
  - SH 40: "**The count.** D2 was met in 0 of 2 registered and 0 of 1 adaptive readings, 3 looks in total."
  - HO 14-15: "...0 of 2 registered and 0 of 1 adaptive readings, 3 D2 looks in total."
- **Records:** RULES.md 7 ("the statement every campaign-level sentence must carry"); `claims/CAMPAIGN_SUMMARY.json` statement. The RD headline and AB Q2 carry it.
- **Corrected:** append to both "; one AGG_POSITIVE comparison passes a true null in about 0.07-2% of replicates under six moderate NRM1 known laws and about 8% when one version or split dominates, and with 3 looks the chance that at least one passes by chance is at most 3 times the per-look rate."

**R15. Adaptive labels without their prefix.**
- **Quotes:**
  - The RD 41-48 table, IA_R1 column: "TASK_FEASIBLE", "NOT_SEPARATED (flags ...)", ..., "TASK_ONLY_FEASIBLE_NO_PROTECTION (adaptive)".
  - SH 39: "D2: NOT_SEPARATED."
  - AB 82: "**IA_R1 (OSG; ADAPTIVE_R1; never IA's result), NOT_SEPARATED:**"
- **Records:**
  - `claims/IA_R1/DECISIONS.json`: every label is `ADAPTIVE_R1__...`.
  - PROTOCOL 6: the tag goes "as a prefix of every label"; RULES.md 7 says the same.
  - TABLES.md section 3 and the headline (`ADAPTIVE_R1__NOT_SEPARATED`) already use the prefix.
  - Precedent: DDR C20.
- **Corrected:** write the recorded labels (`ADAPTIVE_R1__TASK_FEASIBLE`, `ADAPTIVE_R1__NOT_SEPARATED`, ..., `ADAPTIVE_R1__TASK_ONLY_FEASIBLE_NO_PROTECTION`). In SH 39: "D2: ADAPTIVE_R1__NOT_SEPARATED."

**R16. Quoted labels without the reading's scope sentence.** Low severity; CS did not raise it.
- **Quotes:**
  - The RD table's D2 and outcome-class rows (IA_REG and IA_R1 carry no scope).
  - SH 25-26: "IA_SFG and IB_CSF are both NOT_SEPARATED ... and both are TASK_ONLY_FEASIBLE_NO_PROTECTION."
- **Records:**
  - PROTOCOL 6.2: "Every sentence quoting a D2 label or the outcome class repeats the reading's scope sentence".
  - RULES.md 6: ro_scope.
  - The D2.scope field in each DECISIONS.json.
  - IA's GENERIC_ZTILT scope appears only at SH 98-99 and RD 327. IB's second half ("no claim about SEX information beyond the twin's release") appears nowhere.
- **Corrected:** add one note under the RD table: "Scope (D2.scope): IA_REG and IA_R1: the RO controls are the generic release-only tilt of the twin, not the operation itself; the same operation driven only by the twin's release was not built. IB_REG: SEX-blind; SBC and RO are degenerate by construction; D2 was assessed against the twin and the SEX-blind spending path only; no SEX-targeting claim and no claim about SEX information beyond the twin's release is possible." At SH 26, add "(scopes in RESEARCH_DECISION.md)".

**R17. The tables do not hold every number.**
- **Quote:** SH 56: "`TABLES.md` / `TABLES.json`: every number, generated by `ipr.report` from the saved records."
- **Records:**
  - `ipr/report.py` writes utility, AUCs, comparisons, decisions, data-rich readers, D2R records, strongest, beside, spend and summary.
  - It does not hold MODEL_FIT statistics (`ia/`, `ib/`), utility SEs and reserve bindings (EVAL_MS), the verifiers' numbers or the CS power figures.
  - The figures in R3 are in no record at all.
- **Corrected:** "`TABLES.md` / `TABLES.json`: the utility, recovery, comparison, decision, data-rich, robustness, strongest-release, beside, spend and summary tables, generated by `ipr.report` from the saved records (MODEL_FIT statistics are in `ia/` and `ib/`)."

**R18. Wrong time for IPR1_REFS.**
- **Quote:** SH 75: "IPR1_REFS was pre-written at 01:47Z".
- **Record:** `grid_specs/IPR1_REFS.json` has written_utc 01:42:53Z. It was committed in 091d0c88f at 01:47:39Z.
- **Corrected:** "IPR1_REFS was pre-written at 01:42:53Z (committed 01:47:39Z, 091d0c88f) and not run on its own".

**R19. The evidence index omits reader custody.**
- **Quote:** SH 79: "| Custody | `custody/BACKUP_VERIFICATION.json` [PENDING] |"
- **Records:** `custody/READER_CUSTODY_IPR1_MAIN_S0R2k0.json` and `..._IPR1_R1_S1R1k0.json` (all_ok; 5 refitted readers reproduce the grid's scores bitwise); deviation 10; commit 508a7c3c2. No document mentions reader custody or deviation 10.
- **Corrected:** "| Custody | reader custody `custody/READER_CUSTODY_*.json` (5 refitted winner readers reproduce the grid bitwise; the first attempt failed on a composite recipe, deviation 10); backup `custody/BACKUP_VERIFICATION.json` [PENDING] |".

**R20. Publication stated as done.**
- **Quote:** SH 82-83: "It is published under `publications/pcrl-input-and-pair-release-v1/code/ipr/` at the publication commit recorded in `PUBLICATION_RECEIPT.json`;"
- **Record:** `PUBLICATION_RECEIPT.json` does not exist. `publication/pcrl-input-and-pair-release-v1` (local and origin) is e909fb8c0, the registration receipt.
- **Corrected:** "[PENDING: published under ... at the publication commit recorded in `PUBLICATION_RECEIPT.json`]; the registration receipt e909fb8c0 holds only the registration-time code." Precedent: DDR R20.

**R21. QUICKSTART names the failed reader-custody lane.**
- **Quote:** QS 60: "reader custody: `lanes/READER_CUSTODY.json` (`ipr.readercustody run`);"
- **Records:** queue ledger: IPR_READER_CUSTODY_MAIN_V1 failed (04:51:04-05Z) and its lane stopped. The successful lane is READER_CUSTODY2.json (04:51:42-47Z). See deviation 10.
- **Corrected:** "reader custody: `lanes/READER_CUSTODY2.json` (`ipr.readercustody run`; the first lane, `lanes/READER_CUSTODY.json`, failed on a composite winner recipe, deviation 10);"

**R22. QUICKSTART's summary commands fail as written.**
- **Quotes:**
  - QS 41-42: "...and `python3 -m ipr.claims summary`."
  - QS 54: "`python3 -m ipr.claims summary --decisions claims/IA_REG claims/IB_REG claims/IA_R1 --out claims/CAMPAIGN_SUMMARY.json`"
- **Records:**
  - `ipr/claims.py`: line 350 requires `--decisions` and `--out`; lines 336-339 open the paths as given.
  - QS 4: commands run from the repository root.
  - `claims/CAMPAIGN_SUMMARY.json` "readings" are `results/pcrl_input_and_pair_release_v1/claims/...`.
- **Corrected:** use `--decisions results/pcrl_input_and_pair_release_v1/claims/IA_REG results/pcrl_input_and_pair_release_v1/claims/IB_REG [results/.../claims/IA_R1] --out results/pcrl_input_and_pair_release_v1/claims/CAMPAIGN_SUMMARY.json` in both places.

**R23. QUICKSTART lists a lane that was not run on its own.**
- **Quote:** QS 38: "**MODEL_SELECT utility:** `lanes/REF_EVAL.json`, `lanes/EVAL_ALL.json` ..."
- **Records:** EVAL_ALL.json contains REF_EVAL.json's four units (IPR_REF_EVAL_V1 and IPR_DDR_EVAL_*). Each unit ran once, in EVAL_ALL's order (ledger 03:10:11-03:10:17Z).
- **Corrected:** "`lanes/EVAL_ALL.json` (...; it includes the four units of the pre-written `lanes/REF_EVAL.json`, which was not run on its own)".

## 2. Claims stronger than the rules allow (current files)

| Watch item | Status |
|---|---|
| Partial D2 outcome presented as a result | Clean. None occurred; all three D2 labels are NOT_SEPARATED. |
| Point-only or unresolved differences called reductions | Mostly clean ("no point-only difference is a reduction", SH 93). Residues: R1 (a mechanism stated as a finding), R11, and "cut the occupation view" at SH 36 and HO 18 (A5). |
| "Protection" beyond the declared readers | Clean (RD 6-7, SH 102-104, AB 10, HO 30). |
| SEX-targeting or release-only claim for IB | Clean (SH 96-97, AB 81, HO 33). IB's full scope is not repeated (R16, A11). |
| IA_OSG as IA's result; refit or beside comparisons as D2 | Clean in substance ("never IA's result"; D2R marked descriptive). Labels are unprefixed (R15), and the refit SPEND-point figures have no record (R3). |
| IA's RO comparison read as an information bound | Clean (SH 98-99, HO 44, RD 327). |
| Novelty | Clean: D6 NOT_CLAIMED; antecedents from the cards; "no literature search" stated. |
| Joint design | Clean (D5 per reading, RD 192-198). HO 38 wording (A4). |
| The reserve | IA_OSG's POINT_FEASIBLE_NOT_RESERVE_QUALIFIED is stated (RD 77-78, HO 52), as are the infeasible RO_HI bracket and the non-qualified SBC_HI. The cause of the miss is wrong (R5). |

## 3. User-required report items

| Item | Present and separate? | Open |
|---|---|---|
| Task feasibility | Yes (RD Q1, AB Q1, SH) | R5, R10, R12 |
| Improvement beyond matched controls | Yes (RD Q2, AB Q2, SH) | R2, R6, R15, R16, A1 |
| Robustness to classifier refitting | Yes (RD Q3, AB Q3) | R3, R9, A9 |
| Competitiveness against strong references | Yes (RD Q4, AB Q4) | R13 |
| Material criterion (0.020, kept apart from smaller gains) | Yes (RD Q5, AB Q5, HO 62-65) | R11 |
| Joint-design evidence | Yes (RD Q6, AB Q6) | A4 |
| Novelty | Yes (RD Q7, AB Q7) | none |
| Favourable and adverse evidence | Yes (RD); AB defers to RD | R5, A6, A8 |
| Actual resources | [PENDING] in SH, RD and AB | To date, from the queue ledger at 05:06:42Z: 69 units ended (1 failed), 14,205 CPU-s in units plus 410 CPU-s charged, about 4.06 CPU-h; 39 of 48 configurations. |
| Untested questions | RD yes; AB and HO [PENDING] | A12; the gap in section 4 |
| The public commit | [PENDING] | R20 |

## 4. What was not run, and why

- **FS_SWAP:** not run; the reason is stated (RD 223-228, AB 45-46; decided at 02:45Z, before any MODEL_SELECT read).
- **Declared revisions:**
  - IA_SFG_v1 R1, IB_CSF_v1 R1 and IB_CSF_v1 R2 are stated with their reasons (R4b fixes a wording).
  - IA_SFG_v1 R2's trigger held, and the documents misdescribe it (R4a). Its true reason, that it could not change the release, is stated.
- **No IB revision:** stated (RD 244-247).
- **Gap: IA's second revision slot.** The pair-aware ladder is listed as untested (RD 315-319), but no reason is given for not running it. PROTOCOL 7 allows two IA revisions, and 9 configurations remained. State the coordinator's actual reason.
- **Not mentioned at all:**
  - IA_OSG has no deployment package (A10).
  - The challenger's input-PRAM idea (A12).
  - Off-device custody is marked [PENDING] (HO 85); YOTUO was not attached at 04:37:17Z.

## 5. Table regeneration

- **Before:** at 05:10:09Z I copied TABLES.md and TABLES.json aside. Their SHA-256 (d1b2935e..., 3ffc52a4...) equals the HEAD blobs.
- **Method:** I ran `ipr.report` through a wrapper that points `ipr.common.RESULTS` at a scratch copy of the results folder, so the campaign files were never written. Grids and EVAL_MS records were read, read-only, by the generator.
- **The brief's command** (`--main IPR1_MAIN --extra-bank IPR1_ROB`, 05:11:30Z) gives 25 releases, 120 comparisons and 3 readings.
  - It lacks the 8 IA_OSG* release rows, the 24 IA_OSG / IA_OSG_BLIND comparison rows and the 8 IA_OSG* spend rows.
  - Sections 3-7 and 9 are identical.
  - The brief's command predates the revision.
- **QUICKSTART step 9's command** (adding `--extra-bank IPR1_R1 --extra-bank IPR1_ROB_R1`) gives 33 releases, 166 comparisons and 3 readings.
  - TABLES.md is identical byte for byte.
  - TABLES.json differs only in `utc`.
- **After:** `shasum -c` reports both files OK, and `git diff --stat` on both is empty (05:28:52Z). No error was found in the generated tables.

## 6. Checks that passed

- **Labels and decisions:**
  - Every D1-D6 label, outcome class and flag in the three DECISIONS records.
  - The campaign summary: 3 looks, 0 met.
  - STRONGEST: IA_OSG +0.0017078 against IA_SFG +0.0016147.
  - The headline is verbatim from CS section 2 (2,917 characters compared). Its exclusion bounds recompute to 0.0133 / 0.0183 / 0.0141.
- **D2, D2R and beside numbers:**
  - All D2 means, se values, brackets and costs, including SBC_HI −0.0050 / −0.0061, RO_HI −0.0017 / −0.0008, and IB's SPD_HI −0.0037 with version means 0.0037 / 0.0102.
  - Every refit-cell value in RD Q3 and AB Q3.
  - Every beside and ablation value.
- **D3 and D4:**
  - "Resolved in five of six" recomputes: the upper bounds are −0.0041, −0.0084, −0.0029, −0.0036 and −0.0075, and +0.0032 for IB vs NEURAL_RAW.
  - The DDR_PTC_R1 resolutions.
  - The reference pair AUCs and their ordering.
- **Utility:**
  - 66 ledger rows; every campaign row passes except IA_OSG_RO_HI in version 1 (guard|sex0 at 1.0046 of the cap).
  - Reserve ratios; IB_CSF's accuracy loss is exactly 0; IA_SBC_LO was not evaluated.
- **Income-view evidence:** the numbers in RD 103-113 match their records.
- **Fits:**
  - k* is 1 / 2 (IA_SFG) and 2 / 4 (IA_OSG); the ladder steps match.
  - The failing rungs read 1.95 / 1.69 and 1.07 / 1.29 of budget S3.
  - IB pi* is 0.075, and at pi 0.10 the ratios are 1.22 / 1.001, 1.26 / 1.17 and 1.18.
  - The substituted ladder starts AGE5 against MAR2 in version 0.
- **Timing:**
  - Start 01:19:06Z; receipt verified 02:14:15Z.
  - Every card was committed before its first fitting read (02:38:59Z then 02:39:08Z; 04:11:44Z then 04:11:54Z).
  - Every bank was committed before its grid.
  - The first MODEL_SELECT read was at 02:50:32Z.
- **Access:** 45 label reads, all MODEL_FIT fit or MODEL_SELECT; no CALIBRATION or closed role; 8,001 people in 5,306 households.
- **Configurations:** 39 (23 + 16).
- **Deployment:** 10 packages, all bitwise; every probe refused (SEX, OCCP, PINCP, extra, missing and reordered fields).
- **Verifier counts in VER:** 616 + 308, 190, 312 / 2,568, 336 / 2,544, 5 label reads and 81.8 CPU-s all match the reports.
- **QUICKSTART:** every module, subcommand and lane exists (apart from R21-R23), and `ipr/test_rules.py` holds 28 tests.
- **Paths:** every backticked path exists, except pending closeout files, `PUBLICATION_RECEIPT.json` (R20), VER's backup file (R8) and this review's own path.

## 7. Consistency with the closing statement

- **Applied.** CS C1-C19 are applied in the working-tree documents.
- **Not applied in COORDINATOR_LOG.** At lines 106-107 the log still says "showed no gain" and "the men's occupation guard at pi = 0.10" (C4, C12). Append a dated correction.
- **CS wording that needs a fix:**
  - CS 1.1 gives the SPEND costs as "+0.0043 / +0.0019". The record has +0.0043492 and +0.0018498, so 0.0043 / 0.0018.
  - CS 1.4 and C5 say the spend growth "explains ... the missed reserve". The men's guard binds the reserve (R5).
  - CS C9 gives "45% against 50% slack". Occupation accuracy has 37% (R4b).
  - C3's "at matched spend" (A1).
- **Beyond CS.** The phrase "every other reader" goes beyond C2 (R2).
- **First-pass findings fixed by the rewrite:**
  - the income-leakage claim;
  - "equals" / "is spending";
  - "buys nothing" and "bought nothing";
  - "IA_OSG is weaker than its own controls";
  - "below the strong references";
  - IB's binding constraint;
  - "would nearly equal the operation" (contradicted by `ia/FIT_V0.json` substituted_ladder);
  - "the structural prediction held";
  - "far from the resolution";
  - "pooled 0.0019 / 0.0044" (the record gives 0.0018 / 0.0043).

## 8. Advisory notes

- **A1. "At matched spend/cost" for IB** (SH 31, SH 97, RD 257, RD 292, AB 80, HO 35).
  - The bracket IB_SPD_HI spends 0.0037 nats more than IB_CSF (version 0: 0.0064; version 1: 0.0010).
  - The interpolated point, which is reported only, is +0.0030 (se 0.0072).
  - The rules' scope sentence uses the same shorthand, and this is CS's wording, so it is advisory. Suggest "against its SPEND bracket, which spends 0.0037 nats more".
- **A2.** AB 40 "a fairer budget (S3)": write "a predeclared budget (S3), stricter on income than the designer's" (challenger finding 3).
- **A3.** AB 48-49 "the S3 budget was held back by income": write "the shared ladder was stopped by S3's income components at MAR1".
- **A4.** HO 38 "Not "a joint design"": the card declares IB jointly designed. Write "Not "a joint-design advantage"".
- **A5.** SH 36 and HO 18 "cut the occupation view": write "lowered ... at the point". The verifier's uncommitted A4_GRID record shows the occupation view lower in all 8 registered cells.
- **A6.** RD 260 "The data-rich readers agree": only the LOG_LR recipe reads the occupation view.
- **A7.** RD 214 "its pair is more recoverable than the twin's": add "at the point" (−0.0025, se 0.0070).
- **A8. Registered bets not reported:**
  - IA_OSG_R1's "k* includes WK3 or a COW step (0.35)" occurred: version 1 includes WK3.
  - PROTOCOL 10's expectations (CS 4.9) are not in RD.
- **A9.** D2R per-comparator qualifiers read READER_SEEDS_PARTLY_EFFECTIVE, with n_eff of 13-19 of 24 robustness cells for the decisive comparators. "Genuinely effective" is the refit-level check.
- **A10.** IA_OSG was never packaged for deployment: the check at 04:05Z predates it. Only the verifier's predict recomputation (A1.3) covers the osg payload. State this in the SH deployment row and in QS.
- **A11.** IB "SEX-blind by construction": add that SEX entered only through the MODEL_FIT strength guards, which fixed pi* together with income NLL (CS 4.3).
- **A12.** The challenger's within-field randomized response (input PRAM, finding 20) is missing from the untested list.
- **A13.** The C7 note at AB 69-70 refers to "four such comparisons", but AB never quotes that sentence.
- **A14.** AB Q3's "Component ablations" are registered-cell values. Label them so.
- **A15.** VER 22 "deployment of all 39 configurations" is the predict-function recomputation of the MODEL_SELECT arrays, not a package check. Line 74 says so; align the wording.
- **A16.** `roles/coordinator/write_eval_lanes.py` and `write_r1_lanes.py` hard-code the worktree path, so they fail in another checkout (QS 4).
- **A17. COORDINATOR_LOG, receipt size.** The log (02:14:11Z) says the receipt has "77 sanitized files". The receipt e909fb8c0's tree holds 149 files: 77 under `publications/pcrl-input-and-pair-release-v1/` and 72 under `publications/pcrl-dual-design-release-v1/` (the dual-design receipt's files, carried over). This is for the publication lane.
- **A18.** SH 62-63 and HO 80 cite `roles/engineering_review/REVIEW_CLAIMS.md`. That holds only once this report is saved there verbatim. SH 49 and AB 21 ("quoted verbatim") are now true.

## 9. Resources and what this review did not do

- **Wall time:** 04:55:38Z to 05:30:19Z.
- **CPU:** about 3 CPU-s in all, and no queue unit.
  - Measured with getrusage: the two table regenerations took 0.33 s; the lane, path and calibration scripts took 0.16 s.
  - Estimated: about 55 record-reading `python3 -I` calls and about 45 git, grep and shell calls, about 2.5 s together. The calibration gave 0.015 s per interpreter start and 0.018 s to parse all 97 campaign JSON records.
- **Agents:** none spawned.
- **Repository:** nothing written, changed or committed, and no state-changing git command was run.
- **Scratchpad:** the session scratchpad is shared with the coordinator and other agents. My files are under `scratchpad/eng_claims_review/`; I moved mine there and added nothing else. I read three of the coordinator's scratch scripts (`osg_spend_rob.py`, `osg_costs.py`, `r1_detail.py`) to trace R3 and R6 and ran none of them.
- **Not done:**
  - No label file was read and no real-data computation was run.
  - `ipr.test_rules` was not run: CS reports 28/28, and I counted 28 tests.
  - The stage tests were not run, so QS's "writes nothing" claim is unchecked.
  - `osg_spend_rob.py` was not run, so R3's figures are unsupported, not shown wrong.
  - HO 3 ("no message was sent") cannot be checked.
  - The publication branch was checked only for its tip and tree.
