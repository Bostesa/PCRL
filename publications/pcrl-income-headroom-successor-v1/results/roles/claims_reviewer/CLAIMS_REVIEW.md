# Claims review: PCRL income-headroom successor campaign

*Saved by the coordinator from the engineering and claims reviewer's final hand-back. The reviewer did not write this file itself
("My operating instructions forbid a subagent from writing report files", the same restriction as PROTOCOL_DEVIATIONS.md item 4).
The findings, locations and numbers below are the reviewer's; where the reviewer supplied long replacement paragraphs, they are
summarized here ("replacement text given") and were applied in the documents. The coordinator's disposition of every item is in
`CLAIMS_REVIEW_DISPOSITION.md`.*

**Versions reviewed.** Working-tree files, by SHA-256 prefix: `RESEARCH_DECISION.md` bb45961e; `START_HERE.md` d7763580;
`QUICKSTART.md` 449f719d; `CONFIRMATION_FEASIBILITY.md` ae381640; `UTILITY_RESERVE_RULE.md` 9cab9728; `MECHANISM_CARDS.md`
0da07def; `PROTOCOL_DEVIATIONS.md` a10bfd2a. These versions already include the coordinator's fixes for verifier items V9-D1,
V9-D2, V7-D1, V10-F1 to V10-F6 and V10-F9; those items are not repeated.

**Method.** No person-level data or labels were read; only committed or recorded files and git metadata. DEV values from
`report/DEV_REPORT_RESULTS.json`; feasibility from `CONFIRMATION_FEASIBILITY_RESULTS.json`; nomination from `NOMINATION.json` and
`NOMINATION_TABLE.csv`; MODEL_SELECT values from the ledger, `COMPARISON_TABLE.csv` and `REFERENCE_MS_TABLE.csv`; retired-candidate
values from the validation's `CONFIRMATION_FEASIBILITY.md` and `CONFIRMATION_FEASIBILITY_RESULTS_V2.json`.

**Verdict.**
- Every DEV number and claim label in the seven documents now matches the JSON.
- The nomination description matches the frozen rule.
- The timeline is correct.
- 16 must-fix items remain. They concern wording, attribution, feasibility framing, provenance and pending files.
- Two are material:
  - Against H_LJ on seed 1, the pair benefit is described as "established at clause level", but it is negative in 3 of 4 matched bootstrap reader refits.
  - The reader-refit evidence cited in the recommendation does not bear on the benefit.

## Verified correct (no change needed)

- **DEV results and claims tables** (7 releases; both scorecards) match the JSON: E (17 of 28 unresolved at c_P = 3.3040;
  recipient confidence 7/8 ENHANCED and 8/8 ORIGINAL); R2 (10 of 12 met; Brier 3.77-4.98 SE, NLL 5.75-7.05 SE, seed-1 accuracy
  0.8296 SE); R3 (the 4 accuracy clauses are the unresolved ones); P bounds; N +0.0061 / 0.0217 and −0.0070 / 0.0013; M +0.0285
  (0.0220) / +0.0316 (0.0209); C; D 0.0042 / 0.0015 with intervals including 0; O (six conditions below the rule, 0.862 / 0.550 /
  0.398 SE); reader-sensitive key NW_H_LJ|income|s1 (side BOTH); CV true; descriptive values 0.0142 / 0.0094 and 0.0066 / 0.0062;
  the M and C qualifier numbers; the near-tie sentence (48 of 48 ENHANCED winners flagged).
- **Feasibility.** Every n80, limit, power at 17,510, assurance value, same-size value and governing assignment in the current
  tables matches the JSON. All 24 numbers in the new recipient-margin table recompute exactly. The binding-clause rule was already
  in `ihs/feasibility.py` at the freeze and is unchanged since. Retired-candidate values match the validation: 4,632,356;
  59,928,535; 0.013 / 0.252; 0.102 / 0.161; 0.027 / 0.042 SE.
- **Nomination.** Six qualifiers; near-tie window ≤ 0.6136170 and a set of four; ratios 2.862 / 2.677 / 2.677 / 2.212; the nominee
  0.0032 above the best score; the corrected tie-break wording.
- **Ledger.** 34 counted configurations (H 9, P 14, O 7, CMB 4). The MODEL_SELECT values in `RESEARCH_DECISION.md` and
  `MECHANISM_CARDS.md` match the ledger, the comparison table and the investigator reports.
- **`UTILITY_RESERVE_RULE.md`.** Every ratio recomputes from `REFERENCE_MS_TABLE.csv` with max(own SE, H_LJ SE). The 1.7267 /
  3.633 / 1.37 / 1.10 arithmetic is right, the `RESERVE_RULE.json` hash is aca4d8e4…, and `ihs/test_reserve.py` has 13 checks.
- **Timeline.** Registration commit 74376912c at 02:35:11Z; first fit 02:35:25Z. Receipt 77dd6710f committed 02:35:45Z, pushed
  02:35:49Z. Freeze 930235e9e at 03:03:35Z. REPORTING_LOCK at 03:07:23Z, in commit 8e559efe8 at 03:07:29Z. Receipt 9ad0fb25d
  committed 03:07:50Z, pushed 03:07:55Z. Both receipts are on origin/publication. DEV read 03:08:07Z; verifier replay 03:17:54Z;
  14 reads in the lineage.
- **Engineering.** `DEPLOYMENT_CHECKS.json`: 6 packages, three paths, 7-row batch, five refusals. `ihs/report.py` and
  `ihs/claims.py` hashes equal ANALYSIS_LOCK's. The KeyError in deviation item 6 is at `report.py:234`. Item 7's facts are in
  f615cbf53 (`deploy.py` +3 lines and three swept verifier files). sex0 = SEX code 1 = men (`ihs/fam_o.py:7`).
- **Claims not made.** No document claims novelty, a privacy guarantee, information removal or injectivity.

## Must-fix

1. **Clause-level "established" against H_LJ** (`RESEARCH_DECISION.md`:19-20 and :108-110 (Q2); `START_HERE.md`:22-24). CLAIMS_DRAFT
   claim P says the per-reference legs are "condition values, not separate conclusions". Under ENHANCED, the H_LJ leg also contains
   NW_H_LJ|income|s1, which is false. Matched household-bootstrap refits of the ENHANCED pair winners give seed-1 benefits against
   H_LJ of −0.0082, +0.0073, −0.0058 and −0.0077, against +0.0166 for the full fit (source: `per_seed.S1.refit_roster.aux_sample`;
   the resample seeds are the same for every service, `READER_PLAN.json`). Replacement texts were given for :19-20, Q2 and
   START_HERE :22-24 (also removing "not beyond the 0.020 target", which reads as showing the target false).
2. **Reader-refit evidence in the recommendation** (`RESEARCH_DECISION.md`:152). The fall from 0.627 to 0.594-0.604 is a level
   shift that every release shares (H_LJ 0.610-0.622, NEURAL_RAW 0.601-0.613 under the same refits); it says nothing about the
   benefit. Replacement: the matched refit benefits (H_LJ +0.0075 to +0.0216 on seed 0, −0.0082 to +0.0073 on seed 1, 3 of 4
   negative; NEURAL_RAW −0.0010 to +0.0152 on seed 0, 2 of 4 negative, +0.0129 to +0.0179 on seed 1), descriptive, mixing a
   resampling shift with variance.
3. **Component control turned into equivalence** (`RESEARCH_DECISION.md`:125-127, "at about the same recovery"). C is
   NOT_ESTABLISHED; the BEN|CC clause intervals are −0.0026 to +0.0066 (seed 0) and −0.0134 to +0.0082 (seed 1). Replace with
   "its effect on pair recovery is not resolved … An unresolved clause is not equivalence."
4. **Temperature attribution rests on one-seed family-P controls** (`RESEARCH_DECISION.md`:128-129 and :50; `MECHANISM_CARDS.md`:43-44).
   PC_TEMP_P2 / PC_TEMP_P3 ran on seed 0 only; they shrink family P's twin; family H has no temperature control. Replacement texts
   given.
5. **Untested interpretation; one-seed revisions** (`RESEARCH_DECISION.md`:47-48 and :51-52; `MECHANISM_CARDS.md`:24-26 and :41).
   P's critic-chasing explanation is "Interpretation (not tested further)"; P1, H1 and H2 ran on seed 0 only; H2's Brier was
   +0.00497, below the 0.005 cap (within 0.00003); H2's income view was +0.0113 against NEURAL_RAW. Replacement texts given; add
   "; seed 0 only" in MECHANISM_CARDS.
6. **"relieves" collides with a registered label** (`RESEARCH_DECISION.md`:40 and :132-133; `CONFIRMATION_FEASIBILITY.md`:94-95).
   Claim O reserves OCCUPATION_CONSTRAINTS_RELIEVED; O1's men's guard margin is 1.83 / 1.62 SE on MODEL_SELECT, below its 2-SE
   target, with no DEV evidence. Replacement texts given ("raises the men's guard margin …").
7. **"privacy" wording for the transplanted corrections** (`RESEARCH_DECISION.md`:116; `START_HERE.md`:17). The registered term is
   "recovery-targeting component". Replace "*The privacy step:*" with "*The recovery-targeting step:*" and "privacy-trained" with
   "critic-trained (LOCAL_TREE / JOINT_TREE)".
8. **D1 is claim P's conjunction, and the requirement is not identified** (`CONFIRMATION_FEASIBILITY.md`:45-47 and :89;
   `RESEARCH_DECISION.md`:26-27 and :138-140; `START_HERE.md`:27-28). The composite is E ∧ R2 ∧ P; `conditions()` has no R2 and 38
   conditions (the two zero-SE occupation-vs-H_LJ recipient clauses are dropped); binding margins are 0.40-0.87 SE; at a true slack
   of 0 the D1 limit is 0.162 / 0.163. Replacement texts given ("claim P's registered conjunction …"; "not identified by these
   data").
9. **What drives the planning requirement** (`RESEARCH_DECISION.md`:148-149). In the governing ENHANCED D1 at n80 the weakest
   conditions are NW|H_LJ|income|s1 (0.855) then occupation NLL (0.988); occupation NLL leads only in D2 (0.844) and ORIGINAL D1
   (0.850). Replacement text given.
10. **Claim O is not a conjunct of the composite** (`RESEARCH_DECISION.md`:23-24). Say "Reported beside the composite, not part of
    it: … (claim O)".
11. **NLL never bound the retired candidate** (`CONFIRMATION_FEASIBILITY.md`:27-28 and :92). Its binding clauses were income Brier vs
    Ucal (0.027 / 0.042 SE) and seed-1 accuracy (0.110 SE). Replacement texts given.
12. **Order-of-items note in the deviations file** (`PROTOCOL_DEVIATIONS.md`:55-56). The file was first committed at the freeze
    930235e9e with items 1, 2, 4, 5, 3; item 6 and the note in f615cbf53; item 7 later; item 3's substance is registered in
    PROTOCOL.md §2 and `DEV_GRID_REVIEW.md` at 74376912c, but its text was first committed at the freeze. Replacement text given.
13. **Card timing is self-reported** (`MECHANISM_CARDS.md`:6-7). No CARD.md is in 74376912c; all three were first committed at
    930235e9e, after every family fit (the last fit is ledger entry 64 at 02:55:10Z). Replacement text given.
14. **Roles row** (`START_HERE.md`:60). PROTOCOL.md §13 registers a separate engineering and claims reviewer; name this review.
15. **Pending records and a forward-looking claim** (`RESEARCH_DECISION.md`:156-158; `START_HERE.md`:4 and :46-48;
    `QUICKSTART.md`:6; `PROTOCOL_DEVIATIONS.md`:52-53). BACKUP_VERIFICATION.json, RESOURCE_ACCOUNTING.json, ACCESS_RECEIPT.json,
    HANDOFF.json and PUBLICATION_RECORD.json do not exist yet; INDEPENDENT_VERIFICATION.json, VERIFICATION.md and 9 verifier code
    files are untracked at review time. Make every cited file exist and commit it, or mark it pending; make item 7 name the commit
    that adds the verifier's files.
16. **Ambiguous column header** (`RESEARCH_DECISION.md`:72, "Income AUC"): it is the income-view SEX reader AUC. Replace with
    "Income-view SEX AUC s0 / s1 (ENHANCED)".

## Optional

1. "about 35-fold" → "about 36-fold" (4,632,356 / 128,599 = 36.0; D2 205-fold); "improves the practical feasibility problem" →
   "reduces the plug-in requirement".
2. Note that the binding table uses the successor's own DEV SE (0.87 for accuracy) while R2 uses max(successor, H_LJ) SE (0.83).
3. "disclosure D2" → "the replication's disclosure item D2 (an in-memory extraction; validation POPULATION_INVENTORY.md)".
4. Append to the 0.020 sentence "; this does not show that the target is false."
5. Report the nominee's MODEL_SELECT flags beside the DEV results (READER_UNSTABLE; seed-1 c3 minimum cell +0.0012 over B2TWIN;
   UTILITY_CONFIDENCE_UNRESOLVED (8 of 28)), as FREEZE_CHECK note 3 asked.
6. Add the twin's label UTILITY_HEADROOM_ONLY and its DEV pair benefits against each reference (−0.0143 / −0.0150 against H_LJ,
   −0.0157 / −0.0192 against NEURAL_RAW; DESC|TWBEN).
7. "several winners differ" → "27 of the 48 winners differ between the banks".
8. "of the size of the benefits" → "rough reader-refit SDs of 0.007-0.009 against benefits of 0.012-0.017".
9. List all six conditions below the rule in DEV SE units (men's guard 0.86 / 0.55; occupation NLL vs Ucal 0.40 / 1.48; vs U0
   0.87 / 1.53).
10. "honest held-out checks" → "held-out checks (no gain over T_TREE on CALIBRATION, +0.0003 worse; 0.0016 / 0.0013 on
    MODEL_SELECT)"; note that H's registered Step-1 consequence (CARD.md:89-91) was triggered on CALIBRATION but not on MODEL_SELECT.
11. Cite NOMINATION_RULE.md; disclose that P's allotment was raised from 12 to 14 and OC_BFIT was added after the O1 fits.
12. START_HERE "in order": list the 74376912c files before 77dd6710f; give the roles/statistical_reviewer/ paths for
    LOCK_ENCODING_CHECK.md and FREEZE_CHECK.md; add 930235e9e (03:03:35Z) and 8e559efe8 (03:07:29Z).
13. START_HERE: add "(committed 14 s before the first fit; public receipt 24 s after it, PROTOCOL_DEVIATIONS item 1)"; add N and T.
14. QUICKSTART: add the verifier replay time 03:17:54Z; torch 2.10.0 is not in the committed records (cite or drop); the recorded
    refusal tests cover SEX, extra, missing and reordered fields and a hidden-output request only.
15. PROTOCOL_DEVIATIONS: "L1-L2" → "L1-L3" (LOCK_ENCODING_CHECK.md:172).
16. Item 4: `roles/investigator_o/CARD.md`:148 dates the refusal "~03:20Z", but that line is already in the freeze commit
    (03:03:35Z); the time is wrong; record this.
17. New engineering note: `ihs.common.dev_report_truth` checks only that the lock is committed (`common.py`:253-254) while
    PROTOCOL.md §3 says "committed and published"; the receipt was pushed at 03:07:55Z, before the 03:08:07Z read, so there was no
    timing deviation; publication was procedural, not enforced (the same pattern as item 1).
18. Item 2 "(02:05Z)": add the ledger times of the reference units, 02:09:19Z / 02:09:21Z.
19. `UTILITY_RESERVE_RULE.md`: no change needed.
