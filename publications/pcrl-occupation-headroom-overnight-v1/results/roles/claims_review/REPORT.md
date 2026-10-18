# Claims review of the five decision documents (saved by the coordinator)

**Disclosure.** The claims reviewer (a separate agent, with three parallel forks of its own) returned this review as the text of its
final hand-back at about 08:15Z. The coordinator saved it here unchanged in substance. The review applies to the documents as
committed at 2698a58e7 (unchanged through 254f4b7a0). The reviewer edited nothing and opened no `*.npz`, label, ATTACK or DEV_REPORT
file. Brief: `roles/coordinator/CLAIMS_REVIEW_BRIEF.md`. The coordinator's disposition of every item is at the end.

Abbreviations: RD = RESEARCH_DECISION.md, AB = ADVISOR_BRIEF.md, MH = MANUSCRIPT_CLAIMS_HANDOFF.md, SH = START_HERE.md, MC =
MECHANISM_CARDS.md.

No finding changes the formal outcome (NO_QUALIFYING_CONFIGURATION; DEV_REPORT closed). Four findings change sub-conclusions: the
delivered component, the reserve-keeping frontier, how independent the seeds are, and why promotion failed.

## A. Errors and omissions that change a sub-conclusion

- **A1. The reserve-keeping frontier is misstated; P2_LOC_B80 is missing.**
  - P2_LOC_B80 (local critic, B2_SINGLE, B80) is RESERVE_MET on both seeds (min ratio 2.204 / 2.757).
  - Its score, −0.0170, is the best of any reserve-keeping configuration (P2A_PAIR_B50 −0.0179, P2_PAIR_B50 −0.0223).
  - It sits 0.010 above SUCC (−0.0101; seed means −0.0146 / −0.0056).
  - It is ahead of P2_COMP_B80 in all 8 cells (+0.0118) and of P2_CONT in all 8 (+0.0258).
  - Its occupation-view maximum is +0.029 (P2_PAIR_B50: +0.051).
  - SELECTION_RULE §6 requires the frontier (score against minimum whole-release ratio) when nothing qualifies: P2A_PAIR_B80 −0.0027
    / 0.85; P2_PAIR_B80 −0.0109 / 1.39; P2_LOC_B80 −0.0170 / 2.20; P2_PAIR_B50 −0.0223 / 2.62; B2_STACK −0.0401 / 2.862; INC_TW_PO
    −0.0479 / 3.144. B2_SINGLE+H3 (−0.0470) and INC_H3_PO (−0.0480) are dominated.
  - The PAIR-vs-LOC comparison uses the same nominal budget only: PAIR_B80 breaks the seed-0 reserve (1.39), LOC_B80 keeps it.
  - Sources: NOMINATION_TABLE.csv; UTILITY_FRONTIER.csv; READER_STABILITY.csv (PROT_P2L); protection CARD; SELECTION_RULE §6.
- **A2. "Beats … on both seeds under every reader" is false at B50 and for compression.**
  - P2_PAIR_B50 vs P2_COMP_B80 is SIGN_CHANGES (seed means +0.017 / −0.004). P2_PAIR_B50 vs P2_CONT is SIGN_CHANGES (7 of 8 cells).
    P2A_PAIR_B50 vs compression is SIGN_CHANGES (+0.020 / +0.002).
  - Fixed recipes: P2_PAIR_B80 vs P2_COMP_B80 is positive under 17 of 18 recipes on each seed (minimum −0.0056). P2A_PAIR_B80 vs
    P2A_COMP_B80 is positive under 18 of 18 (seed 0) and 17 of 18 (seed 1). The GB_LONG probe gives SIGN_CHANGES for both.
  - No fixed-recipe run exists for B50 or M90.
  - "Every reader" without "declared" breaks the declared-reader rule.
  - Correction: state B80 only, with these counts, and add "at the B80 level" to MH statement 2.
- **A3. Witness 1's two seeds are not refits.** Family B's card ("Seed caveat (checked 06:24Z)") records:
  - the deployed TREE_V2, ADD_V2 and ADD_V1 bases are bitwise identical on both seeds;
  - B2 and B3 differ by seed only through the layer weights and temperature (tau 1.0034 vs 1.0054);
  - B2_SINGLE's occupation accuracy loss is −0.008749 on both seeds.

  For B2_SINGLE, B2_STACK, B3_CALTR and the P2_* witness, seed-to-seed agreement mostly reflects the seed-specific references.
  (Unverified reviewer inference: A1_PO_16 may be similar; ask investigator A.)
- **A4. The recipient gate is not the only binding failure, and the noise wording overclaims.**
  - All 18 candidates are also BENEFIT_NOT_STABLE (best score −0.0027).
  - Six miss the reserve on one seed: P2_PAIR_B80 1.39, P2_PAIR_M90 0.57, P2A_PAIR_B80 0.85, P2_COMP_B80 1.92, P2A_COMP_B80 1.80,
    P2A_LOC_B80 1.41.
  - Without the recipient gate the outcome would be unchanged.
  - The noise figures (SD 0.0212; p = 0.806 per comparator; about 0.99 against five) come from a normal-noise model at a true zero
    difference. The failure is consistent with split noise, not shown to be noise.
  - "Cannot" holds on this grid only because every H3-anchored release shares H3's income view.

## B. Wrong numbers and missing seeds

1. The gain over TASK_REFIT for B2_SINGLE is about half an SE: −0.000905 (0.001678) / −0.001038 (0.001687), i.e. 0.54 / 0.62 SE.
   A1_PO_16 is 1.4 / 1.5 SE.
2. The fixed-recipe count "9 of 18 (median −0.0001)" for P2_PAIR_B80 vs SUCC is seed 0. Seed 1 is 3 of 18 (median −0.012).
3. P2_PAIR_B80 vs P2_CONT: 18 of 18 on each seed (medians +0.038 / +0.028).
4. The R3 "+0.047" is for P2_PAIR_B80, seed 0. R3's adverse result: P2A_PAIR_B80 vs SUCC under the frozen winner, −0.0025 (0.5884 vs
   0.5859).
5. The learning-curve median change ranges from −0.003 to +0.005.
6. The RD §6 table header says "seed 0 / seed 1", but several rows are pooled 8-cell values.
   - Per-seed recipient maxima:
     - the H3 income view is +0.031 / +0.009, so seed 1 passes;
     - occupation view: B2_SINGLE +0.080 / +0.078, P2_PAIR_B50 +0.038 / +0.051, P2A_PAIR_B80 +0.007 / +0.033 (seed 0 passes);
     - INC_H3_PO income view: +0.027 / +0.055.
   - P2A_PAIR_B80 target values per seed: H_LJ +0.040 / +0.003, NEURAL_RAW +0.038 / +0.0025.
   - The adverse bullet mixes seeds: seed-0 income with seed-1 occupation.
7. The income-view failure cells are seed-0 cells S0R1k0/k1: one household split read with two reader seeds. Seed 1 passes (max
   +0.0087).
8. The occupation-view range tops out at 0.082 (A1_PO_8 vs B2TWIN).
9. RETIRED_B3CTL on seed 1 is −0.020, not −0.021.
10. P2_PAIR_M90 row:
    - it reads +0.039 (vs CONT) and +0.025 (vs COMP), with a seed-1 SUCC mean of −0.0005;
    - the +0.025 is against COMP_B80, not a code cost-matched to M90 (P2_COMP_M90 exists on seed 0 only).
11. B50 rows: P2_PAIR_B50 vs CONT is +0.021 (sign changes). The compression seed means are P2 +0.017 / −0.004 and P2A +0.020 /
    +0.002. The B50 and M90 rows are compared with COMP_B80, so "same budget" does not apply to them.
12. Transfer error: +0.0016 to +0.0029 on seed 0 and −0.0008 to −0.0012 on seed 1. That is 30–54% of the seed-0 slack (0.0054),
    and larger than the about 0.0011 a B80 level leaves unspent.
13. The stacking gain out of fold is −0.0025 / −0.0024; −0.0020 is the MODEL_SELECT difference.
14. The unused-budget list sums to 14, not 15 (80 − 65). One PRIV24 slot is missing.
15. P3:
    - −0.0024 vs +0.0064 is B2_SINGLE only (A1_PO_16 −0.0025 vs +0.0071);
    - 0.616 / 0.618 are P3 / P3A (two witnesses, seed 0);
    - "lost most" holds only for A1_PO_16 (about 67%); B2_SINGLE lost about 40–47%.
16. One-seed values given without the other seed:
    - A0 +0.0086 is A0_HGB_S seed 0; there is no seed 1, and A0_HGB_L is +0.0122;
    - the income repair on seed 1 is −0.0028 / −0.0041;
    - the family-C best on seed 1 is +0.0003;
    - occupation-side ratios per seed: 5.000 / 4.418, 4.881 / 4.299, 4.153 / 3.634.

## C. Wording that breaks an interpretation rule

1. "Task-only continuation twin": the CONT fits read MODEL_FIT SEX (verifier note D6). Use "matched continuation twin (critic weight
   zero)".
2. "Did not replicate on seed 1": use "was not reproduced by the seed-1 refit".
3. RD §6 "two incremental components are delivered" reads as claims C and M established. State the claim-ladder status.
4. RD §1 "household cross-fitting reproduces the direction out of fold" is unsupported. Out-of-fold comparisons against U0 / T_TREE
   favour those in-sample references (investigator A REPORT).
5. "Removes a variance problem" should be "reduces": the committee gain falls from −0.0013 / −0.0031 to −0.0006 / −0.0005.
6. "Down to the reserve (0.57–1.39)": these ratios are below the 2.0 reserve.
7. Only the men's guard improved. The women's guard is at the teacher's level (−0.0091 / −0.0082 vs −0.0093 / −0.0079). AB's
   −0.0041 / −0.0025 is B2_SINGLE only.
8. AB "better occupation quality than SUCC": NLL, Brier and the men's guard are better. Accuracy (−0.00125 vs −0.0020) and the women's
   guard (−0.00476 vs −0.00494) are slightly worse.
9. AB "protects less": use "lowers declared-reader recovery less".
10. AB "2:1-weighted (P3) critics … bracket it": P3 weights the task loss by SEX, not the critics, and ran on seed 0 only.

## D. Omissions

1. All six required comparators fail this campaign's 2-SE reserve:
   - SUCC, H_LJ, B2TWIN, TWJ: 0.321 / 0.624;
   - RETIRED_B3CTL: 0.321 / 0.349;
   - NEURAL_RAW: 1.947 / 1.558.

   The reserve constrains candidates only (verifier note D4).
2. The smaller seed-1 scales come from the smaller seed-1 men's slack (0.0041 vs 0.0054) as well as from the transfer error.
3. RD §7 "every recomputed number matches" leaves out the verifier's non-numerical statuses:
   - access: FAIL_REGISTERED_RULE__DEVIATIONS_AUTHORIZED_DIAGNOSTIC;
   - VER_PROVENANCE: ATTENTION;
   - doc notes D4 and D6 still open;
   - D3: 61 of 201 admitted copies have no expected hash, 59 are unpinned, and the H3+O1 / TW+O1 references rest on the source's
     private copy;
   - fixed-recipe, probe, R3, learning-curve and income-noise numbers were not independently recomputed.
4. The MH disclosure list omits:
   - deviation 17 (family B has no code hash and its code was committed after the first fits; four family-C hashes are absent from
     git);
   - deviations 5, 12, 4 and 21;
   - family B's note that B2 and B3 were chosen after MODEL_SELECT feedback.
5. MC: one card exception (item 10). Items 14 and 17 are code-provenance exceptions.
6. RD §8 should cite DEPLOYMENT_CHECKS_BATCH3.json.
7. Witness 1 is family B's registered control (kind control). B's mechanism, B2_STACK, ranked third because stacking weakened the men's
   guard.
8. INC_TW_PO 3.144 vs the unrepaired task-only HC_STRONG 3.296: the repair lowered it, and the gain over 2.862 comes from dropping H3's
   transplant step. "All seven comparators" = six required comparators plus the matched component.
9. MH statement 1: 2 of the 28 conditions (seed-1 income accuracy vs U0 and Ucal) are unresolved at the Bonferroni level.
10. Low severity:
    - "never entered the grid" carries the label NOT_ADMISSIBLE;
    - "65 of 80" counts seed-specific configurations, while "34" counts configurations over both seeds;
    - 435fe44e5 is the receipt commit, and the registration commit is b1fd591d0;
    - SH "21 dated items": some items carry no time;
    - SH "201 hash-checked copies": 61 have no expected hash;
    - DEV_REPORT's label-free raw inputs were used for the deployment checks;
    - PUBLICATION_RECORD.json did not yet exist;
    - P2A_PAIR_M90 (seed 0 only) has the lowest recorded pair AUC (0.5766) and is not mentioned.

## Checked and found correct (reviewer's list, summarized)

- **Nomination record:** NO_QUALIFYING_CONFIGURATION at 07:45:00Z; rule sha e18f1c68; 34 = 18 + 5 + 11 (verifier replay 34 / 34);
  scores −0.047, −0.022, −0.0027, −0.048, each bound by RETIRED_B3CTL.
- **Recipient gate:** the +0.0309 / +0.0147 values were recorded before any candidate grid; SD 0.021 and p about 0.81.
- **RD §1–§5:** every value not listed above.
- **RD §6 bullets:** the favourable and adverse numbers.
- **RD §7 facts:** start 04:36:11Z; 3.24 h; 4.81 of 14 CPU-h; 65 of 80; receipt verified 05:11:09Z before the first label read at
  05:11:30Z; 0 ATTACK / DEV_REPORT reads; 15 banks; 38 packages; 2.3e-13; the P2 card 71 s after its first fit; every deviation
  number cited.
- **MH:** the claim-ladder status is correct (W–O NOT_EVALUATED, J NOT_CLAIMABLE, V NOT_CLAIMED).
- **Interpretation rules:** respected elsewhere (coalition-aware wording, target separate, R3 not fresh, declared-reader caveat,
  determinism and recalibration caveats).

## Coordinator's disposition (08:16-08:22Z; revised documents committed at 17a1c6580)

All items accepted.
- **Revised:** RESEARCH_DECISION.md, ADVISOR_BRIEF.md, MANUSCRIPT_CLAIMS_HANDOFF.md, START_HERE.md and MECHANISM_CARDS.md.
- **Frontier:** written to NOMINATION_FRONTIER.csv (ohn.frontier) as SELECTION_RULE §6 requires.
- **A3:** checked against the records. A1_PO_16 is a genuine refit on the same people: seed-specific household folds give tree counts
  168 / 154 and different accuracy losses (−0.00987 / −0.00975). Family B's bases are identical across seeds, as the card says.
- **Not adopted:** no item.

**Correction to the A3 disposition above (08:35Z):** the note that "A1_PO_16 is a genuine refit" is wrong. The label-free check
`ohn.seed_identity` (unit OHN_SEED_IDENTITY_A_V2, `checks/A_SEED_IDENTITY.json`) shows that family A's seed-1 model is bitwise the
seed-0 boosting path truncated at the seed-1 tree count: A1_PO_16 154 of 168 trees, A1_PO_8 240 of 279. Only `max_iter` and
`random_state` differ, and the random state is unused without early stopping or subsampling. MODEL_SELECT decisions agree on 98.9% /
98.8% of people. Neither witness has independent seed refits.

## Second pass (08:25-08:32Z, on 17a1c6580) and disposition

| Item | Finding | Disposition |
|---|---|---|
| 1 | ADVISOR_BRIEF: the 18 configurations pass the utility point gate and fail the next one, the recipient point gate (not "the first one checked") | fixed |
| 2 | RESEARCH_DECISION section 3: P2_PAIR_B50 vs CONT reads +0.020 (+0.027 / +0.013) | fixed |
| 3 | P2_LOC_B80 guards: men +0.0007 / +0.0001, women −0.0052 / −0.0062 | fixed |
| 4 | Teacher NLL vs U0 on seed 1: −0.0082 | fixed |
| 5 | 57 of the 65 seed-specific configurations make up the 34 in the nomination table; 8 one-seed task-only configurations were not listed | fixed (named in section 7) |
| 6 | The "not independently recomputed" statement is the claims review's, not the verifier's | moved out of the verifier list |
| 7 | Family A's seeds are probably not independent fits | checked: the seed-1 model is the truncated seed-0 path, bitwise; RESEARCH_DECISION, ADVISOR_BRIEF and the handoff now say so |
| 8 | Deviation 5 concerns families A and B | fixed |
| 9 | 61 copies have no expected hash, 59 of them unpinned | fixed |
| 10 | ADVISOR_BRIEF omits the R3 reversal | added |
| 11 | Distance table: per-seed occupation-view maxima (P2_LOC_B80 +0.025 / +0.029; INC_H3_PO +0.080 / +0.078) | fixed |
| 12 | PUBLICATION_RECORD.json pending | written at publication |
