# Reader-robustness investigator report (partial; saved by the coordinator)

**Disclosure.** The reader-robustness investigator (a separate agent) returned this report as the text of its final hand-back at
about 07:37Z. Its session was then **terminated by a platform safeguard** ("API Error: Opus 5.5's safeguards flagged this session
... Details: `[bio]`", request id req_011Cg9KVY33yJ1WFphkfyZto). The delivered text breaks off at the start of its per-candidate
stability table. The coordinator saved the delivered text below unchanged in substance and did **not** relaunch the agent to finish
the same work (the brief: "If an action is blocked, preserve state and report the actual restriction"). Before the termination the
agent had written its final machine tables in this folder; the per-candidate table the text was about to print is
`RR_FINAL__CANDIDATE_TABLE.csv` / `.json` (8-cell label and mean, seed-0 and seed-1 means, probe-winner, union-winner, oracle and
MLP_BIG / GB_LONG / kNN columns). The index of every file is `RR_INDEX.json`. Code: `code/` (rr_probe.py, rr_split3.py, ...).

---

READER-ROBUSTNESS INVESTIGATOR — FULL REPORT. PCRL overnight occupation-headroom campaign, 2026-10-18. Work ran 06:28–07:37Z.

The data are public-use ACS 2019 1-Year PUMS person records for Florida, MODEL_SELECT development role only (8,001 people, 5,306
households, used adaptively). Readers estimate the recorded SEX column (sex0 = men, sex1 = women). They see only the released
float64 probability vectors and argmax decisions: the income view, the occupation view and the linked pair.

Everything below is diagnostic development evidence and was never used for selection. Lower recovery for these readers is not
protection against every possible reader. Smaller FIT samples cannot establish protection at the full reader budget.

Boundaries kept:
- No ATTACK_FIT, ATTACK_VALIDATION or DEV_REPORT access.
- ohn/devreaders.py and the locked recipe files were not edited.
- Nothing was committed or pushed.
- Label reads went through ohn.common.load_role with purposes beginning "reader robustness: …" (diagnostic probe, fixed-recipe
  learning curves, extra-split diagnostic), plus the grid's own logged reads for my learning-curve banks.

## Headline findings (most consequential first)

1. Protection outputs. Only the benefit over each arm's own matched controls holds on both release seeds and under every reader.
   - That benefit is over the CONT twin (and largely over the COMP code). Example: P2A_PAIR_B80 vs P2A_CONT is STABLE_POSITIVE
     (+0.040). 18 of 18 fixed recipes are positive on each seed, medians +0.048 (seed 0) and +0.033 (seed 1).
   - Superiority over SUCC and RETIRED_B3CTL does not replicate. P2A_PAIR_B80 vs SUCC is +0.018 on seed 0 (18 of 18 recipes
     positive) and −0.010 on seed 1 (3 of 18 positive).
   - The seed shift comes from the release, not the readers. Every PAIR arm loses 0.010–0.031 in pair AUC from seed 0 to seed 1.
     Controls, witnesses and references move at most 0.007. The seed-1 projection scales are smaller (B80 0.72 vs 0.86–0.92). The
     protection investigator's MODEL_FIT out-of-fold readers show the same drop.
   - The best P3 level that meets the seed-0 reserve (P3_PAIR_B80, pair 0.616) beats P2_CONT under every reader (+0.017 fixed
     median). It is behind SUCC and RETIRED_B3CTL under every reader, and 0.018 behind P2_PAIR_B80.
2. The frozen BENEFIT_SHRINKS_WITH_READER_DATA flag is driven by VALIDATION-winner selection noise.
   - When the candidate and the comparator are read by the same frozen recipe (cached readers, no fit), almost no flag survives.
     Only 0–6 of 15 recipes flag, with medians of −0.003 to +0.005.
   - Release-level pair-AUC gains from 1/4 to full FIT are +0.016 to +0.019 under fixed recipes for every release (protection
     outputs, controls, witnesses, references). Under the frozen winner the same gains range from +0.009 to +0.027.
   - My 06:57Z message that P1_PAIR_B80's benefit shrinks was withdrawn at about 07:00Z. Its cause was P1_CONT's full-FIT winner in
     S0R1: VALIDATION 0.654, SCORE 0.620.
3. The 8 grid cells contain only 2 household splits, and both release seeds share them.
   - roles_of() depends only on the household id and R1/R2.
   - For every one of 26 releases, the pair view reads about 0.022 higher on R2 than on R1, on both seeds.
   - Among the 68 reference and witness pairs, the R1−R2 contrast of the benefit correlates 0.76 between seeds.
   - In 6 pairs the sign is set by the split on both seeds: the witnesses (B2_SINGLE, B2_STACK, B3_CALTR) against H3+T_TREE,
     H3+TASK_REFIT and TWJ.
   - A STABLE_POSITIVE label therefore means "stable over 2 household splits × 2 reader seeds × 2 release refits".
   - A third split (R3) behaves like one more independent draw.
4. The per-cell income-view recipient gate fails through noise, not through information.
   - Pooled over 36 pairs of distinct income wires, the split SD of an income-view difference is 0.021. The bootstrap SE is 0.009 and
     the reader-seed SD 0.002.
   - With that noise, a true difference of exactly 0 fails the gate against one comparator with p = 0.81, and against five other
     income wires with p = 0.99. A candidate truly 0.010 safer still fails with p = 0.56.
   - Empirically, 9 of 12 orientations of near-identical income views (|8-cell mean| < 0.005) fail.
   - Resolving ±0.010 would need about 19 independent splits at a true difference of 0, or 5 at −0.010. (Methodological finding only;
     no rule change proposed.)
5. Winners.
   - 97.7% of all 780 VALIDATION winner selections are near ties (gap < 0.010).
   - Winners change between repetitions in 84–92% of opportunities.
   - The candidate's and the comparator's pair winners come from different recipe families in 65% of 1,908 cells (82% different
     recipe).
   - Reading both with the same recipe keeps the sign of the reference conclusions, but it removes most of the reader/split spread.
     Median reader/split SD of the benefit: 0.0080 for the frozen winner, 0.0019 for fixed LOG_LR, 0.0027–0.0036 for OUT_LR / CATI01
     / ET25.
6. Diagnostic probe (MLP 256-128, long histogram gradient boosting, kNN on log p).
   - No design beats the frozen slate on average.
   - The ordering of pair recovery barely moves (Kendall tau 0.82–1.0).
   - The SCORE-max oracle sits 0.001–0.008 above the frozen winner.
   - The frozen labels are reader-sensitive at their boundary. SUCC vs H_LJ is POSITIVE_NOT_STABLE under the frozen winner but
     STABLE_POSITIVE under the probe winner and under 6 of 18 fixed recipes.

## Task 1 — learning curves (FIT fractions 1/4, 1/2, full; VALIDATION and SCORE unchanged)

Learning-curve banks were run through the locked grid code:
- RR_LC_WIT: both seeds; witnesses B2_SINGLE and A1_PO_16 plus the six required comparators and H3+T_TREE.
- RR_LC_PROT_P1: seed 0; P1_PAIR_B80 and P1_CONT.
- RR_LC_PROT_P2: both seeds; P2_PAIR_B80, P2_CONT, P2_COMP_B80.
- RR_LC_PROT_P2B50: both seeds.
- RR_LC_PROT_P2A: both seeds; P2A_PAIR_B80, P2A_CONT, P2A_COMP_B80.

Reference fractions were cached from REF_BANK. The official ohn.grid_summary was run for each bank (GRID_SUMMARY_*.json). A
fixed-recipe check was added: the 15 informative frozen recipes were read from the grid cache with no fit (unit OHN_RR_LCFIX_*).
That check matched the grid winners on 3,132 of 3,132 values.

Mean pair benefit at 1/4 → 1/2 → full FIT, the frozen flag, and how many fixed recipes flag (with the median full-minus-quarter
change):

| Pair | Benefit at 1/4 → 1/2 → full | Frozen flag | Fixed recipes flagging (median change) |
|---|---|---|---|
| References (28 pairs) | – | none | – |
| B2_SINGLE vs SUCC | −0.032 → −0.035 → −0.040 | SHRINKS | 7/15 (−0.003) |
| B2_SINGLE vs H_LJ | −0.015 → −0.015 → −0.023 | SHRINKS | 7/15 (−0.002) |
| A1_PO_16 vs SUCC | −0.030 → −0.035 → −0.037 | SHRINKS | 4/15 (−0.002) |
| A1_PO_16 vs H_LJ | −0.013 → −0.015 → −0.019 | SHRINKS | 6/15 (−0.000) |
| P1_PAIR_B80 vs P1_CONT (seed 0) | +0.024 → +0.023 → +0.014 | SHRINKS | 2/15 (+0.003) |
| P1_PAIR_B80 vs SUCC (seed 0) | −0.010 → −0.018 → −0.019 | SHRINKS | 6/15 (−0.001) |
| P2_PAIR_B80 vs P2_CONT (8 cells) | +0.039 → +0.032 → +0.032 | SHRINKS | 1/15 (+0.001) |
| P2_PAIR_B80 vs SUCC (8 cells) | +0.007 → −0.003 → −0.004 | SHRINKS | 2/15 (+0.001) |
| P2_PAIR_B80 vs H_LJ (8 cells) | +0.024 → +0.016 → +0.014 | SHRINKS | 2/15 (+0.002) |
| P2_PAIR_B50 (8 cells) | – | SHRINKS vs all 9 | 0–5/15 (−0.002 to +0.002) |
| P2A_PAIR_B80 vs SUCC (8 cells) | +0.012 → +0.001 → +0.004 | SHRINKS | 2/15 (+0.001) |
| P2A_PAIR_B80 vs P2A_CONT (8 cells) | +0.047 → +0.037 → +0.040 | SHRINKS | 1/15 (+0.001) |
| P2A_PAIR_B80 vs P2A_COMP_B80 | – | none | 2/15 |

Witness learning-curve flags are only partly supported. Tree readers (OUT_HGB, HGB15/31, ET5) gain more on the witnesses
(+0.025/+0.030 vs +0.019 for SUCC); linear and MLP readers gain about the same. "The 4,000-person grid understates the witness gap"
holds for tree readers only.

## Task 2 — sign stability

All outputs come from the frozen rule (ohn.rules, unchanged). My labels match ohn.grid_summary on every checked pair.

Over the 163 complete 8-cell pairs:
- Labels: NOT_POSITIVE 108, SIGN_CHANGES 28, POSITIVE_NOT_STABLE 19, STABLE_POSITIVE 8.
- Spread: median sd_total 0.0118, made of reader/split 0.0082 and bootstrap 0.0089 in quadrature. The k0-vs-k1 (reader-seed) SD is
  0.0052 and the R1-vs-R2 split SD is 0.0073.
- Sign changes: 63 pairs change sign. In 22, a positive conditional interval (b − 2 × bootstrap SE > 0) sits in one cell while the
  sign changes in another.
- In 4 pairs, intervals exclude 0 in opposite directions: SUCC vs B2TWIN (+ in S0R1, − in S1R2); P2A_PAIR_B80 vs SUCC (+ in
  S0R1/S0R2k1, − in S1R1); P2A_PAIR_B80 vs RETIRED_B3CTL; INC_H3_PO vs B2_SINGLE.
- A readers-fixed interval misses the matched-cell mean in only 1.9% of cells. For pair benefits, the bootstrap covers SCORE
  sampling; the extra spread comes from winner switching and seed-dependent recipes.
- The 8 STABLE_POSITIVE pairs: H3+O1 vs H_LJ and vs TWJ; SUCC vs TWJ; P2A_PAIR_B80 vs A1_PO_16 and vs P2A_CONT; P2_PAIR_M90 vs
  B2_SINGLE, vs P2_CONT and vs TWJ.

STABILITY TABLE PER CANDIDATE. Each row gives the 8-cell label and mean benefit, the seed-0 and seed-1 means, and the fixed-recipe
probe per seed (recipes positive of 18, median). Benefit = comparator AUC − candidate AUC; positive favours the candidate.

[The delivered text ends here (platform termination). The table is in RR_FINAL__CANDIDATE_TABLE.csv / .json.]
