# Family C (SEX-mixture marginalization, MIX): card revisions, appended as they happen

Investigator C, 2026-10-17. Exploratory development on previously used Florida ACS 2019 development records; MODEL_SELECT is a
development role used for selection feedback. Code: `amd/fam_c.py`. Releases: `.private/pcrl_autonomous_method_development_v1/runs/C/`.

## C1 (initial version, as on the card), 2026-10-17 ~17:10Z

- Operation: the T_TREE CatBoost recipe (depth 4, lr 0.05, at most 800 trees, l2 3, thread_count 1, T_TREE CatBoost seeds, the same
  15% household-disjoint internal early-stopping split, refit on all 20,000 MODEL_FIT people) with SEX appended as a 146th input column at
  training time only. Release on every role: w0·P(y | x, SEX=0) + w1·P(y | x, SEX=1), with w = MODEL_FIT SEX shares (0.5103 male /
  0.4897 female on seed 0). Income and occupation are both marginalized.
- SEX-blind control: the identical recipe without the SEX column was refitted and reproduced the stored T_TREE teacher bitwise on every role
  (teacher hashes equal, both tasks). The stored T_TREE is therefore the exact SEX-blind control of the same class, capacity and data.
- Seed-0 result (MODEL_SELECT): 12/12 utility clauses pass (worst normalized −0.19); guard men 0.00439 (above 0.8 × cap), women −0.00797.
  Proxies: pair 0.6263 (T_TREE 0.6487; NEURAL_RAW 0.6238; H_LJ 0.6224), income 0.5479 (bar 0.5500), occupation 0.6211 (bar 0.5824).
  The reductions against T_TREE appear for OUT_LR, OUT_HGB and LOG_MLP on every view; the distinct-output counts are unchanged (7,500 pairs).

## C1 control (not a revision): matched-cost non-SEX shrinkage, 2026-10-17 ~17:16Z

- Reason: the card's control. It separates "less SEX-structured" from "less confident". It is the stored SEX-blind T_TREE with one
  temperature per task on its raw scores. Each temperature is chosen label-free so that mean KL(T_TREE || release) on MODEL_FIT
  features equals C1's (income 0.00132, occupation 0.00639 nats on seed 0). No SEX is used. Config `C1_CTRL_TEMP`.

## C2 (revision 1): marginalize income only, and pair it with the frozen eligible JOINT_TREE occupation output, 2026-10-17 ~17:16Z

- Evidence that motivated it (C1, seed 0):
  - The marginalized occupation output stays at 0.6211, 0.039 above the occupation recipient bar (H_LJ 0.5724 + 0.010). The men's
    guard is already at 0.00439. Removing the SEX-proxy pathway accounts for only ~0.02 of the occupation output's recovery, so no
    marginalization variant can close a 0.039 gap: pooling the propensity to J levels re-introduces SEX-propensity structure, and a
    different constant w only moves the burden between SEX groups.
  - Seed 1 starts the men's guard 0.0016 higher (T_TREE men +0.00059 vs −0.00103 on seed 0), so C1's occupation burden on men
    (~+0.0054 Brier) is likely to break the guard there.
  - Marginalized income, by contrast, costs almost nothing (income NLL excess vs U0 −0.00644 against the teacher's −0.00663). It already
    meets the seed-0 income recipient bar (0.5479 ≤ 0.5500).
- Change: income = the C1 SEX-aware income model, marginalized (no refit on seed 0). Occupation = the frozen JOINT_TREE E20 occupation
  output (T_TREE routing plus the stored stage B theta). This is exactly the occupation output of the eligible reference H_LJ; the
  reproduction is checked to be ≤ 1e-12 on every bank role. Config `C2_MIXINC_JOCC`.
- Question tested: H_LJ's linked pair reads 0.6224 although its occupation output reads 0.5724; its LOCAL income output supplies the
  complementary SEX evidence (coalition synergy). Does a SEX-marginalized income output, with its SEX-proxy pathway removed, supply less
  of that evidence? The matched component control is H_LJ itself: the same occupation output, with LOCAL income.
- Scope note: the occupation output is a frozen historical release (adversarial pair-aware leaf correction), not a family-C mechanism.
  Any result is a property of the hybrid. Revise or abandon if the seed-0 pair proxy is not below both references (0.6238 / 0.6224).

## Outcomes and decision, 2026-10-17 ~17:20Z

- **C1 control, seeds 0 and 1.** Temperatures were income 1.1237 / 1.1208 and occupation 1.1820 / 1.1835. The control reduced nothing.
  - Pair: 0.6489 / 0.6496 (T_TREE 0.6487 / 0.6493).
  - Income: 0.5573 / 0.5589 (identical to T_TREE).
  - Occupation: 0.6412 / 0.6366 (T_TREE 0.6426 / 0.6391).

  At matched label-free cost, "less confident" does not lower recovery. C1's reductions are therefore not a confidence effect.
- **C2, seed 0.** The pair proxy was 0.6277, not below H_LJ (0.6224) or NEURAL_RAW (0.6238), so C2 is abandoned under its own rule and has
  no seed-1 run. Diagnostic `CDIAG_TTINC_JOCC` (counts 0, fits nothing) kept the occupation output fixed and changed only the income
  output:

  | Income output | Pair proxy |
  |---|---|
  | SEX-blind T_TREE | 0.6351 |
  | SEX-marginalized (C2) | 0.6277 |
  | LOCAL (H_LJ) | 0.6224 |

  Marginalization removes about 0.007 of the income output's coalition contribution. That is about 58% of what LOCAL's adversarial
  training removes.
- **C1, seed 1.** All 12 clauses pass. The men's guard fails at 0.00640 > 0.005 (predicted from the seed-1 teacher's +0.00059 start);
  the women's guard is −0.00723. Pair 0.6368, income 0.5461 (bar 0.5476), occupation 0.6286 (bar 0.5953). The SEX-blind refit
  reproduced the seed-1 T_TREE bitwise.
- **No C3 (family stopped early, coordinator concurring).** Constant-w full marginalization is the family's maximal removal of the
  SEX-propensity pathway, and it leaves the occupation output 0.039 / 0.033 above the recipient bar. The men's guard is at 0.88 × cap on
  seed 0 and fails on seed 1. No remaining in-family change can plausibly close that gap at a feasible men's guard:
  - Pooling the propensity to J levels re-introduces SEX-propensity structure. A smooth partial shrinkage w(x) = c + λ(π(x) − c) keeps
    the propensity direction.
  - Another constant w only moves the burden between SEX groups; the release stays in the same SEX-proxy-free class.
  - Stratified per-SEX models would complete the pathway removal for interactions. They can only enlarge the SEX-specific gaps that
    create the men's burden.
