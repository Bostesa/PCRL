# Challenger review of the two designs (campaign `pcrl_input_and_pair_release_v1`)

Challenger. Read 02:03-02:22Z on 2026-10-22 (`date -u`); written from 02:22Z. Reading order of `roles/coordinator/CHALLENGER_BRIEF.md`
followed: the design brief and its correction note, `PROTOCOL.md` and `FAILURE_MAP.md`, `roles/designer_a/` (PROPOSAL, mech_sketch,
FEASIBILITY*.json, feasibility.py, inspect_inputs.py, HANDBACK), `roles/designer_b/` (PROPOSAL, mech_sketch, SPEND_GRID.json,
HANDBACK), then `ipr/ctlkit.py`, `ipr/paths.py`, `ipr/ia/sfg.py`, `ipr/ctl/twin.py`, `cards/README.md`, `cards/CTL_TWIN_v1.md`,
`roles/statistical_reviewer/RULES.json`, `nrm/fa/prc.py` (BUDGETS, HGB, fit_teachers) and `nrm/rules.py`
(`frontier_comparison`, `recipient_screen`, `RECIPIENT_LIMIT`).

**Not opened:** any label file, anything of MODEL_SELECT, any reader cache, any `.private/*/runs` or access content.
**Numerical work:** one check script in this folder, `check_support.py` -> `CHECK_SUPPORT.json`, on the admitted MODEL_FIT features
and the label-free cached out-of-fold task predictions `TASK.pkl` (V0, V1) only; **1.0 CPU-s**, one thread (cap 200). Nothing else was
run on real data; no sketch selftest was re-run.

## 0. Answers to the four questions the coordinator asked

1. **Is either proposal the wrong use of the budget; is a better operation available; are the two equivalent?** Designer A's family is
   the right pick and is mis-instrumented (findings 3-8). Designer B's *registered candidate* is the wrong use of the budget: its
   targeting layer is a diluted form of the closed HQ recipe whose measured outcome was null, and B chose it for a protocol reason
   that no longer exists (findings 10-13). The two proposals are **not equivalent** (finding 1) and SFG is **not** a closed recipe
   (finding 2); CSF-T is not a re-labelling either, but its claim space is one narrow sentence (finding 11). A materially different
   operation that both missed, and the only one in either space that attacks the campaign's dominant measured failure, is a
   **cross-recipient keyed swap** (finding 20). It is unlikely to reach 0.020 either; **no operation in either space plausibly does**,
   and both designers' bottom line is correct.
2. **SFG's control, budget, grid.** The gating SBC is the **permuted-SEX ladder** (findings 5, 6); BLIND is not an SBC path at all
   (it zeroes the component) and is the ablation reported beside; PSEUDO2 / ATTR is not triggered and should not be built. Its
   strength budget must **not** be allowed as written (finding 3): it is not one of S1-S3 and is looser than S3 on exactly the two
   clauses its own feasibility says will bind. A ladder of discrete steps is **not** compatible with the bracket-fineness rule
   (finding 7); the fix is predeclared half-steps and the grid rung k = 1. **PROJ is not needed and should not gate** (finding 8).
3. **CSF-T's D2 reading and D5.** The D2 reading is meaningful only in the rules' narrow `RELEASE_ONLY_OPERATION` sense ("a better
   release-only map than ZTILT at matched spend"), and on B's own numbers it is predicted to be NOT_SEPARATED with the point estimate
   on the wrong side (finding 11). The three D5 controls are **not** well defined (two are unimplemented one-liners at risk of
   collapsing by construction) and are **not** worth 6 configurations for a 0.001 effect (finding 14).
4. **Allocation.** Section 4 below: 2 twin + 20 core + 12 reserve, with 14 left unspent; the 8 configurations the proposals ask for
   that buy nothing are named.

## 1. Equivalence, and against the closed recipes

1. **ADVISORY — the two proposals are not the same operation.** SFG releases `f_inc(h_k(x)), f_occ(h_k(x))` with both task models
   **refitted** on a coarsened partition of x, deterministic, continuous-valued, and uses SEX as a **12-number aggregate ordering
   statistic** of candidate steps. CSF-T keeps the twin and, for a keyed fraction of people, replaces both wires by one of 10 fixed
   decision-cell rows, using a **person-level** release-level propensity aggregated into 10 rates. Different released function classes,
   different SEX information, different stage. No redirect on equivalence grounds. The shared deep structure is worth stating in both
   cards: both are *information removal by merging*, and both run at 0.3-0.7 pair AUC per nat against the closed tilts' 2.4 (benchmark,
   DB_PTC_R1), which is the reason neither can reach 0.020 inside the utility budget.
2. **ADVISORY — SFG is not a closed recipe.** Field generalization is in `FAILURE_MAP.md`'s "untested operations" list; it is no tilt,
   standardization, reweighting, posterior-cell routing, channel, decoder, controller or leaf correction. A's argument that any
   SEX-dependent training target or row weight re-parametrizes the closed STD / IPW / tilt family is accepted as written; it is an
   argument about the population limit, not about a 20,000-row HGB, so the card should state it as a design choice, not a theorem.

## 2. Designer A (SFG): REQUIRED changes

3. **REQUIRED — the strength budget is not predeclared and is looser than S3 where it matters.** PROTOCOL 4: "A card chooses one of
   the three predeclared budgets before its first fit". SFG's budget (income and occupation NLL 0.012, accuracy 0.008, men's and
   women's occupation Brier 0.0045) equals S3 except that **income NLL is doubled (0.012 vs 0.006) and income accuracy loosened
   (0.008 vs 0.006)** — precisely the two clauses A's own feasibility says bind ("at k = 3 the net plug-in income accuracy loss
   (0.009) already exceeds the 0.008 budget"). Against the twin's published MODEL_SELECT room (limit minus the twin's excess: income
   NLL 0.0186 / 0.0181, income accuracy 0.0143 / 0.0110, men's occupation Brier 0.0089 / 0.0073, women's 0.0147 / 0.0137) A's budget
   consumes 66% / 73% / 62% of version 1's room on MODEL_FIT alone, before any MODEL_FIT -> MODEL_SELECT transfer error; S3 consumes
   33% / 55% / 62%. Choose **S3**, with the consequence stated in the card (finding 4), or amend the protocol before registration with
   a named, justified fourth budget — a card-level choice cannot do it. Also correct the card's justification sentence: the twin is
   **0.0086 / 0.0081** better than U0 on income NLL, not "about 0.010-0.012 per task"; occupation is 0.0161 / 0.0128.
4. **REQUIRED — the strength grid must contain k = 1, and the likely CAND is near-trivial.** The selection rule needs *every smaller
   grid strength* to pass, and GRID = (2, 3, 4) has no rung below k = 2. A's own plug-in income spend at k = 2 (MAR suppressed) is
   0.0056 nats, so under S3 (income NLL 0.006) the estimate + 2 SE almost certainly fails and the family reads **D1
   NO_FEASIBLE_STRENGTH** with no fallback. Add k = 1 (MAR3 alone; plug-in income spend 0.00022). Disclose in the card that k* = 1 is
   the modal outcome under S3 and that its plug-in pair change is -0.0003, i.e. a near-twin candidate. Note in mitigation that the
   plug-in spends come from a smoothed surrogate (a cross-fitted HGB *regression of the cached posteriors*), and that coarsening the
   inputs of a boosted-tree model can regularize: the real out-of-fold spend of a refitted HGB may be materially smaller or larger, in
   an unpredictable direction. That is the reason to run the ladder and the selection **first** (they cost CPU, not configurations) and
   to commit control points only afterwards (section 4).
5. **REQUIRED — BLIND is not an admissible SBC path.** PROTOCOL 4 (ii) requires the SEX-dependent fitted input to be replaced "by a
   declared SEX-independent statistic of the same type and dimension ... never a constant, a zero or the removal of the component".
   `order='blind'` sets dI identically to 0 and changes the ordering criterion to -dT: that is the removal of the component. BLIND is
   the family's **targeting-off ablation**, reported beside; it cannot gate D2.
6. **REQUIRED — the gating SBC is the permuted-SEX ladder, and PSEUDO2 / ATTR must not be built.** The permuted path *can* reach
   CAND's spend (its rungs span 0.0004 to about 0.038 nats of plug-in spend), so the protocol's fallback condition for PSEUDO2 is not
   met and ATTR is out of protocol as a gate and a waste as a beside (2 configurations). Record two expected properties of the permuted
   ladder in the card: (a) with a permuted target dI is noise about 0 while dT spans 0.0004-0.024, so the ratio key dI/dT amplifies
   noise by up to 2,700 and the permuted ladder will almost surely start at the cheapest step (MAR3) and keep preferring cheap steps —
   it will be a **near-duplicate of BLIND**, so its points may alias BLIND's and must be counted once; (b) because the order is
   noise-driven it will differ by version: expect `SBC_CONSTRUCTION_DIFFERS_BY_VERSION`. Consequence to accept openly: SFG's SBC tests
   only "ordering by a real SEX signal beats ordering by noise", which is the targeting question but a weak one here.
7. **REQUIRED — a discrete ladder is not compatible with the bracket-fineness rule; declare the refinement now.** A's own plug-in
   spends step by about 2x per rung (0.0004 -> 0.0085 -> 0.0181 -> 0.0346), the same failure the dual-design release-only grid had
   (0.90x to 2.08x). The fineness rule wants a saved point in [0.75 s, s] and one in [s + 2 SE, s + 2 SE + 0.25 s]; a 2x ladder meets
   neither side except by luck. Under `RULES.json` `grid_fineness` this is reported, never gating — but the bracket rule then takes
   Q* at about 2x CAND's spend, and an overspending Q* removes **more** information, so the SBC comparison is biased **against** CAND
   and the result is uninformative rather than negative. What should happen: promote A's R2 halves into the **registered** grid
   (WK3 -> merge only <35 / only >=45; COW3 -> government-only / self-employed-only; MAR1 -> never with previously married first) so
   that the ladder has a rung near CAND's spend on each side, and record `CONTROL_GRID_FINENESS_INCOMPLETE_SBC` with that reading if it
   still fails. A continuous dial would be better still (finding 19).
8. **REQUIRED — PROJ must not gate, and its plug-in numbers are a capacity artifact.** My check (`CHECK_SUPPORT.json`): on 20,000
   MODEL_FIT people there are **17,744 distinct six-field records**, and the twin's released pair takes **19,191 / 19,173 distinct
   values** (V0 / V1) with **92.9% / 92.7% of people on a unique wire**; of the 620 / 631 wire groups with more than one person,
   588 / 604 are a single record. The twin's released pair therefore **identifies the record** for almost everyone, so
   E[SFG release | twin release] *is* SFG up to the regression's capacity: at full capacity PROJ is an exact alias of CAND, and at the
   feasibility script's capacity (120 iterations, 31 leaves) it is an arbitrary smoothing whose strength is set by hyperparameters
   rather than by a declared spend grid. Its reported "spend -0.0006, pair -0.0078" is that artifact, not a dominating control. Under
   the protocol SFG uses no person-level SEX statistic, so its RO path **is** the generic ZTILT; declare
   `ro = {kind: GENERIC_ZTILT, operation_release_only: false}` and `operation_sex_blind: false`. Drop PROJ (2 configurations) or keep
   it only as a beside diagnostic with its learner capacity recorded and the alias risk stated.
9. **REQUIRED — three smaller card corrections.** (a) A's "Gate G1" collides with the protocol's G1 and, worse, is written as a stop
   rule; PROTOCOL 2 says MODEL_FIT reader checks named in a card "are descriptive and never stop the comparison" — rename it and make
   it descriptive. (b) The selection must be computed from the rung's **own** construction (the coordinator's `fit_rung`: T* re-selected
   by the inner household CV, out-of-fold on `NRM_FA_xf_S<v>`, the cached twin's folds), not from the sketch's fixed iteration counts
   on its own folds (`spend_clauses`, seed 104729 + version) — otherwise the selected spend is not the released CAND's spend and is not
   comparable with the cached twin's out-of-fold. (c) The card must not sell the data-processing ceiling as a benefit: at the feasible
   rungs the partition is barely coarser than x (MAR3 leaves 17,679 of 17,744 records distinct, MAR1 16,470), and A's own ceiling at
   k = 3 (0.656) is **above** the twin's realized pair AUC (0.6506), so the removal property has no bite at any feasible strength.
   State it as "no bite below k = 4, which is task-infeasible".

## 3. Designer B (CSF-T): REDIRECT

10. **REQUIRED — the stated reason for choosing the targeted form is no longer true.** B writes "the protocol rules out a SEX-blind
    candidate ... so it cannot meet D2 by construction". `roles/statistical_reviewer/RULES.json` `change_record` item 2 (added about
    02:00Z, before registration) gives a declared SEX-blind operation its own route: SBC and substituted RO are degenerate by
    construction, **D2 gates on TWIN and SPEND only**, met label `SEX_BLIND_IMPROVEMENT_BEYOND_TWIN_AND_SPEND`. The premise of B's
    choice is gone.
11. **REQUIRED — the targeting layer is the closed recipe, diluted.** CSF-T's cells are the twin's own released decision cells and its
    priority is `Var_c(z) / cost` with `z = P(SEX | twin release)`: that is the published privacy-funnel / IBSI objective restricted to
    a function of the task posteriors, which is exactly **HQ** (row 12: "No reduction beyond any control; a task-score-only grouping
    (HQ) reproduced it"), with a 10-cell codebook instead of 16 x 64, cell means instead of fitted decoders, and only a keyed fraction
    (pi about 0.05) of people routed. Every difference from HQ except the keyed fraction makes it **weaker** than an operation already
    measured to produce no reduction beyond any control. If the card keeps the targeted form it must argue this explicitly against rows
    11-12 and accept the rules' narrow scope (`RELEASE_ONLY_OPERATION`: "a better release-only map than ZTILT at matched spend, never
    information beyond the release"), with B's own P6 predicting the point estimate on the wrong side of ZTILT.
12. **REQUIRED — on B's own numbers the targeting costs more than it buys.** CSF-T beats uniform CSF by +0.0023 / +0.0007 (pi 0.10) and
    +0.0040 / +0.0006 (pi 0.15) — 1/18 to 1/4 of one se_agg — while raising the worst S2 ratio from 1.39 to 1.94 at pi 0.10, i.e. it
    **halves the affordable strength**, and its allocation lands on a single cell that differs by version (income 1 / occupation 0 at
    rate 0.28 in V0; income 0 / occupation 2 at rate 0.71 in V1) with an oracle allocation adding only +0.002 / +0.0005. An SBC path
    against it compares two essentially arbitrary single-cell allocations: 4 configurations for a true effect near 0.001. **Redirect:**
    register the **uniform coordinated keyed collapse (CSF)** as CAND with `operation_sex_blind: true` (so SBC and substituted RO are
    degenerate and not built, a GENERIC_ZTILT path reported beside), and keep CSF-T as the *targeting* ablation. This also fixes the
    cost-normalization instability B reports (the per-task spend split differs by version, inviting
    `CAND_STRENGTH_DIFFERS_BY_VERSION`).
13. **REQUIRED — choose S3, not S2, and say which component binds.** B's own `SPEND_GRID.json` for uniform CSF gives
    `binding_s2 = men_occ_brier` at **every** strength (s2_ratio 0.35 at pi 0.025, 0.68 at 0.05, 1.01 at 0.075), so S2 caps pi at about
    0.05 and the expected effect at about 0.004 below the twin. S3 is predeclared and roughly doubles the affordable strength (s3_ratio
    about 1 near pi 0.15), where the plug-in effect is about 0.009 — still unresolvable, but twice as informative for the same
    configurations. The guard the coordinator asked about is confirmed as the binding one for this family: **the men's occupation Brier
    guard**, in both proposals' own numbers (B at every pi; A at k = 4, women's +0.0095).
14. **REQUIRED — the D5 controls are not well defined and are not worth 6 configurations.** `cell_indep` is implemented and clear.
    SEQ_OI and SEQ_IO exist only as one sentence each, are **not** in `mech_sketch.MODES`, and do not say whether the key is shared,
    whether "matched spend" is matched per recipient or in total, or how the second stage's rate is selected. Structurally they collapse
    by construction: with a shared key and one rate per joint cell a sequential order **is** CSF-T
    (`NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL`, no joint-design claim possible), and without a shared key it is the separate design
    (reduced test). `RULES.json` D5 requires that declaration, with its construction argument, in the card before the first fit.
    Since B's own evidence puts the entire coordination effect at **+0.0011 / +0.0001** (pi 0.15), the right choice is: declare joint
    design **not built** (D5 = `NOT_ASSESSED_MISSING_CONTROLS`, which PROTOCOL 6 explicitly allows) and spend 2 configurations on
    **IND as a second ablation** (coordination off), which is the only part of the joint-design question with a mechanism behind it.
15. **ADVISORY — B's closest-implementation and failure claims should be trimmed.** CSF at a feasible pi leaves 95% of people with the
    twin's wire bitwise, and the collapsed 5% are exactly identifiable to a table reader (10 repeated rows). So the pair effect is close
    to a 5% dilution of the twin toward the decision floor (0.6506 -> about 0.647, matching B's own estimate) and failure (4) is neither
    addressed nor worsened; the claim that it "addresses" (2) and (3) holds only for the collapsed fraction. Keep the accurate parts:
    income accuracy exactly 0 (failure 5) and release-only by construction (failure 6).

## 4. Shared control code, utility, recovery, prior art, budget

16. **REQUIRED (coordinator code) — the DQ dither must not be family-tagged.** `ipr/paths.py` line 51 builds
    `tag = f'IPR_DQ_{family}_S{int(version)}'`, and `ctlkit.dq_release` keys the subtractive dither on that tag. Both families share
    CTL_TWIN, so the same delta would produce **different wires** for IA and IB and the campaign would pay twice for an identical
    control. Use a family-independent tag (`IPR_DQ_S<v>`) so equal deltas alias exactly and are saved once (PROTOCOL 4). The RO path
    already has no family tag and will alias across families by construction. This is the single largest budget saving available (up to
    8 configurations).
17. **ADVISORY — the predeclared budgets omit 8 of the 24 G1 clauses.** S1-S3 (and both proposals) carry no overall-Brier and no
    income-Brier component, while G1 has summed Brier excess <= 0.005 per task per reference (U0 **and** Ucal), and the published
    headroom the coordinator supplied is vs U0 only. The men's and women's guards bound occupation Brier approximately; nothing bounds
    income Brier, and income is where SFG spends. Record this as a known gap of the inherited budgets in both cards rather than
    inventing a new component.
18. **ADVISORY — recovery, and where each family is predicted to fail.** For SFG the binding D2 clause is unlikely to be the pair
    benefit: its plug-in income view sits about 0.02 **above** ZTILT's at matched spend (ZTILT flattens income, plug-in income view
    0.526 at scale 0.5), and `nrm.rules.recipient_screen` fails at a point mean above 0.010, so expect
    `RECIPIENT_GATE_FAILED_VS_RO` before any pair reading — the RO path's 4 configurations buy a near-certain, and correctly recorded,
    failure of a gate rather than an informative pair comparison. For CSF the leakage cannot move to the other recipient (both wires
    collapse for the same people) but is simply diluted; LOOKUP is favoured by the 10 finite rows and should be watched. One
    campaign-wide statement belongs in both cards: because the twin's released pair identifies the record for about 93% of people
    (finding 8), a release-only control's power to explain a deterministic operation's gain is set by the **control learner's capacity**,
    not by information; `RULES.json`'s RO scope note already says this, and no card may read RO_FRONTIER as an information bound.
19. **ADVISORY — prior art and novelty.** Both novelty statements are NOT_CLAIMED and both cite the right literature (A: Samarati and
    Sweeney 1998, Sweeney 2002, LeFevre et al. 2006, the privacy funnel of Makhdoumi et al. 2014, Calmon et al. 2017; B: Willenborg and
    de Waal 2001, Hundepool et al. 2012, PRAM of Gouweleeuw et al. 1998, Warner 1965, complementary suppression of Cox 1980). D6 stays
    NOT_CLAIMED for both. Neither did a literature search in this session and both say so; that is sufficient here.
20. **ADVISORY — the operation both designers missed, if the coordinator wants one positive shot.** Every family in row 14 and both
    proposals reduce **own-view** information, while the dominant measured failure is **pair super-additivity** (TWJ: views 0.5606 and
    0.5600, pair 0.6476). The only construction that attacks the link itself is a **cross-recipient keyed swap**: for a keyed fraction
    lambda of people the income recipient keeps `p_I(x)` and the occupation recipient receives `p_O(x')`, where x' is a keyed donor from
    the same occupation **decision cell** (so the decision, and therefore accuracy, is preserved exactly, as in CSF, and each
    recipient's marginal wire distribution and own-view leakage are left almost unchanged while the pair link is destroyed for those
    people). It is SEX-blind (so it gates on TWIN and SPEND under the new rules route), it is release-only, it is not in the closed list
    (row 4's channels replace a vector by a class medoid per recipient; SHUF was a control over donor **tables**, not a released wire),
    it is deployable from the six fields plus a fixed donor bank, and it has a continuous strength dial, which satisfies the
    bracket-fineness rule that both proposals fail. Honest estimate: the cost per swapped person is roughly twice CSF's cell-mean
    collapse (a donor is noisier than the cell mean), so the affordable lambda is about half, and the expected pair effect is 0.005-0.015
    against DQ at matched spend — **short of 0.020**, but the largest plausible value in this campaign and the only test of the
    super-additivity failure. If taken, it is the one `FS_` replacement family of PROTOCOL 7 and needs this challenger and the
    statistical reviewer to read its card first. A second, cheaper instrumentation idea for IA, if A's family is kept: replace the
    discrete ladder by **within-field randomized response (input PRAM)**, which gives the same data-processing removal with a
    continuous dial; its efficiency is SFG's, so it buys instrumentation, not effect.
21. **REQUIRED — budget and allocation.** 48 configurations, 2 of them the shared twin; 12 CPU-hours; two one-thread jobs; research
    window to 09:19Z (7.0 h of wall clock remained at 02:22Z). Recommended allocation, after findings 3-16:

    | Block | Contents | Configurations |
    |---|---|---|
    | CTL | CTL_TWIN (committed card) | 2 |
    | IA core | CAND 2 (k* per version), SBC permuted ladder with the predeclared half-steps 4, ABL = BLIND at matched spend 2 (may alias the SBC points and then cost 0) | 8 |
    | IB core | CAND 2 (uniform coordinated CSF at S3), ABL targeting = CSF-T 2, ABL coordination = IND 2 | 6 |
    | Shared paths | SPEND (DQ) bracket 4 and RO (ZTILT) bracket 4, built **once** on the shared twin with family-independent tags; add up to 4 more only if the two CAND spends are not co-bracketed | 8 (+4 contingency) |
    | Reserve | one revision per family (A's half-step refinement R2 4; B's R1 4), or the `FS_` swap family (CAND 2 + ABL 2, reusing the shared paths) | 12 |
    | Unspent | | 14 |

    **Do not build** (8 configurations the proposals ask for that buy nothing): SFG's ATTR / PSEUDO2 path (2; not triggered, finding 6),
    SFG's PROJ (2; capacity artifact, finding 8), CSF-T's SBC path as a gate (4; true effect about 0.001, finding 12) — and the 6 D5
    configurations (finding 14), which are not in the table above.
    **Order of work:** (i) commit CTL_TWIN and both cards; (ii) run IA's ladder and both families' MODEL_FIT strength selections, which
    cost CPU but **no configuration**, in the two lanes (V0 and V1 concurrently; IA's ladder is the long pole at roughly 1.5-2.5 CPU-h
    per version with the task part shared across orderings); (iii) only then commit the control points, at the spends the selection
    actually produced; (iv) if IA reads NO_FEASIBLE_STRENGTH at S3, its control configurations are never spent and the reserve covers
    the half-step revision or the `FS_` family. CPU estimate for the whole table: about 5-6 of the 12 CPU-hours including the reader
    grid (about 200 CPU-s per configuration) and the D2R robustness cells, so CPU is not the binding constraint — **resolution is**
    (se_agg 0.009-0.018; 2 se_agg = 0.018-0.036 against expected effects of 0.001-0.013).

## 5. Verdicts

- **Designer A (SFG, family IA): PROCEED_WITH_REQUIRED_CHANGES.** Required before the card is committed: findings 3, 4, 5, 6, 7, 8, 9
  (predeclared budget S3 with the corrected headroom sentence; grid rung k = 1 and the predeclared half-steps in the registered grid;
  the permuted-SEX ladder as the only gating SBC, BLIND as the ablation, ATTR not built; `ro = {GENERIC_ZTILT, release_only: false}`,
  `operation_sex_blind: false`, PROJ dropped or beside; the reader gate made descriptive and renamed; selection computed from
  `fit_rung`; the ceiling claim corrected). With those changes it is a clean, cheap closure test of the input side, with D1
  NO_FEASIBLE_STRENGTH and NOT_SEPARATED as the registered-expectation outcomes.
- **Designer B (CSF-T, family IB): REDIRECT.** Register the **uniform coordinated keyed collapse (CSF)** at budget **S3** as CAND,
  declared `operation_sex_blind: true` (D2 gates on TWIN and SPEND; a GENERIC_ZTILT path reported beside); keep **CSF-T as the
  targeting ablation** and **IND as the coordination ablation**; declare the joint design **not built** (D5 =
  NOT_ASSESSED_MISSING_CONTROLS) and do not fund SEQ_OI / SEQ_IO. Findings 10, 11, 12, 13, 14 are the required substance of that
  redirect. If the coordinator wants one shot at a material effect rather than a second closure test, the `FS_` cross-recipient keyed
  swap of finding 20 is where the remaining budget belongs.
