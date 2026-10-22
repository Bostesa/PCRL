<!-- Saved verbatim by the coordinator at 2026-10-21 ~17:30Z from the statistical reviewer's hand-back message (the harness refuses
report-file writes by subagents; earlier campaigns saved such reports the same way). Author: the statistical reviewer. -->

# Statistical reviewer: pre-registration review of the draft PROTOCOL (campaign `pcrl_dual_design_release_v1`)

Reviewed text: `results/pcrl_dual_design_release_v1/PROTOCOL.md`, drafted by the coordinator (SHA-256 ecc85a8f0ff2db08..., unchanged from my first read at 16:53Z through 17:24Z), together with `roles/coordinator/DESIGN_BRIEF.md` (3ea7b86e...). Written 2026-10-21 from 16:52:55Z and returned to the coordinator at about 17:26Z, before any outcome of this campaign existed. I read no label and ran no real-data computation. I read nothing under `admitted/exports`, nothing of MODEL_SELECT and no reader cache. I did not edit PROTOCOL.md.

The formal rules are in `RULES.md`, `RULES.json` and `ddr/rules.py`. Every REQUIRED finding that changes a decision is already implemented there, so the protocol and the rules agree once the wording below is adopted. If a REQUIRED finding is declined, the matching rule must be changed and re-frozen before registration. Line numbers refer to the reviewed draft.

## REQUIRED findings

**1. The reader-seed base is not new (section 5, lines 94-96).** "Reader seeds MODEL_SEED_BASE + 800 + 10 v + k ..., a new base" is false as written.
- FSA1 registered exactly these seeds (its `READER_SPLITS.json`: V0k0 = 2026102005 ... V1k3 = 2026102018) and the same bootstrap base, 2026102200 + 100 v + repetition index.
- FSA1 built no grid and read no MODEL_SELECT record (FSA1 RESEARCH_DECISION, question 7). No realized reader fit used these seeds, so they need not change.

Replace "a new base." with:
> "This base and the bootstrap base below were registered by FSA1 and never used (FSA1 built no grid). The seeds of every realized earlier grid (MODEL_SEED_BASE + 300, 400, 500, 600, 700 + 10 v + k) are forbidden, and the grid is valid only with exactly these reader and bootstrap seeds (RULES.md 9)."

**2. What a matched control must satisfy (section 4, table rows SBC and SPEND, lines 77-78; add a bullet after line 89).** The draft names the controls but not what makes them matched. "Every SEX-specific input replaced by a SEX-blind counterpart" would allow a constant or a removed component. Either one changes the operation and makes the control easy to beat. Add:
> "A matched control point (SBC or SPEND) must (i) be built on the family's own TWIN with the same MODEL_FIT people, folds, learners, seeds, number of fitted components and, for a randomized release, the same draw structure as CAND; (ii) for the SBC path, be CAND's operation with every SEX-dependent fitted input replaced by a declared SEX-independent statistic of the same type and dimension (default: a fixed-seed permutation of the MODEL_FIT SEX column across people), never a constant, a zero or the removal of the component; if the permuted path cannot reach CAND's MODEL_FIT spend at the largest grid strength, the card's declared fallback SEX-blind target (named before CAND's first fit, for example a feature- or task-derived binary of comparable task entanglement) is used, by that MODEL_FIT-only rule; (iii) for the SPEND path, be a SEX-blind spend-only transformation of the TWIN's released vectors; (iv) be named with its construction, grid and saved-point rule in the card before CAND's first fit and be fitted before any MODEL_SELECT read; (v) start at TWIN (strength 0 of every control path). A shared CTL path counts as a family's SPEND path only if it is built on that family's TWIN."

**3. Control-point selection, bracketing and equal effort (section 4, lines 82-88).** Two problems.
- (a) The draft contradicts itself. Line 85 says a control path is chosen "by the same kind of rule or by spend matching to CAND". Line 87 says "the control paths use ... the same selection rule".
- (b) Nothing guarantees a control point that spends at least as much as CAND on MODEL_SELECT. Without one, the inherited frontier rule returns NOT_SEPARATED_FROM_TASK_SPENDING (reason CANDIDATE_SPENDS_MORE_THAN_EVERY_CONTROL_POINT). A control point matched exactly on MODEL_FIT brackets CAND on MODEL_SELECT only about half the time, so D2 would often fail for lack of a bracket rather than on evidence.

Replace lines 85-88 ("A control path is chosen ... the same reader cells.") with:
> "Each control path saves exactly two points, chosen on MODEL_FIT (nested held-out paired total-NLL spend over the TWIN) before any MODEL_SELECT read: the largest grid strength whose spend is at most CAND's, and the smallest grid strength whose spend is at least CAND's plus 2 household-clustered SE of the paired difference (if none reaches it, the largest grid strength, flagged CONTROL_PATH_MAY_NOT_BRACKET before MODEL_SELECT). Duplicates and exact aliases are saved once. The bracket is then read on MODEL_SELECT paired total NLL by the inherited rule. A missing MODEL_SELECT bracket reads NOT_SEPARATED with that reason. This is a limit of the control path, never evidence of equality, and no control point is added after MODEL_SELECT is read. Equal tuning effort: every control path is evaluated on MODEL_FIT on a grid of the same size as CAND's, with the same learners, folds, seeds, fitted components and budgets. The saved points differ by design (CAND: the selected strength; each control path: its two bracket points). Equal evaluation effort: every saved configuration gets the same utility evaluation and the same registered reader cells; robustness cells go to the D2R set only (descriptive)."

**4. D2 (c) must keep the recipient allowance and look below the bracket (section 6.2 (c), lines 111-112).**
- (a) NRM1 G2 required the recipient gate against both matched controls. The draft's (c) uses `frontier_comparison` alone, which checks only the pair benefit and secondary spending at Q*. A candidate could beat its SEX-blind control on the pair while one recipient's own view leaks more than 0.010 beyond the control's (measured failure 3).
- (b) The bracket rule looks only upward. On a non-monotone path (for example, coarsening that helps table readers: measured failure 4), a cheaper control point can be resolved less recoverable than CAND. CAND is then not on the control frontier even when it beats Q*.

Replace (c) with:
> "(c) against the SBC path and the SPEND path (each = TWIN followed by its saved points; aliases once), each separately: `DOMINATES_CONTROL_FRONTIER` (spend-bracketing rule, `nrm.rules.frontier_comparison`), the recipient gate against the bracketing point Q* (both views, both versions), and no cheaper path point with a resolved lower pair AUC than CAND (pair benefit upper bound < 0 and both version means < 0)."

**5. D2 for an infeasible candidate, and the outcome classes (sections 6.2, 6.8 and 7; lines 110, 125-127, 134-135).**
- (a) Section 7 says that after a task-feasibility failure "the protection comparison is reported as not run, never as negative". Section 6.2 lists G1 only as conjunct (a), which leaves both MET_WITHOUT_TASK_FEASIBILITY and a negative partial label possible.
- (b) The class list has gaps, and each one would fall into INCOMPLETE:
  - CAND fails G1 while a smaller saved strength passes;
  - a control path the budget forbade (section 7 says "reported as not run");
  - TWIN fails G1 (TASK_ONLY_FEASIBLE_NO_PROTECTION requires "D1 for TWIN and CAND").
- (c) "G1 fails for CAND at every fitted strength" is ambiguous, because CAND is one strength.

Add to 6.2 (a):
> "if CAND fails G1, D2 is NOT_ASSESSED_TASK_INFEASIBLE (not run; never negative, never protection) and D3 / D4 carry the same label."

Replace 6.8 with:
> "Outcome class per family reading (first applicable): `TASK_INFEASIBLE` (no grid strength met the MODEL_FIT budget, or CAND and every other saved nonzero strength fail G1); `SELECTED_STRENGTH_INFEASIBLE` (CAND fails G1, another saved nonzero strength passes; reported, never promoted to CAND); `INCOMPLETE` (the class cannot be determined from the records: D1 or D2 incomplete or not computed for grid validity, or D2 met with D3 undecided); `PROTECTION_NOT_ASSESSED` (a matched control path not run for budget); `CANDIDATE` (D2 and D3); `COMPONENT` (D2 without D3); `TASK_ONLY_FEASIBLE_NO_PROTECTION` (CAND passes G1, D2 not met; a TWIN failing G1 is the flag `TWIN_INFEASIBLE`). Flag `D3_WITHOUT_D2` (references beaten without separation from matched controls: not a candidate)."

**6. Registered versus adaptive readings, and multiplicity (section 6 line 107, section 7 lines 131-133, sections 10 and 11).** As drafted:
- a revision's reading is "labelled with its revision", but nothing stops it from being quoted as "the family's result";
- nothing makes the replacement family, or a card committed after a MODEL_SELECT outcome, visibly adaptive;
- nothing states how many looks lie behind a positive result.

With two families, up to two revisions each and one replacement family, there can be up to nine D2 looks. One AGG_POSITIVE comparison passes a true null in about 0.07-2% of replicates under the moderate NRM1 laws, and about 8% when one version or split dominates (NRM1 RULES 5.5). The chance that at least one of K looks passes is up to K times that.

Replace line 107 with:
> "Per family, at CAND of its registered card (revision 0, committed before any MODEL_SELECT outcome of this campaign). Every other reading (a revision, the replacement family or a late card) is read under the same rules, carries the tag `ADAPTIVE_<...>` (card committed after a MODEL_SELECT outcome) or `REVISED_<...>` (before) as a prefix of every label (RULES.md 7), is reported beside its family's registered reading and is never presented as a registered result."

Add to section 7 after "(disclosed as adaptive)":
> "A revision triggered by a MODEL_SELECT outcome is adaptive as a whole, even though its parameters are chosen on MODEL_FIT; its G1 on the same MODEL_SELECT people is selection-dependent."

Add to section 11:
> "a non-registered reading presented as registered; a campaign-level statement without the number of D2 looks and their registration status (`ddr.rules.campaign_summary`)."

**7. The budget does not cover revisions with their own controls (sections 7 and 8, lines 131-138 and 142-145).** One configuration is one (family, config, version).
- With two saved points per control path (finding 3), one complete D2 reading costs TWIN + CAND + 2 x 2 control points + ABL = 7 releases = 14 of a family's 18 configurations.
- A revision with its own new SBC and SPEND points needs 5 releases = 10 configurations. That does not fit in the family's remaining 4; it would use those 4 plus the whole replacement / reserve allocation of 6.
- One revised CAND with reused controls does fit.

Unless the protocol says when controls may be reused, "at most two revisions" promises readings that would almost all be NOT_ASSESSED_CONTROLS_NOT_RUN. Add to section 8:
> "One complete D2 reading costs TWIN, CAND, two saved points per control path and ABL: 7 releases = 14 configurations; each card states this count. A revision reuses the family's SPEND path whenever its TWIN is unchanged, and the SBC path only if its operation and TWIN are unchanged (the bracket is read as usual). Otherwise its D2 is NOT_ASSESSED_CONTROLS_NOT_RUN (class PROTECTION_NOT_ASSESSED) unless new control points fit the budget."

**8. Per-version strength selection and the definition of CAND (section 4, lines 82-85).** The default rule does not say:
- whether the strength is selected per version or jointly;
- what happens when the two versions select different strengths;
- what happens when one version has no qualifying strength.

This feeds the outcome class ("every fitted strength"), the strongest-release tie rule and the aggregate, which pools the two versions. Add after "... income accuracy loss 0.004.":
> "The rule is applied per version (as `nrm.fa.prc.select_path`); CAND is the pair of per-version selections, recorded, flagged CAND_STRENGTH_DIFFERS_BY_VERSION if they differ. If either version has no qualifying nonzero strength, the family has no CAND (D1 NO_FEASIBLE_STRENGTH, class TASK_INFEASIBLE; no fallback strength or budget). A card may instead fix one strength for both versions (the smaller of the two selections), stated before the fit."

**9. D2R definition (section 6.3, lines 115-118).** "Label ROBUSTNESS_CONFIRMED / ROBUSTNESS_NOT_CONFIRMED, with the refit-effectiveness check" leaves three things open:
- whether an ineffective refit can still be "confirmed";
- which comparators apply when the strongest release is not CAND;
- the minimum set of the effectiveness check.

Replace the second sentence of 6.3 with:
> "Per comparator (TWIN and the release's own bracketing controls, recomputed by the same bracket rule from its MODEL_SELECT paired costs; NEURAL_RAW and H_LJ reported beside), ROBUSTNESS_CONFIRMED only when `nrm.rules.robustness` confirms on the merged registered + robustness cells AND the refit is REFIT_GENUINELY_EFFECTIVE (R3 / R4 assignment hashes distinct from each other and from R1 / R2, SCORE overlaps recorded in [0, 1), and grid validity on the eight seeds k0-k3 with both seed pairs judged separately, on the minimum set {release, TWIN, bracketing controls}); otherwise ROBUSTNESS_NOT_CONFIRMED with reasons. R3 / R4 are new splits of the same people: robustness to classifier refitting, never replication. Descriptive; never changes D2."

**10. Grid validity must be registered in the protocol (section 5, after line 101).**
- In FRC1 the extra-trees explanation (ER1) was a post hoc repair after a failed validity check. Here it is adopted before any grid output, in a narrower form. The protocol should say so, so that applying it later cannot look post hoc.
- The bootstrap seed must also be checked: a re-run with another bootstrap seed changes every cell SE.

Add:
> "The grid is valid (READER_SEEDS_EFFECTIVE, RULES.md 9) only with the registered reader and bootstrap seeds and when every seed-dependent recipe was checked and every seed-pair equality is explained; on the minimum set (a reading's CAND, TWIN and saved control points) an equality is explained only for an extra-trees reader (ET5, ET25) on a view with at most 8001 // min_leaf distinct released vectors that differs on another view while another seed-dependent reader differs on the same view (FRC1's ER1, adopted prospectively and restricted to its mechanism). Otherwise every D2-D4 label is NOT_COMPUTED_GRID_INVALID, and any later repair is labelled POST HOC."

## ADVISORY findings

**11. Minimum-detectable-effect wording (section 6.2, lines 113-114).** "The minimum detectable pair benefit is about 0.015 (2 se_agg)" mixes two quantities. 2 se_agg (about 0.010 at se_agg 0.005) is the resolution threshold. About 0.015 is the benefit resolved with roughly 80% power per comparison. D2 needs three resolved comparisons plus targeting, so its power is lower. Suggested:
> "A pair benefit of about 0.015 is resolved with roughly 80% power per comparison (threshold 2 se_agg, about 0.010); D2 needs three such comparisons and has less power; a smaller effect reads 'not separated', never 'equal'."

**12. MODEL_FIT selection uses SEX and approximate SEs (section 4, lines 82-84).**
- The S2 budget includes the men's and women's Brier guards. So MODEL_FIT SEX labels enter the selection of CAND and of the SEX-blind controls, as a constraint rather than as an input of the operation. State this.
- The nested household-clustered SE treats the refitted pipelines as fixed. For this reason FRC1 adopted SE_used = max(household SE, SD of the outer-fold means / sqrt 5) (its finding 17). Consider the same for the spend-budget check of CAND and of the bracket points. It only makes selection more conservative.

**13. Stale code documentation (no protocol text).**
- (a) `ddr/devreaders.py`: the module docstring still says "reader-seed base offset 700, bootstrap seed 2026102100" and "MODEL_SEED_BASE + 700 + 10 x version + k". The constants (800, 2026102200), READER_SPLITS.json and the manifest are correct. Its selftest's "new seeds" check forbids only 300-600, not 700 (harmless for +800). The file is listed in `DEVGRID_CODE_LOCK.json`, so either fix the docstring and re-lock before the commit, or record it as a known stale docstring (the constants govern).
- (b) `ddr/refitcheck.py` (not locked) says the refit uses "the frozen strict rule (er1=False, decided 00:31Z)". That is FRC1 text. Its own (a) + (b) + (c) description lacks the extra-trees restriction and the support bound. Under DDR1, `ddr.rules.refit_effectiveness` applies the extra-trees explanation prospectively and takes no `er1` argument. Align the docstring, and have the module call `ddr.rules` instead of re-implementing the explanation.
- (c) `grid_validity` and `refit_effectiveness` require the bootstrap seeds (S<v><R>). A claims module copied from `frc/claims.py` must pass them, or the grid is invalid by rule.

**14. Enhanced winner as a reader-sensitivity report (section 5, line 100).** Earlier finalists were overturned by enhanced (TABLE_CV) readers (READER_SENSITIVE in the frozen-finalist validation). The rules report `READER_SENSITIVE_ENHANCED_<point>` beside D2 whenever the enhanced winner's point benefit against TWIN or a Q* is <= 0 in a version mean; it never gates. Say so in section 5: "the enhanced winner's benefits are reported beside D2 as a reader-sensitivity check".

**15. Registered predictions in the rules' labels (section 10).** Make the predictions checkable. Per family, give:
- probabilities for D2 = IMPROVEMENT_BEYOND_MATCHED_CONTROLS, NOT_SEPARATED, a missing bracket (NO_BRACKET_SBC / NO_BRACKET_SPEND) and PROTECTION_NOT_ASSESSED;
- the predicted pair benefit against TWIN and against each Q*, with an interval.

**16. Randomized releases (sections 1 and 5).**
- For any randomized configuration that enters D2 (CAND or a control point), require D >= 2 evaluated draws on the registered cells, so the MC SE can be estimated. The grid's Monte Carlo replicate members give D = 3 on R1 / R2. Otherwise MC_SE_NOT_ESTIMABLE travels with the reading.
- A randomized CAND's SBC should use the same draw count, seed handling and noise coupling.

**17. TWIN as strength 0 (section 4, row TWIN).** Where the operation has a natural zero, state that TWIN is the operation at strength 0. The strength-0 point of every control path is then exactly TWIN, as the rules assume. Otherwise the card states the difference.

**18. The shared CTL controls (sections 2 and 8).** State what the up-to-6 shared control configurations are and on which TWIN they are built (see finding 2 (v)). A control built on another family's TWIN is not matched.

**19. Ablation and strongest-release reading (sections 2 and 6.3).**
- The strongest feasible nonzero release is chosen on registered MODEL_SELECT cells (winner's curse).
- Its ablation comparisons (strongest vs ABL, ABL vs TWIN) are descriptive and never upgrade D2.
- The pool is the family's own operation configurations only (controls and ablations excluded), each tagged with its card revision.

Say so where the order of work mentions the ablation (line 56).

**20. Known-law limitation (section 5, line 103).** Add the NRM1 RULES 5.5 sentence:
> "With two versions and two registered splits, the aggregate screen's false-pass rate is about 8% when one version or split dominates; such a reading carries that caveat."

## Summary

REQUIRED:
1. seed-base wording
2. matched-control definition
3. control-point bracket and equal-effort rule
4. recipient gate and no cheaper dominating control in D2 (c)
5. NOT_ASSESSED_TASK_INFEASIBLE and the outcome classes
6. registered vs adaptive labelling and look counts
7. budget arithmetic and control reuse by revisions
8. per-version strength selection
9. D2R definition
10. grid validity in the protocol

ADVISORY: 11-20.
