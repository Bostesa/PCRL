<!-- Saved verbatim by the coordinator from the statistical reviewer's hand-back message (the harness refuses report-file writes by subagents). Author: the statistical reviewer. -->

# Statistical reviewer: pre-registration review of the draft PROTOCOL (campaign `pcrl_input_and_pair_release_v1`)

Reviewed text: `results/pcrl_input_and_pair_release_v1/PROTOCOL.md` (coordinator's draft, SHA-256 efa4000830f08ba0..., last modified 01:39:38Z and unchanged while I worked), with `FAILURE_MAP.md`, `EXPOSURE_MAP.md`, `roles/coordinator/DESIGN_BRIEF.md` (b3ce09c0...) and `cards/README.md`. Written 2026-10-22 from 01:40:07Z (`date -u`), before any outcome of this campaign existed. I read no label, nothing under `.private/`, nothing of MODEL_SELECT, no reader cache, and neither designer's directory; I did not edit PROTOCOL.md. The coordinator's note of about 02:00Z (a SEX-blind release-only operation; collapsed sequential orders in D5) is answered in findings 3 and 6.

The formal rules are `RULES.md`, `RULES.json` and `ipr/rules.py` (schema `IPR1-rules-v1`). Every REQUIRED finding that changes a decision is implemented there, so the protocol and the rules agree once the wording below is adopted. If a REQUIRED finding is declined, the matching rule must be changed and re-frozen before registration. Section numbers refer to the reviewed draft.

## REQUIRED findings

**1. The RO path is not well defined for every operation (section 4, table row RO and bullet (iv)).** "Every person-level SEX statistic (a function of x fitted with SEX labels) replaced by its counterpart estimated from TWIN's released vectors" leaves three things open:
- (i) an operation whose SEX dependence is not a person-level statistic (a grouping, a field coarsening or training-row weights chosen with SEX labels): by the literal text it falls to ZTILT, which is not the same operation;
- (ii) a statistic that is not a function of x alone (for example P(occupation | x, SEX));
- (iii) what "counterpart" means.

One construction covers all three. Replace the RO row's "What" cell with:
> "the release-only explanation, declared in the card before CAND's first fit as `ro.kind`. SUBSTITUTED (default): CAND's operation with the MODEL_FIT SEX column replaced, in every SEX-dependent fitted input, by the release-only SEX surrogate P(SEX | TWIN's released log-probabilities of both recipients), fitted on MODEL_FIT and cross-fitted on the household folds (used as a soft label where the fit accepts weights, otherwise one fixed-seed draw; the card says which), on its own strength grid. GENERIC_ZTILT: the generic release-only tilt ZTILT of TWIN (`ipr.ctlkit`) on a fine grid; required when the operation is itself release-only (every SEX-dependent fitted input already a function of TWIN's released vectors, so that the substitution would return the operation itself), and otherwise allowed only as the card's declared fallback. The card also records `ro.operation_release_only`."

Replace (iv) with: "(iv) for the RO path, be the declared SUBSTITUTED or GENERIC_ZTILT construction, with the surrogate (or the tilt's amplitude) cross-fitted on the household folds, so that no MODEL_FIT person's own SEX label enters its own surrogate value;".

It fits the frontier machinery unchanged (TWIN + two saved points, Q* by `nrm.rules.frontier_comparison`; test `t_brackets_equal_nrm_rule` covers all three paths).

**2. D2's label when the operation is itself release-only (section 6.2).** For such an operation, the RO comparison cannot ask whether the gain needs anything beyond the twin's release, because by construction it needs nothing. It asks instead whether this release-only map beats the generic tilt at matched spend. The label must say so. Add to 6.2 after (d):
> "D2 met is labelled `IMPROVEMENT_BEYOND_MATCHED_CONTROLS` when the RO path is SUBSTITUTED (or the declared GENERIC_ZTILT fallback), and `RELEASE_ONLY_MAP_BEYOND_MATCHED_CONTROLS` when the operation is itself release-only. Every sentence quoting a D2 label or the outcome class repeats the reading's scope sentence (`ipr.rules.RO_SCOPE`); a release-only map that passes is never described as using SEX information beyond the twin's release. Partial outcomes are `PARTIAL_<passed>_ONLY`, joined by + in the order TWIN, SBC_FRONTIER, SPEND_FRONTIER, RO_FRONTIER."

**3. A SEX-blind operation (coordinator's note; sections 4 and 6.2).** As drafted, consider an operation that reads no SEX, for example a function of the twin's released vectors and a SEX-independent key. Its SBC path and its RO path are the operation itself. D2 would then either fail by construction (the bracket would be an alias of CAND) or read INCOMPLETE. Make it defined and mechanical. Add to section 4 after (vi):
> "A card states whether its operation has any SEX-dependent fitted input (`operation_sex_blind`; SEX in the MODEL_FIT strength-selection guards does not count). For a SEX-blind operation the SBC and RO comparisons are degenerate by construction (each would be the operation itself): no SBC path and no substituted RO path is built (2 x 2 x 2 = 8 configurations saved), and D2 gates on TWIN and the SPEND path (targeting vs TWIN and the SPEND bracket). A generic ZTILT path, if built, is reported beside and never gates."

and to 6.2: "For a SEX-blind operation D2 met is labelled `SEX_BLIND_IMPROVEMENT_BEYOND_TWIN_AND_SPEND` and is quoted with: no SEX-targeting claim and no claim about SEX information beyond the twin's release is possible."

The conjunction then has two members, so its power is higher. Its per-look false-pass rate is still at most one comparison's (intersection-union).

**4. Planned resolution (section 5, "Planned resolution").** The statement has three problems:
- it mixes the resolution threshold with the detectable effect;
- it uses only the low end of the realized range: at se_agg 0.009-0.018 the 2-SE threshold is 0.018-0.036;
- an effect sitting exactly at the threshold is resolved only half the time, and the bracketing comparisons had the larger SEs.

Replace the sentence with:
> "**Planned resolution**: a comparison resolves when its aggregate mean - 2 se_agg > 0 with both version means > 0. The dual-design grid realized se_agg 0.009-0.011 against its twin, 0.009-0.017 against its SBC and SPEND brackets and 0.008-0.012 against its release-only tilts, so the per-comparison threshold is about 0.016-0.036 and a true benefit of about 2.84 se_agg (0.023-0.051) is resolved with roughly 80% probability (normal approximation; the version-mean condition lowers it). D2 needs every gating comparison resolved: with equal true benefits against TWIN and every bracket and correlation 0.5 between comparisons, its power is about 0.24 at 0.025, 0.56 at 0.035 and 0.90 at 0.050; when the release-only control reproduces part of the gain (0.035 against the others, 0.015 against RO) about 0.29 (reviewer's invented-data calculation, RULES.md 3.2). Smaller effects read 'not separated', never 'equal'."

**5. Control-grid fineness (section 4, last sentence of "What a matched control must satisfy").** "A grid point within 25% of CAND's spend on each side" has three gaps:
- (i) it does not constrain the saved upper point, which must spend at least CAND's + 2 SE. The 2.08x overshoot it targets can therefore recur whenever 2 SE exceeds 0.25 s.
- (ii) it is undefined when CAND's MODEL_FIT spend s <= 0. An input-side operation can regularize and spend less than TWIN.
- (iii) it has no check and no consequence.

Replace the sentence "Every control path's grid is fine enough ... the card states the grid." with:
> "Every control path's grid is fine enough that, on MODEL_FIT and in each version, its saved lower point spends at least 0.75 s and its saved upper point at most s + 2 SE + 0.25 s, where s is CAND's out-of-fold paired total-NLL spend over TWIN and SE the household-clustered SE of the upper point's paired difference (`ipr.rules.grid_fineness`; the dual-design release-only grid, 0.90x / 2.08x, reads COARSE_ABOVE). The check runs on MODEL_FIT before any MODEL_SELECT read; a coarse path gets added grid points on MODEL_FIT (CAND's own path is not re-tuned). When s <= 0, TWIN is the bracket of every path and the check does not apply. A remaining coarse path is flagged CONTROL_GRID_COARSE_<path> beside D2: reported, never gating, never a reason to set a reading aside. The card states the grid."

**6. D5 when sequential orders collapse by construction, and D5's missing records (coordinator's note; section 4 table row "D5 controls" and section 6.6).** NR G6 reads:
- an unnamed comparator as NOT_ASSESSED_MISSING_CONTROLS;
- an unbuilt grid entry the same way;
- a missing paired cost as NOT_SUPPORTED.

Some families have sequential orders that are, by construction, the separate or the joint design (for example, independent per-recipient keys as the separate design). For them, G6 would either demand controls that cannot differ or read a missing record as a negative result. Replace 6.6 with:
> "**D5 joint design:** NRM1 G6 unchanged for a family whose card declares joint design and builds the separately designed recipients and both sequential orders. A sequential order that the card declares, before its first fit and with its construction argument, to equal the separate design by construction is replaced by the separate comparator (reduced test, NR's label, flag D5_REDUCED_<ORDER>_EQUALS_SEPARATE, stated with the label); an order declared equal to the joint design itself gives NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL (no joint-design advantage over sequential design can exist). A declared comparator whose grid entry or paired cost is missing reads INCOMPLETE. Otherwise NOT_ASSESSED_MISSING_CONTROLS, or NOT_APPLICABLE_NOT_JOINT / NOT_APPLICABLE_ONE_OUTPUT_CHANGED (NR). D5 carries D2's NOT_ASSESSED_* / NOT_COMPUTED_* label as D3 and D4 do and never changes the outcome class."

In the D5-controls row add: "(or the declared collapse of a sequential order; the challenger and the verifier check the declaration, by an exact-alias wire check where both comparators exist)".

**7. Revisions cannot reuse a substituted RO path (section 8).** "A revision reuses the family's SPEND and RO paths whenever its TWIN is unchanged" is wrong for a SUBSTITUTED RO path. That path is the operation itself with a release-only surrogate, so a changed operation needs a new RO path.

Budget arithmetic with three paths:

| Item | Configurations |
|---|---|
| Per family: CAND | 2 |
| Per family: three paths x two points x two versions | 12 |
| Per family: ABL | 2 |
| Per family total | 16 |
| Planned allocation: shared twin 2 + IA 16 + IB 16 + D5 6 | 40 |
| Remaining | 8 |
| Revision reusing SPEND and a GENERIC_ZTILT RO (CAND + two SBC points) | 6 (8 with its ablation) |
| Revision needing a new SUBSTITUTED RO path | 10 (does not fit) |

Replace the sentence with:
> "A revision reuses the family's SPEND path whenever its TWIN is unchanged and its points still bracket the revision's MODEL_FIT spend, a GENERIC_ZTILT RO path under the same condition, and the SBC path and a SUBSTITUTED RO path only if its operation and TWIN are unchanged. Otherwise its D2 is NOT_ASSESSED_CONTROLS_NOT_RUN (class PROTECTION_NOT_ASSESSED) unless new control points fit the budget (a revision with new SBC and SUBSTITUTED RO paths needs 10 configurations; 8 remain after the planned allocation)."

**8. Strength records and the by-version flags (sections 4 and 6; DDR1 deviation 8 and closing statement 1.2).** Two defects carried over from DDR1:
- its helper read a recipient-split strength [0.35, 0.65] as "not nonzero";
- two by-version facts were missing from the records: CAND_STRENGTH_DIFFERS_BY_VERSION, and an SBC built by permutation in one version and by PSEUDO2 in the other.

Add to section 4 after "...flagged CAND_STRENGTH_DIFFERS_BY_VERSION if they differ.":
> "A strength is recorded as a number, a component vector (for example recipient-split strengths) or a per-version dict of either; it is nonzero when every component is >= 0 and one is > 0; the tie-break scalar is the sum of all components over both versions. The record carries CAND's per-version strength and each control path's construction per version; the rules flag CAND_STRENGTH_DIFFERS_BY_VERSION and SBC_CONSTRUCTION_DIFFERS_BY_VERSION from them (reported)."

**9. Families and registration (sections 2 and 6).** The rules refuse a reading whose family is not `IA`, `IB` or `FS_<letters/digits>` (CTL is never a reading), and only revision 0 of IA or IB can be registered. Add to section 6, first paragraph:
> "A family reading is IA, IB or FS_<name>; the shared controls (CTL) are never a reading; only revision 0 of IA or IB can be a registered reading."

**10. D2R and the minimum set with three paths (sections 5 and 6.3).** Add to 6.3 after "its bracketing controls":
> "(of the SBC, SPEND and RO paths, recomputed by `ipr.rules.brackets`; a path whose bracket is not recorded makes D2R INCOMPLETE, so a decisive comparator is never dropped silently)"

and to the grid-validity bullet of section 5:
> "the minimum set is a reading's CAND, TWIN and the saved points of its gating control paths; references, ablations, D5 comparators and a reported-beside ZTILT path are off it."

## ADVISORY findings

**11. Fallback when a substituted RO path cannot reach CAND's spend (section 4 (iv)).** A release-only surrogate carries less SEX information than the true labels, so the substituted operation may need larger strengths to spend as much as CAND. Extend its grid beyond CAND's (stated in the card). If its largest strength still does not reach CAND's spend + 2 SE, either:
- keep it: flag CONTROL_PATH_MAY_NOT_BRACKET, and RO then reads NO_BRACKET_RO, a design limit; or
- use the declared GENERIC_ZTILT fallback, chosen by the MODEL_FIT-only rule before any MODEL_SELECT read. The D2 scope sentence then changes accordingly (RULES.md 3.1).

**12. Which parts of "matched" apply to DQ and ZTILT (section 4 (i)).** "The same number of fitted components and the same draw structure" cannot hold for DQ or ZTILT, which are not CAND's operation. Say: "(i) applies in full to SBC and to a SUBSTITUTED RO path; DQ and ZTILT share TWIN, the MODEL_FIT people, folds and seeds, and a randomized control gets at least two evaluated draws on the registered cells, as a randomized CAND does."

**13. Shared twin (section 4, TWIN row).** "Both families share one twin when their task-only models coincide" needs identical learners, settings, folds, seeds and selection effort. An input-side family that re-selects hyperparameters on modified inputs needs a TWIN given the same re-selection on the raw inputs (2 configurations). Otherwise the twin is not equally trained (FAILURE_MAP note 5).

**14. Shared control configurations (sections 4 and 8).** When both families share TWIN, a DQ or ZTILT point at the same strength is the same released wire: save it once, count it once, and list it in both readings' paths.

**15. D5 comparator selection (section 4, D5 row).** G6 refuses a joint claim when CAND's MODEL_SELECT primary cost is resolved above a comparator's. Select each D5 comparator on MODEL_FIT as the smallest grid strength whose spend is at least CAND's (one point each), stated in the card.

**16. Registered predictions in the rules' labels (section 10).** Per family, give:
- probabilities for each D2 label that can occur: the met label that applies to its declarations, NOT_SEPARATED, PARTIAL_..._ONLY with RO failing, NO_BRACKET_<path>, PROTECTION_NOT_ASSESSED;
- the predicted pair benefit against TWIN and each Q*, with an interval.

With the power of finding 4, a prediction of D2 met above about 0.3 needs an expected benefit of at least about 0.035 against every comparator.

**17. Record mapping (section 5).**
- READER_SPLITS.json names reader seeds V<v>k<k>, while the rules read S<v>k<k>. The claims builder must map them.
- The builder must also pass the bootstrap seeds S<v><R>. They are required inputs: without them the grid is invalid by rule.
- The +900 base and the bootstrap base 2026102300 appear in no earlier READER_SPLITS.json of the repository (text search, 01:44Z).

**18. Stale derived docstrings (section 12).** `ipr/rules.py` and `ipr/test_rules.py` are rewritten for IPR1. Other derived modules may still describe DDR1 (offset 800, two control paths). A module that re-implements a rule (as `ddr/refitcheck.py` did) should call `ipr.rules` instead.

## Summary

REQUIRED:
1. RO path definition (SUBSTITUTED surrogate / GENERIC_ZTILT)
2. the release-only operation's own met label and scope sentences
3. SEX-blind operation (degenerate SBC and RO, D2 on TWIN and SPEND, own met label)
4. planned-resolution statement
5. control-grid fineness on the saved points
6. D5 collapsed sequential orders and missing D5 records
7. revisions cannot reuse a substituted RO path (budget)
8. strength records and by-version flags
9. families and registration
10. D2R and the minimum set with three paths

ADVISORY: 11-18.
