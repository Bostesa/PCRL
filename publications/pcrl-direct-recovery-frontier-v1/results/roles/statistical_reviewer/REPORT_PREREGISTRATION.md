> Saved verbatim by the coordinator (2026-10-25, after 03:09:53Z by `date -u`) from the statistical reviewer's hand-back message;
> the harness refused the reviewer's own write of this file. Everything below this note is the reviewer's text, unchanged.

# Pre-registration review of the DRF1 design (statistical reviewer)

Written 2026-10-25 from 03:03Z (`date -u`), before any outcome of this campaign and before registration; returned in the reviewer
hand-back because the harness refused the direct write (to be saved verbatim by the coordinator). Inputs: the coordinator's
`DESIGN_BRIEF.md` (written from 02:36Z), the brief I was given and, for consistency only, sections 5-8 of the draft `PROTOCOL.md` and
the label-free seed / tag fields of `READER_SPLITS.json`; no card, investigator file, label, `.private/` content or MODEL_SELECT record
was read. Each finding is marked REQUIRED (the rules depend on it, or a claim would be unsupported without it) or ADVISORY. The rules
(`RULES.md`, `RULES.json`, `drf/rules.py`) already implement every REQUIRED finding that concerns a decision; the others ask for
records or plan statements that the rules read but cannot produce.

## REQUIRED

**R1. Freeze the eight budgets per version before any fit, identical for DR, OBJ_MI and OPT_S / OPT_L.** The budget (0.6 x the
twin's published MODEL_SELECT slack, min over U0 / Ucal, FRC1 `dev/UTILITY_LEDGER.csv`) is an adaptive setting from MODEL_SELECT
aggregates. Record the 16 numbers, their source file hash and the 0.6 factor in the plan before the first fit, and use the same
numbers for every constrained construction. Every G1 statement about DR must carry the disclosure that the constraint was tuned on
MODEL_SELECT aggregates of the twin, so a MODEL_SELECT G1 pass is not an independent test of constraint transfer (RULES.md 12).

**R2. OBJ_MI must be equally exposed and fitted before any MODEL_SELECT read.** Record, for DR and OBJ_MI side by side: fine cells,
atom capacity, move set, sweep order seed, sweep cap, restart count and restart-choice rule, the law (estimated pi law), the 8
budgets, MODEL_FIT data and the teacher versions. The verifier checks equality field by field. OBJ_MI is a single point per version
and gates D2 by the single-point spend rule (RULES.md 4.2): its pair benefit and recipient gates, no resolved secondary-metric
excess, and no resolved lower OBJ_MI total-NLL spend. Report OBJ_MI's MODEL_FIT nested spend and its MODEL_FIT objective values
beside DR's, and which budget binds for each in each version (MODEL_FIT), so a spend mismatch can be read.

**R3. Save the CTRL bracket points by the registered rule on MODEL_FIT nested held-out spend, with DR's spend from the same nested
estimator.** DR's nested total-NLL spend over TWIN and every CTRL path point's nested spend must come from the same 5 outer
household folds and teacher statistics; the SE in "DR's spend + 2 SE" is the household-clustered SE of each point's pooled held-out
paired difference against DR on those data. Record the full greedy path (spend and SE of every point) so that `grid_fineness` can
check the saved points (RULES.md 4.4). If the fineness reads `CONTROL_PATH_MAY_NOT_BRACKET`, extend the greedy path (more merges)
before any MODEL_SELECT read; otherwise D2 is likely `NO_BRACKET` (a limit of the control path, not a result).

**R4. Record distinct-input counts of all three views for every minimum-set wire and version, on the MODEL_SELECT pool.** The
few-valued and finite-pair explanations (RULES.md 9) read `distinct_inputs['<wire>|V<v>'] = {'income', 'occupation', 'pair'}`
counted on the 8001 MODEL_SELECT people. A finite release's pair view can exceed 320 distinct vectors (up to 1024); without these
counts any seed equality on a minimum-set view invalidates the grid. Minimum set: DR, TWIN, the saved CTRL points, OBJ_MI and SUCC.
Record seed checks for every seed-dependent recipe, every view, every wire, both versions, both repetitions and each seed pair
(`k_pair`), with references on the pair view included (they supply the pair-view seed evidence).

**R5. Usefulness failure: record the reader-stage decision and preserve the release.** If DR fails an original condition on
MODEL_SELECT, the coordinator may stop the competitive reader stage; record `reader_stage = STOPPED_USEFULNESS_FAILURE` and keep the
fitted release (custody). A stop recorded while DR passes G1 makes D2 INCOMPLETE (the reader stage must then run). The optimistic
diagnostic is MODEL_FIT-only and must still be computed so that the failure can be classified (stage 1 or 2).

**R6. Optimistic diagnostic inputs on the same law and mode.** For OPT_S and OPT_L, per version and mode (in-sample exact; nested
held-out): the law tag, total-NLL spend over the twin, AUC* of pair / income / occupation, and the in-sample `within_budget`
(all 8 spends within the R1 budgets). The SEX-blind CTRL path must be evaluated on the same law and in the same mode (TWIN first,
spend 0) and extend at least to the optimistic constructions' spends; otherwise the diagnostic reads `UNDETERMINED` and stage 1
cannot be separated from stages 2 / 3. The plan must state that DR, OBJ_MI and CTRL are fitted and read whatever the diagnostic says
(no preliminary significance, optimistic forecast or novelty gate).

**R7. Exact-alias records.** `cand.alias_of_twin` (bitwise float64 equality of DR's transmitted vectors and TWIN's in both versions)
and the alias relations among CTRL points, TWIN and OBJ_MI must be computed on the wires, never inferred from the mapping table:
DR = TWIN gives `NO_OUTPUT_CHANGE`; DR = OBJ_MI gives the OBJ reason `DR_EQUALS_OBJ_MI` (the objectives reach the same release).

**R8. Registration timing.** Record the UTC time of the first MODEL_SELECT read of this campaign (any configuration: utility, reader
AUC or paired cost of DR, OBJ_MI, a CTRL point or a re-read reference) and each card's commit. A revision's card must carry the
prospective note (observed failure, actual changed operation, trigger MODEL_FIT / SYNTHETIC / MODEL_SELECT) committed before its
refit; at most two revisions. A revision triggered by a MODEL_SELECT outcome is adaptive as a whole. The headline is the registered
reading with the highest revision (the latest pre-outcome card).

**R9. Determinism.** The optimizer must be deterministic given its seed and data; record the seeds per version and the restart
choice (by the MODEL_FIT objective under the estimated law, deterministic tie-break). DR, OBJ_MI and the CTRL path carry no MC term;
a recorded `se_mc` for them is a protocol error. Internal iterations are recorded but never read or counted as specifications.

**R10. D2R inputs.** The robustness grid entries of DR vs the registered Q* and OBJ_MI must contain all three views on R3 / R4
(k0-k3) and R1 / R2 (k2 / k3) cells: the robustness recipient gates are part of `ROBUSTNESS_CONFIRMED`, and failed robustness
recipient gates and sign changes on any comparator (decisive or beside: SUCC, TWIN, NEURAL_RAW, H_LJ and the references present)
are carried into the headline. Q* is the registered bracket; nothing is re-chosen on robustness cells.

**R11. Keep FC and TWIN out of D2 in every sentence.** The family change (DR vs SUCC) is reported as its own outcome and attributes
nothing to the objective; DR beating TWIN is never sufficient alone. Any summary sentence quoting D2 carries its scope; a partial
D2 (`PARTIAL_CTRL_FRONTIER_ONLY`, `PARTIAL_OBJ_ONLY`) is never a component gain.

**R12. Align PROTOCOL.md section 8 with the rules (or re-freeze the rules) before registration.** The draft protocol describes D2 as
a gain "beyond START, the CTRL frontier and OBJ_MI". The rules keep START = SUCC out of D2: the brief requires the family change
(finite family vs SUCC) to be reported separately from the objective comparison, and failure stage 4 ("the mapping beats its
controls but misses strong references") refers to the family controls, while SUCC is one of the strong references (D3). With SUCC in
D2, a DR that beats both family controls but not SUCC would be mislabelled "ties its controls" (stage 3), and D2's power would fall
further (a third comparison). Either change PROTOCOL section 8 to "D2 component gain beyond the CTRL frontier and OBJ_MI (FC vs
SUCC reported separately; SUCC in D3)", or ask the reviewer to re-freeze the rules with START gating; never after an outcome. The
protocol's reader seeds, bootstrap rule and split tags agree with the rules. The protocol's configuration names (DR_AUC, DR_MI,
TWB_C64, DR_CTRL_LO / DR_CTRL_HI) are supported through `drf.rules.comparator_ids` (the twin's `id`, OBJ_MI's `config_id`).

## ADVISORY

**A1. Power.** With se_agg about 0.008 at Q* and 0.007 against OBJ_MI (correlation 0.5), D2 resolves a true benefit of 0.010 /
0.015 / 0.020 / 0.025 / 0.030 against both about 12% / 33% / 61% / 84% / 95% of the time; if OBJ_MI reproduces half of the gain,
about 25% / 40% / 55% at 0.020 / 0.025 / 0.030. D4 needs the 0.020 target resolved against NEURAL_RAW and H_LJ, which needs a true
benefit near 0.020 + 2 se_agg (about 0.035-0.048 at the realized reference se_agg 0.0074-0.0139) for even odds. Expect
NOT_SEPARATED / partial readings to be uninformative about small effects; never read them as equality.

**A2. OBJ_MI spend mismatch.** The single-point rule credits an unresolved spend excess of DR (disclosed by a flag). If the
coordinator wants a stricter attribution, the alternative is the strict point rule (reported beside as
`inherited_point_rule_beside`); switching would have to happen before registration and costs about half of D2's power when both
objectives bind the same budget. Neither rule excludes that a better optimizer for the MI objective would close the gap.

**A3. Estimated-law objective.** DR minimizes the exact AUC under the pi-weighted estimated law. Report, as MODEL_FIT diagnostics,
DR's exact AUC under the true-SEX empirical law in-sample and nested beside its estimated-law value, and the same for OBJ_MI and the
CTRL points: the gap shows how much of the objective is model-based, and helps tell an estimation limitation (stage 2) from a
representation or optimization one.

**A4. Tiny exact instances.** Register before running them the instance generator, the instance count and the criterion that
classifies an optimizer defect (for example: local search within 1e-9 of the exhaustive constrained optimum on every instance, or the
largest gap and the share of instances with a gap), so that a later "optimization limitation" in stage 2 rests on a prospective
criterion.

**A5. Few-valued readers.** On 16- and 64-valued views the LOOKUP recipe is close to the optimal reader; report per view the gap
between the selected winner and LOOKUP on SCORE, and the n_eff table from `grid_validity`, so a reader-seed redundancy is visible
next to every comparison.

**A6. Budget accounting.** Each compared operating point counts as a specification (OPT_S 2, OPT_L 2, DR 2, OBJ_MI 2, CTRL 4 = 12 of
32); the CTRL path itself (greedy merges) is computed once on MODEL_FIT and only its two saved points per version are read on
MODEL_SELECT. A revision of DR re-saves the CTRL bracket points for the revised DR's spend before its MODEL_SELECT read (2-4 more
specifications per revision).

**A7. Bootstrap seed overlap.** The v = 0 bootstrap seeds 2026102500-2026102503 equal SAR1's v = 1 seeds; on the new
`DRF1_devgrid_R{r}` splits they resample different SCORE households, so this is harmless; disclose it in the grid record.

**A8. Prior art.** D6 stays NOT_CLAIMED; the design overlaps privacy funnels and the deterministic information-bottleneck line, and
the earlier SCM / FRC finite-routing work in this repository; a separate review is needed before any novelty wording.
