# Statistical reviewer, wave-1 hand-back (verbatim; delivered to the coordinator at about 21:38Z; harness indentation removed)

# RFA1 statistical reviewer: wave 1 hand-back (contract review, then frozen-rules draft)

Statistical reviewer, campaign `pcrl_reader_fit_aware_release_v1` (RFA1). Written 2026-10-25, 21:12–21:38Z (`date -u`), before any outcome of this campaign existed.

**What I read.** Code and records only:
- `MEASUREMENT_CONTRACT.md`, the 21:06Z draft (105 lines, SHA-256 6d37cb8f…; unchanged when I re-checked at 21:36Z);
- `sources/EDG1_ESTIMATOR_DIAGNOSTIC.md` and `sources/EDG1_REVIEW_REPORT.md`;
- DRF1's `RESEARCH_DECISION.md`, its reviewer's `RULES.md` (sections 0–4) and the `d2r` function in `drf/rules.py`;
- SAR1's `RULES.md` and `RULES.json`, `sar/rules.py`, `sar/test_rules.py`, `nrm/rules.py`, the helpers in `nrm/test_rules.py`, `sar/evaluate.py`;
- the code that defines usefulness (`ohn/rules.py`, `ohn/evaluate.py`, `ihs/evaluate.py`, `ihs/common.py`, `scm_bench/selection.py`, `scm_bench/common.py`, `tsz/evaluate.py`);
- the constants and the `utility_of` call in the coordinator's `rfa/devreaders.py` and `rfa/evaluate.py`.

**What I did not read or change.** No label, no model, no private array, nothing under `.private/` or `<worktrees>/`, nothing of MODEL_SELECT. I made no git change and spawned no agents.

**Files written.** Only these four:
- `rfa/rules.py`
- `rfa/test_rules.py`
- `results/pcrl_reader_fit_aware_release_v1/roles/statistical_reviewer/RULES.md`
- `results/pcrl_reader_fit_aware_release_v1/roles/statistical_reviewer/RULES.json`

Running the tests also left the usual `__pycache__` bytecode. All scratch files are in my scratchpad.

**CPU used:** about 16 CPU-s in total, on one numerical thread. That covers four test runs (about 0.4 s each), the in-memory mutation check (9.3 s), a power simulation (0.1 s), small diagnostic scripts and interpreter start-ups.

---

## A. Review of MEASUREMENT_CONTRACT.md

**Verdict.** The contract is sound in substance and covers every measurement requirement in the brief. It has 7 REQUIRED corrections and 9 ADVISORY items, listed below.
- **Main errors.** The description of se_rs is wrong, and the scope of se_hh is incomplete.
- **Imprecise text.** The material-target wording and the stated location of the usefulness definitions.
- **Undefined trigger.** The trigger for the second split.
- **Underspecified reading.** The learning-curve reading.
- **One false scope sentence.** The claim that "all releases" are continuous.

None of these needs anything re-run; nothing has been run yet.

### Verification of the usefulness definitions and the recipient gate (from the governing code)

The contract's statement is correct: 28 conditions = per version 12 task clauses + 2 within-SEX guards, over 2 versions; caps 0.010 / 0.010 / 0.005 / 0.005, all inclusive.

**Where each piece lives:**

| piece | location |
|---|---|
| clause values: 2 tasks × {U0_S<v>, Ucal_S<v>} × 3 metrics | `scm_bench/selection.py::clauses`, lines 38–49 |
| caps 0.010 / 0.010 / 0.005 | `scm_bench/common.py`, line 45 (`CAPS`) |
| per-person metrics: argmax accuracy; NLL with a 1e-12 probability floor; Brier summed over classes | `scm_bench/common.py::task_vectors`, lines 213–220 |
| household-clustered SEs and the two guards | `ihs/evaluate.py::utility`, lines 75–99 (`cluster_se` / `subgroup_se` at lines 57–72) |
| guard definition: occupation summed-Brier excess vs U0, within SEX 0 (men) and SEX 1 (women) | `ihs/evaluate.py`, line 89 |
| guard cap | `ihs/common.py`, line 63 (`GUARD = 0.005`) |
| call chain | `sar/evaluate.py::utility_of` (lines 160–191; `rfa/evaluate.py::utility_of` at line 165 is the same call) → `ohn/evaluate.py::complete_eval` (line 120) → `ihs.evaluate.utility` |
| U0 and Ucal references | `tsz/evaluate.py::refs_of`, lines 38–46, from the frozen bank |
| names and cap check (a changed cap raises an error) | `ohn/rules.py::conditions`, lines 103–130 (`CAPS` line 30, `GUARD_CAP` line 31) |
| gating at the point, inclusive, `n_conditions` 28 | `nrm/rules.py::point_eligibility`, lines 157–182 (constants at lines 46–49) |

Metric directions: accuracy loss = accuracy(reference) − accuracy(release); NLL excess and Brier excess = release − reference.

**Recipient gate.** `nrm/rules.py::recipient_screen`, lines 356–384 (`RECIPIENT_LIMIT = 0.010` at line 48), applied to both single views inside `comparison` (lines 387–394).
- The gate passes iff the per-version aggregate point difference r_v (candidate AUC − comparator AUC) is ≤ 0.010 in both versions.
- The contract's description is correct.

**Material target.** `target_screen` (lines 344–353) and claims G5 (lines 727–740).

Test `t_usefulness_definitions_at_the_cited_code` checks every one of these line citations against the source text. If any of these files changes, the citations must be re-checked.

### Numbered findings

**A1 REQUIRED — §8: se_rs is misdescribed.**
- The contract says: "The spread over repetitions R1 / R2 and reader seeds k0 / k1 gives se_rs".
- What `nrm/rules.py::aggregate` (lines 255–307) actually computes:
  - the blocks are version × repetition, and the reader seeds are averaged inside each block;
  - se_rs = sqrt(Σ_b (m_b − mean)² / 4) over the four block means.
- So se_rs contains the version difference (the two release-fitting repeats), the split difference and their interaction. Reader-seed variation enters only through the block means.
- Proposed wording: "se_rs: the spread of the four version × repetition block means, reader seeds averaged within each block. It mixes the version difference (two release-fitting repeats), the reader-split difference and their interaction; reader-seed variation enters only through the block means. Each factor has about one degree of freedom, so se_rs is a crude component and the screen is approximate. It is not an estimate of release-fitting variance."

**A2 REQUIRED — §8: the scope of se_hh is incomplete.** The contract must state that:
- each cell's household bootstrap of SCORE (B = 400) holds fixed the release, the fitted readers, the orientation and the reader selected on VALIDATION;
- se_hh is the root mean square of the cell SEs, weighted as in the aggregate. It is an upper bound under any cross-cell correlation, and it does not shrink as cells are added;
- se_hh is conditional on everything fitted. It does not cover refitting the readers, re-selecting the winner or refitting the release.

Two more SEs need the same scoping:
- the usefulness SEs are household-clustered over MODEL_SELECT, with the release fixed;
- the bracket SEs, computed from MODEL_FIT out-of-fold paired costs, are a selection device only.

The brief asks the contract to distinguish household, release-fitting and reader-fitting variation; DRF1's decision text carries the equivalent sentence.

**A3 REQUIRED — §2: where the definitions live.**
- The contract says the definitions are "the authoritative `nrm.rules` / `sar.evaluate` ones". That is imprecise: neither file defines the metrics.
- Replace it with the chain given in the verification above.
- Also state that RFA1's evaluator (`rfa/evaluate.py`) calls that chain unchanged.
- The verifier should hash-check these files:

| file | SHA-256 |
|---|---|
| `scm_bench/selection.py` | 87de258a… |
| `scm_bench/common.py` | d8e1c8a7… |
| `ihs/evaluate.py` | 03f449e2… |
| `ihs/common.py` | 62dff612… |
| `ohn/evaluate.py` | 8e72de6a… |
| `ohn/rules.py` | 8686a78f… |
| `tsz/evaluate.py` | 6cb796af… |
| `nrm/rules.py` | d5df8686… |

**A4 REQUIRED — §2: the material-target wording.**
- "A resolved pair benefit of at least 0.020" reads like a point condition.
- The rule, against each of NEURAL_RAW and H_LJ with both recipient gates, is:
  - aggregate mean − 2·se_agg > 0.020 (strict), and
  - both version means > 0.020.
- At se_agg of about 0.010 this needs a point benefit near 0.040.

**A5 REQUIRED — §3: the budget-sensitivity reading is underspecified.**
- **(a) Define "shrinks" and the comparators.** The rules now do this:
  - per comparator, d_LC = b(FULL) − b(HALF) on the registered cells;
  - the comparators are START, CONT, every gating bracket point or control point, and H_LJ;
  - a limitation exists iff the full-budget point benefit is > 0 and the mean d_LC is < 0. This is a point rule on purpose; the resolved level is reported beside it.
- **(b) Add the converse.** No shrinkage between two budgets is not evidence that information was removed, and budgets above the declared FIT were not tested.
- **(c) Record the inner-episode budget.** A reader-fit-aware objective trains against readers fitted at its inner auxiliary budget. The fitted-reader evaluation is most favourable to such a mechanism when that budget matches the evaluation FIT. The card must record the inner-episode training size; the rules flag a missing or equal-or-larger size.
- **(d) Pair the two budgets.** The HALF readers must reuse the recipes, seeds, VALIDATION, SCORE and the same bootstrap draws, so that the difference has a paired SE.
- **(e) Align the tooling with the contract.**
  - `rfa/devreaders.py` defines `FRACTIONS = (0.25, 0.5, 1.0)`.
  - Its `lc_fit` keeps the FIT households that fall in folds 0–1 of 4 household folds (tag `RFA1_devgrid_LC`). That is about half of FIT, nested inside FIT, not "a half by hash rank".
  - Before any outcome, either drop 0.25 or declare it reported beside the reading, never part of it.
  - Describe HALF as that fold rule.

**A6 REQUIRED — §3: the second-split trigger is undefined.**
- "A useful, promising release" — "promising" is not defined.
- Fix the trigger prospectively, as the rules do: D2R is read at CAND whenever CAND is G1-feasible and the registered grid is valid, whatever D2 shows. If budget prevents it, it reads NOT_RUN and is disclosed.
- D2R is descriptive and never changes D2.
- D2R now also requires the recipient gates on the robustness cells, as in DRF1.

**A7 REQUIRED — §5: "All releases here are continuous network outputs" is false.**
- CAND and its controls are continuous.
- Several references emit few-valued codes, for example W's 16-code occupation view (A1_PO_16) and the code- and tree-based references. SAR1's grid needed a few-valued explanation for exactly such views.
- Proposed wording: "CAND and its controls are continuous network outputs; some references are few-valued; no empirical optimal AUC of any release is used as an objective, selector, gate or claim."

**A8 ADVISORY — §6, the qualification of EDG1.** The three qualifications are correct and match the brief. Three additions:
- (i) EDG1's results concern finite output supports. Its support-specific numbers do not transfer to a continuous CAND.
- (ii) D3 and D4 compare a continuous CAND with few-valued references. A fitted-reader comparison between outputs that differ this much in learnability can favour the harder-to-learn release (EDG1's tendency for finer outputs; size not quantified). The learning curve against H_LJ is the planned check.
- (iii) EDG1's post hoc 94.8–95.8% coverage concerns single-split synthetic comparisons. It does not calibrate RFA1's aggregate screen.

**A9 ADVISORY — §7, DRF1.** Accurate. DRF1's objective was under the propensity-estimated law (DRF1 answer 1); EDG1 lists that objective under "Not simulated".
- "The mechanism … fits its inner readers on realized MODEL_FIT SEX labels" is a statement about the card. Verify it against the final card.

**A10 REQUIRED / ADVISORY — §1, equal opportunities on the release side.**
- **REQUIRED.** The contract or the protocol must state that CAND's operating point and every control's saved points are selected on MODEL_FIT only, by rules declared before any MODEL_SELECT read. The rules make an unrecorded or non-MODEL_FIT CAND selection INCOMPLETE.
- **ADVISORY.** If CAND's rule uses a SEX-based MODEL_FIT diagnostic (for example an inner held-out reader AUC), CAND had a selection opportunity on privacy that the spend-matched control points lack.
  - Disclose it: the rules raise flag `CAND_SELECTED_ON_MODEL_FIT_SEX_DIAGNOSTIC`.
  - Report the controls' value of the same diagnostic beside the reading.

**A11 ADVISORY — §1, "scored once".**
- SCORE households are reused across releases, across the two budgets and for the ignore-one candidates. "Once" means once per (release, cell, budget), and SCORE is never used to fit, orient or select. That reuse is what makes the comparisons paired.
- EDG1 found that ignore-one candidates protect only partly (pair selection loss resolved at π = 0.2).
- The winner's curse on VALIDATION (up to +0.029) is why VALIDATION values are never reported as recovery.

**A12 ADVISORY — §4.** Add: the spread of inner episodes or MODEL_FIT folds is never an SE or a replication; the bracket SEs are a selection device.

**A13 ADVISORY — §2, gates and definitions.**
- The recipient gate is a point rule. Under the NRM1 calibration, a true single-view worsening of exactly 0.010 passes it about 32–34% of the time (SAR1 RULES.md §11).
- Cite the definitions of the confidence conjunction and the reserve (RULES.md §2).

**A14 ADVISORY — §8, "must agree in sign".** The screen actually requires both version means to exceed the threshold. That is sign agreement for threshold 0, but > 0.020 for D4.

**A15 ADVISORY — tooling, not contract text (`rfa/devreaders.py`).**
- The docstring at line 30 says the bootstrap seed is 2026102400 + 100·v + r, but the constant (line 92) and its use (line 314) are 2026102520. The rules use 2026102520.
- The `old` offsets list at line 713 omits DRF1's 1100. The rules forbid offsets 300–1100; the seeds at +1200 do not collide either way.

**A16 ADVISORY — §8, optimizer and inner-episode seeds.** Add: dependence on optimizer and inner-episode seeds beyond the two versions is not covered by any SE. In DRF1, the R1 → R2 search change alone moved pair AUC by up to 0.0076.

### The brief's requirements against the contract

| requirement | where | status |
|---|---|---|
| primary target: the declared finite-data reader procedure (fit on FIT, orient and select on VALIDATION, score once on SCORE) | §1 | met; A11 clarifies "once" |
| equal fitting, sample and selection opportunities | §1 (readers) | met for readers; A10 for releases |
| each recipient and the linked pair reported | §1 | met |
| a lower AUC is evidence about this procedure and budget, not a guarantee | §1 "Meaning" | met |
| empirical optimal AUC only as a diagnostic (not a population value, upper confidence bound or cardinality correction) | §5 | met; A7 fixes the scope sentence |
| EDG1's opposing-bias recommendation qualified (bias of a difference, test size, no universal ordering rule) | §6 | met; A8 additions |
| DRF1 used a propensity law; EDG1 used label-based laws; the actual objective was not tested | §7 | met |
| inner episodes are training data | §4 | met; A12 |
| household / release-fitting / reader-fitting variation distinguished | §8 | partly; A1 and A2 required |
| overlapping fold spread is not replication | §8 | met; A12 |
| versions and splits share people | §8 | met |
| exploratory; no correction for the adaptive search | §8 | met |
| usefulness 28 conditions, caps, recipient gate 0.010 | §2 | met; A3 and A4 required |

---

## B. Rules hand-back (draft for registration; adapt to the final card on your message)

### Files and hashes (SHA-256)

| file | SHA-256 |
|---|---|
| `rfa/rules.py` | e01c7cb7b97b866b283d5fc2405e9fb98a03438007e04242232326a54224755f |
| `rfa/test_rules.py` | 628646cf1615b609937db5b4db5fc59425a3ba82435ab7d6af7027318dbc7c9e |
| `roles/statistical_reviewer/RULES.json` (schema `RFA1-rules-v1`) | 60e16d85e23a4eca01c89bc779623695b56653e0233fd7c961240420b16d2b9a |
| `roles/statistical_reviewer/RULES.md` | bdae0a46c711866daec8596bbe046839ce4058b87be7bc14e878fc3d48813404 |

**Inherited files, verified unchanged** (both checked by `load_rules()`):
- NRM1 `RULES.json` 84f187768817e79ef9f2bb36151bea81f00b8f62d9179c8cd523069550674652
- `nrm/rules.py` d5df868678d483a3452e251d8f24575932d23f0e5d22e3ef1ca3627a35030df1

`nrm.rules` is imported unchanged as NR and never monkeypatched; a test asserts the module identities and both hashes.

### Tests and checks

- **Tests.** `python3 -m rfa.test_rules`: **23 of 23 PASS, exit 0** (invented data only; 0.37 CPU-s on one thread). The JSON output carries both hashes.
- **Mutation check.** I seeded 26 single-site bugs into `rfa/rules.py` in memory; the tests caught **26 of 26**. No repository file was changed. The bugs:
  - offset 1100 not forbidden; B not checked, or not required to be recorded;
  - CONT not gating; an alias accepted without its wire check;
  - a point that spends resolved less credited anyway; cheaper-point domination ignored; a missing bracket passing; targeting ignored;
  - a learning-curve limitation flagged without a full-budget benefit; the paired learning-curve SE ignored; H_LJ dropped from the learning curve;
  - the robustness recipient gate ignored; the reader-budget credit dropped; a reference advantage without D2 promoted to competitive;
  - bracket or CAND selection data unchecked; the confidence PASS read at the point; the bound's sign flipped;
  - a revision registered; the learning curve changing the class; START not checked;
  - the D2R beside list not de-duplicated; the point rule's secondary guard dropped; a not-run control inferred.
- **Power** (normal approximation, invented SEs: START 0.006, CONT 0.007, each control 0.008, correlation 0.5). Probability that D2 resolves, by common true benefit:

| scenario | 0.010 | 0.015 | 0.020 | 0.025 | 0.030 |
|---|---|---|---|---|---|
| all five comparisons | 0.04 | 0.17 | 0.42 | 0.71 | 0.90 |
| ALTCRIT an exact alias (four comparisons) | 0.06 | 0.22 | 0.49 | 0.76 | 0.92 |
| STOPFIT reproduces half the gain | 0.02 | 0.08 | 0.18 | 0.31 | 0.44 |

### Structure that was frozen

**Signs.** NR's, plus two of my own:
- pair benefit = comparator AUC − CAND AUC (positive favours CAND);
- recipient difference = CAND AUC − comparator AUC (positive = CAND's single view is more recoverable);
- paired costs = loss(CAND) − loss(comparator) (negative favours CAND; accuracy enters as a loss);
- bracket spend d = cost(CAND) − cost(point); d ≤ 0 means the point spends at least as much;
- learning-curve change d_LC = b(FULL) − b(HALF); negative means the benefit shrinks.

**Resampling.** NR unchanged: household bootstrap of SCORE (B = 400, recorded); block means over version × repetition; se_agg = max(sqrt(se_hh² + se_rs² + se_mc²), 0.001). RULES.md §3 states what each component covers (the corrected text of A1 and A2).

**Reader grid.**

| item | value |
|---|---|
| reader seeds | MODEL_SEED_BASE + 1200 + 10·v + k |
| forbidden offsets | 300, 400, …, 1100 (180 seeds) |
| bootstrap seeds | 2026102520 + 100·v + repetition index; no seed shared with SAR1 or DRF1 |
| B | 400, must be recorded |
| split tag | `RFA1_devgrid_R{r}` |

All of these match the `rfa/devreaders.py` constants. Grid validity uses the SAR1 explanations.

**D1 — usefulness.**
- G1 gates.
- The confidence conjunction and the reserve are reported, never gating, and never remove a reference.
- CAND's selection data must be recorded as `MODEL_FIT_OOF` or `MODEL_FIT_EPISODES`.
- Labels: `TASK_FEASIBLE`, `CAND_INFEASIBLE`, `NO_FEASIBLE_POINT`, `INCOMPLETE`.

**D2 — gain beyond the mechanism-specific controls.**
- Gating comparisons, in this order: START (NEURAL_RAW) and CONT (both simple comparisons), then STOPFIT, ALTCRIT and RAWJ.
- STOPFIT and RAWJ are paths from START, saved as LO / HI on MODEL_FIT out-of-fold spend and read by `NR.frontier_comparison`, plus the recipient gate at Q* and the cheaper-point domination check.
- ALTCRIT is a path, a single point (DRF1's spend rule), or an exact alias of STOPFIT. An alias needs `altcrit_alias_wire_check` True and is counted once.
- Reader targeting (d) applies against every gating point.
- Labels: `COMPONENT_GAIN_BEYOND_START_AND_MECHANISM_CONTROLS`, `READER_TARGETED`, `NOT_SEPARATED`, or `PARTIAL_<passed>_ONLY`.
  - A partial label states what each failed control leaves open. Without STOPFIT, the fit-aware derivative is not isolated.
- Development bounds (mean ± 2·se_agg) are reported for every gating comparison.

**LC — learning curve.** As in A5, descriptive. A learnability limitation is carried into the headline and never changes a label or the class.

**D3 / D4.** NRM1 G4 / G5 with the reader-budget credit rule. D4 is strict at 0.020.

**D5 / D6.** D5 is `NOT_APPLICABLE_NOT_JOINT`: no local twin is built. D6 is `NOT_CLAIMED`.

**D2R.**
- The decisive set is START, CONT and every gating control's registered bracket or point.
- Each requires the robustness recipient gates.
- Sign changes and robustness recipient failures are carried into the headline.

**Outcomes.** Five separately reported outcomes in one statement: (1) usefulness, (2) gain beyond the mechanism-specific controls, (3) strong references, (4) material target, (5) joint design and novelty.

**Outcome class, first applicable:**
1. `TASK_INFEASIBLE`
2. `SELECTED_POINT_INFEASIBLE`
3. `INCOMPLETE`
4. `NO_OUTPUT_CHANGE`
5. `MECHANISM_NOT_ASSESSED`
6. `INCOMPLETE`
7. `COMPETITIVE_CANDIDATE` (D2 and D3 met)
8. `MECHANISM_COMPONENT_NOT_COMPETITIVE`
9. `REFERENCE_ADVANTAGE_NOT_ATTRIBUTED`
10. `BOUNDED_NEGATIVE_EXPERIMENT` — carries `partial` = D2's label and the bounds

**Registration.** Family `RFA`: revision 0 is registered if its card was committed before any MODEL_SELECT outcome. R1 reads `REVISED_R1` or `ADAPTIVE_R1`, with its own STOPFIT; `FS_<name>` is a replacement family. Labels of non-registered readings are prefixed.

**Campaign summary.** Counts D2 looks, with the multiplicity statement (up to five comparisons per look).

### Integration notes for the coordinator

1. `rfa/claims.py` is still a copy of `sar/claims.py`: it builds SAR1 records (`cont_declared`, `alt_declared`, `paths['CTRL'/'ALT']`, CALIBRATION). It must be adapted to the RFA1 record, documented in the `rfa/rules.py` docstring and RULES.md §10:
   - `paths['STOPFIT'/'ALTCRIT'/'RAWJ']`, `control_kind['ALTCRIT']`, `altcrit_alias_wire_check`;
   - `selection_data` per control;
   - `cand.selection_data`, `cand.operating_point`, `cand.alias_of_start`, `points`;
   - `controls_not_run` keyed by CONT, STOPFIT, ALTCRIT and RAWJ;
   - `learning_curve` with `half` entries keyed by the registered cell names (map the `@0.5` cells to `S<v>R<r>k<k>`), `diff_cell_se`, `budgets`, `inner_episode_fit_size` and its own `grid_valid`;
   - `grid_validity(..., boot_B=400)`;
   - the D2R `comparators` keys START, CONT, STOPFIT, ALTCRIT and RAWJ.
2. The minimum set for grid validity is CAND, START, CONT and every saved STOPFIT, ALTCRIT and RAWJ point.
3. When the final mechanism card arrives, I will adapt the rules before registration if needed (control kinds, gating set, ALTCRIT's form, the third LC fraction, CAND's selection rule). Please send it by message.

### Open questions (answers are recorded before freezing)

- **(q1)** Is ALTCRIT a path (bracketed along its own epochs) or a single point at CAND's own MODEL_FIT rule?
- **(q2)** Does CAND's operating-point rule use a SEX-based MODEL_FIT diagnostic?
- **(q3)** Is the 0.25 FIT fraction dropped, or reported beside?
- **(q4)** Is a paired-bootstrap SE of the FULL − HALF difference produced? (Recommended; otherwise the rules use the RSS bound and flag it.)
