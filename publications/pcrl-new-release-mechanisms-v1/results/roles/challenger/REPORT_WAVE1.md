# Challenger, wave 1 report (saved verbatim with attribution)

Author: the challenger agent (simpler explanations and strong controls) of campaign pcrl_new_release_mechanisms_v1. Returned to the
coordinator at about 14:57Z on 2026-10-20 as its final message. Saved verbatim by the coordinator; nothing below was edited.

---

All three wave-1 deliverables are done and committed (14:57Z, ahead of the 15:35Z target). Both invented-data selftests pass: `nrm.ctl.post` 29/29 and `nrm.ctl.frontier` 17/17. No real data, labels or `.private/` files were touched.

Commits (own files only, explicit paths, no trailers):
- `80306d498`: `nrm/ctl/__init__.py`, `nrm/ctl/post.py`, `nrm/ctl/frontier.py`
- `0da16b41c`: `results/pcrl_new_release_mechanisms_v1/FAILURE_MAP.md`
- `b52bbcc84`: `nrm/ctl/post.py` (adds a decision-preserving random channel)

CPU: 20 CPU-s charged as an upper estimate (`nrm.jobqueue charge --label challenger`) for the selftests and one invented-data scale check.

Two things to fix in my record:
- **Frontier selftest, first run: 15/16.** The test's hand-computed expected label was wrong. The interpolated gap is 0.003, above the 0.002 margin, so the code's `CANDIDATE_AHEAD` was correct. I corrected the case, kept the original point as an extra case, and added one new case; the code was unchanged.
- **Post selftest, first run: 26/26 but with a RuntimeWarning** (0×inf from merged-away cells). Merge costs are now computed over live cells only, with a non-finite guard. Results were identical before and after, and later runs use `-W error`.

---

# 1. Failure map

Written to `<repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/results/pcrl_new_release_mechanisms_v1/FAILURE_MAP.md` (110 lines).

**How it was built.**
- Six read-only extraction passes over the decision documents of the 39 earlier `results/pcrl_*` folders.
- The two residence channel studies are not in this worktree. They were read with `git show` from local branches `research/pcrl-stochastic-channel-v1` (deb6992ea) and `research/pcrl-stochastic-replacement-overnight-v1` (e3415b94d).
- I re-checked 8 key numbers against the sources myself; all 8 matched. Values computed by an extraction pass, not registered endpoints, are marked (*).

**Comparability.** Only rows 2, 5 and 9, and the ACS halves of row 3, are ACS 2019 FL income/occupation (pair AUC about 0.60–0.67). The rest are reused Adult (pair AUC about 0.74–0.90) or ACS 2018 residence. Only deltas and failure modes carry over between settings.

**The 10 rows (operation; dominant failure; control lesson):**
1. **Adversarial penalties and refreshed critics** (Adult). Utility cost: "strength is the lever".
   - RAW-J β0.3: pair 0.802 vs U 0.883, at occupation NLL excess +0.016/+0.012/+0.012 (*).
   - NORM-J sits on the raw-β line (0.784 interpolated vs 0.786 measured).
   - None of these studies had a SEX-blind control matched to the candidate's spend.
2. **Leaf-score corrections** (ACS). Recovery not reduced at a feasible strength.
   - LOC vs W: +0.0046, POSITIVE_NOT_STABLE.
   - The s=1.0 gain (+0.041) fails utility.
   - CONT and LOC absorb the effect, and TASK_REFIT shares the training-size gain. The line is CLOSED.
3. **Tree privacy training and frozen hybrids.** Benefit below 0.020 against eligible references.
   - H_LJ: −0.001/−0.005 vs NEURAL_RAW.
   - H_JJ reaches 0.603/0.606 but is utility-ineligible.
4. **Frozen-bank random, sampled and replacement channels.** The bank was infeasible or recovery was not reduced.
   - The identity action alone fails occupation NLL (+0.0043/+0.0054).
   - No randomized release was ever attacked on ACS FL.
5. **SCM, IBSI and shared-cell merging** (ACS). Benefit not separated from spending.
   - SCM 0.6234/0.6326 vs ORD_NLLM 0.6570/0.6579, but SCM spent more on DEV.
   - No ordinary release reached SCM's spend.
   - The 0.001-nat match did not carry over.
   - Its "TWIN" used SEX through the within-SEX guard.
6. **Calibration and learned decoders.** Utility cost. The decoder removes no information, and temperature ties D0.
7. **Bottlenecks and finite codes.** Utility cost.
   - JOINT gives +0.0336, but at occupation LL upper bound 0.0121.
   - With spend held, the benefit is +0.0022.
   - Joint ≈ sequential.
8. **Grouped feature selection.** Infeasible on a weak full-input baseline; drops for one recipient vanish in the linked pair.
9. **ACS income codes** (B3CTL and its successor).
   - The benefit is below 0.020 and uses about the whole income Brier budget.
   - It is reader-sensitive, and confirmation is infeasible.
   - Missing: continuous-mixture and matched-temperature controls.
10. **Task-conditioned representations and adapters.** Benefit below 0.020. Local, unconditional and sequential twins reproduce it.

**Cross-family notes.**
- Drops of 0.020 or more always came with a failed clause or with spend beyond every evaluated control. With spend held, benefits were about 0.002–0.018.
- Coalition terms rarely beat local or sequential versions.
- Leakage moves between recipients.
- Readers decide near 0.01.
- Undertrained baselines flattered candidates.
- Everything is development evidence.

**Operations not found in the records read:**
1. SEX-blind coarsening of a full-size twin as a path that brackets a candidate's spend.
2. Flattening that keeps the decision and top-class confidence but replaces the remainder with SEX-blind class shares.
3. A continuous-mixture control, and a temperature-matched control for transplants.
4. Any randomized release attacked on ACS FL; the simplest baseline would be randomized response on the decisions.
5. Coalition-aware correlated randomness (positive on synthetic data only).
6. Input-side generalization or erasure on the six raw fields or the 145-column encoding.
7. Training-time SEX-balanced reweighting.
8. The MMP mixture over SEX values (data-free only so far).

---

# 2. Controls catalogue (full text)

**A. Simplest explanations every new family must rule out**
- **H1 Task spending.** Lower recovery is bought by coarser, flatter or less confident task outputs. Tested by C2 and C3 on the frontier (section D).
- **H2 Unequal training.** The candidate's model differs from its twin in data, size, epochs, tuning or SEX-labelled exposure. Tested by C1 and C8.
- **H3 Reader artifact.** Vector scale or shape changes so that fixed readers underfit, while no information is removed. Tested by C3, the 16-recipe bank including LOOKUP, and C6.
- **H4 Post-hoc equivalence.** Post-hoc SEX-aware grouping of the family's own twin reproduces it (the closed SCM/IBSI line). Tested by C4.
- **H5 Generic noise.** A randomized family's benefit equals SEX-blind replacement noise, or its own deterministic counterpart at matched spend. Tested by C5.
- **H6 Displacement or reader underfit.** The pair AUC falls because leakage moved to one recipient, or because the pair reader underfits. Tested by C6.
- **H7 Coalition illusion.** The joint term adds nothing beyond local or sequential versions. Tested by C7.

**B. Fairness rules common to every control**
- **F1 Data roles.**
  - Control release parameters are fitted on MODEL_FIT: task labels always, and SEX only for SEX-aware controls.
  - Calibration parameters (cell posteriors, temperatures) are fitted on CALIBRATION, or on MODEL_FIT out-of-fold predictions.
  - MODEL_SELECT labels are never used. MODEL_SELECT enters only through `nrm.evaluate`, `nrm.screen` and `nrm.devreaders` outputs, exactly as for candidates.
  - The code enforces this: fit roles must be MODEL_FIT or CALIBRATION, SEX-aware paths require MODEL_FIT SEX, and SEX-blind paths refuse any SEX input.
- **F2 SEX-blind means no SEX anywhere.** Not in fitting, guards (SCM's TWIN broke this), posterior estimation or level choice. Control strengths are chosen by spend only, never by recovery.
- **F3 Same contract.**
  - Income (n,2) and occupation (n,5) float64 vectors with argmax decisions, for all six roles.
  - Randomized controls emit one draw per person per recipient. Utility is computed per emitted draw and averaged over a predeclared R, and readers see one draw.
- **F4 Same readers.** The formal `nrm.devreaders` grid with the same splits, cells, seeds and bootstrap; the pair winner includes the ignore-one readers. The same screening split in `nrm.screen`.
- **F5 Both versions.** Every control runs in v0 and v1, anchored to that version's twin.
- **F6 Equal effort.**
  - The twin gets the candidate's architecture, capacity, training people, epochs, early-stopping rule, MODEL_FIT tuning trials and calibration.
  - Near the candidate's spend, post-processing paths get about as many evaluated strengths as the candidate had nominee choices. Otherwise the comparison is frontier to frontier, with a selection-adjusted margin from the statistical reviewer.
- **F7 Predeclared knobs** (`nrm/ctl/post.py` constants):
  - K0 = 256 initial cells.
  - Levels: income (64, 32, 16, 8, 6, 4, 2); occupation (128, 64, 32, 16, 8, 5).
  - Temperatures T (1.25, 1.5, 2, 3); smoothing ε (0.05, 0.1, 0.2, 0.3).
  - λ (0.5, 1, 2); replacement q (0.05, 0.1, 0.2, 0.4).
  - Jeffreys α = 0.5; fixed seeds.
- **F8 Task-matched variants.** Post-process only the recipient(s) the candidate changes (for example occupation only, with income = twin). Add the both-recipient variant when the candidate changes both.

**C. The controls**

**C1. Equally trained task-only twin (TWIN; built by the designer).**
- *What.*
  - The family's own code path with every SEX-dependent term at zero: critic weight 0, no SEX-dependent guard, no SEX reweighting.
  - Identical in architecture, capacity, data, training size (full MODEL_FIT if the candidate uses it), household folds, seeds, epochs, early stopping and calibration.
  - For corrections and continuations: the zero-critic continuation (CONT). For transplanted components: the same component applied to the twin.
- *Fair fit.* Same tuning budget on MODEL_FIT folds; never tuned on MODEL_SELECT; `kind='twin'`.
- *Explains it if* either:
  - twin linked recovery − candidate ≤ m; or
  - the candidate spends less than its own twin (BELOW_CONTROL_RANGE). That means the twin is undertrained and must be refit before any claim.
- Every path in C2–C5 starts at the twin.

**C2. SEX-blind coarsening path (BLIND; `Coarsener('blind')`).**
- *What.*
  - Per task, 256 label-free cells of the twin's vectors: quantile bins of income p1, KL k-means for occupation.
  - Merged greedily by the smallest loss of plug-in task information (agglomerative information bottleneck), with snapshots at the predeclared levels.
  - The released vector is the cell posterior, so the argmax can change in mixed cells; that cost is counted.
  - The coarsest levels (income 2, occupation 5) approximate the decisions-only floor.
- *Fair fit.*
  - Partition on the twin's deployed MODEL_FIT vectors (label-free). Merges on MODEL_FIT task labels, using out-of-fold twin vectors if the family cross-fits; otherwise in-sample, recorded as such.
  - Posteriors are calibration parameters. Fit them on the larger permitted sample: MODEL_FIT out-of-fold, or CALIBRATION via `refit_posterior` for levels ≤ 32, or label-free means (`posterior='mean'`).
  - Record the roles, with `kind='control'` and `n_calibrator_fits`.
- *Explains it if* the section-D verdict is CONTROL_EXPLAINS on the primary or any secondary axis, in either version.
- *Credit.* The candidate is credited only with CANDIDATE_AHEAD in both versions. EXTEND_CONTROL or AHEAD_VS_SPARSE_PATH means no claim until the path is extended or densified; this was the SCM failure.

**C3. Flattening (FLAT; `temperature`, `smooth`; data-free).**
- *Properties.*
  - The argmax is kept, so there is no accuracy cost.
  - The map is invertible for a fixed knob, so the best achievable recovery is unchanged.
  - On the income view it is a strictly monotone map of p1, so LOOKUP, decision and quantile-binned tree readers are invariant (up to float rounding). LR/MLP readers may not be.
- *Calibration-matched variant.* For families that change calibration (decoders, Platt maps, transplants): the twin with one CALIBRATION-fitted temperature per task. This was the missing control in income_headroom_successor.
- *Explains it if* FLAT reaches the candidate's recovery at no more than its spend; the benefit is then a reader artifact.
- *Expected result* with a sound slate: a flat recovery profile. A drop of m or more for any recipe flags a scale-sensitive slate.

**C4. Post-hoc grouping of the twin: AWARE, SHUF, RMERGE (`Coarsener('aware' | 'shuffled' | 'random')`).**
- *What.* Same cells and levels as C2.
  - AWARE merge score = task-information loss − λ·(SEX information removed), with MODEL_FIT SEX only.
  - SHUF = AWARE with SEX permuted (seeded).
  - RMERGE = random pair merges (seeded).
- *Explains it if* any of these holds:
  - AWARE reaches the candidate on the frontier: the learned family is no better than post-hoc SEX-aware grouping (closed line).
  - SHUF is within m of AWARE at matched spend: SEX targeting adds nothing over the merge pattern.
  - The candidate is not ahead of RMERGE: noise.
- AWARE is also the post-hoc upper reference that a learned family must beat.

**C5. Random channel without task adaptation (RAND; `replace_random`).**
- *What.*
  - With probability q, a person's vector is replaced by the twin vector of a uniformly random MODEL_FIT person (a SEX-blind bank).
  - `same_class=True` replaces only within the same predicted class: decision-preserving, zero accuracy cost.
  - One draw per person per recipient, R draws, independent seeds per recipient, coupling recorded.
- *Explains it if* RAND matches a randomized family on the frontier.
- *Also required.* The family must face its own deterministic counterpart (same bank: identity or best deterministic map) at matched spend, and the identity action must first be a utility witness.

**C6. Ignore-one-recipient readers and linked recovery (READ; `nrm.devreaders`).**
- *What.* Report income-view, occupation-view and pair AUCs. Linked recovery is the VALIDATION-chosen winner among the pair readers and the ignore-occupation and ignore-income readers.
- *Explains it if* either:
  - a single-view SCORE AUC exceeds the pair AUC (underfit: use the maximum); or
  - the pair falls while one recipient rises beyond its non-worsening allowance relative to the twin and references (displacement).

**C7. Coalition controls (LOCAL, S12, S21).**
- *What.* The same family with per-recipient terms only, and the two sequential versions.
- *Rule.* A coalition claim needs joint − best(LOCAL, S12, S21) > m on linked recovery. Earlier families measured 0.000–0.005.

**C8. Exposure and structure controls.**
- The same training people and the same SEX-labelled exposure as the candidate.
- For codes, a continuous-mixture control (the unquantized score) to separate coarsening from mixing.

**D. Frontier comparison without an exact cost match (`nrm.ctl.frontier`)**
1. **Points.** Per release and version: spend from `nrm.evaluate`, and linked recovery as the mean over that version's 4 registered grid cells, with paired household-bootstrap SEs.
2. **Axes.**
   - Primary: the maximum normalized clause usage, max_i excess_i / limit_i, over the version's 12 clauses plus the 2 men's/women's occupation Brier guards (`clause_usage`).
   - Secondary: occupation NLL excess vs U0, income NLL excess vs U0, and men's occupation Brier excess.
   - Spend is always measured, never taken from the fit.
3. **Control frontier.**
   - The staircase over realized control points (`frontier_step`): the lowest recovery at spend ≤ c. The control never gets an unrealized point.
   - A time-sharing interpolation between staircase corners (`frontier_lin`), used only as a second condition.
4. **Verdicts** (`compare`):
   - CONTROL_EXPLAINS: a realized control point has spend ≤ c + tol and recovery ≤ r + m.
   - BELOW_CONTROL_RANGE: the candidate spends less than every control point.
   - EXTEND_CONTROL: no control point reaches the candidate's spend.
   - AHEAD_VS_SPARSE_PATH: ahead on the staircase only.
   - CANDIDATE_AHEAD: the staircase gap and the interpolated gap both exceed m, with the candidate's spend bracketed.
   - Across axes (`compare_axes`): explained on any axis means explained. Across versions: credit only when CANDIDATE_AHEAD in both versions.
5. **Predeclared tolerances.**
   - tol = 0.10 usage units on the primary axis (0.001 nats on an NLL clause, 0.0005 on a Brier clause, 0.001 accuracy).
   - 0.001 nats on the NLL axes and 0.0005 on the Brier axes.
   - Margin m comes from the reviewer's `nrm/rules.py` (for example, two paired SEs of the recovery difference at the bracketing points); m = 0 for descriptive reads.
   - The 0.020 material target stays separate, against NEURAL_RAW and H_LJ.
6. **Why no exact match.** The SCM 0.001-nat match failed against a paired spend SE of 0.0016–0.0020 nats. Bracketing plus a staircase only requires that the path covers the candidate's spend. It also reports `extra_cost`: the additional spend the control needs to reach the candidate's recovery.
7. **Choosing which control levels to save** (by spend only):
   - Predict each level's spend from the MODEL_FIT plug-in information loss (`record()['plugin_task_mi_nats']`).
   - Save the levels that bracket the candidate's measured spend, plus one stronger level.
   - If the result is EXTEND_CONTROL, save the next level as a logged adaptive step.

**E. Budget and open questions**
- **CPU is negligible.** Eight coarsener fits at n = 20,000 and K0 = 256 took 2.4 CPU-s on invented data.
- **Configurations are the binding cost.**
  - Mandatory per family: TWIN (2) + BLIND at 3 levels (6) = 8.
  - Conditional:
    - AWARE (2) for SEX-targeted grouping claims;
    - calibration-matched FLAT (2) for calibration-changing families;
    - RAND (2–4) for randomized families;
    - LOCAL/S12/S21 (2 each) for coalition claims.
  - Two families come to roughly 16–30 of the 60. Families that share a twin should share its control paths.
- **Q1.** Can a deterministic post-processing ladder of an already saved twin be recorded as one configuration per (control, version) rather than one per level?
- **Q2.** The access layer requires a committed card before any control fit. I can write `results/pcrl_new_release_mechanisms_v1/cards/CTL_CARD_v1.md` once the families' twins are known.
- **Not implemented yet:**
  - a joint income × occupation partition, for candidates that couple the two outputs;
  - top-1-preserving remainder flattening (failure-map item 2);
  - the family-specific continuous-mixture control.

---

# 3. Code and selftests

Files are in `<repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/nrm/ctl/`. They use numpy only, accept arrays only, and load nothing.

- **`__init__.py`**: package description.
- **`post.py`**:
  - `temperature`, `smooth`;
  - `Cells` (quantile or KL k-means partition);
  - `merge_path` (blind, aware, shuffled or random);
  - `Coarsener` (`fit`, `refit_posterior`, `apply`, `cell_ids`, `record`);
  - `replace_random` (with a `same_class` option);
  - `plugin_mi`, plus the role guards.
- **`frontier.py`**: `clause_usage`, `staircase`, `frontier_step`, `frontier_lin`, `compare`, `compare_axes`.

**`python3 -W error -m nrm.ctl.post selftest` → SELFTEST PASS: 29/29 checks.** All checks passed, in these groups:
- identity checks for temperature and smoothing;
- 4 temperatures and 4 smoothing values (valid output, argmax kept);
- MODEL_SELECT refused for both partition and posterior;
- the SEX-aware path refuses a CALIBRATION estimation role, and the SEX-blind path refuses a SEX input;
- blind occupation levels 32, 16, 8, 5 (valid, at most L distinct vectors);
- task information non-increasing along the path, and a bit-identical refit;
- AWARE removes more SEX information than blind and shuffled at L=8, and blind keeps more task information than random;
- income levels 16, 6, 2;
- CALIBRATION posterior refit (cells unchanged, roles recorded);
- label-free mean posterior;
- random replacement (rate about q, draws differ), and same-class replacement keeps every decision.

At L=8 on invented data, plug-in SEX MI was blind 0.164, aware 0.0036, shuffled 0.148, random 0.033 nats. Plug-in task MI was blind 0.380, aware 0.374, shuffled 0.380, random 0.060 nats.

**`python3 -W error -m nrm.ctl.frontier selftest` → SELFTEST PASS: 17/17 checks.** All checks passed:
- staircase, step and interpolated frontiers;
- 9 verdict cases covering all 5 verdicts, including the tolerance case;
- `extra_cost` and the gaps;
- `clause_usage` and mismatched-clause refusal;
- explained on a secondary axis → explained overall.

**Scale check.** On invented data at n = 20,000, all 8 fits took 2.41 CPU-s (actual K0 was 256 for both tasks), and applying every level to 8,000 rows took 0.24 CPU-s.
