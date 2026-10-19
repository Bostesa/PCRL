<!-- Attribution: written by the engineering / claims reviewer (an agent assignment of this campaign) and returned as text in its
final hand-back (its platform rules forbid subagents from writing report .md files). Saved verbatim by the coordinator at 00:41Z
(2026-10-19, date -u); not coordinator analysis. The patch files in PATCHES_WAVE1/ were written by the reviewer. The coordinator's
disposition of every finding is adjudication A2 in roles/coordinator/ADJUDICATIONS.md. -->

## Engineering review, wave 1

**Window.** 00:18:20Z to 00:39:21Z (`date -u`). I read no label, ran no real-data fit and committed nothing. Besides the code, I read the label-access log and the run records as metadata, and ran invented-data probes.

Patch files in `PATCHES_WAVE1/`: `ENG1-01.diff`, `ENG1-02.diff`, `ENG1-05.diff`, `ENG1-08.diff`, `ENG1-09.diff`. At 00:39:21Z each one passed `git apply --check` against tip cd14296ae, and all five apply together. None has been applied, and I edited no production file.

**Code reviewed** (SHA-256 prefixes):
- common d2aba3fc
- evaluate 0154e14f, including `conditional_cmd`
- deploy 73b77ac1
- sizefit de5d9cb4
- control fd778afb
- correct 46cfb5e5
- nested a63fb59e
- select 1ffd66cf
- rules be06bfa8
- claims 1c73f252 and banks fac72f04, for how they feed the rules

**State during the review.**
- The trigger had already been evaluated at 8da1ad661: TRIGGER_FIRES.
- By 00:31Z, 9 A1 size fits and 7 control Stage B runs were done, with no SUPPORT_FAILURE.
- The conditional-stage code was frozen at 368c9ce93.
- No SIZE_RESULTS, NESTED, SELECTION or REALIZED record existed yet.

**Summary.**
- 1 BLOCKER (conditional and currently latent), 1 MAJOR, 9 MINOR.
- One correctness issue I found mid-review was fixed by its owners before the freeze (ENG1-03); I verified the fix.
- No held-out household reaches any upstream fit in the nested estimator; the frozen H3 income is a disclosed exception (ENG1-10).
- Sign conventions agree across evaluate, rules, nested, select and claims.

### ENG1-01 — BLOCKER (conditional). Apply before `python3 -m tsz.evaluate size`
- **Where:** `tsz/evaluate.py` lines 278-279 (size contrasts), 299-300 (control level), 316-317 (DiD), 338-340 (control-also-meets-reserve), 375 (encoding). Root cause on the rules side: `tsz/rules.py:211-216`.
- **Defect:** when a cell is missing, the code skips it (`continue`), and `summarize_orderings` then labels whatever orderings remain. RULES 2 and 3.3 say a missing ordering makes the label INCOMPLETE.
- **Failure scenario (reproduced on invented data):** with O3 missing, `size_summary` gives `SIZE_GAIN_RESOLVED` from three orderings. A missing cell can come from:
  - a class-support failure;
  - a crashed control Stage B (TEACHERS present, STAGE_B absent, so the T_TREE cell enters and TASK_REFIT is dropped);
  - running `size` before every fit has finished.
- **Why it is serious:** `SIZE_RESULTS.json` is write-once, so a premature run permanently records the primary endpoint, control-explains labels and IR readings computed on partial orderings. Separately, `control_also_meets_reserve` silently drops a missing recipe cell.
- **Patch `ENG1-01.diff`:**
  - `_size_cells()` refuses to read MODEL_SELECT until every registered A1, A1_ADM, T_TREE and TASK_REFIT cell is either fitted or recorded in SUPPORT_FAILURE.json.
  - Missing cells become NaN placeholders, so the label is INCOMPLETE.
  - `control_also_meets_reserve` returns 'INCOMPLETE' when a cell is missing.
  - The interpretation now receives `ok is True`; otherwise the truthy string would fire IR3.
  - `missing_cells` is recorded in SIZE_RESULTS.
- **Process note:** evaluate.py is registered code, so this needs a dated amendment before the size stage.
- This file also carries ENG1-06 and ENG1-07.

### ENG1-02 — MAJOR (operational). Apply before any `tsz.correct realized`
- **Where:** `tsz/correct.py:601-603`, against `tsz/select.py:49-57` and `76-79`.
- **Defect:** `realized(v)` only requires `SELECTION_V<v>`. `select.run` refuses after any MODEL_SELECT read by `tsz.correct`, and that check is not specific to a version.
- **Failure scenario:** the lane plan runs `realized` after SELECT, outside the lanes. If `realized --version 0` runs before lane V1's SELECT, the registered V1 selection is blocked and would need a deviation.
- **Patch `ENG1-02.diff`:** `realized` requires both selection files. `evaluate.conditional_cmd` and `banks` already require both selections to be committed.

### ENG1-03 — fixed before use, verified
- **Defect (now fixed):** the identity release r_0 = softmax(log p_W + 0·Δ) was not bitwise equal to W. On invented data 19,757 of 20,000 rows differed (max 4.4e-16). This broke RULES 8.3 ("at s* = 0 every arm equals W exactly") and the exact-alias zero cells RULES 11 relies on.
- **Fix in place:** `correct.release()` at correct.py lines 73-78 is used at lines 336, 345 and 556, and at nested.py lines 208-216. `conditional_cmd` and `banks` use W itself at s = 0.
- Also fixed by the owners before the freeze:
  - `correct.verify` now checks the production witness path.
  - `control_teacher` now uses the hash-checked `control.load_teachers`.

### ENG1-04 — MINOR, retrospective (no effect)
- **Where:** `tsz/evaluate.py:178-221` (`trigger_cmd`).
- **Defect:** RULES 7 T3 says a disagreement between the two deployment records "blocks the trigger until resolved". `trigger_cmd` does not implement this:
  - it never reads `deploy/DEPLOYMENT_CHECKS_W100.json`;
  - it would have written a write-once TRIGGER_DOES_NOT_FIRE if `DEPLOY_W100.json` had been missing or malformed;
  - it does not check package names.
- **Checked now:** the two records agree.
  - Verifier: versions "0" and "1" have bitwise and refusals_all_pass both true, names A1:A1__S100__V0 and V1.
  - Coordinator: all_bitwise and all_refused true for W100_V0_B2 and W100_V1_B2.
  - So TRIGGER_FIRES stands.
- **Action:** state the agreement, with both files' SHA-256, in the trigger or closeout record. Do not re-run the trigger. No patch.

### ENG1-05 — MINOR (statistical reviewer to decide)
- **Where:** `tsz/rules.py:353`.
- **Defect:** IR4 fires only when the size explanation is UNRESOLVED. RULES 3.6 also lists `SIZE_EFFECT_UNRESOLVED` alone. On invented data, `interpretation('ALL_SIZES', 'SIZE_EFFECT_UNRESOLVED')` returns only ['ALL_SIZES'], and the FULL_NOT_REPRODUCED case returns only ['IR2'].
- **Patch:** `ENG1-05.diff` (one line), or a RULES.md erratum instead. Decide before SIZE_RESULTS.

### ENG1-06 — MINOR
- **Where:** `tsz/evaluate.py:352-367`, the reproduction step.
- **Defect:** it skips source roles missing from the new release, and reads RELEASE.npz and OCC.json without their hash checks. RULES 5 covers every role the source stored.
- **Effect today:** none. sizefit stores all six roles, and `sizefit.repro` already compared them all bitwise.
- **Patch:** in ENG1-01 (hash-checked loaders; a missing role counts as a failure).

### ENG1-07 — MINOR (reporting)
- `min_class_count` is looked up as a top-level FIT.json key; it actually lives under `class_support`.
- The Stage B count `n` and epoch order are missing from the records.
- Pairwise subset-to-subset differences exist only for A1, but RULES 3.4 asks for them per method.
- The S100 control reproduction (`runs/CONTROL/S100__V<v>/REPRO.json`) is not shown in SIZE_RESULTS.
- ENG1-01 fixes the first two; the others belong in the decision documents.

### ENG1-08 — MINOR
- **Where:** `tsz/common.py:295-297`.
- **Defect:** a label-free `load(..., subset=...)` silently returns all MODEL_FIT rows. No caller does this today.
- **Patch `ENG1-08.diff`:** refuse a subset on label-free reads.

### ENG1-09 — MINOR
- **Where:** `tsz/nested.py:231-232`, 288, 302.
- **Defect:** the npz is written before its JSON, and either file blocks a retry. A crash between the two writes makes the step refuse permanently.
- **Patch `ENG1-09.diff`:** an npz without its JSON is renamed as an incomplete attempt and kept.

### ENG1-10 — MINOR (disclosure)
- **Where:** nested.py:242 and correct.direction.
- PAIR's critic reads the frozen H3 income S<v>, which was fitted on all MODEL_FIT, including each held-out outer fold. It is the only upstream input not refitted per fold. It is used only by PAIR, and only in training.
- The H3 MODEL_FIT row order cannot be checked by row id: its RELEASE.json has no row-id hash. The order is the admitted export order, as in the source campaigns.
- Keep the H3_NOTE disclosure next to every PAIR nested estimate and C4. No patch.

### ENG1-11 — MINOR
- **Where:** `tsz/select.py:39`.
- **Defect:** the pre-selection guard blocks the registered selection if any MODEL_SELECT read's purpose contains "correction(s)", "corrected", or upper-case LOC, PAIR or CONT. For example, the purpose "replay the correction stage inputs" matches.
- **Current log:** 2 MODEL_SELECT reads, none flagged.
- **Action:** keep those words out of verifier and reader purposes until both selections exist.

### ENG1-12 — note, no defect: conditional evaluation and claims wiring
- `conditional_cmd`:
  - point status covers 14 conditions per version (`value <= cap`, same as scm_bench);
  - the occupation-side reserve uses each release's own SE;
  - W is used at s = 0;
  - it requires both selections to be committed.
- `claims.py` wiring:
  - C2 through C6 read the correct comparisons;
  - recipient views are candidate minus comparator and the pair view is comparator minus candidate, matching `devreaders.paired`, RULES 9 and `rules.recipient_gate`.

### Checked with no defect found
- **common:** the ROLE × FIELD × OPERATION × PURPOSE policy, the registration gate, closed roles, whole-household subsets (ceil gives 6,669 and 9,336), and the entry-point check.
- **evaluate:** per-person metrics match RULES 3.1; `_loss_change` and paired_diff orientation; level = loss(A1) − loss(control); DiD from the per-person double difference with cluster and subgroup SEs; encoding = loss(FIT-only) − loss(admitted); S100 aliasing.
- **deploy:** refusals at the wrapper, CSV and underlying levels; stored wire equals the evaluated release on all six roles; hidden outputs refused.
- **sizefit:**
  - inner folds come from all MODEL_FIT units and are then restricted to the subset;
  - T* is the first argmin from `cv_select`, then refit;
  - the FIT-only dictionary is applied to every role;
  - class-support rule enforced;
  - at S100 it reproduces the source (T* 168 and 154).
- **control:**
  - `run_arm_cyclic` matches `leafcorr.run_arm` line for line except the marked change;
  - the switch happens at n ≥ 10,240;
  - the run records show S50 cells cyclic with 800 updates each and S100 unchanged.
- **nested:** every fold's dictionary, witness (inner CV plus 5-fold base), routing and correction is fitted only on rows with ncv ≠ k; dictionary hashes are cross-checked between steps; costs are measured against the fold's own witness, with SEs clustered over all MODEL_FIT households, in the orientation `select_strength` expects.
- **select:** `skey` and strength-set wiring.
- **rules:** checked against RULES.md for the reserve, margin pattern, control rules, reproduction, trigger, `select_strength` (prefix rule and PASS_AFTER_FAIL), stability with distinct-cell degrees of freedom, target, gate, equal-reserve frontier and `claim_labels`.

### Not checked
- `devreaders.py` beyond `paired()`.
- `seedcheck.py`, `tables.py` (new and untracked at 00:38Z).
- `jobqueue`, `lane`, `closeout`, `custody`.
- The verifier's code.
- I did not re-run the CatBoost or torch selftests, to save the CPU budget; I read their records instead.
