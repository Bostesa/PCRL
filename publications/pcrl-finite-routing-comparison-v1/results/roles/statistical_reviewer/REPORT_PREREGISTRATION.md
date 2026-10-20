# Statistical reviewer, pre-registration report (saved verbatim with attribution)

Author: the statistical reviewer of campaign `pcrl_finite_routing_comparison_v1`. Returned to the coordinator at about 23:25Z on
2026-10-20 (its own times: 23:08:34Z-23:24:44Z). Saved verbatim; nothing below the line was edited (the harness's two-space
indentation of each line was removed).

---

## Statistical reviewer report: FRC1 rules frozen (campaign `pcrl_finite_routing_comparison_v1`)

I started at 2026-10-20T23:08:34Z and finished at 23:24:44Z (`date -u`). I read no label, ran no real-data computation, fitted nothing, spawned no helpers, did not commit or push, and did not switch branches. CPU is charged at `python3 -m frc.jobqueue charge --label statistical_reviewer --cpu-s 30` (exit 0), an upper estimate.

### Files I wrote (SHA-256 at hand-back)
- `<repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/results/pcrl_finite_routing_comparison_v1/roles/statistical_reviewer/RULES.md`: 2236eb243de4868dad1fa91b8672d5f703735a2cb25cfaf0cb5a48d319f00235
- `.../roles/statistical_reviewer/RULES.json` (schema FRC1-rules-v1): e1896c08200aafb7fb83a548444d8d648c2332ab3fa96531e143cc1cd6ddb73c
- `<repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/frc/rules.py`: fd60d7b170a475c630eb69a0760f2566823da381e5330eb2bb58bcd3efd27c76
- `<repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/frc/test_rules.py`: 0d4c51c08136ecd3b0be5b1b20f4ad28993706de5f7614cfc1bb0930bc264d42

`frc/rules.py` imports `nrm.rules` unchanged and never monkeypatches it. The inherited files are verified unchanged: NRM1 RULES.json 84f18776…, nrm/rules.py d5df8686…, nrm/test_rules.py d03bbed2…, ohn/rules.py 8686a78f…. The new functions are:
- `qpe_label` and `share_removed`
- `construction_choice`
- `qual` and `q2_increment`
- `capacity_gate`: reported only; no function reads it.
- `gate_reproduction`: tolerance 1e-12.
- `capacity_decision`: covers T5, T6 and stop. Results present without their trigger are ignored and flagged. A repair counts only if its card was committed before the fit.
- `d1`, `d2`, `decisions`: D1–D6 plus the outcome class. D3 and D4 are NRM1 G4 and G5, read at the same configuration as D2.
- `strongest_nonzero` and `refit_effectiveness`
- `forbidden_seeds`, `registered_seed`, `grid_validity`: these wrap NRM1's `grid_validity` and replace only its seed check. They forbid offsets 300, 400, 500 and 600, and require every seed to equal exactly MODEL_SEED_BASE + 700 + 10v + k.

### Tests
`python3 -m frc.test_rules`: **15 of 15 PASS, exit 0**, invented data only. The tests cover:
- missing inputs giving INCOMPLETE
- a version disagreeing in sign
- exact aliases (B1 = A, PRC_S2 identical to TWIN)
- ties (an exactly-zero difference goes to A, an inclusive Q2 tie passes, a tie in strength goes to the smaller lambda)
- the cascade 64 → 128 → T5 → 256 → T6 → repair → stop
- a failing original capacity gate leaving the decision unchanged
- seeds at +600 rejected even though NRM1's own check accepts them

As an extra check, I seeded 10 single-line bugs into an in-memory copy of `frc/rules.py`; the suite caught all 10. No file was changed by that check.

### Disclosure (known before freezing)
The published NRM1 values already show that A at (16, 64), i.e. PRC_TWIN, meets QUAL. Its worst value/cap is 0.100 in v0 and 0.510 in v1, and the tightest Q2 ratio is 0.6 (occupation NLL). So T5, T6 and stop can only be reached if B1 is the chosen construction and then fails QUAL. I did not change the Q2 increments.

### Changes requested of the protocol (exact fixes are in RULES.md section 10)
- **C1:** `frc/devreaders.py` still has NRM1's reader settings: offset 600, tags `NRM1_devgrid`, bootstrap seed 2026102000, and forbids only 300/400/500. Change these to 700, `FRC1_devgrid_R<r>` and 2026102100, and forbid 600, before E2. The frozen rule refuses offset-600 or unregistered seeds.
- **C2:** T1 reads no SEX, so the men's/women's Brier rows of the Q-PE report should come from T3 (MODEL_SELECT) only.
- **C3:** Section 6.2 is circular. Choose the construction at each capacity separately, then apply QUAL to the chosen construction only.
- **C4:** D1 has no label for "B1 chosen and qualified but Q-PE only POINT_ONLY". Add `QUALIFIED_B1_REPAIR_NOT_RESOLVED`; it is not a repair claim.
- **C5:** If HQ or SHUF has no strength that meets S2, the matched control is the identity (TWIN), flagged `<name>_AT_IDENTITY`. If PRC has none, D2–D4 are `NOT_ASSESSED_NO_FEASIBLE_STRENGTH`, with no fallback to another budget.
- **C6 (forking path):** D3 and D4 are read only at the configuration selected at S2.
- **C7:**
  - Define the ORD frontier path as [TWIN, the distinct ORD_S1–S3].
  - Require grid entries and paired costs against every point on that path.
  - Report realized spending against HQ_S2 and SHUF_S2 beside the result; they are matched on budget, not on actual spending.
- **C8:** Use the exact "genuinely effective refit" definition: split assignment hashes differ, SCORE overlaps are recorded and lie in [0, 1), and grid validity holds on all 8 seeds.
- **C9:** Add the QUAL disclosure above to the protocol.
- **C10:** A T6 repair is judged by the same choice rule and QUAL; its card cannot change them.
- **C11:** ENGINEERING_REPAIR means D1 = REPAIRED and D2 not met. `D3_WITHOUT_D2` is a flag, not an outcome class.

### Not settled
- The 2-SE household-clustered SE on nested cross-validated losses ignores fold-to-fold training variability, so it can be optimistic. The MODEL_SELECT sign check only partly guards against this.
- NRM1's known limitation carries over to D2–D4: about an 8% false-pass rate when one version or split dominates.
- G1 at PRC_S2 is read on the same MODEL_SELECT people that chose the protection capacity, so it depends on that selection.
- I did not check whether the `frc` code records select_path aliases (an S2 selection equal to S1's).

Two things the coordinator needs to know when using the code:
- `frc/test_rules.py` imports its invented-data helpers from `nrm.test_rules`, which is unchanged.
- `decisions()` needs grid entries for PRC_S2 against TWIN, SHUF_S2, HQ_S2, every ORD point and all nine references. Without them it returns INCOMPLETE.

---

# Second pass (saved verbatim with attribution)

Returned to the coordinator at about 23:41Z (its own times: 23:30:38Z-23:40:18Z), after the coordinator's adjudication. Saved
verbatim (the harness's two-space indentation removed). The coordinator then asked for one correction (D3 / D4: an asymmetric
reference is credited on the non-table winner condition only, because the nested MODEL_FIT LOOKUP cannot exist for references);
see the third pass below.

## Statistical reviewer: second pass done — every adopted item is implemented, tests 17 of 17 pass

Second pass ran 23:30:38Z to 23:40:18Z (`date -u`), before any outcome. I read no labels, used no real data, launched no sub-agents, and made no commits or pushes. I read ADJUDICATION_PREREG.md and the challenger's REVIEW_PREREGISTRATION.md, then changed the rules files. CPU is charged at `python3 -m frc.jobqueue charge --label statistical_reviewer --cpu-s 20` (exit 0), on top of the 30 s charged in the first pass.

### Files and new SHA-256
- `results/pcrl_finite_routing_comparison_v1/roles/statistical_reviewer/RULES.md`: 8bae83416340309e556ee6a799db6d59d103eb9be2e8e12352d2a321f08e6f82
- `.../roles/statistical_reviewer/RULES.json`: 09834b7e97e376eb7bf216c5ae57dc236862df71ec54f12dcfdcefd00fb0c75c
- `frc/rules.py`: 4a02ea95c321213d67334ccbd9095543e17205040c430e6515edd7692b86e14d
- `frc/test_rules.py`: 248ff2405c6a77eae4d8f3f9783a351a46804681476b1a2bd33ffd2ba5485025

The inherited files are unchanged (NRM1 RULES.json 84f18776…, nrm/rules.py d5df8686…, nrm/test_rules.py d03bbed2…, ohn/rules.py 8686a78f…), and `nrm.rules` is still imported as is. RULES.md section 13 is the change record, dated with `date -u`. Section 12, the first freeze's completion record, is kept as written; section 14 is the new completion record.

### Tests
- `python3 -m frc.test_rules`: **17 of 17 PASS, exit 0** (invented data only, about 0.08 CPU-s).
- I also seeded 17 single-line bugs into an in-memory copy of `frc/rules.py`; the suite caught all 17. No file was changed by that check.

### What I implemented (your items 1–6)
1. **QUAL Q2 now uses the S2 budget** (`Q2_INCREMENTS`): occupation NLL 0.008, income NLL 0.004, occupation accuracy loss 0.006, income accuracy loss 0.004, guards 0.0030, occupation summed Brier 0.0030, income summed Brier 0.0020.
   - The disclosure text is in RULES.md and RULES.json, worded as you gave it.
   - A test checks the published guard arithmetic: 0.00255 + 0.0030 = 0.00555 > 0.005 fails, while the old S1 form (0.00405) passes.
2. **Q-PE uses SE_used** = max(household SE, SD of the 5 fold means / √5), via `fold_means` and `se_used`.
   - It reads `nested[v]['folds'][i]['paired_fold_means'][metric]`, or the alternative `nested[v]['fold_means'][metric]`.
   - If the fold means are missing, are not exactly 5, or are not all finite, the label is INCOMPLETE.
   - The reported side rows fall back to the household SE and carry the flag `FOLD_MEANS_MISSING_V<v>`.
3. **D2 is rebuilt as five conjuncts:**
   - (a) G1.
   - (b) TWIN: `nrm.rules.comparison` must pass.
   - (c) `nrm.rules.frontier_comparison` must return DOMINATES against the paths [SHUF_S2, SHUF_BR], [HQ_S2, HQ_BR] and [TWIN, ORD_S1..S3, ORD_BR]. Only existing points are used and aliases count once. An empty HQ or SHUF path is compared against [TWIN] with the flag `<name>_AT_IDENTITY`; that path gives no credit when PRC spends more than TWIN.
   - (d) F4 (`f4`): a failure gives `NOT_MET_F4`, which is not a stop.
   - (e) Reader targeting (`targeting`), checked against TWIN and every bracketing point Q* chosen by the frontier check; a failure gives `READER_TARGETED`.

   Other D2 points:
   - New label `NOT_ASSESSED_EXACT_ALIAS` (the family stops), set from the input flag `prc_s2['every_nonzero_selection_alias_of_twin']`.
   - The partial labels stay, but the comparisons are renamed TWIN, SHUF_FRONTIER, HQ_FRONTIER and ORD_FRONTIER.
   - Label precedence: grid validity, then no feasible strength, then exact alias, then INCOMPLETE, then NOT_MET_F4, then READER_TARGETED, then COMPONENT / MET_WITHOUT_TASK_FEASIBILITY, then NO_REDUCTION, then PARTIAL.
4. **D1** gets `POST_HOC_REPAIR_QUALIFIED` for any T6 repair, never `REPAIRED_TASK_ONLY_RELEASE`, so a repair can never yield ENGINEERING_REPAIR. A repair counts only with `card_committed_before_model_select_read` (or `card_committed_before_fit`) set to true.
5. **D3 reader-budget rule** (`_d3`): a reference flagged `READER_BUDGET_ASYMMETRIC` is credited only if both 3(e) conditions hold against it. Otherwise it counts as not beaten and D3 becomes NOT_SUPERIOR, with the reason `_NOT_CREDITED` or `_INPUTS_MISSING`.
6. **The capacity record** now stores, for each capacity, `other_construction` and `other_would_qualify` (None when that construction's utility is missing). These are recorded and never used.

### Things you should know or decide
- **D4 (NRM1 G5) is not covered by the reader-budget rule**, because your instruction named only D3. Say if you want it applied to D4 too.
- **Asymmetric references may lose credit by default.** A reference flagged asymmetric with no nested MODEL_FIT LOOKUP input is not credited. That is likely for the continuous references, so a flagged reference will count as not beaten unless those inputs exist.
- **D2 needs these records as inputs:**
  - `paths`, `alias_of` and `costs_vs` for every path point
  - `f4`
  - `nolookup` per-cell benefits for TWIN, every path point and the 9 references
  - `nested_lookup`; for TWIN, F4's difference is used if absent, with the flag `TWIN_NESTED_LOOKUP_FROM_F4`
  - the per-reference `reader_budget` flags
  - the `folds` field of the nested record
- **Not settled:**
  - The bracket rule that picks HQ_BR, SHUF_BR and ORD_BR is yours; `d2` takes those points as given.
  - SE_used is still only a rough guard against fold-to-fold dependence, because the 5 outer-training sets overlap by 75%.
  - If a control point is an exact alias of TWIN, its path reduces to [TWIN] without the `AT_IDENTITY` flag.
- **Process note:** one python heredoc was used for the constant edits in `frc/rules.py`. This campaign's brief does not forbid it.

---

# Third pass (saved verbatim with attribution)

Returned at about 23:45Z (its own times: 23:42:15Z-23:44:06Z). Saved verbatim (indentation removed). The coordinator accepted the
three-valued interpretation it asks about (a missing input is never the reason for a failing label, and it does not block a failure
that another reference already determines).

## Statistical reviewer, third pass: reference reader-budget correction applied to D3 and D4 (17 of 17 tests pass)

I worked 23:42:15Z to 23:44:06Z (`date -u`), before any outcome. No labels, no real data, no sub-agents, no commits or pushes. CPU is charged with `python3 -m frc.jobqueue charge --label statistical_reviewer --cpu-s 10` (exit 0). That makes 30 + 20 + 10 s across the three passes.

### New SHA-256
- `results/pcrl_finite_routing_comparison_v1/roles/statistical_reviewer/RULES.md`: 2b7e0be8672cb5eb32a195e91c21b4d6b67284b5a14e7a1a584fdc4ffe256fd0
- `.../roles/statistical_reviewer/RULES.json`: a0dc0fc7f377c26339df0c96e6abda9784d725ccab8cb204e1cfa7d789029510
- `frc/rules.py`: effcc20c2f903aab3d1a6ec7b69d3eaaedd48d56f043fa64f2b6635c20b33d3a
- `frc/test_rules.py`: 4cdcb9bb3b1be5703e65e49c135b9917c7f82207d866198f3c461c444acbb68d

The inherited files are unchanged (nrm/rules.py d5df8686…, NRM1 RULES.json 84f18776…).

### What changed
- **The rule:** a reference flagged `READER_BUDGET_ASYMMETRIC` is credited iff the non-table winner's point benefit against it (the `nolookup` extra) is > 0 in both version means. A new function, `reference_credit`, implements this. The nested MODEL_FIT LOOKUP condition no longer applies to references.
- **Missing `nolookup` inputs** make that reference INCOMPLETE; it is never counted as "not beaten".
- **D3 (`_d3`) and D4 (`_d4`)** both use the rule; D4 covers NEURAL_RAW and H_LJ.
- **D2 is unchanged:** its condition (e) still needs both point conditions against TWIN and every bracketing point.
- **Records updated:**
  - RULES.md: the D3/D4 paragraph in section 6, item 7 of change record section 13 (dated), and the completion record in section 14.
  - RULES.json: `decisions.reader_budget_rule_references`, `D3`, `D4`, and `change_record`.

### One interpretation for you to confirm
You said a missing input makes D3/D4 INCOMPLETE. I applied that only where the missing reference could change the result. The rule can only withdraw credit, so:
- A NOT_CREDITED reference turns a met D3 into NOT_SUPERIOR, and any D4 level into NOT_MET.
- An INCOMPLETE flagged reference turns a met D3, or a D4 level above NOT_MET, into INCOMPLETE. This makes the outcome class INCOMPLETE through D3.
- A D3 or D4 that already fails on another reference's own result stays NOT_SUPERIOR or NOT_MET. The incomplete reference is listed under `incomplete_references` and is not counted as failing.

So a missing input is never the reason for a failing label. Say if you want INCOMPLETE in every case instead; that is a one-line change.

### Tests
- `python3 -m frc.test_rules`: **17 of 17 PASS, exit 0.**
- The renamed test `t_d3_d4_reader_budget_asymmetry` covers:
  - credit unaffected by a negative nested LOOKUP;
  - not credited, giving NOT_SUPERIOR / COMPONENT;
  - missing input, giving D3 INCOMPLETE and outcome INCOMPLETE;
  - missing input while another reference fails on its own, giving NOT_SUPERIOR;
  - D4 not credited (NOT_MET), D4 missing (INCOMPLETE), a D4 that is already NOT_MET staying NOT_MET, and D4 without G1 (MET_WITHOUT_TASK_FEASIBILITY).
- I also seeded 20 single-line bugs into an in-memory copy of `frc/rules.py`. The suite caught all 20, including:
  - the reference rule dropped from D3 or from D4;
  - a missing reference input counted as "not beaten";
  - the nested LOOKUP condition required for references.

  No file was changed by that check.

Nothing else was changed from the second-pass freeze.
