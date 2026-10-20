# Claims and selection rules of the finite-routing comparison campaign (FRC1; statistical reviewer; schema FRC1-rules-v1)

**Status.** Writing started 2026-10-20T23:08:34Z (`date -u`). The completion time is recorded at the end of this file. Written
before any outcome of this campaign: I read no label, ran no real-data computation, fitted nothing and called no loader that
reads labels.
- Machine-readable constants: `RULES.json` (schema `FRC1-rules-v1`). Implementation: `frc/rules.py` (pure functions; imports
  `nrm.rules` unchanged as NR; `load_rules()` refuses a RULES.json that disagrees with the module or a changed NRM1 rule file).
  Invented-data tests: `python3 -m frc.test_rules` (prints JSON; exit 0 iff all pass).
- Inputs read: the coordinator's AGENT_CONTEXT.md and draft PROTOCOL.md (written from 23:06Z); the NRM1 RULES.md / RULES.json,
  `nrm/rules.py`, `nrm/test_rules.py`; `nrm/fa/prc.py` (`spend`, `select_path`, `cluster_se`, `BUDGETS`, `CONFIGS`);
  `nrm/evaluate.py` (`paired_costs`, `point_status`, `reserve_of`); `ohn/rules.py` (`conditions`); the card FA_PRC_v1 section 4
  (capacity gate wording); the published GATE records of designer A (structure and the published spends); the reader-seed
  constants of `frc/devreaders.py` (no outputs).
- **Disclosure (known before freezing).** (1) Every NRM1 published record listed in PROTOCOL section 0. (2) Under the adopted S2
  form of Q2 (section 13), A at (16, 64) (= PRC_TWIN) meets Q1 and **fails** Q2: its version-1 men's guard 0.00255 + 0.0030 =
  0.00555 > 0.005. It meets the superseded S1 form (0.00405 <= 0.005). So `QUALIFIED_WITHOUT_PROBABILITY_REPAIR` can occur only
  at (24, 128) or (32, 256), where A's MODEL_SELECT utility is unknown. QUAL(A, (16, 64)) is known from published records and is
  not a prediction. (3) `frc/devreaders.py` at 23:16Z still carried NRM1's reader constants; the coordinator reports them fixed at
  about 23:17Z (change C1).
- Second pass (23:30-23:5xZ): I also read the coordinator's ADJUDICATION_PREREG.md and the challenger's REVIEW_PREREGISTRATION.md
  and implemented every item adopted for the rules (section 13). Still no label, no real data, no outcome.

**What every rule here is.** A development rule on previously and adaptively used records (MODEL_FIT and MODEL_SELECT were used
by earlier campaigns): nothing here is confirmation. No rule changes an original limit. The two-SE reserve and the original
capacity gate are reported, never gating. QUAL and point feasibility are never called confidence preservation. Every uncertainty
statement is asymptotic and model-based. Versions 0 and 1 share every training row and the MODEL_SELECT people: "both versions"
is a procedure-repeat requirement, never replication. Sign convention: every paired difference is loss(first) - loss(second) on
the same people (negative favours the first); accuracy enters as a loss.

## 0. Inherited unchanged (by reference to NRM1 RULES.md)

Original limits (NRM1 section 0); G1-G7 and the useful-component label (section 1); scope sentences (section 2, plus "Point
feasibility, QUAL and the reserve are not confidence preservation"); the nine-reference original benchmark with its published
point labels and alias structure (section 3); point eligibility and the reserve (section 4); the aggregate paired screen, its
labels, the recipient gate and the old every-cell diagnostics (section 5); reader-stability summaries and the A-SC1 recipe
criterion (section 6, except the seed offsets, section 9 below); the frontier rule (section 7); candidate selection (section 8;
reported beside, never a D label); randomized releases (section 9). All through `nrm.rules`, imported unchanged (RULES.json
84f18776..., nrm/rules.py d5df8686...). Nothing in NR is monkeypatched (`t_nrm_rules_unchanged_not_monkeypatched`).

## 1. Q-PE labels per capacity (`qpe_label`; PROTOCOL 6.1)

Input: T1 nested MODEL_FIT records per version (`B1-A`, `B0-B1` paired held-out differences as (estimate, household-clustered
SE), the 5 outer-fold means of each paired difference (`folds[i]['paired_fold_means'][metric]`, or `fold_means[metric]`), and
the spends over CONT of A, B1, B0) and the T3 MODEL_SELECT paired costs B1 vs A (`nrm.evaluate.paired_costs` format).
d_v = nested occupation NLL B1 - A of version v. **SE = SE_used_v = max(household-clustered SE, sample SD of the 5 outer-fold
means / sqrt(5))** (adopted, adjudication finding 17); fold means absent or not exactly 5 finite values: `INCOMPLETE`.

| Label | Condition (first applicable) |
|---|---|
| `INCOMPLETE` | a nested (estimate, SE) missing or non-finite |
| `PE_ERROR_NOT_SUPPORTED` | d_v >= 0 for some v (a tie at 0 is not supported; an exact alias B1 = A lands here) |
| `PE_ERROR_SUPPORTED` | d_v + 2 SE_v < 0 for both v, and the MODEL_SELECT paired occupation NLL B1 - A < 0 at the point for both v |
| `INCOMPLETE` | d_v + 2 SE_v < 0 for both v, MODEL_SELECT value missing |
| `PE_ERROR_POINT_ONLY` | d_v < 0 for both v otherwise |

Reported beside (never gating): the same nested reading (`RESOLVED_NEGATIVE_BOTH` / `POINT_NEGATIVE_BOTH` /
`NOT_NEGATIVE_SOME_VERSION` / `MISSING`) for income NLL, total NLL, summed Brier and accuracy loss of both tasks with their
MODEL_SELECT point values; the men's / women's occupation Brier from MODEL_SELECT only (T1 reads no SEX: change C2); B0 - B1;
and the **share removed** = (spend_A - spend_B1) / spend_A of the nested occupation NLL spend over CONT, per version, defined iff
spend_A > 0 (`NOT_DEFINED_NONPOSITIVE_A_SPEND`), pooled as mean numerator / mean denominator.

Reported rows use SE_used where their fold means exist, else the household SE with flag `FOLD_MEANS_MISSING_V<v>`.

Limitation: the household-clustered SE of nested cross-validated differences treats the refitted pipelines as fixed. The fold
term adds the between-fold spread, but with 5 overlapping outer-training sets (75% shared) it is itself a rough, one-sided guard.
The MODEL_SELECT sign check remains a separate guard, not a correction.

## 2. Construction choice (`construction_choice`; PROTOCOL 6.2)

Applied **per capacity**, on MODEL_FIT nested point estimates only (MODEL_SELECT never chooses): `B1` iff the paired occupation
NLL and the paired total (income + occupation) NLL B1 - A are both < 0 in both versions; otherwise `A` (ties at exactly 0 go to
A). Without a `tot_nll` row the point total is inc_nll + occ_nll (exact for point estimates on the same people; flag
`TOTAL_FROM_TASK_SUM`). The chosen construction is used consistently afterwards at that capacity (QUAL, TWIN, PRC, HQ, SHUF, ORD,
ablation). Choosing per capacity removes the circularity of "the capacity where the protection would run" (change C3).

## 3. QUAL (`qual`; PROTOCOL 5)

QUALIFIED iff, in both versions, Q1 and Q2 hold for the chosen construction's actual MODEL_SELECT utility record:
- **Q1** = G1 of the version: 12 clauses + 2 guards at the point, original caps, inclusive (`nrm.rules.point_eligibility`; a
  missing version or a re-capped / malformed record is `INCOMPLETE`).
- **Q2**: value <= cap - d for every condition (inclusive; cap - d in double precision), with d:

d = the **S2 budget** (the assessment budget of D2 and its HQ / SHUF controls; adopted, adjudication finding 1):

| Condition (both references U0 and Ucal) | d | Largest value / cap allowed |
|---|---|---|
| occupation NLL excess | 0.008 | 0.2 |
| occupation accuracy loss | 0.006 | 0.4 |
| occupation summed Brier excess | 0.0030 | 0.4 |
| men's and women's occupation Brier guards | 0.0030 | 0.4 |
| income NLL excess | 0.004 | 0.6 |
| income accuracy loss | 0.004 | 0.6 |
| income summed Brier excess | 0.0020 | 0.6 |

The record lists per version the failed conditions, the binding condition and each slack after the increment. QUAL is a gate for
starting the comparison, not a claim. Slack for the S2 budget at the twin does not guarantee G1 at PRC_S2 (MODEL_SELECT costs over
the twin can exceed the MODEL_FIT budget); then D2 is `MET_WITHOUT_TASK_FEASIBILITY` or a partial label, never met. No fallback
budget (the challenger's b*) exists. MODEL_SELECT SEX enters QUAL through the guards (an adaptive utility selection common to PRC
and its controls; disclosed).

## 4. Original capacity gate and reproduction (`capacity_gate`, `gate_reproduction`; reported)

- **Original gate** (FA_PRC_v1 section 4, unchanged definition): per version, the construction's nested held-out spend over CONT,
  estimate + 2 SE <= 0.004 nats NLL and <= 0.003 accuracy loss, per task. `ORIGINAL_CAPACITY_GATE_PASS` iff all four metrics pass in
  both versions; `ORIGINAL_CAPACITY_GATE_FAIL` otherwise; `INCOMPLETE` if a value is missing. Computed for A, B1 and B0 at every
  capacity and reported. No function reads it (`t_capacity_gate_reported` shows a failing gate leaves the capacity decision
  unchanged).
- **Reproduction** (PROTOCOL 6, gates T1): T1's A must match every number of the GATE record (CONT and twin nested means; twin spend
  estimates and SEs) within 1e-12 absolute and reproduce its gate label: `REPRODUCED` / `NOT_REPRODUCED` / `INCOMPLETE`. An
  existing capacity without `reproduced = true` is `STOPPED_FOR_DIAGNOSIS` in the capacity decision.

## 5. Capacity order, triggers and stop (`capacity_decision`; PROTOCOL 6.3-6.6)

A deterministic function of the recorded inputs (per capacity: reproduction flag, nested record, MODEL_SELECT paired costs, the
utility of A and B1):
1. For (16, 64), then (24, 128): choose the construction (section 2); evaluate QUAL of the **chosen** construction only. The first
   QUALIFIED capacity is the protection capacity: `PROTECT` (capacity, construction, its Q-PE label). A capacity whose records are
   missing or incomplete stops the walk (`INCOMPLETE`, awaiting it); a later capacity is never used to skip an earlier one.
2. **T5** only if neither existing capacity qualifies: `TRIGGER_T5` until the (32, 256) record exists; then the same steps.
3. **T6** only if (32, 256) does not qualify: `TRIGGER_T6` until a repair is recorded or declined (`'DECLINED'`). A repair is
   labelled POST_HOC; it counts only with `card_committed_before_model_select_read = true` (or the stricter
   `card_committed_before_fit`; otherwise `REPAIR_WITHOUT_COMMITTED_CARD_IGNORED`), is judged by the same construction choice and
   QUAL (its card cannot change QUAL, the increments or any original limit) and can give only D1 = `POST_HOC_REPAIR_QUALIFIED`.
4. **Stop** otherwise: `STOP_FAMILY_FA` with the statement "no task-only finite release qualifies: family FA stops and protected
   routing remains untested".
- Each step records the chosen construction and whether the other construction would have qualified (`other_construction`,
  `other_would_qualify`; None if its utility is missing). It is recorded, never used: the chosen construction is final for its
  capacity, and a failure moves to the next capacity even when the other construction would qualify.
- A (32, 256) or repair result present without its trigger is ignored and flagged (`T5_RESULT_WITHOUT_TRIGGER_IGNORED`,
  `REPAIR_WITHOUT_TRIGGER_IGNORED`). Tested cascade: 64 fails -> 128 fails -> `TRIGGER_T5` -> 256 fails -> `TRIGGER_T6` -> repair
  fails -> `STOP_FAMILY_FA` (`t_capacity_cascade`).

## 6. Decisions D1-D6 and the outcome class (`d1`, `d2`, `decisions`; PROTOCOL 10)

**D1 task-only repair** (at the protection capacity):

| Label | Condition |
|---|---|
| `REPAIRED_TASK_ONLY_RELEASE` | B1 chosen, QUALIFIED, Q-PE `PE_ERROR_SUPPORTED` |
| `QUALIFIED_B1_REPAIR_NOT_RESOLVED` | B1 chosen, QUALIFIED, Q-PE `PE_ERROR_POINT_ONLY` (new label, change C4) |
| `POST_HOC_REPAIR_QUALIFIED` | the protection capacity is a T6 repair (whatever its construction and Q-PE label) |
| `QUALIFIED_WITHOUT_PROBABILITY_REPAIR` | A chosen and QUALIFIED (possible only at (24, 128) or (32, 256)) |
| `NO_QUALIFYING_TASK_ONLY_RELEASE` | `STOP_FAMILY_FA` |
| `INCOMPLETE` | the capacity decision is not final |

**D2 protection component beyond matched controls** (revised, section 13), at the PRC configuration selected at S2 (an alias of
an earlier budget's selection is that configuration, with its brackets). Labels in order of precedence:
1. grid validity (section 9) missing or false: `NOT_COMPUTED_GRID_VALIDITY_UNKNOWN` / `NOT_COMPUTED_GRID_INVALID`;
2. no strength meets S2: `NOT_ASSESSED_NO_FEASIBLE_STRENGTH` (no fallback budget); a capacity stop:
   `NOT_ASSESSED_PROTECTED_ROUTING_UNTESTED`;
3. every nonzero selection transmits exactly the twin's vectors: `NOT_ASSESSED_EXACT_ALIAS` (the family stops);
4. a needed input missing (G1 utility, a grid entry, a paired cost or secondary metric, the F4 record, a targeting input):
   `INCOMPLETE` (never skipped);
5. then the five conjuncts:
   - (a) **G1** at the point;
   - (b) **vs TWIN**: `nrm.rules.comparison` passes (pair benefit `AGG_POSITIVE` and the recipient gate, both views, both versions);
   - (c) **vs SHUF, HQ and ORD**: `nrm.rules.frontier_comparison` = `DOMINATES_CONTROL_FRONTIER` against the control paths
     [SHUF_S2, SHUF_BR], [HQ_S2, HQ_BR] and [TWIN, distinct ORD_S1..S3, ORD_BR] (MODEL_SELECT paired total-NLL costs; only points
     that exist, aliases counted once). The brackets *_BR are chosen on MODEL_FIT by the coordinator's registered rule. An empty
     SHUF or HQ path is compared at [TWIN] (flag `<name>_AT_IDENTITY`), which gives `NOT_SEPARATED_FROM_TASK_SPENDING` whenever
     PRC spends more than TWIN: no credit without a control that spends at least as much;
   - (d) **F4**: nested held-out pair LOOKUP AUC_twin - AUC_prc minus 2 household-bootstrap SE > 0 in both versions;
   - (e) **reader targeting**: against TWIN and every bracketing point Q* that (c) selected, the non-table pair winner's benefit
     (the grid's `nolookup` extra, per-cell on the 8 registered cells) is > 0 in both version means AND the nested MODEL_FIT
     held-out pair LOOKUP benefit AUC_control - AUC_prc (P1) is > 0 in both versions (for TWIN, F4's difference is used if no
     separate value is given; flag `TWIN_NESTED_LOOKUP_FROM_F4`).
6. `NOT_MET_F4` if (d) fails (not a stop; the protected releases are still read); else `READER_TARGETED` if (b) and all of (c)
   pass but (e) fails; else `COMPONENT_BEYOND_MATCHED_CONTROLS` (the only D2 "met": (a)-(e) all hold) or
   `MET_WITHOUT_TASK_FEASIBILITY` ((b)-(e) hold, no G1); else `NO_REDUCTION_BEYOND_ANY_CONTROL` (none of (b), (c) passes); else
   `PARTIAL_<passed>_ONLY`, the passed comparisons joined by `+` in the order TWIN, SHUF_FRONTIER, HQ_FRONTIER, ORD_FRONTIER (for
   example `PARTIAL_TWIN_ONLY`). Flags: `PRC_S2_ALIAS_OF_TWIN`, `<name>_AT_IDENTITY`, `READER_TARGETED_<comparison>` (a passed
   comparison whose targeting fails). The minimum detectable pair benefit is about 0.015 (2 se_agg); "not separated" is the
   default reading. **No partial outcome is a component claim.**

**D3** = NRM1 G4 (all nine point-eligible references, intersection-union, recipient gates) and **D4** = NRM1 G5 (0.020 vs NEURAL_RAW
and H_LJ; its levels kept), both at the **same** configuration as D2 (change C6). **Reader-budget rule for references, D3 and D4**
(adjudication finding 7 as corrected by the coordinator, section 13): a reference comparison flagged `READER_BUDGET_ASYMMETRIC`
(input per reference, `frc.readerbudget.contrast`, pair view) is credited iff the **non-table winner's** point benefit (the grid's
`nolookup` extra against that reference) is > 0 in both version means. The nested MODEL_FIT LOOKUP condition of D2 (e) does not
apply to references (they are never refitted on MODEL_FIT). Missing `nolookup` inputs make that reference `INCOMPLETE`, never "not
beaten". The rule can only withdraw credit, so it is applied in three-valued form: a NOT_CREDITED reference turns a met D3 into
`NOT_SUPERIOR` and any D4 level into `NOT_MET`; an INCOMPLETE flagged reference turns a met D3, or a D4 level above NOT_MET, into
`INCOMPLETE`; a label already failing on its own results stays failing (the incomplete references are listed, not counted as
failing). **D5** = `NOT_ASSESSED_MISSING_CONTROLS` unless
the separate design and both sequential orders exist (then NRM1 G6); ABL_LOC alone is not that control set. **D6** =
`NOT_CLAIMED`.

**Outcome class:** `CANDIDATE` iff D2 met and D3 met; `COMPONENT` iff D2 met and D3 not; `ENGINEERING_REPAIR` iff D1 =
`REPAIRED_TASK_ONLY_RELEASE` and D2 not met (a POST_HOC repair never gives it); `FINITE_NEGATIVE_RESULT` otherwise; `INCOMPLETE`
if D1, D2 or D3 is incomplete or not computed for grid validity. Flags: `D3_WITHOUT_D2` (references beaten without separation from
the matched controls: not a candidate), `POINT_FEASIBLE_NOT_RESERVE_QUALIFIED` (G1 at PRC_S2 without the reserve: near-limit,
never called robust), `PROTECTED_ROUTING_UNTESTED`, `FAMILY_STOPS_EXACT_ALIAS`. NRM1 `select_candidate` may be run over the full ledger and reported beside; it never changes a D label.

## 7. Strongest feasible nonzero release and the classifier refit (`strongest_nonzero`, `refit_effectiveness`; PROTOCOL 10.4)

- **Strongest feasible nonzero release**: among PRC configurations with lambda > 0, G1 (`POINT_ELIGIBLE`), not an alias of an
  earlier configuration and not ALIAS_ZERO against TWIN, with a complete registered grid: the largest registered-cell aggregate
  pair benefit x against TWIN (it may be negative). Ties (|dx| <= 1e-12) go to the smaller lambda, then config_id. None:
  `NO_FEASIBLE_NONZERO_RELEASE` (no refit, no ablation). It is chosen on registered MODEL_SELECT cells (winner's curse):
  descriptive; robustness cells and the ablation never change D1-D6.
- **Genuinely effective refit**: `REFIT_GENUINELY_EFFECTIVE` iff (a) R3 and R4 use different household splits: assignment hashes
  recorded and distinct from each other and from R1 / R2, and every pairwise SCORE household overlap involving R3 or R4 recorded,
  finite and in [0, 1); and (b) the FRC1 `grid_validity` (section 9) holds on all eight seeds S<v>k<k>, k = 0..3, with the
  recorded seed checks of the robustness cells and the minimum set = the strongest release and its matched controls. Otherwise
  `REFIT_NOT_GENUINELY_EFFECTIVE`, and the cells of the failed part ("robustness repetitions" and / or "robustness reader seeds")
  are counted once. The NRM1 `robustness` label is reported beside.

## 8. Required records (the coordinator's inputs)

1. T1 per capacity and version: B1 - A and B0 - B1 paired (estimate, SE) for occ_nll, inc_nll, **tot_nll (per-person summed,
   clustered)**, occ / inc Brier and accuracy loss, each with its 5 outer-fold means (`folds[i]['paired_fold_means']`); spends
   over CONT of A, B1 and B0 for the four gate metrics; the reproduction result.
2. T3 per capacity: the utility records of A and B1 (`nrm.evaluate` format) and `paired_costs` of B1 vs A.
3. P1 / P2: the select_path record per budget (NOT_FEASIBLE recorded), `every_nonzero_selection_alias_of_twin`, the bracket points
   HQ_BR / SHUF_BR / ORD_BR (or None), `alias_of`, the F4 record {'version': {v: {'diff', 'se'}}}, and the nested held-out pair
   LOOKUP benefits {id: {v: AUC_id - AUC_prc}} against TWIN and every path point.
4. E2 grid entries (`paired()` format) of PRC_S2 against every point of the three paths and the nine references; `paired_costs`
   of PRC_S2 against every path point (with `total|nll` and the 8 secondary metrics); the `nolookup` extra's per-cell pair
   benefits against TWIN, every path point and the nine references; the pair-view `frc.readerbudget.contrast` flag per reference;
   the seeds, the recorded seed checks and distinct-input counts; READER_SPLITS.json with assignment hashes and SCORE overlaps.

## 9. Reader-seed validity of the FRC1 grid (`forbidden_seeds`, `registered_seed`, `grid_validity`)

`READER_SEEDS_EFFECTIVE` iff NRM1's A-SC1 recipe-equality parts (ii) and (iii) hold (nrm.rules.grid_validity, called unchanged;
only its reader-seed messages are replaced) and the local seed part holds: the seeds of the required keys (S0k0, S0k1, S1k0, S1k1;
k2 / k3 added for the refit) are recorded distinct integers; every key has the form S<v>k<k> and its seed equals the registered
MODEL_SEED_BASE + 700 + 10 v + k (2026101205 + 700 + ...); none equals MODEL_SEED_BASE + {300, 400, 500, 600} + 10 v + k (k = 0..9;
80 forbidden seeds, a local equivalent of `nrm.rules.forbidden_seeds`, which reads its own constant (300, 500)). Requiring the
registered values (not just non-reuse) prevents a re-run with fresh seeds after an outcome. Claims need `grid_valid` recorded and
true (NR rule). Split tags `FRC1_devgrid_R<r>`; bootstrap seed 2026102100 + 100 v + repetition index; B = 400.

## 10. Changes requested of the protocol (apply before registration)

Adjudicated by the coordinator (ADJUDICATION_PREREG.md, 23:30Z): C1 already fixed; C2, C3, C4, C6, C8, C10, C11 adopted; C5 adopted
for PRC and superseded for HQ / SHUF by the brackets (an empty path is compared at the identity); C7 adopted with ORD_BR added; C9
adopted in its S2 form. The rules implementing them are in section 13; the text below is kept as requested at 23:24Z.

- **C1 (reader constants; exploitable).** `frc/devreaders.py` still has offset 600, tags `NRM1_devgrid_R*`, bootstrap seed
  2026102000 and forbids only 300 / 400 / 500. Set `SEED_BASE_OFFSET = 700`, tags `FRC1_devgrid_R<r>`, `BOOT_SEED = 2026102100`,
  and forbid 300 / 400 / 500 / 600 before E2. The frozen rule refuses offset-600 or unregistered seeds.
- **C2 (section 6.1, inconsistent with section 3).** T1 reads no SEX, so the "men's / women's occupation Brier" rows of the Q-PE
  report cannot come from the nested MODEL_FIT estimator. Fix: "Reported beside: ... and, from T3 only (MODEL_SELECT), the men's
  and women's occupation Brier". Keep T1 SEX-free, as in FA_PRC_v1.
- **C3 (section 6.2, circular).** Replace "at the capacity where the protection would run" with "at each capacity, separately
  (MODEL_FIT evidence only); the capacity order of 6.3 then applies QUAL to that capacity's chosen construction only".
- **C4 (section 10.1, gap).** B1 chosen + QUAL + `PE_ERROR_POINT_ONLY` has no D1 label (the text sends it to
  NO_QUALIFYING_TASK_ONLY_RELEASE, which is false). Add `QUALIFIED_B1_REPAIR_NOT_RESOLVED`; it is not a repair claim and does not
  give ENGINEERING_REPAIR.
- **C5 (section 8, undefined case).** If HQ or SHUF has no strength meeting S2, state: "the matched control at S2 is the identity
  (TWIN) under the same budget rule; the comparison is the TWIN comparison, flagged <name>_AT_IDENTITY". If PRC has none, D2-D4
  are NOT_ASSESSED_NO_FEASIBLE_STRENGTH (no fallback to another budget).
- **C6 (section 10.3-10.4, forking path).** State that D3 and D4 are assessed at the same S2-selected configuration as D2, never
  at PRC_S1 / PRC_S3, the strongest nonzero release or an ablation; those are reported beside.
- **C7 (section 10.2, frontier path).** Define the ORD path of D2 as [TWIN, the distinct saved ORD_S1..S3 at the protection
  capacity] and require E2 grid entries and paired costs of PRC_S2 against every point of it (`nrm.rules.frontier_comparison`
  returns INCOMPLETE otherwise). State that HQ_S2 and SHUF_S2 are matched on budget, not on realized spending: report their paired
  total NLL against PRC_S2 beside (not gating); the frontier check is the spending control.
- **C8 (section 10.4, undefined).** Replace "different household splits (overlap recorded) and ... changed with the reader seed
  (recorded checks)" with the definition of section 7 here (assignment hashes, overlaps in [0, 1), eight-seed grid validity on
  the minimum set).
- **C9 (section 0 / 13, disclosure).** Add: "From the published NRM1 values, A at (16, 64) meets QUAL (worst value / cap 0.100 /
  0.510 <= 0.6, the tightest Q2 ratio); T5, T6 and the stop branch are reachable only if B1 is chosen and fails QUAL."
- **C10 (section 11, repair).** Add: "a T6 repair is judged by the same construction choice and QUAL; its card cannot change QUAL,
  its increments, the original limits or the reported status of the capacity gate".
- **C11 (section 10.7).** Read "ENGINEERING_REPAIR (D1 only)" as "D1 = REPAIRED_TASK_ONLY_RELEASE and D2 not met"; name
  `D3_WITHOUT_D2` as a flag, never a class.

## 11. Not settled here

- Q-PE's 2-SE criterion: the adopted SE_used adds the between-fold spread (section 1), but 5 overlapping outer-training sets give
  a rough guard, not a valid CV variance estimate (none exists cheaply). The verifier may replay T1 but cannot remove this.
- The bracket rule for HQ_BR / SHUF_BR / ORD_BR (MODEL_FIT nested total-NLL spend) is the coordinator's; `d2` takes the saved
  points as given and does not re-derive them.
- With two versions and two registered splits, the NRM1 limitation of section 5.5 (false pass about 8% when one version or split
  dominates) applies unchanged to D2-D4.
- I did not review whether `nrm.fa.prc.select_path` aliasing (S2 = S1 selection) is recorded by `frc` code; `d2` takes the record
  as given.
- The protection capacity is selected by QUAL on MODEL_SELECT, and G1 of PRC_S2 is then read on the same people: that G1 is
  selection-dependent (a capacity that happened to look slack is more likely to be chosen). Disclosed with D2; not correctable
  without a fresh role, which is closed.

## 12. Completion record of the first freeze (superseded by sections 13-14; kept as written)

- Completed 2026-10-20T23:24:27Z (`date -u`, read before this section was written). Started 23:08:34Z.
- `python3 -m frc.test_rules`: 15 of 15 PASS, exit 0 (invented data only; about 0.04 CPU-s per run). A scratch mutation check
  (10 single-line seeded bugs of `frc/rules.py`, applied in memory only; no file changed) was caught by the suite 10 of 10 times.
- SHA-256 at completion:
  - RULES.json e1896c08200aafb7fb83a548444d8d648c2332ab3fa96531e143cc1cd6ddb73c
  - frc/rules.py fd60d7b170a475c630eb69a0760f2566823da381e5330eb2bb58bcd3efd27c76
  - frc/test_rules.py 0d4c51c08136ecd3b0be5b1b20f4ad28993706de5f7614cfc1bb0930bc264d42
  - (this RULES.md: its own hash is given in the hand-back report, computed after this line was written)
- Inherited files verified unchanged at completion: NRM1 RULES.json 84f18776..., nrm/rules.py d5df8686..., nrm/test_rules.py
  d03bbed2..., ohn/rules.py 8686a78f....
- No label was read and no real-data computation was run. Out-of-queue CPU charged through `frc.jobqueue charge` (label
  statistical_reviewer).

## 13. Change record (second pass, 2026-10-20T23:30:38Z to 23:39:58Z `date -u`; before any outcome, no label read)

On the coordinator's instruction, after reading ADJUDICATION_PREREG.md and the challenger's REVIEW_PREREGISTRATION.md, every
item adopted for the rules was implemented. No original limit changed; the reserve and the original capacity gate stay reported
only.
1. **QUAL Q2 at S2** (finding 1): increments occupation NLL 0.008, income NLL 0.004, occupation accuracy loss 0.006, income
   accuracy loss 0.004, guards 0.0030, occupation summed Brier 0.0030, income summed Brier 0.0020 (section 3). Disclosure in the
   status block: A at (16, 64) meets Q1, fails Q2 (0.00255 + 0.0030 = 0.00555 > 0.005), meets the S1 form;
   `QUALIFIED_WITHOUT_PROBABILITY_REPAIR` only at (24, 128) or (32, 256). Test: `t_qual` (the published-guard arithmetic).
2. **Q-PE SE_used** (finding 17): max(household SE, SD of the 5 outer-fold means / sqrt 5) per version; missing or malformed
   fold means give INCOMPLETE (section 1). Tests: `t_qpe_fold_se`.
3. **D2 conjuncts** (findings 3, 5, 9; C5, C7): (a) G1; (b) TWIN comparison; (c) frontier dominance against the SHUF, HQ and ORD
   control paths with brackets; (d) F4 (`NOT_MET_F4`, not a stop); (e) reader targeting (`READER_TARGETED`);
   `NOT_ASSESSED_EXACT_ALIAS`; partial labels kept, with comparisons renamed TWIN, SHUF_FRONTIER, HQ_FRONTIER, ORD_FRONTIER
   (section 6). Tests: `t_d2_conjuncts_partials_and_aliases`.
4. **D1 `POST_HOC_REPAIR_QUALIFIED`** for a T6 repair, never `REPAIRED_TASK_ONLY_RELEASE`; the repair card must precede any
   MODEL_SELECT read of its outputs (finding 11). Tests: `t_d1_labels`, `t_decisions_and_outcome_class`.
5. **D3 reader-budget rule** (finding 7). Superseded by item 7 below.
6. **Capacity record** of the other construction, recorded and never used (finding 10). Test: `t_capacity_cascade`.
- Unchanged: Q-PE label ladder, construction choice, capacity order and triggers, the original gate, reproduction, strongest
  nonzero release, refit effectiveness, reader seeds, D4-D6, the outcome class ladder.
- RULES.json carries the same changes (`qual`, `qpe.se_used`, `qpe.outer_folds`, `decisions.D1/D2/D3`, `capacities`,
  `change_record`); `load_rules()` now also checks the D2 comparison names and the fold count.

**Third pass (2026-10-20T23:42:15Z to 23:43:30Z `date -u`; coordinator correction, before any outcome, no label read).**
7. **Reference reader-budget rule corrected** (`reference_credit`, `_d3`, `_d4`): a reference flagged `READER_BUDGET_ASYMMETRIC`
   is credited iff the non-table winner's point benefit against it is > 0 in both version means; the nested MODEL_FIT LOOKUP
   condition no longer applies to references (that input cannot exist for them, so requiring it would have made every flagged
   reference "not beaten" automatically); missing `nolookup` inputs make the reference INCOMPLETE; the same rule now applies to D4
   (NEURAL_RAW, H_LJ). Applied in three-valued form (section 6), because the rule only withdraws credit: a reference that is
   INCOMPLETE never becomes the reason for NOT_SUPERIOR / NOT_MET, but a label that already fails on its own results stays a
   failure. Test: `t_d3_d4_reader_budget_asymmetry`. RULES.json: `decisions.reader_budget_rule_references`, `D3`, `D4`,
   `change_record`. Nothing else changed.

## 14. Completion record (second pass, updated by the third pass)

- Second pass completed 2026-10-20T23:39:58Z (RULES.json 09834b7e..., frc/rules.py 4a02ea95..., frc/test_rules.py 248ff240...,
  17 of 17 PASS). Third pass completed 2026-10-20T23:43:52Z (`date -u`, read before this line was written).
- `python3 -m frc.test_rules`: 17 of 17 PASS, exit 0 (invented data only; about 0.1 CPU-s). The scratch in-memory mutation check
  (20 single-line seeded bugs, including F4 in one version only, reader targeting and the fold SE ignored, the POST_HOC label
  dropped, the reference rule ignored for D3 or D4, a missing reference input counted as not beaten, the nested LOOKUP condition
  required for references, aliases not deduplicated) was caught 20 of 20 times; no file was changed by it.
- SHA-256 at completion (third pass):
  - RULES.json a0dc0fc7f377c26339df0c96e6abda9784d725ccab8cb204e1cfa7d789029510
  - frc/rules.py effcc20c2f903aab3d1a6ec7b69d3eaaedd48d56f043fa64f2b6635c20b33d3a
  - frc/test_rules.py 4cdcb9bb3b1be5703e65e49c135b9917c7f82207d866198f3c461c444acbb68d
  - (this RULES.md: its own hash is given in the hand-back report)
- Inherited files unchanged: NRM1 RULES.json 84f18776..., nrm/rules.py d5df8686..., nrm/test_rules.py d03bbed2..., ohn/rules.py
  8686a78f....
- No label was read and no real-data computation was run.
