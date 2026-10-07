# Baseline fairness review (lra)

**Owner.** Role F (claims and custody reviewer), 2026-10-07.

**Scope.**
- Prompt §7 (budgets and arms), §8 (search rules, starts, matched control bank, equal ceilings) and §10 (nomination
  pools), checked against the executable code before SCIENCE_LOCK.
- The review covers the port and then the edits by roles A, C and D.
- Findings went to the lead, who owns every locked file. This file changes nothing that it reviews.

**Classes.**
- **REQUIRED**: blocks SCIENCE_LOCK until it is resolved, or until it is registered as an explicit, disclosed deviation.
- **RECOMMENDED**: should be fixed before the lock, but does not block it.
- **NOTE**: recorded only; no change needed.

## 1. Inputs reviewed

| Round | File | Owner | sha256 at review (prefix) | State |
|---|---|---|---|---|
| 1 (04:12–04:25Z) | lra/select.py | D | c91fda94ed3f | port (HEAD df90e6a) |
| 1 | lra/mapper.py | C | 8e496f597b56 | port |
| 1 | lra/run.py, lra/eval_lock.py, lra/infer.py, lra/lock.py | A / D | HEAD df90e6a | port |
| 2 (04:40–04:50Z) | lra/select.py | D | fcee1ee0e4b2 | working tree (D's F01–F14 edits, uncommitted) |
| 2 | lra/mapper.py | C | b69347ab8366 | working tree (C's edits incl. R-4, uncommitted) |
| 2 | lra/run.py | A | a73dc1e83105 | HEAD 89f2206 (R-1, R-3 adopted at ab81a97) |
| 2 | lra/eval_lock.py / lra/infer.py / lra/family.py | D | f3f87eb21a98 / fa48ad0c741b / 8fa49f3a464c | working tree |
| 2 | lra/audit.py | D | aea8a73aef4b | working tree (84-code composition bank) |
| 2 | PKG/SELECTION_RULES.json | D | 0a26f2bd6039 | working tree |
| 2 | PKG/SEARCH_RULES.json | C | 23449678e1e1 | in flux; compared with `lra.mapper.rules()` (§3, R-7) |
| 2 | PKG/FIT_MANIFEST.json | A | d1be9c52fce2 | HEAD |
| 2 | PKG/PROTOCOL.md | A | 8db501781189 | working tree |

The SCIENCE_LOCK hashes of these files supersede this table. Round 3 (§5) re-checks the files actually locked.

## 2. Checklist

| # | Check | Rule | Status |
|---|---|---|---|
| S1 | T\* is the closed privacy-untrained list: C-TASK, D0/D1 DIRECT-TASK and FINE-TASK, U, **CLASS-ONLY with both decoders**, F0; RAW-J excluded | §10 | **OK after R-1.** `select.T_STAR_CLOSED` holds 9 ids, including U\|CLASS\|i1o1\|D1; none is privacy-trained |
| S2 | P\* is drawn from every declared private arm (D0 privacy, D1 fixed-map privacy, weighted, constrained) and guarded against T\* | §10 | **OK.** `private_code` gives 77 code arms. Privacy-trained references are truthfully labelled (`privacy_trained`) but are not in the code-only pool (finding F05). Diagnostic d0s ids are never candidates |
| S3 | C\* is drawn from the D0/D1 old privacy maps and the weighted controls, with no constrained arm; ordinary comparator | §10 | **OK** (72 candidates) |
| S4 | N\* is drawn from the constrained arms and guarded against both T\* and C\* | §10 | **OK.** A constrained fit must be FEASIBLE on every seed. A missing or corrupt fit record is now TECHNICAL (FIT_RECORD_TECHNICAL_FAILURE), never CONSTRAINED_FIT_INFEASIBLE (R-5, resolved in D's working tree) |
| S5 | C_pair\* is every eligible release other than K-JOINT-PAIR | §10 | **OK, conservative.** The pool is every scored id except K-JOINT-PAIR and fit-infeasible constrained units. It includes privacy-untrained releases (CLASS with both decoders) and the continuous references, so claim C faces the strongest comparator. Consequence registered in PRIOR_ART §5.8 and PREDECESSOR_GATE_DIAGNOSIS §6 |
| S6 | J\* is K-JOINT-PAIR only, guarded against both T\* and C_pair\*, and fit-feasible | §10 | **OK** |
| S7 | Fixed-map D1 privacy releases keep the privacy-trained label | §10 | **OK.** `parse_id` uses the base family, and the display field `training` is kept. CLASS\|D1 is privacy-untrained (base CLASS) |
| S8 | Weighted controls face the unchanged inner utility rules and guards, and no fitting budgets | §8, §10 | **OK.** Same `gate_record` eligibility on every seed and task; same T\* guard in P\*. `SPEC[W-*]["constrained"]` is False and there are no local caps |
| S9 | Original inner eligibility on EVERY seed, with no headroom buffer; one configuration across seeds | §10 | **OK** |
| S10 | Ordering keys and tie rules are frozen | §10 | **OK.** Token states are a finite positive integer for codes and null for continuous releases (F06); continuous releases are +∞ for ordering only |
| S11 | Fallbacks are deterministic, take the minimum shortfall and are classified (utility / guard / missing comparator / technical) | §10 | **OK** (`fallback_class`). A missing guard comparator with an eligible candidate keeps a fixed DESCRIPTIVE_ONLY fallback (F04) |
| S12 | Aliases are computed for every role, comparators and Q included, on permitted rows only; one real representative is named; identical_to_untrained is disclosed | §10 | **OK** (F02/F03/F07/F09/F11; `permitted_rows` excludes assessment and HEAD_VALIDATION rows) |
| S13 | The scored list follows §13 | §13 | **OK, with R-10.** It covers the roles or fallbacks, U, decisions alone, the best D1 fixed-map control and its D0, the best weighted control, C-TASK, the 5 constrained arms, RAW-J, FARE, F0, LEACE, and the same-map D0 of P\* and of C-TASK |
| M1 | B controls are the D1 versions of the EXACT admitted D0 maps, with assignments unchanged | §8 | **OK.** `stage_d1` fits 27 ids per seed (the prompt's 26 plus CLASS\|D1, R-1) × 3 = 81 units on OSF_DEFENSE_FIT labels, and asserts tokens bitwise equal to the D0 release |
| M2 | D controls are complete: 4 families × 6 λ × 3 seeds = 72 | §8 | **OK.** `fit_chains` / `new_fit_ids`; new mapping fits are 72 + 15 + 3 = 90 (FIT_MANIFEST counts) |
| M3 | D controls use the same decoder, starts, caps and single-move neighbourhood as the new arm, without the new budgets | §8 | **OK.** W-LOCAL weights are (1, ½, λ/2, 0) = T + λ(I1+I2)/2; W-SEQ and W-JOINT weights are (1, ½, λ/2, λ) = T + λΦ. Same D1 `State`, caps 8/64, 4 nearest + 4 best (each arm's own objective, N-6), same sweeps and the same 8e6 ceiling. `rules()["weighted_controls"]` registers the correspondence W-LOCAL↔K-LOCAL, W-SEQ-ab↔K-SEQ-ab, W-JOINT↔K-JOINT-SINGLE |
| M4 | JOINT-SINGLE, SEQ-12 and SEQ-21 get the same TOTAL proposal-evaluation ceiling as JOINT-PAIR; actual work is recorded | §8 | **OK after R-4.** EVAL_CEILING is 8,000,000 for every arm. Joint: 8e6/#witnesses per witness. SEQ: 8e6/(2·#starts) per stage. LOCAL: 8e6/(2·#starts) per recipient. The pair step's pool screening and pair evaluations now both stop at the share (granularity ≤ one cell's targets). Actual evaluations, share and stop reason are recorded per stage and start |
| M5 | The temporary sequential partner is not required to be feasible and is not a constant | §7 | **OK.** Stage 1 enforces only the first recipient's constraints, with the partner at CLASS-ONLY. Stage 2 enforces the second recipient's with the first frozen, which is equivalent to both, since a frozen recipient's budgets and cap depend only on its own map. The FINAL release is checked against both |
| M6 | C-TASK starts from no privacy-trained map, keeps the FINE-TASK start and reports its D1 initialisation, refined map and D0 references separately | §7, §8 | **OK, with R-9 open.** The only start is D0 FINE-TASK, and SEX cannot reach the search (`sex=False`). `ctask_report.d0_external_references` is still null (lcr N-8) |
| M7 | Starts and witnesses are matched; no infeasible witness is eligible by its name | §8 | **OK and matched.** K/W-LOCAL start from {C-TASK}. K/W-SEQ-ab start from {C-TASK, the six D0 SEQ-ab maps}. K-JOINT witnesses are {C-TASK, K-LOCAL, K-SEQ-12, K-SEQ-21, D0 FINE-TASK, the 24 D0 privacy maps}; W-JOINT(λ) uses the same set with W-\*(λ). Infeasible witnesses are EXCLUDED_INFEASIBLE_WITNESS. D0 DIRECT-TASK and CLASS are not witnesses (N-5) |
| M8 | Pair bank: 8 own-budget proposals per recipient (non-improving included); 4 by Φ and 4 by task loss; atomic product; only a feasible strict Φ improvement accepted | §8 | **OK** |
| M9 | Local caps I_i ≤ I_i(C-TASK) of the same seed, recorded exactly before privacy search | §7 | **OK** (`refs_from_ctask`) |
| M10 | Decoder solves cached only by exact sufficient statistics | §8 | **OK** |
| M11 | Starts and accepted moves are persisted and replayable | §8 | **OK in design** (trace.json per new__ unit; replay in `lra.mapper`; role E replays independently). Custody covers the traces (lra.closeout unit inventory) |
| X1 | No control is omitted for being strong | §8 | **OK.** Banks A (81 D0), B (81 D1 incl. CLASS\|D1), C (3), D (72), E (15) and F (U, decisions alone, RAW-J, FARE, F0, LEACE) are all in the manifest, scored list or reference set. None is dropped by any result-dependent rule |
| X2 | CLASS stays in the T\* pool | §10 | **OK** (R-1). The decision floor consequence is registered (PROTOCOL §6; PRIOR_ART §5.8) |
| X3 | P\* may be a control | §10 | **OK.** P\*'s pool includes the D0 and D1 fixed maps and the weighted controls. If one wins, it is delivered and named by its real construction (PRIOR_ART §6.1) |

## 3. Findings register

| ID | Class | Finding | Status |
|---|---|---|---|
| R-1 | REQUIRED | CLASS\|D1 was excluded from the T\* pool and never audited | **RESOLVED** (A at ab81a97; D's T_STAR_CLOSED, audit and composition bank) |
| R-2 | REQUIRED | `eval_lock.technical_validity` required FIXTURE_GATE.json GATE_MET; `infer` hard-coded `gate_met=True` | **RESOLVED in D's working tree** (engineering_ready / engineering_gate(EL)); re-check in round 3 |
| R-3 | REQUIRED | The CBP_LOCAL_ONLY environment variable skipped every pushed-lock check | **RESOLVED** (A at ab81a97) |
| R-4 | RECOMMENDED | The pair-step screening ignored the ceiling, so one step could overshoot a share by one screening pass | **RESOLVED in C's working tree** (screening and pairs stop at the share; `rules()["pair_step"]["ceiling"]`) |
| R-5 | REQUIRED | `fit_feasible` mapped an unreadable fit record to CONSTRAINED_FIT_INFEASIBLE (source finding 1) | **RESOLVED in D's working tree** (FIT_RECORD_TECHNICAL_FAILURE) |
| R-6 | REQUIRED | Aliases were computed only for P\*, N\*, J\* (findings 7/11) | **RESOLVED in D's working tree** (every role, comparators and Q) |
| R-7 | REQUIRED | SEARCH_RULES.json is stale against `lra.mapper.rules()` | **OPEN** (C/A): regenerate and verify before SCIENCE_LOCK |
| R-8 | RECOMMENDED | PROTOCOL text and code disagree in three places | **OPEN** (A) |
| R-9 | RECOMMENDED | C-TASK's D0 external references are null (lcr N-8) | **OPEN** (A/C) |
| R-10 | RECOMMENDED | Score CLASS\|D1 on the assessment beside decisions alone | **OPEN** (D/A) |
| R-11 | RECOMMENDED | Witness lineage work is outside the joint arms' ceiling | **OPEN** (C, reporting only) |
| R-12 | RECOMMENDED | FIT_MANIFEST should list the d0s__ same-map D0 diagnostic units | **OPEN** (A) |
| N-5 | NOTE | D0 DIRECT-TASK and CLASS are not JOINT witnesses | Recorded |
| N-6 | NOTE | Each neighbourhood's "best 4" uses the arm's own objective | Recorded |
| N-9 | NOTE | Constrained arms have no restoration phase | Recorded (PROTOCOL §7) |
| N-10 | NOTE | Acceptance is per cell within a sweep | Recorded |

### R-1 REQUIRED: CLASS|D1 was outside the T\* pool. RESOLVED

**The problem.** The port registered CLASS|D1 as a "diagnostic (not a candidate, not in code_ids, never audited or
nominated)". Prompt §10 says: "Do not remove CLASS from the comparator pool to manufacture a win. On Adult it may or may
not satisfy the full utility contract after D1; measure that under the new registration. If it qualifies and is
strongest, the privacy claim still compares against it." §16 also asks for CLASS's measured attacks.

**Resolution.**
- CLASS|D1 is a fixed-map D1 code (dec__s{k}__U_CLASS_i1o1_D1), audited, in the 84-code composition bank and in
  T_STAR_CLOSED.
- Counts are reconciled as a registered addition: 81 D1 units, 84 codes per seed, 89 configurations, 267 inner units
  (FIT_MANIFEST, SELECTION_RULES, PROTOCOL §6).
- The decision-floor consequence for claims A and C is registered.

### R-2 REQUIRED: the old gate wiring. RESOLVED in the working tree

**The problem.** `eval_lock.technical_validity` required FIXTURE_GATE.json = GATE_MET. lra has only the historical
SOURCE_FIXTURE_GATE.json (GATE_NOT_MET), so the evaluation lock would have refused forever. `infer.label_from` passed
`gate_met=True`, hard-coded readiness (finding 10).

**Resolution.**
- `technical_validity` calls `lra.run.engineering_ready()` and records the verdict and its sha256.
- `infer` reads the lock-bound verdict through `engineering_gate(EL)` and refuses unless it is ENGINEERING_READY.
- `family.overall_label` takes the verdict as a required argument.

### R-3 REQUIRED: a local-only push bypass. RESOLVED

The CBP_LOCAL_ONLY environment variable skipped every "pushed" check, both in `lock.verify_lock` and in
`run.engineering_ready`. Now `verify_lock` refuses any stage while it is set, `engineering_ready` always requires
origin, and the tests monkeypatch `on_origin`.

### R-4 RECOMMENDED: the pair-step ceiling. RESOLVED in the working tree

**The problem.** In the port, `_pair_step` screened every fine cell of both recipients (about 48.8K exact evaluations at
full caps) without checking the remaining share. It then let pair evaluations use a budget computed before screening. A
step that began just under the share could overshoot it by about one screening pass, roughly 18% of 8e6/29.

**Resolution.**
- Screening and pair evaluations both stop at the share.
- A step that reaches it ends the stage at `eval_ceiling`.
- The rule is registered in `rules()["pair_step"]["ceiling"]` and `["equal_work"]["granularity"]`.

### R-5 / R-6 REQUIRED: source findings 1 and 7/11. RESOLVED in the working tree

See S4 and S12. D's REVIEW_FINDINGS_DISPOSITION.json is the authoritative record.

### R-7 REQUIRED: SEARCH_RULES.json is stale

**The problem.** At 04:44Z, `lra.mapper.rules()` and SEARCH_RULES.json differed in `equal_work`, `module`, `pair_step`,
`receipts`, `replay`, `schema`, `starts_notes`, `trace` and `weighted_controls`. The file's `rules_sha256` was still
lcr's (ec0fa66c…). This was checked under lra.sema as F:rules-compare.

**Fix.** Regenerate with `python -m lra.mapper rules --out …/SEARCH_RULES.json` from the final mapper.py, confirm
equality as parsed JSON, and lock both.

### R-8 RECOMMENDED: PROTOCOL text against code

A code/text disagreement is a defect (PROTOCOL preamble).
- **§11** says U composes through "all 83 codes per seed". The code and AUDIT_COMPUTE use 84 (CLASS|D1 is in the bank).
- **§10** defines C_pair\* as "every release except K-JOINT-PAIR". The code also excludes fit-infeasible constrained
  units (SELECTION_RULES `constrained_fit_feasibility`). Add "and except fit-infeasible constrained units".
- **§10** lists T\* with "CLASS-ONLY". Write "CLASS-ONLY (D0 and D1)", as SELECTION_RULES does.
- Cosmetic: the `run.code_ids()` docstring still says "D1 fixed-map (26) … = 83".

### R-9 RECOMMENDED: C-TASK's D0 external references (lcr N-8, carried)

`stage_ctask` passes no `refs`, so `ctask_report.d0_external_references` is null. Prompt §7 asks for the D1
initialisation, the refined map and the D0 external references to be reported separately.

The D0 DIRECT-TASK and FINE-TASK fitting losses already exist in the dec__ records (`fitting.L_D0` / `B_D0`). Report them
beside C-TASK in BUDGET_FEASIBILITY.csv, or pass them as `refs["D0"]`. This is reporting only; no fit changes.

### R-10 RECOMMENDED: score CLASS|D1 on the assessment

The scored list contains "decisions alone" as D0 CLASS only. CLASS|D1 is scored only if it becomes T\*.
- DECISION_FLOOR_AND_FEASIBILITY.csv reports CLASS feasibility "and its measured attacks".
- The CLASS pair (D0, D1) is the cleanest same-token decoder ablation.

Adding U|CLASS|i1o1|D1 as a fixed, pre-registered scored diagnostic, before EVALUATION_LOCK, is cheap: its token family
aliases D0 CLASS. It is not a winner-selection device.

### R-11 RECOMMENDED: report witness-lineage work beside claim C

**What happens.** The JOINT arms start from K-LOCAL and K-SEQ outputs that were fitted under their own 8e6 ceilings.
JOINT-SINGLE and JOINT-PAIR inherit the same witnesses, so the SINGLE-versus-PAIR comparison is matched.

**The gap.** If C_pair\* is a sequential arm, J\* had that arm's map as a start plus its own share.

**Fix.** Report per unit both its own evaluations and its witnesses' evaluations (cumulative lineage) in
OPTIMIZATION_RECEIPTS. Name this in any claim-C sentence.

### R-12 RECOMMENDED: d0s__ diagnostic units in FIT_MANIFEST

The same-map D0 decodes of C-TASK (3 units) and of P\* (data-dependent: none if P\* is a D0 map or a fixed-map D1 whose
paired D0 is an admitted pol__ unit) are written by `lra.select.stage_d0same`. They are diagnostic releases, not fits.
List them, with the rule that determines their number, so ACTUAL_WORK_ACCOUNTING reconciles.

### Notes carried from lcr (unchanged)

- **N-5.** D0 DIRECT-TASK lies outside the fine-cell family (prompt §5), and CLASS-ONLY is the sequential stage-1
  partner, not a witness. Both stay T\* candidates.
- **N-6.** "Best 4" ranks by each arm's own objective. That is the stronger choice for each weighted control, so it does
  not weaken the controls.
- **N-9.** If C-TASK violates a fitting budget on a seed, K-LOCAL is infeasible there by construction. Report it; it is
  not a technical failure.
- **N-10.** Within a sweep, each cell's best improving feasible move is applied in a fixed order. METHOD_CARD says so.

## 4. Fairness conclusions before SCIENCE_LOCK

- **Every control the prompt requires is present, and none is omitted for being strong** (X1):
  - D0 legacy (bank A);
  - D1 fixed-map calibration controls on the exact admitted maps (bank B, privacy-trained where the map was);
  - C-TASK;
  - the full 72-unit weighted bank;
  - the five constrained arms;
  - the references.
- **The new arms get no structural advantage.** Weighted and constrained arms share the decoder, starts, witnesses, caps,
  neighbourhood, sweeps and the per-unit ceiling. The paired arm's extra neighbourhood is paid for inside the same total
  ceiling (R-4).
- **Comparators are at least as strong as the prompt requires.** T\* keeps CLASS with both decoders. C_pair\* includes
  untrained and reference releases. P\* may be any control.
- **Remaining blockers from the fairness side:**
  - R-7 (regenerate SEARCH_RULES.json);
  - confirming R-2, R-4, R-5 and R-6 in the files actually locked.
- R-8 to R-12 are recommended.

## 5. Round 3 (re-check of the locked files)

To be filled at SCIENCE_LOCK with the locked hashes of mapper.py, select.py, run.py, eval_lock.py, infer.py,
SEARCH_RULES.json and SELECTION_RULES.json.
