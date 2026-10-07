# Baseline fairness review (lcr)

**Owner.** Role F (claims and custody reviewer).

**Scope.** Prompt §8 (search rules and matched controls) and §10 (inner nomination), checked against the frozen text and
the executable code before SCIENCE_LOCK. Findings went to the lead, who owns every locked file. This file changes nothing
that it reviews.

**Classes.**
- **REQUIRED**: blocks SCIENCE_LOCK until it is resolved, or until it is registered as an explicit, disclosed deviation.
- **RECOMMENDED**: should be fixed before the lock, but does not block it.
- **NOTE**: recorded only; no change needed.

## 1. Inputs reviewed

| Round | File | Owner | sha256 at review |
|---|---|---|---|
| 1 (00:15–00:20Z) | PKG/SELECTION_RULES.json | A | 5db911ef520a91f55ff6c73b19a594bacad0fbea286e98b4bc862865288d890f |
| 1 | lcr/select.py | A | 780e417edbc11274ee42cc7c26acf923719f5a9471a9f756a063514b28ab26f4 |
| 1 | lcr/family.py | A | 3511d38cc55f2a2f64a9256d465804b184ce0378e1bfc1bcc84f50d87e4938ca |
| 1 | PKG/LABEL_TRUTH_TABLE.json | A | e4d20999a69215d8abad7110d0b4013a2304edad889388f4b0c7aafe4db07ab6 |
| 1 | PKG/PROTOCOL.md | A | 393a01b82fc69b1f14bd2277eb8d1cbeee0bd090fef84c3db742a680aca616bd |
| 1 | lcr/eval_lock.py | A | 2668a1bcb5ea08edf761cc581f582d849851fa34e80b5a844efc164c248b2be8 |
| 1–2 | lcr/run.py (stage_d1, stage_ctask, _fit_args, fit_chains) | A | 5d673851d90208abc5b8faf36f05e736ae46dc3bf1fbdb6bd4cdfafbf1d0dbcb |
| 2 (00:20–00:25Z) | lcr/mapper.py, including `rules()` (SEARCH_RULES.json had not been written yet) | C | d384e5256ef53bc10b51f4a3438ad0317418497332c2e79dc77db8fdcf77c1fc. A later edit (b3bfd756…834e) leaves every frozen constant checked here unchanged. |
| 2 (re-check of round-1 patches) | lcr/select.py | A | 22902068ce01de02c36d9e668cde08f7fef61698031fd0ebade4a767da9e41c0 |
| 2 | lcr/family.py | A | d54a03b381b6a04dcdacb2ecdea459b4188d6d49266e130f997069bd8986bb1b |
| 2 | PKG/LABEL_TRUTH_TABLE.json | A | 30afa11c877e71f4ea71b65277c94127bdca28f98fa82b802fd4cec9be386409 |
| 2 | PKG/SELECTION_RULES.json | A | 397f7a75e597aebe480520635692fbd50371c7f5b732bc3d2e05bbf8cd8fae87 |
| 3 (00:26Z, R-4 fix at e227909) | lcr/select.py | A | fc33af63f6a5225b50791de381900b424879536096fcec99de05f5c5e9ba2816 |
| 3 | PKG/SELECTION_RULES.json | A | e1ad5bb5b014d7d1a33fe367dec2b364ddf7e2d00159fd9777ad01d705f1a1ee |
| 3 | PKG/LABEL_TRUTH_TABLE.json | A | e9024fe165847489e16b129ff9bdeb5534ded38fee2eaa6d973bccfabcc8a1fe |
| 3 | lcr/family.py | A | 97296e8bbd7c4523ef331c2a95458eff9bd35eacc3613ab1524851263d79fb15 |
| 4 (00:40Z) | PKG/SEARCH_RULES.json | C | 5da202e4daffa9cec5185be2a59c85a3c1705f3a7e975ba100ca3c5e723b1666. It equals `mapper.rules()` plus `rules_sha256` ec0fa66cc420759c8e789037e2de1fbd22bb68a9aed379217ca166b32ff0b44c, compared as parsed JSON |
| 4 | lcr/mapper.py | C | b3bfd7567edc1b73ed8d2c3fa121d322266c01602c262b0255c49c226403834e. Against the round-2 reading, the changes are: synthetic timing helpers added, the unused `_labels_from` removed, and one redundant slice removed in `_cell`. The search semantics are unchanged |
| 4 (00:41Z) | PKG/FIT_MANIFEST.json | A | 0420e92cefab24db91e7b014595dac67dd991a5c3d4031564cd8b14662ca6ed0. 249 code units = `run.code_ids()` × 3 seeds exactly: 81 admitted D0, 78 D1 fixed-map decodes, 3 C-TASK, 72 weighted and 15 constrained, so new mapping-pair fits = 90. Every unit name equals `run.unit_for`. 264 inner audits = `run.scored_ids()` × 3 exactly (249 codes, 6 sources, 9 references). Aliases are recorded after fitting by hash identity and never add fits. There are no auxiliary fine-state jobs (the label-blind fine partitions are admitted). The real-data control jobs are listed. **OK** |
| 5 (00:44Z, R-5 fix at a3d16bd) | lcr/select.py; PKG/SELECTION_RULES.json | A | e672c15b8946e46c9739cf0ccdbb7865ae1c44f8bb0f412ea367e2b957623eba; 160e4a5acb96ce047d7744b668e803a61c82d5ff775157e74b0e03da91ae115b |

## 2. Checklist (prompt §7, §8, §10)

| # | Check | Rule | Status |
|---|---|---|---|
| S1 | T\* is the closed privacy-untrained list (C-TASK, D0/D1 DIRECT-TASK and FINE-TASK, U, CLASS-ONLY, F0), with RAW-J excluded | §10 | **OK**. `select.T_STAR_CLOSED` holds 8 IDs, and none is privacy-trained |
| S2 | P\* is drawn from all declared private arms, the calibrated old maps and weighted controls included, and is guarded against T\* | §10 | **OK**. There are 77 code arms. Privacy-trained references are not candidates, and this is now stated in SELECTION_RULES (N-2) |
| S3 | C\* is drawn from the D0/D1 old maps and the weighted controls, with no constrained arm, and is an ordinary comparator | §10 | **OK** (72 candidates) |
| S4 | N\* is drawn from the constrained arms and is guarded against both T\* and C\* | §10 | **OK**. A constrained fit must be fit-feasible on every seed (R-4, resolved) |
| S5 | C_pair\* is every eligible release other than K-JOINT-PAIR | §10 | **OK**: every scored ID except K-JOINT-PAIR and except infeasible constrained fits (R-4, resolved; the lead chose exclusion) |
| S6 | J\* is K-JOINT-PAIR only, guarded against both T\* and C_pair\* | §10 | **OK**. It must also be fit-feasible (R-4, resolved) |
| S7 | Fixed-map D1 privacy releases keep the privacy-trained label | §10 | **OK**. `parse_id` uses the base family. The display field `training` has been added (R-2, resolved) |
| S8 | Weighted controls meet the unchanged inner utility rules and guards | §8, §10 | **OK**. They use the same `gate_record` eligibility on every seed and task, and the same T\* guard in P\* |
| S9 | Original inner eligibility is applied on EVERY seed, with no headroom buffer | §10 | **OK**. The stale cbp reason code has been removed (N-3, resolved) |
| S10 | Ordering keys and tie rules are frozen | §10 | **OK** |
| S11 | Fallbacks are deterministic, take the minimum shortfall, and are classified | §10 | **OK**. CONSTRAINED_FIT_INFEASIBLE has been added as a selection reason. Fit-feasible candidates rank first in both fallback branches (R-5, resolved) |
| S12 | The scored list follows §13 | §13 | **OK** (`eval_lock.scored_labels`) |
| S13 | The overall-label truth table follows §12 | §12 | **Resolved (R-1)**: the prompt-literal order was adopted in code, the truth table and the tests |
| M1 | B controls are the D1 versions of the EXACT fixed D0 maps (DIRECT, FINE, the full six-weight privacy bank), with assignments unchanged | §8 | **OK**. `run.stage_d1` fits 26 IDs per seed × 3 = 78 units. It uses the admitted D0 policy.json and `decode_policy` on OSF_DEFENSE_FIT labels, and asserts that tokens are bitwise equal to the D0 release |
| M2 | D controls are complete: 4 families × 6 λ × 3 seeds = 72 | §8 | **OK** (`run.fit_chains`). The total of new fits is 72 + 15 constrained + 3 C-TASK = 90 |
| M3 | D controls use the same decoder, starts, caps and single-move neighbourhood as the new arm, without the new budgets | §8 | **OK**. Weights are W-LOCAL (1, ½, λ/2, 0) = T + λ(I1+I2)/2 and W-SEQ/W-JOINT (1, ½, λ/2, λ) = T + λΦ. The same D1 `State`, CAPS 8/64 and neighbourhood construction apply (N-6). There are no fitting budgets and no local caps (`capI` None) |
| M4 | JOINT-SINGLE, SEQ-12 and SEQ-21 get the same TOTAL proposal-evaluation ceiling as JOINT-PAIR, and actual work is recorded | §8 | **OK**. `EVAL_CEILING` = 8,000,000 for every arm. One evaluation is one exact (cell, target) change or one atomic pair, and pair-pool screening counts. Joint arms get ceiling/#witnesses; SEQ arms get ceiling/(2·#starts) per stage. Evaluations, solves, memo hits and CPU are recorded per start. The ceiling probably binds only JOINT-PAIR (N-7) |
| M5 | The temporary sequential partner is not required to be feasible and is not a constant | §7 | **OK**. Stage 1 enforces only the first recipient's constraints, with the partner at CLASS-ONLY (`partner_constraints_enforced: false`). Stage 2 enforces the second recipient's constraints with the first frozen. That is equivalent to enforcing both, because a frozen recipient's budgets and cap depend only on its own map. The FINAL release is checked against both |
| M6 | C-TASK starts from no privacy-trained map, keeps the FINE-TASK start, and reports its D1 initialisation, refined map and D0 references separately | §7, §8 | **OK**. The only start is D0 FINE-TASK. SEX cannot reach the search, because `Problem` is built with `s=None` and `State.sex` is False; SEX is used only afterwards to record I_i(C-TASK). The `ctask_report` covers the D1 initialisation, the refinement and the D0 decoder on the same maps; the external D0 references are null (N-8) |
| M7 | Starts and witnesses are matched; no infeasible witness is eligible by name | §8 | **OK and matched** (`run._fit_args` and `mapper.registered_starts`). K/W-LOCAL start from {C-TASK}. K/W-SEQ-ab start from {C-TASK, the six D0 SEQ-ab maps}. K-JOINT witnesses are {C-TASK, K-LOCAL, K-SEQ-12, K-SEQ-21, D0 FINE, the 24 D0 privacy maps}; W-JOINT(λ) witnesses are the same set with W-\*(λ). Infeasible witnesses are recorded as EXCLUDED_INFEASIBLE_WITNESS. Feasible unchanged witnesses are candidates. D0 DIRECT-TASK is not a witness (N-5) |
| M8 | Pair bank: 8 own-budget proposals per recipient, non-improving ones included; 4 by Φ and 4 by task loss; atomic Cartesian product; acceptance only of a feasible strict Φ improvement | §8 | **OK**. The pool is every own-feasible neighbourhood candidate. 4 proposals are kept by one-sided ΔΦ (ΔI12 + ½ΔI_i) and 4 more by Δ(L_i + ½B_i), with canonical ties. The 8 × 8 product is evaluated with both moves applied, both decoders recomputed and exact feasibility checked. The step runs after each single sweep |
| M9 | Local caps I_i ≤ I_i(C-TASK) of the same seed are recorded exactly before the privacy search | §7 | **OK**. `refs_from_ctask` supplies them, and the passed C-TASK map's exact MI is checked bitwise against them and recorded |
| M10 | Decoder solves are cached only by exact sufficient statistics | §8 | **OK**. A memo entry is reused only if (n, y, s) are bitwise equal, and every applied move is re-folded and compared bitwise |

## 3. Findings register

| ID | Class | Finding | Status |
|---|---|---|---|
| R-1 | REQUIRED | The overall-label precedence disagreed with prompt §12 | **RESOLVED** (lead, round 1) |
| R-4 | REQUIRED | An INFEASIBLE constrained fit can be nominated | **RESOLVED** (commit e227909) |
| R-5 | RECOMMENDED | Descriptive fallbacks do not rank fit-feasible constrained units first | **RESOLVED** (commit a3d16bd) |
| R-2 | RECOMMENDED | Privacy-trained references were displayed as privacy_trained = False | **RESOLVED** |
| R-3 | RECOMMENDED | The basis for the §17 decoder sentence was not registered | **RESOLVED** |
| N-1 | NOTE | Label when B passes without A | Recorded |
| N-2 | NOTE | Continuous privacy-trained references are not P\* candidates | **RESOLVED** (note added) |
| N-3 | NOTE | Stale HEADROOM_SELECTION_FAILURE reason code | **RESOLVED** |
| N-4 | NOTE | There is no D1 CLASS-ONLY control | Recorded |
| N-5 | NOTE | D0 DIRECT-TASK is not a JOINT witness | Recorded |
| N-6 | NOTE | Neighbourhood "best 4" uses each arm's own objective | Recorded (request (c) withdrawn) |
| N-7 | NOTE / RECOMMENDED | The equal ceiling probably binds only JOINT-PAIR | Open |
| N-8 | NOTE / RECOMMENDED | The C-TASK record has null D0 external references | Open |
| N-9 | NOTE | Constrained arms have no restoration phase | Recorded (feeds R-4) |
| N-10 | NOTE | Acceptance is per cell within a sweep | Recorded |

### R-1 REQUIRED: the overall-label precedence disagreed with prompt §12. RESOLVED

**The disagreement.**
- Prompt §12 says: "No method claim passes, Q passes: CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION".
- `lcr/family.overall_label` and LABEL_TRUTH_TABLE gave INCOMPLETE_OR_INVALID precedence over a passing Q.
- PROTOCOL §12 listed them in the opposite order.

**Resolution.** The lead adopted the prompt-literal order in `family.py`, in the truth-table rules (6 and 7 swapped) and in
the tests. Incomplete claims stay displayed. This was verified in round 2: family.py d54a03b3…, the truth table
30afa11c….

### R-4 REQUIRED: an INFEASIBLE constrained fit can be nominated. RESOLVED (commit e227909)

**The problem.**
- When a K- arm finds no feasible candidate, `mapper.fit_unit` still writes a release, with record `status`
  "INFEASIBLE".
- That release is a descriptive copy of the first candidate:
  - for K-LOCAL, or for K-JOINT when every witness is excluded, it is the unchanged first start map, which can be
    C-TASK;
  - for K-SEQ, it is an infeasible refined map.
- `select.candidate_rows` reads only the inner (aud__) records. It never reads the fit record's `status` or
  `deployed.feasible`.
- So an infeasible unit can become P\*, N\* or J\*, or a C_pair\* member, under the "constrained" label. That would
  credit a constrained search whose output breaks its own budgets. In the K-LOCAL case, it could even present a
  privacy-untrained C-TASK map as privacy-trained.

**Patch** (sent to the lead). In `select.candidate_rows`, for arm "constrained", record

```python
fit_ok = all(rec.get("status") == "FEASIBLE" and (rec.get("deployed") or {}).get("feasible") is True
             for rec in (R.rec(R.unit_for(k, cid)) for k in SEEDS))
```

- Set `row["fit_feasible"]` to this value and make `ordinary` require it.
- Give the role a distinct selection reason, CONSTRAINED_FIT_INFEASIBLE, under NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE. It
  is not a technical failure: the release is "infeasible under this registered decoder". Routing it through
  `technical_failure` would make the whole role INVALID.
- Register the rule in SELECTION_RULES and the truth table, and test it.
- For C_pair\*, the lead may either exclude such units or keep them. Keeping them is conservative for J\*. Either way the
  choice is registered, and such units are never nominees.

**Resolution (verified 00:26Z, select.py fc33af63…).**
- `select.fit_feasible(cid)` requires record `status` FEASIBLE and `deployed.feasible` True on every seed for K- arms.
- `ordinary` now equals inner-ordinary AND fit-feasible.
- `pick` gives the reason CONSTRAINED_FIT_INFEASIBLE when no candidate is fit-feasible. That is a selection outcome, not a
  technical one.
- The C_pair\* pool excludes infeasible constrained units.
- The rule is registered in `SELECTION_RULES.constrained_fit_feasibility`, together with the N-9 consequence, and is
  tested in test_select.

### R-5 RECOMMENDED: descriptive fallbacks do not rank fit-feasible units first. RESOLVED (commit a3d16bd)

- When a role has no nominee, the fallback ranking is `(ordinary_shortfall, guard_shortfall, ordering keys)`.
- An infeasible constrained fit that is inner-ordinary has `ordinary_shortfall` 0, because the shortfall is computed from
  inner utility only.
- It can therefore rank ahead of a fit-feasible constrained unit that has a small positive inner shortfall.
- §15 packages the constrained "candidate/fallback with truthful status". An infeasible-fit fallback would carry
  `fit_feasible: false`, but it would displace a feasible one.
- **Patch.** In `pick`, prefix both fallback keys with `byc[e["config"]].get("fit_feasible") is False`, so that False
  sorts after True:

```python
fb = min(ev, key=lambda e: (byc[e["config"]].get("fit_feasible") is False, round(e["ordinary_shortfall"], 12), ...)
```

- Register the change in `SELECTION_RULES.fallback_ordering`.
- This is descriptive only and never changes a nominee.

**Resolution (verified 00:44Z; select.py e672c15b…, SELECTION_RULES.json 160e4a5a…).**
- `pick` prefixes both fallback keys (the valid-guard branch and the missing-guard branch) with `fit_feasible is False`.
- `SELECTION_RULES.fallback_ordering.rule` states that fit-feasible candidates come first.
- `lcr/tests/test_select.py::test_feasible_fallback_ranks_before_an_infeasible_one` covers it.

### R-2 RECOMMENDED: privacy-trained references were displayed as privacy_trained = False. RESOLVED

The lead added a display-only `training` field ("privacy-trained reference" for RAW-J, FARE and LEACE) to the rows, the
CSV and SELECTION.json. `private_all` is unchanged.

### R-3 RECOMMENDED: the basis for the §17 sentence "Learning the probabilities [did/did not] improve confidence" was not registered. RESOLVED

The rule is now registered as `LABEL_TRUTH_TABLE.report_rules.decoder_sentence`:
- It compares the best D1 fixed-map privacy control with its paired D0 release.
- It uses the paired per-task D1 − D0 log-loss and Brier contrasts, with the same draws and the primary z. The contrasts
  are nominal and descriptive.
- The sentence reads "did" iff all four upper bounds are < 0. Otherwise it reads "did not clearly".

### N-1 NOTE: the label when B passes without A

`overall_label` then returns CONSTRAINED_SEARCH_INCREMENT_ESTABLISHED alone. The wording for this case is in
PRIOR_ART_AND_CLAIM_SCOPE.md §6.2.

### N-2 NOTE: continuous privacy-trained references are not P\* candidates. RESOLVED (note added)

- This follows the cbp convention.
- The choice changes nothing here: on the identical admitted inner rows, RAW-J, FARE and LEACE were ordinarily
  ineligible in cbp, with shortfalls of 0.727, 4.23 and 15.6.
- SELECTION_RULES.roles.P\*.note now says so.

### N-3 NOTE: a stale HEADROOM_SELECTION_FAILURE reason code. RESOLVED

The code has been removed from the truth table.

### N-4 NOTE: there is no D1 CLASS-ONLY control

Bank B in §8 covers DIRECT-TASK, FINE-TASK and the privacy maps only.

### N-5 NOTE: D0 DIRECT-TASK is not a JOINT witness

That is correct: §5 says it need not lie in the fine-cell family. It remains a T\* candidate (D0 and D1) and is on the
scored list.

### N-6 NOTE: the neighbourhood's "4 best" targets are ranked by each arm's own objective

**What the code does.** Each neighbourhood holds the 4 nearest targets by KL(cell mean ‖ target's D1 vector), plus the 4
targets with the best exact change in the arm's objective:
- Φ, or I_i for LOCAL, in the constrained arms;
- T for C-TASK;
- T + λΦ for the weighted controls.

**Assessment.** This is the same construction for every arm. For each weighted control, it is the stronger choice for its
own objective, so it does not weaken the control.

**Request (c) withdrawn.** My round-1 request (c) was to rank weighted controls by the privacy objective alone. I withdraw
it.

**How ranking and feasibility combine.** Ranking is over every alive same-class target. Feasibility filters only at
acceptance. The constrained arms follow the same rule.

### N-7 NOTE / RECOMMENDED: the equal evaluation ceiling probably binds only JOINT-PAIR

**Joint arms.** The 29 witnesses get 8,000,000 / 29 ≈ 275,862 evaluations each.
- A sweep over both recipients at full caps evaluates about 64 × 7 + 768 × 63 ≈ 48.8K (cell, target) changes.
- Five JOINT-SINGLE sweeps therefore need about 244K, which fits.
- JOINT-PAIR adds about 48.8K of pool screening per pair step, plus 64 pair evaluations. It reaches its share around
  sweep 3.

**Sequential arms.** Their stage shares (8e6 / 14 ≈ 571K) do not bind.

**Assessment.** This follows §8 (equal TOTAL ceiling). It handicaps the arm under test, not a control, so it is
conservative for claim C.

**RECOMMENDED:**
- (i) Confirm with C's synthetic full-bank timing.
- (ii) Report per-witness stop reasons and evaluation counts in OPTIMIZATION_RECEIPTS.json.
- (iii) State in METHOD_CARD that a claim-C failure in which JOINT-PAIR stopped at `eval_ceiling` is a budget-limited
  result.

The unused shares of excluded witnesses are not reallocated. That is deterministic, and the same for SINGLE and PAIR.

### N-8 NOTE / RECOMMENDED: the C-TASK record has null D0 external references

- `run.stage_ctask` passes no `refs`, so `ctask_report.d0_external_references` is null.
- The D0 DIRECT-TASK and FINE-TASK fitting losses exist in the dec__ records (`fitting.L_D0` and `fitting.B_D0`).
- §7 asks for these to be reported separately. Report them beside the C-TASK D1 initialisation and the refined map in
  BUDGET_FEASIBILITY.csv or METHOD_CARD.

### N-9 NOTE: the constrained arms have no restoration phase

- If C-TASK violates a fitting budget on some seed, K-LOCAL is infeasible on that seed by construction, because C-TASK is
  its only start.
- The same holds for every infeasible start or witness.
- This is a registered design consequence. It must be reported, and it feeds R-4.

### N-10 NOTE: acceptance is per cell within a sweep

- Within a sweep, the mapper applies the best improving feasible move of each cell's neighbourhood, in a deterministic
  order. It does not make one global steepest move per sweep.
- This is the same for every arm. METHOD_CARD should describe it that way.

## 4. Remaining work (round 3, before SCIENCE_LOCK)

- SEARCH_RULES.json: **done**. It equals `mapper.rules()` of mapper.py b3bfd756…. Any later mapper edit before
  SCIENCE_LOCK must regenerate the file and keep the frozen constants checked above (SWEEPS 5, 4 + 4 neighbourhood, 4 + 4
  pair bank, `EVAL_CEILING` 8e6, TOL/TIE_TOL 1e-12, BUDGET_MARGIN 1e-10, CAP_MARGIN 0).
- FIT_MANIFEST.json: **done** (see §1, round 4). It is complete, and nothing is weakened or omitted.
- R-5: **resolved**. Every finding is resolved or recorded as a NOTE. Nothing blocks SCIENCE_LOCK from the fairness side.
