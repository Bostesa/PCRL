# E_ENGINEERING_REVIEW: hcal engineering review, role E (independent verifier), phase 0

Reviewer: role E. Date: 2026-10-07. Scope: the final code before ENGINEERING_LOCK, checked against PROTOCOL.md (§2–§10),
CALIBRATION_RULES.json, SELECTION_RULES.json, PRIMARY_FAMILY.json, LABEL_TRUTH_TABLE.json, CONTROL_PLAN.json and
REVIEW_FINDINGS_DISPOSITION.json. No label was read. No outer__, oprob__ or oatt__ unit was opened. No hcal file was
edited.

**Reviewed bytes** (first 16 hex digits of each file's sha256):

| File | sha256 (16) |
|---|---|
| hcal/ids.py | d5e283f70c06861a |
| hcal/data.py | f5d9cd072e60c255 |
| hcal/admit.py | 72d9f312757dea27 |
| hcal/calib.py | 1d25eb820c381832 |
| hcal/bank.py | db202e5ffa52da31 |
| hcal/controls.py | d8405873af763ccb |
| hcal/select.py | 4c773dce3c86c6b0 |
| hcal/family.py | 8939408d27aa516b |
| hcal/stages.py | 1c05d8d4493eec99 |
| hcal/infer.py | d3bee2d207fbb026 |
| hcal/eval_lock.py | 6986cd93b15e8c75 |
| hcal/assess.py | f104a94d20aa20a0 |
| hcal/deploy.py | 23eb54ef3ec8f993 |
| hcal/engineering.py | a6e00eb767b0d905 |
| hcal/run.py | 7599d87d1571a838 |
| test_bank.py | c127133ae8471232 |
| test_calib.py | dc5233cea50e8c08 |
| test_controls.py | 4bfa9e67ddc63551 |
| test_data_select.py | b4295f4ffebdf489 |
| test_deploy.py | f3091e93c49213f6 |
| test_infer.py | 0d1a81c255521031 |
| PROTOCOL.md | 0e499ea3c9cf7603 |
| CONTROL_PLAN.json | 61db2d18d3a59d80 |
| ENGINEERING_CHECKS.json | 87dc00e27e8d00ee |

Any later change to these files needs a re-check of the affected findings.

**Method.**
- Read every listed module and test against the protocol text.
- Wrote an independent replay (`verify_inner.py`) and a synthetic self-test (`selftest_verify_inner.py`).
- On the real admitted data, with no labels:
  - The role split reproduces ROLE_MANIFEST.json exactly.
  - All 171 frozen-bank tables pass the hash and consistency checks.
  - The bank-order reconstruction matches all 756 admitted legacy bank views and U's interface bank.
  - Role E's own fresh-view and hygiene reconstruction is bitwise equal to `hcal.bank.fresh_views` on 4 real partitions (view fingerprint and kept-column receipts). That comparison was a one-off scratch run; the verifier itself imports no hcal module.

## Summary

| Severity | Count | Items |
|---|---|---|
| BLOCKING | 1 | E-B1: controls verdict key mismatch |
| SHOULD-FIX | 7 | E-S1 … E-S7 |
| NOTE | 11 | E-N1 … E-N11 |

E-B1 fails closed: no wrong science can come out of it. But with it in place the study cannot nominate or open the
assessment, so it must be fixed before ENGINEERING_LOCK.

## BLOCKING

### E-B1: the controls verdict is read from a key that does not exist

`hcal/controls.py:488` builds `all_ok` inside `verdict_`, and `:518` returns it only as `"verdict": {...,"all_ok"}`.
There is no top-level `all_ok`. Three consumers read the top level:

- `hcal/select.py:339`: `sel["controls_all_ok"] = bool(ctl.get("all_ok"))`.
- `hcal/eval_lock.py:98`: `"controls_all_ok": bool(ctl.get("all_ok"))`.
- `hcal/stages.py:404`: `R.event("controls", all_ok=summ.get("all_ok"))`.

**Effect, even when every control passes:**
- `select_all` sets `controls_all_ok` to False. Through `select.py:340-343` it then rewrites T\* and P\* to TECHNICAL_FAILURE.
- `eval_lock` sets `technical_validity.ok` to False, so `hcal.assess` refuses (`assess.py:93`).
- If it went unnoticed, the label would read INCOMPLETE_OR_INVALID.

lra read `verdict.all_ok` (`lra/eval_lock.py:144-147`). The synthetic tests check `s["verdict"]["all_ok"]` only, and no
test exercises select_all or eval_lock on a real summary, which is why this slipped through.

**Fix:** read `ctl["verdict"]["all_ok"]` in all three places, or also write a top-level `all_ok` in
`summarise_controls`. Add one test that runs select_all's control gate and eval_lock's `technical_validity` on a
`summarise_controls` output.

## SHOULD-FIX

### E-S1: the P\* guard is worded two ways, and they differ by one ulp

| Source | Wording | Form |
|---|---|---|
| PROTOCOL.md:353 | "AUC_i(P\*) ≤ AUC_i(T\*) + 0.005" | sum form |
| hcal/select.py:208-209 (code) | margin `(ref+0.005) − rec ≥ 0` | identical to the sum form |
| SELECTION_RULES.json:15, hcal/select.py:312 | "AUC_v(P\*) − AUC_v(T\*) ≤ 0.005" | difference form |

The two forms disagree at the boundary. Example: 0.705 − 0.70 = 0.0050000000000000044 > 0.005, while 0.70 + 0.005 ≥ 0.705.

**Fix:** align the SELECTION_RULES wording to the protocol's sum form; the code is already correct. The replay uses the
protocol form and records any disagreement between the two forms.

### E-S2: the utility stage reads fitting-row labels outside the registered allowlist

`hcal/stages.py:190-194` and `:179-180` compute per-release metrics on OSF_DEFENSE_FIT (the `"fit"` metrics, for all 292
releases × 3 seeds). They use the inherited "fitting" procedure. The protocol limits that procedure:

- PROTOCOL.md:97: fitting labels are for "the source constant predictor only".
- PROTOCOL.md:215: utility is evaluated on calibration, TRAIN_MATCHED and inner rows.

The metrics are descriptive only, but they go beyond the registered allowlist.

**Fix (before SCIENCE_LOCK):** either drop `roles["fit"]`, or amend §2 and §4. The replay does not reproduce these
metrics.

### E-S3: the decoder-variant test is tautological

The required category "unchanged primary privacy predictions across decoder variants" is covered only by
`tests/…/test_bank.py:409-443`. That test calls `assign_variants` (`hcal/bank.py:676`), which maps every variant to the
same dict, and then asserts that the dict equals itself.

The real property is in the consumers, and none of them is tested on it:
- `select.recovery`, `select.py:175-179`;
- `infer.partition_key`;
- `eval_lock.attack_keys`;
- the `assess` attack jobs.

**Fix:** add a test asserting that every decoder variant of a partition resolves to the same `com__` record and the
same `oatt__` unit, and that no per-variant attack path exists.

### E-S4: the "missing comparator" selection test cannot fail

`tests/…/test_data_select.py:214-220` asserts `status == "NO_ELIGIBLE" or parse_release(T*)[0] is None`. In this
fixture T\* resolves to a U variant, so the second clause is always true. As a result,
`select.py:252-255` (NOT_SELECTED_NO_COMPARATOR) is never exercised.

**Fix:** make the U variants ineligible as well, then assert `T*` = NO_ELIGIBLE and `P*` = NOT_SELECTED_NO_COMPARATOR.

### E-S5: untested selection paths

No test covers any of these:
- **Ordering and tie keys:** worst-seed normalised excess, then mean summed NLL, then registered ID order, after r12 rounding (`select.py:182-183`).
- **Inclusive boundaries:** the benefit at exactly 0.02, and the Ucal\* gate.
- **Control gate:** select_all's controls → TECHNICAL_FAILURE path (`select.py:339-343`); see E-B1.

**Fix:** add these four cases to `test_data_select.py`.

### E-S6: no independent solve at K = 6

The certificate-against-an-independent-solve tests (`tests/…/test_calib.py:185`, `:208`) cover K = 2 (bounded Brent)
and K = 3 (SLSQP) only. K = 6, the occupation recipient and the main use, is only checked bitwise against
`lra.decoder` (`:123`).

**Fix:** add one K = 6 independent check. A Frank–Wolfe vertex gap works: role E's self-test measures a gap of
≤ 4.8e-16 relative on lra-certified K = 2, 3 and 6 solutions, and it detects 1% feasible perturbations.

### E-S7: the controls stage reads bank files without hash checking

`hcal/stages.py:393` calls `CT.run_controls` without `load_bank_fn`. The default in `hcal/controls.py:457-459` loads
`bank__*.npz` with no admitted-hash check. Every other stage goes through `run.bank` (`hcal/run.py:124-132`), which
refuses a hash mismatch.

**Fix:** pass `load_bank_fn=R.bank`.

## NOTE

**E-N1. Fresh-view hygiene drops columns that are duplicates on the fit rows only.**
- Where: `hcal/bank.py:251-275`, `:308-331`; PROTOCOL §5 "Fresh-view hygiene".
- The rule is label-free, deterministic and registered.
- "Exact duplicate on F" includes columns that differ off ATTACK_FIT_NEW. `test_bank` column 2 expects exactly this.
- So for a token unseen in ATTACK_FIT_NEW, readers see only the first-kept table's values.
- Dropping columns that are constant on F also removes the random-init MLP weights on unseen-token one-hots.
- Both are legitimate attacker changes. The admitted legacy banks still see these tokens through AUDIT_FIT.
- Real-data scale, seed 0: v2 drops 39–40 constant columns (i8o64 partitions). CLASS|i1o1 v1 drops 2 duplicates (the decision one-hot equals the token one-hot).

**E-N2. The union rule's "running best" is the current winner's value.**
- Where: `hcal/bank.py:614-619`. This matches lra's `composed_source_bank`.
- With values 0, 0.6e-12 and 1.2e-12, the third bank wins.
- The replay implements the same reading.

**E-N3. The U-source refit has no view-fingerprint check.**
- Where: `hcal/bank.py` `u_entry` sets `view_fingerprint=None` for `lra_source`, so `refit_selected` skips that check.
- The bitwise refit of the stored INNER predictions still guards it.

**E-N4. The role receipt does not assert the expected 4,061 ATTACK_FIT_NEW groups.**
- Where: `hcal/data.py:175-180` leaves it out of `ok`; it is only recorded at `:181-182`.
- The realised split is correct: 4,061 groups and 4,064 rows, which the replay re-derives.

**E-N5. `hcal.data.labels_for` has no SCIENCE_LOCK gate.**
- Where: `hcal/data.py:255-278`.
- The "no label before the pushed SCIENCE_LOCK" rule rests only on `run.main`'s lock verification. Process discipline covers ad-hoc scripts.

**E-N6. A non-finite utility metric would crash selection instead of failing a gate.**
- Where: `hcal/run.py:51-64`. `_finite` turns non-finite floats into null in unit records.
- `gate_seed`'s `float(None)` would then raise rather than mark the release ineligible.
- The finiteness flags exist, so this is fail-loud, not silent.

**E-N7. The truth table shows criteria PASS/PASS for rows where nothing ran.**
- Where: `LABEL_TRUTH_TABLE.json`, cases "engineering blocked" and "inputs unavailable".
- Not reachable from `infer`, which always passes `engineering_ready` and `inputs_available` as True.

**E-N8. The alias disclosure hashes raw token IDs.**
- Where: `hcal/select.py:224-237`, `:278-280`.
- A map that equals a task-only map up to token relabelling is not flagged.
- Accepted per disposition "B-other" as "exact deployed bytes".

**E-N9. Deploy accepts T-TOKEN32 decoders, and some refusals are untested.**
- Where: `hcal/deploy.py:44`. T-TOKEN32 is diagnostic only, so accepting it is harmless.
- `test_deploy` tampers only H-GLOBAL-TEMP. H-TOKEN32 tampering and a `--schema-sha256` mismatch are not tested.

**E-N10. The audit plan counts MEAN-only eligibility.**
- Where: `hcal/select.py:148-166`. An lra partition whose only eligible variant is MEAN is still audited.
- This is conservative, and accepted as registered (disposition B-other).

**E-N11. Two stage-requirement lists are incomplete.**
- Where: `hcal/lock.py` `STAGE_REQUIRES["utility"]` omits `hcal/select.py` and `hcal/stages.py`; `hcal/run.py` `STAGE_MODULES["utility"]` omits `hcal/select.py`.
- `check_loaded_modules` after the stage catches any unlocked module, so this is not a gap.

## Protocol conformance

Every area conforms to the protocol except two qualified rows: controls (E-B1, E-S7) and selection (E-S1).

| Area | Code | Finding |
|---|---|---|
| Roles, split, representatives, disjointness, allowlist, HEAD_VALIDATION refusal, assessment sealing | `data.py:68-183`, `:255-296` | Conforms. Replay PASS (counts, purity, 13 disjointness pairs, all manifest hashes). See E-S2, E-N4, E-N5. |
| H-TOKEN32 / T-TOKEN32 objective, prior mu_t, fallbacks, certificates | `calib.py:240-355` | Conforms (A = y + 32 mu; reserved → q0; n_cal = 0 → q0; lra certificate; strict class margin). |
| H-GLOBAL-TEMP / H-CLASS-TEMP | `calib.py:370-540` | Conforms (unclipped LSE NLL; convex derivative; bounded bisection; boundary KKT; α = 1 identity; < 50 → α = 1; all tokens transformed). |
| U temperatures | `calib.py:544-625` | Conforms (log-input floor; α = 1 → P exactly; argmax == teacher decision on every row; normalisation on transformed rows, finding B-2). |
| Common bank and union | `bank.py` `fresh_views` / `fresh_family` / `common_record` / `_union` | Conforms (legacy D0 then D1, or D1 only, then fresh; complete fresh view with every registered table in order, then hygiene; fixed orientation; first-in-order ties 1e-12; seed 0–2 mean). See E-N1, E-N2. |
| U composition | `bank.py` `u_entry` / `compose_u`; `stages.py:285-302` | Conforms (lra SRC\|U composed bank, then fresh banks of audited partitions in registered order; stored predictions reused). |
| Controls | `controls.py` | The common-bank design matches PROTOCOL §5 (rewritten) and CONTROL_PLAN.json: one S\*, union on half A, evaluated on half B, per-pipeline results diagnostic, source limits unchanged. Not conforming until E-B1 is fixed; see also E-S7. |
| U0 gate, Ucal\*, Ucal gate, audit pruning | `select.py:65-171` | Conforms (qpc margin forms, inclusive; ties within 1e-12 → identity, global, class; non-diagnostic variant on every task and seed, plus always-audited partitions). |
| T\* / P\* | `select.py:186-288`, `:324-345` | Conforms on pools, gates, ordering and disclosures. See E-S1 (wording) and E-B1 (control gate). |
| 23 slots, z, B, bootstrap, label precedence, criteria validity | `family.py`, `infer.py` | Conforms (z = NormalDist().inv_cdf(1 − 0.05/46); retention = a − 0.8b − 0.2c; INVALID_TECHNICAL / INVALID_NO_COMPARATOR, finding B-3). |
| Assessment gate | `data.py:186-246`, `assess.py:49-146`, `eval_lock.py` | Conforms (`hcal.assess` caller only; lock committed, pushed and byte-identical; code hashes; technical validity incl. E PASS). Blocked by E-B1. |
| Deploy refusals | `deploy.py` | Conforms (83 columns in order; teacher, schema, policy and decoder hash bindings; bitwise table reproduction; strict argmax; extra or reordered columns, raw-score / fine-ID / unknown flags → exit 2). See E-N9. |

## Tests against the required correctness categories

ENGINEERING_CHECKS.json: 125 nodes, all categories covered.

| Category | Verdict |
|---|---|
| Identity probabilities and predictions | Meaningful (bitwise α = 1 paths; a solve returning 1.0; U identity rows unnormalised within check_probs). |
| Known non-identity optimum | Meaningful (K = 2 closed form for temperature; K = 2 Brent for token32). |
| Zero / small counts, class fallbacks, boundary α, absent classes | Meaningful (n_cal 0/1/2/40; reserved tokens with calibration rows; 49/50 class threshold; absent class; both boundaries; U class fallback). |
| NLL derivative / convexity; per-token certificates vs an independent solve | Meaningful (finite differences, second differences, monotone g). K = 6 gap: see E-S6. |
| Exact group partitioning and duplicates | Meaningful (duplicate-row groups, smallest-row-ID representative, label-free determinism, sha256 rank). The impure-group refusal (`data.py:94-96`) is untested (minor). |
| Unchanged primary predictions across decoder variants | Tautological; see E-S3. |
| Attacker orientation, unseen tokens / pairs, ignoring one recipient | Meaningful (anti-correlated reader stays below 0.5; unseen token → ATTACK_FIT_NEW prior; unseen tuple → INNER-selected fallback rule; pair bank order). |
| Denominator / seed alignment; valid / missing nominee / comparator | Partly. infer seed-alignment and union per-seed checks are meaningful; the selection-side missing-comparator test is vacuous (E-S4); ordering ties are untested (E-S5). |
| Deployment; controls on synthetic plants; fit rows strictly ATTACK_FIT_NEW; refit replay | Meaningful. Missing integration test of the controls summary consumers (E-B1). |

## Contracts `verify_inner.py` relies on

Please keep these stable after ENGINEERING_LOCK, or tell role E before changing them.

- **`cal__s{k}__<safe(p)>`**
  - `tables.npz`: `"{fam}|q{i}"`.
  - `tables.json`: `"{fam}|r{i}"` with `q`, `n_cal`, `y_cal`, `status`, `parameter_count`, `alpha` / `alphas`, `certificate.status`, `certificate.classes[c].{status, solve_status}`.
  - `record.json`: `families`.
- **`calU__s{k}`**
  - `u.npz`: `"{fam}|p{i}"`.
  - `record.json`: `tables["{fam}|r{i}"].{alpha, alphas, certificate}`.
- **`util__s{k}`**: `releases[rid].{inner, cal, tm}[task].{acc, logloss, brier, const_acc, gain, n, const_class}`, plus `preserved{"1","2"}` and `finite`.
- **`run/audit_plan.json`** and **`run/utility_table.json`**: as written by `stages.py:217-235`.
- **`fam__s{k}__<p>`**
  - `record.json`: `roles.fit`, `fit_row_id_sha256`, `sel_row_id_sha256`, `n_fit`, `attacker_seeds`, `meta.{tables, table_sha256, hygiene{v1,v2,pair,fit_rows}}`, `view_fingerprint`, `selection`, the auc/ce means, seed-0 and per-seed values, and `tables`.
  - `inner_preds.npz`: `keys_fresh`, `P_fresh`, `SEL_fresh_{crit}_{w}`, `sel_row_id`.
- **`com__s{k}__<p>`** and **`com__s{k}__SRC_U`**: `recovery.{banks[].name, winner, ce_winner, auc, ce, *_seed0, *_per_seed, bank_seed0, winner_detail, partition, seed}`.
- **`run/selection.json`**: `ucal_star.release`; T\* and P\* `status`, `status_if_controls_had_passed`, `release`, `ranking[].{release, mean_pair_auc, pair_benefit, auc_per_seed}`, `rejected[].{release, reason}`; `controls_all_ok`; `disclosures`.
- **`run/controls.json`**: `verdict.all_ok`. Informational only; not replayed.

## Status of `verify_inner.py`

Location: `verification/verify_inner.py`. No hcal import, which is checked at exit.

**Gating.** Labels are read only after SCIENCE_LOCK.json is byte-identical on origin. Every label read is recorded.

**Current state (pre-lock run):**
- 1_ROLE_SPLIT: PASS.
- 0_FROZEN_BANK: PASS.
- Everything else: PENDING (no calibration units yet, labels not allowed).
- verdict: PENDING.

**Updated for the final code:**
- The fresh-view hygiene receipt and the view fingerprint are re-derived from the bank and the stored tables.
- The selection status is read through `status_if_controls_had_passed`.
- U is composed only when every audited fresh bank is present.
- `controls.json` is read at `verdict.all_ok`.

**Self-test** (`selftest_verify_inner.py`, synthetic, no hcal import): PASS.

**Run with:**

```
~/PCRL/.venv/bin/python -P <WT>/hcal/sema.py --label E:verify_inner -- env OMP_NUM_THREADS=1 PCRL_HCAL_PRIVATE_CACHE=$HOME/PCRL_eval_cache_private/hcal_v1 ~/PCRL/.venv/bin/python <WT>/results/pcrl_heldout_calibration_v1/verification/verify_inner.py
```
