# Statistical reviewer, wave-1b hand-back (verbatim; delivered to the coordinator at about 21:51Z; harness indentation removed)

# RFA1 statistical reviewer — wave 1b: the rules are final for registration (card RF_RFIT_v1)

**Completed 2026-10-25T21:49:38Z** (`date -u`), before any outcome of this campaign. I read the card `cards/RF_RFIT_v1.md` (SHA-256 f484c258…) and grepped `rfa/fam/fit.py` for its gate and bracket labels; that is code only. I read no label or data, made no git change and spawned no agents. Total CPU for both waves is about 36 CPU-s on one thread; this round's tests, mutation check and patch scripts took about 20 s of it.

## Final hashes (SHA-256)

| file | SHA-256 |
|---|---|
| `rfa/rules.py` | 76baa04a34762ecb85e2140ef2254b56f3714df1bfd79c9c58fa4edaf32a266c |
| `rfa/test_rules.py` | b00482c4fe924c9af4aa62c2503c773a34e3caac16f51f9690ebbf67403d0319 |
| `roles/statistical_reviewer/RULES.json` (schema `RFA1-rules-v1`) | 4e03270cf47d7353f81010d69035bf4e833994891df902ce14778754ee1e3916 |
| `roles/statistical_reviewer/RULES.md` | 1548b6de7c3ed9f08d66a2088b660fed9acc3ba19b805229c11a184dfdf50fe9 |

The inherited files are unchanged: NRM1 `RULES.json` 84f18776… and `nrm/rules.py` d5df8686…; `load_rules()` checks both. `nrm.rules` is imported unchanged and never monkeypatched.

## Test result

- **`python3 -m rfa.test_rules`: 24 of 24 PASS, exit 0**, 0.44 CPU-s, on invented data only.
- The new test `t_card_rf_rfit_v1` runs the card's record shape: family FAM, `RF_FULL`, `RF_TASK`, `RF_STOP_LO` / `RF_STOP_HI`, `RF_ALTC_HI`, `RF_RAWJ_HI`.
- The in-memory mutation check caught 35 of 35 seeded bugs: the 26 from wave 1 and 9 new ones for the card adaptation.

## Changes for the card (RULES.md section 0.2, RULES.json `change_record`)

1. **Family.** FAM is an alias of RFA, canonicalised to RFA, so two registered readings under the two names are refused. `RF_RFIT_v1` is revision 0; R1 is revision 1.
2. **LO not saved.** Record `lo_not_saved: ['ALTCRIT', 'RAWJ']`, which raises flag `LO_NOT_SAVED_SPEC_BUDGET_<key>`.
   - Those two paths are START + HI, so the cheaper-point check covers START only.
   - A bracket point that does not exist is simply left out of `paths`. A missing HI then usually reads NO_BRACKET.
3. **NEAR_ALIAS.** The rules apply the card's rule to the recorded `altcrit_median_relative_lag`: near alias iff the lag is strictly below 0.01.
   - **My interpretation of "not counted as an independent control":** a near-alias ALTCRIT **still gates**, because an extra conjunct can only make D2 harder. It is never counted as independent: flag `ALTCRIT_NEAR_ALIAS_OF_STOPFIT_NOT_INDEPENDENT` and scope sentence `D2_SCOPE_NEAR_ALIAS`.
   - A recorded `altcrit_near_alias` that disagrees with the lag rule makes D2 INCOMPLETE. A missing lag raises flag `ALTCRIT_LAG_NOT_RECORDED`.
   - If you intended near-alias to mean "dropped from gating", tell me before registration.
4. **F0 gate.** New CAND status `MECHANISM_INERT`, read from the `f0` record:
   - D1 reads `NOT_FITTED_MECHANISM_INERT`; D2–D5 and the learning curve read `NOT_ASSESSED_MECHANISM_INERT`; the outcome class is `MECHANISM_INERT`.
   - If only one version is inert, flag `MECHANISM_INERT_IN_ONE_VERSION_ONLY` is raised.
   - A FITTED reading with an inert version, or the inert status without the inert record, is INCOMPLETE.
   - Disclosure flag `F0_SEX_BASED_FIT_GATE_DISCLOSED` is raised whenever F0 is recorded.
5. **CAND's selection.** Selection data `MODEL_FIT_OOF`. The record key is now `selection_uses_sex_recovery_diagnostic` (False here); the flag is renamed `CAND_SELECTED_ON_MODEL_FIT_SEX_RECOVERY_DIAGNOSTIC`.
6. **Learning curve.** HALF is FIT fraction 0.5 under the inherited `RFA1_devgrid_LC` fold rule (folds 0–1 of 4). The 0.25 fraction is not run, and the paired SE comes from the shared bootstrap draws.
7. **Falsifiers.** Reported in `decisions()['falsifiers']` and as `FALSIFIER_<k>_TRIGGERED` flags; they never change a label.
   - F0 comes from `f0`.
   - F2 is TRIGGERED iff STOPFIT's frontier label at Q* is neither `DOMINATES_CONTROL_FRONTIER` nor `SEPARATED_ON_PRIMARY_COST_ONLY`. It reads `NOT_ASSESSABLE_NO_BRACKET` when there is no bracket.
   - F4 is TRIGGERED iff the learning curve reads `LEARNABILITY_LIMITATION`.
   - F1, F3, F5 and F6 come from your `falsifier_records`, and read NOT_RECORDED otherwise.
8. **New outcome class order:** `MECHANISM_INERT`, `TASK_INFEASIBLE`, `SELECTED_POINT_INFEASIBLE`, `INCOMPLETE`, `NO_OUTPUT_CHANGE`, `MECHANISM_NOT_ASSESSED`, `INCOMPLETE`, `COMPETITIVE_CANDIDATE`, `MECHANISM_COMPONENT_NOT_COMPETITIVE`, `REFERENCE_ADVANTAGE_NOT_ATTRIBUTED`, `BOUNDED_NEGATIVE_EXPERIMENT`.

## Exact top-level keys `decisions(rec)` reads

Ids are the card's. The same list is in the `rfa/rules.py` docstring and RULES.md section 10.

**Registration:**
- `family`: `'RFA'` or `'FAM'`
- `card`: `'RF_RFIT_v1'`
- `revision`: `0`
- `card_committed_before_any_model_select_outcome`: bool
- `grid_valid`: True / False / None

**START, CAND and CONT:**
- `start`: `{'id': 'NEURAL_RAW', 'evals': {0: utility, 1: utility}}`
- `cand`: `{'status': 'FITTED' | 'NO_FEASIBLE_POINT' | 'MECHANISM_INERT', 'config_id': 'RF_FULL', 'evals', 'operating_point': {0: …, 1: …}, 'selection_data': 'MODEL_FIT_OOF', 'selection_uses_sex_recovery_diagnostic': False, 'alias_of_start': bool}`
- `f0`: `{0: 'MECHANISM_ACTIVE' | 'MECHANISM_INERT', 1: …}`
- `points` (optional): `[{'config_id', 'evals'}]`. It is needed only to tell `TASK_INFEASIBLE` from `SELECTED_POINT_INFEASIBLE` when CAND fails G1.
- `cont`: `{'config_id': 'RF_TASK', 'evals'}`

**Controls:**
- `paths`: `{'STOPFIT': ['RF_STOP_LO', 'RF_STOP_HI'], 'ALTCRIT': ['RF_ALTC_HI'], 'RAWJ': ['RF_RAWJ_HI']}`
- `control_kind`: `{'ALTCRIT': 'PATH'}`
- `lo_not_saved`: `['ALTCRIT', 'RAWJ']`
- `altcrit_median_relative_lag`: float; `altcrit_near_alias` is optional
- `selection_data`: `{'STOPFIT': 'MODEL_FIT_OOF', 'ALTCRIT': 'MODEL_FIT_OOF', 'RAWJ': 'MODEL_FIT_OOF'}`
- `controls_not_run` (optional; budget only): keys CONT, STOPFIT, ALTCRIT, RAWJ
- `alias_of` (optional): `{id: earlier id | 'START' | 'CONT'}`

**Grid and costs:**
- `grid`: `{id: {'pair', 'income', 'occupation'}}`, NR grid entries over the 8 registered cells, for `NEURAL_RAW`, `RF_TASK`, every saved control point and the nine references.
- `costs_vs`: `{id: {v: {'total|nll': {'value', 'se'}, plus the 8 secondary metrics}}}`, for `NEURAL_RAW`, `RF_TASK` and every saved control point.
- `nolookup`: `{id: {'cells': …}}`, for the same ids as `grid` (NEURAL_RAW, RF_TASK, every saved control point and the nine references).
- Optional: `enhanced`, `reader_budget`, `mc_flags`, `joint` (None), `novelty`.
- `fineness` (optional): `grid_fineness(spends)` with `spends[key][v] = {cand, lower, upper, upper_se, max[, grid]}` and `selection_data: 'MODEL_FIT_OOF'`. For ALTCRIT and RAWJ, `lower` is the recorded, unsaved LO spend.

**Robustness, learning curve and falsifiers:**
- `d2r` (optional): the output of `d2r({'grid_valid', 'release': 'RF_FULL', 'refit': refit_effectiveness(..., boot_B=400), 'rob': {id: merged entries}, 'comparators': {'START': 'NEURAL_RAW', 'CONT': 'RF_TASK', 'STOPFIT': …, 'ALTCRIT': …, 'RAWJ': …}})`. Each control comparator is a Q* id, None, or a `brackets()` entry.
- `learning_curve` (optional; the reading is NOT_RUN without it): `{'grid_valid', 'budgets': {'FULL', 'HALF'}, 'inner_episode_fit_size', 'half': {id: entry}, 'diff_cell_se': {id: {cell: se}}}`.
  - Key the `half` cells by the registered names S<v>R<r>k<k>, not the `@0.5` names.
  - Give `half` entries for NEURAL_RAW, RF_TASK, every gating Q* and H_LJ.
- `falsifier_records` (optional): `{'F1' | 'F3' | 'F5' | 'F6': 'TRIGGERED' | 'NOT_TRIGGERED'}`

**Helper calls:**
- `grid_validity(seeds, checks, minimum_set, distinct_inputs, boot_seeds=…, boot_B=400)`. The minimum set is RF_FULL, NEURAL_RAW, RF_TASK and every saved control point.
- `brackets(costs_vs, paths, kinds, alias_of, origin='NEURAL_RAW')`.
