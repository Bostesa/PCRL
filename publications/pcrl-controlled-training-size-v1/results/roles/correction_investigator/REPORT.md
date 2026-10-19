<!-- Attribution: written by the conditional-correction investigator (an agent assignment of this campaign) and returned as text
in its final hand-back, because the harness refused report files written by that agent. Saved verbatim by the coordinator at about
00:33Z (2026-10-19, date -u); not coordinator analysis. Frozen files: tsz/correct.py 46cfb5e5..., tsz/nested.py a63fb59e...,
tsz/select.py 1ffd66cf... Coordinator notes on its open issues: the optional nested CONT (descriptive) is run after SELECT; the
investigator's `tsz.correct realized` command is not run, because `tsz.evaluate conditional` records the realized MODEL_SELECT
costs at every strength (avoiding a redundant MODEL_SELECT read); purposes of corrected-release reads name the correction. -->

# Conditional-correction investigator: readiness report
Campaign `pcrl_controlled_training_size_v1` (exploratory development on previously used records; not confirmation); written about 00:30Z on 2026-10-19 (date -u). This covers code, invented-data selftests and a label-free benchmark only. No real label was read and no real-data fit was made. The trigger fired at 00:27Z.

## Commands
- `tsz/correct.py`:
  - `base --version V` (python3)
  - `direction --version V --arm CONT|LOC|PAIR` (torch)
  - `releases --version V` and `verify --version V` (python3, label-free)
  - `realized --version V` (after selection; operation `benchmark`)
  - helpers `witness_predict`, `correction_delta`, `predict(v, arm, s, X, names)`, `predict_many`, `release`
- `tsz/nested.py`:
  - `witness --version V [--folds a-b]` (python3, resumable per fold)
  - `routing --version V` (CatBoost)
  - `correct --version V --arm LOC|PAIR|CONT` (torch)
  - `bench` (label-free)
- `tsz/select.py`: `run --version V` (reads no label)
- Outputs (write-once, hash-recorded): `runs/CORRECT/V<v>/`, `runs/NESTED/V<v>/`, `results/<campaign>/conditional/SELECTION_V<v>.json`

## Implementation
- **Label access.**
  - `tsz.correct` uses operation `fit` on S100: `base` reads occupation; `direction` reads occupation and SEX.
  - `tsz.nested` uses `select_ncv` on S100: `witness` reads occupation; `routing` reads occupation and income (income only for the internal-split class-support repair); `correct` reads occupation and SEX.
  - `tsz.select` reads no label.
  - Each module calls `load` itself, so the logged entry point is always the authorized module.
- **Registered constants.** Read from RULES.json and checked against `tsz.rules.STRENGTHS`: the strength set {0, .01, .03, .1, .2, .3, .5, .7, 1} and the guard-'none' targets {0.005, 0.005}. Real-data commands refuse if either is missing. The five statistics must equal `ohn.protect.SLACK_OF` (asserted); `cost_summary` provides the cluster and subgroup SEs over MODEL_FIT households.
- **W = W100_v.** `releases` uses W100_v's stored arrays (`tsz.sizefit.release`). Deployment uses W100_v's own package (`tsz.sizefit.predict`).
- **Base.** `base` reruns the source pipeline (`ohn.fam_a.run_pipeline`, A1_PO_16, seed v, OHN1_crossfit outer folds, inner folds OHN_A_cv_S<v> restricted to each pipeline). It gates on two things: the S100 FIT-only dictionary is the identity, and the refit deployed release equals W100_v bitwise on all six roles. The comparison with the admitted source (release, T*, folds, out-of-fold vs CROSSFIT.npz) is recorded. It gates only when the deployed release equals the source bitwise, in which case the out-of-fold array must be bitwise too.
- **Full-pool routing** = `tsz.control.load_teachers('S100', v)['occupation']`, installed write-once into `CORRECT/V<v>/ROUTING`. Equality with the admitted teacher and dictionary agreement are recorded and checked.
- **Directions** use `ohn.protect.train_correction` unchanged (P2 recipe, guard 'none'), one full pipeline per arm, seed 'TSZ1_V<v>', fold 'full'. The arms are identical except for the critic term. PAIR reads the frozen H3 income S<v> wire, during training only.
- **Releases.** r_s = softmax(log p_W + s·Delta), with r_0 equal to W bitwise (RULES 8.3). Delta is recomputed in python3 on the deployed path and compared with the training-time Delta (recorded). `predict()` from features must then reproduce every array bitwise.
- **Nested estimator.** Uses the 10 TSZ1_ncv10 household folds. For each fold, on the 90%:
  - FIT-only dictionary;
  - A1_PO_16 witness plus its own 5-fold base, with folds computed on all MODEL_FIT units and restricted;
  - T_TREE routing, seed v, recoded with the fold dictionary, parity checked on the training and held-out rows;
  - LOC / PAIR corrections, seed 'TSZ1_V<v>', fold 'ncv<k>'.

  Fold k is then scored with its own witness and routing. Per-person increments are taken against the fold's witness (s = 0) at every strength, pooled over 20,000 people, and summarized per strength with `cost_summary`. Every submodel's training size is recorded (deployed witness, its 5 cross-fitting pipelines, routing internal split, correction). Steps are cross-checked by row indices, dictionary hash, teacher hash and file hashes. The disclosure is in each table.
- **Selection.** `rule_table` (the one adapter) feeds `tsz.rules.select_strength(cost_table, version)`. The record holds: the selection per arm (LOC primary), CONT@s*_LOC and CONT@s*_PAIR, the rule's full audit trail, the cost tables, the nested JSON and npz SHA-256s, the RULES / rules.py hashes, a digest of the access log at selection time, and the guard against a prior MODEL_SELECT read of a corrected release.

## Departures and why
From `cts/nested.py`:
1. 10 registered outer folds and operation `select_ncv`.
2. The nested base folds (OHN1_crossfit, OHN_A_cv_S<v>) are computed on all units and restricted to the 90%, as the brief asks, instead of fresh per-fold tags.
3. The s = 0 release is defined as W itself, bitwise, not softmax(log p_W), which differs by about 1e-16. This matches RULES 8.3 and makes the identity row exactly zero; increments at s > 0 are against the witness, as in `cts.nested`.
4. The registered 9-value strength set replaces the old 0.0–1.0 ladder, with new stream names.
5. targets are {0.005, 0.005} instead of {1, 1}; theta is unaffected (checked bitwise).
6. Atomic per-fold directories make the step resumable, and more diagnostics are recorded.
7. CONT is accepted as a descriptive arm.

From the source P2 recipe:
- One full pipeline per arm, with no cross-fitted direction pipelines and no end projection; the nested estimator plus the frozen rule replaces them.
- p_W is the refitted W100_v; routing is the S100 control teacher; strengths come from a finite registered set.
- The training core is unchanged.

## Benchmark
Fold 0 (18,005 people, 12,003 households), one thread, through the queue:

| Step | CPU seconds |
|---|---|
| witness | 115.5 |
| routing | 6.5 |
| correction LOC / PAIR | 26.2 / 25.9 |
| full-pool direction | 24.9 |

- Per version (LOC + PAIR): about 2,056 s. Both versions: 1.14 CPU-h (1.29 with CONT); 1.4–1.6 CPU-h with the 1.25 margin.
- Wall time: about 35–38 minutes with two parallel lanes.
- Caveats: the invented labels shift T* and tree counts a little. Inner-CV fits always run 1,000 iterations, so the witness cost hardly depends on the labels. The correction figures used T = 800, so they are upper bounds.

## Lanes
Per version: BASE → DIR_CONT, DIR_LOC, DIR_PAIR → RELEASES → VERIFY → NCV_WITNESS_F0_4 → NCV_WITNESS_F5_9 → NCV_ROUTING → NCV_CORRECT_LOC → NCV_CORRECT_PAIR → SELECT. Optional NCV_CORRECT_CONT after SELECT. `realized`, and then the readers, come after SELECT. BASE runs first so that a W100 or control mismatch stops the lane before the long nested units.

## Open issues
1. **Guard false positives.** `tsz.select` refuses to write the selection if any earlier MODEL_SELECT read looks like a read of a corrected release. It recognises one by entry point `tsz.correct`, or by a purpose containing `correction` / `corrected` (any case) or the upper-case tokens `LOC`, `PAIR`, `CONT`. If an unrelated purpose contains one of these, the selection will refuse; the 00:31Z access log has no match. Please have evaluator and reader purposes for corrected releases name the arm.
2. **Selection file location.** It is written to `results/<campaign>/conditional/SELECTION_V<v>.json` (constant `tsz.select.SEL_DIR`). Change it there if you want it elsewhere.
3. **Routing dictionary.** Nested routing is recoded with each fold's FIT-only dictionary, which matches the `tsz.control` / `tsz.sizefit` convention.
4. **Disk.** The release arrays are about 54 MB per version, in the private tree.
5. **Write-once.** Every output is write-once; to repeat a unit, the old outputs must be moved and a new unit id used. A crash in the nested witness step can simply be resumed, because completed folds are skipped.
