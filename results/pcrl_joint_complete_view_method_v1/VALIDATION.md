# Validation

## Before any real fit

| Check | Result |
|---|---|
| Admission | All 39,205 regenerated raw rows match the admitted record keys. Raw-file and label-manifest hashes match. Labels re-derive exactly. The corrected input contract has 83 columns (`DATA_ADMISSION.json`). |
| Prior-work and theory review | Independent and read-only (`review/`). Verdict: low novelty; the projection is GEM/A-GEM. Five specification statements were corrected before the lock (`THEORY_AND_LIMITS.md`). |
| Projection | Exact active-set solution vs SLSQP: 400 test cases plus degenerate cases (runner), and 5,900 cases at ≤ 2.5e-13 (reviewer). |
| Fixtures | 9 → 11 tests pass. They cover the XOR attacker positive control, the XOR training positive control (J coalition 0.52 vs L 1.00, tasks kept), the task = S conflict (reported, not repaired), constant withholding, gradient signs, seed aliasing, centring, the affinity of centred logits, and the A1/A2 regressions. |
| Fixture-stage technical repairs | Critic whitening, per-step ZCA, and 5 critic steps at learning rate 3e-3. All were found on fixtures and registered before the lock. |
| Timing pilot (synthetic) | About 0.9 CPU-s per protection epoch for J; the schedule was not reduced. |
| End-to-end plumbing (synthetic) | warm → train → FARE → inner → select → outer → infer ran on synthetic data before the lock. |
| Lock | `b67664f` was pushed at 19:45Z; the first real fit started at 19:46Z. |

## During the ladder

| Check | Result |
|---|---|
| Units | 216/216 planned units are complete and hash-verified (`UNIT_MANIFEST.csv`). 87 pre-A1 versions are quarantined, not deleted. 0 failed, 0 budget-unrun. |
| Amendment A1 | Released centred logits were −∞ under probability underflow (U, income). Fixed with the decision-function form; hard decisions were asserted unchanged; made before any inner result. |
| Amendment A2 | The race audit's cell-conditional attacker had a fixed K = 2. Fixed; no F/F0 outer output existed. The 14 completed neural outer units are unaffected and kept. |
| Projection exactness in training | Max constraint violation over 82,080 projected steps: 6.8e-13. No rescue triggered; no nonfinite gradients. |
| LEACE native check | 48/48 erasure units are within tolerance on the fitting rows (cross-covariance about 1e-15). Held-out max correlation with SEX: 0.014–0.021 (vs 0.31 for U). |
| Selection lock | `197f323` was pushed before any outer scoring; `jcv.outer` refuses otherwise. |
| Audit controls (attacker_fit / attacker_val only) | Label-permutation nulls 0.50–0.54 (none above 0.55). Planted leaks 0.91–0.95, all detected, on local and coalition views of J, L and F. |
| Deployment | `jcv.deploy` reproduces a saved release bitwise and refuses an 84-column input with an appended protected field. |

## Backup and restore

- The drive copy `<drive>/private_jcv_v1_20261003` holds 2,085 files. All were re-read uncached (F_NOCACHE; not a cold-disk unmount) and all match.
- Restored from the drive copy alone, the candidate J, the L control and both official FARE trees reproduce their releases, centred logits, probabilities and decisions with **zero** difference.
- The FARE trees re-encode the permitted inputs in the official environment.

## Independent replay

See `INDEPENDENT_VERIFICATION.json` and `verification/replay_jcv.py`. The section below is completed from the verifier's report.

**Result: 42 PASS, 0 FAIL, 4 WARN, 10 INFO.**
- An import guard refuses `jcv`, `oar`, `odx`, `cap`, `stored_model_eval`, `report` and `pcrl`, and the verifier asserts none of them was loaded.
- The maximum absolute difference is **0** for:
  - deployment-graph replays (neural units from `model.pt` + LEACE npz + heads; FARE releases);
  - inner utility and recovery;
  - selection values;
  - all 18 + 30 endpoint points and SEs;
  - levels;
  - the utility table;
  - sampled attacker refits.
- The LEACE native check recomputed on the fitting rows is ≤ 5.2e-14 relative.
- Claims recompute as NOT_ESTABLISHED.

**WARN items (none changes a decision):**
1. **`5d` F0 status label.** F0's own gates pass, but it is labelled GATES_FAILED because it is anchored to F's infeasible occupation configuration. This is disclosed in `SELECTION_LOCK.json`, and C\* is None regardless.
2. **`8c` coalition table recording.** In 16 coalition banks the ignore-other-view attacker won. The outer record then stores the local view's slate table under the coalition key, so the coalition's own slate losses are not in the record. The verifier refit the coalition slate for 2 cases (s2/J, s1/U) and confirmed the local attacker wins on validation log loss. This is a recording gap only.
3. **`10c2` A1 wording.** The probabilities changed at round-off (≤ 5.6e-17). An erratum has been added to `AMENDMENT_A1_2026-10-03.md`.
4. **`10h` locked document hash.** `UNIT_MANIFEST.csv` changed after the lock. The change was the post-run `status` column and the replacement of the two `<F-selected>` F0 placeholders per seed by the concrete names. The planned unit list and every other column are unchanged (diffed against `a984a1e`).

**Additional items from the verifier's final report:**
- **A1 row count.** The erratum is corrected: 195 rows per seed, 390 non-finite entries.
- **FARE certificate reporting.** The wrapper's refusal (UNAVAILABLE in `NATIVE_VS_AUDIT.csv`) and the descriptive record-identity recomputation (`FARE_CERTIFICATES_A3.json`: vacuous or unavailable) are now documented in `AMENDMENT_A3_2026-10-03.md`.
- **FARE seeds.** FARE seeds are not independent fits; this is added to the limits.
- **Scope of check 8c.** The verifier refit non-selected attackers (the coalition's own slate) for two views to audit the bank decision. No defense was trained.
