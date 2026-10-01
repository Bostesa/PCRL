# Independent verification — summary

(Saved by the owner from the role's final report; the role's own write was refused by the harness.)

All ten checks F1–F10 have standalone fixtures under `../fixtures/` with JSON outputs in
`../fixtures/outputs/`; results are recorded in `independent_checks.json`. Verdicts: 7 confirmed;
F2, F6, F8 partially. Original PCRL code was executed only in `fixtures/orig_compare.py` (PCRL venv)
against read-only `git archive` exports of origin/main@55e4cb1d1 and
research/pcrl-submission-finish-v1@55c0c5a35. `fixtures/run_all.sh` reruns everything in ~3 min.
No repo/branch/cloud changes; no real-data fits.

## EARLY FLAGS

1. **One-hot certificate understates R² on origin/main** — `pcrl/purposes/verification.py:77-104`
   accumulates the Gram matrix in float32 (encoder emits float32, `certificates.py:327-336`);
   dominant-axis casts to float64. Stored audits: certificate never above the float64 value; up to 50%
   lower relatively (0.00327 vs 0.00652), ≤ 0.0033 absolute; nothing flips at τ=0.05 (56/60 either
   way). Synthetic ill-conditioned case: origin/main returns R²=0.0 and certifies where true R² = 0.578.
   The paper's "identity validated 33/33, max residual 0.0021" is this artifact; the identity is exact
   algebra (residual ~1e-15) when weights use the scored rows and per-class R² is not clipped.
   submission-finish-v1 casts to float64 (correct) but is not merged into origin/main.
2. **Dominant axis = max over the K class indicators.** Proven and checked: best linear contrast ≤
   (K−1) × that maximum, so a K=3 "all indicators small, contrast large" case is impossible (observed
   ratios 1.33 balanced, 1.71 imbalanced). With K=10: dominant axis 0.047, one-hot 0.041 (both pass),
   one contrast 0.364.
3. **Unsupported classes (F3)** — origin/main, absent middle class: certificate passes (0.035) while
   dominant axis reports R²=1.0 for that class; absent top class: silently dropped; all-constant labels:
   certificate R²=1.0. Fixed branch fails closed only when `num_classes` is passed; a singleton class is
   scored as valid (one row's leverage, 0.654 on pure noise when that row is an outlier).
4. **durable-guarantees README "21 certificate-approved" is wrong** (`README.md:57-59`, `:141`): 3 of
   the 21 rows were certificate "breach" → 18 approved, 15 collapsed, 3 survived; the 3 survivors are 2
   distinct measurements.
5. **PCRL counts mix checkpoints** — 56/60 headline from final-epoch audits; selected checkpoints give
   54/60 (the baseline the rebuttal pilot uses). Round 5 final vs best: Adult 21→23, Diabetes 18→16.
   Erase pilot: 60/60 strict, but its selector found 0 feasible epochs in 9/9 seeds and STATUS reads
   COLLAPSED; only one checkpoint stored, so final-vs-best cannot be checked.
6. **durable-guarantees Tier 2 = defense-aware attacker on one release, not an insider** —
   `battery.py:60-103` fits on clean training rows, scores one noised draw; `README.md:46-49`,
   `battery.py:15-21` claim pre-noise access or averaging. Averaging exists only in `averaging_attack.py`
   (breach at N=16, 0.560). Knows-Q attacker (0.982) beats the LRT (0.856) on subspace channels. Gaussian
   toy: knowing σ never gives access to h (analytic 0.624 vs insider 0.760).
7. **Withdrawn accuracy guarantee still live on origin/main** — `certified_accuracy_bound(0,.5,2)`
   returns 0.5 and `generate_report` prints it (`certificates.py:611`). 20-observation counterexample
   (R²=0, 90% accuracy) reproduces exactly. `fix/retire-accuracy-guarantee` unmerged.
8. **Nested predictors (F4)** — ACS coalition audits on all three inspected refs include nested
   singletons and H ancestors (`acs_coalition_audits.py:180-196, :310`; file identical across refs).
   `pcrl_utility_extension_v1` lacked J ancestors (documented in `pcrl_stochastic_channel_v1/slate.py:1-21`).
   Without nested predictors the fixture shows spurious combination effects −0.16 to −0.20 and
   increments −0.06 to −0.14.

## Other results
- F6: a constant column produces nonzero, mostly negative AUC changes (5/6 seeds); paired CI over
  evaluation people covers zero in only 4/6 seeds because it ignores refit randomness; across refits the
  CI covers zero and the spread equals a pure refit control.
- F7: all controls pass on two seeds; one control sat on the threshold and was redesigned (original
  output kept as `F07_first_attempt_noise1.35.json`).
- F8: stored figures reproduce exactly: 1/36; r=0.7949, ρ=0.8482 (n=20); 0.7991/0.8282 (n=27); 55/60
  dominant-axis passes; median amplification 1.43, max 12.65. hmda/diabetes continuous-cost shard files
  are not in git, so the merged file was used.
- F10: exact examples show leakage conditioned on a coarsened label and on the full output can differ in
  either direction.

## Limits
Stored JSONs trusted as written; F1 stress and F4 cases deliberately extreme; ACS audit code read, not run.
