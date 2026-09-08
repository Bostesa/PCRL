# Frozen scalar prediction release: closing audit

Declared September 7, 2026 before attacker fitting or fresh-test generation.
This closes the fixed-task toy benchmark; there will be no outcome-driven
attack or representation sweep. The original four result directories remain
immutable. This is an exploratory follow-up with reused validation examples,
not independent confirmation of a broad privacy hypothesis.

## Objects and policy

Use the exact saved C_task_only final encoder and task heads from
redesign_20260907_nonlinear_upstream_v1 for seeds 0,1,2. Freeze parameters and
buffers in evaluation mode, use the original observation standardizer, and
release the native scalar U prediction for P1 and V prediction for P2.
P1 forbids V,S; P2 forbids U,S; their concatenation permits U,V and forbids S.
No final eraser applies to these scalar outputs and no encoder/head is fitted.
Oracle true U/true V and exposed (U,V,S) in both views are diagnostic controls.
One known-leaky full representation, seed2 E_dual_0.01 from the release pilot,
uses its exact saved final erasers without refitting. Re-evaluate its saved
training adversary and saved continued-B audit as additional references.
Never load full-representation weights into a scalar attacker.

Keep the unchanged independent Gaussian latents and fixed nonlinear generator
X=(r+.2r^3)Q2+b, r=LQ1, Q seeds20260908/20260909. The first four streams remain
representation_train4096, calibration2048, attacker_fit2048, validation2048,
RNG500000+100*s+i for i=0,1,2,3. Only representation_train fits task probes;
only attacker_fit fits new attackers. Calibration is used only to verify
historical split identity. Every preprocessing statistic and intercept is
fitted on the designated fitting rows, never validation or final test.

Fresh test: 4096 examples per seed, RNG **1100004,1100104,1100204** respectively.
These are distinct from all four previous test streams. Generate each only
after all that seed's model/checkpoint/restart/family selections are saved.
No fitting occurs after this boundary. IDs are RNGseed*100000+row index.
Hash input/target/split IDs, all frozen model parameters/buffers, erasers and
cached releases before and after fitting and evaluation.

## Fixed audit budget

For every release, fit each view against only its relevant forbidden targets:
P1(V,S), P2(U,S), concatenation(S), with exact target-name/column alignment.
All methods receive the same three families:

* Affine OLS: float64, existing fit-only centering/scaling and intercept,
  rcond1e-10; one fit, no tuning.
* MLP: existing 32x32 ReLU architecture with input width matching the release
  (1,1,2 for scalars;8,8,16 for the full reference) and output widths2,2,1.
  Two fresh restarts, each1800 Adam updates, lr.001, batch256, squared error
  on fitting-standardized targets. Validation raw-target MSE at update0,
  every40 and1800 selects a separate full checkpoint per target, then restart;
  ties prefer earlier restart then earlier step. Minibatches shuffle fitting
  rows repeatedly; save actual schedules/counts. No early stopping of fitting.
  Each restart sees460800 row presentations; two see921600 per view/network.
  This exceeds the successful continued adversary's900 original+640 added
  updates (394240 presentations per trajectory). It uses2048 distinct fitting
  examples, whereas the inherited adversary also saw3072 representation-training
  examples. Thus update/exposure counts do not imply identical attack strength.
* HistGradientBoostingRegressor from installed sklearn: independent per target,
  squared_error,200 iterations,15 leaves,lr.05,l2_regularization1,
  early_stoppingFalse, other installed1.8 defaults. One fixed configuration;
  no extra tree search/restarts. Record200 full fitting-data passes per target
  (409600 row presentations); tree passes are not equivalent to Adam steps.

MLP/base RNG seed1200000+100*s+10*view; restarts and schedules are recorded
by the fitting module. The same seed/budget applies across release interfaces.
Task probes receive affine OLS and independently fitted MLPs with the same
1800-update/two-restart budget, base seed1300000+100*s+10*purpose, appropriate
authorized scalar target, fitting representation_train only. Native scalar
prediction R² is reported separately. No utility-head selection uses test.

For each forbidden relationship, each family's fitted model is fixed before
test; also select the strongest family using validation R² (ties alphabetical).
Report every family's validation/test score and the validation-selected family
score. Test family maxima may be descriptive only, never selection inputs.
No new representation or method/configuration selection is performed.

## Scoring and interpretation

R²=1-SSE/SST, with SST centered at the evaluated target mean. Predictions are
returned in raw target units. Negative scores are preserved as worse-than-mean
prediction, not negative information. Undefined scores fail closed. These
continuous targets have variance/sample-count coverage rather than class counts.
Covariance statistics are not used as held-out predictive leakage.

Existing primary utility floor: independent MLP probe R²>=.99 separately for
U and V. Also report whether both native task predictions meet.99. Protection
requires every individual forbidden relationship<=.05 and concatenated S<=.10
for **all three** fitted audit families, with per-seed validation and test
feasibility reported separately. No purpose/seed averaging hides failures.
Finite empirical passes do not establish universal privacy, unseen-family
robustness, reusable-task sufficiency, or PCRL novelty.

## Runtime and reproduction

Existing Apple M4 Pro24GiB, CPU, one numerical-library thread; no dependencies,
paid resources, other-process interruption, PR or push. Time seed2 first,
including full-reference audit and all fitting/preprocessing. Record a
conservative twice-first-seed estimate for seeds0/1 before expanding. The whole
authorized stage has an approximately two-hour experiment budget. No budget
change is planned; any needed common reduction must precede final testing.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m pytest tests/test_prediction_release_attackers.py tests/test_prediction_audit.py -q
.venv/bin/python experiments/run_prediction_audit.py --out results/redesign_20260907_prediction_audit_v1 --prepare-only
/usr/bin/time -p .venv/bin/python experiments/run_prediction_audit.py --out results/redesign_20260907_prediction_audit_v1 --seeds 2
/usr/bin/time -p .venv/bin/python experiments/run_prediction_audit.py --out results/redesign_20260907_prediction_audit_v1 --seeds 0 1
.venv/bin/python scripts/summarize_prediction_audit.py --out results/redesign_20260907_prediction_audit_v1
```
