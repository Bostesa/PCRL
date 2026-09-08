# Frozen-release adversary diagnostic — protocol frozen before fitting

Question: why did saved training adversaries underestimate leakage recovered
by independent auditors? This is a diagnostic of fixed checkpoints, not new
representation training, a protection improvement, or a method search.
Validation reuse is exploratory. Prediction-only release passed the previous
fixed-task pilot; no full-representation method passed. That result stands.

Use prior results/redesign_20260907_nonlinear_release_v1 read-only. Seed2 is
run first, including E_dual_0.01 and D_fixed_0.01. Then seeds0 use those same
keys; seed1 uses E_dual_0.1 and D_fixed_0.1, which remain diagnostic fallbacks
that failed the .99 validation task floor. Seeds0/2 passed both validation
utility floors but failed protection. Never replace these labels or select
new encoder checkpoints. Seed2 E.01 is explicitly included even if a saved
selection were to omit it. All saved encoders, heads and final calibrated
erasers are immutable; evaluation mode, no gradients, no eraser refitting.

## Data and evaluation

Unchanged generator: L=(U,V,S,E1..E5) iid independent centered unit Gaussians,
r=LQ1, X=(r+.2r^3)Q2+b. Sign-corrected QR Gaussian Q1/Q2 use seeds
20260908/20260909; b=linspace(-.4,.4,8). P1 prohibits V,S; P2 prohibits U,S;
combined access permits U,V and prohibits S. Original observation statistics
are reconstructed from representation training and checked against saved
manifests. Original four streams: 500000+100*s representation training4096,
500001+100*s calibration2048, 500002+100*s attacker_fit2048,
500003+100*s validation2048. Only attacker_fit enters diagnostic optimizers;
validation selects attackers. Neither representation training nor calibration
is reused as diagnostic attacker fitting data.

Fresh diagnostic tests: n4096 each, RNG1000004/1000104/1000204 for seeds0/1/2.
These seeds are declared before any fitting; test is generated only after all
seed-specific attacker selections and any declared bridge decision are saved.
No fitting after that boundary. Previous final tests are not used for fitting,
selection, or scores. IDs RNG*100000+row, split arrays/hashes saved.

Score every forbidden target separately: p1_V,p1_S,p2_U,p2_S,combined_S.
Raw-target predictive R²=1-SSE/SST, with evaluation-mean-centered SST. Predictor
centering/intercepts never use evaluation labels. Negative scores remain
unclipped, undefined scores fail rather than count as protection. Empirical
covariance is not a predictive score; no covariance certificate is used here.
Near-zero measured R² is not a general privacy guarantee.

## A/B/C/D comparison and fixed budget

A: exact saved final training adversary, no fitting, using its saved step400
training-release input mean/std and saved target mean/std, evaluated on the
saved final calibrated release. Reproduce all original validation scores
within1e-6 (dense saved P reconstructs the original factorized erasure in
float64 with only operation-order rounding). Preserve original weights.

B: two trajectories cloned from A's exact weights, reset Adam state for each.
C: two freshly initialized trajectories with the same training architecture.
B/C both use original step400 input and output coordinate conventions without
refitting or silently replacing normalizers. Normalize individual views then
concatenate; denormalize each output using its correct saved target statistics.
B retains prior900-update exposure on3072 representation-training rows. C has
none. Added fitting budget is matched, not cumulative lifetime exposure.

Both: three32x32ReLU MLPs, input8/8/16 output2/2/1, target lists(V,S),(U,S),(S).
Same original mean over five normalized target MSEs; minimize positive MSE.
Adam lr.001, betas(.9,.999), eps1e-8, no weight decay; batch256,80epochs on2048
rows =640updates/trajectory,1280total for each B/C method (three networks per
update),327680 row presentations per method. Every row appears80times in
each trajectory. Fresh initialization seeds970000+100*s+restart; shared B/C
batch permutation seeds980000+100*s+restart, restart0/1. Same batches per
candidate and update. Validate at0/every5epochs=0/40/.../640 updates. Select
minimum raw validation MSE independently per target, earliest candidate/epoch
ties. Save full selected states, common-count curves, schedules, actual
selected and total updates, exposure counts. No budget extensions.

D: load the exact saved independently selected audit checkpoints and their
original preprocessing; this reuses validated audit fitting, not old test
scores. Evaluate on the same validation and new diagnostic test examples.
Original audit budget: two starts,80epochs,640updates/start/network, with
validation selection at0/every5epochs per target. Same32x32ReLU hidden widths
but each network predicts all U,V,S (three outputs), mean of three standardized
target MSEs, separate network optimizer per view. Input/target statistics are
fit on attacker_fit, using variance mask/tolerance1e-10; saved training stats
use training-eraser holdout/input std floor1e-6 and update-pool targets.
Original D seeds700050+100*s+10*p+restart also drive its batch permutations.
D therefore differs in target sharing/output capacity, preprocessing, initial
states and batch streams. Its saved learning curves and selection metadata
are retained as reference; no claim of a pure capacity comparison is made.
Linear audit reference checkpoints are also re-evaluated, separately labeled.

One optional controlled bridge is predeclared, seed2 E.01 only: if any target
has validation D-max(B,C)>.05, repeat C with feature and target normalizers
fit on attacker_fit final releases (std floor1e-6), retaining identical fresh
weights, architecture, loss, batches and budget. This changes the preprocessing
block only and never rebases inherited weights. If the condition fails, do
not run it. No other factor grid or adaptive attack-budget extension.

## Frozen-release and refresh-mismatch evidence

Save dense final maps, parameter and buffer hashes, original checkpoint file
hashes, fitting IDs, individual and combined cached release hashes. Verify
all state and representative validation outputs unchanged after fitting and
testing. Never fit a final eraser. Saved training maps375 and400 are available.
The last50 adversary updates used map375; encoder update400 precedes refresh400,
and no adversary update follows refresh400. Thus the final saved normalizer
was refreshed without further adversary training. A preserves precisely the
reported step400 semantics, so historical evidence is not silently corrected.

On the same final encoder and validation examples, evaluate saved weights with
saved map375/375stats, map400/400stats, finalmap/375stats, finalmap/400stats and
map375/400stats. Explicitly label each change. Save release RMS differences.
This isolates effects of changing maps/statistics at fixed final encoder; it
does not reconstruct the unsaved encoder399/adversary last-training inputs or
prove why joint training failed. Do not recreate joint training as an original
checkpoint. Additional fitting plus freezing co-occur, so catch-up success
alone cannot establish that moving representations caused the gap.

Oracle trueU/trueV and deliberately exposed(U,V,S) controls reuse their exact
saved independent MLP/linear audit checkpoints; score the common validation
and new test draws. Report finite-sample spurious leakage and high recovery,
including combined S. Controls are diagnostics, not hyperparameter selectors.

## Runtime, preservation and reproduction

Existing local Apple M4 Pro24GiB CPU, one numerical-library thread. Time complete
seed2 (both releases, all added attacks and controls), record estimate twice
that for seeds0/1 before expanding. Target <15minutes; no external resources,
downloads, existing process interruption, pushes or PR. Any necessary common
budget reduction must be declared before testing; none is planned.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m pytest tests/test_frozen_release_diagnostic.py tests/test_nonlinear_release.py -q
.venv/bin/python experiments/run_frozen_release_diagnostic.py --out results/redesign_20260907_frozen_adversary_v1 --prepare-only
/usr/bin/time -p .venv/bin/python experiments/run_frozen_release_diagnostic.py --out results/redesign_20260907_frozen_adversary_v1 --seeds 2
/usr/bin/time -p .venv/bin/python experiments/run_frozen_release_diagnostic.py --out results/redesign_20260907_frozen_adversary_v1 --seeds 0 1
.venv/bin/python scripts/summarize_frozen_release_diagnostic.py --out results/redesign_20260907_frozen_adversary_v1
```
