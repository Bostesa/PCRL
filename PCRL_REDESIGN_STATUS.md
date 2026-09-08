# PCRL redesign, first implementation pass

Scope: repair evaluation correctness, retire the invalid universal accuracy
certificate, and run a small continuous Gaussian purpose-conflict pilot locally.
No remote changes, historical-result edits, model downloads, or parameter sweeps.

Starting branch: `ablations-facct-2026-07-24`.
Starting commit: `15dbcc3c3338f6707e7a0b3901d738d99651a77d`.
Tracked files were clean; pre-existing untracked reviews and result directories
are preserved. No AGENTS.md exists in this checkout or its ancestors; read the
README, reproducibility guide, requirements, and local tool settings.

Confirmed: unsupported OvR columns return R²=1; constant targets can return
misleading 0/1 scores; selection applies the per-purpose threshold to concatenated
constraints; epoch 0 follows training; the canonical runner freezes a randomly
initialized backbone; training reactivates backbone dropout; fit_erase_layer fits
one shared union eraser. The active accuracy-bound function asserts the false
classification guarantee and has report/certificate and experiment callers.

Prior evidence inspected read-only: `cross-purpose-rebuttal-2026-05-18` at
`d392112` (unified cross-purpose evaluation), `rebuttal-evidence` at `5739d3b`
(erase-layer/rank/VICReg evidence), local `results/rebuttal/cross_purpose/`, and
`results/rebuttal/erase_layer_pilot_aws/HEADLINE_PARTIAL.md`. Reuse existing LEACE,
LoRA and proxy-dual code; no branch merge. Prior scores are historical evidence,
not revalidated by this pass.

Hardware: Apple M4 Pro, 14 CPU cores, 24 GiB RAM, macOS 15.6 arm64. Pilot restricted
to CPU and one numerical-library thread. Existing .venv: Python 3.13.7, torch
2.10.0, numpy 2.4.2, sklearn 1.8.0. No other process was stopped.

Repairs completed:

- Fixed class schemas and support/variance masks; undefined scores are NaN,
  excluded from gradient/dual updates and cannot pass feasibility. Aggregate,
  dominant-axis, null-space and MLP OvR audits propagate coverage; MLP majority
  predictions are fitted on training data. NumPy scorers preserve continuous
  floating targets. Legacy trainer/BIOS consumers handle undefined scores.
- Selection, violation reporting and canonical checkpoint choice use each
  constraint's threshold, including the separate concatenated threshold.
  Validation leakage is now a full-split empirical fit, explicitly distinguished
  from held-out prediction. Missing coverage cannot produce a feasible label.
- `initialization.pt` is epoch 0 after any closed-form calibration and before
  gradient updates. Primal/vCLUB optimizer counts and selected checkpoint
  encoder/head/vCLUB/dual states agree with their selected epoch.
- Frozen backbone BatchNorm and dropout remain in eval mode; adapter dropout
  follows training mode. `--backbone-init random|pretrained` records provenance;
  pretrained requires an explicit local checkpoint and records its SHA256.
  No pretraining was performed in this pass.
- `--erase-mode shared_union|per_purpose` preserves the union baseline and adds
  independently calibrated prohibited sets. Structural erasure requires
  `--lora-target repr_proj_only` so upstream features remain stable.
- Disabled `certified_accuracy_bound` and its dependent nonlinear certificate;
  active reports no longer claim those bounds. Exact 20-observation regression:
  both conditional means 0, affine least-squares R² 0, threshold accuracy .9,
  majority .5. Active docs corrected; manuscript theorem source is absent
  (the checkout contains paper tables only).

Commands/evidence:

```sh
# Reconnaissance: git status --short; git branch -a; git rev-parse HEAD;
# git log / git diff HEAD against both named prior branches; rg scorer/callers.
.venv/bin/python -m pytest tests/test_redesign_conflict.py -q
# First-seed run, then a measured estimate of 2.56 s for the remaining seeds:
/usr/bin/time -p .venv/bin/python experiments/run_redesign_conflict.py --out results/redesign_20260907_gaussian_v1 --seeds 0
/usr/bin/time -p .venv/bin/python experiments/run_redesign_conflict.py --out results/redesign_20260907_gaussian_v1 --seeds 1 2
# Final integrated regressions (OMP/OPENBLAS/MKL/VECLIB threads all set to 1):
.venv/bin/python -m pytest -q tests/test_redesign_training.py tests/test_scoring_support.py tests/test_redesign_conflict.py tests/test_accuracy_bound.py tests/test_per_class_constraints.py tests/test_proxy_lagrangian.py tests/test_dominant_axis.py tests/test_erase_layer.py tests/test_lora.py tests/test_linear_adapter.py tests/test_folktables_fixes.py tests/test_composition.py tests/test_multiclass_impossibility.py
git diff --check
```

Final integrated result: **156 passed in 2.58 s** (3.37 s process wall time),
two upstream `torch.jit.script` deprecation warnings. Component suites also
passed (training 42; scoring 35 with 2 MLP tests initially deselected; retirement
7; expanded scoring/MLP 27). The final suite includes the MLP tests. Runner
`--help`, affected-module compilation, and diff whitespace checks passed.
The full Adult training/download integration test was intentionally not run.

Pilot: three fixed seeds, Gaussian U,V,S + five independent nuisance columns,
fixed invertible mixing, 2048/2048/2048/4096 representation-fit/attacker-fit/
validation/test rows. Continuous LEACE interface verified in installed source
and official upstream documentation. No test-based selection or tuning.

| Method | Mean P1 task R² | Mean P2 task R² | Mean worst individual leakage R² | Mean combined S R² |
|---|---:|---:|---:|---:|
| No erasure | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Shared union | -0.000483 | -0.000361 | 0.000263 | -0.001933 |
| Independent per-purpose LEACE | 0.999634 | 0.999694 | 0.000038 | -0.002490 |
| PCRL post-erasure LoRA | 0.999634 | 0.999694 | 0.000038 | -0.002490 |

Separate erasure solves the intended conflict. Negative predictive R² is
retained, not clipped into a privacy pass. PCRL ties independent LEACE
(maximum attack-score difference 2.22e-16); this restricted linear post-erasure
adapter with refitted affine heads validates wiring, not PCRL efficacy or
novelty. Full categorical V2 training was regression-tested on tiny synthetic
fixtures, not run on a real dataset or claimed as the regression pilot.

Composition: δ=.01 gives each view population R²≈.00009999 and combined R²=1;
δ=.1 gives .00990099 and 1; δ=0 gives zero for both. Exact zero-covariance
composition remains valid. Independent finite-sample and Gaussian checks pass.

Pilot runtime: seed 0 process 1.28 s; seeds 1+2 process 1.03 s; total **2.31 s**,
with **0.8264 s** inside seed computations. No pilot run was invalidated or
superseded. All new evidence is in `results/redesign_20260907_gaussian_v1/`:
`TABLE.md`, `PROTOCOL.md`, `config.json`, `mixing.json`, `seed_*/metrics.json`,
`seed_*/fitted_arrays.npz`, step-0/20 adapter checkpoints, adaptation histories,
`composition_stress.json`, execution/test logs, provenance hashes, and source
snapshot. Raw metrics include every view/signal, covariance matrices, ranks,
sample counts, target variances and split seeds. Historical artifacts unchanged.

Most informative next experiment: retain this fixed split protocol and compare
independent per-purpose LEACE with PCRL **upstream** adaptation on a small fixed
nonlinear observation map using explicitly task-pretrained starting features.
That lets adaptation change available task information; linear post-erasure
adaptation with optimal affine heads cannot add such information.

## Completed nonlinear upstream pilot (2026-09-07)

Continued on the same branch/starting commit without a PR or push. Re-read the
prior status/protocol/table and verified raw Gaussian summaries and executed
source hashes. All 33 prior-pass source files matched their saved hashes before
this status append. No repository/ancestor AGENTS.md exists. Existing code,
uncommitted work, and historical result files were preserved.

New implementation: `experiments/run_nonlinear_conflict.py`,
`experiments/nonlinear_conflict_training.py`,
`experiments/nonlinear_conflict_probes.py`,
`scripts/summarize_nonlinear_conflict.py`, and `tests/test_nonlinear_conflict.py`.

The fixed invertible nonlinear generator is X=(r+0.2r³)Q2+b, r=LQ1, for eight
independent Gaussian latents. Q1/Q2 seeds are 20260908/20260909. Five separate
splits have 4096/2048/2048/2048/4096 representation-training/calibration/attacker/
validation/test rows. Preprocessing uses designated fitting rows only. Source,
configuration, thresholds and protocol were frozen before the first real run.

Shared 8→32→32→8 GELU encoder: 1280 genuine U/V task-pretraining updates, selected
solely by validation task MSE (epoch80 for all three seeds). Direct task-head
test R² before→after pretraining:

| Seed | U before | U after | V before | V after |
|---|---:|---:|---:|---:|
| 0 | -0.073677 | 0.997564 | -0.085859 | 0.996595 |
| 1 | -0.072528 | 0.997365 | -0.027740 | 0.996927 |
| 2 | -0.033339 | 0.997494 | -0.089205 | 0.995018 |

A is frozen/no erasure; B frozen/separate LEACE; C task-only upstream LoRA then
LEACE; D matched upstream LoRA with protection strengths .1 and1 then LEACE.
C/D share identical starting checkpoint tensors, task heads, minibatch schedules,
1314 trainable parameters and400 steps. The five pre-erasure continuous ridge
R² constraints add strength/5 Σλ(R²−τ) to task MSE; τ=.05 individual/.10 combined,
projected dual ascent is reused as a heuristic. No erasure is fitted/applied
during training. Final LEACE is calibrated separately after freezing each
encoder, then independent task/attacker probes are fitted. Nonzero upstream
task/protection gradients, parameter changes and exact matching were verified.
This omits categorical V2 warm-start/VICReg/vCLUB and asserts no optimizer guarantee.

All methods receive affine probes plus 32×32 MLP probes with two initializations,
80 epochs and validation-only selection. Test data are generated only after
all fitting/selection decisions are recorded. Full-grid validation selected
D.1/D1/D1 for seeds0/1/2, each explicitly **infeasible**. No erased arm met all
five linear+MLP thresholds on any seed, in validation or final test.

Final-test mean ± sample SD across three seeds; task utility averages U and V:

| Arm | MLP task utility | Worst individual MLP leakage | Combined S MLP leakage |
|---|---:|---:|---:|
| A frozen, no erasure | .99687 ± .00046 | .99627 ± .00044 | .50619 ± .13244 |
| B frozen + LEACE | .99664 ± .00023 | .32638 ± .04239 | .15629 ± .04055 |
| C task-only + LEACE | .99700 ± .00017 | .32796 ± .06735 | .16221 ± .04645 |
| D strength .1 + LEACE | .94189 ± .05022 | .27106 ± .11993 | .10850 ± .01222 |
| D strength1 + LEACE | .90702 ± .07367 | .24533 ± .20247 | .09088 ± .00524 |

LEACE's paired MLP utility loss A−B is only .000236±.000248; C recovers
.000360±.000108 over B. Thus this benchmark has very little erasure-induced
utility headroom. D provides no utility advantage over C: D1−C is
−.089977±.073537 MLP task R². It trades away utility for lower leakage on some
targets; D1 even worsens worst individual leakage on seed0. Independent linear
attacks alone would pass all erased arms; MLP attacks reverse that conclusion.
Negative R² is not negative information; no result establishes universal privacy.

Executed and saved commands:

```sh
# With OMP/OPENBLAS/MKL/VECLIB numerical threads set to 1:
.venv/bin/python -m pytest tests/test_nonlinear_conflict.py tests/test_redesign_conflict.py tests/test_proxy_lagrangian.py -q
.venv/bin/python experiments/run_nonlinear_conflict.py --out results/redesign_20260907_nonlinear_upstream_v1 --prepare-only
/usr/bin/time -p .venv/bin/python experiments/run_nonlinear_conflict.py --out results/redesign_20260907_nonlinear_upstream_v1 --seeds 0
/usr/bin/time -p .venv/bin/python experiments/run_nonlinear_conflict.py --out results/redesign_20260907_nonlinear_upstream_v1 --seeds 1 2
.venv/bin/python scripts/summarize_nonlinear_conflict.py --out results/redesign_20260907_nonlinear_upstream_v1
```

**31 tests passed in1.35 s**, including11 new protocol regressions and a tiny
complete five-arm run testing serialization and the selection/test boundary;
two upstream torch.jit deprecation warnings. Relevant modules compile; diff
whitespace checks pass. No full-audit repetition or large dataset runs.

Seed0 measured10.87 s process wall time; recorded estimate21.74 s for seeds1/2
before expansion; actual21.36 s. Total experiment wall time **32.23 s**, total
inside-seed compute **30.563 s**, on existing Apple M4 Pro24GiB, one CPU thread.
No budget/generator changes, superseded pilot, downloads, paid resources or
interruption of other experiments.

New evidence: `results/redesign_20260907_nonlinear_upstream_v1/` contains frozen
`PROTOCOL.md`, `config.json`, generator and source snapshots, `TABLE.md` (per-seed
scores), `ANALYSIS.md` (all target means/SD and paired comparisons),
`paired_comparisons.json`, `seed_*/metrics.json`, pretraining/adaptation/probe
checkpoints, calibration maps, split IDs/hashes, initial gradient and matching
diagnostics, runtime logs and `verification.json`. Current executed source
hashes match all six frozen source files.

Next experiment: keep this generator and five-split protocol fixed; test whether
a nonlinear protection objective reduces the observed residual cross-task/S
leakage while maintaining a predeclared .99 mean task-R² floor. Compare against
the same task-only control. This addresses the demonstrated privacy gap; another
utility-rescue sweep has little headroom here. Three-seed evidence is preliminary.

## Completed nonlinear release/adversary pilot (2026-09-07)

Continued from `results/redesign_20260907_nonlinear_upstream_v1/` on the same
branch and commit15dbcc3c3338f6707e7a0b3901d738d99651a77d. Read current project
instructions, status, prior protocol/table/analysis and training/audit code.
No AGENTS.md exists in checkout/ancestors. Verified all six prior executed
source hashes and exact fitting-split/preprocessing manifests for all three
saved checkpoints. Preserved all671 files in the prior evidence directory
and all prior source edits; this status section is an append. No PR/push.

Confirmed the previous protection objective was pre-erasure differentiable
affine ridge R² with projected dual weighting; it did not train a nonlinear
adversary or apply training-time erasure. Its near-costless LEACE utility loss,
nonlinear privacy failures and lack of feasible PCRL benefit remain evidence.

Added `experiments/run_nonlinear_release.py`,
`experiments/nonlinear_release_training.py`,
`scripts/summarize_nonlinear_release.py`, and `tests/test_nonlinear_release.py`.
Fresh output: `results/redesign_20260907_nonlinear_release_v1/`.

Frozen generator, seeds0/1/2, purpose policy, task-pretrained checkpoints and
four fitting/validation streams are unchanged. New test RNG seeds are
900004/900104/900204, committed in PROTOCOL.md before testing. No old test
scores were copied; all saved references were re-erased/re-audited and evaluated
on the common new test. Primary utility now requires .99 for EACH purpose,
superseding the prior suggested mean floor. Every individual forbidden-target
threshold is .05, combined S .10, both linear and MLP audits must pass.

Controls: inaccessible oracle true-U/V release; existing saved task-head
scalar predictions; saved task-only full views+LEACE; both old protection
checkpoints as references; new matched task-only training. Scalar releases
have1+1 dimensions versus8+8 full views: success applies to the two fixed tasks,
not a claim of equal reusable downstream representations.

New matched C/D/E training: same pretrained tensors, rank4 LoRA at all three
encoder Linear layers,1314 trainable adapter/head parameters,400 Adam.001
encoder updates. Training erasers fit/refresh every25 steps on the first1024
representation-training rows; updates use the other3072. Eraser fit is
detached float64, application differentiable float32. Task heads and nonlinear
adversaries consume these erased releases. Final erasers are freshly fitted
on reserved2048-row calibration after encoder freezing for every erased arm.
This moving-eraser, stop-gradient approximation differs from the prior C.

Three32×32 ReLU adversaries predict P1(V,S),P2(U,S),combined S. Targets normalize
from update rows only. Adversaries minimize mean normalized MSE; encoder loss
is raw mean task MSE + mean λ*(1−adversary MSE−threshold). All arms have100
adversary warmup +2 updates per encoder step=900 total, same initialization,
architecture, schedules and fitting rows. D fixes λ=.01 or .1; E starts at
the matching weight w and projects λ+.02*w*violation to[0,1]. E is standard
constrained adversarial training, with no novel mechanism/optimizer guarantee.
Task-only C has λ0 but the same training/adversary budget. Dual changes are
the intended D/E difference. No exact-zero same-batch erasure statistic is
used as protection loss. Initial task/protection gradients and actual adapter
changes are nonzero; frozen base weights and all matched initial tensors
were verified, including equality to the prior C initialization checkpoint.

Independent audits reuse the previous affine and32×32 MLP protocol,80 epochs,
two validation-selected initializations/checkpoints per target, with task
probes fitted on representation training and leakage probes on attacker fit.
All configurations, references, scalar controls and exposed-U/V/S diagnostics
get the same audit effort. Selection requires both validation utility floors,
then minimizes max of all ten constraint-specific leakage/threshold ratios.
If no utility-feasible candidate exists, a clearly labeled diagnostic fallback
is saved. All fitting and selections precede fresh-test generation.

Results (mean ± sample SD, independently fitted MLP task R²):

| Release | U utility | V utility | Worst individual MLP leakage | Combined S MLP leakage | Joint pass validation/test |
|---|---:|---:|---:|---:|---|
| Oracle | .999990±.000002 | .999994±.000004 | −.000131±.000476 | −.003137±.000688 |3/3;3/3|
| Prediction-only | .997930±.000097 | .996819±.000643 | −.000304±.000561 | −.001906±.001522 |3/3;3/3|
| Saved task-only full+LEACE | .997570±.000230 | .996245±.000820 | .350252±.062275 | .156921±.061658 |0/3;0/3|
| New matched task-only full+LEACE | .990291±.007611 | .990237±.004300 | .331666±.024183 | .147321±.106316 |0/3;0/3|
| Saved selection D* | .991865±.003484 | .992330±.000208 | .230895±.064414 | .102306±.080658 |0/3;0/3|
| Saved selection E* | .990760±.001345 | .992254±.000265 | .203319±.034479 | .107760±.043096 |0/3;0/3|

*D/E select weight.01 on seeds0/2. On seed1 neither configuration meets both
validation task floors; the weight.1 rows above are diagnostic fallbacks,
not utility-feasible primary results. These all-three-seed descriptive means
must not disguise that failure. New C meets both validation floors only on
seed1. On fresh test, D selected meets utility on seeds0/2, E only seed0:
E's U scores on seeds1/2 are .989998/.989969, strictly below .99.

Native prediction-only direct U/V R² is .997769±.000124/.996770±.000689.
Thus the simpler scalar release already solves these two fixed tasks under
the declared finite audits in all three seeds. Oracle finite-sample leakage
is near zero with small signed fluctuations; no positive-information meaning
is assigned to negative predictive R². Exposed-target MLP audits recover
forbidden targets with test R² .998817–.999550; affine recovery is1.

Nonlinear protection reduces some individual leakage while passing both
utility floors on some seeds, but no full representation meets every target.
There is no feasible PCRL advantage. All-three-seed paired E−D differences,
including the explicitly diagnostic seed1 pair: U−.001105±.002408,
V−.000076±.000075, worst individual MLP−.027575±.030165,
combined S MLP+.005454±.056628. These are mixed tradeoffs, not dominance at
matched leakage or consistent per-purpose utility. Per-seed/all-config/linear
and target-specific values and paired differences are in TABLE/ANALYSIS/JSON.

Limitations observed: the new moving-eraser task-only procedure itself loses
utility relative to the saved prior C. Training adversaries can underestimate
leakage substantially: seed2 E.01's maximum validation transfer R² is .04702,
versus .25249 for its independently trained MLP audits. Warmup reduces attack
MSE and gradients are active, so this is neither an unused penalty nor proof
of protection. No numerical instability or invalid/superseded pilot occurred.
Three-seed empirical audit passes are not universal privacy guarantees.

Executed commands (numerical threads1; full commands frozen in PROTOCOL.md):

```sh
.venv/bin/python -m pytest tests/test_nonlinear_release.py tests/test_nonlinear_conflict.py tests/test_redesign_conflict.py tests/test_proxy_lagrangian.py -q
.venv/bin/python experiments/run_nonlinear_release.py --out results/redesign_20260907_nonlinear_release_v1 --prepare-only
/usr/bin/time -p .venv/bin/python experiments/run_nonlinear_release.py --out results/redesign_20260907_nonlinear_release_v1 --seeds 0
/usr/bin/time -p .venv/bin/python experiments/run_nonlinear_release.py --out results/redesign_20260907_nonlinear_release_v1 --seeds 1 2
.venv/bin/python scripts/summarize_nonlinear_release.py --out results/redesign_20260907_nonlinear_release_v1
```

**44 tests passed in1.57s**, including13 new release tests, a complete tiny
ten-arm fixture, matching, loss/dual signs, detached erasure with nonzero
upstream gradients, designated fitting pools, and sealed fresh-test selection.
Two existing torch.jit deprecation warnings; compilation/diff checks pass.
Seed0 measured25.45s process wall; saved remaining estimate50.90s before
expansion; actual seeds1/2 took51.53s. Total **76.98s wall**, **75.114s** inside
seed computations on Apple M4 Pro24GiB, one CPU thread. No budget change,
remote resources, model downloads or interruption of other experiments.

Evidence includes PROTOCOL/TABLE/ANALYSIS, raw seed/target metrics, selections,
config/generator/test seeds, checkpoint/source hashes and snapshots, all
checkpoints/probes/erasers/training schedules, gradient/adversary/dual histories,
runtime logs, hardware.json, preservation hashes and verification.json.

Single next experiment: keep these encoder/final-eraser checkpoints frozen
and run an adversary catch-up diagnostic with the independent audit budget.
Measure whether training adversaries recover the already-demonstrated leakage
once the release stops moving. This tests the observed training-adversary gap
before adding more encoder optimization; the successful scalar control remains
the simpler solution for the present fixed tasks.

## Completed frozen-release adversary diagnostic (2026-09-07)

Continued from results/redesign_20260907_nonlinear_release_v1 on unchanged
branch/commit15dbcc3c3338f6707e7a0b3901d738d99651a77d. Read repository
instructions, status, previous protocol/table/analysis, saved training/audit
checkpoints and relevant source. No AGENTS.md in checkout/ancestors. Preserved
all1661 prior release evidence files and existing source edits. No PR/push,
encoder/head updates, final eraser refits, generator change or method sweep.

New code: experiments/run_frozen_release_diagnostic.py,
experiments/frozen_release_adversaries.py,
scripts/summarize_frozen_release_diagnostic.py,
tests/test_frozen_release_diagnostic.py. Fresh evidence directory:
results/redesign_20260907_frozen_adversary_v1/.

The original seed2 E.01 discrepancy is the SAME P2→U target on the SAME2048
validation examples and final calibrated release: saved training adversary
R²=.047024004933239, independent MLP=.252486578007161. All five original
training scores and audit scores reproduce (max float64 audit roundoff9e-16).
No scorer/target alignment bug. INSPECTION.md/json records exact checkpoint
identities, normalization arrays, data/exposure, architecture and objectives.
Training predicts2/2/1 prohibited outputs, audit3/3/3 U,V,S; both32×32ReLU.
Training had900 updates on3072 representation-training rows with changing
maps and no validation checkpoint selection; D had two640-update trajectories
on2048 fixed attacker-fit rows and per-target validation selection. Input and
target normalization fit different designated pools. The last50 adversary
updates used map375; encoder400 and map/statistics refresh400 followed, with
no adversary fitting on that refreshed or final-calibrated map.

Diagnosed exact selected D/E checkpoints in all3seeds. Seeds0/2 weight.01
remain validation utility-eligible/protection-infeasible; seed1 weight.1
remains utility-infeasible diagnostic fallback. All encoders/heads/erasers
frozen in eval mode. Individual+combined final releases cached and hashed;
all parameter/buffer/map and representative-output checks unchanged. Never
refit a final eraser; reconstruct its saved dense affine map in float64.

A is the saved adversary without fitting. B continues its exact weights;
C starts fresh with identical architecture. Both retain the ORIGINAL saved
input/output coordinates, reset Adam, and use the same two batch trajectories
on attacker_fit,80epochs/640updates each,1280total optimizer updates and327680
row presentations per method. Same original mean-five normalized MSE,
Adam.001/batch256, per-target minraw validation MSE at0/every40updates. B retains
its prior900-update exposure; only ADDED fitting is matched to C. D reloads
exact original selected independent auditors/preprocessing. Re-evaluate all
A/B/C/D on fresh common diagnostic tests with predeclared RNG1000004/1000104/
1000204 and4096rows. Save all selections before generating each seed's test.
No fitting afterward. Prior validation reuse is exploratory, not independent
confirmation of the whole research hypothesis. Optional seed2 preprocessing
bridge (D−max(B,C)>.05 validation) did not trigger; no bridge or budget expansion.

Fresh diagnostic test P2→U R², every selected/fallback checkpoint:

| Seed | Frozen release | A saved | B continued | C fresh | D original audit |
|---|---|---:|---:|---:|---:|
|0|D .01|.184283|.322835|.244539|.178969|
|0|E .01|.178296|.295546|.235011|.161485|
|1|D .1 diagnostic|.014562|.190674|.182234|.119567|
|1|E .1 diagnostic|.003978|.205868|.185437|.116696|
|2|D .01|.069398|.318425|.241836|.230533|
|2|E .01|.073824|.274720|.234644|.209296|

Across all30identical-target/checkpoint pairs, B improves over A30/30, by
.041983–.418441 testR²; B exceeds C28/30 and D30/30. C exceeds D29/30.
Seed2 E.01 P2U validation A/B/C/D=.047024/.287370/.271926/.252487;
its combined-S fresh test A/B/C/D=−.022268/.340394/.282557/.114252.
Thus inherited adversaries recover more leakage with added frozen-release
fitting; fresh initialization does not help more. Original architecture,
loss and saved preprocessing can exceed independent audit leakage, so a
capacity/coordinate change is not necessary to recover the original miss.

Eraser sensitivity is real but not a unique causal attribution. Holding final
encoder+adversary fixed, seed2E.01 P2U validation is.192596 with375map/stats,
.069501 with400map/stats, .047024 withfinalmap/400stats. Finalmap/375stats gives
.052381: preserving earlier standardization alone barely helps. Old maps
were fitted to earlier encoders and change released information. Catch-up
also changes fitting pool, resets Adam, adds updates and selects validation
checkpoints; success does not prove representation movement caused the miss.
No source repair was justified by a scorer/serialization error; the reported
post-refresh deployment mismatch remains explicit historical evidence.

Oracle MLP fresh-test forbidden scores range−.004473 to.000996; exposed U,V,S
MLP recoveries .998823–.999535, including S. No unexpected control failure.
Negative R² is not negative information; finite audit success is no universal
privacy guarantee. Better attacks do not improve protection: release states
never change. The broader prediction-only success/full-representation failure
remains; no PCRL novelty or need for reusable representations is established.

Commands (numerical threads1):

```sh
.venv/bin/python -m pytest tests/test_frozen_release_diagnostic.py tests/test_nonlinear_release.py -q
.venv/bin/python experiments/run_frozen_release_diagnostic.py --out results/redesign_20260907_frozen_adversary_v1 --prepare-only
/usr/bin/time -p .venv/bin/python experiments/run_frozen_release_diagnostic.py --out results/redesign_20260907_frozen_adversary_v1 --seeds 2
/usr/bin/time -p .venv/bin/python experiments/run_frozen_release_diagnostic.py --out results/redesign_20260907_frozen_adversary_v1 --seeds 0 1
.venv/bin/python scripts/summarize_frozen_release_diagnostic.py --out results/redesign_20260907_frozen_adversary_v1
```

Six new focused regressions plus13 prior release regressions:19passed
(regression_tests_final.log); target alignment, exact saved preprocessing,
weight clones, reset/matched budgets, validation selection, BN/dropout and
map integrity, and sealed nonpilot test isolation. Two existing torch.jit
warnings; compilation and diff checks pass. Actual execution: seed2 wall4.16s,
recorded remaining estimate8.32s, actual seeds0/1 wall7.23s; total11.39s wall,
9.180s internal. Existing Apple M4 Pro24GiB, one CPU thread. No superseded
experiment, budget change, paid resource or process interruption.

Evidence: PROTOCOL.md, TABLE.md (every target/per-seed gain and remaining audit
gap, means/sampleSD), ANALYSIS.md, INSPECTION.md/json, paired_target_metrics.json,
learning_curves.png/pdf, seed_*/metrics.json, selected weights and original
coordinates, all actual schedules/learning curves, caches/IDs, selections,
hardware/runtime/reproduction logs, source snapshots and verification.json.

Next action: add a mandatory frozen-final-release catch-up audit before future
protection decisions, reusing saved coordinates and per-target validation
selection. Validate that evaluation change on existing checkpoints before
considering any encoder retraining. Retain scalar prediction release as the
simpler solution for these fixed tasks.

## Completed stronger prediction audit and research/application decision (2026-09-07)

Continued on unchanged branch/commit15dbcc3c3338f6707e7a0b3901d738d99651a77d.
Read current instructions, this status, all four preceding redesign protocols,
available analyses/tables and relevant release/training/audit sources. No
AGENTS.md exists in checkout/ancestors. The working tree was inventoried before
editing. All2662 historical result files and existing source changes are
preserved; no PR, push, new dependency, data/model download or paid resources.

**Implementation/evaluation work:** added
experiments/prediction_release_attackers.py, experiments/run_prediction_audit.py,
scripts/summarize_prediction_audit.py and eight targeted tests. Fresh scalar
attackers use exact saved native prediction outputs (1/1/2 dimensions), original
preprocessing and frozen eval-mode encoders/heads. No representation training or
eraser refit. MLP32x32ReLU, two1800-step Adam trajectories per view,256 rows/update,
460800 presentations/trajectory; fixed200-iteration HistGradientBoosting plus
affine OLS. The same budget audits oracle/exposed controls and one known-leaky
seed2E.01 full release. Compatible original/continued full adversaries are also
replayed in saved coordinates, never transferred to scalar-width inputs.

Fresh test RNG1100004/1100104/1100204 was committed in the protocol before fitting;
test arrays are generated only after saved validation selections. No fitting
receives test/calibration examples. Independent task probes fit representation
training data. All forbidden targets/families are reported separately with raw
MSE, predictive R², target variance and support counts. Negative R² is retained.
The prior final-map mismatch cannot substitute for an audit: these attackers
fit the exact immutable final release. No new core-method repair was necessary.

**Established observations:** prediction-only passes both .99 task floors and
all15 leakage inequalities (.05 individual/.10 combined, three attack families)
for each of three seeds on validation and fresh test. Native direct predictions
also meet both utility floors on every seed. Test mean±sampleSD:

| Metric | Prediction-only |
|---|---:|
| MLP task U | .998035 ± .000100 |
| MLP task V | .996853 ± .000832 |
| Direct native U | .997853 ± .000101 |
| Direct native V | .996745 ± .000874 |
| Worst individual affine leakage | −.000576 ± .000328 |
| Worst individual MLP leakage | −.000203 ± .000619 |
| Worst individual tree leakage | −.032221 ± .007508 |
| Combined S affine leakage | −.002246 ± .001531 |
| Combined S MLP leakage | −.003586 ± .001822 |
| Combined S tree leakage | −.063554 ± .017054 |

The unchanged full E.01 seed2 release still fails: fresh MLP P1→V .349013,
P1→S .149281,P2→U .296702,P2→S .185202,combinedS .352950; trees recoverP2→U
.195354 and combinedS .125327. Oracle MLP scores range−.006001 to .000325;
exposed-target minima across all targets includingS are affine1,MLP .999811,
trees .996258. Controls show competence at the declared budgets. Full per-seed,
per-target, validation/test results and paired prediction-minus-oracle differences
are saved. Prediction-only remains empirically sufficient for these fixed tasks.

**Application evidence and decision:** investigated exactly HAR, Diabetes and
ACS using local data, source and primary references. No real-data representation
pilot was admitted: none supplies both a coherent independently motivated
reusable-release interface and meaningful reserved task family. This is not a
proof that richer releases are never useful or a demand for a signed deployment
contract. Existing prediction-bank/trusted-service alternatives remain adequate
or unexcluded for the stated tasks. ACS custom research is the strongest lead,
but its concrete allowed query/task family is missing.

Added experiments/screen_pcrl_applications.py and seven focused tests. Earlier
oracle-label checks are explicitly retrospective and reproduced. A fixed learned
HAR reference was specified before execution: task-fit/attacker/development
participants13/4/4,4697/1361/1294 windows, official train only; StandardScaler
fits task rows only. Binary active/sedentary task accuracy .999227, while actual
hard output yields forbidden six-way activity accuracy .363215 versus prior
.198609 and log-loss gain .681384nats. This diagnoses an incoherent absolute
activity restriction, not a PCRL training failure. UCI-supplied feature
normalization was not reconstructed; the separate exact-label contradiction
does not rely on features. Diabetes task-label/demographic associations are
reported as development evidence, with no medical or universal privacy claim.

**Prior work and unresolved hypotheses:** docs/PCRL_PRIOR_WORK.md checks the
five requested starting papers and close adversarial/transfer/wearable work
using primary papers and official repositories where available. Task-aware
erasure, configurable policies, held-out task transfer and fresh-attacker failure
diagnostics have close precedents. Conditional maximal-leakage assumptions and
composition bounds are distinguished from finite predictive audits. The old
universal classification certificate remains retired. The remaining question is
whether a justified reusable interface creates a measurable problem beyond a
competitive prediction bank and whether a specific mechanism improves it under
strong independent final-release attacks. No real-data F/G difference, useful
transfer gain or new coalition advantage is established here.

**Unsupported claims:** finite attack failure is not universal privacy; scalar
success is not future-task sufficiency; equal update counts across different
interfaces/exposure histories are not equal attack strength; a projected dual
optimizer, LoRA, multiple purposes, implementation size and test counts are not
research novelty. Three seeds support preliminary descriptive comparisons only.

**Validation/runtime:** eight audit tests passed2.39s; seven application tests
passed1.05s. One initial application test failed due to a platform/subprocess
mock; only that test fixture was corrected. No experimental run was invalidated.
Independent review checked100MLP trajectories/70selected checkpoints, actual
exposure schedules and30split-ID pairs, and exactly replayed380scores. Scalar
checkpoint/validation replay differences are0. All12executed audit source hashes
and2662historical files remain unchanged. Compilation and diff checks passed.

First audit seed2 took21.01s process wall,19.626s internally including full
reference. Recorded remaining estimate42.02s before expansion; actual31.23s.
Audit total52.24s wall,48.777s internal. Development screen5.073s internal,
6.074s command wall, including loading; learned task+attacks .06777s. Existing
Apple M4 Pro14cores24GiB, one CPU numerical thread. No common budget reduction,
method sweep, multi-day work or interruption of other jobs.

Fresh evidence: results/redesign_20260907_prediction_audit_v1/ and
results/redesign_20260907_application_screen_v1/. Both contain protocols,
tables/analyses, raw metrics, source/configuration snapshots, split/checkpoint
identities, logs, runtimes and reproduction commands. The audit additionally
saves all attacker checkpoints/schedules/learning curves, selection-before-test
records, release caches, summary/paired JSON, audit_scores.png/pdf and independent
review. Research/application documents: PCRL_NEXT_STAGE_RESEARCH_REPORT.md,
docs/PCRL_APPLICATION_SELECTION.md, docs/PCRL_PRIOR_WORK.md.

Commands actually executed, with numerical-library threads set1:

```sh
.venv/bin/python -m pytest tests/test_prediction_release_attackers.py tests/test_prediction_audit.py -q
.venv/bin/python experiments/run_prediction_audit.py --out results/redesign_20260907_prediction_audit_v1 --prepare-only
/usr/bin/time -p .venv/bin/python experiments/run_prediction_audit.py --out results/redesign_20260907_prediction_audit_v1 --seeds 2
/usr/bin/time -p .venv/bin/python experiments/run_prediction_audit.py --out results/redesign_20260907_prediction_audit_v1 --seeds 0 1
.venv/bin/python scripts/summarize_prediction_audit.py --out results/redesign_20260907_prediction_audit_v1
.venv/bin/python experiments/screen_pcrl_applications.py --out-dir results/redesign_20260907_application_screen_v1
.venv/bin/python -m pytest -q tests/test_application_screen.py
```

**Research recommendation: redefine the problem.** Close the original toy
benchmark with its empirically successful scalar release. Do not claim a method
or broader security-paper contribution from this evidence. Single next experiment,
after defining one coherent ACS recipient/task/query policy: compare a competitive
prediction bank with frozen features on genuine reserved authorized tasks and
task-output leakage using person/household-aware example splits, before training
PCRL. A useful transfer gain that simple outputs/trusted computation cannot
deliver would most change this decision; the required application specification
is still missing.


## Review publication packaging (2026-09-07)

The user explicitly authorized committing and pushing the completed redesign,
superseding the earlier no-push instructions. Destination verified as
`https://github.com/Bostesa/PCRL.git`; authenticated GitHub access is available.
Remote state was fetched before packaging. Current research branch
`ablations-facct-2026-07-24` has no existing remote counterpart; publication can
create it without changing main or overwriting history. No PR is requested.

The review entry point is [docs/PCRL_REVIEW_INDEX.md](docs/PCRL_REVIEW_INDEX.md).
All 27 pre-publication tracked repairs matched the first-pass source manifest;
all four later frozen runner/dependency manifests matched the executable source.
The complete implementation/helper modules, tests, main reports and compact
original evidence are packaged. The README adds a current review notice; this
append records publication rather than revising any experimental result.

[Publication notes](docs/PCRL_PUBLICATION.md) distinguish committed evidence,
local-only artifacts and fresh-fitting instructions. The manifest in
`results/redesign_publication_20260907/artifact_manifest.json` covers all 3,570
original files: 250 original evidence files are included verbatim; 3,320 files
remain local, including checkpoints, erasers, cached arrays/schedules, duplicate
source archives and redundant metadata. No public checkpoint archive is claimed.
Original provenance hashes and starting commits are not replaced by the new
publication commit. Unrelated pre-existing reviews/results remain unstaged.

Packaging validation is recorded separately in
[VALIDATION.md](results/redesign_publication_20260907/VALIDATION.md). Completed
research validation is reused; no research program or ACS experiment is rerun.
Remote-commit and authenticated report-access verification occurs after the
commit/push and is reported with the final publication URLs.

Established observations and the research decision remain unchanged:
prediction-only release passed the strengthened fixed-task audits; full
representations remained leaky; no PCRL method advantage or universal privacy
guarantee has been established. No real-data method pilot was justified.

## Dated appendix — 2026-09-07: ACS task-identity transfer

Starting commit: `5f162ab37cc9e1caa391e6c964711fa85cc86a14`; branch
`ablations-facct-2026-07-24`, remote `https://github.com/Bostesa/PCRL.git`.
No AGENTS.md exists in this checkout or its ancestors. Original tracked work
was clean; unrelated untracked work and all historical experiment evidence were
preserved. The user authorized this experiment and committing/pushing compact
results to the research branch; main, force-pushes and PRs remain excluded.

**Framing amendment.** The original screen remains intact. A defined one-time
release with identical downstream learning access for banks/features is enough
to investigate transfer. Trusted services remain deployment alternatives, not
a requirement to disprove before running a scientific benchmark. Public ACS
is a simulation dataset, with no confidentiality claim against public linkage.

**Implementation.** Added a separate raw2018 ACS loader with19–34/PWGTP>0 cohort,
whole-household sampling/splits, missing-label masks, ten-field allowlist, and
fit-only numeric/category preprocessing. Excluded all task fields, direct answer
families, SEX/RAC1P recodes, fertility/migration applicability aliases and COW.
No historical loader or output was overwritten. Source-only64/64/32 encoder and
independent tree banks use income/ESR/PUBCOV distributions plus a supported
eight-state joint bank. Reserved MIG/JWMNP labels enter only post-freeze heads.
Seven releases include the simple probability bank, rich neural/tree banks,
features, matched-storage feature compression, PCA and full allowed covariates.
Independent logistic/MLP transfer heads and logistic/MLP/tree attribute audits
use fixed schema2/9, separate fitting/validation pools and saved selection before
testing. Person-weighted sensitivity reuses unchanged selected predictions.

**Established observations.** Across three seeds on the fixed30,000-person cohort,
D minus B residence log-loss differences are −.004596/−.014570/−.016133 nats;
D minus C differences −.012076/−.015786/−.027920. Mean D residence loss is
.498762±.006353; PCA is .487411±.005595 and full inputs .490646±.006126.
Feature compression to26 dimensions retains .498699±.007746. Commute D is
.681786±.003885, neural bank .681481±.002771, tree bank .681863±.004760:
a practical tie under a weak task/reference result. Attribute information
remains recoverable; D SEX AUROC .631907±.010057, RAC1P loss1.228515±.032005
versus prior1.290043±.025603. Rare race categories lack support in some pools;
undefined macro metrics are explicit rather than averaged away.

**Validation and execution.**30 focused tests passed2.26s. An artificial full
pipeline check completed1.249s without ACS model outcomes; schema preparation
1.053s. Complete seed0 process wall53.266s; remaining estimate106.532s saved
before expansion; actual remaining105.348s. Total158.614s, CPU Apple M4 Pro,
24GiB, one numerical thread. Frozen-state/maps/caches/selection checks passed.
No run was invalidated, grid expanded, task replaced, package/data downloaded,
paid resource provisioned or other process interrupted. Exact commands and
independent replay records are in the new directory.

**Research decision and limits.** A reserved residential task benefits from
retaining information beyond the competitive banks. PCA's stronger result and
weak commute transfer argue against attributing the gain to the learned encoder.
Proceed only to a bounded baseline-led protection-feasibility study with explicit
purpose and output-aware criteria; no PCRL, erasure or adversarial encoder was
trained here. No protection, coalition, universal future-task sufficiency, survey
uncertainty or novelty claim follows. The earlier toy evidence is unchanged.

Artifacts: [decision](results/redesign_20260907_acs_transfer_v1/RESEARCH_DECISION.md),
[protocol](results/redesign_20260907_acs_transfer_v1/PROTOCOL.md),
[table](results/redesign_20260907_acs_transfer_v1/TABLE.md),
[analysis](results/redesign_20260907_acs_transfer_v1/ANALYSIS.md),
[reproduction and omissions](results/redesign_20260907_acs_transfer_v1/REPRODUCTION.md).

## Dated addition — 2026-09-08: bounded ACS protection feasibility

Starting commit `e765ced246be0f1c8d5bf8a131ccae5e576e9bf2`, branch
`ablations-facct-2026-07-24`, destination Bostesa/PCRL. No tracked subsequent
changes were present. All historical evidence, original source hashes and
unrelated local work were preserved. The user authorized this run and a normal
commit/push to the research branch; no PR, main change or force-push.

**Implementation, distinct from research evidence.** Added a small runner reusing
verified original ACS models/caches/splits; fixed-schema joint SEX2/RAC1P9 LEACE
on representation-fitting complete cases, fixed numerical rank stabilization,
affine mean preservation and immutable map/output hashes. No source retraining
or reserved-task labels in release fitting. Ten primary arms plus full-input,
prior and exposed controls receive matched downstream heads for five binary
tasks. Independent auditors use logistic, two120-epoch MLPs and150-iteration
HistGB with minimum leaf20/5. All selections saved before within-run evaluation;
serialized predictions support reporting/replay without refitting. A saved-tree
filename issue was fixed before freeze, not after an invalid scientific run.

**Established development observations.** Reused original test households are
DEVELOPMENT EVALUATION. PCA residence loss .487411±.005595 became .496275±.004211;
compressed features .498699±.007746 became .509683±.004454. Half of positive
residential headroom survived in2/3 PCA development seeds and1/3 validation seeds;
compressed features0/3 on both. Every parent/seed failed preservation of all
three source tasks within.01 nats. All15 maps met raw covariance tolerances,
but nonlinear attribute recovery remained. PCA mean SEX attack loss rose from
.655218 to.678120 and race loss from1.200399 to1.235046, versus priors.692687 and
1.290043. None of96 feature/bank seed/split comparisons meets the joint numeric
.01 utility/.005 attribute margins; one misses by only.00004064 nats, so this
does not establish dominance or impossibility. Rare-category coverage prevents
an all-race/all-attribute policy assessment in every seed. Commute remained weak.

**Execution and validation.** Protocol/config/execution closure frozen before
new fitting.30 focused tests passed4.66s; artificial pipeline2.696s. Seed0 wall
58.022s, remaining estimate116.045s, actual115.312s; total173.334s, M4 Pro CPU,
24GiB, one numerical thread. No budget/target/map change, paid resource, new
download, source regeneration or unrelated-job interruption. Independent replay
passed105 bitwise map applications,1,455 new binary hashes,711 candidate records
and2,844 score sets (maximum discrepancy6.66e−16).234 original parent task metric
dictionaries matched exactly. Two additional reporting regressions later passed,
bringing coverage to32 distinct targeted tests. All run/correction/verification
records are retained.

**Unresolved hypothesis and unsupported claims.** PCA is the strongest simple
residential-transfer parent, but the fully specified utility/recoverability
policy was not met. It remains unresolved whether a standard utility-aware
nonlinear protection baseline can improve this tradeoff without spending source
utility. No purpose conditioning, coalition protection, PCRL efficacy, novel
algorithm, universal privacy, independent confirmation or population inference
has been established. The earlier synthetic and ACS results remain historical.

Evidence: [decision](results/redesign_20260908_acs_protection_v1/RESEARCH_DECISION.md),
[protocol](results/redesign_20260908_acs_protection_v1/PROTOCOL.md),
[table](results/redesign_20260908_acs_protection_v1/TABLE.md),
[analysis](results/redesign_20260908_acs_protection_v1/ANALYSIS.md),
[support](results/redesign_20260908_acs_protection_v1/SUPPORT.md),
[verification](results/redesign_20260908_acs_protection_v1/VALIDATION.md),
[reproduction](results/redesign_20260908_acs_protection_v1/REPRODUCTION.md).
Publish source, tests and compact evidence; keep raw records, fitted maps/models,
caches and duplicate metadata local with hashes. No next experiment was launched.

## Dated addition — 2026-09-08: fixed nonlinear ACS bottleneck

Starting commit `948169361c38fa5d37657fd45c5ab45c84f1fef5`; correct research branch
`ablations-facct-2026-07-24`, destination Bostesa/PCRL. Tracked checkout was clean;
all historical results, subsequent/unrelated untracked objects and original
source hashes were preserved. No AGENTS.md exists in checkout or ancestors.
Current user authorized one fixed pilot plus normal commit/push, no PR/main/force.

**Implementation, separate from findings.** Reused exact original32D PCA and
saved LEACE/banks without refitting. Added strict source-only32→64ReLU→16 mapper,
three binary source heads and reconstruction decoder;60 common base epochs,
20 common adversary warmup, exact tensor and Adam clones,80 matched continuation
with3 detached adversary steps per mapper step. C has observer adversaries;
D adds−.1 mean(attributeCE/fittingpriorentropy). Fixed final iterates, trueepoch0,
phase/gradient/counter/label boundaries and immutable final releases recorded.
Fresh final audits plus120-epoch direct-coordinate catch-up use separate attacker
pools and validation-only choices. No new eraser, LoRA, purpose conditioning,
coalition, data, task search or claimed novel mechanism. Pre-run packaging of
merged candidate selections and inherited-exposure metadata was corrected before
freeze; no scientific run was invalidated or silently replaced.

**Established development observations.** C/D preserve all three source tasks
relative to original PCA+.01nats in all three seeds, on both splits. Primary
D−C development loss differences: residence+.001298±.002680, SEXattack+.010078
±.004332, raceattack+.002580±.001821. Matched-independent race difference is
+.012064±.009089; catch-up largely removes that apparent benefit. Both retain
half original PCA residential headroom in1/3 development seeds and0/3 validation
seeds. No primary feature-versus-bank numerical margin comparison passes(0/48).
No primary race-gain halving passes; race code4 lacks fitting/validation support,
so no complete all-attribute protection assessment is possible. Saved training
adversaries understate recoverability: all12catch-ups improve on them, but only
2/12 beat the selected fresh auditor on development. All candidates are reported.

**Execution and verification.** Configuration and execution closure frozen before
scientific fitting.23 focused tests passed2.17s; artificial pipeline.862s. Seed0
process30.652s, remaining estimate61.305s versusactual58.078s:88.731s total, M4Pro
CPU,24GiB, one numerical thread. No budget change, paid resource, download or
unrelated job interruption. Independent state replay passed42 bitwise outputs,
321 binary hashes, exact forks and saved Adam counters/schedules/exposures.
Independent score replay passed555 candidate records,2,220 score sets and288
model prediction sets(maxerror6.66e−16); all411 historical reference rows identical.
Implementation/analysis/publication wall is separately reported from experiment
execution. No model was refitted for tables or verification.

**Unresolved hypotheses and unsupported claims.** D shows a modest SEX tradeoff,
not a tie; that is insufficient overall feasibility. C's shared residential loss
means compression versus source-focused training remains unresolved. Reconstruction
can conflict with protection and guarantees neither transfer nor privacy. Current
evidence does not justify PCRL-specific mechanisms, novelty, complete race
protection, untouched confirmation, arbitrary-future-task or survey inference.
The proposed fixed16-coordinate PCA control has not been run; no extra search
was launched after outcomes.

Evidence: [decision](results/redesign_20260908_acs_bottleneck_v1/RESEARCH_DECISION.md),
[protocol](results/redesign_20260908_acs_bottleneck_v1/PROTOCOL.md),
[table](results/redesign_20260908_acs_bottleneck_v1/TABLE.md),
[analysis](results/redesign_20260908_acs_bottleneck_v1/ANALYSIS.md),
[validation](results/redesign_20260908_acs_bottleneck_v1/VALIDATION.md),
[reproduction](results/redesign_20260908_acs_bottleneck_v1/REPRODUCTION.md).

## Dated addition — 2026-09-08 UTC: fixed PCA16 diagnostic

Starting commit `96509680b5c4906a692249b18cc15463da8db9aa`, branch
`ablations-facct-2026-07-24`, remote Bostesa/PCRL. Tracked checkout was clean;
historical evidence, local fitted objects and unrelated untracked work were
preserved. No applicable AGENTS.md was present in the checkout or ancestors.
The user authorized this diagnostic and normal commit/push, without PR/main/force.

**Implementation, separate from observations.** Added one compact runner using
the exact first 16 columns of each saved PCA32 interface before fitting-specific
standardization. No representation was fitted, no components selected, no map
whitened/rotated, no eraser or training adversary added. Original PCA component
hashes/order and cached releases replayed exactly. Existing cohort, household
pools, masks, 2048-label utility rows and attacker rows remain unchanged. Fresh
PCA16 heads/auditors use the previous recipes. Primary audit reporting now uses
the matched five independent candidates for all releases; historical C/D
catch-up-inclusive scores stay separately identified. All reference predictions
and metrics are reused with original hashes and no regeneration.

**Established development observations.** Mean residence log loss PCA16
.495062±.002095, PCA32 .487411±.005595, C16 .500295±.003986, D16 .501592±.002660.
PCA16−C16 is −.005232±.005723 and PCA16−D16 −.006530±.003566 nats. Under person
weighting these shrink to −.000523±.009670 and −.001563±.006129. PCA16 meets the
original-PCA+.01 source reference for every task/seed/split; half residential
headroom holds in3/3 development and2/3 validation seeds. Primary PCA16 SEX/race
attack gains are .032739/.077966 nats; race recovery exceeds C/D on average.
Incomplete race support and original exposed-control failures persist. Equal
stored dimensions alone do not account for the measured C16 loss, but this does
not isolate objectives, optimization or geometry as its unique cause.

**Execution and checks.** Protocol/configuration and actual execution closure
frozen before fitting. Ten focused tests passed2.62s, including an artificial
20-candidate pipeline. Complete seed0 took7.324s and projected21.973s for three;
actual total19.663s (including setup, heads/auditors, scoring and process startup),
M4 Pro CPU, one numerical thread. No budget changes or historical model reruns.
New-only replay checked60 candidates/240 score sets (maximumerror2.22e−16),120
bitwise model predictions,21 exact slices,60 standardizers,21 primary/48 family
choices,27 MLP schedules and57 referenced hashes. No scientific run was invalidated.
Runtime versus total implementation/analysis/publication is recorded separately.

**Unresolved and unsupported.** PCA16 is a stronger compact transfer reference,
not a uniformly dominating release, a privacy guarantee or a PCRL advance.
PCA32 still transfers better; learned C/D favor source utility and race
recoverability; weighted sensitivity limits the headline improvement. Existing
evaluation households remain DEVELOPMENT EVALUATION. The one proposed next
change is initializing the same C/D mapper to reproduce PCA16 exactly before
otherwise unchanged matched training; that comparison has not been run.

Evidence: [decision](results/redesign_20260908_acs_pca16_v1/RESEARCH_DECISION.md),
[protocol](results/redesign_20260908_acs_pca16_v1/PROTOCOL.md),
[table](results/redesign_20260908_acs_pca16_v1/TABLE.md),
[paired analysis](results/redesign_20260908_acs_pca16_v1/ANALYSIS.md),
[validation](results/redesign_20260908_acs_pca16_v1/VALIDATION.md),
[reproduction](results/redesign_20260908_acs_pca16_v1/REPRODUCTION.md).

## Dated addition — 2026-09-08 UTC: PCA16-initialized matched C/D

Starting commit `de9a7e499c30802e320fec7ac01f0d6f72af8a48` on
`ablations-facct-2026-07-24`, Bostesa/PCRL. No subsequent tracked changes or
applicable AGENTS.md were present. Existing unrelated untracked work, fitted
objects, raw data and all historical result bytes were preserved. The user
explicitly authorized normal commit/push to this branch, without PR/main/force.

**Implementation and checks, distinct from research.** Added optional analytical
PCA16 mapper initialization and frozen I/W snapshots to the existing training
and runner. Original random defaults retain identical artificial numerical
checkpoints/Adam states. Actual new initialization preserves original source-head,
decoder, preprocessing and adversary RNG identities; all32 signed hidden pairs
remain trainable. Maximum PCA16 parity error4.77e-7 versus fixed1e-5 tolerances,
with nonzero unused-readout gradients. Exact C/D forks,60/20/80 epoch schedules,
example batches, objectives and final-state selection are unchanged. All training
finishes before snapshot utility fitting; only final C/D receive new audits and
catch-up. Historical PCA32/PCA16/LEACE/C/D/banks/controls are verified references.

**Established development observations.** Residence I/W/C_init/D_init means are
.495062/.498230/.499489/.501559. W−I is+.003168±.002616 unweighted, but
−.002566±.004596 weighted; the mean warmup loss is not robust to PWGTP. Final
C_init−oldC is−.000805±.002869, D_init−oldD−.000033±.000492. All three source
margins pass every stage/seed/split; final residential half-headroom remains1/3
development seeds. C_init leaks more SEX/race than oldC on the primary means;
D_init leaks more SEX and practically ties oldD's race/residence. Independent
D−C race attack-loss gain .022542 becomes .010327 with inclusive catch-up and
reverses in seed0. No complete policy pass, no PCRL method advantage.

**Execution/evidence.** Frozen protocol/configuration/execution dependencies
preceded scientific fitting.18 focused checks plus a26-test comparison-helper
check (five repeated) passed; miniature fixtures used no ACS outcomes. Seed0
30.884s projected92.653s for three; actual91.787s with no budget changes or
invalidated scientific runs. New-only replay checked204 candidates,816 score
sets(maxerror4.44e-16),408 bitwise prediction sets,84 release arrays, schedules,
actual Adam states and615 unchanged historical records. Independent/catch-up
selectors, inherited260 fitting passes and all9 race categories remain visible.
No raw/object download, reference regeneration, historical retraining or sweep.

**Unresolved and unsupported.** This initialization did not solve the final
tradeoff under the fixed schedule; it does not show impossibility. Stage changes
do not uniquely identify objectives, geometry or optimization. Weighted/source
counterevidence and sparse race support preclude broad privacy/population claims.
These remain DEVELOPMENT EVALUATION households. The one suggested next change,
a fixed PCA16-output preservation penalty during common warmup, is not executed.
Neither baseline success nor failure establishes PCRL novelty.

Evidence: [decision](results/redesign_20260908_acs_pca16_init_v1/RESEARCH_DECISION.md),
[protocol](results/redesign_20260908_acs_pca16_init_v1/PROTOCOL.md),
[stage/final tables](results/redesign_20260908_acs_pca16_init_v1/TABLE.md),
[paired analysis](results/redesign_20260908_acs_pca16_init_v1/ANALYSIS.md),
[validation](results/redesign_20260908_acs_pca16_init_v1/VALIDATION.md),
[reproduction](results/redesign_20260908_acs_pca16_init_v1/REPRODUCTION.md).

## 2026-09-08 UTC — bounded overnight PCA16 preservation study

Completed from reviewed commit `f197933a9ff57b5702b3547d0b8788f15c80189e` on `ablations-facct-2026-07-24`. All8 preservation conditions×3seeds, stage utility, direct/affine teacher diagnostics, full primary final audits, and the full predeclared nested360-epoch extension are complete. See [the research decision](results/redesign_20260908_acs_preservation_v1/OVERNIGHT_RESEARCH_DECISION.md), [matrix](results/redesign_20260908_acs_preservation_v1/EXECUTED_MATRIX.json), [validation](results/redesign_20260908_acs_preservation_v1/VALIDATION.md), and [reproduction](results/redesign_20260908_acs_preservation_v1/REPRODUCTION.md).

Every final mean improves residence over beta0 under both weights. Strong persistent preservation retains almost all affine teacher structure and teacher-like attribute leakage; the measured D−C protection difference becomes small. Warmup-only preservation gives a limited favorable tradeoff, with higher source losses inside the original margins and important weighted/seed qualifications. All216 non-control audit trajectories retain epochs5–80 checkpoints, so360 does not change release scores or rankings. Race code4 remains unassessable; no new privacy or PCRL novelty claim.

Measured scientific unit/audit wall672.27s includes one1.15s optional-loader preflight failure. The [documented amendment](results/redesign_20260908_acs_preservation_v1/OPERATIONAL_RECOVERY.md) adapted a historical manifest schema before optional fitting; all original freezes, core results and subsequent/unrelated work are preserved. Core and extended replay verify3,456 new/saved prediction sets bitwise across their declared scopes; maximum score discrepancy6.66e-16. Raw data, fitted models/Adam states, releases, predictions and caches remain local with hashes. Compact source/evidence are published to the authorized research branch; no main change, PR, force-push, new data, paid compute or unrelated job interruption.

The single proposed next method experiment is fixed attribute-residualized teacher distillation with ordinary source gradients and matched strong audits. It remains a hypothesis and was not launched. Earlier conclusions and proposed-next-step statements above are historical evidence.
