# PROTOCOL — `pcrl_utility_extension_v1`

Locked before any new 2018 or 2017 outcome for any new release. This is NEW development, not a
retroactive success rule for `pcrl_competitive_method_v1`; that study's negative verdict and
criterion stand unchanged (`results/pcrl_competitive_method_v1/RESEARCH_DECISION.md`, 7f961d5c).

Starting state (full SHAs): experiment `7f961d5c7f6f0562efcb25a27a77bb5221c279a7`; manuscript
`72a6383320e866cf5dbda1a8465c00c3fe068a3b`; direct adversarial `106de9afa58cebbc26e34fb782e539e2a0881108`
/ `69e790af36c5ca53203dab17b757a8e3415ee934`; invariant `73903b7f28df68284285f0610a4036beb32b208f`;
locked 2017 transport `349efa454afd907389760fd1f59fd8806a215efd`.

## 0. Boundaries
* **2016 is sealed and unused**: no transform, label, fit, score, plot, selection, archive
  manifest entry or cloud bundle file (the archive planner rejects `2016`/`ss16` paths).
* 2018 and 2017 are development resources; the original locked 2017 result keeps its status.
* Residence and commute are excluded from every fit, decoder target, checkpoint selection, gate
  and nomination. They enter only their designated probes. Results are reserved-from-training
  capability, not blind evaluation (this project has inspected them repeatedly).
* Terminal 2's blocking items are observed: one declared correction level for primary claims
  (section 7); measured rank/support for every erasure (LEACE-on-R records realised projection
  rank and class support); counts reported as planned / fitted / distinct / duplicate.

## 1. Question
Can a small new channel R, appended to the frozen J release, add useful capability while keeping
the ADDITIONAL measured sensitive recovery over J small? A better method may improve utility at
comparable protection; it need not improve protection at comparable utility.

## 2. Tiers and gates (machine-readable, `_scheduler/gates/T*.json`)
* **T0**: private archive + restore works; execution bundle restored and every file sha256-
  verified on AWS; restored A0/J mappers re-execute within 1e-5 of the stored releases; a restored
  stored probe re-predicts its stored validation probabilities and log loss within 1e-6;
  independent stop installed and rehearsed; forecast within the $30 compute guard.
* **T1** (a) exploratory utility-first reanalysis of the stored 372-slot study
  (`tier1_reanalysis.py`, new family m = 2480, not the old m = 3900); (b) cloud re-audit of
  `ref_J`, `ref_A0`, `ref_leace_A0` (release identity asserted against the stored arrays);
  (c) bounded calibration: exactly the eight recipes in `calibrate.py`, rule for "stronger"
  frozen there. **Gate**: identity asserted and positive control — standard-slate validation
  increment of A0 over H > 0.01 nats on A/SEX and A/RAC1P (seed mean, both weightings). If no
  recipe is stronger, the standard slate is retained and added stress strength is recorded as
  NOT established. Cross-platform audit reproduction of historical A0/J (test split, all
  endpoints) is reported against a 0.003-nat tolerance; all new-versus-J comparisons use the
  cloud re-audit of J under the identical slate, so they do not depend on it.
* **T2 pilot** (42 nominal units): r = 2 x {L1, L2, C1} x beta {1, 3, 10, 30} x anchors {0, 1, 2}
  = 36; unprotected learned `U_r2` x 3; deployable PCA residual `P_r2` x 3. **Gate** (validation
  only, point-estimate screening thresholds, not confidence guarantees): some X configuration
  common to all anchors (i) reduces validation residual-reconstruction MSE by >= 10% versus no
  extension in >= 2 of 3 anchors; (ii) passes the historical source allowance (income_binary,
  civilian_at_work, public_coverage validation log loss <= E_pca + .01, every anchor, both
  weightings); (iii) mean validation sensitive increment over J <= .001 nats on each of A/SEX,
  AB/SEX, A/RAC1P, AB/RAC1P under both weightings, no anchor > .003; (iv) H/J byte parity,
  inference-input restrictions and probability integrity (asserted per unit). A required role
  that is unsupported makes the gate UNASSESSABLE; no missing value is replaced by zero.
  FAIL -> no larger grid; the pilot is reported as negative/inconclusive with which leg failed.
* **T3** (135 nominal slots, pilot slots reused by hash): r {2, 4, 8} x policies x beta x anchors
  = 108; U and P controls per width = 18; LEACE applied only to each unprotected R = 9. Before any
  test-split outcome: select at most two C1 configurations passing the T2 screen, by mean
  validation reconstruction reduction; tie-break lower width, then config ID; one global config;
  `SELECTION.json` written and synced with its timestamp. None passing -> no nominee.
* **T4** (only if nominated): section 6.

## 3. Fixed design (METHOD.md for definitions)
Networks: R = g(x), x = PCA_32 standardised with J's frozen standardiser, g = 32 -> 64 ReLU -> r,
last layer zero-initialised (step 0 = zero extension). Residual head E: [std(H_A, Z_J), R] ->
64 ReLU -> 16, zero-initialised. D0: std(H_A, Z_J) -> 64 ReLU -> 16 (60 epochs, Adam 1e-3,
batch 256), fitted on p0_fit U mapper_fit; 5-fold household cross-fitted residuals for training
rows (salt `pcrl_utility_extension_v1/d0_crossfit/v1`); final D0 for inference/evaluation.
var_T fixed = mean per-coordinate variance of out-of-fold T on mapper_fit; degenerate if
var_T < 1e-4 var(Z_A0) (then units are recorded DEGENERATE, not rescaled).
Baselines p0J_j: MLP[64,32] on [H view, Z_J], 60 epochs on p0_fit. Attackers: predecessor
families (linear, mlp64, mlp64_32), zero-initialised corrections on [H view, Z_J, R].
Training: 600 mapper updates, 100 attacker warmup, 5 attacker updates per mapper update,
refreshes at 25/50/75% (catch-up 40), Adam 1e-3, batch 256, checkpoints every 100 incl. step 0.
Policies exactly the predecessor's (L1 local mean; L2 = 2 x local; C1 local + coalition mean;
L1/L2 spend the coalition slots on local replicas); gains are raw nats, clamped at zero, max over
families; no per-role renormalisation; local race and coverage in training; commute audit-only.
Selection: argmin over checkpoints of [fresh recon + beta x fresh penalty] on monitor, each
checkpoint with its OWN fresh attacker slate (300 updates) and fresh decoder (300 updates);
ties -> earliest step. Validation reconstruction (gate): fresh decoder (40 epochs) on p0_fit U
mapper_fit cross-fitted targets, MSE on `source_validation`, with and without R.
Class masks: rows with missing labels masked, never imputed; class support per fold recorded by
the audit harness.

## 4. Audit (identical primary opportunities)
Unchanged `run_dev_2018.evaluate_seed`: independent logistic, restarted MLP[64,32], HistGB and
kernel families on A/AB roles, canonical B candidates, routed H ancestors, utility probes for the
five tasks, validation-only candidate selection, budgets 120/360. Primary scope
`kernel_expanded_independent`, budget 360. New releases and the re-audited references get the
same slate; no extra catch-up. Exact duplicates (content hash over every wire array of every pool)
are audited once. Fitted attackers are KEPT (no compaction). If a calibration recipe is adopted it
is added symmetrically to every interface including H.

## 5. Units and counts
Nominal: T2 42, T3 135 in total (pilot slots included). References per anchor: ref_J, ref_A0,
ref_leace_A0. Counts reported as nominal / fitted / distinct released / exact duplicates /
degenerate / failed.

## 6. Tier-4 decision (development candidate)
For a nominee versus J (cloud re-audit): mean residence-loss reduction >= .003 nats with the
adjusted interval entirely below zero, AND the one-sided adjusted upper bound on increased
recovery <= .001 nats for EACH of A/SEX, AB/SEX, A/RAC1P, AB/RAC1P, both weightings; original
source allowances; H/J parity; no hidden materially worse forbidden role (all eleven reported).
Zero-margin results are reported too. The .001 tolerance is a practical allowance, not zero harm
or a privacy theorem. Competitiveness additionally requires the same against `ref_leace_A0` at
comparable protection; coalition specificity additionally requires improvement over matched
L1/L2 at comparable usefulness with no concealed local-race cost. Transport to already-used 2017
(no encoder retraining; fresh auditors on the established partitions) is exploratory.

## 7. Uncertainty and multiplicity (one declared level)
Paired household-cluster bootstrap (predecessor `ClusterBootstrap`, 2000 replicates), seed-
averaged per-person test-loss differences, both weightings. **Primary family** = selected
configurations x comparators {J, leace_A0, matched L1, matched L2, U_r, P_r, LEACE_r} x 5
endpoints x 2 weightings; studentized Bonferroni estimate +/- z_{1-alpha/(2m)} SE (one-sided
bounds z_{1-alpha/m}), alpha = .05, m its realised size. This single level is used for every
primary contrast, positive or negative. The full grid (all configurations vs J) is a separate
exploratory family with its own adjustment, reported descriptively and never used to nominate.

## 8. Operations
Cloud host: detached systemd scheduler; persistent in-host deadline timer; CloudWatch idle-CPU
stop alarm; EventBridge Scheduler deadline stop (role limited to task-tagged instances); both
stop paths rehearsed against the verified instance ID before START. Compute guard $30 estimated
(operating ceiling $50 incl. storage); 45-minute closing reserve. Stopping does not delete EBS;
disposable compute is terminated only after outputs are archived and verified.
