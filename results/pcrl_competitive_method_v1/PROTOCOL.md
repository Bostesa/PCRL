# PROTOCOL — `pcrl_competitive_method_v1`

Locked before any new 2018 or 2017 outcome for any new release. Companion documents:
`METHOD.md` (definitions), `MATRIX.json` (every nominal slot and its audit priority),
`DEPENDENCY_MANIFEST.json` (code, fixture and document hashes; `python -m
experiments.pcrl_competitive_method_v1.freeze --verify` fails on drift),
`CORRECTIONS.md`, `SELECTION_DIAGNOSIS.md`. Amendments go in `RUN_STATUS.md` with
timing, outcomes seen, affected units and whether refitting is required.

## 0. Boundaries

* **2016 is sealed and unused.** No transform, label, fit, score, plot or selection.
* 2018 and 2017 are **development** resources. The first locked 2017 experiment keeps
  its historical status; nothing here transfers that status to this study.
* Residence and commute are excluded from every fit, nuisance, checkpoint selection,
  hyperparameter choice and panel nomination. They enter only their designated probes
  after releases are locked. This project has inspected them repeatedly, so results
  are **reserved-from-training capability**, not blind evaluation.
* A favourable development result earns a **prospective candidate**, never a
  confirmation.

## 1. Research questions

* **Diagnostic** (done, `SELECTION_DIAGNOSIS.md`): the previous run's trajectories all
  moved; selection preferred step 0 by a real margin; the zero-gain clamp was not the
  driver.
* **Track N**: at fixed protection formulation, does lowering `gamma` improve the
  measured tradeoff? Does J initialisation help? Does C1 beat matched L1/L2 there?
* **Track E**: on the same frozen strong channel, does removing coalition-conditioned
  residual directions beat local-only removal (`L`, `LX`) at equal retained rank? Does
  it beat marginal partial erasure, rank-matched PCA and random compression, the
  untouched channel, `J`, and ordinary LEACE/SPLINCE?

## 2. Matrix (MATRIX.json)

Track N 168 nominal slots (144 main + 24 `beta = 0` continuations); Track E 204 (90
core + 90 controls + 18 full-span + 6 ordinary erasure on J); **372 total**, plus the
untouched references `ref_A0`, `ref_J` per anchor. Nominal slots are not distinct
systems; fitted, infeasible, duplicate, audited and pending counts are reported
separately.

## 3. Audits (2018)

* Machinery: the predecessor's `run_dev_2018.evaluate_seed`, unchanged — independent
  logistic, restarted `MLP[64,32]`, boosted trees and kernel families for A/AB roles,
  canonical B candidates, legally routed `H` ancestors, utility probes for the five
  utility tasks, validation-only selection, budgets 120 and 360.
* **Primary scope `kernel_expanded_independent`, budget 360** (matched exposure). The
  catch-up scope is reported separately and never merged. New arms get **no** extra
  attack families; historical arms get **no** extra catch-up in the primary comparison.
* **Identity checks (registered validation):** `ref_A0` and `ref_J` must reproduce the
  historical matched-scope audits of the same channels exactly. Seed-0 `ref_A0` did
  (before lock).
* **Deduplication**: releases bitwise identical on every wire array of every pool are
  audited once; the duplicate inherits the canonical audit and is counted as such.
  Identity is never inferred from similar scores.
* **Numerical integrity**: every stored probability matrix is checked for finiteness,
  range, nonzero row mass and row sums before a unit completes. A failure quarantines
  the unit's outputs with provenance; at most **three** retries on unchanged inputs;
  a unit failing three times is a scoped stop, other units continue. No cause is
  attributed without evidence.
* **Compaction** (disk): after a unit validates, `predictions.npz` is reduced to the
  keys any registered selection can reach (both splits, every weighting x budget x
  scope, recomputed with the predecessor's own `point_from_records`); kept arrays are
  asserted bitwise equal to the originals and the original sha256 is recorded.
* **Audit queue priority** (fixed now): 1 = everything except the two reduction slices;
  2 = Track E `k in {1, 6}`; 3 = Track N `gamma = .01`. See §9.

## 4. Candidate panel (selected WITHOUT residence or commute)

Candidate configurations are common across all three anchors:

* neural family: C1 at each `(init, gamma, beta)` with `beta in {1, 3}` — 16 configs;
* projection family: `C` at each `(channel, k)` — 10 configs.

**Eligibility** (all must hold on every anchor, both weightings, **validation split**):

1. exact `H` parity (asserted at release);
2. historical source-probe allowance: each of `income_binary`, `civilian_at_work`,
   `public_coverage` has validation log loss at most `.01` nats above the historical
   `E_pca` parent (the frozen registry's `source_preservation` rule, evaluated on the
   validation split).

**Ranking** within the eligible set, lexicographic:

1. minimise the **worst normalised positive sensitive increment**: for each endpoint
   `e in {A/SEX, AB/SEX, A/RAC1P, AB/RAC1P}`, the anchor-mean over seeds of
   `max(0, gain_release(e) - gain_H(e))` on the validation split, unweighted, divided
   by the fixed training H-only class entropy of the target (`SEX`, `RAC1P` priors of
   the frozen registry); take the maximum over the four endpoints; local and coalition
   endpoints both count;
2. lower anchor-mean training teacher distortion (neural: selected-checkpoint monitor
   distortion vs A0; projection: normalised distortion on representation-fit rows);
3. config ID (lexicographic string).

Take the top **two** per family. For each nominated neural config carry matched L1 and
L2 (same init, gamma, beta); for each nominated projection carry matched `L` and `LX`
(same channel, k). At most 12 configurations x 3 anchors = 36 interfaces.

**Fallback** if a family has no eligible config: no nominee is published for it. For
diagnostic transport only, `N_J_g010_C1_b100` (neural) and `E_J_C_k2` (projection) are
carried, labelled **ineligible**, never relabelled a success.

The panel is written to `PANEL.json` and committed **before** its 2018 evaluation
(test-split) metrics are opened in any table. The full 2018 grid remains available
for descriptive frontier analysis, labelled exploratory and distinguished from the
prospective panel.

## 5. Comparisons, uncertainty, adjustment

Paired household-cluster bootstrap (predecessor `ClusterBootstrap`, 2000 replicates,
same cohort clusters), seed-averaged differences of per-person test losses of the
selected candidates, both weightings from the same predictions.

**Primary family P** (searched for superiority/noninferiority claims): for every panel
configuration,

* vs `J` (historical J, matched scope), `leace_A0`, `splince_A0`, `optnet16_C1`;
* vs its matched local controls (neural: L1, L2; projection: L, LX);
* projection only: vs marginal, PCA, random at the same `(channel, k)`; vs `leace_J`;
  vs its untouched starting channel (`ref_A0` / `ref_J`);
* neural only: vs its `gamma = 1` counterpart and vs its other-initialisation
  counterpart;

x 5 endpoints (4 sensitive + `same_residence`) x 2 weightings. Adjustment: studentized
Bonferroni `estimate ± z_{1-alpha/(2m)} x SE`, `alpha = .05`, `m` the realised size of
P (the predecessor's amendment-3 form; the percentile at `alpha/m` is not estimable
from 2000 replicates).

**Exploratory family X** (whole grid; never used to nominate): gamma vs 1 at fixed
init/policy/beta; J vs A0 init at fixed gamma/policy/beta; C1 vs L1 and vs L2 in every
matched cell; C vs L, C vs LX, C vs marginal/PCA/random at every `(channel, k)`; every
Track N and Track E release vs `J` and vs `leace_A0`. Same adjustment over the
realised size of X. Reported with effect sizes, per-anchor directions and signs, never
only significance counts.

## 6. Decision levels (reported separately)

* **Mechanism works**: the transformation/optimisation achieves its declared finite
  objective (training moments reduced as specified; selection reconstructs; ranks as
  requested).
* **Competitive development tradeoff**: a panel release has at least one sensitive
  endpoint with an adjusted superiority interval entirely below zero against **`J` and
  against at least one of `leace_A0`/`splince_A0`/`optnet16_C1`**, while
  (a) residence utility is noninferior to that comparator at the `.001` margin
  (one-sided adjusted bound), (b) every other sensitive endpoint is noninferior at the
  practical `.001`-nat tolerance (also shown at zero margin), (c) source eligibility
  holds. Required under **both weightings** for the strongest label. If unresolved,
  it is called unresolved, not failed equivalence.
* **Coalition-specific benefit**: that release also improves on its matched local
  controls at comparable rank/usefulness.
* **Candidate for confirmation**: the same fixed configuration survives the stronger
  attack (§7) and the descriptive 2017 check (§8). No fresh-year confirmation is
  claimed.

Retained for continuity and reported separately: the historical `.01` residence
reference gain over `H`, the original source allowances, the historical half-headroom
criterion, and the historical `.001` point-estimate coordination rule under its own
name. Discrete Pareto plots of residence gain against each of `A/SEX`, `AB/SEX`,
`A/RAC1P`, `AB/RAC1P` separately, with eligibility and retained rank marked; the four
risks are never collapsed into a mean.

## 7. Stronger attack (frozen now, before any panel outcome)

Held out of every training ensemble and of the standard audit slate: a wider, deeper
network, **`MLP[256, 256, 128]`**, ReLU, Adam `lr 1e-3`, weight decay `1e-5`, batch
256, **2 fresh initialisations** (seeds `31000 + 100*anchor + j`), 240 epochs with
snapshots every 40 epochs, trained on the historical `attacker_fit` subset (4096,
predecessor indices), snapshot and initialisation selected by `attacker_validation`
log loss, scored once on test. Roles: `A/SEX`, `A/RAC1P` on `wire/A`; `AB/SEX`,
`AB/RAC1P` on `wire/AB`; and the same recipe on the `H`-only views, so increments over
`H` use the same attacker on both sides. Applied identically to every panel release,
its matched controls, `ref_J`, `ref_A0`, `leace_A0`, `splince_A0`, `optnet16_C1`.
Reported as absolute recovery and increment over `H`, both weightings, beside the
standard audit. No universal 720-epoch expansion is repeated.

## 8. 2017 exploratory transport

The fixed panel and its matched controls, with `J`, `A0`, `leace_A0`, `splince_A0`,
`optnet16_C1` references, are carried to the already-used 2017 pools: **no 2017
encoder or eraser fitting**; the 2018-fitted maps are applied to the 2017 frozen
channel, and only fresh attackers and utility probes are fitted on the existing 2017
fit/validation partitions (predecessor transport machinery). At most 12 candidate +
24 matched-control interfaces before reuse. All transport results are exploratory.

## 9. Compute and schedule

Measured `2026-09-19T02:54Z`: swap 0 used, compressor 2.8 GB, free+inactive ~8.8 GB,
load ~3.5, 33 GB disk free. Pilot: Track E 30 s per anchor (all 68 maps); Track N
11 s per unit at 0.77 GB RSS; one audit ~41 s at ~1.0 GB RSS, ~26 MB after
compaction. On that measurement this study runs **one worker per anchor (3 workers)**,
single BLAS/OpenMP thread each, disjoint output trees, one orchestrator lock; swap is
re-measured between stages and concurrency drops to one worker if swap use exceeds
2 GB. Session ceiling `09:47Z`; final hour reserved.

Projected: fits ~15 min; ~300 distinct 2018 audits ~80 min; panel + 2017 + stress
~75 min. The full matrix is expected to fit. **Prospective uniform reduction**, if the
2018 audits have not reached priority 3 by `06:45Z`: stop launching priority-3 units
(Track N `gamma = .01`) for all anchors; if priority 2 is unfinished by `07:00Z`, stop
launching it too (Track E `k in {1, 6}` for every matched method). Any unlaunched unit
is recorded as pending with a resume command. Nothing is shortened selectively.

## 10. Reporting

Every table and figure has machine-readable backing and source hashes. Headline numbers
are replayed from stored predictions by an independently written scorer. All
configurations and all anchors are reported; per-anchor tables accompany pooled ones;
a local race cost is never hidden behind a coalition SEX gain.
