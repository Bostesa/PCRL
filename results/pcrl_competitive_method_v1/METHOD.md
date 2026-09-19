# METHOD — `pcrl_competitive_method_v1`

Written before any new 2018 or 2017 outcome was opened for any new release. The only
audit run before this lock is `ref_A0` seed 0 — the **untouched A0 channel**, whose
outcome is already published — used as a timing measurement and as an identity check
(it reproduced the historical audit of a bitwise-A0 channel exactly: 144,778 numeric
metrics, 0 differences, predictions bitwise equal).

Implementation: `experiments/pcrl_competitive_method_v1/`. Tests:
`tests/pcrl_competitive_method_v1/`. Predecessors, full SHAs:

| Study | Commit |
|---|---|
| Direct adversarial (HEAD / evidence) | `106de9afa58cebbc26e34fb782e539e2a0881108` / `69e790af36c5ca53203dab17b757a8e3415ee934` |
| Manuscript v3 (Terminal 2) | `ce5ba24a0a0ed1848c4029a922ac66fa629d71fc` |
| Invariant repair + external baselines | `73903b7f28df68284285f0610a4036beb32b208f` |
| Nonlinear / rank | `c37807e4f568ef38e5528fc09c1506083278bf4d` |
| Original locked 2017 transport | `349efa454afd907389760fd1f59fd8806a215efd` |

This study is based on `106de9af` on branch `research/pcrl-competitive-method-v1`.

---

## 0. Attribution and what is not new

* Adversarial transferable representations: Madras, Creager, Pitassi & Zemel, ICML 2018,
  <https://proceedings.mlr.press/v80/madras18a.html>. Track N's alternating training
  is theirs in form; the residual-logit attacker, household folds and fresh-probe
  checkpoint selection are the predecessor study's, reused verbatim.
* LEACE: Belrose et al. 2023, <https://arxiv.org/html/2306.03819v4>. Its affine erasure
  of a *complete* cross-covariance span, with a guarantee against all affine predictors
  under convex losses on the transformed variable, is theirs. **Track E's partial
  projection inherits none of that guarantee** (§2.6).
* SPLINCE, SARL, OptNet-ARL and the conditional-moment references are the ones pinned
  in `results/pcrl_invariant_baselines_v1/BASELINE_ADAPTATIONS.md` at `73903b7f`;
  SPLINCE and OptNet are reused through that study's verified wrappers.
* **The residual spectral moment objective of this project** (`experiments/
  acs_residual_spectral.py`, locked 2017 study `349efa45`) already defines the moment
  Gram matrix Track E ranks by. §2.5 states the relation algebraically. **Track E does
  not introduce a new eigensystem**; it applies that eigensystem's hard-removal limit
  to a different input space.

Empirical improvement and novelty are assessed separately. Any novelty statement is an
absence-of-evidence statement after a bounded search, never a claim of priority.

## 1. Track N — neural utility and initialisation

Width 16, one architecture: `standardise(PCA_32) -> Linear(32,64) -> ReLU ->
Linear(64,16) -> Z`, Z-only source heads `Linear(16,1)` for `income_binary` and
`civilian_at_work`.

**Initialisation** `in {A0, J}`. Both are recovered from their 2018 checkpoints and
asserted bitwise identical to their released wires on all seven pools (A0 via the
predecessor loader; J via `run_fit_e.frozen_channel`). J's architecture is identical
to A0's and its input standardiser is asserted identical, so **no wrapper, distillation
or approximation is involved**: the J arm starts from J's trained weights.

**Teacher.** The frozen A0 channel for **both** initialisations, with distortion
normalised by A0's representation-fit variance. J therefore starts at a nonzero
distortion — measured on seed 0 as `0.634` (mapper_fit) / `0.641` (monitor) in these
units. That is a property of the design.

**Objective**, identical to the predecessor except for `gamma`:

```
U(theta) = mean_{income, employment} CE(Z-only head) + gamma * ||f_theta(x) - t_A0(x)||^2 / var_A0
mapper:  U + beta * policy_penalty(G)        policies L1, L2, C1 exactly as predecessor §5.1
```

`gamma in {0, .01, .1, 1}`, `beta in {1, 3}`, policies `{L1, L2, C1}` with the
predecessor's slot schedule (L1/L2 spend the two coalition-equivalent slots on
independently initialised local replicas), three anchors: **144 main units**. Plus
`beta = 0` continuations (policy C1 attackers still train, penalty weight zero) for
each initialisation x gamma: **24 units**. Total **168**.

Unchanged from the predecessor: three differentiable correction families, 100
attacker warmup updates, 5 attacker updates per mapper update, refreshes at 25/50/75%,
Adam `1e-3`, batch 256, **600 mapper updates**, checkpoints every 100 **including step
0**, head warmup 60 updates with the mapper frozen (so step 0 releases the untouched
starting channel bitwise), per-seed minibatch order shared by every unit, optimiser
seeds shared across matched arms. **No ensemble-size sweep**; its cost is reported.

**Checkpoint selection**: `argmin_t [C_t + gamma*D_t + beta*P_t]` on the internal
`monitor` fold, every checkpoint (step 0 included) scored against its **own freshly
initialised attacker slate at the same 300-update budget** (predecessor amendment 1).
Ties to the earlier step. No residence, commute, downstream or test pool. The
`gamma` in the selection score is the unit's own `gamma`.

**Historical reuse.** The 18 `gamma = 1, init = A0` units at `beta in {1, 3}` coincide
with predecessor arms `dax16_{L1,L2,C1}_b{100,300}`. They are **refit** and asserted
bitwise identical (channel hashes on every pool) to the historical releases rather
than imported; seed 0 `C1_b100` was verified in the timing pilot (identical hashes on
all 7 pools, identical monitor scores, same selected step 300).

## 2. Track E — coalition-conditioned partial projection

### 2.1 Input and whitening

Starting channels: frozen `A0` and frozen `J`, separately. `z` a row vector, `d = 16`.
On representation-fit rows: `mu`, `Sigma = (z-mu)'(z-mu)/n`, symmetric
eigendecomposition, relative tolerance `1e-10 * lam_max` defines the supported rank
`r`; `R = Q_s lam_s^{-1/2}`, `S = lam_s^{1/2} Q_s'`, `v = (z - mu) R`. `R S` is the
orthogonal projector onto the supported subspace (identity at full rank). No ridge.
Measured on seed 0: `r = 16` for both channels at every tolerance in
`{1e-14,...,1e-6}`; condition numbers 783 (A0) and 4992 (J).

### 2.2 Nuisance and residuals

For each trainable role `j` (`A/public_coverage`, `A/SEX`, `A/RAC1P`, `AB/SEX`,
`AB/RAC1P`), `p_j(s | h_j)` is the predecessor's service-only `MLP[64,32]` family
(60 epochs, Adam `1e-3`, batch 256) fitted by **5-fold household cross-fitting** inside
representation-fit (`sha256('pcrl_competitive_method_v1/nuisance_crossfit/v1|SERIALNO')
mod 5`). Out-of-fold `e_j = onehot(s) - p_j(h_j)`; rows with a missing label are
masked out, never imputed. The nuisance is fitted once per anchor and shared by both
channels and every policy. Cross-fitting limits self-fitting; it proves neither
nuisance correctness nor conditional independence.

### 2.3 Bases

Nonredundant service coordinates are chosen by an outcome-free rule: in fixed column
order, keep a column of `H` if it raises the numerical rank of `[1, kept]` at relative
tolerance `1e-6` (above float32 rounding of the probability pairs, which sum to 1 only
to ~9e-8). Verified: `H_A -> 2` coordinates (income, employment), `[H_A,H_B] -> 3`.

* `b_A` = constant + 2 linear + 3 quadratic = **6** coordinates;
* `b_AB` = constant + 3 linear + 6 quadratic = **10**;
* `b_LX` = `b_A` + **10** deterministic random Fourier features of the two `H_A`
  coordinates (seed `20269018`, Gaussian frequencies with bandwidth = median pairwise
  distance on <=512 training rows, uniform phases) = **16**, used for `A/SEX` and
  `A/RAC1P` only; `A/public_coverage` keeps `b_A`.

Monomials and RFF features are scaled to unit training standard deviation; the constant
keeps scale 1; nothing is centred. Scales are fixed from training inputs.

### 2.4 Moments, kernels, policies

```
G_j = mean_{valid j}[ v' (b_j(h_j) kron e_j)' ]        (r x B*K), v globally centred
K_j = G_j G_j' / max(||G_j||_F^2, eps),   eps = 1e-8 * max_j ||G_j||_F^2   (per channel, anchor)
K_local = mean(K_{A/cov}, K_{A/SEX}, K_{A/RAC1P})       K_coal = mean(K_{AB/SEX}, K_{AB/RAC1P})
L  : K_local          C : K_local + K_coal          LX : mean(K_{A/cov}, K^LX_{A/SEX}, K^LX_{A/RAC1P})
```

`L2 = 2 K_local` is an **exact eigenspace alias** of `L` at every fixed `k` (the
eigenvectors of `cK` equal those of `K` for `c > 0`; tested by comparing projectors).
It is registered as an alias and not fitted. Likewise any uniform positive reweighting
of the whole policy matrix — for example "equal mass" — is analytically redundant for a
fixed-rank projection.

**LX** has the same labels, no `H_B`, and matches the nominal target-feature count of
`L` plus `C` (6 + 10 = 16). It is a richer local conditioner, **not** a function-class
match to the joint-view polynomial: a C-versus-LX difference is not attributable to the
coalition view alone.

### 2.5 Release and relation to the residual spectral objective

`U_k` = leading `k` orthonormal eigenvectors of the policy matrix (deterministic order;
a tie straddling `k` is flagged `boundary_degenerate`), `P_k = I - U_k U_k'`,

```
z_out = mu + v P_k S          ambient width 16, realised rank r - k
```

`k in {1, 2, 4, 6, 8}`. Proved and tested (§4): idempotence, realised rank `r - k`,
exact preservation of the training mean, the row convention against a column-vector
reference, and **equivariance** under orthogonal reparameterisation of the input
channel (`z -> zO` gives `z_out -> z_out O`, including with repeated covariance
eigenvalues, because the whitening's internal basis ambiguity cancels between `v`,
`P` and `S`).

**Relation to earlier work, algebraically.** `experiments/acs_residual_spectral.py`
builds, for each role, `P_role = mean_attr trace_normalise( sum_c G_c G_c' )` with
`G_c = V'(b * residual_c)/n`. Since `sum_c G_c G_c' = G G'` for the class-stacked
`G = V'(b kron e)/n`, **`P_local`, `P_coalition` and `P_marginal` there are exactly
`K_local`, `K_coal` and the marginal kernel here**, up to (i) the input space `V`
(there: whitened residual RFF features of the 32 PCA inputs; here: the whitened frozen
neural channel) and (ii) the nuisance family and folds. That study released the top
eigenvectors of `U - lambda P`. For `lambda -> infinity` with the retained rank held at
`r - k`, those converge (off degeneracies) to the bottom `r - k` eigenvectors of `P`,
which is `P_k` here. **Track E is therefore the utility-free, infinite-penalty limit of
the same eigenproblem, applied to a different input channel and unwhitened back into
that channel's metric.** It is not presented as a new eigensystem.

### 2.6 What the projection does and does not do

* It is **rank-reducing**, not an invertible shrinkage: `k` directions are removed and
  cannot be recovered from `z_out`.
* It is **partial**. A partial projection generally leaves nonzero training
  cross-moments (tested). The full-span projection zeroes them (tested) — but on these
  channels the moment Gram is **full rank**, so the full-span map is the constant
  `mu` (measured on seed 0 for every policy and both channels). That is reported as an
  outcome, audited once per distinct release, and is informationally equivalent to
  `H`; a finite learner with an expanded input may still behave slightly differently.
* The top-`k` eigensystem maximises `trace(U' K U)` over `k`-frames in the whitened
  metric. It does **not** optimise attacker recovery, residence capability or source
  covariance, and preserves none of them by construction.
* **LEACE's complete-erasure guarantee is not claimed.**

### 2.7 The stationarity result, with its exact scope

Model: fixed-offset multinomial logit on the valid rows of role `j`,

```
logits(h, z) = l0_j(h) + Theta . ( b_j(h) kron v_out ),   v_out = v P_k   (no free intercept)
```

with `l0_j` held fixed. The average cross-entropy is convex in `Theta`, and its
gradient at `Theta = 0` is exactly `-G_j^out`, the declared cross-moment evaluated on
`v_out` with `e_j = onehot(s) - softmax(l0_j(h))`. Hence **if `G_j^out = 0`, zero
correction is stationary and therefore globally optimal within this family** on these
rows. Conditions, all necessary:

1. the offset is the **same** `l0_j` that defines `e_j` (our `e_j` is out-of-fold, so
   the theorem applies to the cross-fitted offset, not to a full-sample refit);
2. the same valid-row mask and denominator;
3. no free per-class intercept: its gradient is `-mean(e_j)`, which cross-fitting does
   not make zero (tested: nonzero in the fixture);
4. the correction is linear in `b(h) kron v_out`.

It says nothing about retrained `H`-only baselines, nonlinear auditors, other
correction families, or population quantities, and a vanishing moment with a
misspecified nuisance guarantees nothing (tested counterexample: `s = 1[|v| > c]` has
all linear moments ~0 and is recovered by a quadratic probe).

## 3. Track E controls

Per starting channel, rank and anchor:

* **marginal partial erasure** — same machinery with `b = 1` and globally centred
  one-hot targets for SEX, RAC1P, public_coverage (no nuisance). A partial adaptation,
  **not complete LEACE**;
* **PCA** retaining `r - k` principal directions of `z` in the *original* metric
  (generic variance-ranked compression; whitened PCA would be meaningless);
* **one** random `r - k` subspace in the whitened metric, seed
  `20269500 + 1000*anchor + k`, **shared by A0 and J** and by every comparison at that
  `(anchor, k)`; one draw is a limited baseline, never "the best random subspace".

Diagnostics: full-span `L`/`C`/`LX` on both channels (18); ordinary LEACE (rank-
stabilised, predecessor wrapper) and SPLINCE (predecessor wrapper, silent LEACE
fallback disabled, infeasibility reported as `SCOPED INFEASIBLE`) on `J` (6).
Historical `leace_A0`/`splince_A0` are reused, never refit.

## 4. Mathematical acceptance tests (run before any new outcome)

`tests/pcrl_competitive_method_v1/test_projection_math.py`, 15 tests, all passing at
lock: whitening round trip and `k = 0` identity; row vs column convention; projector
idempotence, realised rank, mean preservation, replay idempotence; scaled-alias
projector equality; orthogonal equivariance (generic and repeated eigenvalues);
full-span moment zeroing and partial non-zeroing; masked moment with global centring
and valid denominator; basis dimensions (and under float32 rounding); gradient `= -G`
at zero correction, vanishing after full-span removal, optimality of zero correction,
non-vanishing intercept gradient; the quadratic counterexample; the synthetic
interaction example (sign of the `z-s` relation flips with `H_B`: local moments ~0,
joint-view moments >10x larger, leading joint direction aligned with the signal,
|corr| > .95); the correlated authorised/sensitive example (removing the shared
direction raises sensitive loss by >.3 nats and source loss by >.05 nats).

## 5. Release contract

Unchanged from the predecessor and asserted per pool per release: `wire/A = [H_A, Z]`,
`wire/B = H_B`, `wire/AB = [H_A, Z, H_B]`, float64, service coordinates bitwise
identical in value, dtype and order. `Z` at inference needs only the A-side inputs
(`PCA_32` via the frozen standardiser, then the frozen mapper, then a fixed affine
map); **`H_B` is never an A-side input** — it enters only the fitting of `K_coal`.
No sensitive label and no row identifier at inference.
