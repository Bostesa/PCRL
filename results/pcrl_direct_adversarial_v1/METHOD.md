# METHOD — `direct_adversarial_channel_refinement`

Written **before** any new interface was audited and before any 2018 or 2017 outcome
was opened for the new arms. Every scale, seed, subset, fold and constant below is
prospective and is hashed into `PROTOCOL_FREEZE.json`.

Implementation:
`experiments/pcrl_direct_adversarial_v1/{inputs,channel,attackers,train,run_fit}.py`.

Predecessors, resolved to full SHAs and preserved unchanged:

| Study | Branch | Commit |
|---|---|---|
| ACS 2017 transport (locked) | `residual-spectral-20260910` | `349efa454afd907389760fd1f59fd8806a215efd` |
| Nonlinear / rank | `research/pcrl-nonlinear-rank-v1` | `c37807e4f568ef38e5528fc09c1506083278bf4d` |
| Invariant repair + external baselines | `research/pcrl-invariant-baselines-v1` | `73903b7f28df68284285f0610a4036beb32b208f` |
| Manuscript (Terminal 2) | `research/pcrl-manuscript-integrated-v2` | `3c6ada82e719489656da820b72ce8425d2079be0` |

This study is based on `73903b7f`. It does **not** touch the residual spectral moment
penalty, which is closed by `RESEARCH_DECISION.md` at `73903b7f`.

---

## 0. What this is, and what it is not

The strongest developed protection channel in this line of work is `J`: the frozen
neural auxiliary channel trained against a nine-role observer ensemble with
coefficients `individual = -0.1`, `coalition = -0.1`. `J` beats every spectral arm and
ties the external erasure baselines. This study asks whether a **directly adversarial
refinement** of that same channel — a refreshed attacker ensemble, residual-logit
parameterisation, an explicit incremental-gain penalty and coalition conditioning —
improves the measured utility/disclosure frontier beyond `J`, beyond erasure and
beyond OptNet.

**Nothing in the ingredient list is new.** Attribution, stated before any number:

* **Adversarial fair and transferable representations** — Madras, Creager, Pitassi &
  Zemel, ICML 2018, <https://proceedings.mlr.press/v80/madras18a.html>. Adversarial
  training of an encoder against a discriminator, and the transfer framing, are theirs.
* **LEACE** — Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff & Biderman, 2023,
  <https://arxiv.org/abs/2306.03819>. Closed-form affine concept erasure; the erasure
  controls here are its application to a new input channel.
* **SPLINCE** and **OptNet-ARL** — the primary sources pinned in
  `results/pcrl_invariant_baselines_v1/BASELINE_ADAPTATIONS.md` at `73903b7f`, reused
  verbatim and not re-derived.
* **Attacker ensembles, attacker refresh, residual/offset parameterisation, and
  representation compression** are all established. None is presented as invented here.

The question is **empirical**: whether a careful adaptation of these ingredients to
this project's immutable-service and recipient/coalition access contract produces a
useful measured improvement. **Novelty is established separately from performance**,
and any novelty statement is phrased as absence of evidence after a bounded search.

Four explicit non-claims:

* Finite alternating optimisation is **not** a solved minimax problem and confers no
  protection certificate.
* A low measured incremental gain means the **declared finite differentiable family**
  did not find one inside its budget. It is not independence, not differential privacy,
  not a bound on `I(S;Z|H)`, and not protection against arbitrary attackers.
* The incremental gain `g = CE(p0) − CE(q)` is **not** a new conditional-information
  estimator. Subtracting the fixed `CE(p0_j)` creates **no new encoder gradient at
  all**: `p0_j` does not depend on `theta`. What the baseline changes is which
  constraints are active under the `max(0, ·)` and how the reported quantity reads.
* Teacher distortion is a **useful-capability proxy**. It is not a guarantee that
  residence capability transfers, and §9 tests whether it predicts residence at all.

---

## 1. The starting channel, recovered and proved

The frozen `A0` auxiliary channel is recovered from
`results/redesign_20260909_acs_fixed_predictions_v1/seed_N/training/A0/final.pt` as

```
standardise(PCA_32)  ->  Linear(32,64) -> ReLU -> Linear(64,16)  ->  Z          (A0)
heads: Linear(16,1) for income_binary and civilian_at_work
```

**Width is verified, never assumed.** The reconstructed channel is asserted
**bitwise identical** to the released 2018 wire `wire/A/{pool}[:, 4:]` on all seven
pools before it is used (`inputs.load_frozen_state`). Measured on seed 0 during
recovery: `max_abs_difference = 0.0`, `bitwise_identical = True`, on every pool.
The nominal design assumption of width 16 therefore **holds**, and no distillation
substitute is needed. A reconstruction that were not bit-exact would be a **new fit**
and would not be permitted to inherit the `A0` identity; the assertion enforces that.

### 1.1 Widths

| Width | Starting channel |
|---|---|
| **16** | **Identical** to the frozen `A0` mapper and heads, parameter for parameter |
| **8** | The same `A0` mapper followed by **one** deterministic training-only PCA projection of the `A0` output |

The width-8 compressor is `Linear(16,8)` with `W = P^T`, `b = -mu P`, where `mu` and
`P` are the mean and top-8 right singular vectors of the `A0` output **on the
representation-fitting rows only**. Its heads are the exact algebraic image of the
`A0` heads under that projection (`w8 = w16 P`, `b8 = b16 + w16 mu`), so both widths
descend from one object rather than two unrelated fits. The whole path stays
differentiable, so the compressor is fine-tuned along with the mapper.
**The PCA sees no reserved label and no outcome.**

### 1.2 Common warm start

Within each `(seed, width)` the Z-only source heads are warmed for **60** updates with
the mapper and compressor **frozen**. Because only the heads move, the **released
channel is bitwise unchanged** by warming, so `teacher` is unambiguous and the warmed
state is byte-identical across every policy, beta and repeat at that `(seed, width)`.
Its hash is recorded (`fit_context.json:widths.*.warmed_state_hash`).

Minibatch orders are generated **once per seed** (`default_rng(20265018 + seed)`,
batch 256) and consumed in the same order by every policy, beta, ablation and repeat,
so matched arms see paired minibatches.

---

## 2. Release contract and label boundaries — unchanged

Recovered from the frozen registry and **not** renegotiated:

* `H_A` = the full income/employment probability vectors, **4** coordinates.
  `H_B` = the full coverage probability vector, **2** coordinates.
* `A` receives `[H_A, Z]`; `B` receives `H_B`; `AB` sees the union `[H_A, Z, H_B]`.
* `Z` is computed from `PCA_32` and the `A0` standardiser alone — inputs already
  permitted at `A` inference. **No sensitive label and no row identifier at inference.**
* `H_B` **may** condition coalition training attackers (it is in the `AB` view) and is
  **never** added to the `A` release nor required as a new `A`-side input.
* Every release wire is float64 with `wire/A[:, :4] == H_A` and `wire/AB[:, -2:] == H_B`
  **bitwise**, asserted per pool per arm (`run_fit.validate_cache`, `inputs.build_wires`).
  Bitwise preservation of the service array is **not** the same as source accuracy
  across years; source probes are scored separately.

**Reserved labels.** `same_residence` and `commute_over20` never enter encoder
training, checkpoint selection, early stopping, calibration or release selection. The
only label helper that can reach them has those two keys **deleted from its output**
before it returns (`inputs.authorised_source_labels`), so a later typo cannot index
them. They are used afterwards only for their designated probes and for exploratory
frontier analysis. Because these tasks have been inspected repeatedly across this
project's history, **the overall development process is not called blind to them.**

**Trainable protected roles** — exactly the five previously allowed ones:
`A/public_coverage`, `A/SEX`, `A/RAC1P`, `AB/SEX`, `AB/RAC1P`. The reserved commute
label is **not** pulled into protection training merely because `A/commute_over20` is
an audited forbidden task. The audit uses the **full historical eleven-role forbidden
registry**, including the unchanged `B` roles, with the same missing-category handling.

---

## 3. Internal household folds

`representation_fit` (10 513 rows) is split at the **household** level by
`sha256('pcrl_direct_adversarial_v1/internal_household_fold/v1|<SERIALNO>')`:

| fold | share | use |
|---|---|---|
| `p0_fit` | `[0.00, 0.25)` | fit the frozen service-only baselines `p0_j` |
| `mapper_fit` | `[0.25, 0.80)` | the alternating game |
| `monitor` | `[0.80, 1.00]` | attacker-refresh decisions and checkpoint selection |

Households never straddle a fold. Realised sizes (seed 0): 2790 / 5655 / 2068. All of
this lives **inside** representation-training households; none of it is a downstream
audit pool.

---

## 4. Service-only baselines and residual-logit attackers

For each protected role `j`, `p0_j` is an `MLP[64,32]` on **that role's `H` view
alone** (`H_A` for a local role, `[H_A, H_B]` for a coalition role), fitted on `p0_fit`
with Adam (`lr 1e-3`, batch 256, 60 epochs), then **frozen before the mapper moves**.
Fitting `p0_j` on a fold disjoint from `mapper_fit` prevents a self-overfitted baseline
from manufacturing apparent gains.

Every full-view attacker sees the same service view **plus `Z`** and is a **residual
logit correction**:

```
q_jk(s | H, Z) = softmax( logits_p0_j(H) + d_jk(H, Z) )                        (1)
```

The output layer of every `d_jk` is **zero-initialised**, so the zero correction
recovers `p0_j` exactly and `g_jk = 0` at initialisation by construction. The attacker
learns interactions between `H` and `Z`; conditioning is never replaced by label strata.

**Empirical incremental attack gain**, on a fixed minibatch or evaluation fold:

```
g_jk(theta) = CE(p0_j) − CE(q_jk(theta))                                       (2)
G_j(theta)  = max( 0, max_k g_jk(theta) )                                      (3)
```

The zero option in (3) is the service-only predictor. `torch.max` returns the **first**
maximiser and the `(slot, family)` order is fixed, so ties resolve deterministically.

**Reported** losses use a fixed probability clip of `1e-12` with row renormalisation,
the repository scorer's own rule; clipping rates are logged.

### 4.1 The training ensemble

Three **differentiable** correction families per role slot:

| family | architecture |
|---|---|
| `linear` | `Linear(d, K)` |
| `mlp64` | `Linear(d,64) → ReLU → Linear(64,K)` |
| `mlp64_32` | `Linear(d,64) → ReLU → Linear(64,32) → ReLU → Linear(32,K)` |

Widths 64 and 64/32 are the repository's established observer widths and are locked
here before any outcome. **Trees and kernels are not in the gradient** and are never
pretended to be differentiable.

**Held out of every training ensemble** and reserved for the audit: the audit slate's
own `logistic`, its two restarted `MLP[64,32]` trajectories under independent audit
seeds, both `HistGB` boosted-tree configurations, and the random-Fourier kernel ridge
family. At least one independent audit architecture is therefore out of training by
construction.

### 4.2 Locked slot schedule and matched attacker exposure

Every policy spends **five** role slots and the **same total attacker optimiser-step
budget**:

| policy | slots |
|---|---|
| `C1` | `A/public_coverage`, `A/SEX`, `A/RAC1P`, `AB/SEX`, `AB/RAC1P` |
| `L1`, `L2` | `A/public_coverage`, `A/SEX`, `A/RAC1P`, **replica** `A/SEX`, **replica** `A/RAC1P` |

The local policies spend their two coalition-equivalent slots on **independently
initialised local replicas** of the two sensitive local roles; `G_j` for those roles
then maxes over **six** attackers rather than three. This is a deliberate
strengthening of the local controls so that C1 is not merely the arm that received the
compute. It is locked here and its effect is disclosed in every comparison.

**`L2` matches `C1`'s nominal total group weight, not its gradients and not its
difficulty.** That limitation is reported wherever the contrast appears. Actual
per-role update counts are logged per fit.

### 4.3 Inner fitting and refresh

* **Attacker warmup**: 100 attacker minibatch updates against the unchanged channel
  before the mapper moves.
* **Alternation**: 5 attacker minibatch updates per mapper update.
* **Refresh**: at 25%, 50% and 75% of the mapper budget, one MLP per role slot is
  re-initialised and given a **bounded** catch-up of 40 updates. The refreshed and the
  incumbent attacker are compared on the **`monitor` fold** and the stronger (lower
  cross-entropy) is kept. Refreshed families, in order: `mlp64`, `mlp64_32`, `mlp64`.
* All of this stays inside representation-training households.

### 4.4 Sign convention, verified

The mapper minimises `U + beta * policy_penalty`; each attacker minimises its own
cross-entropy. **Stronger recovery raises the mapper's penalty and the mapper step
attempts to reduce it.** This is verified on a synthetic fixture in `VALIDATION.md`
and is visible in the training-only pilot recorded in `PROTOCOL.md` §7: with
`beta = 0` the monitor penalty **rises** 0.059 → 0.083 as the attackers learn, and at
`beta = 1.0` it **falls** 0.059 → 0.012. The two players are not reversed and the
service baseline is not being optimised in place of the channel.

---

## 5. The shared utility objective

Identical in every new neural arm:

```
U(theta) = mean_{income_binary, civilian_at_work} CE( Z-only head )
           + 1.0 * teacher_distortion(theta)                                   (4)

teacher_distortion = mean_{rows, coords} ( f_theta(x) − t(x) )^2 / var_ref     (5)
```

`t` is the **frozen starting channel at that width** (the warm-start release, which
head warming leaves bitwise unchanged). `var_ref` is the mean per-coordinate variance
of `t` on the **representation-fitting rows**, fixed once per `(seed, width)` and never
recomputed from an optimised channel. Coefficient **1.0**, fixed.

The source heads read **`Z` alone**. They never see `H`, so copying the service output
cannot trivially satisfy the auxiliary-capability objective. The heads train with the
mapper; **separately fitted probes are used for every final evaluation**.

### 5.1 Policies and the beta grid

```
local_agg = (1/3) [ G_{A/public_coverage} + G_{A/SEX} + G_{A/RAC1P} ]
ab_agg    = (1/2) [ G_{AB/SEX} + G_{AB/RAC1P} ]
```

| policy | penalty |
|---|---|
| `L1` | `local_agg` |
| `L2` | `2 * local_agg` |
| `C1` | `local_agg + ab_agg` |

`beta in {0.1, 0.3, 1.0, 3.0}`, in **this objective's** fixed scale. These are **new
coefficients and are not numerically comparable** to the historical `J` coefficients
(`-0.1`), to the spectral `lambda` grid, or to OptNet's multi-lambda form. The grid is
**not** adjusted after observing residence or the final attacks.

### 5.2 Budget, checkpoints and selection

* Mapper update budget **600**, checkpoints every **100** (7 checkpoints including the
  unmoved initial point). Set from the representative **training-only** timing pilot of
  `PROTOCOL.md` §7; the nominal budget is met in full and no reduction is applied.
* Identical budgets across matched policies and strengths. **Test performance is never
  used to extend a weak arm.**
* Optimisers: Adam, `lr 1e-3`, `betas (0.9, 0.999)`, `eps 1e-8`, batch 256 — the
  repository's own trainer settings.
* **Every checkpoint is saved**, together with the selected index and its evidence.
* **Checkpoint selection** uses one preregistered monitor score on the internal
  `monitor` fold:

```
monitor_score = U(theta)|monitor + beta * policy_penalty(theta)|monitor         (6)
```

  Each checkpoint is scored against **its own freshly initialised attacker slate**,
  trained for the **same fixed budget** of 300 attacker minibatch updates against that
  frozen channel on `mapper_fit`, and read on `monitor` (`RUN_STATUS.md` amendment 1).
  Equal budget across checkpoints is the point: neither the contemporaneous slate
  (weaker early) nor the final slate (specialised to the final channel, and therefore
  understating what is recoverable early) is a neutral yardstick, and **both biases
  point the same way**. The contemporaneous and final-slate scores are still computed
  and stored, and are used for the training-attacker-versus-fresh-auditor analysis.

  This is a **selection** yardstick, not a protection measurement: 300 fresh updates on
  three differentiable families is far weaker than the 2018 audit slate, and every
  protection claim rests on the audit, never on this number. **No residence, no
  commute, no downstream pool, no test pool.** Selection is `argmin`, ties to the
  earlier step.

Logged for every fit: role-wise gains, the active attacker per role, mapper gradient
norms, source losses, teacher distortion, refresh recoveries and decisions, attacker
warmup losses, per-role label support and clipping rates.

---

## 6. Erasure controls on the new channels

`LEACE` and `SPLINCE` are refit on **each width's no-protection channel**, because that
input channel is genuinely new. The **historical** `leace_A0` and `splince_A0` are kept
and are never replaced. Erasure operates on **`Z` only**; `H_A` and `H_B` are appended
unchanged, for the reason recorded in `BASELINE_ADAPTATIONS.md` §4 at `73903b7f`:
LEACE's `P*` is a single oblique projection over all coordinates and nothing constrains
it to act as the identity on `H_A`.

SPLINCE may preserve `income_binary` and `civilian_at_work` using their
representation-training labels; it **never** sees residence or commute. Its silent
LEACE fallback stays **disabled**: an infeasible arm is reported `SCOPED INFEASIBLE`.

**Width matching is not claimed when effective rank drops.** Ambient width, realised
projection rank and a compression sensitivity are reported for every erasure arm.

---

## 7. Endpoints, attacks and reporting — reused unchanged

Reused verbatim from the transport study and the invariant study, and labelled as
reused: five utility tasks; **eleven** forbidden roles
(`A/{public_coverage, commute_over20, SEX, RAC1P}`,
`B/{income_binary, civilian_at_work, same_residence, SEX, RAC1P}`, `AB/{SEX, RAC1P}`,
`RAC1P` on all nine classes); four family sensitive endpoints
(`recovery/A/SEX`, `recovery/AB/SEX`, `recovery/A/RAC1P`, `recovery/AB/RAC1P`) plus
`utility/same_residence`; the logistic / restarted-MLP / boosted-tree / kernel attack
families with their exact seed formulae; legal `H`-only ancestors and
singleton-to-coalition projections; validation-only selection by minimum **unweighted**
attacker-validation log loss then candidate ID; budgets 120 and 360 with primary 360.

**No new candidate family is introduced** for the new arms, so attack opportunities
are matched without adding anything to the controls. Attack scopes stay distinct:
historical catch-up exposure is never merged into the common fresh scope, and no
equal-total-history claim is made.

---

## 8. Boundaries

* 2018 and 2017 are **development** resources here. The historical first locked 2017
  result keeps its original status and is neither restated nor overwritten.
* **2016 remains sealed and unused.** No transformation, fitting, scoring, label
  inspection or candidate selection involving 2016 occurs in this study, and success
  here does not open it.
* Every number produced by this study is a **development** number on pools that have
  been used repeatedly. No computation on them can undo that.
