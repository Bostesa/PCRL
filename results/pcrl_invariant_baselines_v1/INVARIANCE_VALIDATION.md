# INVARIANCE_VALIDATION — the mathematics, checked before the outcomes

Every tolerance below was declared in `PROTOCOL.md` §4.1 and hashed into
`PROTOCOL_FREEZE.json` **before** any fit. Nothing here was loosened after seeing a
number. The one fixture expectation that was corrected is recorded, with its timing, in
`RUN_STATUS.md` failures item 1 — and it was a fixture expectation, not a measure.

Suite: `tests/pcrl_invariant_baselines_v1/` — **40 fixtures, all passing**
(`pytest tests/pcrl_invariant_baselines_v1 -q`).

---

## 1. The nine required checks

| # | Check | Result |
|---|---|---|
| 1 | Quadratic Frobenius identity vs packed `sqrt(2)` implementation | **PASS**, relative `< 1e-12`, against **two** independent paths: unpacking each symmetric `M` and taking `‖M‖_F^2`, and accumulating `M` directly by explicit summation |
| 2 | Kernel block vs an independently computed dense Gram formula | **PASS**, relative `< 1e-12`; plus a second check against an **explicit finite feature map** (linear kernel, where `phi` is the identity) confirming the `L = (BB')*(EE')` algebra |
| 3 | Objective, reconstruction and each penalty component under **many** rotations, sign flips and permutations, across ranks and seeds | **PASS** at `abs(delta) <= 1e-10 * max(1,abs(value))`. 16 transforms per cell (8 rotations, 4 sign flips, 4 permutations) x ranks {3,5,8} x seeds {0,1} |
| 4 | Analytic gradient vs central finite differences (`h = 1e-6`) | **PASS**, relative `< 1e-6`, 12 random directions |
| 5 | Near-zero derivative along pure rotation directions `W A`, `A` skew | **PASS**, `abs(<G,WA>) <= 1e-9 * ‖G‖_F ‖WA‖_F`, 8 skew directions per rank |
| 6 | Exact `H` parity and legal input dependence | **PASS**. `H_A`'s four coordinates and `H_B`'s two are asserted **bitwise** (`np.array_equal`) on all 7 pools x every arm at release time, and again at evaluation time against the frozen historical anchors. `transform(T, hA, arm)` reads `T` and `H_A` only |
| 7 | Magnitude leakage, XOR with side information, conditional-null with an oracle nuisance, deliberately misspecified nuisance | **PASS** — see §3 |
| 8 | Original fixed-matrix solver replay and old rank/alias behaviour | **PASS**, **bitwise**, all three seeds — see §2 |
| 9 | Repeated serial CPU evaluations and alternate chunk sizes on the same saved objects | **PASS**. Five repeated evaluations agree **bitwise** in-process; chunk sizes {64, 512, 4096, 100000} agree to relative `< 1e-12` |

A **rotation-share ratio is not the correctness gate.** It is unstable when the total
gain is tiny and is not a causal decomposition. The gate is the direct
absolute/relative invariance identities in checks 3 and 5.

---

## 2. Chaining to the historical baseline — bitwise

The closed-form original-moment start is asserted against the predecessor's stored arm
inside the fit itself (`seed_N/closed_form_replay.json`); the run **raises** if it is
not bitwise.

| Condition | Aliases | Bitwise identical | Max abs difference |
|---|---|---|---|
| `spectral_lin16_L1` | `spectral_L1` | yes | `0.0` |
| `spectral_lin16_L2` | `spectral_L2` | yes | `0.0` |
| `spectral_lin16_C1` | `spectral_C1` | yes | `0.0` |
| `spectral_lin8_{L1,L2,C1}` | — | yes | `0.0` |

All three seeds. The predecessor asserted the same maps against the historical
`spectral_*` arms, so this study is chained to the historical baseline exactly.

Frozen-input replay is also exact: the recomputed moment Grams match the stored
`matrix_diagnostics.json` traces with **maximum absolute error 0.0**.

---

## 3. The four falsification fixtures

Two of these exist to show the penalty can be **wrong**. Their expected behaviour was
declared in the fixture body in advance.

| Fixture | What a correct implementation must show | Result |
|---|---|---|
| **Magnitude leakage** | `S` depends on `abs(z)` only, so every *first* moment of `z` vanishes while both repaired blocks fire | first moment `< 0.02`; quadratic block `> 1e-4`; kernel block `> 1e-6`. **PASS** |
| **XOR with side information** | `Z` alone says nothing about `S`; `Z` interacted with `H` does. The intercept is retained so conditional cancellation is representable | `H`-interacted kernel block `> 3x` the constant-basis block. **PASS** |
| **Conditional null, oracle nuisance** | `Z ⊥ S \| H` by construction, so the blocks must concentrate towards zero — but finite-sample values are **not** exactly zero | **PASS**, with a structural distinction found while validating: see §4 |
| **Misspecified nuisance** | A wrong `m(H)` makes the penalty fire although `Z` adds nothing beyond `H` | quadratic `> 10x` oracle, kernel `> 5x` oracle. **PASS** |

The last fixture is the important one for interpretation: **a nonzero penalty is not
evidence of incremental disclosure.** The repair does not change this, and the
limitation is inherited in full.

A **control** fixture confirms the predecessor's quadratic block is still rotation
**sensitive** (it moves by `> 1e-6` relative under random rotations). Without that, the
repair would prove nothing.

---

## 4. A structural limitation found while validating

The two blocks concentrate in **different** variables, and this was not anticipated in
the registered method text. It was measured, recorded in `METHOD.md` §3.7 and
`RUN_STATUS.md` amendment 1, and the frozen subset size was **deliberately not changed**
in response.

* **Quadratic block** averages over all valid rows and concentrates in the pool size `n`:
  `4.9e-4` at `n=2000` -> `3.3e-6` at `n=128000`.
* **Kernel block** averages over the frozen subset and concentrates in `m`, **not** `n`:

| `m` | 32 | 64 | 128 | 256 | **512** | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| mean `D_kernel` under the conditional null | 1.11e-2 | 9.37e-3 | 2.77e-3 | 1.52e-3 | **7.13e-4** | 4.36e-4 | 2.28e-4 |

Decay is roughly `1/m`, as expected for a V-statistic whose retained diagonal
contributes an `O(1/m)` bias. **Consequence:** the kernel block cannot be driven below
its own `m`-dependent floor, for reasons unrelated to disclosure, so a small nonzero
fitted value is not evidence of residual conditional dependence. The floor applies
equally to every arm, policy and rank, so it does not bias comparisons *between* arms;
it bounds what an *absolute* block value can mean.

---

## 5. The predeclared mechanism gate — the headline mechanism result

Computed on the fitted maps **after fitting and before any 2018 or 2017 score was
read** (`MECHANISM_GATE.json`).

| | Predecessor (defective) | **This study (repaired)** |
|---|---|---|
| Rotation-only share of the training gain | mean **0.628**, range 0.367–0.907 | **max abs 1.4e-15** across all 18 cells |
| Gate (`< 0.10`) | would fail | **PASSES, 18/18** |
| Registered forecast Q1 (`< 0.01`) | — | **MET** |
| Direct invariance identity, max abs change in the penalty under a random rotation | large by construction | **5.6e-17** |

Per-cell rotation-only shares are `0.0` exactly in 13 of 18 cells and `O(1e-15)` in the
other 5 — floating-point noise, not a measured quantity. The rotation-only descent finds
nothing because there is nothing to find: the repaired objective is **constant on the
rotation orbit**, so the line search fails immediately at zero improvement.

**Forecast Q1 is confirmed, and it is confirmed structurally rather than empirically.**
The repaired objective is a function on the Grassmannian: it depends only on
`span(W)`, exactly as the information content of the release `Z = V W` does. On real
ACS data at rank 16, `abs(L(WQ) - L(W)) = 1.1e-16`.

**What this does and does not establish.** It establishes that the 63% of the
predecessor's surrogate movement that was provably inert has been removed by
construction. It does **not** establish that the penalty captures all sensitive
information, that the optimiser reaches a useful point, or that anything about measured
disclosure has improved. Those are separate questions and the attack evaluation
answers them.

---

## 6. Optimiser health

All 18 repaired conditions, three seeds:

| Quantity | Result |
|---|---|
| Returned the unmoved initial point | **0 of 18** |
| Feasibility `max abs(W'W - I)` | `<= 6.7e-16` (declared tolerance `1e-13`) |
| Training-objective gain | 0.018–0.128 |
| Selected start | 9 of 18 `original_moment_spectral`, 9 of 18 `perturbed_retracted` — genuine restart spread |
| Degeneracy flags at any role, seed or rank | **none** |
| Released covariance `max abs(Z'Z/n - I)` | at machine precision (whitening is exact for orthonormal `W`) |

Class support is **recorded, not repaired**. `RAC1P` class index 3 has population
support 1, 1 and 0 rows in seeds 0, 1 and 2, so it has **zero** positive support on
every 512-row kernel subset. Such a class still contributes through its residual
`e_c = -m_c(H)`. This is reported per role in `seed_N/fit_diagnostics.json` and was
never rebalanced, stratified or padded.

---

## 7. What the validation does not cover

* Nothing here is a privacy certificate. Vanishing fitted finite moments imply neither
  `Z ⊥ S | H` nor any bound on `I(S;Z|H)`, and no calibration is inherited from KCI or
  RCoT.
* Rotation invariance is a property of the **objective**, not evidence that the
  objective measures the right thing.
* The kernel block is exact **for the declared 512-row subset**, not for the ACS
  population. It is never described as an exact full-data kernel objective.
* Nuisances were held fixed by design. The misspecification fixture shows a wrong
  `m(H)` manufacturing a penalty more than ten times the oracle value under *exact*
  conditional independence.
* The repaired objective is nonconvex. No global-optimality statement is made for it;
  the only such statement in this study is for the unchanged `original` family on its
  fixed matrix.
