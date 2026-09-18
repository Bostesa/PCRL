# VALIDATION — what was checked, and what the checks cannot cover

The mathematics of the repaired objective is validated separately and in detail in
[`INVARIANCE_VALIDATION.md`](INVARIANCE_VALIDATION.md). This file covers the **run**:
integrity, reuse discipline, and the boundaries of every claim.

## 1. Test suite

`pytest tests/pcrl_invariant_baselines_v1 -q` — **40 fixtures, all passing**, at
tolerances declared in `PROTOCOL.md` §4.1 and hashed before any fit.

| Group | Fixtures | Covers |
|---|---:|---|
| Packed Frobenius identity | 2 | packed `sqrt(2)` block vs two independent `‖M‖_F^2` paths; the predecessor's defect reproduced |
| Exact kernel block | 4 | dense-Gram reference; explicit finite feature map; V-statistic diagonal convention; frozen bandwidth multiples |
| Invariance | 8 | many rotations, sign flips, permutations across ranks {3,5,8} and seeds {0,1}; reconstruction; a **control** confirming the predecessor is still rotation-sensitive |
| Gradients | 3 | central finite differences; near-zero rotation-direction derivative; closed form of `METHOD.md` eq. (14) |
| Solver | 3 | closed-form global optimum of the unchanged `original` family; solver pieces asserted **identical objects** to the predecessor's; descent monotone and feasible |
| Numerical stability | 3 | chunk-size invariance; repeated serial evaluation; frozen subset reproducibility |
| Support handling | 1 | subset support recorded, never manufactured |
| LEACE | 3 | generalised recipe reproduces the verified two-attribute wrapper to `2e-11`; exact predicted rank loss; **guarantee is linear-only — a quadratic probe still recovers** |
| SPLINCE | 2 | both published constraints hold; infeasibility detected, not silently relaxed |
| OptNet-ARL | 4 | closed-form player vs explicit ridge solve; differentiation vs finite differences; Theorem 4.1 count without forming an `n x n` matrix; frozen deterministic architecture |
| Falsification | 4 | magnitude leakage; XOR with side information; conditional null; **misspecified nuisance manufacturing a penalty** |

## 2. Chain to the historical baseline — bitwise

The closed-form original-moment start is asserted **inside the fit**, which raises
rather than continuing if it fails. All three seeds, all six conditions:
`bitwise_identical: true`, `max_abs_difference: 0.0`. At rank 16 these arms **are** the
historical `spectral_L1/L2/C1`. Frozen moment-Gram replay is also exact (max abs error
`0.0`).

## 3. Reuse discipline

**33 unique new mapper fits.** Nothing else was fitted, and nothing already scored was
rescored.

| Reused, never refitted or rescored | Count |
|---|---:|
| Historical simple interfaces `H, E, A0, L025, L20, J` | 6 |
| Historical `spectral_*` arms | 8 |
| Predecessor `spectral_lin8_*`, `spectral_nlr{8,16}_*` | 9 |
| Rank-16 original arms, by proved alias | 3 |
| **SARL duplicate fits deliberately avoided by the alias audit** | **6** |

## 4. Numerical integrity

| Check | Result |
|---|---|
| Independent score replay (separate recomputation of every selected per-person loss) | **3264 checks, max abs difference `4.44e-16`** |
| `H_A` / `H_B` parity | **bitwise** (`np.array_equal`) on all 7 pools x every arm, at release time **and** again at evaluation time against the frozen historical anchors |
| Historical subset-index hashes | asserted equal to the stored study's, per seed; a mismatch aborts |
| Feasibility `max abs(W'W − I)` | `<= 6.7e-16` against a declared `1e-13` |
| Concurrency | one fitting worker; audit suites never concurrent with a model job |
| Inconsistent outputs | none observed; none quarantined |

No numerical disagreement was averaged, smoothed or discarded.

## 5. Machine conditions

Apple CPU, 14 cores, 24 GB. **Swap was at capacity for the entire run** (28.2 GB of
29.7 GB used at start; 23% free memory; load average 6.2), with unrelated
`removal-pricing` jobs running throughout and never touched. Peak RSS **369 MB** for
fitting.

The predecessor associated rare load-dependent prediction faults with this condition.
**No such fault occurred in this run.** Memory pressure remains a **suspected**
contributing condition there, not a demonstrated cause, and nothing here upgrades it.

## 6. Ordering that the repair claim depends on

The mechanism gate was computed **after fitting and before any 2018 or 2017 score was
read**. That ordering is recorded in `MECHANISM_GATE.json`
(`"timing": "computed after fitting and BEFORE any 2018 or 2017 score was read"`), was
requested independently by Terminal 2, and is the reason the repair claim is supportable
at all. The fit and diagnostic modules cannot reach an attacker, a residence label, a
commute label or a transport table.

## 7. Errors found and corrected during this run

All recorded with timing in `RUN_STATUS.md`. None changed a fitted map.

1. **A fixture asserted the wrong thing; the code was correct.** The conditional-null
   fixture expected both blocks to concentrate in the pool size `n`. The kernel block
   concentrates in the **subset size `m`** instead, by construction. The fixture was
   split and the underlying property measured and documented as a limitation
   (`METHOD.md` §3.7). `m = 512` was deliberately **not** changed in response.
2. **A report-only OptNet diagnostic was missing a `1/n`.** `utility_signal_ratio`
   recorded `10513.0` where the quantity is an R-squared; the corrected value is
   **1.000**. The field enters no objective, no selection and no fitted map.
3. **A sign convention was described backwards in the document generator.**
   `utility/same_residence` is a **loss** difference, so positive means *less* residence
   capability. Caught by checking the generator's description against the decision rule
   (`residence_difference <= .001` is the pass condition) rather than against intuition,
   and fixed before the report was published.
4. `KernelBlock.build` could not accept synthetic role names; `role_index` is now an
   explicit optional parameter. Production paths unchanged.

## 8. What the validation does **not** cover

* **Nothing here is a privacy certificate.** Vanishing fitted finite moments imply
  neither `Z ⊥ S | H` nor any bound on `I(S;Z|H)`. No calibration is inherited from KCI
  or RCoT.
* Rotation invariance is a property of the **objective**, not evidence that the
  objective measures the right thing. The 2018 results are precisely a case where an
  exactly-correct objective produced worse measured outcomes.
* The kernel block is exact for the **declared 512-row subset**, not the ACS population,
  and has an `m`-dependent floor.
* Nuisances were held fixed. A misspecified `m(H)` manufactures a penalty more than ten
  times the oracle value under **exact** conditional independence, so a nonzero penalty
  is **not** evidence of incremental disclosure.
* All intervals are **development** uncertainty conditional on fitted systems. They do
  not undo repeated use of the 2018 pools and are not replicate-weight or retraining
  variability.
* Three seeds cannot resolve seed-to-seed variability on race recovery.
* Baseline guarantees cover **only the transformed channel**, never the augmented
  release, which still contains `H`. LEACE's own authors conjecture their guarantee does
  not extend to general nonlinear adversaries, and this study's own fixture shows a
  quadratic probe recovering from an erased channel.
* The OptNet multi-lambda multi-attribute form is an **extrapolation** not present in
  its paper, and its mini-batch projector is approximate.
* This is **not** a comprehensive benchmark of every cited method.
* **2016 was not scored, fitted, transformed, tuned on or inspected.** No code path in
  this study can reach it.
