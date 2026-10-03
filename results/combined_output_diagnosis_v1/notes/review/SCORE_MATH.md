# Score decomposition: identities, proofs and numerical precision

**Reviewer note.** I am the read-only mathematics and protocol reviewer, writing on 2026-10-03 against draft PROTOCOL.md
on `research/combined-output-diagnosis-v1` (cut from f7425b15).

- These are elementary, established identities. Nothing here is a new method.
- Every numerical claim is reproduced by `score_math_checks.py`, which uses synthetic arrays only. Its output is in
  `score_math_checks.out`; excerpts are quoted below.
- §7 gives aggregate-only facts about the stored logits. They were computed read-only from the admitted forward
  caches. No per-person value appears.

Notation:
- l = (l0, l1) are the binary logits.
- d = l1 − l0 is the margin.
- c = (l0 + l1)/2 is the offset.
- σ(x) = 1/(1 + e^{−x}) is the logistic function.
- softmax(l)_k = e^{l_k} / Σ_j e^{l_j}.
- S is the sensitive attribute, Y is the task label, and R is a release.

---

## 1. (l0, l1) ↔ (d, c) is an invertible linear map

**Claim.** (d, c)ᵀ = A (l0, l1)ᵀ, where A = [[−1, 1], [½, ½]] and det A = −1. The inverse is
l0 = c − d/2 and l1 = c + d/2.

**Proof.**
- det A = (−1)(½) − (1)(½) = −1 ≠ 0.
- Substituting: c − d/2 = (l0 + l1)/2 − (l1 − l0)/2 = l0, and c + d/2 = l1. ∎

The two coordinates are orthogonal but not orthonormal: the row norms are √2 and 1/√2. Consequences:
- An L2-penalised or standardised attacker fitted on (l0, l1) is a different fitted model from one fitted on (d, c).
- Trees are not rotation-invariant either.
- Only the *composition* "fitted model ∘ inverse map" is guaranteed to give identical predictions.

**Floating point.**
- For general float64 inputs, the round trip c ∓ d/2 is exact only up to about 2 ulp. On 10⁶ random rows, 631,318
  were not bit-exact (max error 1.8e−15).
- The stored logits are float64 **upcasts of float32** forward values. For those inputs, l1 − l0 and l0 + l1 are
  exact in float64: their 24-bit significands fit in 53 bits unless the exponents differ by more than about 28. Halving
  is exact. So the round trip is bit-exact:

```
float64 inputs: max |roundtrip - l| = 1.776e-15  rows not bit-exact: 631318
float32-upcast inputs: rows not bit-exact: 0
```

On the real stored binary heads, the round trip was bit-exact on every row (§7).

## 2. Probabilities, decisions and proper losses depend on d only

**Claim.** softmax(l) = (σ(−d), σ(d)).

**Proof.**
- e^{l1} / (e^{l0} + e^{l1}) = 1 / (1 + e^{l0 − l1}) = σ(d).
- The other coordinate is 1 − σ(d) = σ(−d). ∎

Consequences:
- **Decision.** argmax(l) = 1[d > 0], with numpy's tie rule. On ties l0 = l1, `argmax` returns index 0, and
  1[0 > 0] = 0.
  - In IEEE arithmetic, fl(l1 − l0) = 0 iff l1 = l0 (gradual underflow).
  - sign(fl(l1 − l0)) = sign(l1 − l0) (correct rounding preserves sign).
  - So the identity holds bit-exactly, not just mathematically.
- **Every probability-based quantity is a function of d.** This includes:
  - any proper scoring rule (log loss −log σ((2y−1)d), Brier, and so on);
  - calibration;
  - every threshold rule;
  - ranking, and therefore task ROC-AUC.

  The offset c enters none of them.
- **Hard decision.** It is a function of sign(d) only.

```
max |softmax(l)_1 - sigma(d)| = 1.67e-16   max |softmax(l)_0 - sigma(-d)| = 1.67e-16
argmax(l) != 1[d>0] rows (incl. 2 exact ties): 0
```

**Caveat.** "Depends on d only" does not mean "d is necessary". A recipient who needs only decisions needs only
sign(d). Which contract a recipient needs is a design choice and must be stated (assignment, Stage 1).

## 3. Subtracting the offset leaves softmax unchanged (binary)

**Claim.** softmax(l − c·1) = softmax(l), and l − c·1 = (−d/2, d/2).

**Proof.**
- Softmax is invariant to adding the same constant to every coordinate:
  e^{l_k − c} / Σ_j e^{l_j − c} = e^{l_k} / Σ_j e^{l_j}.
- Also, l0 − c = (l0 − l1)/2 = −d/2 and l1 − c = d/2. ∎

Numerically:
- `scipy.special.softmax` subtracts the row max, giving (−|d|, 0) from either input. So the two probability vectors
  were bit-identical on 10⁶ random rows (max difference 0.0).
- (−d/2, d/2) and (l0 − c, l1 − c) can differ in the last bit for general float64 inputs. They were identical on the
  float32-upcast real data (§7).
- **Recommendation:** define the centred surface as (−d/2, d/2) computed from d. Then it is a function of d
  bit-exactly.

## 4. Multiclass analogue

For K classes:
- the offset is c = (1/K) Σ_k l_k;
- the centred vector is l̃ = l − c·1, which satisfies Σ_k l̃_k = 0 and so has rank K − 1.

**Claims.**
1. **Reconstruction.** l = l̃ + c·1, so (l̃, c) ↔ l is invertible. It is a linear bijection between ℝᴷ and
   {Σ l̃ = 0} × ℝ.
2. **Softmax invariance.** softmax(l̃) = softmax(l), for the reason given in §3.
3. **Same decision.** argmax(l̃) = argmax(l).
4. **Equivalent non-redundant coordinates.** The K − 1 differences (l_k − l_0), k ≥ 1, are an invertible linear
   recoding of l̃: l̃_k = (l_k − l_0) − (1/K) Σ_j (l_j − l_0).

**Proof.** Each claim follows by direct substitution, using softmax translation invariance. ∎

**Floating point.** Centring by the mean is bit-exact only when the division by K is exact (K a power of two) and the
sum is exact:

```
K=3: max|softmax diff|=2.2e-16 max|sum centred|=1.8e-15 recon rows not bit-exact=18330 max recon err=8.9e-16
K=4: max|softmax diff|=0.0e+00 max|sum centred|=0.0e+00 recon rows not bit-exact=0     max recon err=0.0e+00
K=5: max|softmax diff|=4.4e-16 ...                       recon rows not bit-exact=25423 max recon err=8.9e-16
K=6: max|softmax diff|=4.4e-16 ...                       recon rows not bit-exact=33063 max recon err=8.9e-16
```

So "(centred, c) reconstructs l" holds to about 1 ulp, not bit-exactly, for the 5-class and 6-class heads. The study
should report the maximum reconstruction error rather than assert bit equality (real data: §7).

### 4a. The common refitted head HEAD__A outputs log-probabilities

If the released vector is λ = log softmax(z), then:
- the margin is unchanged: λ1 − λ0 = z1 − z0 = d;
- the offset is a fixed even function of d:

c_λ = −½[softplus(d) + softplus(−d)] = −½[|d| + 2 log(1 + e^{−|d|})].

So for this stratum the full release is (d, g(d)): a deterministic function of the centred release. Full and centred
carry identical information *by construction*, and the draft protocol says so correctly.

More generally, for multiclass log-softmax the offset is mean_k λ_k = −LSE(l̃), which is a function of l̃.

```
log-prob head: max|d_logprob - d| = 8.9e-16  max|c - g(d)| = 3.6e-15  max|g(d)-g(-d)| = 0.0
```

## 5. Information versus fitted-attacker AUC

**(a) Data processing.** For any deterministic measurable f, I(S; f(R)) ≤ I(S; R).

- **Proof.** S → R → f(R) is a Markov chain, so the data-processing inequality applies.
- **Equality** holds iff f(R) is sufficient for S, that is, iff S ⫫ R | f(R).
- **Consequences:**
  - I(S; centred) ≤ I(S; full).
  - I(S; hard) ≤ I(S; centred).
  - I(S; prob) ≤ I(S; centred), with equality in the unsaturated binary case, where prob and centred are bijective
    (§6).
- **Population-optimal AUC is also monotone under garbling.** The Bayes score P(S = k | R) has the
  likelihood-ratio-optimal ROC (Neyman–Pearson). Any score of f(R) is also a score of R. So for each one-vs-rest
  class, AUC*(f(R)) ≤ AUC*(R), and the same holds for the macro mean.

**(b) Fitted attackers do not inherit this ordering.** A finite-sample, validation-selected attacker on R can score
lower *or* higher than one on f(R). Typical causes:
- inductive bias, for example axis-aligned trees that cannot form d from (l0, l1) cheaply;
- overfitting to irrelevant coordinates;
- the limits of the hyperparameter grid.

Synthetic demonstrations, both removing deterministic functions of (l0, l1):

```
stump GBT on (l0,l1,d): 0.940  -> remove d (a function of (l0,l1)): 0.917      # removal LOWERS fitted AUC
LR (n_fit=300) on (l0,l1,junk(l)): 0.812  -> remove junk: 0.963                # removal RAISES fitted AUC
```

So a measured R(full bank) > R(centred) is *consistent with* the offset carrying S-signal, but does not prove it. A
measured R(full) < R(centred) does not contradict (a). This is why the bank must contain the ignore-offset
candidates, and why a selected candidate that ignores the offset must be labelled (STATS_REVIEW §4).

**(c) Information orderings do not imply AUC orderings between non-nested releases, even at the optimum.** Consider
two releases for a balanced binary S:
- **Release A** reveals S exactly with probability 0.1 and is otherwise uninformative.
- **Release B** is a binary symmetric channel with flip probability 0.4.

```
A: I=0.1000 bits, Bayes AUC=0.595;   B: I=0.0290 bits, Bayes AUC=0.600
```

A has more than three times the information of B, but a lower optimal AUC.

The study's endpoints are differences in *fitted* AUC. They should never be described as information differences, and
"lower recovery" should never be read as "less information".

**(d) Fixtures for the planned controls.** If the offset encodes S and the margin encodes only Y, centring removes all
of S's information:

```
fixture offset-carries-S: I(S;full)=0.5000 bits  I(S;centred)=0.0000 bits
```

The converse fixture is the margin carrying all of S with an independent offset. There I(S; full) = I(S; centred), and
any measured FC > 0 is pure attacker variance or parameterisation.

## 6. A public fixed head on released features adds no information

**Claim.** Let Z be the released features and h a fixed, publicly known deterministic map. Then
I(S; Z, h(Z)) = I(S; Z).

**Proof.**
- By the chain rule, I(S; Z, h(Z)) = I(S; Z) + I(S; h(Z) | Z).
- Since H(h(Z) | Z) = 0, the conditional term is 0. ∎

```
public fixed head: I(S;Z)=0.049022  I(S;Z,h(Z))=0.049022
```

**Scope:**
- **What "fixed" requires.** If h is fitted from data, the claim holds conditionally on the fitted h. That requires
  h to be independent of the person being attacked, for example trained on other records, and to be known to the
  attacker.
- **It covers the "own-head" view** (features + head fitted on those features).
- **It does not cover features + the historical clean logits** when the features are treated: the clean logits are a
  function of the *untreated* representation, not of Z. That is exactly the output bypass.
- **It can still raise fitted-attacker AUC.** h(Z) can be a useful engineered feature (§5b), so a measured increase
  in the complete view is not a "new information leak".

## 7. Saturation in floating point

Let d* be the smallest double for which a given implementation returns exactly 1.0. The thresholds below were found
by bisection over representable floats.

| Quantity | float64 | float32 |
|---|---|---|
| Largest value below 1 | 1 − 2⁻⁵³ = 0.99999999999999988898 | 1 − 2⁻²⁴ |
| σ(d) or softmax p1 returns exactly 1.0 (scipy `expit`, naive 1/(1+e^{−d}), scipy `softmax`) | d ≥ 36.73680056967711 = 53 ln 2. Then e^{−d} ≤ 2⁻⁵³ and 1 + e^{−d} rounds to 1. | d ≥ 16.635532 = 24 ln 2 |
| Correctly-rounded σ(d) would round to 1 | d ≥ ln(2⁵⁴ − 1) = 37.4299. The common implementations saturate about 0.7 earlier. | d ≥ 25 ln 2 = 17.33 |
| σ(d) returns exactly 0.0 via the exp-overflow path (scipy 1.17 `expit`, naive formula) | d ≤ −709.7827 | d ≤ −88.72 |
| σ(d) returns exactly 0.0 via the stable path (e^{d}/(1+e^{d}), scipy `softmax`) | d ≤ −745.1332 = ln 2⁻¹⁰⁷⁵. Subnormal (reduced precision) from d ≤ −708.3964. | d ≤ −103.97 |

**Correction to the draft (REQUIRED).** The draft says "|d| ≥ 36.7 in float64 gives p1 ∈ {0, 1}". That is wrong on
the negative side. The behaviour is asymmetric:
- p1 rounds to exactly **1** for d ≥ 36.737;
- p1 rounds to exactly **0** only for d ≤ −709.78 or d ≤ −745.13, depending on the implementation;
- for −709 < d < −36.7, p1 is a tiny but exact-enough number from which d is fully recoverable.

**When is the margin recoverable from a probability release?**
- **From p1 alone** (with p0 implied as 1 − p1), resolution degrades before saturation. The spacing of doubles
  below 1 is 2⁻⁵³, so the recovery error is about 2⁻⁵³·e^{d}. Recovery is impossible once p1 = 1.0:

  ```
  d=20: |d - logit(p1)|=3.6e-08   d=25: 4.2e-06   d=30: 1.0e-03   d=36.7: 0.66   d=36.8: inf
  ```

- **From both stably computed probabilities (p0, p1)**, d = log p1 − log p0 is exact (error 0.0 in the test) for
  every |d| < 708. It is lost only when the smaller probability underflows to 0.

So "probabilities determine the margin" holds exactly in arithmetic. In float64 it holds whenever the release stores
both probabilities and the smaller one is ≥ 2⁻¹⁰²² (|d| ≲ 708). If only p1 is stored, it holds only for d ≲ 25–30.

**Real stored logits** (aggregate, read-only; all values are float64 upcasts of float32):

| Head | Facts |
|---|---|
| Adult income_prediction (primary) | max \|d\| = 3.82 / 18.45 / 17.47 for seeds 0 / 1 / 2. **No row reaches the float64 threshold.** 92 rows (s1) and 11 rows (s2), out of 39,205, have \|d\| ≥ 16.6, so they **would saturate if any probability were computed or stored in float32**. |
| HMDA underwriting (primary) | max \|d\| ≤ 4.13. No saturation in either precision. |
| Both primary heads, all seeds | (d, c) round trip bit-exact; centred (−d/2, d/2) = l − c bit-exact; max \|softmax₁ − σ(d)\| ≤ 1.7e−16; argmax ≠ 1[d > 0] on 0 rows. |
| Adult education_assessment (4-class) | Centred-logit range up to 4,304. **In float64, softmax gives max p exactly 1.0 on 29.5 % (s1) and 28.1 % (s2) of all rows, and at least one probability exactly 0 on 28.4 % / 21.4 %** (a logit gap above 745). On those cells the probability surface is a *strict* garbling of the centred logits, not a reparameterisation. |
| Adult employment_analysis (6-class) | Pmax = 1.0 on 0.0–0.3 % of rows. |
| HMDA pricing_analysis (5-class) | No saturation. Mean-centring reconstruction is off by about 1 ulp on 16k–37k rows of 77k (K not a power of two). |

Implications:
- **(i)** For the primary binary cells, prob and centred are information-equivalent on every row. So any assessment
  difference R(centred) − R(prob) is attacker parameterisation, not information. The previous study's full-vs-prob gap
  (0.07) therefore splits into an offset part (full − centred) and a parameterisation part (centred − prob). Only the
  first is about the offset.
- **(ii)** For S3, Adult education seeds 1 and 2 have materially saturated probabilities. There R(centred) − R(prob)
  is *not* only parameterisation. The Stage 1 log loss must be computed as −log_softmax(l)[y] in float64, which is
  finite for finite logits, not as log(clip(softmax)). Otherwise misclassified saturated rows give infinite or clipped
  losses whose value depends on an arbitrary ε.
- **(iii)** All probability surfaces should be computed and stored in float64, and the dtype recorded. A float32 path
  (torch, some GBT libraries, float32 caches) would saturate 103 Adult income rows.

## 8. Summary of what the identities do and do not license

| Statement | Status |
|---|---|
| Centred logits preserve the probability vector, decision, accuracy, log loss, calibration and ranking exactly | **True** (§2–4). Holds to ≤ 1 ulp in float64. Bit-exact on the stored binary heads. |
| Centred logits are information-equivalent to probabilities | True for unsaturated binary heads (all primary rows). **False** for Adult education seeds 1–2 (§7). |
| Removing the offset cannot increase I(S; release) | **True** (§5a) |
| Removing the offset cannot increase *fitted* recovery | **Not implied** (§5b) |
| A lower fitted AUC means less information | **Not implied** (§5c) |
| Hard decisions preserve accuracy | **True by construction.** Says nothing about proper loss, calibration or ranking. |
| A public fixed head on the released features adds no information | **True** (§6), but can raise fitted AUC. It does not apply to clean logits released alongside treated features. |
