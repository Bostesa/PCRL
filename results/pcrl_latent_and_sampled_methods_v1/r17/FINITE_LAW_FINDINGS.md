# R17 independent finite-law checks

Status: data-free engineering checks for the new latent and sampled-method portfolio. These fixtures use exact rational enumeration except for natural-log NLL. They do not use Adult rows, candidate math, fitting code, or a fresh privacy attack. A toy advantage is a capability check, not an Adult result or a privacy guarantee.

Run from the portfolio worktree:

```sh
python3 -m unittest discover -s results/pcrl_latent_and_sampled_methods_v1/r17 -p 'test_finite_laws.py' -v
```

## Hard public-code laws

For binary codes, the oracle uses `kρ(x,x′)=ρ^Hamming(x,x′)`, with `ρ=1/2`. Empirical MMD² includes ordered within-group pairs and diagonals. The coalition kernel is the product of the two recipient kernels on *aligned* public-code pairs. This rational kernel is an independent miniature check of the proposed hard-code and product-kernel geometry, not a claim that it is the campaign's frozen bandwidth choice.

| Law | Recipient 1 MMD² | Recipient 2 MMD² | Pair MMD² | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Balanced null: all `(SEX,a,b)` combinations | 0 | 0 | 0 | No protected distinction in any view. |
| XOR: `SEX=a⊕b`, each `(a,b)` once | 0 | 0 | 1/4 | Both marginals conceal SEX; the aligned pair reveals it perfectly. |
| No complementarity: `SEX=a`, `b` independent | 1 | 0 | 3/4 | Recipient 1 already reveals SEX; the second code adds no recovery. MMD magnitudes across different kernels/views are not comparable information quantities. |
| Removable bit: code `(task,SEX)` becomes `(task)` with independent balanced task | positive | — | — | The task remains exactly decodable while local hard-code MMD becomes zero. |

For XOR with a two-scale recipient kernel averaging `ρ∈{1/2,1/4}`, the product pair-kernel MMD² is **25/64**, while both local MMD² values remain zero. Algebraically, a one-scale XOR pair has `MMD²=(1−ρ)²`. For R06's proposed four-scale kernel `k(a,b)=¼Σγ exp(−γ Hamming(a,b))`, `γ∈{0.25,0.5,1,2}`, the XOR pair MMD² is `(1−¼Σγ exp(−γ))² ≈ 0.27863983050495`; both local values remain zero. This uses the *product of averaged recipient kernels*. A local-only term cannot detect this complementary clue. MMD is still a fitted-law surrogate: a small empirical value need not prevent fresh nonlinear recovery.

## Sampled scores and public identities

For one draw from `(.9,.1)` or `(.6,.4)` with equal probability and true class 0:

| Scored release | NLL | Summed multiclass Brier |
| --- | ---: | ---: |
| Actual sampled output, expectation over draws | `−½ log(.54)` ≈ 0.308093 | 0.170 |
| Unsampled mean vector `(.75,.25)` | `−log(.75)` ≈ 0.287682 | 0.125 |

Thus an evaluator that scores the mean vector understates the actual one-draw Brier by **0.045** and NLL by about **0.020411** in this law. The mean is not the recipient's released vector. A separate 20-row calibrated-confidence law has two contexts with the same predicted class but class-1 frequencies 0.1 and 0.4; context-specific calibrated outputs beat the decision-only common vector `(.75,.25)` on both NLL and Brier. This demonstrates why decision-only privacy cannot automatically qualify for the confidence requirement.

Deduplication occurs **before** per-input action normalization. For listed choices `[A,A,B]` with unit raw masses, the public choice law is `(A:1/2,B:1/2)` under the oracle's fixed first-occurrence convention. Normalizing three local slots first gives `(A:2/3,B:1/3)` and changes the channel. The implementation must freeze its own duplicate/tie convention before fitting; R17's convention is a checkable example, not a candidate training rule.

Global public IDs are a function of the fixed emitted probability vector and its deterministic first-argmax decision, with globally duplicate vectors merged. In the fixture, two source tokens list `[A,B]` and `[B,A]`; one chooses local slot 0 and the other local slot 1, but both emit `A` with the **same global ID**. A source-specific ID or local action position would reveal a distinction absent from the promised release. The oracle averages each person's public action law within SEX groups and obtains identical group laws in this swapped-slot case, while the local action positions themselves differ by group. Joint action masses are summed by the actual `(public ID 1, public ID 2)` pair: two `1/4` actions yielding the same pair give it mass `1/2`. This aggregation must precede the empirical-law privacy objective and complete-view attack representation. The fitted implementation should canonicalize its fixed public bank from exact stored vector values and a frozen tie rule, not a post hoc approximate-equality tolerance. If any per-person metadata varies and is transmitted, it also belongs in public identity; the present contract permits only globally fixed model/version metadata.

The independent oracle does not certify population privacy, classifier robustness, optimization quality, utility feasibility, or the real-data candidate. R04's final replay should check candidate outputs and public-ID values against the locked release interface, and should independently recompute sampled losses from actual draws.
