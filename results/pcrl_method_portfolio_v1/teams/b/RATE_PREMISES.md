# Team B rate and frozen-law premises

**Scope:** prefit mathematics for the declared ideal sampler and S1's frozen five finite laws. This is not a fitted-model result, a SEX-AUC guarantee, or a theorem about a particular saved batch. S1's `shared/SYNTHETIC_LAWS.json` is frozen at SHA-256 `b2e68f5dec7dd690afa9389c44c0c2f98a78b30cde25c20a48751665f35fdc53`. The exact enumeration checker is separately frozen at SHA-256 `43de24d94b9da09e62f417e1a3506075a1d79459bd0bc087406df51d7f7fee93`. S3 admitted and ran that checker as `B-FROZEN-LAWS-009`; it exited 0. This verifies the analytic law definitions, not the neural optimizer or an Adult outcome.

## What the Gaussian KL actually bounds

Let `X` follow a specified distribution, and let ideal private Gaussian draws satisfy `q(z1,z2|x)=q1(z1|x)q2(z2|x)`. With `r_i=N(0,I)` and `q_Z` the induced joint marginal,

```
E_X[ KL(q1(.|X)||r1) + KL(q2(.|X)||r2) ]
  = E_X KL(q(.|X)||r1*r2)
  = I(X;Z1,Z2) + KL(q_Z || r1*r2)
  >= I(X;Z1,Z2).
```

The first equality needs conditional independence of the ideal sampler noises given `X`; the second is the KL chain identity. If the encoder reads only `X` and independent sampling randomness, `S-X-(Z1,Z2)` is a Markov chain, so `I(S;Z1,Z2) <= I(X;Z1,Z2)` by data processing for that same distribution. This is a broad, usually loose information bound. A nonzero rate can still permit high SEX AUC. The fitting-row average KL is an empirical objective and does not certify the population rate, finite-attacker recovery or the held-out privacy target.

The deployed release is a fixed artifact generated from HMAC-derived Torch pseudorandom streams and a private key. Domain separation is a reproducibility and engineering measure, not exact Gaussian independence. A deterministic keyed batch has no unconditional information-theoretic rate certificate from the displayed identity. The actual sampled feature/head/metadata views must therefore be attacked afresh.

If a public head and calibration are deterministic functions of released `z_i`, then under the ideal law and **constant** metadata they add no Shannon information beyond `z_i`. A person-varying flag, fallback, length, identifier or sampler state can add information and must be included in the measured complete view. `B-DVIB` releases deterministic real means: its Gaussian KL term is mean-code regularization, not an upper bound on the information in those means.

## What the five frozen laws test

| Law | Exact premise | Failure it exposes |
|---|---|---|
| `L1_FEASIBLE_NULL` | `S`, `B1`, `B2` are independent fair bits. The two useful class scores disclose only their own `B_i`, so local and pair ideal SEX recovery are at chance. | A pipeline reporting privacy gain below the decision/view floor or spurious SEX recovery is wrong. |
| `L2_REMOVABLE_DISTINCTION` | Each input includes `(B_i,S)` while `Y_i=B_i`. Dropping `S` preserves task labels and class decisions and makes both views SEX-independent. The original fine scores depend on `S`. | Teacher-score imitation can fight genuine removal; an unchanged fine score appended after a feature bottleneck would bypass it. |
| `L3_COMPLEMENTARY_XOR` | Recipient 1 sees `(B1,U)` and recipient 2 `(B2,U XOR S)` for independent fair bits. Each local view is SEX-independent, but the two clue bits together recover `S` exactly. | Local-only privacy training can miss a coalition-only clue. Pair readers must inspect aligned people and include ignore-one-recipient candidates. |
| `L4_NO_COMPLEMENTARITY` | Both views contain the same `C=S XOR E`, with `P(E=1)=1/5`; task bits are otherwise independent. The pair posterior equals the posterior from `C` alone. | A coalition advantage should not be invented where there is no additional pair clue. The ideal binary-clue SEX AUC is `0.8` locally and jointly. |
| `L5_CONFIDENCE_VS_CLASS` | `S=H`; `P(Y1=1|H)` is `0.60` or `0.95`, while the hard decision is always class 1. The class-only probability `0.775` is optimal without `H`, but the two precise probabilities identify `S`. | Decision-only privacy can be ineligible for confidence utility. Precise confidence has strictly lower expected NLL and multiclass Brier; the Brier gain is `2 Var(P(Y1=1|H)) = 0.06125`. |

These premises show capability and impossibility in their *specific* finite laws, not prevalence in Adult. The checker must verify exact source-law strings/parameters and rational masses, then report separate tests of null, removable, complementarity, redundancy and confidence behavior. A neural optimizer succeeding or failing on one law is an algorithm result distinct from these analytic premises. No synthetic result may choose a better Adult coefficient, threshold or noise draw after the campaign lock.
