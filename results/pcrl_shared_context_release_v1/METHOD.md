# Shared-context release: method

This is a supervised, residence-utility **2018 development** mechanism. The encoder privately computes a per-person 17-token law and releases one persistent keyed token beside the unchanged service vector H_A.

Nothing here is label-free. The historical T0 code and teacher posterior were fitted with residence labels, and the new policies are trained with residence and protected labels on construction rows only.

Binding details are in `PROTOCOL.md`, `PROTOCOL_AMENDMENTS.md` (M1–M5, O1) and `DESIGN_SPEC.md`.

## Release law

For legal inputs x = (X_A, H_A, stored T0, teacher posterior p, residual r, historical risk):

    q(z|x) = B[T0(x), z] + Σ_m A[k(x), m] · 1{d_m(x) = z}
    B ≥ 0, A ≥ 0, 0 ≤ η ≤ 1,  Σ_z B[t,z] = 1 − η  (every t),  Σ_m A[k,m] = η  (every k)

- **Contexts.** k(x) is a hard context. K=1 is global. K=4 is a 2×2 split at nuisance-role PWGTP-weighted medians of r and of the frozen local SEX-risk max-probability, with a registered K=2 fallback.
- **Policies.** The d_m are frozen deterministic 17-token policies:
  - d_0 = D17, exactly;
  - d_1 is task-only;
  - d_2 is local-priced;
  - d_3 is coalition-priced;
  - d_4 is all-priced at doubled prices.
- **Witness and embedding.** With η=0 the class embeds every T32 kernel. B=D17, A=0, η=0 is the exact D17 witness.
- **Identification.** η is not identified, since D17 is also a column. Richer decisions are measured by within-T32 variation of the fitted per-person law, not by η.

## Policies: one approximate pricing step (cost-sensitive reduction)

- **Prices.** λ are the nonnegative duals of the round-0 fixed-bank T32 LP. A price group with all-zero duals uses the registered fallback λ_j = 1/|group|.
- **Priced cost.** Per person and token, g_i(z) = u_i(z) − Σ_j λ_j a_ij(z), where:
  - u is the task cross-entropy of the frozen decoder. It is cross-fitted on construction rows: two household folds, each fold's decoder scoring the other fold.
  - a_ij is attacker j's cross-entropy for the true protected class at token z.
- **Oracle (M2).** For each token, a regularized boosted regressor on nuisance_train fits the paired improvement over D17, Δ_i(z) = g_i(z) − g_i(D17(x_i)). The D17 column is exactly 0.
- **Switching rule (M1).** The policy deviates to argmin_z Δ̂(z|x) only when min_z Δ̂ < −0.002 nats, with ties going to D17. This shares one codebook with D17 by construction.

## Fitting

- **Round 0**, using the predecessor Branch A machinery on the same roles and seeds:
  - the decoder is fitted on nuisance_train under the coverage law (current + D17 + uniform)/3, with the slate chosen on inner_selection;
  - the attack bank is fitted on audit_fit and validated on inner_selection;
  - cut coefficients are computed on coefficient_split.
- **Rounds 1–6.** Each round:
  1. solves the LP over (B, A, η);
  2. refits the decoder on the per-person law;
  3. fits best-response attackers;
  4. rebases every cut to ρ = min over the bank of L_a(D17) on the same rows;
  5. re-solves.

  The LP takes one of two forms:
  - **U form:** minimize ½C_U + ½C_W subject to L_a ≥ ρ − 0.001.
  - **P form:** maximize τ with AB/SEX cuts ≥ ρ + τ, the other cuts ≥ ρ − 0.001, and task ≤ D17 + 0.001.
- **Closing refit (M3)** on the last round's own law.
- **Final selection (M4, M5).** Selection is among rounds ∪ {D17 witness} that are feasible on the enlarged bank:
  - U form: lowest inner_selection task;
  - P form: largest AB/SEX slack.

  If only the witness is feasible, the unit is `WITNESS_FALLBACK`, an exact D17 alias.

For fixed decoder and bank, the step is an LP in 32·17 + K·M + 1 variables with one inequality per retained cut. Decoder refits, policy learning and context construction make the whole procedure nonconvex. LP optimality certifies only the fixed bank. The attacker refits and the independent audit are the only privacy evidence (math review, finding 3).

## Controls

| Control | What it is |
|---|---|
| T32_U / T32_P | Same pipeline with η fixed to 0 (parent-restricted) |
| DET_SEL1 / DET_SEL4 | Exhaustive deterministic choice of one policy per context (5 or 625 assignments), on the closing bank |
| RD_TASK | Richer deterministic task-only policy |
| RD_PRIV | Richer deterministic privacy-priced policy with a bisected multiplier |
| ADV_B1/B2 (task-selected) and ADV_B1_P/B2_P (privacy-selected) | PPAN-style adversarial categorical encoder with exact 17-token expectation |
| D17 | Exact reference |
| Q_HIST | Historical task-directed Q, the external 17-token reference |
| J, H_ONLY | Continuity references |

## Deployment

- **Code.** `release.load_law(unit_dir)` computes q(·|x) from pinned objects and legal inputs only; hb, labels, ids, households and weights are refused.
- **Session.** `release` session objects emit one HMAC-keyed persistent token per record ID. Changed inputs for a known ID are refused.
- **Wire.** `h_a` plus `token`.
- **Not analyzed.** Fresh redraws and composition with other releases.

## Complexity and cost

- **Parameters beyond the T32 kernel.** At most K(M−1)+1 mixing parameters: 5 at K=1 and 17 at K=4. The policy learners are separate models: 17 boosted regressors per non-D17 policy, fitted on about 4,100 construction people per anchor.
- **Runtime.** One unit is roughly O(n·17) frozen-predictor evaluations per round plus the decoder and attacker refits, which dominate.
