# Method implementation notes (NM / T32 / DET_SEL)

Owner: method implementer. Written 2026-09-24 before any scientific fit. No outer row, no
committed number. Code: `experiments/pcrl_shared_context_release_v1/{channel,policies,contexts,fit_nm,release}.py`;
tests: `tests/pcrl_shared_context_release_v1/test_{channel,policies,contexts,fit_nm,release}.py`.

## CLI

```
# once per anchor (shared by every variant of that anchor)
python -m experiments.pcrl_shared_context_release_v1.fit_nm build-bank --anchor A --out <private>/bank_aA
# registered K decision from the three banks' coefficient_split census
python -m experiments.pcrl_shared_context_release_v1.fit_nm decide-k --banks <bank_a0> <bank_a1> <bank_a2>
# one unit
python -m experiments.pcrl_shared_context_release_v1.fit_nm fit --anchor A --variant {NM1_U,NM1_P,NM4_U,NM4_P,T32_U,T32_P} \
    --bank <bank_aA> --out <private>/aA_<variant> [--rounds 6] [--nm4-k {4,2}] [--smoke]
# exact deterministic selector (amendment M1.3), from the finished NM*_U unit of the same K
python -m experiments.pcrl_shared_context_release_v1.fit_nm fit --anchor A --variant DET_SEL4 --from <aA_NM4_U> --out <private>/aA_DET_SEL4
```
Every output path must contain a `private` component. `--smoke` = 30% household subsample,
at most 1 round, refuses output paths inside the repository. NM4 variants refuse to run
without an explicit `--nm4-k` (the registered fallback decision is cross-anchor).
The infrastructure dispatcher loads a finished unit with `release.load_law(unit_dir)`
(kind `nested`, `RELEASE_SPEC.json`), which works for NM*, T32* and DET_SEL* units.

## What each piece does

- `channel.py`: x = [vec B (32x17), vec A (KxM), eta]; equalities `sum_z B[t,z] + eta = 1`,
  `sum_m A[k,m] - eta = 0`; utility LP (min 0.5U+0.5W), privacy-first LP (max tau; AB/SEX cuts
  `>= rho + tau`, others `>= rho - delta`, caps `C_v <= C_v(D17) + .001`), phase I, HiGHS with the
  TAC tolerances, primal replay, reconstructed dual bound, e(B)/e(A)/eta sparsity, calibration via
  the inherited `AR/reference.calibrate_reference` on the B blocks plus a second witness
  (B=0, A[:,D17]=1, eta=1) that must reproduce the B=D17 losses to 1e-10 (catches any row/weight/
  context accounting mismatch between the blocks). `fix_eta_zero` builds exactly the TAC `solve_p1`
  matrices; the solution is bit-identical to `solve_p1` (test) and the T32 round-0 unit is
  bit-identical to the bank's AR round-0 LP (test + recorded per unit).
- Nonalias gate (never eta): per-person TV to D17 row, within-T32 spread V (TV to the state mean,
  U and PWGTP), max pairwise TV within a state, affected unique households, per-state V quantiles,
  deterministic-emission report, stochastic weight fraction; plus the within-T32-averaged re-score
  (T32 projection re-scored with the same frozen decoder and bank, and on inner_selection).
  Recorded on coefficient_split and inner_selection at every round checkpoint.
- `policies.py`: legal-input guard (`x, ha, token_codes|T0, teacher_p|p, residual|r, risk` only;
  `hb, labels, ids, households, weights, role, loss ...` raise `PermissionError`), 49 features
  `[X_A 32, H_A 4, logit p, r, risk 11]` standardized on nuisance_train, 17 HGB regressors per policy
  (registered hyper-parameters), D17-anchored switch with TAU = 0.002 (ties to D17, then lowest id),
  price groups with the registered all-zero-dual fallback, alias ledger, disagreement diagnostics.
- `contexts.py`: PWGTP-weighted medians on nuisance_train of stored r and frozen SEX-risk max-prob;
  K=1/2/4 rules saved together; coefficient_split census (unique households, household-weight ESS);
  registered cross-anchor fallback.
- `fit_nm.py build-bank`: AR round-0 decoder (`fit_a.fit_frozen_decoder`, AR seeds) and attack bank
  (`fit_a.build_attack_bank`, sources H/D17/Q/coverage), round-0 T32 LP (`alternate.channel_update`)
  and duals (`fit_b.fixed_bank_dual_prices`), frozen nuisance, 2-fold household cross-fitted task
  decoders (M1.2), priced attack losses on nuisance_train (`fit_b._attack_loss_rows`), 4 switched
  policies, alias ledger on coefficient_split, context rules + census, write-once inventory.
- `fit_nm.py fit`: round 0 reuses the bank's decoder/attacks; rounds r>=1 refit the decoder on the
  per-person coverage law `(q_i + D17[t_i] + 1/17)/3`, fit A and AB best-response slates on the
  current per-person law (AR recipe/seeds), rebase the whole bank, re-select among retained decoders
  on inner_selection under the current law, re-solve. Final round = AR rule. Write-once checkpoints,
  `COMPLETE.json` with SHA inventory; resume verifies instead of refitting.

## Deviations / interpretation choices (flag for the coordinator)

1. Policy cost target: the spec formula `g_i(z) = u_i(z) - sum_j lambda_j a_ij(z)` is used literally
   in per-person nats (so TAU=0.002 nats is meaningful); U vs W cuts differ only in the dual price,
   the population weighting is carried by the regressor's PWGTP-mean-1 sample weights. The
   `fit_b.priced_original_person_rows` helper (which also multiplies each term by its own 1/n or
   PWGTP/sum weight) is therefore not used for the target; its attack replay (`_attack_loss_rows`) is.
   People missing a protected label contribute no attack term for that target; task-unlabelled
   people are excluded from the regression.
2. Early stopping: sklearn's built-in early stopping is not household-grouped, so the boosting
   count is the argmin of the staged validation MSE on a household-hash 20% split, and the model
   is refitted deterministically on the 80% households with that count.
3. Price-group fallback reads "role group" as the policy's price group (local = A-view cuts,
   coalition = AB-view, all = all cuts); lambda_j = 1/(cuts in group), doubled for all_priced_x2.
4. P variants run the alternation with the privacy-first LP in every round and use the AR final-round
   rule (feasible on the final rebased bank at floor rho - delta, then lowest inner-selection task).
5. DET_SEL uses the source NM unit's LAST round (its decoder cost blocks and final rebased bank),
   over the alias-pruned retained columns (M^K may be < 625 if aliases were removed).
6. Seeds equal the AR seeds (`20260924 + 10000 anchor + 100 round`, attacks +50000); cross-fit
   decoders +80000(+fold), nuisance +70000, policies +90000+index. With identical inputs, T32_U
   therefore reproduces the AR Branch A center (same decoders, attacks and LP in every round).
7. Round-0 attack bank is fitted before the policies exist (spec order), so no attacker has seen a
   policy column until round 1; the math review's warning about unseen-token pricing applies to
   round 0 costs.

## Infrastructure hand-off

Every finished NM/T32/DET_SEL unit also writes the infrastructure descriptor `release.json`
(`laws.write_descriptor`, kind `nested`, pins `RELEASE_SPEC.json`, the params file and
`COMPLETE.json`), so `laws.load(unit_dir)` dispatches to `release.load_law`. The frozen bank is
external to the unit and pinned by the SHA-256 of its `BANK_COMPLETE.json` (plus the policy-bank and
context-rule SHAs); `release.load_law(unit, bank_dir=...)` overrides the location after a restore.

## Mac smoke (engineering only; 30% subsample, anchor 0, OMP=2; no number is scientific)

| step | wall | max RSS |
|---|---|---|
| build-bank (AR r0 decoder + 156-spec bank + LP/duals + nuisance + 2 cross-fit decoders + 4 policies + contexts) | 100 s | 0.72 GB |
| NM4_U, 2 rounds (r0 + 1 alternation round) | 40 s | 0.70 GB |
| T32_U, NM1_P, 2 rounds each | ~37 s each | similar |
| DET_SEL4 (625 assignments) | 3 s | 0.44 GB |
Disk: bank 51 MB, unit 14 MB (smoke). Engineering observations only: in smoke the early-stopping
argmin kept 10-90 of 150 boosting iterations; the local-priced group hit the registered all-zero-dual
fallback; T32_U reproduced NM4_U when the NM4 LP returned eta=0; NM1_P returned interior eta with
almost every row stochastic (the math review's time-sharing prediction).

## Amendment M2 (paired oracle) — implemented

`policies.fit_paired_oracle(standardized_features, costs, base_tokens, pwgtp, households, *, seed) -> (PairedOracle, record)`
fits the registered HGB recipe on Delta_i(z) = g_i(z) - g_i(D17(x_i)) (all rows; the D17 column is
forced to 0 at prediction). `policies.switched_policy_from_oracle(name, d17_tokens, standardizer, oracle,
*, tau=TAU, pricing=None)` returns the frozen legal-input policy (deviate iff min Delta_hat < -tau).
`PairedOracle.predict_delta(z_std, base_tokens)` gives the (n,17) paired predictions.
Caveat from a synthetic probe (scratchpad, not a test): pairing removes person-level noise shared by
all tokens (raw oracle 91% -> paired 10% deviation), but when token-specific noise is ~0.1 nats the
fitted mean paired difference of some token still falls below -tau by chance and the paired policy
deviates for almost everyone. TAU is not scaled by estimation error.
