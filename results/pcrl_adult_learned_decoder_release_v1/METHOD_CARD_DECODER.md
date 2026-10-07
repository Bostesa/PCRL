# Method card, decoder, correctness-gate and deployment part (lra; role B; the lead merges it into METHOD_CARD.md)

Code: `lra/decoder.py`, `lra/fixtures.py`, `lra/deploy.py` (a port of the predecessor's `lcr/` at 091afc2). Proofs and
checks: `MATH_REVIEW.md`. Fixture laws (pinned, unchanged copy) and the correctness gate: `FIXTURE_LAWS.json` and
`ENGINEERING_GATE_RULE.json`; CORRECTNESS_LOCK governs them.

## 1. What D1 is

D1 is the single learned decoder shared by every new arm (C-TASK, the 72 weighted controls, the 15 constrained units)
and by every fixed-map calibration control (`<D0 id>|D1`, 27 per seed including the registered CLASS|D1). For one token t of one recipient, it turns the token's
fitting sufficient statistics into the released probability vector:
- n_t: fitting count;
- y_t: true-label counts;
- s_t: sum of teacher probability vectors;
- d_t: the teacher-predicted class of the token.

**Problem.** Minimise, over u in the class-dominant simplex {sum u = 1, u >= 0, u_d >= u_k}:

    sum_k y_k (-log q_k) + 0.5 sum_rows ||q - onehot(Y)||^2 + kappa KL(pbar || q),
    q = (u + eps 1 + eps e_d) / (1 + (K+1) eps),   pbar = s / n,   kappa = 32,   eps = 1e-12.

Here q is the ACTUAL released vector: the source smoothing, applied as an affine map.

**Fixed choices (not tuned).**
- kappa = 32 teacher pseudo-observations per occupied token.
- Brier coefficient 0.5.
- eps = 1e-12, float64, natural logs.
- No SEX enters the decoder. SEX may influence the partition only, and is never a deployment input.

**Training-label use (disclosed).**
- D1 reads true task labels of OSF_DEFENSE_FIT rows, through `lra.data.labels_for(D, "fitting", "OSF_DEFENSE_FIT")`
  in the runner (`lra.run.fit_data`), only after SCIENCE_LOCK.
- This is a material change from the label-blind D0 codebook.

## 2. How it is solved

The problem is strictly convex: the Hessian is diagonal and positive for n >= 1. The solve is exact and deterministic,
with no general-purpose solver in the loop.
- The multiplier of sum u = 1 at a given u_d = t follows from exact pooling,
  nu(t) = max_m -(g_d + the m smallest g_k) / (1 + m).
- Every other coordinate has a closed form (the positive root of a quadratic), clipped to [0, t].
- The remaining residual R(t) is strictly increasing and is bisected exactly 64 times on [1/K, 1], down to adjacent
  doubles.
- Cost: about 0.32 s CPU per 40K-token batch (K = 6); about 3.2 ms per 256-row call.
- `solve_token` is the M = 1 batch. Rows are independent, so batch and scalar u and q are bitwise equal.

**Frozen projection.**
- After the solve, the frozen repair runs: clip at 0, cap at u_d, divide by the k-ordered sum. This makes u
  class-dominant EXACTLY in float64.
- Exact class-dominance gives a strict argmax d on the released q (MATH_REVIEW Theorem 2).
- A repair larger than PROJ_TOL = 1e-9 refuses the solve.
- Every certificate is computed on the final released q.

## 3. Certificates and tolerances (`lra.decoder.TOLERANCES`; DECODER_CERTIFICATES.json via `aggregate_certificates`)

**Per token.**
- primal residuals: |sum u - 1| and |sum q - 1| (the latter <= 1e-12);
- min u (>= 0);
- margin q_d - max_{k != d} q_k (> 0);
- the active zero and tie sets;
- pooled multiplier nu;
- relative stationarity and dual infeasibility (<= 1e-9 each, relative to the largest gradient term);
- projection magnitude (<= 1e-9);
- bisection bracket (<= 2 ulps);
- objective (varying part and full prompt objective);
- converged.

An uncertified row raises and is never released. Synthetic fuzz values: stationarity <= 6e-16, dual infeasibility 0,
projection <= 1.2e-16.

**Aggregation.** `certificate_summary(table)` summarises one recipient table. `summary_violations(summary)` lists any
registered tolerance it misses. `aggregate_certificates(records)` builds the DECODER_CERTIFICATES.json body from the
d1 / ctask / fit unit records, which already carry `certificates = {"1": summary, "2": summary}`. It reads committed
records only: no solve and no refit. Every table is listed with its violations, and the totals state whether every
table is within tolerance.

**Inputs refused.**
- n = 0 (the fallback applies instead);
- non-integer labels, or labels not summing to n;
- teacher sums not summing to n (1e-9 n);
- d not the teacher's class (1e-9);
- any kappa or eps other than the registered values (kappa = 32, eps = 1e-12 and the Brier coefficient 0.5 are
  module constants asserted at import).

**Tables.**
- decode_policy re-encodes the fitting rows, and requires the tokens to be exactly the deployed ones.
- It requires the token counts to equal the policy's.
- It requires the canonical teacher sums (fine-cell accumulation order = qpc token_tables) to agree with the row sums
  within 1e-9 n_t.

## 4. Fallback rules (registered before any fit)

- A token with n_t = 0 keeps the pinned D0 decoded vector, token_proto, bitwise. This covers an absent predicted class
  (the qpc fallback token, smooth(uniform, class)) and any other token without fitting rows.
- Its label counts are zero. No labels are invented and no supervised statistic is attached.
- `DecoderTable.validate` refuses fallback flags other than exactly {n_t = 0}, and fallback vectors other than the
  policy's.

## 5. Cache

`TokenCache` is keyed on the EXACT bytes of (y, s, n, d, K, kappa, eps), with -0.0 mapped to +0.0.
- Every hit re-verifies the stored key, and cached arrays are read-only.
- A stale reuse would need changed label counts or teacher sums to produce the same bytes, which is impossible. Tests
  cover this, including a roundoff-size change of 1e-13 in the teacher sums.
- `solve_many` solves its misses in one batch. Because rows are independent, its results never depend on what was
  already cached.

## 6. Artefacts and bindings

**decoder.json (`lra.DecoderPair`).** Per recipient it stores:
- K and the token classes;
- n_t, y_t, s_t, u_t, q_t and the fallback flags;
- the per-token certificates;
- the sufficient-statistic hash and a content hash.

At the pair level it stores:
- the D1 config id;
- the policy-pair fingerprint and the policy's own config;
- the teacher/schema binding, copied from the policy pair;
- decoder_sha256, the canonical-JSON hash of the body.

`load_decoder_pair` checks:
- the hash;
- the bitwise smoothing u -> q;
- the strict argmax and normalisation;
- the policy fingerprint and binding;
- and it re-solves every supervised token from the stored statistics, which must reproduce u and q bitwise.

**Configuration convention (accepted by the lead).**
- New fits: policy config = the lra id (e.g. `U|K-JOINT-PAIR|i8o64|D1`), and decoder config = the same id.
- Fixed-map D1 controls: policy config = the admitted D0 id, and decoder config = that id + `|D1`. The D1 artifact
  never overwrites the D0 map: the unit is `dec__s{k}__<id>` and the D0 map stays in `pol__s{k}__<id>` (distinct ids,
  tested).

**release.npz (`release_arrays_d1`).**
- Exactly `row_id, tok1, q1, hard1, alpha1, tok2, q2, hard2, alpha2`.
- Tokens are the policy's canonical token IDs and q the D1 vectors.
- Checked: decisions equal the teacher's, normalisation, and strict argmax at the decision.

## 7. What the decoder can and cannot do

**It can:**
- Improve true-label log loss and Brier on a FIXED code when the frozen teacher is miscalibrated. The real heads are
  imperfect.
- Thereby make a more private (coarser or SEX-mixing) partition usable within the fitting budgets.
- (The predecessor called this the "decoder-enabled" route; in lra it is measured on Adult, not gated on fixtures.)

**It cannot:**
- **Remove information from an unchanged token.** The recipient receives the full token identity, and the decoder is
  a public deterministic function of it. So I(S; complete interface) and every complete-interface Bayes risk are
  unchanged (MATH_REVIEW Proposition 5; fixture check C1).
  - A finite attacker's score on the probability-only view may change in either direction.
  - That change is not credited as privacy.
  - Information reduction requires a changed partition.
- **Change any decision.** Accuracy, recalls and the confusion matrix equal the teacher's.
- **Beat a calibrated teacher in population.** If the teacher equals the true conditional, the token mean is the unique
  population minimiser of log loss and Brier (MATH_REVIEW Proposition 4; fixture check C2).
- **Certify held-out confidence.**
  - In-sample caveat: the U encoders and heads were fitted on OSF_DEFENSE_FIT (osf protocol). D1 calibrates the
    teacher's IN-SAMPLE outputs, and L_i(U) and B_i(U) on the fitting rows are in-sample too.
  - Fitting-row budget slack can therefore overstate held-out confidence.
  - The unchanged inner and assessment rules decide.
- **Be a Bayes rule.** It is a regularised registered decoder. "Violates a budget under D1" means infeasible under this
  registered decoder, not mathematically infeasible.
- **Support global-optimality claims.** Its convexity certifies the fixed-token solve only. The discrete partition
  search (lra.mapper) is heuristic, and no global optimum is claimed.

## 8. Correctness stage and the engineering gate (prompt section 9)

**What changed from the predecessor.** The predecessor's performance gate (trigger: a privacy-trained D1 release
0.01 nats below the strongest feasible task-only release) could not trigger on its bank: CLASS|D1 was affordable on
F2–F4, and every class-preserving release has at least the decision-only information (the disclosure floor,
MATH_REVIEW section 7). Its GATE_NOT_MET is preserved as `SOURCE_FIXTURE_GATE.json`. It is never this study's verdict.
lra replaces that gate with a CORRECTNESS-ONLY launch gate on the SAME four pinned laws.

**Laws** (`FIXTURE_LAWS.json`, copied unchanged, file sha256 24f70745…, laws_sha256 5c5e3bda…). Four fixed families,
each with N = 4096 exact expected counts:

| Family | Purpose | Shape | Static task gains | Mapping pairs |
|---|---|---|---|---|
| F1_CALIBRATED_NULL | the teacher is the true conditional; labels are the exact expectation | K = 2 / 2 | 0.25 / 0.23 | 4096 |
| F2_MISCALIBRATED | overconfident binary head; 3-class head with a binding class-dominance cell; SEX only in recipient 2 | K = 2 / 3 | 0.19 / 0.10 | 32768 |
| F3_COMPLEMENTARY_XOR | SEX = b1 XOR b2 (w.p. 3/4); local information exactly 0 | K = 2 / 2 | 0.22 / 0.28 | 4096 |
| F4_REDUNDANT | one clue bit visible to both recipients; local = pair information | K = 2 / 2 | 0.22 / 0.28 | 4096 |

They are KNOWN, ALREADY-OPENED regression cases, not fresh mechanism evidence.

**Engine (unchanged).**
- The D0 bank: FINE-TASK, CLASS, DIRECT-TASK and the 24 old-objective privacy maps.
- Their D1 versions, assignments unchanged.
- C-TASK, the 24 W- and the 5 K- arms through `lra.mapper.fit_unit` in fixture mode. They now persist traces.
- The exhaustive canonical-partition oracle.

**Mandatory checks E01–E12** (prompt checks 1–12; exact text and tolerances in `ENGINEERING_GATE_RULE.json`,
generated by `lra.fixtures.engineering_gate_rule()`):
- E01: integer atoms, class routing, label counts, law hashes.
- E02: fixed-token D0/D1 full-interface information.
- E03: F1 calibrated null.
- E04: final-vector D1 certificates (class, simplex, KKT, projection, plus an independent Frank-Wolfe gap).
- E05: LL/Brier row-level reconstruction.
- E06: budgets and local caps on EVERY accepted deployed state, including atomic paired moves.
- E07: the temporary CLASS partner of a sequential stage is not required feasible; the final release meets both
  budgets.
- E08: incremental terms and cached solves equal a from-scratch replay of the persisted starts and moves.
- E09: exhaustive enumeration reproduces the published oracle tables, with heuristic gaps labelled (not failures).
- E10: the decision disclosure floor; CLASS affordability recorded, never a failure.
- E11: launch, truth-table and refusal wiring, through registered pytest node IDs.
- E12: the 14 inherited review findings, each with a passing regression test.

**Verdict.** ENGINEERING_READY iff every mandatory check passes; otherwise ENGINEERING_BLOCKED.

These outcomes are registered as non-blocking: CLASS zero-leakage or affordable; no superior fixture release;
constrained ties weighted; joint ties sequential; the source GATE_NOT_MET; heuristic gaps; a constrained arm infeasible
under this decoder. The old trigger, T* and route are computed as descriptive output only.

**Outputs.**
- `ENGINEERING_GATE_RESULT.json`: verdict, reasons, every check with its details, rule, laws and source-gate hashes,
  and descriptive output.
- `correctness_oracle/<FID>_{partitions_r1,partitions_r2,pair_I12,arms}.csv`.
- Private units `cor__<FID>` under `<PRIVATE_CACHE>/lra_v1/run/units/`. Each holds the mapper records and traces, the
  policies, the decoders, the releases, the D0 start maps and the fine partitions, so the independent verifier can
  replay every search.

**Command.** The registered stage runs through the lead's `lra.run --lock <CORRECTNESS_LOCK> --stage correctness`
(one process, no shard, under the semaphore). `python -m lra.fixtures laws` verifies the pinned laws.
`python -m lra.fixtures rule [--write]` prints or writes the rule.

**Modules the stage loads.**
- lra: `lra.fixtures`, `lra.decoder`, `lra.mapper`, `lra.run`; `lra.lock` for the binding check.
- qpc: `qpc.compress`, `qpc.kmeans`, `qpc.release`, `qpc.partition`.
- dpc: `dpc.utility`, `dpc.partition`, `dpc.compress` (through qpc).
- E11/E12 run their registered pytest node IDs in one child process of the stage.

## 9. Deployment (`python -m lra.deploy`)

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m lra.deploy --unit <teacher unit> --policy <policy.json> \
        [--decoder <decoder.json> [--decoder-sha256 <hex>]] --X <input.npz> --schema <schema> --out <release.npz> \
        [--schema-sha256 <hex>] [--seed k]

**What it does.** This is a thin wrapper around `qpc.deploy`: the same 83-column schema checks, teacher forward
application and output writer.
- Output is ONLY tokens, decoded probabilities and decision per recipient.
- D0 configurations deploy without a decoder. CLASS|D1 is a registered fixed-map control and deploys with its decoder.
- D1 configurations require their own decoder.json, bound to the same policy fingerprint, teacher and schema. Its hash
  and bitwise re-solve are verified.

**Refused (exit code 2):**
- extra, reordered, reversed, renamed or missing columns, non-finite values, and extra input arrays;
- a mismatched teacher (a re-bound policy, or a different teacher unit with the original policy and decoder);
- a decoder of another map, or a tampered, stale, hash-mismatched or relabelled decoder;
- raw-score, fine-ID, teacher-probability, logit, latent, label, sensitive-attribute and debug exports;
- unknown flags and abbreviations;
- unregistered configurations: off-bank lambda or caps, a D1 id without a decoder, a decoder for a D0-only id, a
  family inconsistent with its id.

Tests: `lra/tests/test_deploy.py`, 11 tests on a synthetic teacher (cbp deploy-test pattern).
