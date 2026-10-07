# Method card, decoder, fixture and deployment part (role B; the lead merges it into METHOD_CARD.md)

Code: `lcr/decoder.py`, `lcr/fixtures.py`, `lcr/deploy.py`. Proofs and checks: `MATH_REVIEW.md`.
Fixture laws and gate: `FIXTURE_LAWS.json` and `FIXTURE_GATE_RULE.json`. Both were written before any fixture algorithm
ran; FIXTURE_LOCK governs them.

## 1. What D1 is

D1 is the single learned decoder shared by every new arm (C-TASK, the 72 weighted controls, the 15 constrained units)
and by every fixed-map calibration control (`<D0 id>|D1`). For one token t of one recipient, it turns the token's
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
- D1 reads true task labels of OSF_DEFENSE_FIT rows, through `qpc.data.labels_for(D, "fitting", "OSF_DEFENSE_FIT")`
  in the runner.
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

## 3. Certificates and tolerances (`lcr.decoder.TOLERANCES`; DECODER_CERTIFICATES.json via `certificate_summary`)

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

**Inputs refused.**
- n = 0 (the fallback applies instead);
- non-integer labels, or labels not summing to n;
- teacher sums not summing to n (1e-9 n);
- d not the teacher's class (1e-9);
- any kappa or eps other than the registered values.

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

**decoder.json (`lcr.DecoderPair`).** Per recipient it stores:
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
- New fits: policy config = the lcr id (e.g. `U|K-JOINT-PAIR|i8o64|D1`), and decoder config = the same id.
- Fixed-map D1 controls: policy config = the admitted D0 id, and decoder config = that id + `|D1`.

**release.npz (`release_arrays_d1`).**
- Exactly `row_id, tok1, q1, hard1, alpha1, tok2, q2, hard2, alpha2`.
- Tokens are the policy's canonical token IDs and q the D1 vectors.
- Checked: decisions equal the teacher's, normalisation, and strict argmax at the decision.

## 7. What the decoder can and cannot do

**It can:**
- Improve true-label log loss and Brier on a FIXED code when the frozen teacher is miscalibrated. The real heads are
  imperfect.
- Thereby make a more private (coarser or SEX-mixing) partition usable within the fitting budgets.
- That is the "decoder-enabled" route of the fixture gate.

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
  search (lcr.mapper) is heuristic, and no global optimum is claimed.

## 8. Fixture stage and mechanism gate (prompt section 9)

**Laws** (`FIXTURE_LAWS.json`). Four fixed families, each with N = 4096 exact expected counts and atom weights in
multiples of 1/4096:

| Family | Purpose | Shape | Static task gains | Mapping pairs |
|---|---|---|---|---|
| F1_CALIBRATED_NULL | the teacher is the true conditional; labels are the exact expectation | K = 2 / 2 | 0.25 / 0.23 | 4096 |
| F2_MISCALIBRATED | overconfident binary head; 3-class head with a binding class-dominance cell; SEX only in recipient 2 | K = 2 / 3 | 0.19 / 0.10 | 32768 |
| F3_COMPLEMENTARY_XOR | SEX = b1 XOR b2 (w.p. 3/4); local information exactly 0; pair 0.1308 nats while both bits are visible | K = 2 / 2 | 0.22 / 0.28 | 4096 |
| F4_REDUNDANT | one clue bit visible to both recipients; local = pair information | K = 2 / 2 | 0.22 / 0.28 | 4096 |

- Each family has 4 fine cells per predicted class, and caps of 2 tokens per class.
- Budgets: L_i <= L_i(U) + 0.005 and B_i <= B_i(U) + 0.003.
- The lambda grid is as on Adult.

**Arms run on every fixture.**
- D0: FINE-TASK, CLASS, DIRECT-TASK and the 24 old-objective privacy maps (unchanged qpc search).
- The D1 versions of those exact maps, with assignments unchanged.
- C-TASK, the 24 W- controls and the 5 K- arms, through `lcr.mapper.fit_unit` in fixture mode with the full registered
  start and witness sets.

**Oracle.** Exhaustive canonical same-class partitions, independent of the mapper, with:
- exact-law MI;
- row-level losses under D0 and D1;
- the exhaustive optimum of every registered own problem.

**Gate.**
- T* is the feasible task-only D1 candidate with the smallest I12. The candidates are the refined C-TASK, FINE-TASK|D1,
  DIRECT-TASK|D1 and CLASS|D1.
- A privacy-trained D1 candidate qualifies when it meets every condition of section 3 of the execution prompt:
  - feasible: both budgets, decisions, strict argmax and caps;
  - within the local caps, I_i <= I_i(C-TASK);
  - I12(T*) - I12 >= 0.01 nats;
  - its own accuracy gain >= 0.03 on both tasks.
- The candidates are the D1 fixed maps of the 24 D0 privacy maps, the W- controls and the K- arms.
- GATE_MET iff the mandatory checks C1-C7 all pass AND at least one fixture triggers.

**Route** (only when GATE_MET).
- ASSIGNMENT_SEARCH_ROUTE iff, on some triggered fixture, a W- or K- arm qualifies and either no D1 fixed map
  qualifies or the new arm is at least 0.01 nats lower in I12.
- Otherwise DECODER_ENABLED_ROUTE.
- A descriptive flag records whether the D0 version of a qualifying fixed map already qualified.

**Mandatory checks.**
- C1: fixed-token information.
- C2: calibrated null (F1 only).
- C3: budget enforcement.
- C4: decision preservation.
- C5: term reconstruction.
- C6: heuristic labelling against exhaustive references.
- C7: law integrity.

**Outputs.**
- `FIXTURE_GATE.json`: verdict, reasons, route, per-fixture checks, trigger, references, arm table and hashes.
- `fixture_oracle/<FID>_{partitions_r1,partitions_r2,pair_I12,arms}.csv`.
- The private units `fix__<FID>`.

**Command.**
- The registered stage runs through the lead's `lcr.run --lock <FIXTURE_LOCK> --stage fixture`.
- `python -m lcr.fixtures laws` verifies the laws file. `python -m lcr.fixtures run --out DIR` runs the same stage
  code.

**Modules the fixture stage loads.**
- lcr: `lcr.fixtures`, `lcr.decoder`, `lcr.mapper`, `lcr.run`.
- qpc: `qpc.compress`, `qpc.kmeans`, `qpc.release`, `qpc.partition`.
- dpc: `dpc.utility`, `dpc.partition`, `dpc.compress` (through qpc).

## 9. Deployment (`python -m lcr.deploy`)

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lcr.deploy --unit <teacher unit> --policy <policy.json> \
        [--decoder <decoder.json> [--decoder-sha256 <hex>]] --X <input.npz> --schema <schema> --out <release.npz> \
        [--schema-sha256 <hex>] [--seed k]

**What it does.** This is a thin wrapper around `qpc.deploy`: the same 83-column schema checks, teacher forward
application and output writer.
- Output is ONLY tokens, decoded probabilities and decision per recipient.
- D0 configurations deploy without a decoder.
- D1 configurations require their own decoder.json, bound to the same policy fingerprint, teacher and schema. Its hash
  and bitwise re-solve are verified.

**Refused (exit code 2):**
- extra, reordered or missing columns, and extra input arrays;
- a mismatched teacher;
- a decoder of another map, or a tampered, stale or hash-mismatched decoder;
- raw-score, fine-ID, teacher-probability, logit, latent and debug exports;
- unknown flags;
- unregistered configurations: off-bank lambda or caps, a D1 id without a decoder, a decoder for a D0-only id, a
  family inconsistent with its id.

Tests: `lcr/tests/test_deploy.py`, 6 tests on a synthetic teacher (cbp deploy-test pattern).
