# Method card — lra: learned decoder D1 and confidence-constrained mapper on Adult

This card merges the role cards, which stay in the package unchanged: METHOD_CARD_DECODER.md (role B: decoder,
correctness gate, deployment) and METHOD_CARD_MAPPER.md (role C: mapper, traces, replay).

**Registration.** Frozen in SCIENCE_LOCK. The launch gate was correctness-only: ENGINEERING_READY on the four pinned lcr
fixture laws (ENGINEERING_GATE_RESULT.json).

**Scope.**
- This is a fixed-task output release. It is not general-purpose features, and not a pipeline with the encoder/LoRA or
  stochastic-release lines.
- Changing the decoder of an unchanged token removes no information. The recipient receives the complete token
  identity.
- D1 can improve utility and can make a more private partition usable; it cannot erase information.
- In the calibrated-teacher null, no population proper-loss gain exists.
- The fixed-token D1 problem is convex. The full joint discrete program is NOT convex, and no global optimum is claimed.
- Fitted MI is not a population guarantee.
- The sequential arms are matched adaptations, not the official Taylor–Vippathalla–Coon solver.
- No novelty is claimed: hard utility constraints, privacy-funnel objectives (arXiv:1402.1774), proper-loss calibration,
  greedy partition search and sequential collusion accounting (arXiv:2601.21859v2) are established ideas
  (PRIOR_ART_AND_CLAIM_SCOPE.md).

**Training-label use.** OSF_DEFENSE_FIT true task labels enter every D1 decoder, C-TASK, the weighted controls' task
objective and the fitting budgets. SEX enters the partition search, the local caps and the MI diagnostics. Neither is a
deployment input (EXPOSURE_LEDGER.md).

# Part A — decoder, correctness gate and deployment (role B)

Code: `lra/decoder.py`, `lra/fixtures.py`, `lra/deploy.py` (a port of the predecessor's `lcr/` at 091afc2). Proofs and
checks: `MATH_REVIEW.md`. Fixture laws (pinned, unchanged copy) and the correctness gate: `FIXTURE_LAWS.json` and
`ENGINEERING_GATE_RULE.json`; CORRECTNESS_LOCK governs them.

### 1. What D1 is

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

### 2. How it is solved

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

### 3. Certificates and tolerances (`lra.decoder.TOLERANCES`; DECODER_CERTIFICATES.json via `aggregate_certificates`)

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

### 4. Fallback rules (registered before any fit)

- A token with n_t = 0 keeps the pinned D0 decoded vector, token_proto, bitwise. This covers an absent predicted class
  (the qpc fallback token, smooth(uniform, class)) and any other token without fitting rows.
- Its label counts are zero. No labels are invented and no supervised statistic is attached.
- `DecoderTable.validate` refuses fallback flags other than exactly {n_t = 0}, and fallback vectors other than the
  policy's.

### 5. Cache

`TokenCache` is keyed on the EXACT bytes of (y, s, n, d, K, kappa, eps), with -0.0 mapped to +0.0.
- Every hit re-verifies the stored key, and cached arrays are read-only.
- A stale reuse would need changed label counts or teacher sums to produce the same bytes, which is impossible. Tests
  cover this, including a roundoff-size change of 1e-13 in the teacher sums.
- `solve_many` solves its misses in one batch. Because rows are independent, its results never depend on what was
  already cached.

### 6. Artefacts and bindings

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

### 7. What the decoder can and cannot do

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

### 8. Correctness stage and the engineering gate (prompt section 9)

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

### 9. Deployment (`python -m lra.deploy`)

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

# Part B — mapper, traces and replay (role C)

Owner: role C (mapper engineer). Tests: `lra/tests/test_mapper.py`, run on synthetic data only. Frozen rules:
`SEARCH_RULES.json`, written by `python -m lra.mapper rules` (its `rules_sha256` is also recorded in every unit
record). Role C code reads no Adult row, label or SEX column. The lead runs the real stages (`lra.run --stage ctask`,
`--stage fit`) after SCIENCE_LOCK.

**What is new.** The source study fitted codebook probabilities to the teacher (D0 = smoothed token mean) and swept a
privacy weight. This mapper:

- decodes every token with the shared learned decoder D1 (`lra.decoder`, role B);
- changes the coarse assignments with the actual fitting task losses;
- enforces explicit fitting budgets and local information caps.

Task labels and SEX on OSF_DEFENSE_FIT are used for fitting. This is a material change from label-blind codebook
fitting. The fine partitions remain fixed and label-blind; only the coarse assignments and the decoded vectors use
fitting labels.

**Source of the engine.** `qpc/compress.py` at 7f3ec67 is not edited. It is imported only for `labels_from_policy`,
`class_labels` and `check_sex`. The following qpc mechanics are re-implemented for the new objective:

- sufficient-statistic tables;
- canonical labels;
- exact n·log n plug-in MI;
- vectorised single-cell move deltas.

qpc's greedy merge-to-cap phase is **not** used. Every start is already a capped map, and every proposal moves one
whole fine cell to an existing token.

### 1. Interface

```
fit_unit(arm, fine_dict, T, tr, Y_fit, S_fit, lam=None, starts=None, witnesses=None, refs=None, meta=None)
    -> (record, {"policy.json": dict, "decoder.json": dict, "release.npz": dict of arrays, "trace.json": dict})
refs_from_ctask(ctask_record) -> {"I_ctask": {"1", "2"}, "ctask_pair_fingerprint", "L_U", "B_U"}
replay_unit(record, fine_dict, T, tr, Y_fit, S_fit, refs=None, *, trace, files=None, per_state=True, starts=None)
    -> report                                    (section 10; correctness checks 6, 7, 8)
replay_trace(problem_inputs, record, trace, files=None)   alias; problem_inputs = {fine_dict, T, tr, Y_fit, S_fit[, refs]}
```

**Arms.** `C-TASK`, `W-LOCAL`, `W-SEQ-12`, `W-SEQ-21`, `W-JOINT` (these need λ on the registered grid), `K-LOCAL`,
`K-SEQ-12`, `K-SEQ-21`, `K-JOINT-SINGLE`, `K-JOINT-PAIR`.

**Inputs.**

- `fine_dict` is the admitted `fine.json` ({"fine1", "fine2"}).
- `T` holds the teacher outputs over ALL rows (`row_id, p1, p2, d1, d2`).
- `tr` holds the OSF_DEFENSE_FIT indices.
- `Y_fit` is {1: income, 2: occupation_group}, aligned with `tr`.
- `S_fit` is binary SEX, aligned with `tr`.

**Starts and witnesses.** These are {config ID: `policy.json` dict}, keyed by `lra.run` IDs. Only the cell→token map
of each is used. Its fine fingerprints must equal `fine_dict`, and its D1 decoder is recomputed.

**Outputs.**

- `policy.json` is the qpc.PolicyPair of the token map. Its `token_proto` are the D0 means, which are **not
  released**. `config.config` is the lra ID, m1 = 8, m2 = 64, and both binding hashes are present.
- `decoder.json` is `lra.decoder.decoder_pair_dict`.
- `release.npz` is `lra.decoder.release_arrays_d1` over all rows. q there is the actual released D1 vector.
- `trace.json` is the persisted search trace of the unit (section 9). It is plain finite JSON, written by `lra.run.save`
  like decoder.json, and it is **not** inside record.json.

**Stable record fields used downstream.** `status` is `FEASIBLE` or `INFEASIBLE`. The others are
`deployed.feasible`, `final_state_terms`, `winner`, `token_counts` and `pair_fingerprint`. The trace adds
`trace_schema`, `trace_sha256` (sha256 of the trace's canonical JSON), `trace_counts` (starts, moves, pairs) and
`trace_cpu_s`. Each refined stage or start in `record.starts` carries `evals`, `ceiling_share`, `stop` and `sweeps`,
and these are mirrored in the trace's termination receipts.

**Refusals.**

- an unknown arm;
- λ given to a non-W arm, or λ off the grid;
- unregistered or missing start/witness keys;
- K- arms without `refs["I_ctask"]`;
- `I_ctask` not bitwise equal to the exact MI of the passed C-TASK map;
- `refs` U losses that differ from those recomputed on the rows;
- caps or budget different from the registered profile;
- missing binding hashes;
- a start that exceeds the caps or mixes classes;
- deployed fine-cell counts on the fitting rows that differ from the partition's n.

**Fixture mode.** `meta["fixture"] = True` (for role B's fixtures) permits other caps, budgets and λ, and a subset of
the source maps. It is recorded as `fixture_mode`.

### 2. Fitting quantities

All quantities are computed on the N OSF_DEFENSE_FIT rows, through the deployed fine-cell routing of the frozen
teacher. For each token t of recipient i:

- n_t is its count;
- y_t holds its exact label counts;
- s_t is the canonical sum over member fine cells of `fine.S`, accumulated from zeros in increasing fine index. This
  is bitwise `lra.decoder.token_stats` and qpc `token_tables`;
- the SEX counts form an exact int64 table, together with the pair table n(s, t1, t2).

The fitting terms are:

| Term | Definition |
|---|---|
| q_t | `lra.decoder.solve_batch(y_t, s_t, n_t, d_t)`, the actual released vector (κ = 32, ε = 1e-12, class-dominant, certified) |
| L_i | (1/N) Σ_t Σ_k y_t[k] · (−log clip(q_t[k], 1e-12, 1)) (`token_losses`) |
| B_i | (1/N) Σ_t Σ_k y_t[k] · Σ_j (q_t[j] − [j = k])², the source multiclass Brier |
| I_i, I12 | plug-in MI of SEX with the full token, and with the token pair (natural log, 0 log 0 = 0) |
| T | L1 + L2 + 0.5 (B1 + B2) |
| Φ | I12 + 0.5 (I1 + I2) |

**Exact state values.** Each term is the sum of the sorted nonzero per-token (or per-cell) contributions. A state value
is therefore a function of the partition alone, not of the search path or slot numbering (tested bitwise). Candidate
deltas use vectorised difference formulas; the test agreement is ≤ 1e-12.

**Final recomputation from the deployed release.** After the search, L and B are recomputed with dpc `per_row` on the
released rows of `tr`, and I and I12 with dpc `mi_plugin` on the released tokens. This is `record.deployed`. Parity
with the state values is required: ≤ 1e-10 for L and B, ≤ 1e-12 for MI. The released q is checked to be bitwise
equal to the search's q for every token. Synthetic parity is about 1e-16 and q is bitwise equal.

### 3. Budgets, caps and structure (one profile, all seeds)

**Utility budgets.** L_i ≤ L_i(U) + 0.005 nats and B_i ≤ B_i(U) + 0.003. U is the continuous U teacher on the same
fitting rows, under the dpc `per_row` conventions. These are fitting budgets applied to accepted moves. They are not a
population confidence certificate, and they do not replace the inner or assessment utility rules.

**Local caps (K- arms).** I_i ≤ I_i(C-TASK) of the same seed. The exact record bits come in through `refs_from_ctask`
and are cross-checked against the passed C-TASK map. They are fitting-law constraints, not held-out AUC guarantees.

**Margins.**

- In the search, L and B must satisfy value ≤ limit − 1e-10, and I must satisfy value ≤ cap − 0. MI uses the same
  exact function as the cap, so the K-LOCAL start (equal to C-TASK) is exactly feasible.
- On the deployed release, value ≤ limit with no tolerance. A winner that is feasible in the search but not on the
  deployed rows raises.

**Decision preservation** is structural:

- tokens never cross predicted classes;
- B's decoder certifies a strict argmax equal to the class;
- `release_arrays_d1` re-checks every row;
- the mapper re-checks the fitting rows.

**State caps.** At most 8 income / 64 occupation tokens per teacher-predicted class. Caps are upper bounds. Tokens are
never created, and an emptied token is removed. Actual counts are recorded (`token_counts`, `tokens_total`).

**No restoration phase (F N-9).** A local or sequential start that violates the enforced constraints of its stage is
not refined. It is recorded as `INFEASIBLE_START` and is ineligible. In particular, if C-TASK violates a fitting
budget on a seed, K-LOCAL (whose only start is C-TASK) is infeasible on that seed **by construction**. K-SEQ then has
only its source-map starts. The arm reports `status = "INFEASIBLE"` when no candidate is feasible, and its files are
then a descriptive release of the first candidate. It is "infeasible under this registered decoder and search", not
mathematically infeasible.

### 4. Arms

| Arm | Objective | Enforced | Search |
|---|---|---|---|
| C-TASK | T: each recipient separately on L_i + 0.5 B_i | none (budget slack recorded) | start = source FINE-TASK, D1-recomputed; **no SEX enters the search** (tested by permuting SEX) |
| W-LOCAL(λ) | T + λ(I1+I2)/2: per recipient L_i + 0.5 B_i + λ I_i/2 | none | start C-TASK |
| W-SEQ-ab(λ) | T + λΦ | none | as K-SEQ-ab |
| W-JOINT(λ) | T + λΦ | none | as K-JOINT-SINGLE; witnesses C-TASK, W-LOCAL(λ), W-SEQ-12(λ), W-SEQ-21(λ), FINE-TASK, CLASS and the 24 source privacy maps (30) |
| K-LOCAL | I_i per recipient | own budgets + own cap | start C-TASK |
| K-SEQ-ab | Φ | stage 1: a's only; stage 2: b's; final: both | see below |
| K-JOINT-SINGLE | Φ | both budgets + both caps | joint single moves |
| K-JOINT-PAIR | Φ | both budgets + both caps | K-JOINT-SINGLE + one atomic paired step after each single sweep |

**C-TASK report.** `ctask_report` gives the following separately:

- the D1 initialisation (the FINE-TASK map with D1);
- the refined map;
- the same two maps with the D0 decoder;
- optional external D0 references from `refs["D0"]`.

**Sequential (K-SEQ-ab, W-SEQ-ab).** For each start:

1. **Stage 1.** Recipient a starts from the start's a-map. Partner b is held at its CLASS-ONLY view: one token per
   predicted class, because the decision is always disclosed. a is optimised under Φ (or T + λΦ), enforcing only a's
   budgets and cap. The partner is **never required feasible**: its class-only constraint values are recorded and are
   typically violated (tested). It is never replaced by a constant.
2. **Stage 2.** a is frozen. b starts from the start's b-map and is optimised with b's constraints. a's freezing is
   asserted.
3. **Final.** The final release must satisfy both recipients' constraints.

The starts are C-TASK plus the six source SEQ-ab maps (one per λ), in that order.

**Joint (K-JOINT-*, W-JOINT).** There are 30 witnesses, in this order: C-TASK; LOCAL, SEQ-12 and SEQ-21 (the
constrained arms for K-, the same-λ weighted arms for W-); source FINE-TASK; source CLASS (`U|CLASS|i1o1`, added
2026-10-07 on the lead's decision as a "feasible source mapping", prompt section 8); then the 24 source privacy maps.
Source DIRECT-TASK is not a witness. It is a Stage A identity map on its own assignment cells, not a cell→token map of
the fine partitions. Every witness is evaluated unchanged against all constraints.

- A **feasible** witness becomes an unchanged candidate and is also refined. (For W-JOINT every witness is
  admissible.)
- An **infeasible** witness is recorded as `EXCLUDED_INFEASIBLE_WITNESS` and is never eligible by its name.

The winner has the lowest exact objective among eligible candidates. Ties follow the registered witness order, with
refined before unchanged. The winner therefore never has a higher Φ than any feasible unchanged witness (fitting
objective only).

### 5. Neighbourhood, acceptance and stopping (frozen)

**Proposal.** One whole fine cell moves to an existing alive token of its predicted class.

**Candidate set per cell.** The exact objective change is computed for **every** alive same-class target; each one
counts as a proposal evaluation. The neighbourhood is the deduplicated union of:

- the 4 nearest targets by KL(fine-cell mean teacher probability ‖ target's current released q);
- the 4 targets with the best exact objective change. For C-TASK this is the task objective, without SEX.

Ties are broken by (value, target canonical index = lowest member fine index).

**Acceptance (F N-10).** This is **per-cell best-improving in one deterministic sweep order. It is not a global
steepest-descent move.**

1. Fine cells are visited in increasing index, recipients 1 then 2. In local arms each recipient is searched alone; in
   sequential arms, one recipient per stage.
2. For each cell, the best strictly improving (Δ < −1e-12) feasible member of its neighbourhood is applied at once.
   Ties within 1e-12 go to the lowest canonical target.
3. On constrained recipients, improving neighbourhood members that fail a budget or cap are counted as
   `rejected_by_budget` (L, B, I).
4. Every applied move is re-checked on exact state values: a strictly lower objective, and every enforced constraint.
   A move that fails is reverted and counted as `rejected_on_exact_recheck` (0 in every synthetic run).

**Sweeps and stopping.**

- There are at most 5 sweeps per stage or start, with one ordering per start.
- Each sweep restarts from the coherent accepted state.
- The stop reason is recorded per stage or start (F N-7): `no_change_sweep` (no accepted single or paired move),
  `sweep_cap` or `eval_ceiling`.
- The search is a greedy local search. No global optimum is claimed.

### 6. Paired step (K-JOINT-PAIR only; after each single sweep; at most one accepted pair per step)

1. **Pool.** For each recipient, the pool is every neighbourhood candidate of every fine cell at the post-sweep state
   that satisfies **that recipient's own** budgets and local cap. Non-improving candidates are included, so the bank
   is not restricted to moves that already improve Φ alone.
2. **Retain.** Keep 4 by exact one-sided Φ change, then 4 more (not already kept) by exact task-loss change
   L_i + 0.5 B_i. Ties are broken by (value, fine cell, target canonical index).
3. **Evaluate.** Every pair in the 8 × 8 Cartesian product is evaluated atomically: both moves are applied, the joint
   table and both decoders are recomputed, every constraint is re-checked on exact values, and the state is reverted.
4. **Accept.** The best feasible strict Φ improvement is accepted (< −1e-12); ties go to the first in (i, j) order.
   It is then re-checked once more.

An infeasible paired update is refused and fully reverted (tested).

**Ceiling (F R-4, fixed 2026-10-07).** The pool screening and every pair evaluation count toward the start's remaining
share, and both stop at it. The share is checked before each screened cell and before each pair. A step that reaches
it records `hit_ceiling` and stops the stage at `eval_ceiling`. Before this fix the screening (about one full pass over
both recipients) ran unchecked, and the pair budget was taken from the pre-screening remainder. One step could then
overshoot a witness share by about 49K evaluations, about 18% of 8e6/29. Results do not change when the ceiling does
not bind (tested: `test_pair_step_stops_at_its_ceiling_share`).

### 7. Equal work (F N-7)

**Ceiling.** Every unit has the same TOTAL proposal-evaluation ceiling, `EVAL_CEILING = 8,000,000`. One proposal
evaluation is one exact (cell, target) objective change, or one atomic pair; pair-pool screening counts.

**Allocation.**

| Arm type | Share |
|---|---|
| joint | ceiling / #witnesses per witness |
| sequential | ceiling / (2 · #starts) per stage per start |
| local | ceiling / (2 · #starts) per recipient |

So K-JOINT-SINGLE, K-JOINT-PAIR, K-SEQ-12 and K-SEQ-21 have equal totals. The share is checked before each fine cell,
and in the paired step before each screened cell and each pair. A stage's evaluations are therefore at most its share
plus (cap − 1) of its recipients, which the replay checks. Unused share is not transferred between starts.

**Recorded.** For every start and unit: actual evaluations, ceiling share and stop reason per stage or start, decoder
solves, solve calls, memo hits, misses and verified entries, accepted moves, paired evaluations and accepted pairs, and
CPU.

**Interpretation rule.** A claim-C (paired joint) failure where JOINT-PAIR starts stop at `eval_ceiling` is a
**budget-limited** result, not evidence that pairing cannot help. The stop reasons in the synthetic timing are in
section 12. In that run one K-JOINT-PAIR start stopped at `eval_ceiling`. Real data could differ, and the per-start stop reasons and evaluations are recorded in the record and in the trace termination receipts.

### 8. Decoder-solve cache

Each state keeps a memo per (recipient, class, fine cell, slot). An entry is the D1 solve of the canonical statistics
of (slot XOR {cell}).

- Entries are filled from the canonical fold.
- An entry is indexed by the slot's member-set version id. Every membership change gets a fresh id, and undo restores
  the old id.
- Every applied move re-folds the two entries it uses and requires **bitwise** equality of (n_t, y_t, s_t). A stale
  entry raises "stale cache key".
- So a solve is never reused across different label counts or teacher sums. Tests cover a 1-ulp teacher-sum change, a
  label-count change and random move/undo sequences under full bitwise verification.
- Misses are solved in one `solve_batch` per class pass. Rows are independent and bitwise equal to `solve_token`. A
  test that replaces the batch path with row-by-row scalar solves gives **identical moves and policy**.

### 9. Persisted trace (`trace.json`, schema `lra-mapper-trace-v1`)

The predecessor's verifier could not independently replay the incremental mapper checks, because start labels and move
traces were not saved. Every unit of every arm now returns a trace. It holds coherent starts, accepted moves,
sufficient-statistic hashes and termination receipts (prompt section 8). Its sha256 is `record.trace_sha256`, the
sha256 of `json.dumps(trace, sort_keys=True, separators=(",", ":"), allow_nan=False)`. Floats round-trip bitwise through
JSON.

**Labels.** A canonical label is the lowest member fine index of the cell's token. Labels are given per recipient
("1", "2") as an int list over ALL F fine cells, including cells without fitting rows.

**Top level.**

- schema, arm, config, lam, kind (local / seq / joint), fixture_mode, constrained, sex_used_in_search;
- objective_weights, eval_ceiling, caps, budget_allowance, rules_sha256;
- hash_rule and move_fields (the HASH_RULE and MOVE_FIELDS constants);
- starts;
- winner: start, position, kind, status, canonical labels (= `labels_from_policy` of the released policy.json), and
  objective.

**Per start.**

- name and registered position;
- `labels`, the canonical start labels;
- stages;
- `final_labels` (only when a refined candidate exists) and `eligible`;
- joint arms only: `joint_witness`, with the unchanged terms, `feasible_all` and the status (REFINED or
  EXCLUDED_INFEASIBLE_WITNESS). An excluded witness has no stages.

**Per stage.**

- `stage` (r1, r2 / seq1, seq2 / joint), its recipients and enforced recipients, and `start_labels`;
- `start_terms`, and the stage-start state hashes (`start_stats_sha256`, `start_q_sha256`);
- status: REFINED or INFEASIBLE_START. An infeasible start has no moves and a null termination receipt;
- the ordered accepted moves and the termination receipt. The receipt holds stop, sweeps, evals, ceiling_share,
  accepted, pairs_accepted, pair_steps, objective_start and objective_end.

**Sequential stages.**

- **seq1** starts from {a: the start's a-map, b: CLASS-ONLY}. Its `partner` holds the recipient, view CLASS-ONLY, the
  class labels, `constraints_enforced: false`, and its constraint values at the start and the end of the stage.
- **seq2** starts from {a: the frozen seq1 end map, b: the start's b-map}. Its `frozen` holds the recipient and the
  labels.

**Local stages.** r2 starts from r1's end state.

**Per move.** Each move is one atomic entry:

- step, sweep, type (single or pair), parts, terms_after (all state terms) and objective_after.
- A part holds r, f, `from_canon` and `to_canon` (the canonical labels of the source and target tokens BEFORE the
  move), and from_slot / to_slot.
- A part also holds `stats_sha256` and `q_sha256` for [from-token, to-token] AFTER the move. These are exactly the
  memo-served statistics and the cached solve the search used. A hash is null for an emptied token, and the q hash is
  null for a token with n_t = 0.
- Single moves carry the vectorised `delta`, dL, dB, dI and dI12.
- A pair lists r = 1 then r = 2. Each part carries its one-sided dL, dB, dI, dI12, dPhi_one_sided and dTask on the
  pre-pair state.

Applying lab[f] = to_canon and re-canonicalising gives the next state.

**Hashes (`HASH_RULE`).**

- A token's stats hash is sha256(n_t `<i8` ‖ y_t `<f8` ‖ s_t `<f8`).
- A token's q hash is sha256(q_t `<f8`).
- The stage-start state hash runs over recipients 1 then 2 and their nonempty tokens in canonical order, hashing
  (label ‖ n ‖ y ‖ s); the q hash runs over the same tokens with n_t > 0.

The record keeps its compact move log `[sweep, r, f, from_slot, to_slot, delta]`. The replay requires it to equal the
trace's single moves, together with stop, sweeps and evals.

Synthetic cost (section 12): trace building is under 1% of fitting CPU. A trace is up to about 15 MB of JSON per W-JOINT
unit; the full bank's sizes are in section 12.

### 10. Independent replay (`replay_unit`; correctness checks 6, 7, 8)

`replay_unit(record, fine_dict, T, tr, Y_fit, S_fit, refs=None, *, trace, files=None, per_state=True, starts=None)`
takes the inputs given to fit_unit, the record and the trace. `files` is optional: {policy.json, release.npz}, which
enables the release checks. `starts` is optional: the start/witness policy dicts, whose labels are cross-checked.

It never raises on a defect. It raises ValueError only for malformed problem inputs.

It returns:

```
{ok, arm, config, max_abs_diff, max_abs_diff_by_term, max_delta_abs_diff, n_states, bitwise_equal_states, n_moves,
 n_solves_checked, fresh_solves, partner_infeasible_states, tolerances, checks{... check6, check7, check8},
 per_state[{start, stage, step, kind, enforced, feasible_enforced, feasible_both, max_abs_diff}],
 violations[{code, start, stage, step, detail}]}
```

**From-scratch evaluator (`_Scratch`).** It uses none of State, its slots, its version-indexed memo or its incremental
tables.

- Per state, every token's statistics are folded from zeros over its member cells in increasing fine index.
- Each token is solved FRESH with `lra.decoder.solve_batch`. Solves are memoised only by the exact member set, which is
  a pure function of the partition and the fixed data.
- Losses come from `token_losses`, summed by the same sorted-nonzero rule.
- MI comes from the fine-cell SEX table, which is built once from the fitting rows and aggregated by the labels.

**Procedure.** The replay rebuilds each start from its persisted labels and checks the labels are canonical,
class-pure and within the caps. It then derives every stage start (the seq1 CLASS-ONLY partner, the seq2 frozen map,
local r2 continuing from r1) and compares it with the trace. Moves are applied IN ORDER; a paired move is applied
atomically.

**Checks at every accepted state.**

- Recorded terms vs from-scratch must agree within `REPLAY_TERM_ATOL` = 1e-12. Bitwise equality is counted
  separately.
- Vectorised deltas, and the pair one-sided deltas, must agree within `REPLAY_DELTA_ATOL` = 1e-12. The objective delta
  is allowed 1e-12 · (1 + Σ|w|).
- The statistics hashes and cached q of both affected tokens must equal the fresh fold and the fresh solve
  **bitwise**.
- Every accepted move must be a strict improvement.
- The stage's enforced budgets and local caps must hold on from-scratch values, with the search margins.
- A paired state must satisfy both recipients' constraints.
- An eligible constrained candidate must satisfy both recipients' constraints at every state outside seq stage 1.

**Unit-level checks.**

- **References:** U losses, caps, allowance, I_ctask (bitwise equal to the exact MI of the C-TASK start), the eval
  ceiling, the rules hash, the trace hash and the objective weights. starts_given must be in registered order, and must
  be the full registered set outside fixture mode.
- **Sequential:** the stage-1 partner holds the CLASS-ONLY labels, and its enforced set is the first recipient only. A
  stage-1 start that meets its own constraints must have been refined; otherwise the replay reports
  PARTNER_REQUIRED_FEASIBLE. The frozen map never moves. The final release meets both budgets, and both caps on K-
  arms.
- **Joint:** an infeasible witness is EXCLUDED, and never refined or eligible.
- **Termination receipts:** they must agree with the moves, the registered share (evals ≤ share + cap − 1), the stop
  rule and the record.
- **Winner and release:** the winner is recomputed from scratch with the registered rule; the record's
  final_state_terms are checked; with files, the released policy labels, tokens, decisions and q (bitwise equal to the
  fresh solves), and row-level parity.

**Violation codes (`REPLAY_CODES`).**

- TERMS_MISMATCH, DELTA_MISMATCH, STATS_HASH_MISMATCH (stale cache key), CACHED_SOLVE_MISMATCH;
- NOT_STRICT_IMPROVEMENT, ENFORCED_CONSTRAINT_VIOLATED, PAIR_INFEASIBLE;
- PARTNER_NOT_CLASS_ONLY, PARTNER_REQUIRED_FEASIBLE, FROZEN_MAP_CHANGED;
- FINAL_INFEASIBLE, FINAL_STATE_MISMATCH, RELEASE_MISMATCH, WINNER_MISMATCH, INFEASIBLE_WITNESS_ELIGIBLE;
- MOVE_INVALID, START_MISMATCH, TERMINATION_INVALID, TRACE_INCOMPLETE, REFERENCE_MISMATCH, TRACE_HASH_MISMATCH.

**Gate mapping (`REPLAY_GATE`).**

| Gate check | Passes when none of these codes is present |
|---|---|
| check 6 | ENFORCED_CONSTRAINT_VIOLATED, PAIR_INFEASIBLE, FINAL_INFEASIBLE |
| check 7 | PARTNER_NOT_CLASS_ONLY, PARTNER_REQUIRED_FEASIBLE, FROZEN_MAP_CHANGED, FINAL_INFEASIBLE |
| check 8 | TERMS_MISMATCH, DELTA_MISMATCH, STATS_HASH_MISMATCH, CACHED_SOLVE_MISMATCH, START_MISMATCH, TRACE_INCOMPLETE, TRACE_HASH_MISMATCH |

`partner_infeasible_states` counts stage-1 states where the CLASS-ONLY partner violated its own constraints. These
were tolerated, as check 7 requires.

**Synthetic result.** On all ten arms:

- every replayed state is **bitwise** equal to the from-scratch rebuild (max_abs_diff 0);
- deltas agree to within 5e-16;
- every cached solve equals its fresh solve bitwise;
- K-JOINT-PAIR's accepted pairs pass.

Verifier E mirrors this independently in `PKG/verification/replay_lra.py`.

### 11. Matched-control rules verified against the code (prompt section 8), 2026-10-07

| Rule | Verdict |
|---|---|
| Neighbourhood: 4 nearest decoded-probability targets ∪ 4 best exact-objective targets, deduplicated, canonical ties; C-TASK ranks by the task objective and SEX does not enter (s = None; permutation test) | matches |
| Paired step: per recipient, 8 proposals that satisfy that recipient's own budgets and local cap, including non-improving ones; 4 ranked by one-sided Φ and 4 by task loss L_i + 0.5 B_i; 8 × 8 Cartesian product, atomic (both decoders and the joint table recomputed); accept only a feasible strict Φ improvement; after each single sweep | matches |
| JOINT-SINGLE has the same total ceiling as JOINT-PAIR (8e6, E // #witnesses per witness, same 30 witnesses) | **needed change**: the pair-step pool screening ignored the share (F R-4). Fixed in section 6 |
| Both sequential orders have the same total ceiling, split across their two stages (E // (2 · #starts) per stage per start) | matches |
| Constrained LOCAL starts from C-TASK | matches |
| Sequential arms start from C-TASK and the six source SEQ-ab maps (D1 recomputed) | matches |
| Joint arms use feasible witnesses: C-TASK, LOCAL, SEQ-12, SEQ-21, FINE-TASK, the 24 source privacy maps; an infeasible witness is never eligible | matches. **Added: source CLASS as a witness** (lead decision 2026-10-07, "feasible source mappings"), giving 30 witnesses; run._fit_args matches |
| D bank (W-) arms use the same decoder, start/witness structure, caps, single-move neighbourhood and ceiling as the corresponding K- arms | matches. The "4 best" component and acceptance rank by each arm's own exact objective (T + λΦ, or T + λ(I1 + I2)/2). This is registered in SEARCH_RULES `weighted_controls` |

### 12. Synthetic timing (TIMING.json["fitting"])

`python -m lra.mapper timing --seeds 0 --out TIMING.json` ran once under `lra.sema` (label `C:fit-timing`, one
thread) on SYNTHETIC real-shaped data, 2026-10-07 04:44-05:01Z.

**Shapes.**

- 39,170 rows, of which 15,434 are fitting rows;
- income K = 2, with 2 predicted classes × 32 fine cells;
- occupation K = 6, with 5 predicted classes × 128 fine cells, class 5 a fallback (F = 64 / 641);
- caps 8 / 64.

The labels come from the miscalibrated synthetic law. The source D0 maps (FINE-TASK, CLASS and the 24 privacy maps)
were fitted with plain qpc. That setup took 42 CPU-s and is not counted.

**What was measured.**

- all 30 new mapping-pair units of one seed, through `fit_unit` with the registered start and witness sets. This
  includes the trace build, the decoder.json / release.npz build and the deployed recomputation;
- the 27 D1 fixed-map decodes per seed (a FINE-TASK map stands in for DIRECT-TASK, and CLASS|D1 is included);
- an independent `replay_unit` of every unit on its JSON-loaded trace, with files, timed separately.

**Measured code.** The run used mapper.py b69347ab, which had 29 joint witnesses. The final 6f619c94 differs only by
the CLASS witness added after launch, and the replay order check. TIMING.json `code_note` records this, with
`code_unchanged_during_run = False` for that reason only. The extra witness costs one unchanged evaluation per joint
unit and is EXCLUDED unless it is feasible.

| Arm | Units (1 seed) | fit CPU-s total | mean | max | trace CPU-s (in fit) | trace MB max | record MB max | replay CPU-s total |
|---|---|---|---|---|---|---|---|---|
| C-TASK | 1 | 3.2 | 3.2 | 3.2 | 0.02 | 0.47 | 0.0 | 1.7 |
| K-LOCAL | 1 | 2.3 | 2.3 | 2.3 | 0.01 | 0.36 | 0.0 | 1.1 |
| K-SEQ-12 | 1 | 5.6 | 5.6 | 5.6 | 0.04 | 1.09 | 0.1 | 3.0 |
| K-SEQ-21 | 1 | 7.5 | 7.5 | 7.5 | 0.05 | 1.34 | 0.1 | 3.6 |
| K-JOINT-SINGLE | 1 | 19.6 | 19.6 | 19.6 | 0.13 | 3.67 | 0.2 | 10.0 |
| K-JOINT-PAIR | 1 | 22.3 | 22.3 | 22.3 | 0.11 | 3.72 | 0.3 | 9.2 |
| W-LOCAL | 6 | 4.3 | 0.7 | 0.9 | 0.02 | 0.04 | 0.0 | 0.8 |
| W-SEQ-12 | 6 | 107.3 | 17.9 | 18.8 | 0.80 | 3.65 | 0.2 | 43.4 |
| W-SEQ-21 | 6 | 105.0 | 17.5 | 18.3 | 0.71 | 3.58 | 0.2 | 37.1 |
| W-JOINT | 6 | 445.5 | 74.2 | 77.5 | 2.98 | 15.36 | 0.9 | 128.3 |

Totals for one seed:
- the 30 new units took 723 CPU-s, of which 4.9 CPU-s (0.68%) built the trace;
- the 27 D1 fixed-map decodes, including CLASS|D1, took 5.6 CPU-s;
- traces totalled 142 MB of JSON (48 MB with zlib) and records 8.4 MB;
- the independent replay of all 30 units took 238 CPU-s, with every unit `ok` (replay_ok_all_units = True);
- worker peak RSS was 548 MiB.

Projection for 3 seeds (3 C-TASK + 72 weighted + 15 constrained = 90 new units, plus 81 D1 decodes):
- fitting: 2184 CPU-s = **0.61 CPU-h**, or 1.21 CPU-h with a ×2 margin;
- traces: about 0.42 GB of JSON; records: 25 MB;
- an independent replay of every unit: 0.20 CPU-h, which is not part of fitting.

Search behaviour (synthetic, descriptive):
- every unit is ['FEASIBLE'];
- stop reasons seen: ['eval_ceiling', 'no_change_sweep', 'sweep_cap'];
- units with a start or stage stopped at `eval_ceiling`: ['U|K-JOINT-PAIR|i8o64|D1'];
- the largest unit used 4,433,346 of the 8,000,000 evaluations (U|W-JOINT|i8o64|l0.01|D1);
- K-JOINT-PAIR used 2,220,049 evaluations, with 2,560 pair evaluations and 17 accepted pairs;
- K-JOINT-SINGLE used 1,142,548.

**R-4 seen in practice.** Under the fixed pair-step ceiling, one K-JOINT-PAIR start stopped at `eval_ceiling`. The
pre-fix run (01:08Z) gave the same unit 2,241,950 evaluations, 2,752 pair evaluations and 17 accepted pairs; the fixed
run gives 2,220,049, 2,560 and 17. So the old screening let that start overshoot its 275,862 share without recording
`eval_ceiling`. Every other unit reproduces the pre-fix evaluation counts exactly. On the test world, the old and new
code give identical moves and policy when the ceiling does not bind. At a share of 272 the old code reached 456-464
evaluations per start, while the new code stays at most share + cap - 1.

On Adult, with 30 witnesses, the share is 266,666. A K-JOINT-PAIR start may bind, and the interpretation rule of
section 7 then applies.

**Recommendation.** Run the FULL bank (90 new units and 81 D1 decodes) with no reduction, in 2 shards under
`lra.sema`:

- the `ctask` stage first (3 units);
- then `lra.run.fit_chains`, constrained chains first;
- the `d1` stage is independent.

Fitting is about 0.61 CPU-h, or 1.2 CPU-h with a ×2 margin, against the 3 CPU-h fitting line. The traces take about
0.42 GB of plain JSON. Replaying every unit costs about 0.2 CPU-h more, for B and E.

### 13. Tests (`lra/tests/test_mapper.py`, synthetic)

There are 26 tests, all passing (13.3 s under `lra.sema`, one thread, mapper.py 6f619c94).

**Fixture.** Synthetic only (`cbp.fit.synthetic_teacher` plus `synthetic_labels`):

- seed 4, 4,000 rows of which 2,400 are fitting rows;
- fine partitions of 8 / 12 cells per class, caps 4 / 6;
- fixture-mode budget 0.05 / 0.03, so that the constrained arms have feasible starts;
- this shape was chosen so that K-JOINT-PAIR accepts paired moves.

All ten arms run on it, including W-LOCAL, W-SEQ-12, W-SEQ-21 and W-JOINT at λ = 0.01. The joint witnesses include
source CLASS.

**Tests carried over from the predecessor (17).**

| Test | Checks |
|---|---|
| `test_incremental_terms_match_from_scratch_after_moves` | Over 20+ random moves with all four terms active: every exact state term is bitwise equal to a freshly built state on the same partition; every vectorised delta (dL, dB, dI, dI12) matches the exact change within 1e-12; undo restores terms, tables and labels bitwise |
| `test_deployed_recomputation_parity_every_unit` | For every fitted unit: state vs deployed row-level terms ≤ 1e-10; released q bitwise equal to the search q; independent per_row recomputation from release.npz |
| `test_budgets_and_local_caps_enforced_including_paired` | All five K- arms are FEASIBLE with feasible deployed releases; L, B and the local cap hold on the deployed rows; every eligible refined candidate is feasible; the paired step ran and evaluated pairs |
| `test_paired_step_refuses_infeasible_update` | With any state where both recipients moved declared infeasible, no pair is accepted, every pair evaluation is counted as infeasible, and the state is bitwise unchanged |
| `test_pair_bank_keeps_non_improving_proposals` | At most 8 distinct retained proposals per recipient, and non-improving proposals are retained |
| `test_class_preservation_and_caps` | Released decisions equal the teacher decisions on all rows; q argmax equals the decision; token counts are ≤ the caps; decoder.json loads, binds to the pair and matches the record hash |
| `test_start_over_caps_refused` | A start above the caps is refused |
| `test_sequential_temporary_partner_not_required_feasible` | The CLASS-ONLY partner violates its budgets in stage 1, yet stage 1 refines and the final release is feasible |
| `test_determinism` | A re-fit gives an identical record (excluding CPU/wall), and identical policy, decoder hash, release arrays and **trace** |
| `test_memo_misses_on_changed_statistics` | After a move, exactly the changed slots' entries miss and equal fresh solves; a stored entry with a 1-ulp teacher-sum change or a changed label count raises "stale cache key" |
| `test_memo_version_lookup_equals_verified_lookup` | Random move/undo/refresh sequences, with full bitwise verification of every entry |
| `test_scalar_decoder_path_gives_identical_moves` | Replacing solve_batch with row-by-row M = 1 solves gives identical accepted moves and policy |
| `test_ctask_uses_no_sex` | Permuting SEX leaves the C-TASK map identical; the report has D1-init, refined and D0 views |
| `test_equal_ceilings_and_ceiling_stop` | Every K- arm records the same 8e6 ceiling. With a ceiling of 200, joint starts stop at `eval_ceiling` within their share, and the sequential stages stay within their halves |
| `test_joint_infeasible_witness_never_eligible` | Witnesses are evaluated in registered order; infeasible ones are EXCLUDED and never win; the winner's Φ ≤ every feasible unchanged witness |
| `test_refusals` | Missing I_ctask; I_ctask off by 1 ulp; unregistered key; λ on a K- arm; off-grid λ; a non-registered budget outside fixture mode; missing JOINT witnesses; missing binding |
| `test_rules_are_finite_json_and_registered_starts` | SEARCH_RULES content: start counts 1 / 7 / **30**, the witness order (…, FINE-TASK, CLASS, 24 source maps), tolerances, sweeps and budget values |

**New trace and replay tests (9).**

| Test | Checks |
|---|---|
| `test_trace_persisted_for_every_arm_and_round_trips` | Every arm returns trace.json. It is JSON round-trip equal, its hash equals record.trace_sha256, and it is not in record.json. Every start has labels over all F cells, and every refined stage has a complete termination receipt. Every move part has stats and q hashes. The winner labels equal the released policy. SEQ stage 1 records the CLASS-ONLY partner (constraints_enforced false), and stage 2 the frozen map |
| `test_replay_passes_on_every_arm_including_accepted_pair` | replay_unit on the JSON-loaded trace, with files, is `ok` for all ten arms. Every check is true, max_abs_diff ≤ 1e-12, deltas ≤ 1e-12, and the move counts match. Every accepted K-state meets its enforced constraints, and K-JOINT-PAIR has ≥ 1 accepted pair whose states satisfy both budgets and both caps. On K-SEQ, check 7 holds with partner_infeasible_states > 0. The replay with the start maps passes |
| `test_replay_catches_corrupted_move` | A move redirected to another same-class token gives TERMS/STATS/DELTA (check 8 fails). A move to a non-token gives MOVE_INVALID. A corrupted dL gives DELTA_MISMATCH. Corrupted terms_after give TERMS_MISMATCH. A dropped move gives not ok. An untouched record hash gives TRACE_HASH_MISMATCH |
| `test_replay_catches_stale_cache_key` | A trace part carrying the pre-move statistics hash gives STATS_HASH_MISMATCH. A real injected defect, a cached solve served under the correct statistics key but holding the pre-move solve, is NOT noticed by the search: its deployed parity still passes, but the replay reports CACHED_SOLVE_MISMATCH and check 8 fails. (On a constrained arm, the mapper's own final deployed-feasibility guard refused such a fit, so the weighted SEQ arm is used.) |
| `test_replay_catches_infeasible_paired_update` | A real defective paired step (own-budget screen and joint re-check skipped) accepts pairs. The replay reports PAIR_INFEASIBLE at those pair steps and check 6 fails; the honest unit passes |
| `test_replay_catches_partner_wrongly_required_feasible` | Trace-level: seq1 enforced [1, 2] gives PARTNER_REQUIRED_FEASIBLE. A real defect, stage 1 requiring the CLASS-ONLY partner feasible, makes every stage-1 start INFEASIBLE_START (the baseline is impossible by construction); the replay reports PARTNER_REQUIRED_FEASIBLE and check 7 fails. A partner that is not CLASS-ONLY gives PARTNER_NOT_CLASS_ONLY |
| `test_replay_catches_ineligible_witness_and_final_mismatch` | An excluded witness marked eligible gives INFEASIBLE_WITNESS_ELIGIBLE (source CLASS is exercised). A 1e-9 change to final_state_terms gives FINAL_STATE_MISMATCH. A 1-ulp change to a released q gives RELEASE_MISMATCH |
| `test_replay_requires_registered_witness_order` | Swapped witnesses (consistent hash) give START_MISMATCH |
| `test_pair_step_stops_at_its_ceiling_share` | F R-4. An unbounded step records the screening plus the pair evaluations. With budget 7 it aborts in the screening (no pair evaluated). With budget = screening + 5 it stops after ≤ 5 + cap − 1 pairs. Both set hit_ceiling and stay within budget + cap − 1 |


### 14. Not claimed

- No global optimality: the search is a greedy local search with ≤ 5 sweeps.
- No convexity of the discrete program: only the fixed-token decoder is convex.
- No population MI, SEX-AUC or confidence bound from the fitting budgets.
- No equality of held-out gains from fitting-objective dominance.
- The sequential arms are matched adaptations, not the official Taylor solver.
- A D1 decoder change on an unchanged token does not remove information from that token. Every arm's privacy terms
  use full token identities.
