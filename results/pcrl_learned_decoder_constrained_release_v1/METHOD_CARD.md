# Method card — learned decoder D1 and confidence-constrained mapper (lcr)

This card merges the two role cards. They stay in the package unchanged: METHOD_CARD_DECODER.md (role B: decoder,
fixtures, deployment) and METHOD_CARD_MAPPER.md (role C: mapper).

**Status.** The registered mechanism gate returned GATE_NOT_MET (FIXTURE_GATE.json; RESEARCH_DECISION.md), so no Adult
fit was run.

| Component | What it ran on |
|---|---|
| Decoder (D1) | the four known-law fixtures (228 certified decoders); synthetic tests |
| Mapper | the fixtures (constrained and C-TASK arms EXHAUSTIVE_OPTIMAL against their own exhaustive references on F2–F4; a heuristic in general); synthetic tests and timing |
| Deployment | the admitted D0 Q map on Adult (DEPLOYMENT_RECEIPT.json); the D1 path on synthetic data only |

Every Adult statement below describes how the code WOULD run on Adult. None of it is an observed Adult result.

**What D1 can and cannot do.**
- Changing the decoder of an unchanged token removes no information (C1 holds bitwise on every fixture).
- In the calibrated-teacher null, D1 equals the mean decoder within 8.1e-13 (C2).
- On the miscalibrated fixture F2, D1 made all 27 of the fixture's fixed mean-decoder maps (task-only and
  privacy-trained) budget-feasible, against 0 under D0, at identical information. This was not tested on Adult.

**Prior art.** Hard utility constraints, privacy-funnel objectives (arXiv:1402.1774), proper-loss calibration, greedy
partition search and sequential collusion accounting (Taylor, Vippathalla and Coon, arXiv:2601.21859 v2) are
established. Their combination here is not claimed as algorithmic novelty. The sequential arms are matched
adaptations, not the official Taylor solver. The full joint discrete program is not convex; only the fixed-partition
decoder is.

# Part A — decoder, fixtures and deployment (role B)

Code: `lcr/decoder.py`, `lcr/fixtures.py`, `lcr/deploy.py`. Proofs and checks: `MATH_REVIEW.md`.
Fixture laws and gate: `FIXTURE_LAWS.json` and `FIXTURE_GATE_RULE.json`. The laws (dd1cf23) predate every algorithm
run on them. The gate rule was first registered at dd1cf23 and clarified at 47beb1f, after role E's oracle had
enumerated the laws (correctness checks and descriptive definitions only; VALIDATION.md §4). FIXTURE_LOCK governs
both. (Merged-card note; role B's original wording is kept in METHOD_CARD_DECODER.md.)

### 1. What D1 is

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

### 3. Certificates and tolerances (`lcr.decoder.TOLERANCES`; DECODER_CERTIFICATES.json via `certificate_summary`)

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

### 7. What the decoder can and cannot do

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

### 8. Fixture stage and mechanism gate (prompt section 9)

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

### 9. Deployment (`python -m lcr.deploy`)

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

# Part B — mapper (role C)

Owner: role C (mapper engineer). Tests: `lcr/tests/test_mapper.py`, run on synthetic data only. Frozen rules:
`SEARCH_RULES.json`, written by `python -m lcr.mapper rules` (its `rules_sha256` is also recorded in every unit
record). Role C code reads no Adult row, label or SEX column. The lead runs the real stages (`lcr.run --stage ctask`,
`--stage fit`) after SCIENCE_LOCK.

**What is new.** The source study fitted codebook probabilities to the teacher (D0 = smoothed token mean) and swept a
privacy weight. This mapper:

- decodes every token with the shared learned decoder D1 (`lcr.decoder`, role B);
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
    -> (record, {"policy.json": dict, "decoder.json": dict, "release.npz": dict of arrays})
refs_from_ctask(ctask_record) -> {"I_ctask": {"1", "2"}, "ctask_pair_fingerprint", "L_U", "B_U"}
```

**Arms.** `C-TASK`, `W-LOCAL`, `W-SEQ-12`, `W-SEQ-21`, `W-JOINT` (these need λ on the registered grid), `K-LOCAL`,
`K-SEQ-12`, `K-SEQ-21`, `K-JOINT-SINGLE`, `K-JOINT-PAIR`.

**Inputs.**

- `fine_dict` is the admitted `fine.json` ({"fine1", "fine2"}).
- `T` holds the teacher outputs over ALL rows (`row_id, p1, p2, d1, d2`).
- `tr` holds the OSF_DEFENSE_FIT indices.
- `Y_fit` is {1: income, 2: occupation_group}, aligned with `tr`.
- `S_fit` is binary SEX, aligned with `tr`.

**Starts and witnesses.** These are {config ID: `policy.json` dict}, keyed by `lcr.run` IDs. Only the cell→token map
of each is used. Its fine fingerprints must equal `fine_dict`, and its D1 decoder is recomputed.

**Outputs.**

- `policy.json` is the qpc.PolicyPair of the token map. Its `token_proto` are the D0 means, which are **not
  released**. `config.config` is the lcr ID, m1 = 8, m2 = 64, and both binding hashes are present.
- `decoder.json` is `lcr.decoder.decoder_pair_dict`.
- `release.npz` is `lcr.decoder.release_arrays_d1` over all rows. q there is the actual released D1 vector.

**Stable record fields used downstream.** `status` is `FEASIBLE` or `INFEASIBLE`. The others are
`deployed.feasible`, `final_state_terms`, `winner`, `token_counts` and `pair_fingerprint`.

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
  is bitwise `lcr.decoder.token_stats` and qpc `token_tables`;
- the SEX counts form an exact int64 table, together with the pair table n(s, t1, t2).

The fitting terms are:

| Term | Definition |
|---|---|
| q_t | `lcr.decoder.solve_batch(y_t, s_t, n_t, d_t)`, the actual released vector (κ = 32, ε = 1e-12, class-dominant, certified) |
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
| W-JOINT(λ) | T + λΦ | none | as K-JOINT-SINGLE; witnesses C-TASK, W-LOCAL(λ), W-SEQ-12(λ), W-SEQ-21(λ) and the 25 source maps |
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

**Joint (K-JOINT-*, W-JOINT).** Every witness is evaluated unchanged against all constraints.

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

### 7. Equal work (F N-7)

**Ceiling.** Every unit has the same TOTAL proposal-evaluation ceiling, `EVAL_CEILING = 8,000,000`. One proposal
evaluation is one exact (cell, target) objective change, or one atomic pair; pair-pool screening counts.

**Allocation.**

| Arm type | Share |
|---|---|
| joint | ceiling / #witnesses per witness |
| sequential | ceiling / (2 · #starts) per stage per start |
| local | ceiling / (2 · #starts) per recipient |

So K-JOINT-SINGLE, K-JOINT-PAIR, K-SEQ-12 and K-SEQ-21 have equal totals.

**Recorded.** For every start and unit: actual evaluations, decoder solves, solve calls, memo hits, misses and verified
entries, accepted moves, paired evaluations and accepted pairs, and CPU.

**Interpretation rule.** A claim-C (paired joint) failure where JOINT-PAIR starts stop at `eval_ceiling` is a
**budget-limited** result, not evidence that pairing cannot help. In the synthetic timing (section 9), no K- or
W-unit start reached the ceiling.

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

### 9. Synthetic timing (TIMING.json["fitting"])

`python -m lcr.mapper timing --seeds 0` was run once under `lcr.sema` (label `C:fit-timing`, one thread) on
SYNTHETIC real-shaped data, using the final code (`code_sha256_at_start` recorded;
`code_unchanged_during_run = True`). The shapes are:

- 39,170 rows, with 15,434 fitting rows;
- income K = 2, with 2 predicted classes × 32 fine cells;
- occupation K = 6, with 5 predicted classes × 128 fine cells and class 5 a fallback (F = 64 / 641);
- caps 8 / 64.

The labels come from a deliberately miscalibrated synthetic law. The source D0 maps (FINE-TASK and the 24 privacy
maps) were fitted with plain qpc. That setup took 44 CPU-s and is not counted. The bank run was:

- all 30 new mapping-pair units of one seed, through `fit_unit` with the registered start/witness sets, including the
  decoder.json / release.npz build and the deployed recomputation;
- the 26 D1 fixed-map decodes, with a FINE-TASK map standing in for the DIRECT-TASK map.

| Arm | Units (1 seed) | CPU-s total | mean | max |
|---|---|---|---|---|
| C-TASK | 1 | 2.9 | 2.9 | 2.9 |
| K-LOCAL | 1 | 2.1 | 2.1 | 2.1 |
| K-SEQ-12 | 1 | 5.4 | 5.4 | 5.4 |
| K-SEQ-21 | 1 | 7.3 | 7.3 | 7.3 |
| K-JOINT-SINGLE | 1 | 19.8 | 19.8 | 19.8 |
| K-JOINT-PAIR | 1 | 22.9 | 22.9 | 22.9 |
| W-LOCAL | 6 | 4.7 | 0.8 | 0.9 |
| W-SEQ-12 | 6 | 111.1 | 18.5 | 19.0 |
| W-SEQ-21 | 6 | 105.2 | 17.5 | 18.5 |
| W-JOINT | 6 | 448.5 | 74.8 | 77.5 |

**Totals.**

- The 30 new units took 730 CPU-s.
- The 26 D1 fixed-map decodes took 5.6 CPU-s.
- Worker peak RSS was 479 MiB.

**Projection for 3 seeds (90 new units + 78 decodes).** 2206 CPU-s =
**0.61 CPU-h**, or 1.23 CPU-h with a ×2 margin. The qpc/cbp
real/synthetic ratio was about 1.0. This is against the 3 CPU-h fitting line and the 20 CPU-h study.

**Search behaviour.** This is synthetic and descriptive only.

- Status of every unit: ['FEASIBLE'].
- Maximum state-vs-deployed parity: 1.0e-15.
- Stop reasons seen: ['no_change_sweep', 'sweep_cap'].
- Units with any start stopped at `eval_ceiling`: none.
- The largest unit used 4,433,346 of the 8,000,000 evaluations (U|W-JOINT|i8o64|l0.01|D1).
- K-JOINT-PAIR used 2,241,950 evaluations, with 2,752 pair evaluations and 17
  accepted pairs.
- K-JOINT-SINGLE used 1,142,548 evaluations.

So the equal ceiling does not bind on these shapes. The lead's note N-7 expected PAIR to bind at about sweep 3; it did
not here. Real data could differ, and the per-start stop reasons are recorded.

**Optimisations applied before this measurement.** Engineering only; no rule changed.

- The decoder memo is version-indexed, with a bitwise audit of every applied move.
- One batched refresh is made per accepted move.
- Role B vectorised `_state` over K, bitwise identical: a 256-row solve went from 6.6 to 3.2 ms.
- An earlier synthetic run of the same bank, on code before these optimisations, gave per-unit evaluation counts
  that were identical for all 25 units it completed, at 1.77× the CPU.

**Recommendation.** Run the FULL bank, with no reduction. Use 2 shards under `lcr.sema`: the `ctask` stage (3 units)
first, then `lcr.run.fit_chains`, constrained chains first. The `d1` stage is independent.


### 10. Tests (`lcr/tests/test_mapper.py`, synthetic)

17 tests, all passing (about 4 s, under `lcr.sema`, one thread). The fixture is synthetic: 3,000 rows, 1,600 fitting
rows, fine partitions of 6 / 9 cells per class, caps 3 / 5, and a fixture-mode budget of 0.05 / 0.03, so that the
constrained arms have feasible starts. All ten arms run on it.

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
| `test_determinism` | A re-fit gives an identical record (excluding CPU/wall), and identical policy, decoder hash and release arrays |
| `test_memo_misses_on_changed_statistics` | After a move, exactly the changed slots' entries miss and equal fresh solves; a stored entry with a 1-ulp teacher-sum change or a changed label count raises "stale cache key" |
| `test_memo_version_lookup_equals_verified_lookup` | Random move/undo/refresh sequences, with full bitwise verification of every entry |
| `test_scalar_decoder_path_gives_identical_moves` | Replacing solve_batch with row-by-row M = 1 solves gives identical accepted moves and policy, so the batch path changes no accepted move |
| `test_ctask_uses_no_sex` | Permuting SEX leaves the C-TASK map identical; the report has D1-init, refined and D0 views |
| `test_equal_ceilings_and_ceiling_stop` | Every K- arm records the same 8e6 ceiling. With a ceiling of 200, joint starts stop at `eval_ceiling` within their share, and the sequential stages stay within their halves |
| `test_joint_infeasible_witness_never_eligible` | Witnesses are evaluated in registered order; infeasible ones are EXCLUDED and never win; the winner's Φ ≤ every feasible unchanged witness |
| `test_refusals` | Missing I_ctask; I_ctask off by 1 ulp; unregistered key; λ on a K- arm; off-grid λ; a non-registered budget outside fixture mode; missing JOINT witnesses; missing binding |
| `test_rules_are_finite_json_and_registered_starts` | SEARCH_RULES content: start counts 1 / 7 / 29, tolerances, sweeps and budget values |


### 11. Not claimed

- No global optimality: the search is a greedy local search with ≤ 5 sweeps.
- No convexity of the discrete program: only the fixed-token decoder is convex.
- No population MI, SEX-AUC or confidence bound from the fitting budgets.
- No equality of held-out gains from fitting-objective dominance.
- The sequential arms are matched adaptations, not the official Taylor solver.
- A D1 decoder change on an unchanged token does not remove information from that token. Every arm's privacy terms
  use full token identities.
