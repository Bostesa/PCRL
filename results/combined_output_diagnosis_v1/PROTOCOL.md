# Protocol: output-leak diagnosis and replication

**Frozen in LOCK.json before any new fit.** The math and protocol reviewer's required changes are incorporated; see
`notes/review/STATS_REVIEW.md` and `notes/review/SCORE_MATH.md`.

> **This is planned analysis of repeatedly used development data. Prior outputs motivated the study. Naming a primary
> endpoint or committing a lock does not turn the existing assessment pool into fresh confirmation. New comparisons
> are development evidence, and no population privacy guarantee is inferred from attacker performance.**

**Base evidence.** research/combined-output-aware-removal-v1 @ f7425b15cb89bee841f5b1f9dd290d05a879ae1e. The
output-aware lock is 3130c8d4e48e4446854d0744e34b22997d985430 and the benchmark is
70f978ffc0afff55ecdf49cf1a74908cc3db5f49.

**No new data.** No new population, cohort, year or checkpoint family is opened. Adult (adult.data / adult.test) and
HMDA CA-2023 were used by every prior PCRL and durable-guarantees study.

## Data, roles, encoders, heads

**Encoders and labels.**
- Six frozen PCRL Round-4 encoders: Adult and HMDA × seeds 0, 1, 2.
- Admitted forward caches: representations and frozen-head logits, float64 copies of the float32 forward.
- The purpose task labels and the declared sensitive attributes, with their full declared classes and no rebinning.

**Roles: `oar-roles-v1`.**
- The 17 Adult / 42 HMDA training-overlapping test records are removed from every scored role. This is reconstructed
  from record identities; repair R1 fixes the label truncation.
- 20 % of attacker_fit groups form a certification set, used only by stage 5.
- Assessment membership is common to every surface, head, purpose and seed of a dataset.
- Encoder-validation-split exposure cannot be checked: those records are not admitted.
- Collapse collisions, where distinct records share an encoder vector, are kept and described (repair R2).

**Support.**
- 100 / 30 / 100 per class on attacker_fit / attacker_val / assessment. A pair is supported iff both its classes are.
- Unsupported classes and pairs are NOT_ESTIMABLE and never pooled. `COVERAGE_AND_SUPPORT.csv` lists every role and
  every pair.

**Heads.**
- **Frozen heads:** the PCRL Round-4 task heads, whose logits are in the forward caches.
- **Common refitted head:** HEAD__A from the output-aware study (LR grid, defense_fit minus the `oar-head-v1` holdout,
  selected on that holdout), reused by hash. Its outputs are log-probabilities.
- **Common refitted probe**, for utility only: U2__A (LR grid, attacker_fit / attacker_val).

## Stage 1: usefulness (mandatory)

**Statistics.** Per cell × seed, on assessment rows:
- accuracy, balanced accuracy, confusion matrix, per-class recall, prediction and true-class frequencies;
- log loss computed as −log_softmax(l)[y] in float64, never as the log of clipped probabilities (the education heads
  are saturated);
- task ROC-AUC (macro one-vs-rest for multiclass, with class coverage stated).

**Constant predictor.** Chosen on attacker_fit only: the majority class (for accuracy) and the class frequencies (for
log loss). Gain = accuracy − constant accuracy, reported per seed and as the mean over seeds. Discordant-row counts are
reported for every gain.

The same statistics are computed for U2__A and HEAD__A. Hard decisions have, by construction, the same accuracy as the
head they come from; probability-based utility is UNAVAILABLE for them.

## Stage 2: score decomposition (mandatory)

**Decomposition.**
- **Binary:** margin d = l1 − l0, offset c = (l0 + l1)/2. The `centred` surface is d (one column, non-redundant), and
  p1 = σ(d).
- **Multiclass:** c = mean_k l_k, and the centred vector is l − c (K columns). `dc` = (centred, c) for both cases.
- softmax, argmax and every proper task loss are unchanged when c is removed.

**Saturation, all computed and stored in float64.**
- p1 rounds to exactly 1.0 for d ≥ 36.7368 (= 53 ln 2).
- p1 rounds to exactly 0.0 only for d ≤ −745.13 with the stable softmax used here.
- Rows with |d| ≥ 16.6355 are counted, because they would saturate in float32.
- Saturation counts are reported per cell and seed.

**Surfaces.** Each uses the benchmark slate: L / GBT / MLP, NL = GBT vs MLP on attacker_val log loss, refitted at
attacker seeds 0, 1, 2.

| Surface | Content |
|---|---|
| `full` | (l0, l1) |
| `dc` | (d, c) |
| `centred` | d |
| `prob` | softmax |
| `offset` | c |
| `hard` | one-hot argmax |
| `REF` | label-only P(s \| true task label), a diagnostic that is never a release; plus the constant prior |

**Banks.** These are fit-free and validation-only. Selection uses the NL attacker's attacker-seed-0 validation log loss
recorded in each unit, once per encoder seed; ties go to the earlier-listed candidate; the L family is not a
candidate. The selected candidate for each seed is reported.
- **Ignore-offset bank (IO):** {centred, prob}.
- **Full bank (FB):** {full, dc, centred, prob}. FB contains every IO candidate.

**Strata.**
- Historical frozen head: primary.
- Common refitted head HEAD__A: secondary. Its log-probability offset is a deterministic function of d, so FB ≡ IO in
  information.

**Exactness checks.**
- argmax = 1[d > 0];
- softmax₁ = σ(d), maximum error reported;
- (d, c) ↔ (l0, l1) round trip.

**Controls.** In each primary cell, on seed 0, fit and validation rows only: a real-data shuffled-label null and a
planted leak, on `full` and `centred`.

## Primary family (30 elementary endpoints; frozen-head stratum)

**Estimate.** The original-sample value of the mean over encoder seeds of the per-seed statistic. For recovery, average
over attacker seeds first. Each bootstrap replicate is computed exactly the same way.

**Recovery R.** Supported-class macro one-vs-rest AUC on all assessment rows.

**Pair AUC (i < j).** Score P_j / (P_i + P_j) from the selected attacker, on the records of groups i and j, with j as
the positive class. The orientation is fixed in advance. If P_i + P_j = 0 the row scores 0.5; the number of such rows
is reported. The worst supported pair is also summarised.

| Block | Endpoints | Count |
|---|---|---|
| Usefulness | U-frozen-{adult,hmda}: Acc(frozen) − Acc(const) > 0.01; U-refit-{adult,hmda}: Acc(U2__A) − Acc(const) > 0.01 | 4 |
| Macro recovery | FC-{ds}: R(FB) − R(IO) > 0.02; CH-{ds}: R(IO) − R(hard) > 0.02 | 4 |
| Pair recovery | FC and CH for Adult sex pair (0,1) and all 10 HMDA race pairs | 22 |
| **Total** | 4 + 4 + 2·(1 + 10), asserted in `odx/family.py` | **30** |

**Declared before any fit:**
- **14 NOT_ESTIMABLE slots.** These are the HMDA pairs involving race group 3 (95 attacker_fit rows) or group 4 (53),
  for FC and CH. The slots stay in the denominator.
- **2 exact aliases.** FC-adult-pair0-1 ≡ FC-adult and CH-adult-pair0-1 ≡ CH-adult, because the attribute is binary.
  They are kept in the count and verified equal, but not counted as extra support.
- **UNAVAILABLE** applies to an estimable slot lost to a technical failure or to the budget; it stays in the
  denominator.

**Inference.**
- Paired group bootstrap over assessment record groups (canon_key units). Adult has no household IDs and HMDA has no
  applicant IDs.
- Multinomial unit weights, with the same draws for every unit of a dataset (B = 1,999, seed 20261021).
- SE = sd of the replicates with ddof = 1.
- Simultaneous two-sided 95 % Bonferroni intervals, estimate ± z·SE, with z = Φ⁻¹(1 − 0.05/60) = **3.143980**. This is a
  normal approximation, and it is stated as one.

**Decision.**
- **PASS** iff lower > target.
- **NOT_ESTABLISHED** otherwise.
- **NOT_ESTIMABLE** or **UNAVAILABLE** as declared.
- One-sided use of two-sided intervals keeps the family's false-PASS rate ≤ 0.025.

**Flags.**
- **IDENTICAL_BY_SELECTION:** both sides resolve to the same fitted attacker on every seed. Then SE = 0 and the result
  is NOT_ESTABLISHED; nothing is divided by the SE.
- **NORMAL_APPROX_WEAK:** fewer than 30 discordant rows.
- **NEAR_BOUND:** AUC > 0.98 or accuracy > 0.99. A percentile interval is added.
- **FC offset-attributable:** only if an offset-using candidate (`full` or `dc`) is selected by FB on every encoder
  seed.
- **Telescoped sum:** FC + CH = R(FB) − R(hard) is reported. It is the analogue of the prior full-minus-hard result.

**Relation to prior inference.** The output-aware study used one-sided percentile bounds at α = 0.05/12 (B = 20,000,
seed 20261011). This study uses two-sided normal Bonferroni intervals at 0.05/30 (B = 1,999, seed 20261021), as the
assignment requires. The per-endpoint level is about 5 times stricter here, so decisions are not directly comparable.
Intervals are conditional on the fitted attackers and heads, with no retraining variance; between-seed spread is
reported separately.

## Secondary families (locked before their fits)

**S3 replication (34; z = 3.180426).**
- For each of the 14 admitted pairs, FC and CH: 28 endpoints.
- Frozen-head usefulness for each of the 6 purposes: 6 endpoints.
- The two primary cells and their usefulness rows are deliberately **kept** in S3 as a within-family replicate. They
  are not new evidence.
- Multiclass heads use the centred K-vector.

**S4 multi-recipient (6; z = 2.638257).**
- Adult income + employment, attribute race, which is disallowed for both purposes in PERMISSION_TABLE.
- Row identities, roles, labels and exclusions are asserted identical across the two purposes.
- Nine slots per seed: income alone, employment alone and the pair, each under three contracts. For the single
  recipients the contracts map to: full = FB, centred = IO, hard = hard.
- The pair bank contains the pair attacker and both single-recipient attackers (ignore-the-other candidates).
- Endpoints: R(pair) − R(income) and R(pair) − R(employment), each > 0.02, for each contract.
- A combination finding for a contract **requires both of its contrasts to PASS**.
- Both tasks' utility is reported.

**S5 conditional FARE (4; z = 2.497705).** See stage 5.

## Stage 5: conditional FARE

Run only if the mandatory stages fit within budget.

**Replay.** The existing Adult / HMDA FARE, untreated, target-LEACE and zero-fairness-twin results under the complete
contract (features + own head), with gain over constant and retained share. The HMDA s1 one-cell nominee is reported
separately.

**New-cell screen.** This uses no assessment data.
- **Candidates:** every admitted (purpose, attribute) cell except the two primary cells.
- **Eligibility:**
  - the frozen head's attacker_val accuracy minus the attacker_fit-majority constant is ≥ 0.03 on average and > 0 on at
    least 2 of 3 seeds;
  - **and** the untreated U2 probe (LR grid on attacker_fit / attacker_val) has validation gain ≥ 0.03 on average and
    > 0 on **every** seed;
  - **and** every sensitive class has ≥ 100 defense_fit rows, with ≥ 2 supported classes.
- **Choice:** the highest mean frozen-head validation gain.
- **Ties:** broken lexicographically on the strings (dataset, purpose, attribute), which in effect selects the
  alphabetically first attribute of the purpose.
- If the selected task is a deterministic recoding of an input feature, that is noted.

**Grid.** The frozen six settings plus the zero-fairness twin. The official FARE pin and adaptations are verified to
support the new class counts before the fits.

**Nominee, validation only.**
- U2 validation accuracy ≥ untreated − 0.01, **and** (A_F − const) ≥ 0.8·(A_A − const).
- Among those, the lowest validation macro AUC; ties go to the lower id.
- If no configuration qualifies: **NO_FEASIBLE_NOMINEE**. There is no fallback "success". If any seed has no feasible
  nominee, or FARE admission fails, the S5 slots cannot PASS.

**S5 endpoints (Bonferroni 4).** Acc = U2 probe accuracy on assessment; complete contract = features + own head.
1. R(LEACE target, complete) − R(FARE, complete) > 0.02.
2. R(FZ, complete) − R(FARE, complete) > 0.02.
3. Acc(FARE) − Acc(A) > −0.01.
4. Retention, as the linear contrast mean[A_F − 0.8·A_A − 0.2·const] > 0.

The FARE certificate is reported separately and only under its stated premises.

## Runtime

- At most 10 elapsed hours and 12 CPU-hours, including verification.
- At most 2 heavy workers, with total memory under 6 GiB.
- The final 90 minutes are reserved for verification, backup and reporting.
- Runner ledger budget: 6 CPU-h.
