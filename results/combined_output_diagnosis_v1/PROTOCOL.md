# Protocol: output-leak diagnosis and replication (DRAFT, pending reviewer sign-off; frozen in LOCK.json)

> **This is planned analysis of repeatedly used development data. Prior outputs motivated the study. Naming a primary
> endpoint or committing a lock does not turn the existing assessment pool into fresh confirmation. New comparisons
> are development evidence, and no population privacy guarantee is inferred from attacker performance.**

Base evidence: research/combined-output-aware-removal-v1 @ f7425b15cb89bee841f5b1f9dd290d05a879ae1e (output-aware lock
3130c8d4e48e4446854d0744e34b22997d985430; benchmark 70f978ffc0afff55ecdf49cf1a74908cc3db5f49). No new population,
cohort, year or checkpoint family is opened. Adult (adult.data/adult.test) and HMDA CA-2023 were used by every prior
PCRL and durable-guarantees study.

## Data, roles, encoders, heads

- Six frozen PCRL Round-4 encoders (Adult, HMDA × seeds 0,1,2), admitted forward caches (`rep_p*`, frozen-head logits
  `logits_<purpose>`), the purpose task labels and declared sensitive attributes (full declared classes; no rebinning).
- Roles = `oar-roles-v1` (output-aware study): benchmark roles with the 17 Adult / 42 HMDA training-overlapping test
  records removed from every scored role (from record identities, not the role text column), and 20 % of attacker_fit
  groups as a certification set (used only by the conditional FARE stage). Assessment membership is common to every
  surface, head, purpose and seed of a dataset. Encoder-validation-split exposure cannot be checked (records not admitted).
- Collapse collisions (distinct records sharing an encoder vector) are kept and described, never deleted.
- Support: 100 / 30 / 100 per class on attacker_fit / attacker_val / assessment; a pair is supported iff both classes
  are; unsupported classes/pairs are NOT_ESTIMABLE (never pooled).
- **Frozen heads** = the PCRL Round-4 task heads (logits in the forward caches). **Common refitted head** = the
  output-aware study's release head on untreated features `HEAD__A` (benchmark U2 LR grid, fitted on defense_fit minus
  the `oar-head-v1` holdout, selected on that holdout) — reused by hash; its outputs are log-probabilities.
  **Common refitted probe** (utility only) = `U2__A` (LR grid, attacker_fit / attacker_val).

## Stage 1 — usefulness (frozen heads; mandatory)

Per cell × seed on assessment: accuracy, balanced accuracy, confusion matrix, per-class recall, prediction frequencies,
log loss (softmax of logits), task ROC-AUC (binary; macro OvR otherwise). Constant predictor = attacker_fit majority
class (accuracy) and attacker_fit class frequencies (log loss). Gain = accuracy − constant accuracy. Also for U2__A and
HEAD__A. Hard decisions: accuracy identical to the frozen head by construction; probability-based utility UNAVAILABLE.

## Stage 2 — score decomposition (mandatory)

Binary logits (l0, l1): margin d = l1 − l0, offset c = (l0 + l1)/2, centred = (−d/2, d/2), p1 = σ(d).
Multiclass (K): offset c = mean_k l_k, centred = l − c·1 (K columns, sum 0), probabilities = softmax(l), decision = argmax.

Surfaces (same attacker slate, roles, seeds; benchmark slate L/GBT/MLP, NL = GBT vs MLP on attacker_val log loss,
refit at attacker seeds 0,1,2):
`full` = (l0,l1) BANK · `dc` = (d, c) · `centred` · `prob` · `offset` (c alone) · `hard` (one-hot argmax) · `const`
(attacker_fit prior) · `label` (true task label; diagnostic reference, never a release).
- **Full bank** (the strongest selected full-output attacker): candidates = slate on (l0,l1), slate on (d,c), and the
  `centred` and `prob` units as ignore-offset candidates; validation log loss selects (ties: earlier listed).
- Matched canonical ablations: every surface also reported with its own slate.
- Strata: historical frozen head (primary) and common refitted head HEAD__A (secondary; its log-probability offset is
  a deterministic function of the margin, so full ≡ centred information by construction — reported, not a finding).
- Exactness checks: argmax(l) = 1[d>0]; softmax(l)₁ = σ(d) (max error); attacker on (d,c) ∘ linear map = attacker on
  full; saturation counts (|d| ≥ 36.7 in float64 gives p1 ∈ {0,1}).
- Real-data controls per primary cell (seed 0, fit/val only): shuffled-label null and planted leak on `full`, `centred`.

## Primary family (30 elementary endpoints; frozen-head stratum)

Statistic per endpoint: mean over encoder seeds of per-seed values (recovery: mean over attacker seeds first).
Recovery R = supported-class macro OvR AUC on all assessment rows (benchmark estimator). Pair AUC for (i<j): score
P_j/(P_i+P_j) of the validation-selected attacker on records of groups i and j, positive class j (orientation fixed).

| Block | Endpoints | Count |
|---|---|---|
| Usefulness | U-frozen-{adult,hmda}: Acc(frozen) − Acc(const) ≥ 0.01; U-refit-{adult,hmda}: Acc(U2__A) − Acc(const) ≥ 0.01 | 4 |
| Macro recovery | FC-{adult,hmda}: R(full bank) − R(centred) ≥ 0.02; CH-{adult,hmda}: R(centred) − R(hard) ≥ 0.02 | 4 |
| Pair recovery | FC and CH for Adult sex pair (0,1) and all 10 HMDA race pairs (i<j over 0..4); pairs with an unsupported group are NOT_ESTIMABLE (slot kept) | 22 |
| **Total** | 4 + 4 + 2·(1+10) | **30** |

Inference: paired group bootstrap over assessment record groups (canon_key units; Adult has no household IDs; HMDA no
applicant IDs) — multinomial unit weights, the same draws for every view/seed/surface (B = 1,999, seed 20261021);
SE = sd of replicates. Simultaneous two-sided 95 % Bonferroni intervals est ± z·SE with z = Φ⁻¹(1 − 0.05/60) = 2.9352
(normal approximation, stated). Decision: PASS iff lower > target; NOT_ESTABLISHED otherwise; NOT_ESTIMABLE for
unsupported slots. Intervals are conditional on fitted attackers/heads (no retraining variance); between-seed spread is
reported separately.

## Secondary families (locked before their fits)

- **S3 replication (34):** for each of the 14 admitted purpose/attribute pairs: FC and CH macro contrasts (28) + frozen-
  head usefulness for each of the 6 purposes (6). Same estimator; Bonferroni over 34. Multiclass heads use centred K-vectors.
- **S4 multi-recipient (6):** Adult income + employment, attribute race (disallowed for both per PERMISSION_TABLE).
  Nine slots per seed (income alone, employment alone, pair) × (full, centred, hard). Pair bank includes each single
  recipient's attacker as an ignore-the-other candidate. Endpoints: R(pair) − R(income) and R(pair) − R(employment) ≥
  0.02 for each contract. Bonferroni over 6. Also both tasks' utility.
- **S5 conditional FARE (4, only if budget remains):** see FARE section.

## Stage 5 — conditional FARE (useful-task requirement)

Replay: existing Adult/HMDA FARE, untreated, target LEACE, zero-fairness twin under the same complete contract
(features + own head), with gain over constant and retained share; HMDA s1 one-cell nominee separately.
New cell screen (no assessment data): candidates = all admitted (purpose, attribute) cells except the two primary
cells; eligible iff frozen-head attacker_val accuracy − attacker_fit-majority constant ≥ 0.03 on average over seeds,
> 0 on ≥ 2 seeds, and every sensitive class with ≥ 100 defense_fit rows plus ≥ 2 supported classes. Choose highest mean
validation gain; ties → lexicographic (dataset, purpose, attribute). FARE grid = the frozen six settings + zero-fairness
twin; nominee on validation only: U2 val accuracy ≥ untreated − 0.01 AND (U2 val acc − const)/(untreated − const) ≥ 0.80;
lowest validation macro AUC; ties → lower id; none → NO_FEASIBLE_NOMINEE (no fallback success). S5 endpoints (Bonferroni
4): R(LEACE, rep+head) − R(FARE, rep+head) ≥ 0.02; R(FZ, rep+head) − R(FARE, rep+head) ≥ 0.02; Acc(FARE) − Acc(A) ≥
−0.01; retained gain share ≥ 0.80 (delta-method on ratio, reported as approximation).

## Runtime

≤ 10 elapsed hours, ≤ 12 CPU-h incl. verification, ≤ 2 heavy workers, < 6 GiB; final 90 minutes reserved.
