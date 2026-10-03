# Matched useful-head comparison: results (Stage C)

**Status: development evidence on reused assessment rows; not fresh confirmation.**

| Item | Value |
|---|---|
| Lock | `a0de449` (protocol, code, inputs, units and families), pushed before any fit |
| Cell | Adult `income_prediction` / `sex`, Round-4 encoders, seeds 0, 1, 2 (all reported) |
| Assessment | 5,243 records |
| Verification | Independent replay: see `INDEPENDENT_VERIFICATION.json` and `VALIDATION.md` |

## 1. What was compared

Each arm releases the outputs of an **actually deployable task head**: a logistic-regression head from one common family, fitted on that arm's own features only. The heads, LEACE maps and FARE trees are reused unchanged from the output-aware study. **84 new attacker units (3,924 model fits) and 4 real-data controls** were run.

| Arm | Features released to the head |
|---|---|
| A | Untreated PCRL `rep_p0` |
| B | Official target LEACE |
| F | Validation-selected official FARE nominee (10 / 50 / 10 cells on seeds 0 / 1 / 2) |
| F0 | The matched zero-fairness tree |

**Construction check.** Every head emits log-probabilities, so its offset is an exact function of its margin (to within 1e-12). The full logits, the (margin, offset) pair, the centred margin and the probabilities therefore carry **the same information**. Differences among them below are attacker-fitting variation. On F/F0, whose outputs take only 10–50 values, the surfaces are recodings that the attackers fit identically.

## 2. Actual head utility (`ACTUAL_HEAD_UTILITY.csv`)

Values are means over the three seeds, with per-seed values in the CSV. The constant (majority class on attacker_fit) has accuracy 0.7492. Recall is reported for the minority class, >50K.

| Arm | Accuracy | Balanced accuracy | Minority recall | Log loss | Gain over constant | Historical probe U2 accuracy (separate) |
|---|---|---|---|---|---|---|
| A | 0.8318 | 0.723 | 0.506 | 0.378 | 0.0826 | 0.8306 |
| B | 0.8312 | 0.722 | 0.503 | 0.382 | 0.0820 | 0.8304 |
| F | 0.8219 | 0.721 | 0.519 | 0.390 | 0.0727 | 0.8237 |
| F0 | 0.8278 | 0.728 | 0.527 | 0.374 | 0.0786 | 0.8307 |

All four heads are genuinely useful: a gain of 7–8 points, about half of high earners recalled, and better log loss than the constant (0.563).

## 3. Primary family (19 endpoints; Bonferroni z = 3.0078)

| Endpoint | Point | Simultaneous interval | Target | Decision | Seeds 0 / 1 / 2 |
|---|---|---|---|---|---|
| Acc(B) − Acc(A) | −0.0006 | [−0.0016, 0.0003] | > −0.01 | **PASS** | −0.002 / 0.000 / −0.000 |
| Gain B | 0.0820 | [0.0672, 0.0968] | > 0.03 | **PASS** | |
| Retention B | 0.0159 | [0.0129, 0.0189] | > 0 | **PASS** | |
| Acc(F) − Acc(A) | −0.0099 | [−0.0152, −0.0046] | > −0.01 | **NOT_ESTABLISHED** | −0.002 / −0.007 / **−0.020** |
| Gain F | 0.0727 | [0.0571, 0.0884] | > 0.03 | **PASS** | |
| Retention F | 0.0066 | [0.0006, 0.0126] | > 0 | **PASS** (near the bound; seed 2 −0.0015) | 0.008 / 0.014 / −0.001 |
| Acc(F0) − Acc(A) | −0.0040 | [−0.0086, 0.0005] | > −0.01 | **PASS** | 0.001 / −0.012 / −0.001 |
| Gain F0 | 0.0786 | [0.0634, 0.0939] | > 0.03 | **PASS** | |
| Retention F0 | 0.0125 | [0.0073, 0.0178] | > 0 | **PASS** | |
| Centred: R(A) − R(B) | 0.0148 | [0.0105, 0.0191] | > 0.02 | **NOT_ESTABLISHED** (excludes zero; below target) | 0.049 / −0.000 / −0.004 |
| Centred: R(A) − R(F) | 0.2345 | [0.2169, 0.2520] | > 0.02 | **PASS** | 0.174 / 0.236 / 0.293 |
| Centred: R(A) − R(F0) | 0.1252 | [0.1115, 0.1390] | > 0.02 | **PASS** | 0.061 / 0.130 / 0.184 |
| Centred: R(B) − R(F) | 0.2197 | [0.2011, 0.2382] | > 0.02 | **PASS** | 0.125 / 0.236 / 0.297 |
| Centred: R(F0) − R(F) | 0.1092 | [0.0923, 0.1261] | > 0.02 | **PASS** | 0.113 / 0.106 / 0.109 |
| Hard: R(A) − R(B) | −0.0003 | [−0.0012, 0.0005] | > 0.02 | NOT_ESTABLISHED | |
| Hard: R(A) − R(F) | 0.0076 | [−0.0040, 0.0192] | > 0.02 | NOT_ESTABLISHED | −0.002 / 0.002 / 0.024 |
| Hard: R(A) − R(F0) | −0.0020 | [−0.0063, 0.0022] | > 0.02 | NOT_ESTABLISHED | |
| Hard: R(B) − R(F) | 0.0080 | [−0.0036, 0.0195] | > 0.02 | NOT_ESTABLISHED | |
| Hard: R(F0) − R(F) | 0.0097 | [−0.0023, 0.0216] | > 0.02 | NOT_ESTABLISHED | |

**Recovery levels** (output-only, mean over seeds; 90 % marginal SE in `inference.json`):

| Arm | Centred margin | Full logits | Probabilities | Hard decision |
|---|---|---|---|---|
| A | 0.774 | 0.774 | 0.766 | 0.544 |
| B | 0.759 | 0.755 | 0.757 | 0.544 |
| F | 0.539 | 0.548 | 0.548 | 0.536 |
| F0 | 0.648 | 0.648 | 0.648 | 0.546 |

## 4. What this means

**1. LEACE keeps the useful head but barely changes what its output reveals.**
- The head on LEACE features is as accurate as the untreated head (−0.0006, PASS).
- Its output alone still recovers sex at 0.759, against 0.774 untreated.
- The difference, 0.015 [0.011, 0.019], is **a smaller difference, below the registered 0.02 target**. Its interval excludes zero, so it is not "no difference".
- It comes almost entirely from seed 0 (0.049); seeds 1 and 2 show none.

**2. FARE's head output reveals far less, and the fairness term, not only compression, accounts for about half of that.**
- FARE's head output recovers 0.539, against 0.648 for the zero-fairness tree of the same size and 0.774 untreated.
- Of the 0.234 reduction, compression (A → F0) gives 0.125. The fairness weight (F0 → F) gives another **0.109 [0.092, 0.126], PASS**, and does so on every seed.
- This differs from the earlier useful-task cell (Adult employment/age_group). There the zero-fairness tree matched FARE to within +0.007 [0.002, 0.013], below target.

**3. But FARE's head costs about one accuracy point, and non-inferiority within one point is not established.**
- Acc(F) − Acc(A) = −0.0099, lower bound −0.0152. Seed 2 loses 2.0 points.
- Its gain (0.073) and the 80 % retention criterion pass, the latter only just (lower bound 0.0006; seed 2 alone fails it).

**4. When only the decision is released, every arm leaks about the same small amount.**
- Hard-decision recovery is 0.536–0.546 for all four arms; no hard-decision contrast is established.
- The decision carries the same accuracy as the scores (it is the same decision), so for decision-only recipients the feature defense makes no measurable privacy difference.
- **Post hoc, descriptive:** the untreated head's decisions (0.544) leak about as much as FARE's *scores* (0.539; difference +0.005, 90 % interval [−0.003, 0.013]). They are about one point more accurate (+0.0099 [0.007, 0.013]).

**5. The untreated score hides sex from a linear reader but not from a nonlinear one** (post hoc, descriptive).
- A linear attacker on the untreated head's margin recovers only 0.515 (on B's margin, 0.510). Nonlinear attackers on the same single number recover 0.773 (and 0.759).
- This is consistent with PCRL's linear constraint on the representation. It is the same pattern as the benchmark's representation result: the linear check passes, and recovery happens outside its scope.

**6. Secondary family** (33 endpoints; z = 3.1718; `SECONDARY_COMPLETE_RELEASE_ENDPOINTS.csv`):
- **Features plus own output:** the features dominate. Recovery is A 0.816, B 0.817, F 0.550, F0 0.646, and every A/B-vs-F/F0 contrast passes.
- **A vs B:** never established; the features-only gap is 0.005 [0.002, 0.008], excluding zero but below target.
- **Offset:** 0 for every arm, as it must be for log-probability heads (NOT_ESTABLISHED). For F and F0 it is exactly 0 by selection, because the banks resolve to recodings with identical predictions.
- **Bypass:** adding the historical clean PCRL logits to F or F0 features raises recovery by 0.238 / 0.143 (PASS), back to 0.788 / 0.789. For A and B it adds nothing, because their features already reveal 0.81–0.82.

## 5. Registered predictions vs outcome (pre-fit predictions, PROTOCOL §6)

| Prediction | Outcome | Hit? |
|---|---|---|
| Centred A−B PASS | NOT_ESTABLISHED (0.015) | **miss** |
| Centred A−F, A−F0 PASS | PASS, PASS | hit |
| Centred B−F NOT_ESTABLISHED | PASS (0.220) | **miss** |
| Centred F0−F NOT_ESTABLISHED | PASS (0.109) | **miss** |
| Hard A−B, A−F, A−F0 PASS | all NOT_ESTABLISHED | **miss** (3) |
| Hard B−F, F0−F NOT_ESTABLISHED | NOT_ESTABLISHED | hit |
| Offset NOT_ESTABLISHED for every arm | yes | hit |
| Bypass PASS for B, F, F0; not A | F, F0 PASS; A and **B** not | partial (B miss) |
| Features plus output tracks features only | yes | hit |

**4 of 10 primary recovery predictions were correct.** The misses run in both directions:
- LEACE removed less from the head output than predicted.
- FARE's fairness term removed more than predicted.
- Hard decisions are near-uninformative for every arm, so no hard contrast could reach 0.02.

## 6. Controls and limits

**Controls.**
- Shuffled-label nulls: 0.479–0.511.
- Planted leaks: 0.899–0.927, all detected.
- They cover output-only centred (A), output-only prob (B), output-only hard (F) and features plus hard (F).

**Limits.**
- One cell, three encoder seeds, reused rows.
- Intervals are conditional on the fitted heads and attackers, with no retraining variance.
- FARE's tree on seed 1 has 50 cells; on seeds 0 and 2 it has 10.
- FARE is applied to stored representations.
- No repeated-query or multi-recipient attack in this cell.
- Post-hoc diagnostics (§4, items 4–5) are labelled as such and are not in any family.
