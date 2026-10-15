# Prospective inference design for a fresh-population confirmation

STATISTICS role draft, 2026-10-15. Prospective only: no historical label changes. Inputs: `SCIENCE_LOCK.json`, `RESEARCH_DECISION.md`, `PRIMARY_ENDPOINTS.csv` (INNER, G = 2,234 duplicate groups).

## 0. Clauses and nulls

The frozen candidate P has two seeds, both scored on the same people. With the six required comparators the claim has **J = 42 clauses**:

- **24 utility clauses.** θ_j is the mean per-person loss difference vs U0 or Ucal. Caps: 0.010 for accuracy and NLL, 0.005 for summed Brier. Accuracy vs U0 and vs Ucal is the same statistic, so 20 clauses are distinct.
- **6 benefit clauses.** β_c = mean_s[AUC_pair(c) − AUC_pair(P)], margin 0.020.
- **12 recipient-excess clauses.** ε_rc = mean_s[AUC_r(P) − AUC_r(c)], cap 0.010.

Each null is stated in the failing direction: H0j: θ_j ≥ c_j; H0β: β_c ≤ 0.020; H0ε: ε_rc ≥ 0.010. These are margin-shifted nulls (FDA 2022 §II.A, p.4).

Inference is conditional on the frozen models, readers and comparators. Duplicate groups are the random units. The two seeds are fixed releases, not a sample of the training recipe.

## 1. Design 1: two-sided Bonferroni simultaneous bounds (historical)

**Rule.** For each endpoint, form θ̂_k ± t_{1−α/(2K),G−1}·SE_k with α = 0.05; at K = 78 the multiplier is 3.418. A clause PASSES iff its whole interval lies on the passing side. The claim requires every clause to pass.

**Controls.** P(all K intervals cover) ≥ 1−α under any dependence, so every statement read off them (passes, violations, shortfalls) is jointly valid at FWER α.

**Cost.** For a single conjunction, Design 1 *is* an IUT with clause tests at one-sided α/(2K): 0.00032 at K = 78, or 0.0006 at K = 42. That bounds false confirmation by α/(2K). For equal per-clause power (0.8–0.9), it needs more people than an IUT at 0.025: 1.95–2.12× at K = 42 and 2.10–2.31× at K = 78. K also grows with comparators, duplicated clauses and non-claim endpoints.

## 2. Design 2: one intersection-union test (IUT)

**Hypotheses.** H0 = ∪_j H0j (some clause fails) against H1 = ∩_j H1j (all clauses hold).

**Clause tests.** One-sided at level α, with q = t_{1−α,G−1}: utility and excess clauses reject iff θ̂_j + q·SE_j ≤ cap; benefit clauses reject iff β̂_c − q·SE_c > 0.020.

**Decision.** CONFIRMED iff every clause test rejects.

**Type I error.** If θ ∈ H0, then θ ∈ H0k for some k, so P(∩_j R_j) ≤ P(R_k) ≤ α. No adjustment and no dependence model are needed. Berger (2014) states this result ("size at most α") and credits the proof to Theorem 1 of Berger (1982). Berger (1982) and Berger & Hsu (1996) were inaccessible; the proof above is self-contained.

**Conditions.**

1. Each clause test is level α for its own null, uniformly over all other parameters.
2. All clause definitions are fixed before labels open.
3. With approximate tests, the IUT's size is at most the worst clause test's true size: errors do not accumulate, but are not removed. FDA (p.4): error rates "are correct if the statistical test is appropriate"; calibrate (§4).

**Conservativeness and power.** Size reaches α only where one clause sits on its boundary and the others lie deep in their alternatives. Global power is at most min_j power_j, equals the product under independence, and is at least 1 − Σβ_j under any dependence (the worst case per EMA 2002 §2.1.1, p.3). Every added clause costs power; duplicated or high-slack clauses cost little.

**Regulatory support.**
- FDA §III.A.1 (p.7): "only one path that leads to a successful outcome… no concern with Type I error rate inflation".
- FDA §III.B (p.8): Type II error rises (0.8 × 0.8 ≈ 64%).
- FDA §III.C.1 (p.9): "more than two co-primary endpoints should be carefully considered because of the loss of power". This design has 42.
- ICH E9 §2.2.5 (p.7) and EMA §2.5.2 (p.5) agree.

### What is and is not controlled

**Controlled.** P(CONFIRMED | some clause false) ≤ α. Clause statements ("clause j holds") are issued only together upon confirmation, so they are jointly controlled at α.

**Not controlled.** The displayed per-clause bounds are marginal: under independence, 42 bounds at 97.5% all cover with probability 0.975^42 ≈ 0.35. Counting passes after a failed conjunction, calling violations or shortfalls, and ranking clauses are therefore descriptive. If needed, add Design-1 simultaneous intervals as a labelled secondary display.

**Orientation, not a re-analysis.** At INNER precision the IUT would not confirm. Recipient excess vs H_DPC_JOINT would pass marginally (upper 0.0081), but that pass has no error-controlled meaning.

### Several claims, candidates or populations

The IUT protects one pre-specified conjunction. Separate claims need FWER control across claims:
- a Bonferroni split with Σα_m ≤ α (FDA App. 1, p.18);
- a fixed sequence ("no further testing once the sequence breaks"; App. 5, p.20);
- serial gatekeeping (App. 7, pp.21–22), e.g. testing a J_TREE claim only after the primary is confirmed.

Nested conjunctions S_1 ⊂ S_2 ⊂ … that share clause tests need no split. The first claim that contains a false clause rejects with probability ≤ α, and its rejection region contains every larger claim's.

"Confirmed for all" candidates or populations is one larger IUT. "Confirmed for at least one" is a union and needs α/m each, or a fixed order.

Picking the best candidate, population, reader, seed or comparator subset on confirmation data lets false confirmation approach mα; the plan "should not be changed after unmasking" (FDA p.6). Dropping clauses defines a narrower claim; that is allowed only before labels open, and must be reported as such.

## 3. Recommended α and decision rule

**Recommendation.** One-sided α = 0.025 per clause, so q = t_{0.975,G−1} ≈ 1.96. Equivalently, the relevant end of each clause's two-sided 95% interval must lie on the passing side. CONFIRMED iff all J clauses pass; no failing-direction claims.

**Justification.**
- FDA §II.B (p.4): "the most widely used values for α are 0.05 for two-sided tests and 0.025 for one-sided tests".
- ICH E9 §5.5 (p.25): one-sided α at "half the conventional type I error used in two-sided tests".

This keeps the historical false-favourable rate for each decision and drops only the multiplicity penalty.

**Not one-sided 0.05.** It doubles the false-confirmation risk. FDA §III.C.1 (p.9) calls raising α for co-primary endpoints "not acceptable". It saves only 18–21% of the sample. A smaller α may be fixed before data; a larger one never to fit a budget.

## 4. AUC clauses

**Statistic.** With frozen reader score s and SEX label Y:

Â = (N1N0)^{-1} Σ_{Y_i=1} Σ_{Y_k=0} [1{s_i>s_k} + ½·1{s_i=s_k}]

This is a two-sample U-statistic. Each clause is T = Σ_m w_m·Â_m over services and seeds, on the same people.

**Primary SE: group influence function** (LeDell et al. 2015, Theorems 4.1 and 5.1, the latter for clustered units; unclustered, it reproduces DeLong-type components).

1. Per person, φ_i = (F̂0(s_i) − Â)/π̂1 if Y_i = 1, else φ_i = (Ŝ1(s_i) − Â)/π̂0. Here F̂0 is the share of Y=0 scores below s_i, Ŝ1 the share of Y=1 scores above s_i, ties count ½, and π̂y = Ny/N.
2. Per clause and group, φ_i^T = Σ_m w_m·φ_i^(m) and Φ_g = Σ_{i∈g} φ_i^T.
3. Then SE = N^{-1}·[G/(G−1)·Σ_g (Φ_g − Φ̄)²]^{1/2}.

Pairing is automatic. Utility clauses use φ_i = d_i − d̄. The bound T̂ ± t_{0.975,G−1}·SE is deterministic.

**Sensitivity analysis.** Run a paired duplicate-group bootstrap with one resample matrix shared by all clauses. Use one-sided BCa or percentile bounds at 2.5%, B ≥ 10,000 (2,048 draws leave about 51 in the tail) and a pre-registered seed. The primary governs if they disagree.

**Calibration.** Bootstrap methods "often require simulations to ensure… adequate Type I error rate control" (FDA App. 6, p.21). LeDell et al. (§7.1) found 92–93% coverage at n = 1,000 and 94–95% at n ≥ 5,000. Simulate boundary rejection rates at the planned G before locking.

**Freeze before confirmation SEX labels open:**
- readers, fit on attacker-train people disjoint from confirmation people;
- winners and orientation, chosen on attacker validation;
- confirmation scores, written and hashed;
- artifacts, the grouping rule, tie and loss conventions, and control gates.

The slate envelope (maximum over readers) is selected on assessment data, so it is descriptive only.

## 5. Why conditional power from development estimates is optimistic

**Selection.** L_TREE won a FIT2 nomination among about 60 candidates. Confirmation is triggered by a favourable INNER readout, on a reused population, under a design chosen after earlier INNER outcomes. Selected estimates "are likely to substantially overestimate" their targets because they were "lucky" (Efron 2011, Abstract, §1). ICH E9 §3.5 (p.17): sample size "will be underestimated" when the assumed difference is too favourable.

One bias runs the other way: the historical codes were selected on INNER, which may understate the benefit against them. Plan over a range of true values.

**Margins.** Clause power ≈ Φ(slack_j/SE_j − 1.96), with SE_j ∝ G^{-1/2}. If any true slack is ≤ 0, global power is at most α at the boundary, and tends to 0 inside the null, for every G. More people cannot rescue a false clause.

**Binding clauses at INNER precision** (p: one-sided INNER p-value for the clause null):

| Clause | Point (SE) | Slack/SE | p |
|---|---|---|---|
| benefit vs H_JOINT_CODE | 0.0177 (0.0071) | −0.32 | 0.63 |
| benefit vs H_SEQ21_CODE | 0.0150 (0.0073) | −0.68 | 0.75 |
| benefit vs H_DPC_JOINT | 0.0138 (0.0075) | −0.82 | 0.79 |
| s1 occupation Brier vs U0 | 0.00422 (0.00324) | +0.24 | 0.41 |
| s0 occupation accuracy | 0.0058 (0.0077) | +0.54 | 0.29 |

The other occupation clauses sit at +0.39 to +1.30 SE; all remaining clauses are at ≥ 2.16 SE. INNER is not evidence that the clauses in the table hold.

- **Full competitive conjunction.** At face value, the historical-code benefits make it futile at any G.
- **Seed-1 Brier clause.** For 90% clause power it alone needs about 184× the INNER groups, ≈4.1×10^5 (≈8.0×10^5 under Bonferroni K = 42; ≈3.4×10^5 at one-sided 0.05). It rests on 0.24 SE of slack, so it is unstable.

**Recommendation.** Simulate global power from the joint IF or bootstrap distribution of all clause statistics. Use a pre-declared grid of true values that includes shrunken and boundary values. Never tune α to G.

## Sources read

- FDA (2022), *Multiple Endpoints in Clinical Trials*: §§II.A–C, III.A.1–C.1 (pp.4–9); App. 1, 5–7 (pp.18, 20–22).
- ICH E9 (1998): §§2.2.5 (p.7), 3.5 (p.17), 5.5 (p.25).
- EMEA CPMP/EWP/908/99 (2002): §§1 (p.1), 2.1.1 (p.3), 2.5.2 (p.5).
- R. L. Berger, errorstatistics.com guest post (31 Jul 2014) and reply (1 Aug 2014).
- LeDell, Petersen & van der Laan (2015), *Electron. J. Statist.* 9:1583–1607: Theorems 4.1 and 5.1, §7.1.
- Efron (2011), *JASA* (author preprint): Abstract, §1.

**Named, not read:** Berger (1982, *Technometrics*); Berger & Hsu (1996, *Statist. Sci.*); DeLong et al. (1988, *Biometrics*).
