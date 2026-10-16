# Designer 2 cross-critique of designer 1 (MMP)

I read designer 1's proposal only after both hashes were logged; my PROPOSAL.md is unchanged.
- **Check:** unit `D2_XC_MMPGAUSS_V1` (script `xc_mmp_gaussian.py`). It places MMP inside my calibrated linear-Gaussian coalition model.
- **What the model fixes:**
  - T_TREE-level signal;
  - R² = 0.82;
  - α = 0.6;
  - d′²_X ∈ {0.40, 0.49, 0.64}.
- **What it scans:** within-SEX gaps γ (female minus male; income −0.05/−0.10/−0.20, and two occupation directions), 81 settings. The gaps are assumed, not measured.

## 1. Strongest objection
MMP is a parameter-free point. Its effect and its price are both set by the same unmeasured quantity, the within-x SEX gaps g.

**What it removes, and what it leaves.**
- It deletes Cov(Δ, π) = E[(π − π̄)² g].
- It leaves Cov(m, π) untouched: the within-SEX task effects of SEX-correlated fields such as hours, class of worker and schooling. The coalition keeps reading that part (designer 1's failure mode 1, and law R).
- It therefore reaches the material target only if most of the occupation SEX signal *is* the inferred-SEX term, which is exactly the unproven premise F1.

**What the model shows.**
- MMP's pair falls to ≤ 0.606 in only 24 of 81 settings. That requires:
  - a gap along occupation's SEX-specific direction that accounts for 60–100% of that direction's SEX covariance when d′²_X ≥ 0.49, or about 50% when d′²_X = 0.64.
- With smaller gaps, the pair stays at 0.583–0.659.

**Why it cannot trade along the frontier.** The Brier price, the NLL price and the argmax flips all scale with the same gap. Where MMP is strong it pays most and flips most decisions. In law F it fails the accuracy cap even in its ideal case.

**Subgroup exposure.** The price falls on both SEX groups, and men's seed-1 T_TREE headroom over U0 is only −0.0002 occupation Brier (`SUBGROUP_TABLE.csv`).

**What I cannot hold against it.** In the same model MMP never worsens the pair (0 of 81 settings), and the pair's excess over occupation alone stays ≤ 2.9× the income-only signal. The shared constant π̄ keeps the two outputs aligned, so the amplification failure seen in LOCAL does not apply to MMP.

## 2. Distinct or overlapping?
**They overlap substantially.** Both use SEX only on MODEL_FIT to build a propensity-aligned direction, and both act on it deterministically, with no critic.
- MMP's release, η − (π − π̄)g, is itself a propensity-driven shift. Its coefficient is fixed at the task gap, and it is applied to both outputs.
- RCCS is a propensity-driven shift of the occupation output only. Its coefficients are optimised for the pair-conditional criterion, and income is held fixed.

**They differ in four ways.**
1. **Where the coefficient comes from.** MMP identifies its coefficient from task data, so its price is exact and it cannot overshoot: it removes only the inferred-SEX term. RCCS chooses its coefficients with a SEX criterion and can inject counter-signal.
2. **Income.** MMP changes income, cheaply. RCCS never does.
3. **Target.** RCCS targets the cross-output contrast; MMP does not.
4. **Pooling.** MMP can pool pure-proxy directions, which is a genuine Bayes-level effect. RCCS adds proxy information.

At equal occupation price in the model, RCCS gives the lower pair in 54 of 81 settings and MMP in 27. The difference ranges from −0.032 to +0.026; MMP wins where the income gap is large. The two methods are **complementary, not duplicates**.

## 3. What this changes in my own assessment
These points are stated here only; PROPOSAL.md is not edited.
- **(a) RCCS is more dependent on reader limits than I said.** It injects π̂, which is built partly from pure-proxy fields. Wherever T_TREE pooled such directions, it can therefore only *raise* population Bayes recovery. MMP has the cleaner "removal" interpretation.
- **(b) Freezing income leaves cheap protection unused.** MMP's income price is Var(π)γ₁², which is 0.0002–0.003 summed Brier in the scan. MMP on income followed by RCCS on occupation is at or below both single arms in **81 of 81** settings. My primary arm is probably not the best version of my own idea, and a pilot should include this combination as an arm.
- **(c) My amplification argument is narrower than my proposal implies.** It condemns per-recipient *optimised* protection (LOCAL, H_LJ). It does not condemn a shared-driver identity such as MMP.
- **(d) Neither premise is supported by any published number.** MMP's Stage-0 gate on MODEL_SELECT is cheaper than mine and needs no optimisation, so it should run first.
- **Net:** my decision remains `PROPOSAL`. My confidence in RCCS *as specified* is lower than my confidence in the MMP-income + RCCS-occupation combination.
