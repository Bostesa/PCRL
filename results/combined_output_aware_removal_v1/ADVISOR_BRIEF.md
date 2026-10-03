# Advisor brief: corrected benchmark, and does a broader defense protect the whole release?

2026-10-03. This is development evidence on data that has already been used. The full account is in
RESEARCH_DECISION.md.

## What was done

1. **Corrected the earlier benchmark, without rewriting it.**
   - Two arithmetic errors in exploratory tables were fixed: an unpaired-seed contrast and a numerically unstable
     correlation.
   - Six wording problems were fixed.
   - The 24 headline results reproduce exactly from the saved predictions.
2. **Exposure check.** About 0.1–0.3 % of assessment rows duplicate training records. Removing them changes none of
   the 24 headline decisions.
3. **A new, pre-registered comparison** (protocol pushed before any fit):
   - the official published **FARE** defense, a fair decision-tree representation that comes with a certificate;
   - target-only LEACE, noise and controls;
   - on Adult (sex) and HMDA (race), across 3 encoder seeds.

   We also tested what changes when the recipient gets only the prediction's decision instead of its full output.

**Cost.** About 0.9 CPU-hours for the new fits and inference, and about 1.8 CPU-hours including the corrections,
custody and exposure re-scoring. The ceiling was 12 hours; everything ran on the laptop. All 12
primary questions were decided. The independent verification is reported in VALIDATION.md.

## Findings in plain terms

- **The released output's detail is the main leak.**
  - Full output: an attacker recovers the protected attribute at about AUC 0.77.
  - Probabilities only: 0.70. The raw logits carry an extra per-person number that no decision needs.
  - The decision alone: about 0.51, near chance.

  The decision has, by construction, exactly the same task accuracy as the full output.
- **The clean output bypasses every feature defense.** Protect the features with FARE or with noise, but keep releasing
  the original output, and recovery returns to about 0.77–0.79.
- **FARE versus LEACE at the same task cap:**
  - FARE's coarse cells bring feature recovery to about 0.55 (Adult) and 0.50 (HMDA). LEACE stays at 0.81–0.86.
  - Under the same output contract (features plus a head built only on those features), FARE's release is also far
    lower.
  - **Adult:** FARE loses about 0.7 points of accuracy. That is not provably within our 1-point margin, and one seed
    drives it. *Not competitive by our rule.*
  - **HMDA:** FARE passes every test, but the task barely beats a constant guess (90.3 % vs 88.6 %). On one seed, FARE
    picked a single-cell "release nothing" encoder. On the other two it kept the full accuracy at chance-level
    recovery.

  So this is genuinely favourable but on an easy-to-protect task. It is *not evidence of a generally better method.*
- **Compression is part of the effect.** The same tree without the fairness term already lowers recovery to about
  0.61–0.65. The fairness term does the rest.
- **FARE's own certificate was not informative here.** With about 1,500 certification rows it was unavailable or
  vacuous in almost every case. That reflects our sample size; it does not mean FARE is unsafe. The certificate is
  about demographic parity, not attacker AUC, and it does not cover the output bypass.

## Limitations

- Already-used development data.
- Two cells, three seeds.
- FARE is applied to stored representations, not raw inputs.
- HMDA underwriting has little task signal to protect.
- Some feature-identical rows cross the train/test boundary; they are reported.
- Repeated-query attacks are not covered.

## Decision for the next meeting

**Should the combined paper treat the release contract as a primary variable?** That is: full output vs probability vs
decision, and a head built only from the protected features. The evidence points to the output contract as a larger
lever than the feature defense.

**If FARE is pursued, where?** Run the same locked protocol on a cell with real task lift and a larger certification
sample, so that its certificate can be meaningful. No new algorithm is indicated.
