# Advisor brief: first stored-model pilot (CELL-A)

2026-10-02. This is development evidence on data that has already been used. The full account is in
RESEARCH_DECISION.md.

## What was run

| Item | Detail |
|---|---|
| Encoder | One frozen encoder: PCRL Round-4 Adult seed 0, the one the AAAI audit attacked. No defense was retrained. |
| Rows | PCRL Adult test split (15,060 rows). The encoder never trained on it, but it was used before, so this is not fresh confirmation. |
| Units | 26: 8 untreated disallowed purpose/attribute pairs, plus income/sex with Gaussian noise at 6 levels × 3 release seeds |
| Fits | 2,902 attacker and probe fits; 0.19 CPU-hours on the laptop; no cloud |
| Protocol | Locked and pushed before any fit (Amendment 1 records every change from the preparation plan) |
| Verification | An independent replay from the saved predictions found **0 FAIL across 19,466 checks** |

## Three separate findings

1. **Implementation compliance.**
   - The paper's own linear check reproduces on all 8 pairs.
   - One pair (income/race) already fails it.
   - For 3 education pairs, the stored "pass" came from a numerically broken float32 calculation. Recomputed
     properly, they still pass.
2. **Generalisation of the paper's own quantity.** The same R², measured on held-out rows, stays below the 0.05
   threshold for 7 of 8 pairs, with simultaneous bounds across 16 endpoints. The eighth is unresolved. **No case
   where the check passes but fails to generalise.**
3. **Recovery outside the guarantee's scope.**
   - A nonlinear attacker recovers every attribute well above AUC 0.55 (0.68–0.82).
   - The task outputs leak more than the true label alone would for 7 of 8 pairs (+0.08 to +0.23 AUC).
   - For income/race, the class-contrast diagnostic finds 3× the linear leakage the pooled check reports.

## Strongest favourable and adverse comparisons

**Favourable.** Where the linear certificate passes, it holds on unseen rows. The failures are not overfitting of
the check; they are attackers and surfaces the check never claimed to cover. The paper should say exactly this,
not "the certificate is broken".

**Adverse.** Noise on the representation pushes representation-only recovery below 0.55 from σ = 2. But:
- once task outputs are also released, recovery stays at about 0.67 at every noise level;
- the original task head loses all of its useful accuracy by σ = 0.5;
- a refitted probe keeps only about 40 % of the clean gain at σ = 2.

On this cell, noise protects one surface and costs most of the utility.

## Is a larger stored-artifact benchmark justified?

**Yes, within limits.**

The pipeline is validated and the decomposition separates metric, attacker and surface effects cleanly. It can be
extended with no defense training to the frozen interfaces that exist:
- the remaining Round-4 seeds;
- HMDA Round-4;
- the NeurIPS Round-5/7 headline checkpoints;
- noise arms on the other pairs.

**Not without your decision:** a matched projection/LEACE comparison. The original projection matrices were never
saved, so it would need a defense refit. The multi-purpose coalition scenario is also not yet run.

## Decisions for the next meeting

1. Extend the frozen-artifact benchmark to the interfaces listed above (estimated a few CPU-hours, laptop only)?
2. Authorize projection/LEACE refits on attacker-fit rows, so that the most common defense family has a matched
   comparison?
3. Add the multi-purpose coalition scenario now, or after the single-recipient benchmark is complete?
