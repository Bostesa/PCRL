# Advisor brief: matched attribute-removal benchmark

2026-10-03. This is development evidence on data that has already been used. The full account is in
RESEARCH_DECISION.md.

## What was run

| Item | Detail |
|---|---|
| Encoders | 6 frozen PCRL Round-4 encoders: Adult and HMDA × 3 seeds. None was retrained. |
| Defenses | **Official LEACE** (concept-erasure 0.2.4, default settings), fitted on the encoders' training rows:<br>(B) erasing the target attribute;<br>(C) erasing all disallowed attributes of the purpose.<br>Also Gaussian noise at 6 levels. |
| Cells | Primary: Adult income/sex and HMDA underwriting/race. Registered extension: 12 more pairs. |
| Units | 435 of 882 registered. All Tier-1 units ran; the remaining noise units were cut by the pre-registered CPU budget rule. |
| Fits | 60 LEACE maps; 44,824 attacker and probe fits |
| Cost | 5.0 CPU-hours on the laptop; no cloud |
| Protocol | Locked and pushed before any fit (commit 3274fe1) |
| Verification | Independent replay; see VALIDATION.md |

## Findings

1. **LEACE does exactly what it promises, and only that.**
   - It passes its own check on every map.
   - The removal of the linear signal carries over to unseen rows: canonical correlation ρ₁² goes from 0.025 (Adult) and 0.21 (HMDA) to
     0.0006 and 0.0001.
   - Erasing only the target costs essentially no task accuracy (within 0.13 pp).
2. **It does not stop nonlinear recipients.**

   | Cell | Untreated recovery (AUC) | Under LEACE |
   |---|---|---|
   | Adult sex | 0.815 | 0.81–0.82 |
   | HMDA race | 0.870 | 0.86 |

   The same holds on all 14 pairs.
3. **Task outputs keep the leak alive.** Noise strong enough to stop representation recovery (AUC 0.51–0.53)
   leaves recovery from the released outputs at 0.76–0.78. The outputs carry 0.18–0.24 AUC more information than the
   true label would.
4. **Noise is expensive.**
   - On Adult, a refitted probe keeps about 40 % of its task lift.
   - On HMDA, all of the lift is lost: the probe equals a constant predictor.
   - The original task head is useless at σ\*.
5. **PCRL's own check.**
   - Linear leakage on unseen rows was not shown to exceed the threshold anywhere (no C2). The Adult income/sex check
     generalises; HMDA underwriting/race is unresolved.
   - PCRL's stored check **fails on 12 of 42 untreated pair × seed combinations**. That includes HMDA
     underwriting/race on 2 of 3 seeds, and Adult income/race on all 3. Every value reproduces exactly.
   - The leakage varies a lot by encoder seed: Adult nonlinear recovery is 0.67 / 0.91 / 0.87. The pilot's encoder
     was the least leaky of the three.

## For the paper

The pilot's pattern replicates across 6 encoders, 2 datasets and 14 pairs. It now comes with a matched,
off-the-shelf eraser that behaves the same way.

The defensible framing is: **linear certificates, whether PCRL's or LEACE's, hold where they claim to and transfer to
unseen rows, but they do not cover nonlinear recipients or released task outputs.**

Avoid "the certificate is broken".

## Decisions for the next meeting

1. Is the next step an eraser that acts on outputs or nonlinear statistics, run under this same matched protocol?
   More pairs or seeds of the same kind would not change the conclusion.
2. Should the 447 unrun noise units be completed? That is about 3 CPU-h. It is optional, because the 4 completed
   noise pairs already agree.
3. When should a fresh, unopened cohort be used for confirmation, and under which frozen protocol?
