# Designer B, closing report for family B (saved verbatim with attribution)

Author: Designer B (family B) of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 16:30-16:31Z on
2026-10-20, after the coordinator's 16:28Z message with the NRM1_MAIN results (MAIN_V2) and the decisive-rule outcomes. Designer B read
no MODEL_SELECT output itself. Saved verbatim; nothing below the line was edited.

---

DESIGNER B: closing report for family B (FSC). Written 16:30Z, to be saved verbatim.

Family B closes. It is a finite negative result, reached by the stop rule I registered in advance (F1 holds). I did not fit or read any MODEL_SELECT output myself; the outcome numbers below are from your message, and the diagnostic numbers are from my own fit-time records.

**Predictions against outcomes**

| Prediction | Outcome |
|---|---|
| P1b: FB2_BASE passes 24 clauses + 4 guards in both versions | TRUE |
| P2b: SEX lowers pair AUC against the base by about 0.006 (S3) / 0.015 (S5) | Observed +0.0069 (se 0.0077) and +0.0120 (se 0.0073). Close in size, but not resolved. |
| P3b: SEX beats the MINF control at both strengths (p ≈ 0.45) | FALSE. Margin over MINF is +0.0027 / +0.0019 with se 0.0089, so SEX is not separated from task spending. The CTL_REM half is still pending. |
| P4b: all noisy configurations stay task-feasible at S5 | FALSE. MINF_S5 version 1 exceeds the occupation NLL cap (1.005 × cap). SEX_S5 is feasible. |
| P5b: the 0.020 target is not met | TRUE. SEX_S5 (0.6496) is still more recoverable than NEURAL_RAW (0.6361) and H_LJ (0.6327). |
| P6b (MEAN), P7b (FB2P) | Not testable. Neither was fitted under the rules (FB2P not run; MEAN cannot qualify). |
| Card v1, P1 | FALSE. The joint-trained twin failed version 1. |
| Card v1, F3 | Held: training through the channel added nothing. |

**Why the channel did not separate from SEX-blind noise**

1. **Most of the recovery drop came from spending, not from targeting.**
   - At S5: SEX lowered pair AUC by 0.0109 (0.6605 → 0.6496), and SEX-blind MINF by 0.0100 (→ 0.6505).
   - At S3: 0.0074 against 0.0046.
   - Noise of this size lowers recovery whichever way it is aimed. The extra effect of aiming it, 0.002-0.003, is a third of a standard error.
2. **The targeting worked for the linear model it was designed against, but not for the readers.**
   - In my fit-time records, SEX cut the in-sample linear SEX discriminability by 18-20% at S3 and 36-40% at S5. MINF cut it by only 6-7% and 11-14%.
   - The readers' AUC barely reflected this difference. Their SEX information therefore sits mostly outside the single global discriminant direction: in decisions, local patterns and nonlinear structure.
   - A rank-1 channel built from one global geometry for all people is the wrong target for a bank of flexible readers.
3. **The noise budget caps the effect.**
   - The base's second-order cost at S5 is about 0.022 nats. In my records, SEX_S5 spends 0.0144 nats on occupation and 0.0087 on income; MINF_S5 spends 0.024 on occupation.
   - Noise that small can only blur the vector by about its own amplitude. The bulk of the SEX signal is task-relevant variation that is far larger.
4. **The starting point.**
   - The strongest task base is also the most recoverable release: 0.6605, against 0.6361 and 0.6327 for the target references.
   - Even a perfectly targeted 0.012 reduction leaves it above both, so this line could not reach G5.

**Recommendations**

Would not recommend:
- any further strength, coupling, penalty, R or cap tuning of this channel;
- reopening FB2P or MEAN, or channel-trained learners (F3);
- raising the strength, since MINF_S5 already exceeds the occupation cap.

Would recommend:
- Report family B as **EXPERIMENTAL_NO_ADVANTAGE**. Randomized logit noise lowers recovery roughly in proportion to what it spends, but not beyond matched SEX-blind noise. This agrees with the earlier frozen-bank and LP-mixture studies ("no randomness advantage").
- Before designing any new output channel, run a label-light diagnostic in a future campaign: on fixed readers, how much SEX information sits in decisions, in each recipient's confidence, and in the linked residual.
- The 0.024-0.028 AUC gap between the strongest base and NEURAL_RAW / H_LJ points at the predictor's own information content, not at post-processing. Targeted noise cannot close that gap within the utility limits.

**Resources and process**
- CPU: 0.138 CPU-h (496 s). This is 400 s charged out of queue (upper estimates) plus 96 s in four queue units (FB_FIT and FB2_FIT, each version).
- 28 configurations used: FB 18 and FB2 10. FB2P and MEAN were not spent.
- 12 label reads, all through the access layer with committed cards.
- Two read-only research agents ran early (finished by about 14:45Z); none were used after the rule.
- Commits on research/pcrl-new-release-mechanisms-v1:
  - code: ef0bc1ddf, 39414150e, e4da5a8fc, d14ce4526, 0a8b2de9d, 2fa4e64ac;
  - cards: 48fbac50f (v1), b163f3233 (v2), c4ded0dd4 (v3).
- Disclosed wording slip: card v2 says HC_STRONG's STEP1.json holds "CV metrics only". It holds CV fold/grid metadata and no per-person arrays; the stated limitation is unchanged.
