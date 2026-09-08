# Source exposure control and the three-probability banks

**Teacher-only source banks outperform the corresponding K representations on all three mean source losses under both scoring weights, while losing residential transfer.** This holds for both fixed native heads and independently fitted downstream probes. It is a tradeoff between released interfaces and training objectives, not evidence that the K representation creates information or restores all original source utility.

Each bank starts with exactly the same K mapper and genuine original source heads as its teacher-matched K-C/K-D conditions. It uses the same source examples, eligibility masks, fixed three-task BCE mean, Adam defaults, 60 base plus 80 continuation mapper epochs and batch schedules. Its sole objective is source loss: preservation, reconstruction and protection coefficients are zero. No observers or catch-up are fitted for banks. Skipping observer work does not advance mapper schedules. K main conditions retain beta=1, and K-D additionally applies the real-attribute protection gradient. Thus K-C versus bank changes the preservation objective and published interface while controlling source exposure and usable architecture; K-D additionally differs in protection.

All sampled representation-fitting rows have known labels for the three source tasks in this cohort. Actual exposures per task are:

| Seed | Distinct fitting examples | Mapper steps, 60+80 epochs | Label presentations per source task |
| --- | ---: | ---: | ---: |
| 0 | 10,513 | 5,880 | 1,471,820 |
| 1 | 10,428 | 5,740 | 1,459,920 |
| 2 | 10,551 | 5,880 | 1,477,140 |

These counts match the K conditions exactly. Repeated presentations are not additional independent labels. Native heads are the fixed final training heads: no downstream selection, calibration or refitting is applied to them. Their source-validation and **DEVELOPMENT EVALUATION** rows and masks match across conditions. Separately, each published release gets the same 2,048-label logistic/MLP downstream-head recipe, with validation selection. Native and probe results remain separate; neither is selected after comparing their evaluation performance. Comparing a source-trained student with a small direct-teacher probe alone would confound source exposure, optimization and function class.

The bank publishes only positive probabilities in order **income >$50k, civilian employed and at work, public coverage**. Its internal 16-coordinate mapper is never treated as the bank's released interface. Both banks and main K conditions have 2,179 usable model parameters; the stored unused decoder and zero raw columns are detailed in [INPUT_BOUNDARY.md](INPUT_BOUNDARY.md).

## Native source losses

Cells are three-seed development mean log loss, **unweighted / PWGTP**. Lower is better. The same fixed native predictions receive both scores. Every seed, source-validation result, sample SD and paired contrast is preserved in [NATIVE_SOURCE.csv](NATIVE_SOURCE.csv), [NATIVE_AGGREGATE.csv](NATIVE_AGGREGATE.csv), and [NATIVE_PAIRED.csv](NATIVE_PAIRED.csv).

| Release | Income | Civilian at work | Public coverage |
| --- | --- | --- | --- |
| E K C | .329459 / .339431 | .368793 / .352854 | .512070 / .517717 |
| E K D | .329595 / .339604 | .368558 / .352580 | .512710 / .518307 |
| E bank | .322172 / .331232 | .358638 / .341599 | .505968 / .512575 |
| S K C | .314153 / .326857 | .336640 / .322560 | .511886 / .519031 |
| S K D | .314128 / .327230 | .336612 / .322588 | .512207 / .519245 |
| S bank | .308485 / .320962 | .327182 / .309233 | .508223 / .515900 |

Bank superiority is a mean result, not a claim about every seed/task. Income improves in every seed for all four K comparisons under both weights. E's native civilian-at-work comparison favors K in seed2, while seeds0/1 favor the bank. Unweighted public coverage favors E-K-C in seed0 and S-K-C/K-D in seed2; these exceptions do not reverse the means. F has access to raw PCA32 and substantially better mean source performance than these K/bank conditions, so the bank result is not superiority over the full-input controls.

## Transfer cost and recovery

The independently selected 2,048-label probes also give the banks lower mean losses on all three source tasks than their K-C/K-D counterparts under both weightings. However, their residential losses are markedly higher:

| Interface | E residence U / PWGTP | S residence U / PWGTP |
| --- | --- | --- |
| K C | .505432 / .482860 | .510575 / .485461 |
| K D | .505551 / .483033 | .510701 / .485466 |
| Bank | .521901 / .497591 | .526645 / .500173 |

K has lower residential loss than its bank in **every seed under both weights**. Mean advantages across the four K conditions are .015944–.016469 unweighted and .014558–.014731 weighted. This is useful reserved-task flexibility beyond the three source-probability interface, but it comes with higher measured attribute recovery: the banks have lower mean independent SEX and RAC1P gains than every teacher-matched K final under both weights. Main catch-up-inclusive audits add observer exposure unavailable to banks and are reported separately. [TABLE.md](TABLE.md) and [PAIRED.csv](PAIRED.csv) retain all five tasks, audit scopes and seed signs.

All K finals and banks still fail the combined original PCA32+.01 three-source probe allowance in all three seeds under both weightings. Better mean source use than another restricted interface is not full source preservation. The original residential/attribute margins and nine-category race support limits remain unchanged. This control supports a source/transfer/recovery tradeoff within the allowed teacher information; it establishes neither representation necessity nor privacy or novelty.
