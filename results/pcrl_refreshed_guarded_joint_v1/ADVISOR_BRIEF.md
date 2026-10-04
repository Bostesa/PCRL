# Advisor brief: refreshed critics with local feedback (J-G)

2026-10-04. This is development evidence on already-exposed Adult rows. A new 3,397-row development assessment was sealed until the evaluation lock was pushed. The pre-fit lock, the Stage B freeze and the evaluation lock were each pushed before the stage they govern.

## What we tried

The predecessor showed two problems:
- its training attackers fell behind the encoder;
- joint protection improved the pair partly by letting one recipient leak more.

The candidate J-G addressed both:
- **Refreshed critics.** Every four epochs it freezes the features, refits strong critics (keeping whichever is better of a continued and a fresh critic), and resumes against them.
- **Local feedback.** It raises a recipient's protection weight whenever that recipient's own leakage exceeds a budget set by the local-only reference.

It was compared with matched local, refreshed, online and official baselines on 3 seeds, a 3-value strength grid and 4 checkpoints.

## What happened

| | Income accuracy | Occupation accuracy | Coalition SEX AUC |
|---|---|---|---|
| Task only (U) | 0.844 | 0.478 | 0.882 |
| Local + feedback (L-G) | 0.843 | 0.479 | 0.867 |
| **J-G (candidate; descriptive)** | 0.841 | 0.481 | 0.855 |
| Joint, old online critics (J-O; C\*) | 0.842 | 0.468 | **0.816** |
| FARE | 0.842 | 0.455 | 0.701 |

1. **No advantage was established.**
   - J-G had no feasible nominee on any seed. The strongest control was the old online-critic joint arm (J-O), and J-G could not match its local protection.
   - Against L-G, J-G lowered coalition recovery by 0.012 [0.006, 0.018], below the 0.02 target, and its occupation-view guard narrowly failed.
2. **Refreshing the critics kept them current but did not protect more.**
   - The measured critic lag fell to about 0 after refits, against 0.06–0.07 nats for online critics.
   - At the same strength, though, the online arm leaked less. Its per-step input whitening makes the penalty gradient about 3× larger. The lever was gradient scale, not critic freshness.
3. **Local feedback never engaged.** The budgets were loose and the multipliers stayed near zero. The predecessor's income-recipient shift did not recur, but not because of the feedback.
4. **Accuracy was preserved.** J-G is within 0.3 points of task-only on both tasks.

## Decision needed

- **Close J-G as EXPERIMENTAL_NO_ADVANTAGE?** The code, models and deploy command are runnable and backed up.
- **If the line continues:** pre-register a study that equalises penalty-gradient scale across critic schedules and sets budgets below the reference, on fresh data. Do not spend a confirmation population on J-G. `CONFIRMATION_PLAN.md` audits candidate populations: ACS 2016–2018 CA are spent, and Texas 2018 is already named by another programme.
