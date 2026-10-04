# Advisor brief: removing mandatory erasure from the joint method

2026-10-04. This is exploratory development evidence on already-exposed Adult rows. The protocol was locked before the fits (`f1d1b76`), and the selection was locked before outer scoring (`b0b9256`).

## What we changed

We started from the ordinary penalty update, which had beaten the projected update last time, and removed the mandatory LEACE map completely:
- **PN:** joint coalition plus local critics.
- **LN:** local critics only.

PN and LN were trained at three strengths on three seeds (18 fits). All other components were exactly the predecessor's. At strength 0, PN and LN reproduce the task-only model bit for bit; this was checked on every seed before any fit.

## What happened

| | Income accuracy | Occupation accuracy | Coalition SEX AUC |
|---|---|---|---|
| Task only (U) | 0.843 | 0.476 | 0.873 |
| **PN, β = 0.1** | **0.842** | 0.473 | **0.840** |
| LN, β = 0.1 | 0.842 | 0.475 | 0.869 |
| PN with erasure (JP, β = 0.1) | 0.773 | 0.440 | 0.817 |

1. **Removing erasure restored accuracy** (+4.3 income points, +1.7 occupation, β-matched). There was no measurable change in nonlinear SEX recovery (+0.003, unresolved). The exact linear guarantee is lost: the linear R² of SEX rises from 0 to 0.23.
2. **PN beat local-only training on the coalition** by 0.029 AUC [0.021, 0.037], at useful accuracy. **But the registered claim is not established**:
   - PN's income recipient leaked slightly more than LN's (+0.010; upper bound 0.019 against a 0.01 guard);
   - on one seed PN missed LN's local allowance at selection.
3. **The training critics were weak.** Fresh critics of the same size, trained on the same frozen views, read more SEX than the online critics in all 45 cases (+0.047 nats on average). At stronger β the online critics sat near chance while information remained.
4. **FARE and the erased arms leak less,** but they fail the utility gates.

## Plain answers

| Question | Answer |
|---|---|
| Accuracy restored? | Yes; PN is at task-only accuracy at β = 0.1. |
| Joint vs local? | 0.029 coalition gain at useful accuracy, but the income-recipient guard failed: **not established**. |
| Vs strongest control? | That control was LN; same answer. |
| Weak critics? | Yes, measured like for like. |
| Guarantees lost? | LEACE's fitted-row linear guardedness; the evidence is attacker-based only. |

## Decision needed

**Should the combined paper report this as a near-miss development result** (a restored-utility trade-off with an established coalition reduction but a failed income-recipient guard), labelled EXPERIMENTAL_NO_ADVANTAGE?

If the method continues, the next registered step on new data addresses the two measured bottlenecks:
- **critic tracking:** fresh-critic restarts;
- **income-recipient leakage:** a per-recipient local weight fixed in advance.
