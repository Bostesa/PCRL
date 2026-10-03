# Multi-recipient output check (S4, secondary, exploratory)

**Contract.** Fixed from the schema and PERMISSION_TABLE before any fit:
- two recipients, Adult `income_prediction` and `employment_analysis`;
- the same encoder seed and the same people;
- attribute **race**, which is disallowed for both purposes.

The first eligible pair in the registered order was used. Row identities, roles, race labels and exposure exclusions
were asserted identical across the two purposes.

**Views.** Nine view slots per seed: each recipient alone and the pair together, each under three contracts (full
logits, centred logits, hard decisions). The pair's attacker bank contained the pair attacker and **both**
single-recipient attackers, as ignore-the-other candidates; the bank was selected on validation only.

**Family.** S4 has 6 endpoints, with Bonferroni z = 2.638 (two-sided). A combination finding requires the pair to beat
**both** single recipients by more than 0.02 AUC.

## Recovery of race

Supported classes are {1, 2, 4}. Values are the mean over 3 encoder seeds; the bracketed values are the simultaneous
lower bounds.

| Contract | Income alone | Employment alone | Pair | Pair − income | Pair − employment | Combination finding |
|---|---|---|---|---|---|---|
| Full logits | 0.845 | 0.775 | 0.917 | +0.073 (0.064) PASS | +0.143 (0.128) PASS | **yes** |
| Centred logits (offset removed) | 0.731 | 0.792 | 0.883 | +0.152 (0.137) PASS | +0.092 (0.078) PASS | **yes** |
| Hard decisions | 0.546 | 0.576 | 0.611 | +0.065 (0.044) PASS | +0.034 (0.023) PASS | **yes** |

**Bank selection.** The pair bank selected the pair attacker on every seed, except the full-logit contract on seed 2.
There it chose the income-only attacker, so that seed contributes a difference of exactly 0. The endpoint still passes
on the mean.

**Utility of both tasks** (frozen heads, assessment):

| Task | Accuracy, s0 / s1 / s2 | Constant | Comment |
|---|---|---|---|
| Income | 0.786 / 0.783 / 0.797 | 0.749 | modest |
| Employment (occupation group) | 0.998 / 0.999 / 0.894 | 0.277 | near a recoding of the occupation input column |

## Reading

- On these heads and people, combining two purpose recipients' outputs adds measurable race recovery over either alone,
  under every contract, including hard decisions alone (0.61 against 0.55 / 0.58).
- Low recovery from one recipient therefore does not imply low recovery from a coalition of recipients.
- This is a measured property of these fitted attackers. It is **not** a composition theorem. A small or absent effect
  elsewhere would not prove a composition guarantee.
- No joint-training claim is possible: nothing was trained jointly here.
