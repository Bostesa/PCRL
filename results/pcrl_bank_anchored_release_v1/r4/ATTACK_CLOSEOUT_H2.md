# R4 fresh attack closeout: effective H1/H2 bank comparison

**Status.** All 38 effective release instances have a complete public
attack receipt: 36 H1 and two P* H2 technical retries. The two original
P* H1 attempts failed after ATT SEX access because their defense-aware
view was empty; they remain failed attempts and supply no AUC. H2 retained
their source release, seed, code/probability/decision/metadata wire and
30-reader/control-fit slate, adding only public deterministic view columns.
The [H2 correction](PSTAR_H2_TECHNICAL_CORRECTION.md) and
[source plan](PSTAR_H2_SOURCE_PLAN.json) record the change.

**Unit accounting.** B0, D, W_Q and P_STAR have one fixed release per
seed (two instances each); P_STAR uses H2. T, P, L, J and S12 have three
preregistered cached realizations per seed (six instances each). This is
38 effective instances, 114 recipient/coalition views and 1,140 planned
reader/control fits, with no best-draw choice. S21 was screened out by its
22/24 original INNER point utility clauses. The nine admitted arms met
24/24 original INNER point clauses across the two seeds; the separate FIT
bank proxy had failures for some arms, including J 10/12, and is retained
as a diagnostic. Point eligibility is not an interval PASS or a population
confidence guarantee.

**Source identities.** B0 attacks the original W_TSTAR witness release;
W_Q attacks the distinct historical Q witness release; P_STAR attacks its
own historical finite bank and metadata, with its own decoder. T, D, P, L,
J and S12 attack their exact R2 selected policies and cached role releases.
No cross-arm recovery result was aliased: every effective instance received
a fresh matched 30-fit attack slate. Each reader sees the actual emitted
recipient wire or aligned coalition pair, including categorical code,
probabilities, decisions and actual metadata. Public constants were
available as side information. Local, pair, probability-only, defense-aware
and ignore-recipient readers were retained. All 114 null, planted and
shuffle control-gate sets passed.

Equal-seed means average the three cached draws within each stochastic
seed, then weight the two source seeds equally. Lower AUC means less SEX
recovery. These are descriptive development numbers from the frozen fitted
attacker bank on reused INNER rows:

| Arm | Recipient 1 | Recipient 2 | Coalition |
| --- | ---: | ---: | ---: |
| B0 | 0.706600165 | 0.818487689 | 0.838251447 |
| T | 0.701529064 | 0.816875749 | 0.836983118 |
| D | 0.706871557 | 0.814011666 | 0.836270170 |
| P | 0.702636618 | 0.811780219 | 0.832300822 |
| L | 0.703003971 | 0.813320245 | 0.834971076 |
| J | 0.701835115 | 0.813937004 | 0.833785730 |
| S12 | 0.702171244 | 0.815440310 | 0.835492561 |
| W_Q | 0.702428284 | 0.824081114 | 0.840998347 |
| P_STAR | 0.701484939 | 0.761952933 | 0.803140866 |

The strongest favorable **new bank** comparison is P versus B0:
coalition AUC falls by `0.005950625110312635`, while recipient-1 and
recipient-2 AUC fall by `0.003963546275752128` and
`0.006707469382694731`. The strongest historical compatible control is
P_STAR, whose coalition AUC is `0.03511058140664014` below B0. The
adverse joint comparison is J against P: J coalition AUC is
`0.001484908003036689` higher. J is `0.030644864299364194` higher
than P_STAR. Thus the new bank methods do not meet the `0.020` coalition
benefit target against the strongest eligible control, and these numbers
do not support a joint-design advantage. These are comparisons of
descriptive means; R0 owns the locked intervals and final claims.

For exact seed/draw accounting, P coalition AUCs are seed 0
`[0.8343679060163394, 0.8395346621546106, 0.8378041652848952]`
and seed 1 `[0.8250370559017364, 0.8287431041211292,
0.82831803765869]`. J coalition AUCs are seed 0
`[0.8366306520556177, 0.8425682992028172, 0.8421991143227244]`
and seed 1 `[0.8293807038147883, 0.8293069584479212,
0.8226286513117514]`. B0 fixed seed AUCs are
`[0.8467447604145864, 0.8297581335185059]`; P_STAR H2 fixed
seed AUCs are `[0.802702286747637, 0.803579444372175]`.
The [114-row AUC table](ATTACK_AUC_ALL_H2.csv) records every other
view and release without selection.

**Verification and custody.** The
[aggregate](ATTACK_AGGREGATE_ALL_H2.json) SHA-256 is
`0a07ee940a8210532d068d2f7ed111c0f0791361d1fbfdceb8e1101c8bd99e3b`;
the table SHA-256 is
`52f1d1c1a7d19450b65953514d8f47e9c97bd2f76899cf84c78b50bb867a1fbb`.
R5's [independent attack replay](../r5/ATTACK_REPLAY_CLOSEOUT.json),
SHA-256 `032d805530f974e50061b3c99b2c5fbac4cd0125fb1f01bb8b0c43c9025bb32d`,
reconstructed all 114 views and replayed all 38 frozen fitted attacks,
with maximum AUC difference `2.22e-16`. Private fitted models,
predictions and person-level releases remain under owner-only `.private`
custody. This is development evidence on repeatedly used Adult roles,
not independent confirmation or an attack-optimality theorem.
