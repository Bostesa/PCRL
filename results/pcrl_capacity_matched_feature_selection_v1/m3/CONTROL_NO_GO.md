# Matched-control no-go at the pre-sensitive task gate

Status: exploratory development no-go. The two scheduled FIT units completed;
no J/T pair was selected and no downstream Adult release, attack, V2,
endpoint, or inference unit ran. This record uses aggregate FIT receipts and
the sealed potential-unit registry only. It does not report an INNER result.

## Frozen selection and stopping rule

The bank contains five masks per recipient, each removing one categorical
atom while retaining four numeric and four categorical atoms with at most
77 encoded columns. T is chosen before a protected-label read: for each
purpose, keep masks satisfying every accuracy, NLL and Brier point cap
against **U0, Ucal and F** on the fitting-role task-validation fold; choose
lowest validation NLL, then mask SHA. The two chosen masks form T. If either
purpose has no eligible mask, publish the ten invalid rule slots and stop
before fitting SEX critics or constructing a sensitive PairTable.

Only after a valid T would J search the same 25 ordered mask pairs. J requires
both task caps and recipient search AUC no more than T's local AUC plus
0.010. It minimizes fitting-fold coalition search AUC; ties use summed task
NLL, both mask SHAs, then pair ID. This is an optimum only within the frozen
bank and critic slate. The nine controls use that same bank and fitted table;
none can be selected from an absent T/PairTable. No control can be validly
scored on INNER without a selected pair, actual release, all three complete
views and their fresh reader/V2 receipts.

## Observed FIT-only decision

| Seed | FIT receipt SHA256 | Control-plan SHA256 | Eligible income masks | Eligible occupation masks | Status |
| --- | --- | --- | ---: | ---: | --- |
| 0 | `be387533d68828ebc45d12132ba5f9c4d1a90092a68160d8af45d78434e70534` | `5aa3c14b2164d18b83f54d74f57b7782464b04dcfd5c982584598525b5d2f42d` | 0/5 | 0/5 | `NO_FEASIBLE_J` |
| 1 | `0984398e414f9fe3c96e5b988483d67cbb2df56f2762e23b21e7828e807f7ccb` | `bbc8ef1883c57f348eb2c9f6d44abedb798eb82a246ac59ab9a5b1a425a09bd0` | 0/5 | 0/5 | `NO_FEASIBLE_J` |

Both task-only receipts give the reason `no utility-feasible task-only income
mask`. Occupation also has zero eligible masks. Even each purpose's *best*
U0 NLL margin over the five masks is below zero: income −0.035983 and
occupation −0.044394 for seed 0; income −0.031630 and occupation −0.048718
for seed 1. Ucal best NLL margins are also negative for both tasks and seeds.
These are fitting-role validation margins against frozen limits, not fresh
assessment estimates. The FIT receipts have null PairTable and joint-selection
SHAs, no final selected head packages and no SEX critic specification.

For **each seed**, J, T, I, S12, S21, SHARED, T_U, I_U, S12_U and S21_U have
status `INVALID_NO_UTILITY_FEASIBLE_PAIR` and `pair_id:null`. The exact alias
policy remains untested: an identical mask pair would have remained a valid
tie; a common raw union alone would not prove the complete transmitted
releases or heads identical. With no selected pair, there is no release alias,
coalition AUC, local AUC, utility pass, sequential comparison or joint-design
claim.

## Registered downstream identities and disposition

The immutable potential-unit registry SHA256 is
`0eaae64a41ac7a279db3048596ce4b0d607f2523ffaacc958fd5b7876f7ae9f2`.
The campaign lock SHA256 is
`93ed6271f30da39e6010784895d105f77a96a17fe573dcaf5209fb8bc932a6cb`.
The mask IDs forming every registered ordered pair are
`M_DROP_EDUCATION_FACT`, `M_DROP_MARITAL_STATUS`, `M_DROP_NATIVE_COUNTRY`,
`M_DROP_RELATIONSHIP` and `M_DROP_WORKCLASS`. For each `s ∈ {0,1}` and each
of their 25 ordered IDs `p = income_mask__occupation_mask`, the following
identities were registered as *potential* units and were not enqueued:

| Stage | Exact registry identity rule | Potential count for seeds 0–1 | Disposition |
| --- | --- | ---: | --- |
| RAW union | `M4-RAW-S{s}-{union_sha256[:16].upper()}` for the six registered raw unions per seed | 12 | Not run: no selected pair/view |
| Release | `M4-REL-S{s}-{p}` | 50 | Not run |
| Fresh attack | `M4-ATT-S{s}-{p}-{view}`, where view is INCOME, OCCUPATION or COALITION | 150 | Not run |
| Real V2 | `M4-V2-S{s}-{p}-{view}`, with the same three views | 150 | Not run |
| Independent saved-release replay | `M5-REPLAY-S{s}-{p}` | 50 | Not run |

The six exact registered RAW suffixes per seed are
`5085DF35D009BF8C`, `782405BE1A3BEEE9`, `860BBC8CFAFD6C86`,
`D76E9CB6A3279D08`, `DAB8998C7840415B` and `E6E7D3F551D27C86`.
The two completed units are exactly `M2-FIT-S0` and `M2-FIT-S1`.

The other registered screen-stage identities are
`M4-ENDPOINTS-FIRST`, `M4-ENDPOINTS-SCREEN`, `M3-INFERENCE-SCREEN` and
`M5-REPLAY-AGGREGATE-SCREEN`. None ran. `M4-ENDPOINTS-FIRST` itself requires
a selected J and T in the FIT plan and verifies the selected RAW, release,
three attack and three V2 receipts before opening INNER labels. The remaining
seed-2 and FINALIST registry entries were conditional potential units and
were not activated: `M2-FIT-S2`, six `M4-RAW-S2-*`, 25 each
`M4-REL-S2-{p}` and `M5-REPLAY-S2-{p}`, 75 each
`M4-ATT-S2-{p}-{view}` and `M4-V2-S2-{p}-{view}`, plus
`M4-ENDPOINTS-FINALIST`, `M3-INFERENCE-FINALIST` and
`M5-REPLAY-AGGREGATE-FINALIST`. Registering 628 potential units did not authorize or
execute them; at this closeout read, the actual queue contains only the two
completed FIT units.

This is a no-go for this **fixed five-mask capacity profile and frozen mean
utility premises**. It does not establish that all feature selection or raw
representation methods fail. Any broader mask bank, different cap or altered
reference is a new development design requiring its own lock and budget.
