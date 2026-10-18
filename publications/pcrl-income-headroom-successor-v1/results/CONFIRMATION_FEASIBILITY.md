# Confirmation feasibility: successor H3_STRONG_LJDIR_PB (planning arithmetic, no new population)

**Data.** Public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, development roles only. The service
predicts income class (2) and occupation group (5) from AGEP, WKHP, COW, SCHL, MAR and POBP; readers estimate the recorded SEX
column from the released predictions. No population, sealed assessment or download was opened for this analysis.

**Source.** `CONFIRMATION_FEASIBILITY_RESULTS.json` (unit IHS_FEASIBILITY_V1, `ihs/feasibility.py`), from the one logged
DEV_REPORT evaluation's private bootstrap replicates (2,048 household draws, 3,955 households) and its published points.
Method as the validation's V2: estimates for n new households ~ MVN(theta, Sigma_DEV x 3,955 / n); joint power by Monte Carlo
(40,000 draws, seed 2026101803); clause critical value c1 = 2.3949991 (three alpha-spending claims). The binding clauses were
chosen by a rule fixed before the evaluation: the 4 utility or guard conditions with the smallest DEV margin in SE units.

**Estimand and limits.** The frozen successor and comparators with the frozen fitted readers and winners, applied to new
independent households from the same Florida 2019 population, unweighted, no transport. The curves do not cover designs that
refit readers or re-select winners, another state or year, or design weights. The two seeds share people. These are planning
numbers under stated scenarios, not statements that the release complies or fails.

## What binds now

| Condition (successor, DEV_REPORT) | Point | SE | Margin in SE units | At 17,510 households |
|---|---|---|---|---|
| occupation NLL excess vs Ucal, seed 0 (inherited JOINT_TREE output) | 0.008665 | 0.003356 | 0.40 | 0.84 |
| men's within-SEX occupation Brier excess vs U0, seed 1 (inherited) | 0.003951 | 0.001908 | 0.55 | 1.16 |
| men's within-SEX occupation Brier excess vs U0, seed 0 (inherited) | 0.003235 | 0.002046 | 0.86 | 1.81 |
| income accuracy loss vs U0 / Ucal, seed 1 | 0.007067 | 0.003377 | 0.87 | 1.83 |

Three of the four binding conditions belong to the inherited occupation output, which no income-side mechanism can change. The
income summed-Brier clauses, which bound the retired candidate at 0.03-0.04 SE, now have 3.8-5.0 SE of margin (NLL 5.8-7.0 SE;
both under the R2 rule's max(successor, H_LJ) SE). The seed-1 income accuracy clause bound both releases: 0.11 SE for the retired
candidate, 0.87 SE here in own-SE units (0.83 SE under the R2 rule). The table uses the successor's own DEV SEs.

## Recipient non-worsening margins (successor single view minus reference, limit 0.010; occupation views against H_LJ are identical releases)

| Clause | ENHANCED point (SE); margin in SE units [at 17,510] | ORIGINAL point (SE); margin in SE units |
|---|---|---|
| income vs H_LJ, seed 0 | −0.0078 (0.0029); 6.11 [12.85] | −0.0163 (0.0028); 9.34 |
| **income vs H_LJ, seed 1** | **+0.0061 (0.0065); 0.60 [1.27]** | −0.0070 (0.0035); 4.87 |
| income vs NEURAL_RAW, seed 0 | −0.0046 (0.0039); 3.79 [7.98] | −0.0044 (0.0031); 4.67 |
| income vs NEURAL_RAW, seed 1 | −0.0071 (0.0029); 5.81 [12.23] | −0.0091 (0.0042); 4.50 |
| occupation vs NEURAL_RAW, seeds 0 / 1 | −0.0616 / −0.0439; 9.77 / 7.88 | −0.0613 / −0.0394; 9.44 / 7.19 |

The seed-1 income recipient clause against H_LJ is the reader-sensitive one: the enhanced bank's H_LJ income winner reads lower
and the successor's reads higher than under the original bank (side BOTH in `report/DEV_REPORT_RESULTS.json`).

## Required households for 80% joint power (governing scorecard: the less favourable; ENHANCED in the plug-in cells)

Definitions: D1 = claim P's registered conjunction (28 utility and guard point gates, 8 recipient clause bounds <= 0.010,
4 pair-benefit clause bounds > 0; 38 conditions in the computation, because the two occupation recipient clauses against H_LJ
compare identical releases); it does not include the composite's reserve condition R2. D2 = D1 with every utility and guard
upper bound at or below its cap; D3 = D2 with every benefit bound above 0.020.

Each cell gives the governing (less favourable) scorecard's value; the other scorecard's value is in brackets where it differs.

| Scenario | D1 | D2 | D3 |
|---|---|---|---|
| plug-in | about 128,600 (ENHANCED) [ORIGINAL 27,700] | about 291,800 | never (limit 0) |
| benefits x 0.75 | about 128,600 (ENHANCED) [30,700] | about 291,800 | never |
| benefits x 0.5 | about 128,600 (ENHANCED) [51,200] | about 291,800 | never |
| binding clauses halfway to their caps | about 157,800 (ENHANCED) [116,100] | about 1.1 million | never |
| binding clauses at an assumed true slack of 0.002 | about 116,100 (ENHANCED) [20,400] | about 174,800 (ENHANCED) [157,800] | never |
| binding clauses at a true slack of 0.001 | about 142,500 (ENHANCED) [77,100] | about 597,500 | never |
| binding clauses at a true slack of 0 | never (limit 0.16) | never | never |
| plug-in + rough reader-refit spread on the benefits (tau 0.007-0.009) | about 1.36 million (ORIGINAL governs; limit 0.83) [ENHANCED 733,300] | about 1.36 million (ORIGINAL) [733,300] | never |
| HYPOTHETICAL transport (benefits x 0.5, every utility margin halved) | about 157,800 (ENHANCED) [116,100] | about 1.1 million | never |

- **Why ENHANCED governs D1.** Its weakest condition is the seed-1 income recipient clause against H_LJ (point +0.0061, SE
  0.0065; the reader-sensitive clause), then the inherited occupation NLL clause. ORIGINAL's is the inherited occupation NLL.
- **Power at the size that exists.** About 17,510 Florida 2019 households could test this release (the sealed 20% and the
  capped-out households; both carry the replication's disclosure item D2, an in-memory extraction (validation
  `POPULATION_INVENTORY.md`), and opening them needs a separate authorization and frozen protocol). At that
  size, D1 joint power is 0.089 (ENHANCED) [0.698 ORIGINAL]; D2 is 0.001 [0.008].
- **Assurance** (flat-prior predictive heuristic centred on the evaluated values): at most 0.30 (ENHANCED) and 0.41 (ORIGINAL)
  at any n for D1.
- **Same-size replication** (3,955 households, plug-in): D1 0.003 (ENHANCED) [0.100 ORIGINAL]; D2 about 0.
- **The 0.020 target** cannot be established at 0.020 under these benefits (all four plug-in benefits are 0.012-0.017); this
  does not show that the target is false.

## Compared with the retired candidate (validation, same method)

| Quantity | Retired B3CTL_UNCOUPLED_MIX50_C6A | Successor H3_STRONG_LJDIR_PB |
|---|---|---|
| binding income Brier slack | 0.03-0.04 SE | 3.8-5.0 SE (income Brier is no longer binding) |
| D1 n80 at the plug-in (governing) | about 4.6 million households | about 128,600 |
| D2 n80 at the plug-in | about 60 million | about 291,800 |
| D1 power at about 17,510 households (ENHANCED / ORIGINAL) | 0.013 / 0.252 | 0.089 / 0.698 |
| D1 assurance limit (ENHANCED / ORIGINAL) | 0.10 / 0.16 | 0.30 / 0.41 |

The successor reduces the plug-in requirement by about 36-fold (D1) and about 205-fold (D2), but the required sizes still exceed
the households that exist for a same-population test, and a zero true slack in any binding clause means 80% power is never
reached; the requirement is not identified by these data.

## Conclusions (each conditional on its scenario)

1. **No realistic same-population confirmation of claim P's conjunction exists for this release.** At the plug-in the governing
   requirement is about 129,000 households (D1) and 292,000 (D2) against about 17,510 available; with the rough reader-refit
   spread added to the benefits, the governing (ORIGINAL) requirement rises to about 1.36 million (ENHANCED about 733,000).
2. **The binding problem has moved.** The retired candidate's income Brier bottleneck no longer binds on DEV for this release;
   what binds now is the inherited occupation output (men's guard, occupation NLL), the seed-1 income accuracy clause, and the
   reader-sensitive income recipient clause against H_LJ. Family O's occupation recalibration improves the men's guard (0.32 /
   0.62 → 1.83 / 1.62 SE) and the occupation NLL margins on MODEL_SELECT, but was not part of the nominee and has no DEV evidence
   (`RESEARCH_DECISION.md`).
3. **A fresh population is not a cure** for these margins, and another state or year would test transfer, not this release.
