# Complete fixed-pair comparison

The full cube has 57,600 per-seed rows: every 5×5 J/Iplus pair, both interfaces, all three seeds, both weightings, four panels, four deltas, three audit scopes, two budgets and each attribute separately. [UTILITY_MATCHES.csv.gz](UTILITY_MATCHES.csv.gz) retains all five utility differences, all eleven forbidden-role gain differences, the two independent source-floor checks, support flags and exact exclusion reasons. [UTILITY_MATCHES_AGGREGATE.csv.gz](UTILITY_MATCHES_AGGREGATE.csv.gz) holds every fixed-pair mean, SD and 0–3 qualifying count; all seed values are included even when that seed fails feasibility.

The following compact view uses expanded catch-up-inclusive 360-epoch SEX recovery. Each cell reports the number of **fixed coefficient pairs** eligible or qualifying in 3/3, 2/3 or 1/3 seeds. Eligibility requires both source floors and the utility comparison; qualification additionally requires lower SEX gain. Counts describe this finite grid, not independent experiments, probabilities or a selected method. No-match findings distinguish close matching from directional comparisons.

## F; unweighted

| Utility panel | δ | Close eligible 3/2/1 | Close lower-SEX 3/2/1 | Directional eligible 3/2/1 | Directional lower-SEX 3/2/1 |
| --- | --- | --- | --- | --- | --- |
| source_only | 0 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 3 | 0 / 0 / 1 |
| source_only | 0.0005 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 2 / 2 | 0 / 0 / 2 |
| source_only | 0.001 | 0 / 1 / 1 | 0 / 0 / 0 | 0 / 2 / 3 | 0 / 0 / 2 |
| source_only | 0.002 | 0 / 2 / 8 | 0 / 1 / 6 | 0 / 5 / 9 | 0 / 1 / 9 |
| residential_transfer | 0 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 3 | 0 / 0 / 1 |
| residential_transfer | 0.0005 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 3 | 0 / 0 / 1 |
| residential_transfer | 0.001 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 4 | 0 / 0 / 1 |
| residential_transfer | 0.002 | 0 / 1 / 5 | 0 / 0 / 4 | 0 / 2 / 9 | 0 / 0 / 7 |
| source_commute | 0 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 0 | 0 / 0 / 0 |
| source_commute | 0.0005 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 0 | 0 / 0 / 0 |
| source_commute | 0.001 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 1 | 0 / 0 / 0 |
| source_commute | 0.002 | 0 / 1 / 6 | 0 / 0 / 5 | 0 / 1 / 6 | 0 / 0 / 5 |
| full_authorized | 0 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 0 | 0 / 0 / 0 |
| full_authorized | 0.0005 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 0 | 0 / 0 / 0 |
| full_authorized | 0.001 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 1 | 0 / 0 / 0 |
| full_authorized | 0.002 | 0 / 1 / 4 | 0 / 0 / 3 | 0 / 1 / 4 | 0 / 0 / 3 |

## F; PWGTP

| Utility panel | δ | Close eligible 3/2/1 | Close lower-SEX 3/2/1 | Directional eligible 3/2/1 | Directional lower-SEX 3/2/1 |
| --- | --- | --- | --- | --- | --- |
| source_only | 0 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 1 | 0 / 0 / 0 |
| source_only | 0.0005 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 6 | 0 / 0 / 2 |
| source_only | 0.001 | 0 / 1 / 3 | 0 / 0 / 2 | 0 / 1 / 11 | 0 / 0 / 3 |
| source_only | 0.002 | 0 / 2 / 8 | 0 / 0 / 2 | 0 / 4 / 9 | 0 / 0 / 3 |
| residential_transfer | 0 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 1 | 0 / 0 / 0 |
| residential_transfer | 0.0005 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 6 | 0 / 0 / 2 |
| residential_transfer | 0.001 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 11 | 0 / 0 / 3 |
| residential_transfer | 0.002 | 0 / 1 / 5 | 0 / 0 / 2 | 0 / 4 / 8 | 0 / 0 / 3 |
| source_commute | 0 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 0 | 0 / 0 / 0 |
| source_commute | 0.0005 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 2 | 0 / 0 / 1 |
| source_commute | 0.001 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 6 | 0 / 0 / 1 |
| source_commute | 0.002 | 0 / 1 / 4 | 0 / 0 / 2 | 0 / 1 / 8 | 0 / 0 / 2 |
| full_authorized | 0 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 0 | 0 / 0 / 0 |
| full_authorized | 0.0005 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 2 | 0 / 0 / 1 |
| full_authorized | 0.001 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 6 | 0 / 0 / 1 |
| full_authorized | 0.002 | 0 / 1 / 4 | 0 / 0 / 2 | 0 / 1 / 8 | 0 / 0 / 2 |

## P; unweighted

| Utility panel | δ | Close eligible 3/2/1 | Close lower-SEX 3/2/1 | Directional eligible 3/2/1 | Directional lower-SEX 3/2/1 |
| --- | --- | --- | --- | --- | --- |
| source_only | 0 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 8 | 0 / 0 / 2 |
| source_only | 0.0005 | 0 / 1 / 7 | 0 / 0 / 5 | 0 / 2 / 14 | 0 / 0 / 7 |
| source_only | 0.001 | 0 / 5 / 10 | 0 / 0 / 12 | 0 / 5 / 17 | 0 / 0 / 13 |
| source_only | 0.002 | 0 / 6 / 15 | 0 / 1 / 14 | 0 / 6 / 19 | 0 / 1 / 14 |
| residential_transfer | 0 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 5 | 0 / 0 / 1 |
| residential_transfer | 0.0005 | 0 / 1 / 3 | 0 / 0 / 3 | 0 / 2 / 11 | 0 / 0 / 6 |
| residential_transfer | 0.001 | 0 / 5 / 9 | 0 / 0 / 11 | 0 / 5 / 17 | 0 / 0 / 13 |
| residential_transfer | 0.002 | 0 / 6 / 15 | 0 / 1 / 14 | 0 / 6 / 19 | 0 / 1 / 14 |
| source_commute | 0 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 6 | 0 / 0 / 0 |
| source_commute | 0.0005 | 0 / 1 / 5 | 0 / 0 / 4 | 0 / 1 / 15 | 0 / 0 / 6 |
| source_commute | 0.001 | 0 / 5 / 10 | 0 / 0 / 12 | 0 / 5 / 17 | 0 / 0 / 13 |
| source_commute | 0.002 | 0 / 6 / 15 | 0 / 1 / 14 | 0 / 6 / 19 | 0 / 1 / 14 |
| full_authorized | 0 | 0 / 1 / 0 | 0 / 0 / 0 | 0 / 1 / 4 | 0 / 0 / 0 |
| full_authorized | 0.0005 | 0 / 1 / 2 | 0 / 0 / 2 | 0 / 1 / 12 | 0 / 0 / 5 |
| full_authorized | 0.001 | 0 / 5 / 9 | 0 / 0 / 11 | 0 / 5 / 17 | 0 / 0 / 13 |
| full_authorized | 0.002 | 0 / 6 / 15 | 0 / 1 / 14 | 0 / 6 / 19 | 0 / 1 / 14 |

## P; PWGTP

| Utility panel | δ | Close eligible 3/2/1 | Close lower-SEX 3/2/1 | Directional eligible 3/2/1 | Directional lower-SEX 3/2/1 |
| --- | --- | --- | --- | --- | --- |
| source_only | 0 | 0 / 0 / 1 | 0 / 0 / 0 | 0 / 0 / 11 | 0 / 0 / 1 |
| source_only | 0.0005 | 0 / 0 / 5 | 0 / 0 / 2 | 0 / 0 / 16 | 0 / 0 / 3 |
| source_only | 0.001 | 0 / 0 / 15 | 0 / 0 / 5 | 0 / 0 / 21 | 0 / 0 / 5 |
| source_only | 0.002 | 0 / 0 / 20 | 0 / 0 / 7 | 0 / 0 / 24 | 0 / 0 / 7 |
| residential_transfer | 0 | 0 / 0 / 1 | 0 / 0 / 0 | 0 / 0 / 9 | 0 / 0 / 1 |
| residential_transfer | 0.0005 | 0 / 0 / 4 | 0 / 0 / 2 | 0 / 0 / 16 | 0 / 0 / 3 |
| residential_transfer | 0.001 | 0 / 0 / 13 | 0 / 0 / 5 | 0 / 0 / 21 | 0 / 0 / 5 |
| residential_transfer | 0.002 | 0 / 0 / 20 | 0 / 0 / 7 | 0 / 0 / 24 | 0 / 0 / 7 |
| source_commute | 0 | 0 / 0 / 1 | 0 / 0 / 0 | 0 / 0 / 11 | 0 / 0 / 1 |
| source_commute | 0.0005 | 0 / 0 / 5 | 0 / 0 / 2 | 0 / 0 / 16 | 0 / 0 / 3 |
| source_commute | 0.001 | 0 / 0 / 15 | 0 / 0 / 5 | 0 / 0 / 21 | 0 / 0 / 5 |
| source_commute | 0.002 | 0 / 0 / 20 | 0 / 0 / 7 | 0 / 0 / 24 | 0 / 0 / 7 |
| full_authorized | 0 | 0 / 0 / 1 | 0 / 0 / 0 | 0 / 0 / 9 | 0 / 0 / 1 |
| full_authorized | 0.0005 | 0 / 0 / 4 | 0 / 0 / 2 | 0 / 0 / 16 | 0 / 0 / 3 |
| full_authorized | 0.001 | 0 / 0 / 13 | 0 / 0 / 5 | 0 / 0 / 21 | 0 / 0 / 5 |
| full_authorized | 0.002 | 0 / 0 / 20 | 0 / 0 / 7 | 0 / 0 / 24 | 0 / 0 / 7 |

## Every local comparator for each primary F J point

All 25 comparisons are shown, including excluded points. Utility differences and gain differences are J minus Iplus. The table uses unweighted expanded catch-up-inclusive 360-epoch SEX results and the primary residential-transfer panel. Each exclusion cell lists the three seed-specific reasons; empty means that seed qualifies. The full cube supplies the PWGTP, independent-scope, 120-epoch and RAC1P counterparts without selecting a favorable scope.

| J β | Local β | Close | Directional | Δ SEX gain | Δ residence loss | Directional exclusions |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0/3 | 0/3 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | s0: no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage, no_strict_SEX_gain_improvement; s2: no_strict_SEX_gain_improvement |
| 0 | 0.025 | 0/3 | 0/3 | 0.00840 ± 0.00407 | 0.00289 ± 0.00581 | s0: utility/public_coverage, utility/same_residence, no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage, no_strict_SEX_gain_improvement; s2: utility/income_binary, utility/civilian_at_work, utility/same_residence, no_strict_SEX_gain_improvement |
| 0 | 0.05 | 0/3 | 0/3 | 0.00613 ± 0.00334 | -0.00592 ± 0.00364 | s0: utility/public_coverage, no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage, no_strict_SEX_gain_improvement; s2: utility/income_binary, utility/public_coverage, no_strict_SEX_gain_improvement |
| 0 | 0.1 | 0/3 | 0/3 | 0.00764 ± 0.00389 | -0.00197 ± 0.00805 | s0: utility/public_coverage, no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/public_coverage, no_strict_SEX_gain_improvement; s2: utility/income_binary, utility/same_residence, no_strict_SEX_gain_improvement |
| 0 | 0.2 | 0/3 | 0/3 | 0.00531 ± 0.00430 | -0.00562 ± 0.01027 | s0: utility/public_coverage, no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage, no_strict_SEX_gain_improvement; s2: Iplus/source/public_coverage, utility/income_binary, utility/same_residence, no_strict_SEX_gain_improvement |
| 0.025 | 0 | 0/3 | 1/3 | -0.00713 ± 0.00548 | 0.00147 ± 0.00502 | s0: utility/civilian_at_work; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/civilian_at_work, utility/same_residence; s2: qualifies |
| 0.025 | 0.025 | 0/3 | 0/3 | 0.00127 ± 0.00690 | 0.00436 ± 0.00214 | s0: utility/civilian_at_work, utility/same_residence, no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/same_residence; s2: utility/income_binary, utility/same_residence, no_strict_SEX_gain_improvement |
| 0.025 | 0.05 | 0/3 | 0/3 | -0.00100 ± 0.00306 | -0.00445 ± 0.00267 | s0: utility/civilian_at_work; s1: J/source/public_coverage, Iplus/source/public_coverage; s2: utility/income_binary, no_strict_SEX_gain_improvement |
| 0.025 | 0.1 | 0/3 | 0/3 | 0.00051 ± 0.00230 | -0.00050 ± 0.00627 | s0: no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/civilian_at_work, utility/public_coverage; s2: utility/same_residence |
| 0.025 | 0.2 | 0/3 | 0/3 | -0.00183 ± 0.00234 | -0.00414 ± 0.00683 | s0: no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage; s2: Iplus/source/public_coverage, utility/same_residence |
| 0.05 | 0 | 0/3 | 0/3 | -0.00832 ± 0.00851 | 0.00352 ± 0.00912 | s0: utility/income_binary; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/civilian_at_work, utility/public_coverage, utility/same_residence; s2: utility/public_coverage, utility/same_residence |
| 0.05 | 0.025 | 0/3 | 0/3 | 0.00007 ± 0.00944 | 0.00641 ± 0.00701 | s0: utility/public_coverage, no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/civilian_at_work, utility/public_coverage, utility/same_residence; s2: utility/income_binary, utility/public_coverage, utility/same_residence, no_strict_SEX_gain_improvement |
| 0.05 | 0.05 | 0/3 | 0/3 | -0.00219 ± 0.00580 | -0.00240 ± 0.00794 | s0: utility/public_coverage, no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/public_coverage, utility/same_residence; s2: utility/income_binary, utility/public_coverage, no_strict_SEX_gain_improvement |
| 0.05 | 0.1 | 0/3 | 0/3 | -0.00068 ± 0.00511 | 0.00155 ± 0.01107 | s0: utility/public_coverage, no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/civilian_at_work, utility/public_coverage, utility/same_residence; s2: utility/income_binary, utility/public_coverage, utility/same_residence |
| 0.05 | 0.2 | 0/3 | 0/3 | -0.00302 ± 0.00496 | -0.00210 ± 0.01037 | s0: utility/public_coverage, no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/civilian_at_work; s2: Iplus/source/public_coverage, utility/income_binary, utility/same_residence |
| 0.1 | 0 | 0/3 | 0/3 | -0.01347 ± 0.00706 | 0.00579 ± 0.00464 | s0: utility/income_binary, utility/civilian_at_work, utility/same_residence; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/same_residence; s2: utility/public_coverage, utility/same_residence |
| 0.1 | 0.025 | 0/3 | 0/3 | -0.00507 ± 0.01034 | 0.00868 ± 0.00119 | s0: utility/income_binary, utility/civilian_at_work, utility/same_residence; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/same_residence; s2: utility/income_binary, utility/public_coverage, utility/same_residence, no_strict_SEX_gain_improvement |
| 0.1 | 0.05 | 0/3 | 0/3 | -0.00734 ± 0.00688 | -0.00013 ± 0.00113 | s0: utility/civilian_at_work, utility/public_coverage; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/same_residence; s2: utility/income_binary, utility/public_coverage, no_strict_SEX_gain_improvement |
| 0.1 | 0.1 | 0/3 | 0/3 | -0.00583 ± 0.00379 | 0.00382 ± 0.00449 | s0: utility/public_coverage; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/same_residence; s2: utility/income_binary, utility/public_coverage, utility/same_residence |
| 0.1 | 0.2 | 0/3 | 0/3 | -0.00816 ± 0.00333 | 0.00018 ± 0.00581 | s0: utility/income_binary, utility/public_coverage; s1: J/source/public_coverage, Iplus/source/public_coverage; s2: Iplus/source/public_coverage, utility/income_binary, utility/same_residence |
| 0.2 | 0 | 0/3 | 0/3 | -0.00632 ± 0.00539 | 0.00443 ± 0.00300 | s0: utility/income_binary, utility/civilian_at_work, utility/same_residence; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/income_binary, utility/civilian_at_work, utility/same_residence; s2: J/source/public_coverage, utility/income_binary, utility/public_coverage, utility/same_residence |
| 0.2 | 0.025 | 0/3 | 0/3 | 0.00207 ± 0.00770 | 0.00732 ± 0.00283 | s0: utility/income_binary, utility/civilian_at_work, utility/same_residence, no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/income_binary, utility/same_residence; s2: J/source/public_coverage, utility/income_binary, utility/public_coverage, utility/same_residence, no_strict_SEX_gain_improvement |
| 0.2 | 0.05 | 0/3 | 0/3 | -0.00019 ± 0.00393 | -0.00149 ± 0.00072 | s0: utility/income_binary, utility/civilian_at_work; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/income_binary; s2: J/source/public_coverage, utility/income_binary, utility/public_coverage, no_strict_SEX_gain_improvement |
| 0.2 | 0.1 | 0/3 | 0/3 | 0.00132 ± 0.00154 | 0.00246 ± 0.00551 | s0: no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/income_binary, utility/civilian_at_work; s2: J/source/public_coverage, utility/income_binary, utility/public_coverage, utility/same_residence, no_strict_SEX_gain_improvement |
| 0.2 | 0.2 | 0/3 | 0/3 | -0.00102 ± 0.00126 | -0.00119 ± 0.00734 | s0: utility/income_binary, utility/public_coverage, no_strict_SEX_gain_improvement; s1: J/source/public_coverage, Iplus/source/public_coverage, utility/income_binary; s2: J/source/public_coverage, Iplus/source/public_coverage, utility/income_binary, utility/same_residence |

Pareto membership in [NONDOMINATED_POINTS.csv.gz](NONDOMINATED_POINTS.csv.gz) uses ordinary componentwise dominance with 1e−12 roundoff, not the utility-matching δ. Every vector is explicitly named in the frozen rules. F, P and their union are separate domains; individual-seed and three-seed-mean frontiers remain separate. A projected two-axis plot cannot establish dominance in a five-task or all-forbidden-role vector. Missing components prevent a full-vector assessment, and numerical race nondominance does not repair category support. [VECTOR_TRADEOFF.csv](VECTOR_TRADEOFF.csv) lists every component that improves or worsens for each fixed J/local pair, including all individual forbidden targets. A SEX benefit alone is not an overall tradeoff improvement.
