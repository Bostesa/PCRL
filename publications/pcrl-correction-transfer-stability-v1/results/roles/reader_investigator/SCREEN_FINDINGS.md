# Screen study: reader investigator's findings (saved by the coordinator, 19:53Z)

Saved verbatim in substance from the reader investigator's returned text (the subagent harness blocks hand-written findings files).
Generated tables: `screen/SCREEN_STUDY.md`, `SCREEN_STUDY.json`, `SCREEN_TABLE.csv`, `SCREEN_OC.csv`. Units: CTS_SCREEN_SIM_V1
(845 CPU-s), CTS_SCREEN_L0B_V1 (920 CPU-s; 3,072 new recipe fits; 120 source and 48 own cache hits), CTS_SCREEN_L2_V1 (460 CPU-s;
1,536 grid fits and 682 LOG_LR calibration / split fits). All rates hold under the stated laws only.

1. **What-if (descriptive).** NO_QUALIFYING_CONFIGURATION is unchanged under AGG1: all 18 two-seed source configurations fail AGG1.
   Even with no recipient screen at all, 12 stop at BENEFIT_NOT_STABLE and 6 at RESERVE_NOT_MET.
2. **L0a (exact aliases).** 380 income and 28 occupation alias pairs: every cell exactly 0; both screens pass all.
3. **L0b (identical releases, separately fitted readers; a true zero shift).** The old gate fails 3 / 24 occupation and 9 / 30 pair
   entries (single cells up to +0.013 / +0.023); AGG1 fails 1 / 30 pair entries (NEURAL_RAW, U +0.0106) and nothing else; income
   passes everywhere.
4. **L1 (cell-level law, one comparison).** At the boundary (+0.010) AGG1 admits at most 0.033 in every setting against the old gate's
   up to 0.137 (income, rho 0.76). On income AGG1 has less power at mu = -0.010 (0.31 vs 0.55). Three repetitions raise AGG1's power
   (income mu -0.010 to 0.43; occupation mu 0 to 0.58) while lowering the old gate's.
5. **L1, multiplicity (12 comparisons).** A candidate truly 0.010 safer everywhere passes everything with probability 0.108 (old) /
   0.032 (AGG1) at rho 0.76 and two repetitions (with the actual comparator wire structure: 0.150 / 0.045). The intersection-union
   error with one bad income comparison is 0.049 (old) / 0.008 (AGG1). Failing either screen at the configuration level is weak
   evidence about the candidate.
6. **L2 (semi-synthetic).** The LOG_LR-calibrated plants (+0.005 / 0.010 / 0.020) were read 3-30 times more strongly by the slate
   (mostly ET5 and LOOKUP winners; realized +0.015 to +0.196), so both screens rejected all 12 rows and the boundary was not probed. A
   boundary-level L2 would need a new registered law (slate-calibrated or a smaller plant); nothing post hoc was run.
