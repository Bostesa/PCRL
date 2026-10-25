# Failure map (campaign `pcrl_reader_fit_aware_release_v1`; coordinator; registered before any label read)

Written 2026-10-25 from 21:20Z from public records only. Rows A-G are the direct-recovery frontier campaign's map
(`results/pcrl_direct_recovery_frontier_v1/FAILURE_MAP.md`, published in c4896479e), which stays the reference for them; rows 1-15 of
the strong-anchor map stay the reference behind it. Rows H-J are added. No conclusion of an earlier campaign is changed here.

| # | Family or recipe | Dominant failure | Status here |
|---|---|---|---|
| A | SAR1 FA: SUCC-anchored residuals | protection profile not kept across versions; D2 not met | closed; not resumed |
| B | SAR1 FB: pair Fisher-discriminant penalties on NEURAL_RAW (FB_PFD v1: both networks, income absorbed the objective; FB_PFO R1: occupation-only, in-batch closed-form quadratic-discriminant best response) | FB_PFO +0.0125 (0.0056) over NEURAL_RAW, not separated from NEURAL_RAW's own recipe continued (ALT); sign change on the robustness split | closed: the Fisher penalty is not resumed. This campaign keeps FB_PFO's START, occupation-only loop, budgets and ALT path but replaces the in-sample closed-form best response by a held-out fitted-reader objective; the card states the operational difference and its isolating control (STOPFIT) |
| C | SAR1 FS: conditional pair-increment residual | no feasible strength | closed |
| D | NRM1 / FRC1 PRC: propensity-routed finite cells | not separated from controls | closed |
| E | FSA1 SV1 exact tilt-line solver | lowers the surrogate only | closed |
| F | SCM / IBSI shared-cell merging | not separated from task spending | closed |
| G | FRC1 / ORD SEX-blind coarsening | large task cost for small reductions | closed |
| H | DRF1 family DR: exact-AUC objective with explicit usefulness constraints in a finite lookup family (R1 / R2) | R1 met 28/28 at the point but tied the MI objective (+0.0076, se 0.0117) and the strong start SUCC; D4 not met; OPT_L useful in-sample only | closed: both revisions used; family DR is not resumed |
| I | EDG1 synthetic estimator diagnostic | no mechanism (a measurement study): same-sample / nested exact AUC* favours coarser outputs, held-out LOOKUP favours finer outputs | its findings enter the measurement contract (with the qualification of section 6 of MEASUREMENT_CONTRACT.md); empirical optimal AUC is not an objective or selector here |
| J | Online alternating critics (NEURAL_RAW's RAW-J recipe; the Adult-line online / refreshed critics of RGJ / SMF / OSF) | protection limited by update strength and by the occupation-accuracy cost; strength / allocation sweeps closed | RAW-J is continued only as the ALT control; an ordinary alternating critic of the candidate's own reader family is the ALTCRIT control; no strength sweep |

Facts the design uses (published): NEURAL_RAW's pair recovery equals its occupation view on the SAR1 grid (0.6253 = 0.6253, income view
0.5336); its point margins are 0.34 / 0.57 of the caps (men's guard v0, income Brier vs U0 v1); SAR1 FB_PFO's selected checkpoint was
S50 in both versions under the occupation budgets reused here.
