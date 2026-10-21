# Research decision: finite-routing comparison (campaign `pcrl_finite_routing_comparison_v1`)

Florida ACS 2019 development cohort (admitted public-use PUMS records), 2026-10-20/21. This continues family A (propensity-routed
finite cells, card FA_PRC_v1) of `pcrl_new_release_mechanisms_v1`, which closed at its capacity stage before any protected routing was
fitted. Development evidence on adaptively used MODEL_SELECT records, for the declared readers only. Lower recovery for these readers
is not protection against every reader.

## Decision

**Outcome class: `ENGINEERING_REPAIR`, under ER1.** ER1 is the post hoc grid-validity repair (`roles/statistical_reviewer/RULES_ER1.md`).
The outcome class is identical under the registered and revision-R1 readings. Under the frozen strict rule, D2-D4 are
`NOT_COMPUTED_GRID_INVALID` and the outcome class is `INCOMPLETE` (`claims/MAIN`, `claims/R1`). The independent verifier replayed
the grid, both ER1 readings and the stability record with separate code. It did not recompute the P1 person-level numbers (spends,
SEs, AUCs, the F4 bootstrap SE). There is a repaired task-only finite release, and the protected routing was actually fitted. There
is no candidate, no useful component, no joint-design result and no novelty claim.

| Decision | Label (both readings, under ER1) | In one line |
|---|---|---|
| D1 repaired task-only release | `REPAIRED_TASK_ONLY_RELEASE` | The 16 x 64 twin with out-of-fold teacher-mean tables (B1) meets the original limits with S2 slack (its two-SE reserve, reported and never gating, is 3.15 / 2.71). The label-refit version (A) does not (version-1 men's guard 0.00255 > 0.002). At 24 x 128 both qualify. |
| D2 component beyond matched controls | `NO_REDUCTION_BEYOND_ANY_CONTROL` | PRC_S2 lowers pair recovery against the twin by 0.0070 (se 0.0040, not resolved) and is not separated from task spending on the HQ, untargeted or ORD paths. |
| D3 improvement over the admitted references | `NOT_SUPERIOR` | Higher point-estimate pair AUC than six of the nine references (resolved for four: H_LJ, B2TWIN, RETIRED_B3CTL, SUCC); no resolved benefit over the other three. |
| D4 0.020 material benefit | `NOT_MET` | 0.0177 / 0.0233 pair AUC above NEURAL_RAW / H_LJ on the registered grid. |
| D5 joint design | `NOT_ASSESSED_MISSING_CONTROLS` | The rule needs the separate design and both sequential orders. They were not built, and the ablations are not that control set. |
| D6 novelty | `NOT_CLAIMED` | Not claimed by rule; the objective has named prior art (privacy funnel, IBSI, FARE). |

**Headline** (the statistical reviewer's required wording, `roles/statistical_reviewer/CLOSING_STATEMENT.md`):

> Development evidence on adaptively used MODEL_SELECT records, for the declared readers only. Under the post hoc grid-validity
> repair ER1, the outcome class is `ENGINEERING_REPAIR`: a repaired task-only finite release, and no protected-routing benefit
> beyond matched controls. Without ER1, D2-D4 are `NOT_COMPUTED_GRID_INVALID` and the outcome class is `INCOMPLETE`. On identical
> cell partitions, replacing family A's label-refit cell tables with out-of-fold teacher means lowered nested held-out occupation
> NLL by 0.0039-0.0049 nats at 16 x 64 cells and 0.0085-0.0090 nats at 24 x 128 cells (each resolved). That is about 36% and
> 62-68% of the label-refit twin's occupation-NLL spend over the continuous task teachers (point ratios). The original 0.004
> capacity gate still fails at both capacities. With the repaired tables, the 16 x 64 twin qualifies under the original limits
> with S2 slack (D1 `REPAIRED_TASK_ONLY_RELEASE`). The propensity-aware routing was then fitted. The registered strength path
> produced one protected partition per version (strength 8, the top of the grid). That partition re-routes 13-14% of MODEL_SELECT
> people and lowers registered-grid linked-pair AUC from 0.6555 (twin) to 0.6485 (benefit +0.0070, se 0.0040, not resolved). A
> grouping control that sees only task scores reaches 0.6489, and untargeted re-routing at PRC_S2's spend reaches 0.6516 (revision
> R1 path). On every control path the reduction is not separated from task spending (D2 `NO_REDUCTION_BEYOND_ANY_CONTROL`). The
> routing is not superior to the admitted references (D3 `NOT_SUPERIOR`), and on the registered grid it stays 0.018-0.023 pair AUC
> above both target references (D4 `NOT_MET`). No comparison is confirmed on the 32 robustness reader cells (24 new, 26-29
> distinct). No component, joint-design or novelty claim is made.

## What decided it

| Step | What was done | Decisive evidence | Outcome |
|---|---|---|---|
| T1-T3: probability estimation (Q-PE) | On the saved twin partitions (16 x 64, 24 x 128; identical cells), three table constructions were compared: A = label-refit decoders (the source); B1 = means of the out-of-fold teacher predictions in each routed cell; B0 = the stored prototypes | Nested held-out occupation NLL B1 - A: -0.00388 (SE_used 0.00083) / -0.00486 (0.00101) at 16 x 64; -0.00848 (0.00131) / -0.00900 (0.00113) at 24 x 128. MODEL_SELECT: -0.00374 / -0.00546 and -0.00722 / -0.00618 (household SE 0.0011-0.0016). B0 and B1 agree within 0.000005 nats of held-out occupation NLL. Occupation spend over the continuous teachers, A / B1: 0.0107 / 0.0068 and 0.0136 / 0.0088 (64 cells); 0.0137 / 0.0052 and 0.0132 / 0.0042 (128) | `PE_ERROR_SUPPORTED` at both capacities; construction B1; the original gate (estimate + 2 SE <= 0.004) fails for every construction |
| QUAL (predefined; Q1 + Q2) | Original limits (24 clauses, 4 guards, against U0 and Ucal) plus the S2-budget slack | 16 x 64: B1 qualifies in both versions (version-1 men's guard 0.00019); A does not (0.00255 > 0.002 after the S2 increment). 24 x 128: both qualify. B1 twin: worst clause / cap -0.164 / 0.037 (reserve, never gating: 3.15 / 2.71; A: 2.31 / 1.29) | `QUALIFIED`, protection at 16 x 64 with B1; D1 `REPAIRED_TASK_ONLY_RELEASE` |
| P1-P2: protected routing | PRC objective: task cross-entropy + strength x (pair + per-recipient plug-in SEX information) + a per-SEX occupation guard. Damped, rule-consistent Lloyd chain over strengths 0.25-8. B1 tables recomputed on each configuration's own out-of-fold partition. Budgets S1-S3 chosen on MODEL_FIT nested estimates (estimate + 2 SE) | S2 and S3 both select strength 8, the top of the grid; the S2 budget did not bind. S1 is infeasible in version 0 and an exact alias of S2 in version 1. PRC_S2 re-routes 14.4% / 12.8% of MODEL_SELECT people relative to the twin (income 4.0% / 3.7%, occupation 11.3% / 10.1%). Fitted pi pair MI 0.0332 / 0.0327 (twin 0.0357 / 0.0358). Nested held-out LOOKUP AUC 0.6227 / 0.6206 (twin 0.6298 / 0.6318). F4: 0.00715 (se 0.00225) / 0.01113 (0.00231) | Protected routing fitted and feasible: PRC_S2 meets the original limits in both versions (worst clause / cap -0.180 / 0.180); F4 passes. One protected partition per version |
| D2: matched controls (registered grid FRC1_MAIN, 8 cells) | Controls: the twin; HQ (routing on task scores only, SEX information only as cell means); SHUF (donor propensity); R1 UNT (SEX-blind keyed second-nearest moves at PRC's spend); ORD (SEX-blind coarsening) | Pair AUC: TWIN 0.6555, PRC_S2 0.6485, HQ_S2 0.6489, SHUF_S2 0.6574, UNT_BR 0.6516, ORD_S2 0.6538, ORD_S3 0.6474. Benefit of PRC_S2 at each bracketing control: HQ_S2 +0.0003 (se 0.0051), UNT_BR +0.0031 (0.0076, R1 reading), ORD_S2 +0.0053 (0.0059). The registered SHUF path never reaches PRC's spend | `NO_REDUCTION_BEYOND_ANY_CONTROL`: every frontier `NOT_SEPARATED_FROM_TASK_SPENDING`; reader targeting fails against HQ_S2 and UNT_BR |
| D3-D4: original benchmark | Nine admitted references | PRC_S2 vs NEURAL_RAW -0.0177 (se 0.0117), H_LJ -0.0233 (0.0087), TWJ -0.0107 (0.0080), SUCC -0.0345, B2TWIN -0.0320, RETIRED_B3CTL -0.0458; vs W +0.0020, TR100 +0.0110, TT100 +0.0065 (point only) | `NOT_SUPERIOR`; `NOT_MET` |
| Classifier stability (descriptive; never a decision input) | Robustness readers (FRC1_ROB: the 8 registered cells plus 24 new cells: repetitions R3/R4 and reader seeds k2/k3; 26-29 of the 32 cells distinct). Ablations at PRC_S2's strength | All 32 cells: vs TWIN +0.0069 (0.0040); vs HQ_S2 +0.0005 (0.0062; 13 of 27 distinct cells opposite); vs UNT_BR +0.0031 (0.0077; versions opposite); vs H_LJ -0.0206 (0.0089, resolved worse). Ablations against PRC_S2: ABL_LOC (pair weight 0, per-recipient terms only) -0.0024 (se 0.0031) and ABL_PAIR (local weights 0, pair term only) -0.0032 (0.0042) on the 8 registered cells; -0.0006 (0.0035) and -0.0022 (0.0038) over all 32 cells | Every comparison `ROBUSTNESS_NOT_CONFIRMED`; no component separable |
| Refit effectiveness | frc.rules strict rule (frozen; ER1 deliberately not extended to it) | The new splits pass (four distinct assignments, SCORE overlap 0.30-0.31). The strict seed rule fails with 240 seed-dependent equalities on the minimum set (384 in all) | `REFIT_NOT_GENUINELY_EFFECTIVE` (reason `READER_SEEDS_NOT_EFFECTIVE`) |
| Strength probe (label-free; decision rule registered before the run) | The full-pool chain continued to strengths 16-128 | Version 0: one partition at every strength from 0.25 to 128. Version 1: one partition from 0.25 to 8, and a move at 16 (39.3% re-routed, fitted MI 0.03224 vs 0.03273) | No revision R2 (version 0 saturated); the S3 budget point stays untested |

Descriptively (`frc/refitcheck.py`, not a rule outcome), every seed-dependent tie in the refit bank (384 of 2,880 checks) is an
extra-trees reader on one of the eight finite releases in that bank: ET5 or ET25 on the income view, or ET5 on the occupation view.
These views have 12 or 16 income values and 48 or 64 occupation values, and on them the readers reduce to seed-invariant frequency
tables (the ER1 mechanism). Each tie meets the NRM1 (iii) explanation. Every other seed-dependent check in the bank (2,496 of 2,880)
differed between its two reader seeds. For those extra-trees recipes, the k2 / k3 cells duplicate the k0 / k1 cells, and the
robustness screens count them once (26-29 distinct of 32 cells). The robustness evidence therefore varies the split fully but the
reader seed only partly.

Numbers: `TABLES.md` (generated by `frc.tables`), `claims/MAIN_ER1/` and `claims/R1_ER1/DECISIONS.json`, `claims/STABILITY/`,
`decisions/QUAL_DECISION.json`, `fa/*.json`, `figures/FRONTIER.png`.

## Original benchmark or narrower comparison?

**Neither.** On the original benchmark (the nine admitted references), PRC_S2 is `NOT_SUPERIOR` and the 0.020 target is `NOT_MET`.
Its point-estimate pair AUC is above both target references (NEURAL_RAW 0.6309, H_LJ 0.6252); the difference is resolved for H_LJ.
In the narrower within-family comparison, its reduction is not separated from task spending on any matched-control path. The only
resolved favourable results are on the task side: the probability repair (Q-PE) and the repaired twin's qualification (D1).

## What is closed and what stays open

**Closed by this campaign's rules:**
- the probability-estimation question at the two existing capacities;
- the task-only repair (D1);
- protected routing at the registered budget path. The targeted-versus-untargeted question is closed for this campaign at the
  registered optimizer and budget path (R1 stopping rule);
- the second revision: not made, by the probe rule registered before the probe ran.

**Untested:**
- whether the occupation-NLL spend that remains after B1 is the partition's own cost or remaining per-cell estimation error (other
  per-cell estimators were not tried);
- a traced protected-routing frontier: the registered path realized one protected partition per version. More points need an
  optimizer that moves below strength 16. The S3 budget point is unreachable in version 0 with the registered optimizer, whose
  smallest rule tilt grows with the strength; version 1 moves only at 16, re-routing 39% of people;
- the registered-reading untargeted bracket: no SHUF point reaches PRC_S2's spend;
- an effective-reader-seed robustness check (ER1 was deliberately not extended to the refit);
- other optimizers for the same objective;
- the prospective 32 x 256 capacity (not needed, because 16 x 64 qualified; not run);
- joint design (D5: the separate design and both sequential orders were never built);
- randomized finite releases;
- readers trained on more data;
- repeated queries;
- other cohorts.

## What the evidence suggests for the next decision (descriptive, not a rule outcome)

- **Probability estimation.** Replacing the label-refit tables with out-of-fold teacher means removed about 36% (16 x 64) and 62-68%
  (24 x 128) of the label-refit twin's nested held-out occupation-NLL spend over the continuous task teachers. These are point ratios
  without intervals; the NLL reductions themselves are resolved (3.3-8 se). The remainder (0.0068 / 0.0088 nats at 16 x 64, 0.0052 /
  0.0042 at 24 x 128) was not reduced by either teacher-mean construction; B0 and B1 agree within 0.000005 nats. Whether it is the
  partition's own cost or remaining per-cell estimation error was not tested.
- **PRC_S2 against HQ_S2.** The two reached about the same recovery (benefit +0.0003, se 0.0051) and the same fitted pair MI (0.0332 /
  0.0327 vs 0.0332 / 0.0328). PRC_S2's primary task spend was 0.0026 nats lower (se 0.0014, not resolved). It re-routed fewer people:
  14% vs 33% in version 0, 13% vs 24% in version 1; re-routing is not a registered cost. No efficiency difference is established.
- **Interpolated SEX-blind paths.** Interpolated linearly at PRC_S2's spend (reported, never gating), the ORD path sits 0.0064
  (se 0.0047) above PRC_S2 in both readings. The untargeted path sits 0.0068 (se 0.0054) above it in the revision-R1 reading only; the
  registered untargeted path never reaches PRC_S2's spend. Both are point estimates of about 1.3 se. The gating bracket benefits are
  +0.0053 (se 0.0059) and +0.0031 (se 0.0076).
- **Precision.** Standard errors of 0.004-0.008 are 20-40% of the 0.020 target, and the minimum detectable pair benefit in the D2
  record is about 0.015 (2 se_agg). Differences of this size can be neither excluded nor established with these people.
- **Distance to the material criterion.** D4 asks for a further 0.020 beyond parity with the target references. PRC_S2's shortfall
  from that criterion is 0.038 (NEURAL_RAW) and 0.043 (H_LJ), 5-6 times the reduction it achieved against its own twin (0.0070).

## Limits of this evidence

- **ER1 dependence.** Every D2-D4 label exists only under the post hoc repair ER1. Under the strict rule the outcome class is
  `INCOMPLETE`.
- **One protected point per version.** S2 and S3 both select strength 8, and the S2 budget did not bind. In the label-free full-pool
  chain, every registered strength from 0.25 to 8 gives the same partition in both versions. D2 therefore compares one protected
  partition per version with the controls, not a traced frontier, and nothing here is a dose-response.
- **Adaptive use.** MODEL_SELECT was used adaptively by this and every earlier campaign. R1 was decided after the protected fits, on
  MODEL_FIT evidence only and before any MODEL_SELECT read of a protected release. ER1 was decided after the first claims run and
  before any decision value was read (`PROTOCOL_DEVIATIONS.md` items 5, 6, 9).
- **Reader seeds.** The robustness bank's reader seeds were partly ineffective: the strict refit label applies, and 26-29 of the 32
  cells are distinct.
- **Detectability and multiplicity.** The minimum detectable pair benefit is about 0.015. The observed control differences
  (0.0003-0.0070) are below it, so "not separated" is the default reading, not evidence of equality. Many comparisons were screened
  at 2 se without adjustment; only the Q-PE reductions (3.3-8 se) are clear of this.
- **Data-limited readers.** Every release's pair AUC still rises with reader-FIT data (`TABLES.md` section 9). Measured recoveries
  are lower bounds for better-trained readers, and the PRC_S2 / HQ_S2 ordering changes sign across FIT fractions.
- **Verifier scope.** The P1 person-level spends, SEs, AUCs and the F4 bootstrap SE were not independently recomputed.
- **H_LJ.** Over all 32 cells, PRC_S2 is resolved worse than H_LJ (upper bound -0.0028). There, `ROBUSTNESS_NOT_CONFIRMED` means no
  benefit, not an unstable difference.
- **Rules hash.** The claims records cite the post-ER1 RULES.json hash e2a6af23... (kept at commit 6d6cd0829), not the restored
  registration file (deviation 9).
- **Scope.** One released instance per person; the declared reader grid only; six admitted raw fields; two recipients; one cohort.
  The two versions share every training row and the MODEL_SELECT people (procedure repeats, not replications).

Related records: `ADVISOR_BRIEF.md`, `START_HERE.md`, `VERIFICATION.md`, `PROTOCOL_DEVIATIONS.md`,
`roles/statistical_reviewer/CLOSING_STATEMENT.md`, `roles/coordinator/COORDINATOR_LOG.md`.
