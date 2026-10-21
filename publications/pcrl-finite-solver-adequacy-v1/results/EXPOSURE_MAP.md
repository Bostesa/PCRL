# Exposure map (campaign `pcrl_finite_solver_adequacy_v1`)

Written 2026-10-21 at 02:41-02:42Z (`date -u` read 02:38:37Z before and 02:42:35Z after; an earlier draft said 02:46Z in error). It records which data each stage reads and where SEX can reach a release. Every label
read goes through `fsa.common.load` (logged); features through `fsa.common.features` (label-free).

| Stage | Role | Fields read | SEX reaches | Card |
|---|---|---|---|---|
| exact references | none (invented laws) | none | - | `roles/exact_reference/LAWS.md` |
| replay (P1, unchanged) | MODEL_FIT | features, income, occupation, SEX | the propensity and SEX-conditional occupation teachers of each outer fold (training label), held-out LOOKUP (evaluation label); exactly as in the original P1 | FA_FSA_v2 |
| audit, source | MODEL_FIT (features only) + cached out-of-fold statistics | no label file; the cached statistics are derived from MODEL_FIT income, occupation and SEX labels | the cached propensity and SEX-conditional teacher predictions enter the objective and the PRC routing, and through cell means the HQ routing | FA_FSA_v2 |
| nested (both solvers) | MODEL_FIT | features, income, occupation, SEX | as replay | FA_FSA_v2 |
| pilot fits (only if the condition is met) | MODEL_FIT, cached statistics | as the pilot card states | PRC: person-level propensity and SEX-conditional occupation statistics in the routing rule; HQ: only their twin-cell means; untargeted control and twin: none | pilot card |
| pilot evaluation | MODEL_SELECT | income, occupation, SEX (benchmark utility, paired costs; never a fitting input) | evaluation only | - |
| reader grid | MODEL_SELECT | SEX (FIT / VALIDATION / SCORE reader roles) | evaluation only | - |
| deployment parity | DEV_REPORT | features and raw inputs only, label-free | none | - |
| verification | MODEL_FIT, CALIBRATION, MODEL_SELECT | as needed for replay, with separate code | evaluation only | - |

- Closed throughout: ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT labels, the sealed assessment, capped-out people. There are no
  downloads or new populations.
- CALIBRATION has no SEX label, and no stage of this campaign plans to read CALIBRATION labels.
- At runtime a release reads six fields and emits two probability vectors with their decisions, nothing else. SEX is never a runtime
  input or output.
