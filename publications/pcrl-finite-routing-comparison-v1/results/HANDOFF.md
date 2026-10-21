# Handoff to the manuscript owner and the claims owner

Campaign `pcrl_finite_routing_comparison_v1` (2026-10-20/21). This file is a repository handoff; no message was sent to anyone.

## What can be said

Use the statistical reviewer's required headline in `RESEARCH_DECISION.md` exactly. In short, and only under the post hoc
grid-validity repair ER1 (without it D2-D4 are `NOT_COMPUTED_GRID_INVALID` and the outcome class is `INCOMPLETE`):
- replacing the label-refit tables with out-of-fold teacher means removed about 36% (16 x 64) and 62-68% (24 x 128) of the
  label-refit twin's nested occupation-NLL spend over the continuous teachers (point ratios; the reductions resolved at 3.3-8 se),
  and the repaired 16 x 64 twin qualifies as a finite task-only release (D1);
- the propensity-aware routing was fitted and is feasible;
- its reduction in linked-pair recovery against the twin (+0.0070, se 0.0040) is not resolved and not separated from task spending on
  any matched-control path (D2);
- it is not superior to the admitted references (D3), and the 0.020 target is not met (D4);
- outcome class `ENGINEERING_REPAIR` (under ER1).

Supporting numbers: `TABLES.md` (generated), `figures/FRONTIER.png` / `.csv`, `claims/MAIN_ER1/` and `claims/R1_ER1/DECISIONS.json`,
`claims/STABILITY/`, `decisions/QUAL_DECISION.json`.

## What must not be said

- Not "propensity routing reduces SEX recovery" or "targeting helps": no comparison against a matched control is resolved, and the
  task-score-only grouping control reaches the same recovery.
- Not "probability estimation explained the lost margin" (causal) or "explains the capacity failure": the teacher-mean tables
  removed a share of one spend measure, the remainder's cause was not tested, and the original 0.004 capacity gate still fails for
  every construction.
- Not "efficiency edge" or "matched recovery at lower cost" against HQ_S2: about the same recovery (+0.0003, se 0.0051), spend
  0.0026 lower (se 0.0014, not resolved); no efficiency difference is established.
- Not "PRC beats the SEX-blind paths": the interpolated values (0.0064, 0.0068) are reported, never gating, about 1.3 se.
- Not "strongest setting" as a dose-response: the registered path produced one protected partition per version.
- Not "confirmed", "robust" or "replicated": every robustness comparison is `ROBUSTNESS_NOT_CONFIRMED`. The two versions share their
  training rows and MODEL_SELECT people. All of this is development evidence on adaptively used MODEL_SELECT records.
- Not "the refit was genuinely effective", "seeds effective", "effective under ER1" or "24 independent readers": the frozen rule
  says `REFIT_NOT_GENUINELY_EFFECTIVE`. Use the statistical reviewer's wording in `roles/statistical_reviewer/CLOSING_STATEMENT.md`.
- Not "the S3 budget was tested": S2 and S3 selected the same setting, and the label-free probe shows the registered optimizer cannot
  move further in version 0.
- Not "equivalent to the controls": "not separated" with standard errors of 0.004-0.008 (20-40% of the 0.020 target) is not
  equivalence.

## Records to cite

- **Decisions.** `claims/MAIN_ER1/` (registered untargeted path [SHUF_S2]) and `claims/R1_ER1/` (revision R1 path [SHUF_S2, UNT_BR])
  give identical labels.
- **Strict records, kept.** `claims/MAIN/` and `claims/R1/` hold the strict pre-ER1 records (`NOT_COMPUTED_GRID_INVALID`).
- **ER1.** ER1 is the statistical reviewer's post hoc engineering repair of grid validity, decided before any decision value was
  read. Its text is in `roles/statistical_reviewer/RULES_ER1.md` (moved out of the registration files, deviation 9).
- **Replay.** The independent verifier replayed both readings, the robustness grid and the stability record with separate code
  (`VERIFICATION.md`).

## Open items

- **Off-device custody is pending.** The owner's drive (YOTUO) was not attached during the campaign (checked 22:56Z and 00:59:11Z).
  Only same-device custody exists (`custody/BACKUP_VERIFICATION.json`). When the drive is attached, the safe copy and check is:
  `python3 -m frc.custody offdevice <backup dir> <external drive>`
  A check right after copying is warm. A cold re-check needs an ordinary eject and re-attach first. Nothing is mounted, unmounted or
  forced by the tool.
- **Untested routes** (`RESEARCH_DECISION.md`): the S3-budget region with another optimizer; joint design (the separate design and
  both sequential orders); the prospective 32 x 256 capacity; randomized finite releases; readers trained on more data; repeated
  queries.
- **Rule note for a future campaign.** On the few-valued views of finite releases (here 8-16 income and 32-64 occupation values),
  extra-trees readers reduce to seed-invariant frequency tables. A strict reader-seed rule therefore cannot certify a refit of such
  releases unless those recipes are counted once (`claims/STABILITY/REFIT_EQUALITIES.json`).
- **Erratum to carry.** The ER1 text (`roles/statistical_reviewer/RULES_ER1.md`, frozen) says "16 income, 64 occupation values";
  that is exact for PRC_S2, TWIN, HQ_S2, SHUF_S2 and UNT_BR only (ORD_S2 has 12 / 48, ORD_S3 8 / 32). No rule is affected
  (`CLOSING_STATEMENT.md`, C12).
