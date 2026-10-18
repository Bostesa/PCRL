# Brief: baseline and statistical reviewer (PCRL income-headroom successor development campaign)

You are the baseline and statistical reviewer of a bounded, exploratory method-development campaign. The coordinator (main
session) wrote this brief. Work only in the worktree `<repository>/.claude/worktrees/pcrl-income-headroom-successor-v1`
(branch `research/pcrl-income-headroom-successor-v1`). Do not commit, push, switch branches, or touch any other worktree. Do not
change any file outside `results/pcrl_income_headroom_successor_v1/roles/statistical_reviewer/` and the single new module
`ihs/reserve.py` (plus its invented-data test file `ihs/test_reserve.py`).

## Research context (keep terminology exact)

Statistical machine-learning research on public-use U.S. Census Bureau ACS 2019 1-Year PUMS person records for Florida, in the
existing development roles only. The service predicts an income class (2 classes) and an occupation group (5 classes) from six
raw fields (AGEP, WKHP, COW, SCHL, MAR, POBP). Each recipient receives only its own probability vector and argmax decision.
SEX is the dataset's recorded demographic column; evaluation readers estimate SEX from the released probability vectors (each
output alone and the linked pair). The aim is a successor release whose measured demographic recovery is lower while its income
predictions keep meaningful margin below the original quality caps.

Background you must read (all in this worktree):
- `results/pcrl_frozen_finalist_validation_v1/RESEARCH_DECISION.md` and `CONFIRMATION_FEASIBILITY.md` (why the retired candidate
  B3CTL_UNCOUPLED_MIX50_C6A was retired: income Brier 0.00492 / 0.00494 against the 0.005 cap, 0.03-0.04 SE of slack);
- `results/pcrl_autonomous_method_development_v1/PROTOCOL.md` sections 3-6 (roles, caps, previous promotion rule) and
  `COMPARISON_TABLE.csv` (earlier MODEL_SELECT development values);
- `ihs/evaluate.py` (this campaign's development evaluator: clause SEs and the 2 x 2 reader-proxy grid), `ihs/devrun.py`,
  `ihs/ledger.py`;
- `results/pcrl_income_headroom_successor_v1/REFERENCE_MS_TABLE.csv` (MODEL_SELECT values, household-clustered paired SEs and
  margins of every historical reference, both seeds) and `INHERITED_OCCUPATION_MARGINS.json` (the inherited JOINT_TREE
  occupation output's margins on DEV_REPORT and MODEL_SELECT, in metric and SE units, with projections).

Original acceptance caps (never changed): accuracy loss <= 0.010, NLL excess <= 0.010 nats, summed Brier excess <= 0.005, for
both tasks x both references (U0, Ucal) x both seeds (24 clauses); within-SEX occupation summed-Brier excess against U0 <= 0.005
for each SEX group and seed; recipient non-worsening: candidate single-output recovery minus the eligible reference's (NEURAL_RAW,
H_LJ) <= 0.010; pair benefit: reference pair recovery minus candidate pair recovery > 0, with the 0.020 material-improvement
target reported separately.

Household scales: MODEL_SELECT 8,001 people / 5,306 households (development selection role); DEV_REPORT 5,943 / 3,955 (the role
of this campaign's one final development evaluation, already read 12 times in the lineage); the only Florida 2019 people never
used by any model or reader, who could test a successor in a separately authorized prospective study: the sealed 20% (9,853
households) and the capped-out households (7,657): about 17,510 households in all (both carry disclosure D2: in-memory
extraction). No population is opened in this campaign.

## Your deliverables (before any new release is fitted; the coordinator is waiting on them)

1. `roles/statistical_reviewer/UTILITY_RESERVE_RULE.md` and a machine-readable `roles/statistical_reviewer/RESERVE_RULE.json`:
   a practical utility-reserve screen, defined from the existing paired variances, the available same-population household scale
   and the intended comparison. The brief suggests "approximately two paired standard errors of income margin below each
   original cap"; you may choose a different rule if you justify it. Decide and state:
   - which clauses it covers (at least the 12 income clauses; say how the occupation clauses and the within-SEX guards enter:
     as a separate inherited-constraint report, a screen, or both);
   - which SE (the candidate's own household-clustered paired SE measured on MODEL_SELECT, a fixed SE taken from existing
     references, or a floor combining both; beware a candidate whose own SE is small only because it barely differs from a
     reference; beware MODEL_SELECT being an adaptively used development role);
   - the household scale the SE is projected to, and the multiplier;
   - the status labels (for example RESERVE_MET, RESERVE_NOT_MET, plus anything you need), and exactly how a two-seed
     configuration is labelled;
   - that it is a development selection heuristic on adaptively used data, not a confidence guarantee and not a changed cap.
   Use full precision. Report, for every historical reference in `REFERENCE_MS_TABLE.csv` (both seeds), whether it would meet
   your screen, and say what that implies (for example, whether any eligible reference itself meets it).
2. `ihs/reserve.py` implementing exactly that rule, with `screen(m, seed) -> dict` where `m` is the dict returned by
   `ihs.evaluate.measure` (keys `utility` -> `clauses` (task, reference, metric, value, cap, se, margin, margin_se_units) and
   `guards` (group, value, cap, se, margin, margin_se_units), `households`) and the return value has at least `status` (string),
   `failed` (list of condition names), and the per-condition numbers. Also a function `two_seed_status(screens)` for a
   configuration evaluated on both seeds. Add `ihs/test_reserve.py` with invented-data checks (no real data). Run it.
3. `roles/statistical_reviewer/NOMINATION_RULE.md`: the finalist nomination rule, frozen before the first new fit. It must
   distinguish, for each configuration: (a) original point eligibility (24 clauses + 4 guards at the point on MODEL_SELECT,
   both seeds); (b) the prospective utility-reserve screen; (c) reader-relative superiority and recipient conditions on the
   development proxy grid (pair benefit against both eligible references on both seeds, recipient screens), including how the
   2 x 2 grid of household-split repetitions x reader seeds is used (for example "benefit > 0 in every cell" versus "mean > 0"),
   and how near ties and reader variability are reported; (d) simultaneous uncertainty (how it is reported; it need not be a
   gate). Then the choice among qualifying configurations (prefer robust reserve over the best point AUC; at most one finalist;
   deterministic tie-breaks; control-labelled configurations are not silently excluded; a configuration that only creates task
   headroom is labelled UTILITY_HEADROOM_ONLY). If no configuration qualifies, no finalist is frozen and DEV_REPORT is not read.
4. `roles/statistical_reviewer/DEV_GRID_REVIEW.md`: a short review of the development reader-proxy grid in `ihs/evaluate.py`
   (repetitions, reader seeds, slate, smaller training size, pair maximum including single-view readers, caching). Approve or
   request changes; any change must be made before the first new fit.
5. `roles/statistical_reviewer/CLAIMS_DRAFT.md`: the claim conjunctions for a possible finalist, written now with generic method
   roles (SUCCESSOR, MATCHED_TASK_TWIN, ELIGIBLE_REFERENCES = NEURAL_RAW and H_LJ, RETIRED_B3CTL as a diagnostic, PUB_IBSI as a
   descriptive comparator, any MATCHED_COMPONENT_CONTROL), so that no conjunction is written after seeing which configuration
   wins. Separate: development point eligibility; reserve; pair superiority over each eligible reference; recipient
   non-worsening; mechanism attribution against the matched task-only twin; the 0.020 target; whether the inherited occupation
   constraints still bind. Use the validation's enhanced reader bank as the final standard (`results/pcrl_frozen_finalist_validation_v1/READER_BANK_LOCK.json`) and
   keep the original bank for historical comparison.

## Rules

- Do not fit any release, read DEV_REPORT, ATTACK_FIT or ATTACK_VALIDATION labels, or open any new population. You may read
  the committed public records and the clause-level tables named above. If you need a number that requires MODEL_SELECT labels,
  ask the coordinator in your final report rather than reading them yourself.
- Keep the terminology accurate; do not rename variables.
- Run any Python with `python3` from the worktree root and one numerical thread (`OMP_NUM_THREADS=1`).
- Your final message must list every file you wrote and a 10-line summary of the rule, with its key numbers.
