# Protocol: autonomous development of a different release mechanism (PCRL, 2026-10-20)

Campaign `pcrl_new_release_mechanisms_v1`. Research branch `research/pcrl-new-release-mechanisms-v1` (local; its history contains
private paths). Public record: publication branch `publication/pcrl-new-release-mechanisms-v1` on https://github.com/Bostesa/PCRL.
Campaign start 2026-10-20T14:29:43Z (first action, `date -u`). This protocol, the access policy, the exposure map, the source
admission, the frozen reader splits, the failure map and the statistical reviewer's generic claims and selection rules are
committed and published as a sanitized receipt, verified on the GitHub remote, before any new real-label computation or fit.

## 1. Setting and terminology

- **Data.** Statistical machine-learning research on already admitted public-use U.S. Census Bureau ACS 2019 1-Year PUMS person
  records for Florida: the existing development cohort, its household groups and the roles MODEL_FIT (20,000 people, 13,337
  households), CALIBRATION (2,982 people) and MODEL_SELECT (8,001 people, 5,306 households).
- **Tasks.** Income (2 classes) and occupation (5 official broad groups), with the exact definitions of the pinned manifests.
- **Service and adversary.** The service receives six fields (AGEP, WKHP, COW, SCHL, MAR, POBP). Each recipient receives only its
  own float64 probability vector and argmax decision. Separate evaluation classifiers ("readers") estimate the recorded SEX column
  from those released outputs. "Recovery" means their AUC.
- **Scope of claims.** Lower recovery for the declared development readers is never protection against every reader.
- **Scenario.** The service needs probabilities for recipient-specific thresholds, ranking or deferral. This is an illustrative
  benchmark scenario, not a documented deployment.
- **Confirmation.** No new population, microdata download, sealed assessment or capped-out people are used. ATTACK_FIT,
  ATTACK_VALIDATION and DEV_REPORT labels stay closed throughout. MODEL_SELECT was used adaptively by every earlier campaign: all
  evidence here is development evidence, never confirmation.

## 2. Adversary (declared)

- **Access.** A recipient holds its own released vector and decision for each person it serves. The linked adversary pools both
  recipients' outputs, aligned person by person (perfect linkage). Readers never see raw fields, internal codes or cell ids,
  hidden scores, or any label other than SEX. Any cell identity recoverable from the actual outputs is part of the disclosure
  analysis.
- **Auxiliary sample.** A demographic-labelled sample of people with their released outputs and SEX. In this development grid it
  is the reader-FIT split of MODEL_SELECT (about 3,990 people, 2,655 households per repetition), with reader-VALIDATION (about 1,600
  people) for recipe selection; SCORE (about 2,400 people) is never used for selection. This is smaller than the original full
  auxiliary budget, and nothing here implies otherwise.
- **Fitting budget.** The 16 locked recipes of the inherited grid (OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB,
  LOOKUP, ET5, ET25, HGB15, HGB31, MLP64, MLP6432, CATI01, CATI1), with identical sizes and effort for every release.
  - Pair candidates always include the ignore-income and ignore-occupation alternatives.
  - LOOKUP is a smoothed frequency table on exact output vectors, the reader for identifiable finite output cells.
  - Readers needed to interpret a proposed release (e.g. channel-aware training draws) are added before candidate outcomes and
    given to every applicable control with the same effort.
- **One release per person.** Each target person contributes exactly one emitted vector per recipient (one draw for a randomized
  release). The default auxiliary access is also one draw per person. A multi-draw auxiliary variant is used only if declared
  before outcomes. Repeated-query protection is not claimed.

## 3. Sources and preserved verdicts

- **Latest source.** Publication ed7bc1ccd (`publications/pcrl-controlled-training-size-v1`), verified on the remote, and its
  admission chain (SOURCE_ADMISSION.json).
- **Every earlier verdict stands unchanged:**
  - SUPPORTED_ON_THIS_POOL / CONTROL_EXPLAINS_GAIN and CORRECTION_RECIPE_CLOSED (controlled training size);
  - NO_RECIPE (correction transfer);
  - NO_QUALIFYING_CONFIGURATION (occupation headroom);
  - H3's income limitations.
- **Closed lines.** Leaf-score correction, correction controllers, temperature and nearby strength tuning are closed as the main
  research line.
- **H3 is optional.** It is a benchmark, not a compulsory component. Both task heads may be trained from scratch.

## 4. Roles and resources

- **Assignments.** About seven to ten, in waves. At most six live agents including the coordinator:
  - coordinator / integration owner;
  - two independent mechanism designers and implementers (A, B);
  - a challenger (failure map, simpler explanations, strong controls);
  - statistical reviewer;
  - reader-robustness investigator;
  - independent verifier (separate numerical code);
  - engineering / claims reviewer.

  Agent reports returned as text are saved verbatim with attribution.
- **Budget.**
  - 8 elapsed research hours (to 22:29:43Z) plus 30 minutes of closeout (to 22:59:43Z).
  - 14 CPU-hours in total, including failed attempts, reader fits and agents' numerical work.
  - At most two heavy jobs at once, one numerical thread each, through the one shared queue `nrm.jobqueue`. It hashes the
    importable code at launch, keeps an append-only ledger and refuses work past the CPU or time limits.
  - No paid or cloud compute.
  - Up to 60 new seed- or training-repetition-specific release configurations, fitted matched controls included (counted by
    `nrm.release.save`). Internal cross-fitting, critic, calibration and reader fits are recorded separately.
- **Phases.**
  - First hour: source admission, failure map, proposals, this receipt.
  - Next three hours: prototypes, task feasibility, justified revisions.
  - Next two hours: matched protection comparisons and robustness.
  - Final two hours: decisive ablations, independent replay, deployment and closeout.

## 5. Mechanism development

1. **Failure map** (FAILURE_MAP.md, in this receipt) from the campaign ledger, before any new fit.
2. **Independent proposals.** Two designers propose families independently before comparing them. The challenger checks each for
   operational equivalence with earlier attempts and names the simplest controls. For each family the card states:
   - the exact operation that changes the transmitted output;
   - its closest prior PCRL attempt and what is substantively different;
   - the measured bottleneck it addresses, and why it could improve protection without repeating the observed utility cost;
   - the simplest control that could explain a result, and the falsifying outcome.

   A different name, optimizer, penalty weight, code size or classifier is not automatically a different mechanism. This check
   prevents repetition; it is not a novelty gate. At most three materially different families are allowed.
3. **Card before fit.** Each family's mechanism card (and every later revision card) is committed under `cards/` before the fit it
   governs. The access layer refuses fitting reads without a committed, unchanged card.
4. **Per family**, in order:
   - a minimal raw-field-to-release path, checked on invented data;
   - an equally trained task-only twin with the same architecture and adequate capacity (if it fails utility, the architecture is
     improved or replaced before any protection conclusion);
   - the protection operation along a small predeclared strength path;
   - scoring of the actual transmitted outputs;
   - diagnosis of the dominant failure;
   - then a revision, a replacement, or a stop.
5. **Revisions.** Up to two substantive revisions per family, plus one reviewed replacement of a falsified family, within the
   global configuration budget. The coordinator and the statistical reviewer approve each revision in writing before its fit. No
   amendment changes already-tested hypotheses or old verdicts.
6. **Training exposure.** Release parameters are fitted on the full MODEL_FIT pool where possible; the pool is not halved to create
   diagnostic panels. Calibration is fitted only on CALIBRATION or on correctly separated internal training folds. Out-of-fold
   predictions use household grouping and exclude the checked households from every affected upstream component. Nested and
   non-nested calculations are distinguished, the submodels' training sizes are stated, and their costs are never a guarantee for
   the final full-data model.
7. **MODEL_SELECT** serves logged adaptive model selection and diagnosis only, through `nrm.evaluate` (utility) and `nrm.screen`
   (screening readers on the registered screening halves). No gradient, calibration parameter or continuous release parameter is
   fitted on its labels. A later revision does not become fresh because a new reader split is used.
8. **Controls.** Every family gets:
   - its equally trained task-only twin;
   - its simplest plausible alternative explanation, including a SEX-blind control path of comparable utility cost;
   - equal access to task and calibration data, comparable tuning effort, and the same release contract and reader access.

   Strong inherited neural, tree and code releases are evaluated faithfully from their source packages, and diagnostic or
   utility-ineligible references are labelled as such. If both outputs are jointly designed and a joint-design contribution is
   proposed, matched separate design and both sequential orders are compared; if only one output changes, that is stated. A
   control is never called SEX-blind if its fitting used SEX.

## 6. Release contract

- A release maps the six raw fields (via the admitted encoding) to an income vector and an occupation vector. Decisions are their
  argmax. Nothing else is transmitted.
- A randomized release emits one draw per recipient per person, with frozen noise coupling, seed handling, number R of Monte Carlo
  repetitions and reader sampling access (declared in its card before fitting). Accuracy, NLL and Brier are computed on each
  actual emitted vector and its emitted argmax, then averaged over R independent one-draw-per-person repetitions. An averaged
  vector is never substituted unless it is the actual release. Monte Carlo uncertainty is reported separately from household
  sampling and reader variation.
- Readers train and score under the same declared channel law. Candidate and control randomization effort is matched.

## 7. Utility (original limits, unchanged)

- **Point limits.** Per version v in {0, 1}, 12 clauses (income and occupation x U0_S<v> and Ucal_S<v> x accuracy loss <= 0.010,
  NLL excess <= 0.010 nats, summed Brier excess <= 0.005), plus the men's and women's occupation summed-Brier excess vs U0
  (<= 0.005). That is 24 clauses + 4 guards over both versions, with household-clustered SEs. U0 and Ucal are kept.
- **Recipients and target.** The original recipient non-worsening allowances apply. The 0.020 material-benefit target is reported
  separately from smaller positive improvements.
- **Reserve.** The two-SE reserve (k = 2, H_LJ SE floor) is a separate development robustness heuristic, not a new cap or a
  guarantee. It is applied symmetrically in its own frontier comparison, and point-eligible references without it stay in the
  original benchmark.
- **Matching.** No exact 0.001-nat cost match is demanded. Actual utility vectors, paired cost uncertainty and the observed
  utility-recovery frontier are reported. Similar point costs or equal nominal capacity do not establish matched population utility.

## 8. Readers (frozen splits: READER_SPLITS.json)

- **Grid.** Household-disjoint reader FIT (50%), VALIDATION (20%) and SCORE (30%) subsets of MODEL_SELECT with fresh tags
  NRM1_devgrid_R1 / R2 (registered) and R3 / R4 (robustness repetitions). Reader seeds use the new base MODEL_SEED_BASE + 600;
  k0 and k1 are registered, k2 and k3 are for robustness. That gives 8 registered cells per release. Release models and
  calibrators are never fitted on these households' labels.
- **Fitting and scoring.** Readers are fitted on FIT, selected (recipe and orientation) on VALIDATION, and scored on SCORE, with
  the same recipes and effort for every release. Comparator readers are refitted too. The household bootstrap of SCORE gives paired
  differences.
- **Selected outputs only.** The actual selected outputs are evaluated (identity, deterministic aliases and unequal strengths
  across versions included). No nearby configuration is substituted after selection.
- **Promising bank.** Repeated household splits and genuinely effective optimization seeds are run, near ties and winner changes
  recorded, paired benefits and sign reversals reported, and a small reader-data learning curve added where it could change the
  interpretation. Different seed labels, identical tree paths and deterministic reader outputs are not independent evidence:
  aliases are counted once.
- **Screening split.** NRM1_screen halves of MODEL_SELECT, used by `nrm.screen` for development diagnosis only.

## 9. Claims and selection

The statistical reviewer's frozen generic claims and selection rules (`roles/statistical_reviewer/RULES.md`, `RULES.json`,
`nrm/rules.py`, in this receipt) govern.
- **Claims, kept separate:**
  - task feasibility;
  - incremental recovery reduction beyond matched controls;
  - stable utility-recovery frontier improvement;
  - original-benchmark superiority;
  - the 0.020 material criterion;
  - joint-design contribution;
  - novelty.
- **Selection.** At most one development candidate is selected by the predeclared rule from the complete ledger. A
  control-labelled configuration may win. A smaller positive result may be a useful component without being competitive.
- **Every-cell rule.** The old every-cell rule is kept as a diagnostic beside the prospective aggregate screen. Earlier campaigns
  are not rewritten under the new rule.

## 10. Verification, deployment, custody, publication

- **Verification.** An independent verifier with separate numerical code checks:
  - source admission and role use;
  - training and calibration exposure;
  - task and subgroup metrics;
  - candidate and control selection;
  - actual transmitted outputs;
  - recovery and uncertainty;
  - representative reader refits;
  - deployment from restored artifacts.

  Numerical and procedural status are recorded separately. Every failed check, correction and original result is preserved, and no
  rule is widened after a failure.
- **Deployment.** The selected method (or strongest useful component), its task-only twin and its decisive control are packaged to
  run from the six raw fields and reproduce the evaluated wire. Packages refuse SEX, target, missing, extra and reordered fields at
  the documented and underlying entry points, and export no hidden intermediate output.
- **Custody.** Representative models and readers are backed up and restored from the backup alone. An attached writable external
  drive is used if present; otherwise off-device custody is reported as pending, with the safe copy command. No forced
  mount/unmount, and a warm read is never called a cold check.
- **Publication.** Sanitized code, aggregate results and documents go on the dedicated publication branch. The GitHub SHA and the
  public entry points are verified. The research history stays local, and earlier branches, main, manuscripts, old locks, prior
  publications and source packages are preserved. No messages are sent to people; repository handoff files are authorized.

## 11. Amendments

Amendments are dated, committed before the step they govern, and recorded in PROTOCOL_DEVIATIONS.md (with failed attempts and
timing disclosures). Receipts are never backdated.
