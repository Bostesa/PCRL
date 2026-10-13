# Next direct trial: occupation task-safeguarded refinement

Status: implementation-ready next experiment; **not run**. The present campaign
did not trigger its conditional pilot. Its four fixed gradient batches contained
two adverse directions, while the registered trigger required three. A direct
trial needs a new identity, explicit authorization, fresh preflight, new compute
queue, and new science/reporting locks. It must preserve that current decision.

## Question and competing explanations

Can the implemented occupation safeguard improve complete-view sensitive
recovery while keeping the actual occupation head useful? Modify the occupation
encoder/head of RAW strength 0.3, epoch 5, while freezing income exactly. This is
a test of whether task checks help during heterogeneous training updates. The
two adverse and two favorable starting batches do not establish a stable causal
conflict. Prediction leakage, extra feature leakage, limited critic strength,
and objective/reader mismatch remain competing explanations.

The source is `35f7636a159cb1efcc349ee69fed8b6c94b389c4`, to be replaced by ROOT with the
full final campaign commit. The useful predecessor public source remains
`6e767c5a5ccebcf984e2c4deb678ed7a560413cf`; private execution artifacts are located
through the admitted role-safe manifest and roster, not a public source checkout.
The proposed new identity is `pcrl-occupation-safeguarded-trial-v1`, branch
`research/pcrl-occupation-safeguarded-trial-v1`, with results under
`results/pcrl_occupation_safeguarded_trial_v1/`.

## Mechanism and matched arms

Reuse `occupation_bottleneck/pilot.py`, `pilot_worker.py`, and the corrected
`gradient_worker.py`. Canonical corrected recipe SHA:
`7c4ca6110be3c4774c9cdfdcff46babd44c117a39cf81e47d7f727de7bbf1528`.
Read the live recipe and verify this hash; the earlier r5 recipe JSON predates
the rounding correction. No new teacher, distillation term, controller, critic
family, or strength sweep is added.

| Arm | Occupation model objective | Task safeguard |
| --- | --- | --- |
| CONSTRAINED | CE + lambda × (occupation + pair sensitive penalty) | NLL/Brier displacement constraints and finite-batch checks |
| ORDINARY | Same CE and both sensitive terms | Off |
| LOCAL_TWIN | Same CE and coefficient, pair term removed only from model objective | On |
| TASK_ONLY | CE; privacy model gradient disabled | Off |
| Unchanged RAW | No updates | Source control |

All four continuation arms share starting weights, random draw streams, six
actual-wire critics, optimizer schedule, model proposal count, reference scheme
and selection effort. Both critic banks train even in LOCAL_TWIN and TASK_ONLY.
Income encoder/head tensors remain bitwise equal to their source after every
critic and model update. Each runtime wire is raw83 through its own encoder and
head: income full21, occupation full29, aligned pair full50. Labels never enter
these runtime inputs.

Critics are linear, MLP32 and MLP64 per occupation/pair view, Adam 0.003, 80 warm
updates per critic and five before each model proposal. Use batch256 uniform
draws from inherited FIT folds 0/1, 20 epochs and checkpoints 0/5/10/20. FIT fold2
is selection only. The fixed 256-row detached diagonal reference refreshes
initially and each epoch, retaining every coordinate with strictly positive
scale. One lambda per seed is derived from 16 initial warm draws:
`3 * median(task CE gradient norm) / median(full occupation+pair penalty gradient norm)`.
Both medians must exceed 1e-12, be finite, and lambda must be at most 1e4. An
invalid normalization ends that attempt; do not choose a new coefficient.

Propose SGD displacement `-.01*(g_CE+lambda*g_sensitive)`. In constrained arms,
project against homogeneous occupation NLL/Brier gradient halfspaces and the
0.01 norm ball. Read the actual mixed-dtype displacement back. The corrected
norm-only repair applies identically to all arms, with safety fraction 1e-6 and
at most 16 retries; it does not use task outcomes. Constrained arms retain the
original finite NLL/Brier tolerance 1e-10 and fractions 1/.5/.25/.125. Restore
exact parameters after failed checks and log zero stalls. The checks certify
the update batch only. The ordinary control is a matched bounded SGD control;
this trial cannot establish superiority over all historical Adam recipes.

## Reader and nomination design

Fit the full slate of **18 fresh readers** on each distinct new ATT view under
the inherited group-disjoint split. Use all nine BASE_READERS and all nine
EXTENDED_READERS from `reader_specs.py`: six logistic, six MLP64×32, and six
HGB15×200 fits per view. Orient on ATT validation only. Full-wire recovery uses
all 18 applicable readers; the pair envelope includes its 18 pair readers and
both sets of 18 exact local-ignore aliases. Do not use the inherited base9-only
`primary` envelope as the complete-view endpoint.

Fit income once per source seed for RAW and share its exact frozen bank across
every pilot checkpoint/arm. Epoch-zero and unchanged RAW releases are exact
aliases, subject to queued wire/parity checks. Aliasing removes duplicate fits,
not recipients or reader families. Refit fresh U0 banks as useful reference
anchors. Include ordinary head-row-space releases of U0 and RAW as fixed
interpretation comparators; these are separate disclosure contracts.

Before any fit, lock the following common per-arm selection rule. Across paired
seeds, calculate the largest FIT-fold2 ratio of each utility excess to its
original cap, relative to both U0 and Ucal. Among common epochs whose point
proxy clauses all pass, minimize mean ATT-validation all 18 pair recovery; tie
by lower paired occupation NLL, then earlier epoch. If none pass, retain the
epoch with smallest maximum normalized excess; tie by NLL, pair recovery, then
earlier epoch, and mark it point-ineligible. This is an exploratory training
proxy selection rule. It does not use the earlier unequal-exposure FIT confidence
gate, or establish assessment usefulness. Nominate one common epoch per arm.

## Registered unit counts and resources

With admitted update rows 10,223 and selection rows 5,211, there are 40 proposals
per epoch, 800 per arm and 6,400 across both seeds and four arms. Six critics
receive 24,480 optimizer updates per arm and 195,840 total. There are 32 saved
pilot packages, 32 normalization gradient draws, and 32,640 critic batch draws.

| Fresh reader scope | Logistic | MLP64×32 | HGB15×200 | Total |
| --- | ---: | ---: | ---: | ---: |
| All pilot checkpoints + unchanged RAW, after epoch0/income aliases | 324 | 324 | 324 | 972 |
| U0 full anchors, both seeds | 36 | 36 | 36 | 108 |
| U0 and RAW head-row-space anchors, both seeds | 72 | 72 | 72 | 216 |
| Main fresh reader fits | 432 | 432 | 432 | 1,296 |
| Real-data shuffled/planted controls: U0 and RAW full, both seeds × three views | 24 | 0 | 0 | 24 |

The 972 count has two frozen-income banks plus occupation/pair banks for 26
distinct seed/checkpoint packages: `(2 + 26*2)*18`. The 32 nominal packages have
eight epoch-zero files that collapse to two distinct initial models; 24 later
packages remain. Pair-ignore aliases require no fits. Null/planted fits are
logistic under the current `fit_reader` implementation and are diagnostics,
never deployed SEX inputs.

Independent ATT refits must cover U0, RAW, CONSTRAINED, ORDINARY, LOCAL_TWIN and
TASK_ONLY nominees, both seeds and all three views: 36 targets. This preserves
the eventual most protective eligible control even when assessment usefulness
changes eligibility. Freeze target architectures from ATT validation before new
reporting access; their logistic/MLP/HGB allocation is then counted explicitly. If an
anchor/control identity aliases another, retain an independent attempt receipt.
Private restore/deployment jobs and failures also consume the resource budget.

Use a new maximum of 20 total CPU-hours, ten elapsed hours, $0 cloud, at most two
heavy processes including unrelated jobs, one numerical thread per process,
2 GiB RSS per worker/4 GiB aggregate, and at least 15 GiB free disk. Training
artifacts reserve 256 MiB; reserve another measured allowance for the 1,320 main
reader/control fits and private scores before starting. Do not assert runtime
feasibility from fit counts. Fresh queued engineering timing must show both
seeds, every mandatory reader, independent replay and closeout fit the budget.
If they do not, report preflight infeasibility or an incomplete attempt. Do not
silently reduce reader families or append recipes after outcomes.

## Data, reporting and decision

Reuse only the inherited FIT 15,434 rows/15,428 groups, ATT 4,064/4,061, and INNER
2,235/2,234; verify exact counts and duplicate groups. These people, including
the present campaign's assessment, have already been used. New locks or seeds
do not make a future trial fresh confirmation. Do not open old assessment arrays,
HEAD_VALIDATION, all-role loaders, or acquire another cohort. Keep new trial
INNER truth arrays inaccessible until nominees, all readers/orientations, independent
targets and reporting registries are committed. Document prior exposure.

The five nominated services contribute 120 utility clauses: five services ×
two seeds × two tasks × two references × accuracy/NLL/Brier. Register all twelve
CONSTRAINED privacy contrasts against unchanged RAW, ORDINARY, LOCAL_TWIN and
TASK_ONLY: pair benefit at least .020 and each recipient worsening at most .010.
One frozen combined family therefore has K=132, subject to independent registry
validation before reporting; add explicitly registered diagnostic intervals to
a separately declared family. Do not reuse the current campaign's K values.
Use shared paired duplicate-group bootstrap draws across every arm and both
fitted seeds, 2,048 replicates, and a newly frozen bootstrap seed. The two model
seeds are not independent populations.

The competitive comparator is the most protective complete-view control
among all four point-eligible matched controls: the smallest paired-seed mean
pair AUC, with a prelocked deterministic tie order. Preserve all four comparisons.
Require the CONSTRAINED nominee's 24 original utility clauses, pair benefit,
and both local guards to be established under the registered simultaneous
family. Report point values and PASS/UNRESOLVED/SHORTFALL/VIOLATION/MISSING
separately. A favorable trial is an exploratory repair lead, not a joint design
theorem or a general novelty claim. Independent metrics, recovery envelopes,
selection, interval classifications, checkpoint restoration and label-free
deployment must finish before any competitive claim.

Runnable training code is preserved; future campaign queue/reporting adapters,
new identity locks and direct authorization remain required preparation. The
accompanying execution prompt lists those boundaries and actual entrypoints.
