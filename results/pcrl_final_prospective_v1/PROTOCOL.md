# Final prospective evaluation of frozen PCRL releases on ACS 2016 — protocol

Registered 2026-09-22 before any 2016 feature transformation, fitting or outcome. Branch
`research/pcrl-final-prospective-v1`, based on the task-directed study at
`f4bdf4cd5bf74c634feeec50aef78bff249667e4`. This is a bounded final evaluation of existing releases.
It is not a search for a mechanism. No encoder, teacher, eraser, code, action dictionary, Q, privacy budget
or quantizer is fitted, chosen or tuned. On 2016 the only new fits are the independent attackers and utility
probes, all under the frozen recipe below.

## 1. What this evaluation can and cannot establish

It can establish whether two predeclared operating points transport to a previously unused survey year
under a locked final evaluation, and how they compare with fixed controls. It cannot change the historical
706-endpoint decision on 2018. That decision stands: no adjusted competitive pass, and the Q historical-J
conjunction failed because 7 of 8 adjusted privacy bounds missed the .001 cap. The evaluation also cannot
prove a stochastic-versus-deterministic optimum, and it cannot turn supervised residence prediction into
unseen-task transfer. It does not certify privacy against arbitrary attackers, and it gives no
coalition-specific result (Q constrains only the A view).

## 2. Panel (frozen; see PANEL.json and MODEL_MANIFEST.json)

| Name | Frozen family | Role in this study |
|---|---|---|
| H | original immutable services | service-only reference |
| J | matched historical J release | historical operating-point comparator |
| **Q** | `T0_L_0.01_a17` | **primary hypothesis 1** |
| **D17** | `T0_U_unconstrained_a17` | **primary hypothesis 2**; same code and same 17 actions, no constraint |
| D33 | `T0_U_unconstrained_a33` | control: finer deterministic task control |
| C | `continuous_task` | control: direct supervised prediction |
| E | `leace_supervised_mechanism40` | control: label-matched LEACE adaptation |
| S | `splince_supervised_mechanism40` | control: label-matched SPLINCE adaptation |
| RR75 | `T0_rr_0.75` | fixed simple-randomization control (verified complete for all three anchors) |
| W75 | `T0_withhold_0.75` | fixed simple-randomization control (verified complete for all three anchors) |

These are the ten families, giving thirty anchor interfaces. All three existing anchor fits are used. The
anchors are not independent datasets. The families are common across anchors, and no family is chosen per
anchor or per weighting. The two hypotheses are Q and D17. The other eight families are controls, and none
of them can replace a hypothesis after 2016 is observed. On the archived 2018 test rows no two families
release identical views (`exact_equivalence_groups_2018_test`). The same check is repeated on the 2016
fitting pool before any fit. OptNet is not in the panel because its encoder weights were never saved, so no
transport comparison exists for it. Training labels, training households, action counts, channel widths
and inference inputs for every family are listed in `PANEL.json/families/*/fairness`. The E and S
adaptations keep their actual mechanism40 access to joint SEX/RAC1P labels (S also uses
residence/income/employment preservation). An adaptation of a published method is not that method's
universal optimum.

## 3. Release contract

See RELEASE_CONTRACT.md. In short: A receives [H_A, Z], B receives the unchanged H_B, and the coalition AB
receives [H_A, H_B, Z]. Each person gets one persistent sampled token. The Q row, the private coin, the
code, protected labels and the household index are never on the wire. H must be byte-identical across
interfaces on the same host and rows.

## 4. Data (see DATA_ADMISSION_PLAN.md)

The study reuses the committed, label-blind 2016 admission partition (evidence-paper branch, commit
`0d8f4b67`, salt `PCRL-evidence-review-2016-admission-v1`). That partition is one global household
split shared by every anchor, interface and role:

- fitting: 39,697 people in 26,546 households. Attackers and utility probes both fit on this whole pool,
  as in the predecessor's recipe, where both fit on the union of its two fitting pools.
- attacker validation: 7,911 people in 5,310 households. It selects attack predictors only.
- task validation: 8,006 people in 5,309 households. It selects utility predictors only.
- final evaluation: 23,684 people in 15,928 households. It is scored once, after the evaluation lock.

The code reads final-partition labels only through a loader that requires `EVALUATION_LOCK.json`. That lock
hashes every selected predictor, registry, frozen object and scoring source, and it is committed and pushed
before the labels are read.

## 5. Execution host

A single Linux x86-64 host (c7i.8xlarge, Python 3.12, numpy 2.4.2, scikit-learn 1.8.0, torch 2.10.0+cpu,
BLAS/torch threads = 1 per worker) performs every 2016 transform, encoding, fit and score. Tier A found
the reason: the frozen b(H_A) teacher is a float32 MLP. On macOS-arm64 its outputs differ from the
archived Linux encodings by up to about 1e-7. That difference moves 0.1–0.4% of 2018 fitting rows across a
T0 quantile cut, because those rows lie exactly on cuts derived from their own quantiles. Historical byte
fingerprints and cross-platform toleranced compatibility are therefore reported separately. All primary
comparisons use same-host references. Local replays report tolerances and code flips.

## 6. Attackers and utility probes (ATTACK_RECIPES.json, UTILITY_RECIPES.json)

The predecessor's `catchup` slate is used unchanged, with its hyperparameters, weighting, class-schema
floor and selection rule. Legal ancestors are fixed by what is on the wire:

- every non-H role also gets the same-role H-only candidates;
- AB roles also get all candidates of the release's own A role plus H's B role;
- utility also gets the frozen fixed decoder where one exists, plus the frozen H_A global-offset
  recalibrators.

J is never an ancestor. Selection minimizes 0.5·unweighted + 0.5·PWGTP expected validation log loss, with a
lexical tie-break. The same selected predictions are scored under both weightings. The fixed decoder, the
independent-slate selection and the validation-selected predictor are reported separately. Attack roles
are A/SEX, A/RAC1P, AB/SEX and AB/RAC1P. The two B roles are fitted once per anchor, from H, only as AB
ancestors. The task role is A/same_residence. There are 156 fitting units. Their seeds are
`2016000 + 10000·anchor + 37·role_index`, identical across releases.

## 7. Loss (RANDOMNESS_AND_LOSS.md)

The primary per-person loss is the exact single-release expectation
L_i = Σ_z Q(z|x_i) · loss(f(H_i, z), y_i). It averages losses over tokens. It is not the loss of an
averaged probability. Deterministic releases are the one-action case, and RR/W use their declared
one-release laws. Q(x_i) is used only by the offline evaluator and never enters an attacker.

## 8. Inference (PRIMARY_CLAIMS.json, SECONDARY_CONTRASTS.json, BOOTSTRAP_SPEC.json)

The estimands are dU(M) = CE_task(M) − CE_task(J) and, for each role, dG(M) = CE_J − CE_M. In both,
negative favors M. Each anchor contributes a weighted ratio over its role's complete-case final rows, and the
three anchor ratios are averaged equally. Values are never clamped.

**Primary.** For each M in {Q, D17} there are ten clauses:

- dU ≤ −.003 under both weightings;
- dG ≤ .001 for all four roles under both weightings.

Each clause passes iff its one-sided 97.5% upper bound (estimate + 1.959964·SE, from a common 10,000-draw
household bootstrap) meets the threshold. The candidate passes iff all ten clauses pass. This is an
intersection-union test at α = .025 per candidate and nominal family-wise .05 over declaring either
conjunction. The argument: if a candidate's null holds, some clause k* is false. Declaring the conjunction
requires rejecting k*, which happens with probability ≤ α, so the size is ≤ α. Bonferroni over the two
candidates then gives ≤ .05. No factor of ten is applied inside a conjunction. The bounds are pointwise
component bounds, not simultaneous coverage. Control is nominal and asymptotic, conditional on the fitted
releases, the fitted attackers and their selection. An SE of zero means the bound equals the estimate.
Missing or invalid measurements fail.

**Secondary (separate family).** Q is compared with each of D17, D33, C, E, S, RR75 and W75 on two task
differences and eight sensitive differences. That is 7 × 10 = 70 endpoints, each with a two-sided
simultaneous 95% Bonferroni interval, z = z_(1−.05/140). No error rate is claimed across the two families,
and no secondary result can be promoted to a primary success.

**Descriptive (unadjusted, labeled).** Recovery over H, task differences versus H and J, and D17-versus-Q
recovery differences, each with a marginal 95% interval.

The bootstrap is the predecessor's `paired_household_bounds`, unchanged. Tier A showed that it reproduces
the archived 2018 estimates and SEs for 38 registered endpoints exactly. Only its estimates and SEs are
used here.

## 9. Execution order and stopping

1. Commit this protocol. Terminal 2 reviews it.
2. Run the provenance audit and 2016 admission checks.
3. Transform 2016 features. This step is label-free.
4. Fit and select on the fitting and validation pools.
5. Commit the evaluation lock.
6. Score the whole panel once.
7. Run the bootstrap and generate reports.
8. Run the independent replay, archive and close out.

No step stops early because of an outcome: not when Q fails, not when D17 wins, not when an endpoint is
unfavorable. No conditional nomination or new grid follows.

## 10. Repairs and amendments

Engineering faults are repaired with bounded retries, quarantine and atomic writes. Before final outcomes, a
protocol defect can be amended in a dated file that states the old rule, the new rule and the reason.
After final outcomes, only a genuine code bug can be corrected. In that case the whole affected panel is
rerun from unchanged objects, both outputs are kept, and the timing is disclosed. Thresholds, candidates,
roles, weightings and win rules never change.

## 11. Decision cases

The possible cases are:

- Q passes and D17 does not;
- both pass;
- D17 passes and Q does not;
- neither passes;
- technical validity or freshness fails.

The wording for each case is the one registered in the assignment and reproduced in `RESEARCH_DECISION.md`.
In every case the 2018 development result and its failed conjunction are retained.
