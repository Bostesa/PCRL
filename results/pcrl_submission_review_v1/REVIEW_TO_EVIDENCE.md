# REVIEW_TO_EVIDENCE — review concerns, responses, and what the evidence actually shows

## First, the status of the review record itself

The original paper exists and was submitted: **"One Encoder, Many Purposes: Purpose-Conditioned
Representation Learning with Per-Purpose Linear Leakage Bounds"**, NeurIPS 2026 submission, PDF created
2026-05-07 06:48 EDT (deadline 11:00 that day), local copy `32955_One_Encoder_Many_Purpose.pdf`, 26 pages.
The leading number is consistent with a submission identifier.

**The venue reviews for that submission were not found.** Searched: the whole repository at the pinned
commit, `~/Downloads` (1,395 files), and `~/Documents`, for review text, OpenReview or HotCRP artifacts,
decision notices, meta-reviews and rating/soundness/confidence patterns.

What exists instead, and must not be substituted for venue reviews:

| file | what it actually is |
|---|---|
| `~/Downloads/reviews_outputs_leak_what_they_use.md` | an **AAAI-27 review package for a different paper** ("Outputs Leak What They Use"), three reviews + meta-review |
| `~/Downloads/simulated_reviews_v28.md` | explicitly **"Generated ... by three independent reviewer agents"** — simulated, for that same other paper |
| `~/Downloads/deep-review-round3.md`, `durable-guarantees-review.md`, `post-revision-review.md` | self/agent review passes, again on the AAAI paper |
| `MOCK_REVIEWS_2026-07-24.md` (repo) | self-described mock reviews, that other paper |
| `~/Downloads/REBUTTAL_EVIDENCE_2026-07-24.md` | real rebuttal experiments for the **other** (AAAI) paper: fresh-partition generalisation and the FARE baseline, addressed to that paper's blind-reviewer objections |

**Correction (2026-09-22, Terminal 3 audit T3-16).** An earlier version of this file attributed
`results/rebuttal/**` to the AAAI paper. That was wrong and is withdrawn. All four directories there —
`ablations_facct`, `erase_layer_pilot`, `erase_layer_vicreg_sweep_aws`, `erase_rank8_diabetes_cpu` — are
**this** line of work's rebuttal evidence, verified by reading them: they concern per-dim standard
deviation, the LoRA erasure floor, the published `[128,128]→64` backbone and the Adult/HMDA/Diabetes
grid. The same earlier version listed the fresh-partition and FARE runs as "Appendix A" of this
manuscript; `main.tex` contains neither, and those two runs do belong to the other paper. Both errors are
corrected here and in the claim ledger.

**The reviews exist.** `results/rebuttal/ablations_facct/PREDICTIONS.md` @ `ecaeba46` registers ablations
against concerns attributed to **named reviewer handles and the area chair** ("reviewers AC, AkJK Q1,
NY7k Q2"; "AkJK Q2"). Those handles are venue review identifiers, so the earlier submission was reviewed
and the project worked from the reviews. What is absent is the **review text itself**, which is in no
repository ref. The paraphrases in that registration are project summaries and cannot substitute for the
appendix.

**The prior-review appendix is therefore incomplete and can only be completed by the author.** See
`SUBMISSION_READY_CHECK.md` for the exact document to retrieve.

## The lineage question the author must answer

SaTML requires the reviews of *the latest previous submission of the same paper*. Whether this submission
is "the same paper" is a factual question about content, and it is genuinely close:

* **Same:** research line, protected-attribute audit philosophy, the dominant-axis auditing result and the
  composition/guarantee corrections, all of which are reused here (§III of the paper).
* **Different:** the release contract (there the releaser owns and re-emits the representation; here the
  published service output is immutable and the decision is what to append), the mechanism (per-purpose
  LoRA adapters under a proxy-Lagrangian dual versus a task-directed finite channel), the data (Adult /
  HMDA / Diabetes versus ACS), and every headline result.

A defensible reading is that this is a **successor paper that reuses two contributions** from the earlier
submission rather than a resubmission of it. That reading is the author's to make and to state honestly;
it must not be manufactured by retitling. If the author judges it the same paper, the NeurIPS reviews must
be appended complete and unedited except for anonymisation.

## Concern → evidence map

Sources are project-internal records (the original paper's own corrections and rebuttal runs), **not**
venue review text. Column 2 names what the concern was raised against.

| concern | exact original source | response / change | verified evidence | residual limitation | in the final paper |
|---|---|---|---|---|---|
| A linear probe's $R^2$ certifies little; zero $R^2$ was claimed to bound classification accuracy | `docs/ACCURACY_CERTIFICATE_RETIREMENT.md` @ `0176f149e` | claim **withdrawn**; the API refuses to run **in the evaluated code**, but still ships on public `main` (repair PR prepared, not merged); replaced by the convex-loss statement | exact 20-row sample: $\mathrm{Cov}(h,A)=0$, $R^2=0$, threshold accuracy 18/20 vs 50% majority; `tests/test_accuracy_bound.py` | zero cross-covariance still implies nothing about independence, nonlinear attacks or out-of-sample behaviour | §III, Appendix C item 1 |
| Aggregate one-hot scores may hide rare-class leakage | `results/V2_DOMINANT_AXIS_LATEST_SUMMARY.md` @ `0176f149e` | dominant-axis auditing adopted and reported beside the aggregate | HMDA multi-class mean $R^2_{\mathrm{onehot}}=0.0192$ vs $R^2_{\mathrm{DA}}=0.0609$; one configuration 0.027 vs 0.288; replay of the stored per-class scores reproduces the stored aggregate within $2\times10^{-5}$ in 31 of 33 cells (two exceptions trace to a single-precision aggregate in the historical evaluation code) | not an arbitrary-classifier or population certificate; historical encoder-line checkpoints | §III, Table I |
| Per-purpose bounds may not compose across purposes | original paper §5.5 and `ACCURACY_CERTIFICATE_RETIREMENT.md` | composition stated **only** for exact zero cross-covariance | exact-zero composition holds; $h_1=N+\delta S$, $h_2=N-\delta S$ recovers $S$ while each view has $R^2=\delta^2/(1+\delta^2)$ | approximate leakage can combine adversely; no independence guarantee | §III |
| A shared union eraser might serve conflicting purposes | `results/redesign_20260907_gaussian_v1/TABLE.md` @ `0176f149e` | claim **narrowed**: the union destroys both tasks; per-purpose erasure is the right comparator | shared union $-0.0005/-0.0004$; per-purpose LEACE $0.9996/0.9997$; restricted arm identical to per-purpose | the restricted arm **ties** rather than beats the comparator; continuous-task pilot, not the categorical trainer | Appendix C item 2 |
| A near-optimality certificate was promised | `results/reviewer_dropins/PAPER_PASTE_theory.md` lines 93–127 | explicitly **skipped**; not claimed | the source records the missing implementation/numerical check | a proposal is not a result and is not presented as one | Appendix C item 4 |
| Headline totals may be tuned on the evaluation grid | `results/reviewer_dropins/PAPER_PASTE.md` lines 96–126 | disclosed as same-grid post hoc development evidence | dual floor and skip-warmup developed after examining an earlier evaluation on the same 60-cell grid; no held-out grid validation | a later prospective registration does not retrospectively validate it | Appendix A |

## Concerns attributed to the actual reviewers, and their current status

From `results/rebuttal/ablations_facct/PREDICTIONS.md` @ `ecaeba46`, which registers ablations against
named reviewer handles and the area chair. These are **project paraphrases of reviewer questions, not
review text**, and they do not satisfy the venue's prior-review requirement.

| # | concern (paraphrased in the registration) | attributed to | status now | where it lands |
|---|---|---|---|---|
| R1 | Is the low-rank adapter needed, or would a plain linear adapter do? | AC, AkJK Q1, NY7k Q2 | **unresolved** — the ablation was registered with code and a smoke test, and never run | paper claims no necessity or advantage for the adapter; the missing comparison is stated (App. A) |
| R2 | Is the frozen erasure layer necessary? | AkJK Q2 | **unresolved** — registered, never run; the one existing comparison adds the erasure layer and the constraint together and cannot separate them | stated as untested (App. A) |
| R3 | Hyperparameters tuned on the evaluation grid | reviewers | **partly addressed** — disclosed, not repaired; held-out selection registered and never run | "same-grid post hoc development evidence" (App. A) |
| R4 | Does the method meet its constraint? | reviewers | **partly addressed** — final iterate 56/60 strict with 7/60 also clean; best-checkpoint rule gives 54/60; a later erase-layer arm reaches 60/60 strict but 0/60 clean | strict and clean reported separately under one stated rule (App. A) |

The remaining project-planning concerns (composition, shared eraser, baselines, the accuracy guarantee,
the rank floor) are covered by the rows above and by Appendix C of the paper.