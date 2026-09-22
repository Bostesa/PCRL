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
| `results/rebuttal/**`, `~/Downloads/REBUTTAL_EVIDENCE_2026-07-24.md` | **real rebuttal experiments**, but addressed to *blind-reviewer* objections recorded for the AAAI paper (R2 W1 fresh-partition generalization; R1 W4 FARE baseline) |

So: genuine rebuttal *work* exists; the genuine NeurIPS *reviews* were not located. **The prior-review
appendix is therefore incomplete and cannot be completed without the author.** See
`SUBMISSION_READY_CHECK.md` for the exact item required.

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
| A linear probe's $R^2$ certifies little; zero $R^2$ was claimed to bound classification accuracy | `docs/ACCURACY_CERTIFICATE_RETIREMENT.md` @ `0176f149e` | claim **withdrawn**; API raises `NotImplementedError`; replaced by the narrower squared-loss statement | exact 20-row sample: $\mathrm{Cov}(h,A)=0$, $R^2=0$, threshold accuracy 18/20 vs 50% majority; `tests/test_accuracy_bound.py` | zero cross-covariance still implies nothing about independence, nonlinear attacks or out-of-sample behaviour | §III, Appendix C item 1 |
| Aggregate one-hot scores may hide rare-class leakage | `results/V2_DOMINANT_AXIS_LATEST_SUMMARY.md` @ `0176f149e` | dominant-axis auditing adopted and reported beside the aggregate | HMDA multi-class mean $R^2_{\mathrm{onehot}}=0.0192$ vs $R^2_{\mathrm{DA}}=0.0609$; one configuration 0.027 vs 0.288; convex-combination identity 33/33, max residual 0.0021 | not an arbitrary-classifier or population certificate; historical encoder-line checkpoints | §III, Table I |
| Per-purpose bounds may not compose across purposes | original paper §5.5 and `ACCURACY_CERTIFICATE_RETIREMENT.md` | composition stated **only** for exact zero cross-covariance | exact-zero composition holds; $h_1=N+\delta S$, $h_2=N-\delta S$ recovers $S$ while each view has $R^2=\delta^2/(1+\delta^2)$ | approximate leakage can combine adversely; no independence guarantee | §III |
| A shared union eraser might serve conflicting purposes | `results/redesign_20260907_gaussian_v1/TABLE.md` @ `0176f149e` | claim **narrowed**: the union destroys both tasks; per-purpose erasure is the right comparator | shared union $-0.0005/-0.0004$; per-purpose LEACE $0.9996/0.9997$; restricted arm identical to per-purpose | the restricted arm **ties** rather than beats the comparator; continuous-task pilot, not the categorical trainer | Appendix C item 2 |
| Attackers may only be measuring memorisation of training rows | rebuttal run 1, `REBUTTAL_EVIDENCE_2026-07-24.md` (prediction commit `592c551`, results `da1c94d`) | fresh-partition generalisation run, registered before implementation | no flips; max attacker delta 0.013 AUC | one dataset family; addressed the other paper's audit, carried here only as method provenance | Appendix A |
| A certified competitor was missing from the baselines | rebuttal run 2, same file (prediction `ab76813`, results `099cc2d`) | FARE executed under the same two-tier protocol, reproduction gate bit-exact | passes both tiers on three cells at 0 / 28.3 / 39.0% utility | again the other paper's baseline set | Appendix A |
| A near-optimality certificate was promised | `results/reviewer_dropins/PAPER_PASTE_theory.md` lines 93–127 | explicitly **skipped**; not claimed | the source records the missing implementation/numerical check | a proposal is not a result and is not presented as one | Appendix C item 4 |
| Headline totals may be tuned on the evaluation grid | `results/reviewer_dropins/PAPER_PASTE.md` lines 96–126 | disclosed as same-grid post hoc development evidence | dual floor and skip-warmup developed after examining an earlier evaluation on the same 60-cell grid; no held-out grid validation | a later prospective registration does not retrospectively validate it | Appendix A |
