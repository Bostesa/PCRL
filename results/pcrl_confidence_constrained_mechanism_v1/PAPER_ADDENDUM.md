# Paper addendum (exploratory development evidence; supported, label-free, independently replayed)

**Scope.** This addendum reports only what the locked, verified Stage D run supports. It contains no privacy result.
Do not merge it into the manuscript without the manuscript owner's decision.

**Suggested paragraph.** We asked whether a many-to-one release of the calibrated teacher's scores could keep a
per-person confidence guarantee while coarsening enough to leave room for attribute protection. The guarantee is: for
every input and every possible label, log loss within 0.005 and multiclass Brier within 0.0025 of the calibrated
teacher, with the same predicted class. Any token that obeys this guarantee pins the teacher's score vector to a cell
of total-variation diameter at most e^0.005 − 1 ≈ 0.005.

On three teacher seeds of the Adult development split, at the capacity used throughout (8 income and 64 occupation
tokens per predicted class), we measured coverage and fallback:
- **Occupation:** at least 10,105 of 15,434 fitting rows required a token of their own. No admissible code could cover
  more than 36.6% of fitting rows (a certified packing bound), and 85% of held-out inputs fell back to their raw
  calibrated scores.
- **Income:** the exact optimum covered 37–39% of rows, with 61–63% fallback.

The registered go rule therefore failed in every seed and recipient, and no prototype was fitted. Under a weaker,
calibration-dependent guarantee in expectation (a registered diagnostic), income compressed to 16 tokens with 4–5%
fallback, while occupation still fell back on 44–48% of inputs. These are exploratory development results on a
repeatedly used dataset. They concern utility geometry only, not privacy.

**Source pins.**
- Branch research/pcrl-confidence-constrained-mechanism-v1.
- FEASIBILITY_LOCK commit d5337e552e78b31830f94d85ef1cc4b335b1b8ec.
- Results commit dc57711a1d80fe4868f93008339ff2d02d467f7b.
- Final evidence commit: see HANDOFF.json.

**Do not say:**
- that this shows calibration or compression removed sensitive information;
- that any privacy frontier advantage was shown;
- that average-loss confidence targets are infeasible.
