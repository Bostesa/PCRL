# Bug Report: Shuffle-Misaligned Extraction in `generate_report()`

**Date discovered:** 2026-04-19
**Date introduced:** 2026-04-14 (commit `a50810e`)
**Date fixed:** 2026-04-19
**Severity:** Data-corrupting (silent incorrect results)

## What happened

Commit `a50810e` ("Speed up training and evaluation by eliminating
redundant encoder forward passes") refactored `generate_report()` in
`pcrl/evaluation/certificates.py` to cache representations separately from
labels.  The original implementation extracted both in a single pass over
the DataLoader:

```python
# BEFORE (correct — single pass)
train_reprs, train_labels = _extract_representations_and_labels(
    encoder, train_loader, purpose_idx, attr_name, device
)
```

The refactored version split this into separate functions that each iterated
the DataLoader independently:

```python
# AFTER (broken — separate passes)
train_reps_cache[purpose_idx] = _extract_representations(encoder, train_loader, ...)
# ... later ...
label_cache[("train", attr_name)] = _extract_labels(train_loader, attr_name)
```

Because `train_loader` has `shuffle=True`, each call to `__iter__()` produces
a different random permutation of the data.  The representations from
iteration 1 correspond to sample ordering A, while the labels from
iteration 2 correspond to sample ordering B.  The resulting
`(train_reprs[i], train_labels[i])` pairs are **misaligned** — each
representation is paired with the label of a *different* data point.

The PostHocAuditorSuite (LogisticRegression, RandomForest, SVM, XGBoost)
trained on these misaligned pairs learns nothing meaningful and reports
accuracy at or below the majority baseline, regardless of how much
information the representation actually contains about the attribute.

Test-set data was unaffected because `test_loader` has `shuffle=False`,
so R^2 values remained correct.

## Which results were affected

| Result | Script | Code path | Produced | Status |
|--------|--------|-----------|----------|--------|
| Table 2 (Adult baselines) | run_baselines.py | generate_report() **pre-bug** | March 29 | **Correct** |
| Table 2 (INLP/LEACE) | run_extended_baselines.py | run_compliance_on_reprs() | March 29 | **Correct** |
| Table 3 (HAR) | run_har_real.py | generate_report() **pre-bug** | March 29 | **Correct** |
| Table 4 (cross-purpose) | run_cross_purpose_attack.py | Direct PostHocAuditorSuite, shuffle=False | April 14 | **Correct** |
| Figure 3 (lambda sweep) | run_baselines.py | generate_report() **pre-bug** | March 29 | **Correct** |
| Composition experiment | run_composition.py | run_compliance_on_reprs() | March 29 | **Correct** |
| MINE estimates | run_extended_baselines.py | MINE (no PostHoc audit) | March 29 | **Correct** |
| CelebA v2 | run_celeba_v2.py | generate_report() **post-bug** | April 16 | **Affected** |
| Multi-seed Adult | run_multi_seed.py | generate_report() **post-bug** | April 19 | **Affected** |
| Multi-seed HAR | run_multi_seed.py | generate_report() **post-bug** | April 19 | **Affected** |

**All original paper claims are valid.** Every table, figure, and
quantitative claim in the paper was produced from results generated before
the bug was introduced (March 25-29).  The bug only affected reproducibility
runs performed after April 14.

For the multi-seed run, INLP and LEACE results used `run_compliance_on_reprs()`
(single-pass extraction) and were correct.  Standard, LAFTR, and PCRL results
used `generate_report()` (broken) and had incorrect empirical accuracy values.
LAFTR and PCRL results were accidentally correct because those methods
genuinely suppress attribute information (R^2 near 0), so even correctly
trained auditors would report near-majority accuracy.

## How the bug manifested

The multi-seed Standard encoder on Adult Census reported:
- Sex delta = -0.1% (should be ~+25%)
- Race delta = -0.0% (should be ~+5%)
- Marital delta = -0.9% (should be ~+44%)

These near-zero deltas are impossible for a no-privacy encoder and
immediately flagged the results as suspect.

## The fix

Restored single-pass extraction.  `generate_report()` now calls
`_extract_representations_and_labels()` which extracts representations
AND all needed sensitive attribute labels in **one iteration** of the
DataLoader, guaranteeing row-level alignment regardless of shuffle state.

The speed optimization intent is preserved: representations are still
cached per purpose_idx (one encoder forward pass per purpose), and the
label extraction adds negligible cost since it's just reading from the
batch dict that's already in memory.

## Regression test

`tests/test_generate_report_alignment.py` trains a Standard encoder on
Adult Census (no adversarial training) and verifies that `generate_report()`
reports empirical Sex delta > 15%.  A Standard encoder's representation
trivially leaks Sex (R^2 ~ 0.6); if the empirical auditor can't beat
majority, the extraction is misaligned.

The test deliberately uses a shuffled train_loader to exercise the exact
condition that triggers the bug.

## HAR pair count note

During this investigation, a secondary discrepancy was identified: the
HAR experiment has **3 purpose-attribute pairs**, not 2 as described in
some paper sections.  The `health_monitoring` purpose disallows both
`subject_id` and `activity`, producing:

1. `activity_recognition / subject_id`
2. `health_monitoring / subject_id`
3. `health_monitoring / activity`

This has been the case since the code's initial commit (`a96ee0e`,
March 25).  There is no historical version with only 2 pairs.  Any paper
text describing "2 purpose-attribute pairs" for HAR should be corrected
to "3 purpose-attribute pairs" (from 2 purposes with 1 and 2 disallowed
attributes respectively).

## Lessons

1. **Never iterate a shuffled DataLoader more than once for paired data.**
   If representations and labels must correspond row-by-row, extract them
   in the same loop.  The `run_cross_purpose_attack.py` script already
   knew this — it creates a separate `train_extract_loader` with
   `shuffle=False` for extraction.

2. **"Speed optimization" refactors are high-risk for data pipelines.**
   The refactor was logically correct for non-shuffled loaders and looked
   innocuous.  The shuffle interaction was not tested because the existing
   test suite didn't verify empirical accuracy of a no-privacy encoder.

3. **Include a "known-leaky" baseline in every evaluation test suite.**
   If the test suite had verified that a Standard encoder reports high
   empirical delta, this bug would have been caught immediately.
