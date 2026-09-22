# Final report generation and interpretation

This reporting rule was written while final audits were still running, without reading live comparative or sealed evaluation outcomes. It changes no fit, configuration, selection criterion, endpoint family or uncertainty calculation.

`experiments/pcrl_task_directed_release_v1/narrative.py` provides the pure function:

```python
report = render_reports(
    evidence=accepted_evidence,                  # optional EVIDENCE.json
    prediction_assessment=accepted_predictions, # optional PREDICTION_ASSESSMENT.json
    equivalence=accepted_equivalence,           # optional EXACT_EQUIVALENCE.json
    verification=accepted_verification,         # optional PARALLEL_VERIFICATION.json
    provenance={                               # original accepted-file SHA256 pins
        "EVIDENCE.json": evidence_sha256,
        "PREDICTION_ASSESSMENT.json": prediction_sha256,
        "EXACT_EQUIVALENCE.json": equivalence_sha256,
        "PARALLEL_VERIFICATION.json": verification_sha256,
    },
)
paths = write_reports(report, out_dir=public_report_directory)
```

Only supply the keys for available inputs. The function does not discover files, deserialize models, read ACS records, fit, select, evaluate, bootstrap or replay private artifacts. The caller must verify each accepted file's bytes against its supplied SHA256 before parsing. A separate canonical parsed-content digest binds each supplied object in the output manifest. File hashes are not interchangeable with these content digests.

The output consists of `RESEARCH_DECISION.md`, `PAPER_ADDENDUM.md`, `VALIDATION.md` and `NARRATIVE_MANIFEST.json`. The writer exclusively creates each file and refuses an existing target, including an existing manifest. Use a fresh report location or explicitly manage an earlier draft outside this function; it never silently replaces a report. A partial write remains an explicit incomplete artifact. These reports are additions, not edits to the original manuscript.

## Evidence boundary

The evidence producer must already have validated accepted summaries, frozen selection, original configuration/scientific/inference hashes, the exact contrast family, numerical-version dependencies, and the claim results recomputed from the registered adjusted endpoint checks. The narrative function separately checks the evaluation/test/frozen flags, selection pin, endpoint presence, interval shape, formula-tree consistency and competitive-route eligibility. It does not recreate prospective check thresholds from prose or infer them from endpoint names.

An adjusted competitive pass can be reported only from a passing main competitive formula in accepted evidence, consistent with the frozen route's passed validation screen, complete required families and nonempty eligible comparator union. Descriptive point improvements never establish that pass. Main competitive, historical-J, attribution and stricter diagnostic claims remain separate. Blocked comparisons stay blocked; a missing input stays pending. A failed adjusted criterion is a result within this registered programme, not a population or information-theoretic impossibility result.

The manifest retains all supplied complete-configuration task points, primary recovery points, formula pointers and referenced bounds. The prose reports the selected nominees and the deterministic minimum/maximum balanced selected residence loss difference from J among complete supplied configurations. Those extrema are descriptive summaries, not additional nominees, tests or claims. Ties remain explicit. Missing H/J comparisons are unavailable rather than assigned zero.

Both unweighted and PWGTP scores use the same selected predictions. Fixed decoder, best independent deployment predictor and selected deployment predictor are separate columns. Recovery over H is `CE(H) - CE(candidate)`; positive is additional measured recovery. Recovery increment over J is `CE(J) - CE(candidate)`; positive means more recovery than J. These are not mutual-information estimates or bounds.

## Numerical versions and counts

Use the existing `EVIDENCE.numerical_versions` schema. Do not create a second four-slot registration. A separately supplied `numerical_versions` object requires a `NUMERICAL_VERSION_INDEX.json` provenance pin and must equal the evidence block if both are supplied.

That schema records the registered retry slots, original and current map/audit hashes, single completed retry, installation decision, same-slate re-audit and separate original/current role-fit/reuse counts. The final upstream collector refuses a registered but unfinished retry/install/re-audit dependency. A rejected retry retains its original map and has no replacement re-audit; an accepted retry has one. Operational interrupted/locked invocations remain incidents, not completed fit versions. The report never adds nominal method configurations for numerical retries.

The four slots and their mathematical-equivalence explanation are in `NUMERICAL_RECOVERY_PLAN.md`; no version is selected for favorable validation/evaluation performance. Original feasible fallback maps and audits remain preserved even after accepted replacements. Solver termination, independent numerical feasibility and exact optimality are different facts. Exact kernel-equivalence counts are independent of nominal units, accepted artifact counts, numerical versions and role-fit counts; none counts independent observations.

## Interpretation required in every final handoff

- The method is a supervised residence study. A gain over inherited J alone does not isolate channel design. Label-matched teacher/code/action controls and all required supervised eraser pairs must retain their eligibility/completeness qualifications.
- Ttask and Trisk are equally sized refinements of T0, but not nested with each other. Coarse embedding establishes family inclusion under identical laws/actions; strict improvement and risk-specific attribution require their own evidence. Learned/quantized scores are not claimed to be exact sufficient statistics.
- Fitted CMI is finite, empirical and conditional on the registered coarse context. Measured full-H predictive recovery is a different diagnostic, including at zero fitted budget. No differential-privacy, population, arbitrary-side-information, joint-intersection or repeated-release guarantee is implied.
- `DATED_SPLIT_CLARIFICATION.md` is a late disclosure of existing cross-anchor overlap. Anchor-specific test access restrictions do not imply globally unseen people or labels. Registered adjusted intervals are conditional development diagnostics with model-fitting and selection dependencies remaining; they do not create a retrospective independent holdout. Fresh-year confirmation remains prospective.
- P1–P7 retain the original probability, supported/refuted/unassessed status, source pointer and coverage. They are descriptive bets, not scientific gates. P7 is finite empirical nonconstant feasibility; useful held-out utility is a separate question.
- Historical published scores remain in `HISTORICAL_SCORE_INDEX.json/.md`, with exact source commit/path/hash. They are never substituted into current matched comparisons. Old slates, supervision and data use differ.
- The current-host common audit slate is distinct from the historical expanded/kernel-expanded slate. The optional neural-adversarial baseline was not executed; its optional, lower-priority scope is recorded in [PROTOCOL.md](PROTOCOL.md). The final narrative identifies that omitted comparison and does not claim an exhaustive baseline study.
- The original ownership/service framing and explicit recipient audits may remain valuable without validating every old claim or asserting novelty for established convex/privacy-funnel ingredients. `LITERATURE_SOURCES.json` and `NOVELTY_AND_ASSUMPTIONS.md` preserve the primary-source references. `INTERPRETATION_ADDENDUM_2026-09-22.md` controls the measured-recovery and descriptive-decomposition wording corrections.
- `REVIEW_INDEX.md` is not a substitute for actual venue reviews. This generator does not claim that actual reviews were answered, edit the original manuscript, contact anyone or submit anything. A later authorized submission must separately satisfy the then-current prior-review disclosure policy.

## Verification and privacy

`PARALLEL_VERIFICATION.json` is optional and separately pinned. A reported pass requires positive planned coverage, every planned unit passed, no failed/incomplete unit, an unchanged source/selection closure and the same selection pin. Its executed plan does not automatically verify every unselected artifact or the entire archive. Mathematical replay, exact zero certificates, channel equivalence, archive read-back and a clean restored replay retain their own reports and actual coverage.

The renderer projects declared aggregate fields and never copies unknown arrays, private fields, individual identifiers or free-form incident messages. This is a boundary for this generator, not an all-files privacy audit. Public reports expose aggregate values and hashes; person rows, labels, per-person losses/predictions and fitted model artifacts belong only in the task-owned private encrypted archive. No final report claims archive success without its actual verification evidence.
