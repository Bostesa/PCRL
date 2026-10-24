# Statistical reviewer hand-back, revision 2 of the pre-registration rules (saved verbatim by the coordinator, about 05:11Z on 2026-10-24)

The addendum to the pre-registration report contained in this hand-back is also saved, verbatim, in `REPORT_PREREGISTRATION_ADDENDUM.md`.

---

Statistical reviewer hand-back, revision 2 (SAR1). This revision runs from 04:59:09Z to 05:07:04Z (`date -u`); the first freeze ran from 04:28:04Z to 04:56:14Z. It adds your two changes, the ALT path and declared selection data. Everything stayed label-free and used invented data only. I did not read the investigators' proposals, so the rules stay family-neutral. No agents, no git, no processes killed. Total CPU is under about 40 CPU-s. You already saved the earlier report and hand-back. Below is an addendum to that report, to save verbatim beside it; I cannot write report files myself.

## Files (SHA-256 at 05:06:17Z)
- `results/pcrl_strong_anchor_release_v1/roles/statistical_reviewer/RULES.md`: e830d949e33dddacc20fcaa1caeaab85f83ffde53053eb3be5d34c9152424160 (the changelog is section 14, entry 05:05:54Z)
- `results/pcrl_strong_anchor_release_v1/roles/statistical_reviewer/RULES.json` (SAR1-rules-v1): f5b6a28003c111bcd7734ca2224c11c018f578cc9d09a60687a4f33ba5fabbda
- `sar/rules.py`: 435abbceb685d80d8d2d50108fba0bfeba3410d5ad7098615e830d5de0e76fb8
- `sar/test_rules.py`: 41ab0e2ba198e05c2599d997b43f10898a594ab83450fbf0831c3c8267dd574c
- Inherited files are unchanged and are checked by `load_rules()`: nrm/rules.py d5df8686..., NRM1 RULES.json 84f18776...

## Tests
- `python3 -m sar.test_rules`: **25 of 25 PASS, exit 0** (one thread, about 0.26 CPU-s). Two tests are new: `t_d2_alt_path` and `t_d2_selection_data`. The fineness, D2R and rule-file tests were extended.
- In-memory mutation check (scratchpad only): **33 of 33 seeded bugs caught**, including 9 new ALT and selection-data mutations.

## What changed
**A. ALT path**
- Record fields:
  - `alt_declared` (boolean, from the card). If absent it counts as false, and then no label changes.
  - `paths['ALT']`.
  - `controls_not_run['ALT']`.
- The origin is always START, even when the CTRL path's origin is CONT.
- ALT uses the same two-point bracket rule and the same conjuncts as CTRL (c1-c3), plus targeting (d) against ALT's Q*.
- When declared, ALT gates D2:
  - Partial labels are ordered START, CONT, CTRL_FRONTIER, ALT_FRONTIER.
  - The met label's scope adds "beyond the starting release's own protection operation continued to matched MODEL_SELECT total-NLL spend".
  - A partial result missing ALT says the gain may come from running the start's own operation further.
- Other cases:
  - A declared ALT not run for budget gives NOT_ASSESSED_CONTROLS_NOT_RUN.
  - A declared ALT path that is missing gives INCOMPLETE.
  - An undeclared ALT path is reported beside and never gates (flags ALT_REPORTED_BESIDE_NOT_DECLARED and ALT_NOT_SEPARATED_REPORTED_BESIDE).
- Other places ALT now appears:
  - `brackets(..., alt_origin=)` returns the ALT bracket as well.
  - `grid_fineness` covers ALT.
  - D2R's decisive set adds ALT's registered bracket. `comparators['ALT']` is required when ALT is declared.
  - The grid-validity minimum set adds the ALT points.
  - The multiplicity statement now counts "two to four" comparisons per D2 look.

**B. Selection data**
- The plan records `selection_data[CTRL]` and, when ALT is declared, `selection_data[ALT]`, plus `start_in_sample_on_model_fit`.
- Allowed values are MODEL_FIT_OOF, or CALIBRATION only when START is recorded as in-sample on MODEL_FIT. MODEL_SELECT is never allowed. The SE comes from the same data.
- What D2 does with it:
  - Not recorded: D2 is INCOMPLETE.
  - Not allowed: D2 is INCOMPLETE, with flag CONTROL_SELECTION_DATA_NOT_ALLOWED_<key>. The control must then be re-selected, which counts as a POST HOC repair.
  - CALIBRATION: allowed and disclosed with flag SELECTION_DATA_CALIBRATION_<key>.
- `grid_fineness(spends, start_in_sample_on_model_fit)` reads each path's `selection_data` and gives SELECTION_DATA_NOT_ALLOWED or INCOMPLETE. It is still reported only.
- The configuration text now says CAND is selected by the card's rule on its declared selection data (guard parts MODEL_FIT out-of-fold; never MODEL_SELECT). The plan records this and the verifier checks it; the rules do not.

**C. Your answers are recorded** in RULES.md section 14 and the RULES.json change_record: the whole-release point-margin clause, the targeting gate (d), the few-valued bound of 320, and the budgets (FA 8, FB 12).

## Addendum to REPORT_PREREGISTRATION.md (please save verbatim)
(saved separately as REPORT_PREREGISTRATION_ADDENDUM.md)

## Open questions
None blocking. One remaining choice is yours: if you prefer an absent `alt_declared` to make D2 INCOMPLETE for FB-family readings (instead of "not declared"), require the field in the plan. As written, absent means false, per your instruction.
