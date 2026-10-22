# Verifier addendum brief (coordinator, campaign `pcrl_input_and_pair_release_v1`)

You verified the registered readings IA_REG and IB_REG (your report is saved verbatim as `roles/verifier/VERIFIER_REPORT.md`). Since
then the campaign ran:
- the robustness readings of both registered releases: `claims/IA_REG_D2R/D2R.json` and `claims/IB_REG_D2R/D2R.json` (bank IPR1_ROB);
- IA's first revision, IA_OSG (card `cards/IA_OSG_R1.md`, code `ipr/ia/osg.py`, fit records `ia/FIT_R1_V0.json` /
  `ia/FIT_R1_V1.json`), read under the same rules on bank IPR1_R1 (`claims/PLAN_IA_R1.json`, `claims/IA_R1/DECISIONS.json`; tag
  ADAPTIVE_R1);
- possibly its strongest-release follow-up: `claims/STRONGEST.json`, bank IPR1_ROB_R1 and `claims/IA_R1_D2R/D2R.json`, when present;
- the beside records `claims/BESIDE_IPR1_R1.json` and the data-rich reader record `rich/RICH_IPR1_R1_RICH.json` (descriptive).

Check, with your own code under `roles/verifier/code/` (new files are fine; do not import `ipr.evaluate`, `ipr.devreaders`,
`ipr.rules`, `ipr.claims` or `nrm.rules` for recomputation):
- **A1. Integrity** of every configuration saved since your report:
  - release, wire and model hashes;
  - the card's commit before each save and before the first fitting read with that card;
  - the deployment function recomputes every saved MODEL_SELECT array bitwise;
  - the income output of every IA_OSG* configuration equals CTL_TWIN's bitwise, in every role.
- **A2. Selection and brackets** re-derived from the fit records:
  - k* under S3 with the scan rule;
  - the SBC / SPEND / RO bracket points by the PROTOCOL 4 rule;
  - the BLIND rung;
  - the consistency record of the recomputed single-step statistics.
- **A3. Utility:** the 24 clauses + 4 guards and paired costs of every new member against `EVAL_MS.json`; G1 at the point; the two-SE
  reserve.
- **A4. Grid replays** of IPR1_R1, and of IPR1_ROB_R1 if present, as for IPR1_MAIN:
  - winners;
  - SCORE AUCs from the cached reader scores;
  - bootstrap;
  - paired cells and SEs;
  - seed effectiveness.
- **A5. Decisions:** IA_R1's D1-D6 and every D2 conjunct, the adaptive tag from the card's commit time, and grid validity.
- **A6. Robustness:** both registered D2R records, and IA_R1's if present: refit effectiveness and the per-comparator labels.
- **A7. Access:**
  - every label read up to your check: after the receipt, inside running units, policy-conforming;
  - fitting reads carry their committed card;
  - no MODEL_SELECT read from a fitting module.
- **A8. Code drift:** the queue's `code_changed_during_run` field for every unit since 03:11Z, and what changed.

Report each check as PASS / FAIL with numbers, observations, the exact-versus-approximate split and your CPU seconds.
- **Labels:** read only through `ipr.common.load(..., 'verify', ...)`, inside queue units (`python3 -m ipr.jobqueue run ...`).
- **Light work:** label-free arithmetic may run outside the queue if you state its CPU.
- **Files:** write only under `roles/verifier/`; commit nothing.
- **Code timing:** `roles/verifier/code/` is part of the queue's hashed code manifest. A code file written or edited while a campaign
  unit runs flags that unit `code_changed_during_run` (your `v6_seeds.py`, written at 03:20:24Z, flags both IPR1_ROB grid units).
  Write and finish your new code files at the start, before your first queue unit, and record their write times.
- **Report:** return the report as your final message (do not write a report file). Any time you write must come from `date -u` read
  first, never ahead of the clock.
