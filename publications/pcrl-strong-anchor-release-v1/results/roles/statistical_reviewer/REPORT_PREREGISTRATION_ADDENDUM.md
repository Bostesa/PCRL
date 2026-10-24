# Addendum to the pre-registration report (statistical reviewer; saved verbatim by the coordinator, about 05:11Z on 2026-10-24)

The harness does not let the reviewer write report files; the text below is the reviewer's, unchanged.

---

Addendum (statistical reviewer, 2026-10-24, 05:07Z): ALT path and declared selection data, added at the coordinator's request before registration.

Status of findings 1-15:
- Adopted: 1 (targeting gate), 2 (UM point-margin clause), 5 (few-valued bound 320).
- Resolved: 3. The budgets are FA 8 and FB 12, which is 20 of 36, of which 16 are control specifications.
- Still REQUIRED: 4, 6, 7, 8, 9. The ADVISORY findings stand.

16. **REQUIRED.** PROTOCOL 6 needs an ALT row: "the starting release's own protection operation continued to matched spend; card-declared; origin START; the same bracket rule and frontier conjuncts as CTRL; gating when declared". Section 8 needs ALT_FRONTIER added to D2's comparison list and to the partial order.
17. **REQUIRED.** In PROTOCOL 6's CTRL row (and the new ALT row), replace "MODEL_FIT out-of-fold paired total-NLL spend" with: "spend on the card's declared selection data (MODEL_FIT out-of-fold, or CALIBRATION task-label paired costs when START is in-sample on MODEL_FIT; never MODEL_SELECT), with the SE from the same data". In section 4, "MODEL_FIT-only selection rule" becomes "selection rule on the declared selection data".
18. **REQUIRED, plan file additions:**
   - `alt_declared`.
   - The ALT points: ids, checkpoint or strength, paths, wire SHA-256 per version.
   - The full ALT grid with spends and SEs.
   - `selection_data` per path and `start_in_sample_on_model_fit`.
   - FA's disclosure that H3 income and Ucal are in-sample on CALIBRATION.
   - Paired costs vs CONT whenever CONT lies on a frontier path, either as origin or as an alias of a CTRL point (for example FA's full-scale λ = 0 point).
19. **ADVISORY.** FB's RAW-J E2 checkpoint equals NEURAL_RAW bitwise on MODEL_FIT. Record it as `alias_of: {<id>: 'START'}` only after a MODEL_SELECT wire-hash check. If it aliases, it collapses into the origin and costs no reader cells.
20. **ADVISORY, power** (normal approximation; SEs 0.006 / 0.008 / 0.008; correlation 0.5): FB-like D2 with START, CTRL and ALT has power 0.26 / 0.53 / 0.78 / 0.93 at δ = 0.015 / 0.020 / 0.025 / 0.030. With CONT as well: 0.22 / 0.49 / 0.76 / 0.92. If the RAW-J continuation reproduces half the gain, power is 0.20 at δ = 0.020 and 0.32 at 0.025. So PARTIAL_START+CTRL_FRONTIER_ONLY is a likely FB reading unless the new objective adds a distinct gain.
21. **ADVISORY.** CALIBRATION has 2,982 people, so its bracket SEs are wider and FA's upper CTRL point will tend to spend more than CAND on MODEL_SELECT. That makes the comparison conservative for CAND, and fineness may read COARSE_ABOVE. The in-sample H3/Ucal bias only affects which points are saved and whether they bracket. Q* is always read on MODEL_SELECT by NR's rule.
22. **ADVISORY.** After the initial 20 specifications, 16 remain. A revision with its controls costs 6 (FA-like: CAND 2 + CTRL 4) to 10 (FB-like: CAND 2 + CTRL 4 + ALT 4). That allows about two revisions in total, so plan revisions with that in mind.
