# Statistical reviewer hand-back, pre-registration (saved verbatim by the coordinator, about 04:58Z on 2026-10-24)

The report appended to this hand-back is saved separately and verbatim as `REPORT_PREREGISTRATION.md`.

---

Statistical reviewer hand-back (SAR1). I worked from 04:28:04Z to 04:56:14Z (`date -u`). Everything was label-free and used invented data only. I read no label, no MODEL_SELECT output, no investigator file, no card and nothing under `.private/`. Estimated CPU is under 20 CPU-s (each test run takes about 0.23 CPU-s; the mutation check and the power calculation ran in my scratchpad). I spawned no agents, ran no git and killed no processes.

## Files written (SHA-256 at 04:56:14Z)
- `results/pcrl_strong_anchor_release_v1/roles/statistical_reviewer/RULES.md`: ddae791da6655c643b93b2a36f092a9255833642b56f1e1531508d9e7af2948a
- `results/pcrl_strong_anchor_release_v1/roles/statistical_reviewer/RULES.json` (schema SAR1-rules-v1): 41ba919ebb714e1bb4f7943a57b189e719364a6ab1e745d26e1aba9b13c16498
- `sar/rules.py` (rewritten, not renamed; imports `nrm.rules` unchanged): 96ab52d3edb0380f8a6513ca558fe19cdd31fe831080ad988ddbf0e6bea84e1b
- `sar/test_rules.py` (rewritten): 224a23ae25cdc264ceb0a5ba15d28ae755983c1cfbc84525425ba7c681cda651
- Inherited files are unchanged and are checked by `load_rules()`: nrm/rules.py d5df8686..., NRM1 RULES.json 84f18776...
- **The harness refused my write of `REPORT_PREREGISTRATION.md`** ("subagents return reports as text"). Its full text is at the end of this message. Please save it verbatim with attribution in `roles/statistical_reviewer/`, as was done for the IPR1, DDR1 and FRC1 reports. RULES.md and RULES.json already say that it was returned this way.

## Tests
- `python3 -m sar.test_rules`: **23 of 23 PASS, exit 0** (one numerical thread, about 0.22 CPU-s). pytest is not installed.
- In-memory mutation check (scratchpad `mutate.py`; no repository file changed): 24 single-line seeded bugs, **24 of 24 caught**. The first pass caught 21; I strengthened three tests (a wrong START with complete inputs, a pair-point failure that is not resolved, a CONT origin without a declared CONT).
- The mechanical copy's tests still asserted the IPR1 reader seed (2026102105), so they could not pass as they were.

## What the rules do (summary; full detail in RULES.md)
- **D1:** G1 at the point is the gate. Reported separately and never gating: the confidence conjunction (PASS / UNRESOLVED / VIOLATION per condition, own household-clustered SE, MC SE in quadrature, no floor, z = 2) and the reserve. SUCC's men's guard reads UNRESOLVED.
- **UM:**
  - START's binding condition per version is the condition with the smallest reserve ratio. It must equal the plan's declaration and the frozen table: SUCC guard|sex0 / guard|sex0; NEURAL_RAW guard|sex0 / income|U0|brier_excess.
  - A "resolved improvement" means the pooled paired CAND − START difference has an upper bound below 0 and both versions are below 0. The guard maps to `occupation|men_brier`.
  - MARGIN_RECOVERED also needs G1, the recovery profile retained, and the whole-release point margin not below START's.
- **D2:** gates on START, CONT (only when the card declares it) and the CTRL frontier. The CTRL path is its origin (START or CONT, declared) plus the saved points, read with `nrm.rules.frontier_comparison`.
  - A missing bracket reads NOT_SEPARATED with reason NO_BRACKET. The residual spend mismatch is reported.
  - Reader targeting (d) is kept from IPR1.
  - Partial outcomes are labelled PARTIAL_<passed>_ONLY, with a statement saying what was not separated.
- **D2R** is read at CAND with the registered decisive set. A sign change on any comparator is flagged and carried into `outcomes.statement`. At least one of R3 / R4 is required.
- **D3** lists which references passed and which failed. **D4** is unchanged and never weakened. **D5** applies only if a card claims joint design; otherwise not claimed. **D6** stays NOT_CLAIMED unless separately reviewed.
- **Outcome classes:** METHOD_CANDIDATE / COMPONENT_GAIN_NOT_COMPETITIVE / REFERENCE_ADVANTAGE_NOT_ATTRIBUTED / USEFULNESS_MARGIN_ONLY / FEASIBLE_NO_RELIABLE_GAIN / NO_OUTPUT_CHANGE, plus the inherited infeasible and incomplete classes. Every reading also reports the three outcomes separately in `outcomes`.
- **Grid:** offset +1000; offsets 300–900 forbidden (140 seeds); bootstrap base 2026102400. A new few-valued explanation applies when a view has at most 320 distinct inputs and the recipe's seed is shown to work elsewhere in the same cell. Such equalities are counted once (n_eff) and do not invalidate the grid.
- **Families:** FA, FB, FS_<name>. Only revision 0 of FA / FB is registered.

## REQUIRED / ADVISORY changes to PROTOCOL.md
Reviewed version: ee5a2a15... at 04:31Z. It is now db87f8b8..., modified at 04:32Z and one byte shorter; sections 5–8 were unchanged when I re-checked.
- **R1:** Section 8 must state D2's targeting conjunct (d). The rules keep it.
- **R2:** Adopt the UM definition in RULES.md 3, including my added clause that the whole-release point margin may not fall below START's.
- **R3:** Specify the CTRL path:
  - its origin, and the SE used in "+2 SE";
  - recording the full MODEL_FIT grid;
  - the budget: two saved CTRL points are 4 specifications, so CAND plus CTRL already makes 6, and a declared CONT does not fit DESIGN_BRIEF's limit of 6.
- **R4:** Each card declares `cont_declared` and `ctrl_origin` before its first fit.
- **R5:** Define the few-valued rule (at most 320 distinct inputs, seed shown effective elsewhere) and record distinct-input counts.
- **R6:** D2R scope and the sign-change headline rule.
- **R7:** Use the rules' outcome-class names and the three-outcome statement.
- **R8:** Specify the SE and z of the confidence conjunction.
- **R9:** Commit each reading's plan file before its first MODEL_SELECT computation.
- **A10–A15 (advisory):**
  - A10: the bootstrap seeds coincide with IPR1's v1 seeds; disclose it (harmless).
  - A11: the point rules are lenient, since a true 0.010 worsening passes about a third of the time.
  - A12: D3's SUCC / NEURAL_RAW entry is the same as D2's START comparison, not a second piece of evidence.
  - A13: a CAND bracketed at its origin adds no attribution.
  - A14: carry the 8% version-dominance caveat where it applies.
  - A15: PROTOCOL 14 should say that sar/rules.py and sar/test_rules.py were rewritten, not renamed.

## Open questions for the coordinator
1. Do you adopt the whole-release point-margin clause in UM?
2. Do you keep targeting (d) as a gate?
3. Which budget reading applies: 8 specifications with CONT, or only the upper CTRL point?
4. Is 320 the few-valued bound you want?

If any is declined, the rules must be changed and re-frozen before registration.
