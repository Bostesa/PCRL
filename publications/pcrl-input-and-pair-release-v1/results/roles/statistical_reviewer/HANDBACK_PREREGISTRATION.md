# Statistical reviewer's hand-back of the pre-registration review (verbatim)

Saved by the coordinator after the reviewer's completion on 2026-10-22 (RULES.md completed 02:08:15Z by the reviewer's `date -u`),
verbatim from the statistical reviewer's final message (a subagent of this campaign; its words, not the coordinator's). The report it
asked to be saved is `REPORT_PREREGISTRATION.md` in this folder (saved verbatim from the same message). One change: the absolute
worktree path in the file list is shortened to the repository-relative path (a personal path would be removed by the publication
sanitizer).

---

## Statistical reviewer hand-back: IPR1 pre-registration review and frozen rules

The pre-registration review is finished and the IPR1 rules are frozen. `python3 -m ipr.test_rules` passes 28 of 28 (exit 0, invented data only). I read no label, nothing under `.private/`, nothing of MODEL_SELECT, no reader cache and neither designer's directory. I made no commit and started no background process.

**One thing for you to do:** the harness refused my write of `REPORT_PREREGISTRATION.md`. Its full text is at the end of this message. Please save it verbatim with attribution to `results/pcrl_input_and_pair_release_v1/roles/statistical_reviewer/REPORT_PREREGISTRATION.md`, as was done for the DDR1 report.

### Your ~02:00Z note
Both cases are now general rules, disclosed in `RULES.md`:
- **SEX-blind operation (declared `operation_sex_blind`):**
  - SBC and RO are labelled `DEGENERATE_BY_CONSTRUCTION`, and neither path is built.
  - D2 gates on TWIN and SPEND only, plus targeting against TWIN and the SPEND bracket.
  - The met label is `SEX_BLIND_IMPROVEMENT_BEYOND_TWIN_AND_SPEND`, quoted with the sentence "no SEX-targeting claim is possible".
  - A generic ZTILT path, if built, is reported beside and never gates.
  - If an SBC path or a substituted RO path is given anyway, the reading is `INCOMPLETE`.
- **D5 when the sequential orders collapse (`joint.collapsed`, declared in the card):**
  - An order equal to the separate design gives the reduced test: NR G6's own label, with the separate comparator standing in and the flag `D5_REDUCED_<ORDER>_EQUALS_SEPARATE`.
  - An order equal to the joint design gives `NOT_APPLICABLE_JOINT_EQUALS_SEQUENTIAL`.
  - Without a declaration, an order that was not built reads NR's `NOT_ASSESSED_MISSING_CONTROLS`.

### REQUIRED findings
1. **RO path definition.** The draft's RO definition is ambiguous. SUBSTITUTED means the operation re-fitted with a cross-fitted release-only surrogate in place of SEX, P(SEX | the twin's released log-probabilities). GENERIC_ZTILT is required when the operation is itself release-only, and is otherwise only a declared fallback. The card records `ro.kind` and `ro.operation_release_only`.
2. **Release-only operation.** It gets its own met label, `RELEASE_ONLY_MAP_BEYOND_MATCHED_CONTROLS`. Every quoted D2 label carries its scope sentence. Partial labels are `PARTIAL_<passed>_ONLY` in the order TWIN, SBC, SPEND, RO.
3. **SEX-blind operation.** Degenerate SBC and RO, D2 on TWIN and SPEND only, its own met label (your note).
4. **Planned resolution.** The draft's "0.02-0.025 is the smallest that resolves" is wrong.
   - The realized se_agg of 0.009-0.018 gives a per-comparison threshold of 0.016-0.036, and about 2.84 se_agg is needed for 80% power.
   - D2's power is about 0.24 / 0.56 / 0.90 at a true benefit of 0.025 / 0.035 / 0.050 against every comparator.
   - It is about 0.29 when the release-only control absorbs part of the gain (0.035 against the others, 0.015 against RO).
5. **Grid fineness.** The 25% rule moves to the saved points: lower point at least 0.75 s, upper point at most s + 2 SE + 0.25 s. It is checked on MODEL_FIT before any MODEL_SELECT read, does not apply when s ≤ 0, and is reported only (`grid_fineness`).
6. **D5 records.** The collapsed-order rules above, plus: a declared D5 comparator whose grid entry or cost is missing reads `INCOMPLETE`, not NOT_SUPPORTED. D5 carries D2's gate labels.
7. **Revisions and RO.** A revision cannot reuse a substituted RO path.
   - Budget: the planned allocation is 40 configurations, leaving 8.
   - A revision needing new SBC and substituted RO paths costs 10, so it does not fit.
8. **Strength records.** Strengths may be component vectors, fixing DDR1's deviation 8. `CAND_STRENGTH_DIFFERS_BY_VERSION` and `SBC_CONSTRUCTION_DIFFERS_BY_VERSION` are computed from the records.
9. **Families.** Only IA, IB and FS_<name> are accepted; CTL is never a reading. Only revision 0 of IA or IB can be registered.
10. **D2R with three paths.** D2R's decisive set covers the brackets of all three paths, and an unrecorded bracket makes it `INCOMPLETE`. The grid-validity minimum set is CAND, TWIN and the saved points of the gating paths.

### ADVISORY findings
11. Fallback when a substituted RO path cannot reach CAND's spend (extend its grid, or use the declared ZTILT fallback).
12. Which parts of "matched" (i) apply to DQ and ZTILT.
13. A shared twin needs identical selection effort; an input-side family that re-tunes needs its own twin.
14. A control at the same strength shared by both families is saved once and counted once.
15. Select D5 comparators as the smallest grid strength whose MODEL_FIT spend is at least CAND's.
16. Give registered predictions per D2 label; by finding 4, a D2 prediction above about 0.3 needs an expected benefit of about 0.035.
17. The claims builder must map READER_SPLITS keys V<v>k<k> to S<v>k<k> and pass the bootstrap seeds. The +900 base and 2026102300 appear in no earlier split file.
18. Other derived `ipr/` modules may still have stale DDR1 docstrings; modules should call `ipr.rules` rather than re-implement rules.

### Tests and completion
- **Tests:** 28 of 28 PASS, exit 0, about 0.63 CPU-s. New tests cover:
  - the RO declaration and the release-only met label
  - the SEX-blind case
  - D5 with NR G6, including collapsed orders
  - grid fineness, families, component strengths and the by-version flags
  - three-path D2R
- **Mutation check:** an in-memory check caught 22 of 22 seeded single-line bugs; it changed no file.
- **Unchanged:** `nrm/rules.py` (d5df8686...) and the NRM1 `RULES.json` (84f18776...).
- **Completed:** `RULES.md` at 2026-10-22T02:08:15Z (`date -u`); started 01:40:07Z.

Files are in the worktree root:
- `results/pcrl_input_and_pair_release_v1/roles/statistical_reviewer/RULES.md`
- `results/pcrl_input_and_pair_release_v1/roles/statistical_reviewer/RULES.json` (SHA-256 4c235046c57d8175...)
- `ipr/rules.py` (4ef5fa35e68676a8...)
- `ipr/test_rules.py` (a5e11220eee4bc87...)
- `REPORT_PREREGISTRATION.md` still to be saved by you (text below).
