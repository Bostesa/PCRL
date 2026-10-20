# Engineering / claims reviewer, wave 2 part 1: code (saved verbatim with attribution)

Author: the engineering / claims reviewer of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator after its 16:47Z header
on 2026-10-20 (the coordinator's next `date -u` read follows in the coordinator log), in answer to the coordinator's 16:40Z request.
Read-only; no sub-agents. Saved verbatim; nothing below the line was edited.

Coordinator note (not part of the report): ENG2-06 asked for a fix "before the custody run". The custody run had already finished at
16:46:18Z, and both interpreter passes re-scored all six reader pickles bitwise (custody/BACKUP_VERIFICATION.json), so the predicted false
failure did not occur. Adjudication: roles/coordinator/ADJUDICATIONS.md, A5.

---

Engineering / claims reviewer, wave 2 part 1 (code), 16:47Z

None of the findings below is BLOCKING. Two RECORD items should be fixed before their step: ENG2-06 before the custody run, and ENG2-01 before any further claims run.

Everything was read-only and I ran no probes. I did read three label-free saved outputs: the FINAL claims, the decisive-phase records and the FB2_SEX evaluation records.

**What I confirmed:**
- **Deviation-10 fix** (nrm/claims.py:132 and :146): the distinct-input counts now reach `grid_validity`. FINAL shows `READER_SEEDS_EFFECTIVE`, with 28 equalities explained.
- **FB2P rule 2(a)** (nrm/decisive.py:84-100) is exact. It reads the registered Monte Carlo reading, checks S5 before S3, does not require MINF's G1, and returns NOT_RUN when no strength qualifies.
- **C5 MEAN rule** (decisive.py:116-134) is exact: both declared paths must reach DOMINATES and the candidate must be TASK_FEASIBLE; a missing or INCOMPLETE path means not read.
- **Section 3 with the 16:16Z clarification** (decisive.py:152-210) matches:
  - applicability comes from the Monte Carlo B0;
  - Δ comes from single-instance B0 and B1 in the shared bank;
  - se_Δ is the `se_agg` of per-cell SDs of the paired bootstrap-draw differences;
  - the B0_SINGLE_INSTANCE_NOT_RESOLVED flag never changes the label.
- **Access amendment:** committed in c43a7c5a7 at 16:34:09Z, before the only `nrm.readercustody` read at 16:34:18Z (unit NRM_READER_CUSTODY_V1, launched from clean code at c43a7c5a7). Operation 'readers' was logged with entry point `nrm.readercustody:run`. The pickles stay private; their LOOKUP tables are near-individual SEX frequencies, so they must never be published.

**ENG2-01 RECORD (nrm/claims.py:181-184).** Path members without paired costs are silently dropped from both `costs_vs` and the path list before `frontier_comparison`. A missing cost would then produce a label computed on a shorter path instead of INCOMPLETE. That breaks C6 (an INCOMPLETE record is completed first, never read as F1 or as a pass), and `decisive.f1` / `decisive.mean` would read that label. It did not happen in FINAL: all three points of both paths carry costs. Fix: pass every declared member and let `nrm.rules` return INCOMPLETE.

**ENG2-02 RECORD (claims.py:220-221).** When a conjunction over several alternatives fails, the ledger row's alternative is set to None. The selection record then shows G2 = NO_MATCHED_CONTROLS while the claims summary says NOT_MET (FB_SEX_S2 in FINAL). The outcome is unchanged (BELOW_MINIMUM_TIER; it is TASK_INFEASIBLE anyway). Fix: pass the first failing alternative instead of None.

**ENG2-03 MINOR (claims.py:89-93).** Paired costs are merged across `EVAL_MS*.json` in filename order, not by time, so a duplicated comparator would be resolved by tag name. There are no duplicates for FB2_SEX_S3 or S5.

**ENG2-04 RECORD (decisive.py:103-113, `f1`).** It reads only the MINF path. Card v2 §6 (C6) says "DOMINATES at BOTH S3 and S5 against the declared paths", which per C7 are MINF and CTL, and the INCOMPLETE guard covers MINF only. The verdict is unaffected: both paths are NOT_SEPARATED at both strengths, so F1 holds under either reading. Fix: read both paths, or state in the record that F1 is the MINF targeting reading.

**ENG2-05 MINOR (decisive.py:144-145 and :178-179).**
- `f2` maps INCOMPLETE to NOT_READ_INCOMPLETE; read literally, the ruling would say F2 HOLDS. This is the protective direction, and `f2` never ran because MEAN was NOT_FITTED.
- In `perturbation`, a missing c1 gives PERTURBATION_NOT_CONSISTENT; it should be INCOMPLETE. This never ran because FB2P was NOT_RUN.

**ENG2-06 RECORD, fix before custody (nrm/restore_check.py:57-66; custody.py:64-75).** Releases are now split by fit interpreter, but the reader pickles are re-scored in both passes. All six custody readers were fitted under 3.13.7 (torch venv): MLP64 three times, OUT_LR, CATI01 and LOOKUP. The python3 3.14.3 pass re-scores the MLP and logistic-regression readers with a different numpy and possibly a different BLAS build. One non-bitwise score fails that pass, and custody's `all_ok` requires both passes, so the result would be a false custody failure. Fix: skip pickles whose `obj['python']` differs from the pass, and count them in the coverage check.

**ENG2-07 MINOR (nrm/readercustody.py:62-63).** A pair WINNER that is an ignore-one candidate (`IGNORE_income__X`) is not mapped to its single-view recipe. It would fail loudly, and it did not occur here.

**ENG2-08 RECORD (nrm/tables.py:38-40, the section-1 "Pair AUC"; nrm/figures.py:47 and :56, the y axis).** Randomized members are shown from the base instance only (R1 → 0, R2 → 1, replicates excluded). Tables §2-3 and the claims pool the +2/+4 replicates with `mc_cell`. So differences between the section-1 AUCs need not equal the §2 benefits. Fix: label these as "base instance per cell", or pool the replicates per cell before averaging.

**ENG2-09 MINOR (figures.py:58).** The family is inferred from the name prefix ('PRC_' means FA, anything else FB), so other FA names (ORD_*, HQ_*, SHUF_*) would be skipped silently. None is in the banks.

**ENG2-10 MINOR (nrm/evaluate.py:144-145).** The `'normalized'` recompute (value / cap) is correct; it is a clause-only field. Randomized records written before 999e216b0 (for example FB2_SEX_* `EVAL_MS.json` at 15:49Z) keep draw 0's value. No rule, table or figure reads it.

**ENG2-11 MINOR (nrm/ctl/run.py:146-149).** `plan()` checks the base CALIBRATION vectors against the CALIBRATION labels by length only, with no row-id check. Otherwise CTL_REM (bcd9df4fd, ac488d353, 2aa8491a9) is correct:
- only CALIBRATION occupation labels are read;
- spend is predicted out of fold on household folds;
- the level choice uses spend only (Z = 0), with the plan, choice and shares hashes recorded;
- every decision and top probability is kept (a 1e-12 margin; ties are left unchanged);
- the base hash is checked at fit;
- CTL_REM and FB2_BASE were both fitted under 3.14.3, so the restore split covers them.
