# Protocol deviations, amendments and disclosures

Every departure from PROTOCOL.md, every dated amendment (made before the stage it affects), every failed attempt and every timing
disclosure is listed here in the order it happened, with the `date -u` time at which the entry was written. Earlier entries are never
edited; corrections are new entries.

## Registration-time disclosures (written before any new real-label computation)

1. **Worktree directory.** The campaign branch research/pcrl-controlled-training-size-v1 was created (23:53:57Z) from the
   correction-transfer research tip 507024649 inside the worktree directory `.claude/worktrees/pcrl-correction-transfer-stability-v1`,
   because this session is bound to that directory. The previous branch is unchanged; only the directory name is inherited.
2. **Known source margins.** The coordinator and the statistical reviewer had read the published source records of the full-data
   A1_PO_16 package (WITNESS_SELECTION.json of the occupation-headroom campaign: it met the full reserve, minimum occupation-side ratio
   4.30) and the correction-transfer training-size diagnostic before this registration. The trigger and the strength-rule constants
   were therefore set with that knowledge (RULES.md states which constants depend on published source values).
3. **Stage B epoch order at S50 (control recipe adaptation, decided before any fit).** The replication's Stage B loop draws each
   epoch's 40 batches of 256 rows from one permutation of the training rows, which assumes at least 10,240 people. Every S50 subset
   has fewer (9,934-10,046), so the unchanged loop would leave a short last batch (S50_O2: 62 rows; S50_O3: 30) or, for S50_O0 /
   S50_O1, a short batch and one EMPTY batch per epoch (a step with no task gradient). For subsets below 10,240 people the control
   extends the same permutation cyclically to 10,240 positions, so every update has exactly 256 rows and every subset person is used
   once before any is reused; for 10,240 or more people the unchanged loop runs (bitwise equal; task-refit investigator's selftest).
   The update count (20 x 40 x 256) is unchanged, so smaller subsets are passed over more often (recorded per fit).
