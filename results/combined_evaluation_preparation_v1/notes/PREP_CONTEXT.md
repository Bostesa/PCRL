# Shared context — stored-model evaluation preparation (2026-10-02)

Owned worktree: /Users/nathansamson/PCRL/.worktrees/combined-evaluation-preparation-v1
(branch research/combined-evaluation-preparation-v1, based on research/combined-evidence-reconciliation-v1
@07a9ca3ffaf27133b6cf955b3b8d526682961e8d). Package dir: results/combined_evaluation_preparation_v1/.
No AGENTS.md/CLAUDE.md exists in either repo.

AUTHORIZED: preparation, evaluator implementation, synthetic validation, recounts of existing artifacts,
frozen forward passes of existing checkpoints (no gradient updates), private backup.
NOT AUTHORIZED: new defense training or defense refits (incl. closed-form LEACE/projection refits on new
rows), new scientific attacker or utility-probe fitting on real data, opening new data/states/years,
cloud compute, submission metadata changes, merging into main, editing other worktrees/branches/history.
Synthetic fixtures may fit tiny models on synthetic data.

THE EXTERNAL DRIVE (YOTUO) IS NOT MOUNTED today (diskutil shows no external disk). Do not call drive
artifacts lost. Record drive-dependent operations as PENDING with the exact archive member.
Local substitutes available now:
- Drive inventories (member -> sha256/size): /Users/nathansamson/storage-relocation-20260930/inventories/*.json.gz
  and RESTORE.md / manifest.json in /Users/nathansamson/storage-relocation-20260930/.
- Members extracted yesterday (exact bytes, verify by sha256 vs inventory) under session scratch
  /private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad/recon/
  (inventory/dgx, inventory/parts/ck, verification/dg_drive, verification/D_drive, verification/A, verification/B).
- Private stable cache (not git): ~/PCRL_eval_cache_private/checkpoints/v2_{adult,hmda}_s0_final.pt
  (PCRL Round 4 seed 0 = the encoders durable-guarantees loads; sha256 1cfc2fef…, e29d0367…, verified vs inventory).
- Raw data on laptop: /Users/nathansamson/PCRL/data/{adult,hmda_processed,hmda_raw,diabetes,diabetes_processed,folktables}
  (already-used development data only). CelebA images NOT local (relocated).
- PCRL code: /Users/nathansamson/PCRL (primary checkout belongs to another session — read refs via git -C ... show;
  your own worktree is the place to add code). Python with torch: /Users/nathansamson/PCRL/.venv/bin/python;
  system python3 has numpy/sklearn/scipy.
- durable-guarantees read-only clone: /private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad/dg @956f5c88.

READ FIRST (reconciled evidence, in this worktree):
results/combined_evidence_reconciliation_v1/{UPDATED_MEETING_BRIEF.md, MISSING_FILES_EXACT.md, EVIDENCE_INVENTORY.csv,
REVIEW_RESPONSE_MATRIX.csv, VERIFICATION_REPORT.json, CORRECTIONS_TO_PREVIOUS_ASSESSMENT.md, notes/inventory/*}
and results/combined_empirical_preparation_v1/EVALUATION_PROTOCOL_DRAFT.md (+METHODOLOGY_IMPROVEMENTS.md).
Private reviews/transcript: ~/Documents/PCRL_private_review_sources_20261001/ (never copy into the repo; public
pseudonyms N-R1..3, A-R1, A-R2, A-AI).

EMPIRICAL QUESTION: When a release passes a method's stated protection check, what can recipients still recover
under explicitly specified access conditions, and what task performance does the release preserve?
Separate: fitting-sample compliance | held-out generalization | recovery outside a guarantee's scope |
prediction-output leakage | recovery from combined releases. Not every successful attack is a broken theorem.

OUTCOME CATEGORIES (use these exact labels):
C1 implementation fails its own fitting-sample check; C2 fitting-sample check passes but does not generalize to
held-out rows under the tested procedure; C3 attacker or release surface outside the stated guarantee recovers the
attribute; C4 a claimed population guarantee is contradicted under its actual assumptions; C5 evidence insufficient
or quantity not estimable.

WRITE RULES: write only in your role dir results/combined_evaluation_preparation_v1/notes/<role>/ unless your prompt
names other owned paths. The harness may refuse files named like reports (SUMMARY.md, FINDINGS.md); use data files and
*_notes.md, and put refused content in your final reply. Never print script bodies, credentials or env secrets. No
raw individual data, checkpoints or private reviews in the repo. Final reply <= 700 words, EARLY FLAGS first.
