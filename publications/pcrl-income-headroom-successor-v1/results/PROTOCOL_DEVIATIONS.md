# Protocol deviations and engineering records

Items are numbered in the order recorded. This file was first committed at the freeze `930235e9e` (03:03:35Z), with items 1, 2,
4, 5 and 3 in that order; item 6 and an order note were added in `f615cbf53`, item 7 in `d45a17112`, and item 8 and this
header with the engineering and claims review. Item 3 records safeguards made before the first fit; their substance (reviewer
requests S1 / S2) is registered in PROTOCOL.md §2 and `roles/statistical_reviewer/DEV_GRID_REVIEW.md` at `74376912c`, but the
text of item 3 was first committed at the freeze. No committed item has been deleted or changed in substance; corrections are
marked as such.

**1. The registration receipt was pushed 24 seconds after the first new fit started (timing; content unaffected).**
PROTOCOL.md says "a sanitized receipt is pushed to the publication branch before the first fit".
- The registration commit `74376912c` was made at 02:35:11Z. The label gate in `ihs.common.load_role` opens on the committed
  registration files, and investigator P's first two units (`IHS_P_P1_PAIR_B40_S0_V1`, `IHS_P_PC_TWIN_B40_S0_V1`) started at
  02:35:25Z.
- The receipt `77dd6710f` (built from a `git archive` of `74376912c`, 34 files, 0 leak-scan problems) was committed at 02:35:45Z,
  pushed at 02:35:49Z and verified on the remote.
- So every registered rule was committed, hashed and unchangeable before the first fit; only its public push came 24 seconds
  later. The receipt is byte-for-byte the committed registration state. The gate should have opened on the pushed receipt; it
  opened on the commit.

**2. The label gate was added after the two pre-registration reference units (recorded in `ihs/common.py`).**
`IHS_REFERENCES_S0_V1` / `S1_V1` (started about 02:05Z; their ledger rows were appended at 02:09:19Z / 02:09:21Z) read
MODEL_SELECT labels and SEX to evaluate the frozen historical releases on the registered grid; `IHS_INHERITED_V1` summarized them.
No release was fitted before the registration. The gate refusing further MODEL_FIT / CALIBRATION / MODEL_SELECT label reads
before the committed registration was added at about 02:12Z. The verifier reproduced the reference cells bitwise under the final
evaluator (its check 5).

**3. Development-evaluator safeguards S1 / S2 (statistical reviewer, before the first fit).** Atomic proxy-cache writes and a
refuse-to-run fingerprint of the reader code and library versions (`DEV_EVALUATOR_FINGERPRINT.json`) were added to
`ihs/evaluate.py` after the reference units ran. They change no number; the reference cells were not recomputed. The PROTOCOL.md
§2 sentence on interpreter versions was corrected before registration (the validation's reader lock records numpy 2.4.2, the
`.venv` interpreter; the development grid runs under Homebrew Python with numpy 2.4.4).

**4. Platform block: subagent report files.** When investigator O tried to write `roles/investigator_o/REPORT.md`, the harness
refused with the exact message: "Subagents should return findings as text, not write report files. Include this content in your
final response instead." The investigator stopped that action and returned the full report text in its hand-back; the
coordinator saved it with a header saying so. Investigators P and H met the same refusal; the engineering and claims reviewer did
not attempt the write because its operating instructions forbid subagent report files; the independent verifier's report writes
were not refused. No result, number or conclusion was changed by this; the investigators' cards, code and private run records
were written by the investigators themselves. Correction: investigator O's `CARD.md` (line 148) dates its refusal "~03:20Z";
that line was already in the freeze commit (03:03:35Z), so the refusal happened before 03:03Z and the card's time is wrong.

**5. Nomination code safeguards from the reviewer's lock-encoding check (before the official nomination).** The reviewer's
`LOCK_ENCODING_CHECK.md` passed the decision logic and requested engineering changes (N1-N7, C1-C6, L1-L3, F1-F2). They were
made before the official nomination and before any lock: count only complete evaluation rows and require the EVAL_MS record on
disk to reproduce the counted ledger row; check the recorded inputs and the reserve rule hash; report reference winners per
cell, own-twin / component-control cells, the frontier and the best diagnostic package; TARGET_POINT_ONLY requires a bound at or
below 0.020; key-by-key bank comparison with sides; controls checked on ATTACK_VALIDATION and DEV_REPORT; alias-safe bank and
endpoints (references inserted first); the family-D roster additions; a predefined verifier refit roster; the AV gap to the
original winner; stored CATI designs. None changed a dry-run label.

**6. Engineering record: the coordinator's DEV_REPORT unit is recorded FAILED although its evaluation completed.** Unit
IHS_REPORT_V1 (03:08Z) read the DEV_REPORT truth once under the registered event DEV_REPORT_TRUTH_REREAD_BY_IHS_COORDINATOR
(03:08:07Z; appended to this campaign's log and to the lineage log), computed both scorecards and wrote every output
(`report/DEV_REPORT_RESULTS.json`, `report/PRIMARY_ENDPOINTS.csv`, the private endpoint draws and household contributions). Only
the module's final console print then raised `KeyError: 'reader_sensitive_keys'` (`report.py`:234; a key renamed to
`reader_sensitive` during the reviewer's C2 change, in the `__main__` block after `run()` had returned), so the queue recorded
return code 1. The results were read from the written files. There was no second DEV_REPORT read, and `ihs/report.py` was not
edited afterwards: it is hashed in ANALYSIS_LOCK, and the `__main__` print is not part of the evaluation. The verifier's replay
reproduced all four outputs exactly.

**7. Engineering records after the DEV_REPORT read (verifier items V7-D1, V10-F9, and the verifier's code files).**
- `ihs/deploy.py` was edited after REPORTING_LOCK and the DEV read (commit `f615cbf53`, 03:15Z, +3 lines): family H packages now
  also hash the STRONG learner's step-1 files (`runs/H/STEP1/S<s>/HGB.pkl`, `STEP1.json`) in `PACKAGE.json`. The deploy code is
  not covered by any lock; no evaluated output changed (the verifier's check 7 redeployed all 12 DEV_REPORT vectors bitwise).
- `NOMINATION.json` (hashed by FINALIST_LOCK, so not edited) fills a `best_diagnostic_package` field (CMB_P3JR_O1SEXBAL) even
  though a finalist was nominated; NOMINATION_RULE.md defines that term only for the case where nothing qualifies. The documents
  call it "the best development pair score among qualifying configurations".
- Commit `f615cbf53` swept in three of the independent verifier's code files while the verifier was still working (a `git add`
  of the results directory by the coordinator, not the verifier's action); the verifier's final report and code were committed in
  `d45a17112`.

**8. Engineering and claims review, and a gate that is procedural (after the DEV_REPORT read).**
- The engineering and claims reviewer (a separate agent) checked every number, label and timeline entry of the documents and
  raised 16 must-fix wording, attribution, framing and provenance items (two material: an over-strong "established at clause
  level" description of the H_LJ benefit, and a reader-refit statement that did not bear on the benefit). All were corrected
  before publication; no number, label or decision changed (`roles/claims_reviewer/CLAIMS_REVIEW.md`,
  `CLAIMS_REVIEW_DISPOSITION.md`).
- `ihs.common.dev_report_truth` checks that REPORTING_LOCK is committed (`common.py`), while PROTOCOL.md §3 says "committed and
  published". The receipt `9ad0fb25d` was pushed at 03:07:55Z, before the 03:08:07Z read, so there was no timing deviation;
  publication before the read was procedural, not enforced by code (the same pattern as item 1).
