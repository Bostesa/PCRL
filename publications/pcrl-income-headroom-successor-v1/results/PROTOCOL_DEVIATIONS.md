# Protocol deviations and engineering records (append-only)

**1. The registration receipt was pushed 24 seconds after the first new fit started (timing; content unaffected).**
PROTOCOL.md says "a sanitized receipt is pushed to the publication branch before the first fit".
- The registration commit `74376912c` was made at 02:35:11Z. The label gate in `ihs.common.load_role` opens on the committed
  registration files, and investigator P's first two units (`IHS_P_P1_PAIR_B40_S0_V1`, `IHS_P_PC_TWIN_B40_S0_V1`) started at
  02:35:25Z.
- The receipt `77dd6710f` (built from a `git archive` of `74376912c`, 34 files, 0 leak-scan problems) was pushed at 02:35:49Z
  and verified on the remote.
- So every registered rule was committed, hashed and unchangeable before the first fit; only its public push came 24 seconds
  later. The receipt is byte-for-byte the committed registration state. The gate should have opened on the pushed receipt; it
  opened on the commit.

**2. The label gate was added after the two pre-registration reference units (recorded in `ihs/common.py`).**
`IHS_REFERENCES_S0_V1` / `S1_V1` (02:05Z) read MODEL_SELECT labels and SEX to evaluate the frozen historical releases on the
registered grid; `IHS_INHERITED_V1` summarized them. No release was fitted before the registration. The gate refusing further
MODEL_FIT / CALIBRATION / MODEL_SELECT label reads before the committed registration was added at about 02:12Z.

**4. Platform block: investigator report files.** When investigator O tried to write
`roles/investigator_o/REPORT.md`, the harness refused with the exact message: "Subagents should return findings as text, not
write report files. Include this content in your final response instead." The investigator stopped that action and returned
the full report text in its hand-back; the coordinator saved it verbatim (with a header saying so). The same handling applies to
any other role whose report file write is refused; each such file says so in its header. No result, number or conclusion was
changed by this; the investigators' cards, code and private run records were written by the investigators themselves.

**5. Nomination code safeguards from the reviewer's lock-encoding check (before the official nomination).** The reviewer's
`LOCK_ENCODING_CHECK.md` passed the decision logic and requested engineering changes (N1-N7, C1-C6, L1-L2, F1-F2). They were
made before the official nomination and before any lock: count only complete evaluation rows and require the EVAL_MS record on
disk to reproduce the counted ledger row; check the recorded inputs and the reserve rule hash; report reference winners per
cell, own-twin / component-control cells, the frontier and the best diagnostic package; TARGET_POINT_ONLY requires a bound at or
below 0.020; key-by-key bank comparison with sides; controls checked on ATTACK_VALIDATION and DEV_REPORT; alias-safe bank and
endpoints (references inserted first); a predefined verifier refit roster; the AV gap to the original winner; stored CATI
designs. None changed a dry-run label.

**3. Development-evaluator safeguards S1 / S2 (statistical reviewer, before the first fit).** Atomic proxy-cache writes and a
refuse-to-run fingerprint of the reader code and library versions (`DEV_EVALUATOR_FINGERPRINT.json`) were added to
`ihs/evaluate.py` after the reference units ran. They change no number; the reference cells were not recomputed. The PROTOCOL.md
§2 sentence on interpreter versions was corrected before registration (the validation's reader lock records numpy 2.4.2, the
`.venv` interpreter; the development grid runs under Homebrew Python with numpy 2.4.4).
