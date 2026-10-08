# M3 simultaneous inference contract

Status: implementation and data-free tests complete; M4 input integration,
source-specific S3 timing for the corrected seed code, and M5 independent
replay remain pending. No Adult inference has run.

The fixed family has **207 named clauses**: each of J and nine control rules
has accuracy, natural-log NLL and multiclass Brier margins for both tasks
against U0, Ucal and F (180 clauses); J has coalition benefit and two local
no-worsening margins against each control (27 clauses). Every margin is
oriented so zero or greater is favorable. The coalition threshold is 0.020
and each local allowance is 0.010. Each task margin uses 0.010 accuracy,
0.010 nats NLL or 0.005 Brier. All thresholds are inclusive.

FIT-invalid rules retain their clause names and receive
`UNRESOLVED_INVALID_FIT_RULE`. They do not gain an interval or favorable
comparison. The active numeric family is fixed by FIT status before INNER
arrays are opened. A resample without both SEX classes makes every otherwise
active clause `UNRESOLVED_UNDEFINED_REPLICATE`; no replicate is dropped.

M4 publishes an immutable private per-seed NPZ and endpoint index after
attack fitting, V2 checks and point scoring. The M3 worker checks the live
S3 unit, its M4 DONE parent/output hash, each seed's exact M2 FIT DONE
receipt and control-plan path, NPZ and every array hash before scoring.
The decoder requires the six fixed local reader IDs and their 18-member
coalition bank, including transported ignore-recipient aliases. It
independently recomputes all point task
metrics and every fixed-reader oriented AUC envelope and refuses any
disagreement with M4's aggregate endpoints. It selects the strongest main
and attribution controls separately using M3's frozen exact-tie rule.

For inference, unique identical probability and reader arrays are interned
per seed by exact content hash. Each of 2,048 PCG64DXSM draws resamples the
2,234 INNER groups with replacement, expands them to people, and applies
the **same positions to both paired model seeds, all rules, all references
and all fixed readers**. Each reader's AUC is oriented by the prespecified
positive/reverse aliases, and each view's maximum is recomputed in every
draw. The PCG64DXSM integer uses all 32 bytes of the public locked seed.
No reader is fitted, tuned or reoriented using INNER.

The 95 percent two-sided interval uses the maximum absolute studentized
deviation across all active clauses. A clause whose every bootstrap value
matches its point within 1e-12 is marked `NUMERICALLY_CONSTANT` with zero
computed width; that is not a structural or population guarantee. The
point-selected strongest control's three privacy clauses are mapped back
into this full family. Simultaneous coverage over every possible control
clause accounts for this rule selection without changing the fitted masks or
reselecting a bootstrap argmin.

These are development intervals conditional on fitted heads, attacks and
reused INNER rows. They are not independent confirmation or post-selection
population guarantees. M5 separately replays endpoints and intervals from
private saved arrays without importing M3 or M4 scoring code.

## Dated prefit correction after resource smoke (2026-10-08)

S3's data-free role-sized smoke on the preceding inference source completed
2,048 draws and 207 clauses in 409.62 charged CPU seconds with 199 MB peak
RSS. Its private output SHA256 is
`cb0885aaeaf9c91f87b706fc5a1b2ef1046a096f36f95415b0b44d8c791b9a23`.
Before any Adult inference, M3 corrected seed initialization to
consume all 32 locked bytes, required exact seed-specific FIT receipt/plan
custody, and checked the six fixed reader identities. The smoke measures
resource scale for the same arithmetic; it is not a test receipt for these
corrected source bytes or an Adult result.
