# Validation

## Before any new fit

| Check | Result |
|---|---|
| Reuse custody | `LOCK.json` pins:<br>- 81 reused output-aware unit COMPLETE files: heads, attackers, FARE trees, U2 probes, references;<br>- the 3 LEACE map directories;<br>- the 3 FARE selection files;<br>- the forward caches.<br>All were hash-verified at lock time and again by the runner before the fits. |
| Head construction | All 12 reused heads emit log-probabilities (logsumexp = 0 within 1e-12). The offset equals −(softplus(d) + softplus(−d))/2 within 1e-12, so full logits, (d, c), centred and probabilities are informationally equivalent. |
| Families | 19 primary / 33 secondary, asserted in code. z = 3.007787 / 3.171766, asserted to 6 decimals. |
| Tests | 32 pass (`cap/tests`, `odx/tests`, `oar/tests`). They cover family counts and z, the log-probability head identity, the linear retention statistic, and the resolution chain bank → plus → component. |
| Lock | Lock `a0de449` was pushed at 17:53:29Z. The first fit started at 17:53:39Z. All 84 timestamped new units completed after the lock (independent check). |

## After the run

| Check | Result |
|---|---|
| Units | 184/184 accounted for: 180 hash-complete unit directories (48 aliases, 84 attack, 48 banks) and 4 controls. 0 failed, 0 quarantined, 0 budget-unrun. |
| Real-data controls (attacker_fit/val only) | Shuffled-label nulls 0.479–0.511 (none flagged); planted leaks 0.899–0.927 (all detected). They covered output-only centred (A), output-only prob (B), output-only hard (F, finite) and features plus hard (F). |
| Lock re-verification after the run | `ok: true`. No locked file changed. |

## Independent replay (`INDEPENDENT_VERIFICATION.json`, `verification/replay_cap.py`)

An import guard refuses `cap`, `odx`, `oar`, `stored_model_eval`, `report` and `pcrl`. The replay uses its own roles, LEACE application, surfaces, banks, plus rule, resolution, Mann–Whitney AUC, multinomial group bootstrap, bounds and decisions.

**Result: 35 PASS, 0 FAIL, 8 INFO, 2 WARN.**

| Item | Outcome |
|---|---|
| Roles | Rebuilt: 5,243 assessment / 6,065 attacker_fit / 2,235 attacker_val / 1,500 cert / 17 exposure-excluded. They match every unit. |
| Attacker replays | All 528 saved models (NL as0–as2 and L, for the 84 new units and the 48 alias sources) are bitwise identical to the saved predictions. One MLP refitted from scratch reproduces bitwise. |
| Selections | 48/48 banks, 60/60 plus-rule choices and 132/132 resolutions are recomputed exactly. |
| Recovery and endpoints | 192 per-seed points have max difference 0. All 19 + 33 endpoint points match exactly. SE differences are ≤ 3.5e-18. Decisions agree: primary 12 PASS / 7 NOT_ESTABLISHED; secondary 22 PASS / 11 NOT_ESTABLISHED. |
| Cross-study | The arm-A output-only units are bitwise identical to the output-diagnosis study's refitted-head units (12/12). |
| WARN 1 (interpretive) | `U-retain-F` passes only narrowly: its lower bound is 0.00064, 0.32 SE above the target. It is reported as near the bound. |
| WARN 2 (interpretive) | `S-offset-out-F` / `-F0` are exactly 0 because the two sides' predictions are bitwise identical: the finite outputs are recodings. Their NOT_ESTABLISHED holds by construction, not as an empirical finding. This is reported as such. |
| Tie sensitivity | 34 exact validation ties. No endpoint depends on how a tie is broken. |

## Manuscript review (`MANUSCRIPT_REVIEW.md`)

An independent reviewer audited 100 numbers and the wording rules, and found 21 required fixes. **All 21 were applied** before the final build:
- 3 numbers (the LEACE range, the bypass range, HMDA +0.262);
- 2 labels (ρ₁², the 0.0098 bound);
- per-study protocol differences (roles, bootstrap) stated explicitly;
- an undisclosed reuse (below);
- the multiclass claim corrected (the effect passed for HMDA pricing/race);
- 2 citation over-attributions;
- post-hoc labels and non-causal wording;
- "decisions say little" softened;
- the certificate wording;
- the coalition seed-2 sentence;
- "PCRL" removed from the figures;
- the Figure 3 legend.

The final PDF has 10 pages, no overfull boxes and no unresolved references. Every page was inspected.

**Disclosure found in review.** The protocol's "already known" list covered only head accuracies. The full-logit levels of the 48 reused surfaces were already in the output-aware study's committed exploratory tables when the predictions were registered. This is now disclosed in `USEFUL_HEAD_COMPARISON.md` and in the manuscript. It does not change any decision.

## Other corrections found during this assignment

- The 22→8 and 26→19 counts were mis-described in a draft of `CORRECTIONS_AND_SCOPE.md` as best-validation vs final-checkpoint counts. The evidence-map builder caught this, and it was corrected before commit.
- Stage A: all eight requested corrections were recomputed from committed arrays (`report/corrections_recomputed.json`). No registered decision changes.

## Backup and restore

- The versioned private copy `private_cap_v1_20261003` holds 955 files. All 955 were re-read uncached (F_NOCACHE) and match.
- Restored from the copy alone, 4 sampled attackers reproduce their saved predictions exactly (max diff 0).
- **Drive copy** `<drive>/private_cap_v1_20261003`: 955/955 files re-read uncached and matching. Its SHA256SUMS hash is identical to the local copy's, and the 4 restore replays from the drive copy are exact.
