# Designer A, closing report after the capacity stage (saved verbatim with attribution)

Author: Designer A (family A, FA-PRC) of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 15:37Z on
2026-10-20. Saved verbatim; nothing below was edited.

---

DESIGNER A: hand-back after the capacity stage (15:37Z; prepared for saving verbatim with attribution to Designer A)

**Outcome.** The R0 gate (24, 128) FAILED in both versions, so family A closes at the capacity stage with FINITE_CELL_CAPACITY_NOT_ESTABLISHED. No protected or control configuration (PRC_S1–S3, ORD_S1–S3, HQ_S2, SHUF_S2) was fitted, and the propensity-routing question remains untested.

| Twin | Occupation NLL spend over the continuous A1 teachers (v0 / v1) | Occupation accuracy loss (v0 / v1) | Gate |
|---|---|---|---|
| 16 × 64 | +0.0107 ± 0.0013 / +0.0136 ± 0.0013 | +0.0026 / +0.0048 | FAIL |
| 24 × 128 (R0) | +0.0137 ± 0.0014 / +0.0132 ± 0.0014 | +0.0057 / +0.0031 | FAIL |

- Spends are nested held-out on MODEL_FIT over 5 outer household folds, with no SEX read. The registered gate is estimate + 2 SE ≤ 0.004 nats and ≤ 0.003 accuracy loss.
- Income passed in every case (NLL +0.0010 to +0.0015 nats).
- Records: roles/designer_a/GATE_V{0,1}_{16x64,24x128}.json.

**Deviation: 2 contingent configurations were consumed.** I saved PRC_TWIN_C128 V0 and V1 at about 15:34Z. That followed your 15:31Z instruction to "apply R0 as the recorded contingent (+2)", which arrived before the 15:35Z decision asking for the gate only.
- These saves contradict FA_CAPACITY_DECISION.md's line that no full-data twin release is saved and no configuration is consumed.
- The releases are append-only and kept. Please record the discrepancy in the decision record and the ledger.
- They are full kind-twin releases on the shared teachers, so you can still evaluate them.
- I did not rerun the gate under new unit names, because the identical computation was already done: FA_GATEC128_V0_V1 and FA_GATEC128_V1_V1, under the committed card FA_PRC_v1.md and code commit 73cc1381c, run before the decision arrived.

**Why more cells did not help** (label-free; information only, no rule changed; roles/designer_a/CAPACITY_INFO_LABEL_FREE.json):

| | 64 cells | 128 cells | 256 cells | 512 cells |
|---|---|---|---|---|
| Fitted occupation quantization KL, v0 | 0.0098 | 0.0067 | 0.0046 | 0.0031 |
| Fitted occupation quantization KL, v1 | 0.0094 | 0.0064 | 0.0043 | 0.0029 |
| Approximate label-refit decoder estimation term | 0.0055 | 0.0092 | – | – |

- The decoder term rises because the median falls from about 265 to 120 people per occupation cell.
- So the held-out cost stayed flat at about 0.013 nats between 64 and 128 cells.
- Income costs 0.0009 / 0.0004 / 0.0002 at 16 / 24 / 32 cells.
- Your MODEL_SELECT evaluation of the 64-cell twin matches this. It passes the original limits, but quantization used about 0.012 nats of A1's occupation NLL margin and about 0.0045 of men's Brier, and its screening pair AUC (0.642 / 0.645) equals W's.
- A replacement with codeword (unrefit) decoders or a different finite design would need its own card and review.

**Configurations and units** (all rc 0, one thread):

| Unit | Charged CPU-s each | Notes |
|---|---|---|
| FA_TWIN_V0_V1, FA_TWIN_V1_V1 | 38.8 / 38.7 | PRC_TWIN saved; 12 model fits + 10 cross-fitting fits each |
| FA_GATE_V0_V1, FA_GATE_V1_V1 | 35.6 / 35.2 | 16 × 64 gate |
| FA_TWINC128_V0_V1, FA_TWINC128_V1_V1 | 3.9 / 3.8 | PRC_TWIN_C128 saved; reused the cached teachers, no new fit |
| FA_GATEC128_V0_V1, FA_GATEC128_V1_V1 | 35.3 / 35.0 | 24 × 128 gate |

- Configurations saved: 4 (PRC_TWIN × 2 firm; PRC_TWIN_C128 × 2 contingent).
- Queue CPU about 226 s. Out-of-queue CPU charged via nrm.jobqueue charge about 980 s, including the disclosed 740 CPU-s multi-threaded first selftest.
- Labels read: MODEL_FIT income and occupation only, by the twin and gate units with card FA_PRC_v1.md. No SEX label, no CALIBRATION label and no MODEL_SELECT label was read by any family A unit.

**Engineering review notes.** I acknowledge ENG1-07 (save in both versions or neither; record alias-of-twin instead of spending configurations) and ENG1-14 (describe ORD as "SEX-blind mapping, SEX-informed operating point"). I made no code change, because the family closed before stage 2.

**Commits** (my files only, no trailers): 47916952c, 7308e45fa, 300d017a6, f19614b05 (card), 73cc1381c (R0 code), 6d8d83744 (capacity info), 29ec7f83a (gate records). Selftest is 26/26 at 73cc1381c. Nothing of mine is running, and I spawned no sub-agents after your rule.

**Files**, all under <repository>/.claude/worktrees/pcrl-correction-transfer-stability-v1/:
- Code: nrm/fa/prc.py, nrm/fa/release.py, nrm/fa/__init__.py
- Card: results/pcrl_new_release_mechanisms_v1/cards/FA_PRC_v1.md
- Records in results/pcrl_new_release_mechanisms_v1/roles/designer_a/:
  - GATE_V0_16x64.json, GATE_V1_16x64.json, GATE_V0_24x128.json, GATE_V1_24x128.json
  - CAPACITY_INFO_LABEL_FREE.json
  - SELFTEST_WAVE1.json, SELFTEST_WAVE2.json, SELFTEST_WAVE3.json, SELFTEST_WAVE4_R0.json
