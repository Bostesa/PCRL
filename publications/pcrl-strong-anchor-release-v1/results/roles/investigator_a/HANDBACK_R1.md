# Investigator A hand-back for revision R1 (saved verbatim by the coordinator, about 05:38Z on 2026-10-24)

## Investigator A: revision FA_TFRO_R1, built and tested (label-free)

I propose one revision, FA_TFRO_R1: the residual may move only the occupation output, and income is SUCC's H3 vector bitwise. The R1 selftest passes all 19 checks (26.9 CPU-s), and the FA_TFR_v1 selftest still passes (24 checks). The status is REVISED_: the design uses only the two v1 fit records and the committed anchor table. No label was read, no git was run, and no shared code was changed.

### Diagnosis from FIT_V0.json and FIT_V1.json (ρ = 0.3; v0 / v1)
- **D1 – part (a) failed only on components CALIBRATION cannot measure at v1's budgets.**
  - Every point estimate was favourable or near zero:

    | Component | v0 | v1 |
    |---|---|---|
    | Occupation NLL | −0.0160 | −0.0140 |
    | Occupation Brier | −0.0063 | −0.0053 |
    | Occupation accuracy | −0.0091 | −0.0050 |
    | Income accuracy | −0.0081 | −0.0054 |
    | Income Brier | +0.0007 | +0.0019 |
  - The failures (each budget is estimate + 2 SE): income Brier (bounds 0.0043 / 0.0056 > 0.0030), income NLL in v1 and occupation accuracy in v1 (0.0056 > 0.0050).
  - My 0.4-0.6-of-margin budgets sat inside CALIBRATION's 2-SE noise. The income-side spends are also biased in SUCC's favour, because H3 is in-sample on CALIBRATION.
- **D2 – part (b) was not a valid estimator.** On CALIBRATION, where the occupation side is out of sample, CAND's occupation Brier is 0.0063 / 0.0053 better than SUCC's. On MODEL_FIT the same CAND reads 0.0148 / 0.0165 worse: a sign flip of about 0.02 from SUCC's in-sample fit. The guard difference against the λ = 0 LO point (+0.007 to +0.014 per group) is the same size as that offset, so it cannot be separated from it. The cancellation I assumed in v1 does not hold.
- **D3 – within one release the men-minus-women contrast is barely affected.** For the LO point it is −0.0005 / +0.0007 while both groups read about +0.004 over SUCC.
  - CAND's contrast is −0.0072 / +0.0042 at ρ = 0.3.
  - It turns men-heavy as ρ grows: +0.0042 / +0.0061 at ρ = 3 and +0.0156 / +0.0201 at ρ = 10 (exact values in `r1_contrasts.json`).
- **Also:** ρ = 0.3 already recovered most of the occupation quality gap to the teacher, so it is a strong move toward the teacher.
- **v1's declared R1 (pair-only critics) is not used.** Its trigger needs the λ = 0 path's CALIBRATION spends, which the records do not hold when no candidate exists, and it would not fix D2.

### The mechanism change
- **Release:** occupation = softmax(centre(z_S + r_O(φ))), where r_O is a 64-64 ReLU network with 5 outputs and a zero-initialized output layer. The features, task distillation and critics are as in v1. risk(income) is constant; the pair critic sees H3's income with the moved occupation.
- **Why it is structural:** every component CALIBRATION cannot measure without SUCC's bias (all of income) is removed, so the 12 income clauses are SUCC's own. Every layer the task part of the rule measures is out of sample on CALIBRATION.

### The rule (prospective)
- **Budgets are SUCC's own margins to the caps.** The code reads them from `anchors/ANCHOR_TABLE.json`: cap minus the larger of SUCC's U0 / Ucal values, and cap minus SUCC's guard values.

  | Component | v0 | v1 |
  |---|---|---|
  | Occupation NLL | 0.00659 | 0.00705 |
  | Occupation Brier | 0.00488 | 0.00451 |
  | Occupation accuracy | 0.0120 | 0.0140 |
  | Men's guard | 0.00058 | 0.00106 |
  | Women's guard | 0.00994 | 0.00804 |
- **Part (a):** CALIBRATION paired occupation spends over SUCC, each at estimate + 2 SE, must fit within these margins. The code asserts the income spends are exactly 0.
- **Part (b), the new guard estimate:**
  - L = the CALIBRATION occupation Brier change, the population average, measured out of sample.
  - C = the MODEL_FIT out-of-fold men-minus-women contrast of the same release (SEX exists only on MODEL_FIT).
  - Δ_men = L + (1 − p)·C and Δ_women = L − p·C, where p is the MODEL_FIT share of men, with SE² = SE_L² + weight²·SE_C².
  - Each must stay within its margin at estimate + 2 SE.
- **Why this measures the cap:** it is the exact average-plus-contrast decomposition of each group's change, with each piece taken from the only data that can see it.
- **Remaining biases, stated in the card:**
  - any SEX-asymmetric part of SUCC's in-sample term left in C (it looks small and conservative for the λ = 0 path, unknown for the critics);
  - CALIBRATION's SEX mix is assumed equal to MODEL_FIT's;
  - MODEL_SELECT evaluation noise is not covered, so point feasibility is predicted, not guaranteed.
- **Grid:** ρ ∈ {0.3, 0.6, 1, 2, 3, 6, 10}, fitted in increasing order. The scan stops at the first failure, and ρ\* is the last ρ that passed.

### Controls (frozen SAR1 rules)
- **CTRL:** the occupation-only λ = 0 path with origin START (s = 0 is SUCC exactly). It is bracketed by `tfr.bracket` unchanged into FA_TFRO_R1_TASK_LO and _HI on CALIBRATION total-NLL spend, with the same flags and endpoint aliases as v1.
- **CONT:** FA_TFRO_R1_CONT (λ = 0, s = 1; about W), gating.
- **Specifications:** at most 8 (CAND 2, CTRL 4, CONT 2).

### Predictions (registered in the card)

| Quantity | Prediction |
|---|---|
| ρ\* | 0.3: 0.15 · 0.6: 0.20 · 1: 0.25 · 2: 0.15 · 3: 0.08 · 6: 0.04 · 10: 0.03 · none: 0.10 |
| Scan stops on the men's guard estimate | 0.65 |
| CALIBRATION occupation NLL improves on SUCC in both versions | 0.85 |
| MODEL_SELECT men's guard below SUCC's in both versions | 0.80 |
| All 28 conditions met at the point | 0.75 |
| Pair AUC | 0.620 (80% interval 0.600-0.638); occupation view 0.610 |
| Pair benefit over SUCC | −0.013 (80% interval −0.030 to +0.005); P(> 0) 0.20 |
| Benefit over LO / HI | +0.010 / +0.003; resolved over both 0.12 |
| CONT pair AUC | 0.645 |
| D3 / D4 | 0.02 / 0.02 |

My honest expectation is that usefulness margin is recovered but the protection profile mostly is not.

### What I built
- **`sar/fa/tfro.py`** (new module). It reuses v1 helpers unchanged and adds:
  - `preflight`, which passes for both versions, so the SUCC pins are in place;
  - the record `results/pcrl_strong_anchor_release_v1/fa/FIT_R1_V<v>.json` with all fields from your item 5, plus L, C, p and Δ with SEs per ρ.
- **`sar/fa/__init__.py`** now routes `FA_TFRO_R1*` ids to `tfro.predict`; every other FA id still goes to `tfr.predict` exactly as before. This touches the FA deploy entry, not the v1 code paths; v1 saved nothing, so nothing deployed depends on it. Say if you want it handled differently.
- **The selftest covers:**
  - s = 0 is START exactly, and income is bitwise SUCC in every saved release;
  - the numpy forward equals torch, and fits are deterministic;
  - the guard estimate recovers each group's change exactly when its two pieces are consistent;
  - the margins match the anchor table;
  - bitwise deployment through the package router;
  - the scan stops at the first failure and nothing is saved;
  - the record fields.
- **CPU:** about 130 of the 300 design CPU-s used. The fit is estimated at 4-6 CPU-min per version (about 9 at worst).

Files are in `<repository>/.claude/worktrees/pcrl-strong-anchor-release-v1/`:
- `sar/fa/tfro.py`
- `sar/fa/__init__.py`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/CARD_DRAFT_R1.md`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/selftest_R1_output.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/preflight_R1_V0.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/preflight_R1_V1.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_a/r1_contrasts.json` (and its script, plus `r1_diagnose_records.py`, `r1_succ_margins.py`)
