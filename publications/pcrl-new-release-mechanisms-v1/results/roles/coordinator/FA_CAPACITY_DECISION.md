# Family A capacity gate: coordinator decision (about 15:35Z; times from `date -u` at commit)

## Evidence

- **Designer A's registered internal capacity gate FAILED in both versions** (roles/designer_a/GATE_V0_16x64.json, GATE_V1_16x64.json;
  units FA_GATE_V0_V1 and FA_GATE_V1_V1; no SEX read).
  - Nested held-out spend of the 16 x 64 twin over the continuous task teachers (5 outer household folds), v0 / v1:
    - occupation NLL +0.0107 ± 0.0013 / +0.0136 ± 0.0013 (gate: estimate + 2 SE ≤ 0.004);
    - occupation accuracy loss +0.0026 ± 0.0019 / +0.0048 ± 0.0018 (gate ≤ 0.003).
  - The income terms pass.
- **On MODEL_SELECT (nrm.evaluate) the same 64-cell twin passes the original limits**, but most of A1's occupation margin is gone:
  - occupation NLL excess vs U0 −0.0033 / −0.0012 (A1: −0.0161 / −0.0128);
  - men's guard +0.0005 / +0.0026 (A1: −0.0039 / −0.0023), leaving about 0.0024 of men's-guard headroom in v1.
- **Label-free capacity curve** (Designer A; cached out-of-fold posteriors, model outputs only). Fitted occupation KL to the codewords
  at 64 / 128 / 256 / 512 cells: 0.0098 / 0.0067 / 0.0046 / 0.0031 (v0); 0.0094 / 0.0064 / 0.0043 / 0.0029 (v1). Held-out adds about
  0.001-0.004 at 64 cells.

## Decision

- **No change to the gate.** The registered gate is not widened after its failure.
- **The registered next step R0 (the (24, 128) ladder) is run as the gate only.** It is the nested capacity estimate on MODEL_FIT,
  with income and occupation labels and no SEX, under the committed card FA_PRC_v1.md. No full-data twin release is saved, so no
  configuration of the 60 is consumed. The registered R0 outcome is then on record.
- **If the R0 gate FAILS,** family A closes at the capacity stage:
  - label: FINITE_CELL_CAPACITY_NOT_ESTABLISHED;
  - no protected or control configuration is fitted;
  - the two saved 64-cell twins and their MODEL_SELECT utility stay in the ledger.
  - This is a finite negative result about the family's finite-cell route at registered capacities, not an impossibility claim.
    Its replacement, if any, needs its own card and the statistical reviewer's approval.
- **If the R0 gate PASSES** (not expected from the curve), PRC_TWIN_C128 (+2, the registered contingent) is fitted and the
  family's protected path proceeds per its card.
