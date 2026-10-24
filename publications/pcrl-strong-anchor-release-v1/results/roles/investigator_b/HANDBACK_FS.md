# Investigator B hand-back for the replacement family FS_CPI (saved verbatim by the coordinator, about 06:13Z on 2026-10-24)

## FS_CPI is ready to commit: an income-only residual on SUCC, trained against the pair with JOINT occupation frozen

I built the replacement family as you proposed, with three changes I chose myself (listed below; each is yours to reverse). All selftests and the python3 deployment check pass. Design took about 35 CPU-s of the 300 allowed (05:54Z–06:11Z). Everything was label-free: committed records, admitted features and frozen files only. Nothing else in shared code was touched, and there was no git use this time (the earlier read-only `git log` was already reported).

### Operation (sequential, occupation first)
- **Occupation:** SUCC's JOINT E20 vector, bitwise. Every occupation clause and both guards, including the binding men's guard, stay SUCC's own.
- **Income:** SUCC's H3 log-odds plus a residual g. g is a 64-64 network, zero-initialised so that θ = 0 is SUCC. It reads `sar.fa.tfr`'s feature map: compact fields, SUCC's logits, and the HGB teachers' logits.
- **Training:** all 20,000 MODEL_FIT people, using MODEL_FIT SEX only and no task label. The loss is KL(H3 ‖ H3 + g) plus λ times P_FS:
  - P_FS is FB's closed-form whitened pair best response, computed on (trainable income, frozen occupation log-ratios as released), plus a hinge on the income view at START's value.
  - With occupation frozen, D²_pair = D²_occ + D²_income|occupation, and only the second term can move. The partial weights therefore target exactly SUCC's 0.040 super-additive increment.
- **Strength:** λ ∈ {0.025, …, 1.6}, measured in nats of anchor KL per unit of AUC proxy; 400 steps per λ.
- **Rule:** the largest λ such that it and every smaller λ keep estimate + 2 SE within budget on CALIBRATION for the three income components:

  | Income component | Budget |
  |---|---|
  | NLL | 0.0065 |
  | Brier | 0.0035 |
  | Accuracy loss | 0.0040 |

  These are about 0.5 of SUCC's smallest recorded MODEL_SELECT income margins. That is an adaptive, disclosed choice, and it also absorbs a transfer of up to 2x.

### Selection data, stated honestly
- Spends are paired CALIBRATION differences against START_CF: H3 with its stack and Platt maps household-cross-fitted on CALIBRATION, which is H3's own budget estimator (folds `IHS_H_budget_S<v>`).
- The residual, the HGB teachers and H3's trees and HGB learner are all out of sample on CALIBRATION.
- The one remaining in-sample element is β itself, a single parameter chosen on CALIBRATION by H3's rule. This is disclosed.
- MODEL_FIT is not used for spends, because SUCC is in-sample there (FA's D2 showed offsets of about 0.02).

### Controls (origin START, selection data CALIBRATION)
- **CTRL FS_CPI_DQ_HI:** SEX-blind DQ of SUCC's income, with occupation passed through bitwise. One point is saved: the smallest spend ≥ max(CAND + 2 SE, 1.5 × CAND). The 1.5x hedge is prospective, against FB_R1's NO_BRACKET.
- **ALT FS_CPI_BETA_LO / _HI:** SUCC's own LJDIR transplant continued, with β above 0.25 / 0.20 on H3's grid, and stack and Platt maps refitted on CALIBRATION per β. This is exactly `ihs.fam_h`'s recipe. The bracket uses the shared rule (LO ≤ CAND's spend, HI ≥ CAND + 2 SE).
- **CONT:** not declared, because λ = 0 is START bitwise (see the first change below).
- **Not built:** SBC and RO. Both are constructible; they are left out for budget.
- **Specification count:** at most 8 new (CAND 2 + CTRL 2 + ALT 4).

### Three changes from your proposal
1. **Anchor is KL to START, not HGB distillation.** FA's D1 shows that an HGB-distilled path departs from START at every strength, so CALIBRATION's 2-SE noise on income Brier (about 0.0036) defeats the budgets. With the START anchor, spend scales with the protection, λ = 0 leaves g exactly 0, and no task label is read. The cost is that there is no CONT distinct from START. If you want a CONT, the HGB variant provides one, with FA's noise problem.
2. **The income-view term is a hinge, not a full term.** FB_PFD_v1 showed the AUC-scale view term's 1/D amplification spends income on a near-chance view. SUCC's income view (0.529, D ≈ 0.1) would be amplified even more. The hinge only guards the recipient condition.
3. **CTRL is a single hedged HI point; ALT keeps LO and HI.** This fits the 8-spec budget.

### Closest work
- **H3 / LJDIR:** its D_J was trained jointly with JOINT's *own* income and then transplanted. It was never trained against the pair it ships in, and that pair is super-additive by 0.040.
- **ihs family P (PAIO):** this is the nearest. It also froze occupation, but used leaf-score corrections on T_TREE (a closed family), online critics, and a projection budget, and pair training "added only about 0.001". FS_CPI differs in function class, objective (whitened partial weights rather than marginal critic attribution), anchor and base.
- **B3CTL codes:** closed; FS_CPI keeps continuous probabilities.
- **FA_TFR / FA_TFRO_R1:** same features, but online critics and HGB distillation on the occupation side.

### Registered predictions

| Prediction | P |
|---|---|
| λ* ≥ 0.1 in both versions | 0.65 |
| NO_FEASIBLE_STRENGTH in either version | 0.10 |
| All 28 conditions at the MODEL_SELECT point | 0.85 |
| Benefit vs START > 0 at the point / resolved | 0.80 / 0.45 |
| Benefit vs CTRL > 0 at the point / resolved | 0.80 / 0.40 |
| Benefit vs ALT > 0 at the point / resolved | 0.60 / 0.20 |
| D2 met | 0.15 |
| D3 met | 0.10 |
| D4 met | 0.12 |
| Pair − occupation view ≤ +0.015 | 0.30 |

- MODEL_SELECT pair: median 0.603, 80% interval [0.590, 0.615].
- Pair − occupation view: median +0.025.
- Most likely outcome (0.40): feasible, pair 0.598–0.610, a point benefit of +0.006 to +0.018 over SUCC, and not separated from ALT.

**What would contradict the rationale:** λ* ≤ 0.05; the MODEL_FIT pair-minus-occupation gap not shrinking while the income view falls; or ALT or CTRL reproducing the pair drop at matched spend.

### Tests
- **`selftest` (13/13 pass).** Covers:
  - λ = 0 leaves the output exactly 0;
  - the invented conditional increment D²_pair − D²_occ is cut from 0.331 to 0.016 with no rise in the income view;
  - the hinge rule and the hedged CTRL rule;
  - on admitted features with the frozen SUCC files: occupation is SUCC's bitwise, a zero residual reproduces SUCC's income (2e-16), the ALT law at SUCC's β and maps reproduces SUCC's income, and the DQ law matches bitwise.
- **`selftest_stage`: passes** end to end with invented MODEL_FIT SEX and invented CALIBRATION income. All four configs are saved, the bracket branch runs, every release equals `predict`'s output bitwise, occupation is SUCC's everywhere, and it took 5.3 CPU-s. The check that ALT's refitted maps reproduce SUCC's income can only be meaningful with the real CALIBRATION labels, so the stage test reports it without gating.
- **`check_fs_deploy_python3`: passes** through `sar.fs` without torch.
- **CPU per real fit:** about 1.5–2 CPU-min per version (upper estimate 4).

### Notes for the queue
- The fit reads MODEL_FIT `('features', 'sex')` under `fit`, and CALIBRATION `('features', 'income')` under `calibrate`, both under the FS card.
- Release files go under family `FS`; the reading name is `FS_CPI`. The fit record goes to `results/pcrl_strong_anchor_release_v1/fs/FIT_CPI_V<v>.json`.
- Pins for every frozen file and TASK.pkl are recorded, and `predict` refuses if they change.

Files are in `<repository>/.claude/worktrees/pcrl-strong-anchor-release-v1/`:
- `sar/fs/cpi.py`
- `sar/fs/__init__.py`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_b/CARD_DRAFT_FS.md`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_b/fs_selftest_output.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_b/fs_selftest_stage_output.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_b/check_fs_deploy_python3.py` / `.json`
