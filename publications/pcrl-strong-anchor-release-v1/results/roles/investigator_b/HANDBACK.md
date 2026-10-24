# Investigator B hand-back (saved verbatim by the coordinator, 2026-10-24 about 04:52Z)

## Investigator B hand-back: proposal FB_PFD (pair Fisher-discriminant training of the two U0 networks)

All the work was label-free. I opened no label file and computed nothing from MODEL_SELECT. I used about 45 CPU-s of the 600 CPU-s cap (selftests about 40 s, label-free checks about 5 s), working 04:22Z–04:51Z. I did not spawn agents, run git, push, kill processes, or edit shared code.

### Starting point and binding condition
Pair recovery for NEURAL_RAW is the same as its occupation view (0.6253 = 0.6253). What stops its own path is income utility, not the linked pair:
- **Stronger points fail on income first** (SELECTION_DETAIL.json):
  - rho1 E5 reaches proxy pair 0.6065 and fails only version-1 income Brier (1.059 of cap). Its occupation NLL is +0.0048 / +0.0045.
  - rho3 E2 and rho10 E2 also fail income clauses first.
- **The income network is pushed hard for no visible signal** (LOGS.json):
  - The income-view critics read almost nothing: risk(income) is 0.000–0.0045 and exactly 0 on most steps.
  - Yet the income network's weighted privacy gradient is 1.1–2.1x its task gradient.
  - Per-recipient calibration gives income about twice the occupation multiplier (lambda_J 1.57 / 0.88 in v0, 1.82 / 0.77 in v1).
- **Label-free geometry** (`check_release_geometry.json`, frozen MODEL_FIT and CALIBRATION arrays):
  - U0's income log-odds is 68–74% predictable (R²) from its occupation log-ratios.
  - Along NEURAL_RAW's path the income output moves as much as or more than the occupation output (v1 E2: 0.113 vs 0.071 in U0-SD units).

**Hypothesis:** the early-stopped, diagonally normalised critics weight the income coordinate as a proxy for signal the occupation output carries, so the income utility budget pays for occupation-view protection. Protection must not drop income entirely: the logs show the pair becoming super-additive as occupation protection deepens. Also, U0's income view (0.5476 / 0.5526) would fail the recipient condition against NEURAL_RAW, which allows at most 0.5434 / 0.5438.

### The operation
Both networks start from U0 and keep NEURAL_RAW's optimizer and task rows (Adam 1e-4, clip 5, the THP1task stream). The loss is task cross-entropy plus one coalition coefficient λ times P, where:
- P = Σ over the income view, occupation view and pair of [Φ(D_v/√2) − ½].
- D_v² = Δᵀ(S + εI)⁻¹Δ is computed fresh on every 2,048-row batch, from degree-2 polynomial features of the logit differences (whitened by the pooled within-SEX covariance, ε = 1e-3·tr/p).
- So P is the summed AUC (Gaussian approximation) of the exact best quadratic-discriminant reader of each view and of the pair.
- λ = rho × median task-gradient norm / median P-gradient norm, with rho = 1.

Why this is not NEURAL_RAW's or JOINT_TREE's critic objective under a new name:
1. There is no adversary: the reader is re-solved exactly each step, so there is no critic lag and no zero clamp.
2. Full-covariance whitening gives each recipient's coordinates only their incremental weight. On invented data where the income coordinate only proxies an occupation coordinate, its share of the gradient is 0.040, against 0.460 under marginal weighting.
3. One strength covers both recipients, instead of a separate λ per recipient.
4. D is invariant to affine changes of the logit differences, so temperature or bias rescues cannot lower P.
5. Nothing on the closed list is involved: no routing, post-processing, noise, merges, codes, input generalisation, standardisation, reweighting or controller.

### Card choices (CARD_DRAFT.md)
- **Grid:** rho = 1 with the path E1..E10 (40 steps of 256 rows per epoch).
- **Selection rule (MODEL_FIT only):** k* is the largest E_k such that E1..E_k all keep estimate + 2 household-clustered SE within budget on all 8 components. Spend is the out-of-fold paired spend over each fold's own start (folds SAR_xf_FB_S<v>; each fold restarts from U0's recipe refitted on its training rows).
- **Budgets:**

| Component | Budget |
|---|---|
| Occupation NLL | 0.0060 |
| Income NLL | 0.0050 |
| Occupation Brier | 0.0025 |
| Income Brier | 0.0025 |
| Men's occupation Brier | 0.0028 |
| Women's occupation Brier | 0.0028 |
| Occupation accuracy | 0.0060 |
| Income accuracy | 0.0060 |

  That is about 0.5–0.6 of each cap. Every component except income Brier allows more than NEURAL_RAW's own MS spend, so this does not force near-identity updates.
- **Matched control FB_PFD_RAWJ:** NEURAL_RAW's own recipe, step for step, from the same starts and rows, at the first checkpoint whose out-of-fold total NLL spend reaches the candidate's. A recorded check confirms its all-MODEL_FIT E2 weights equal `NEURAL_RAW__rho1.0__E2.pt` bitwise, so it is the anchor's own path continued to the same cost.
- **Zero-protection continuation FB_PFD_TASK:** the same loop with λ = 0, at E_k*.
- **Spec count:** 6 (three configurations × two versions).
- **Revisions:** R1 switches to degree-3 features if the MODEL_FIT HGB pair reader is evaded. R2 switches to rho = 3 if the budget is never reached.
- **Not built:** the SBC/SPEND/RO paths, with reasons stated in the card.

### Key predictions
| Prediction | Probability |
|---|---|
| Candidate passes all 28 conditions at the MS point | 0.75 |
| Pair AUC median 0.610 (80% interval 0.592–0.628) | — |
| Pair benefit vs NEURAL_RAW > 0 at the point | 0.75 |
| Pair benefit vs NEURAL_RAW resolved | 0.40 |
| Benefit vs the matched control > 0 at the point | 0.65 |
| Benefit vs the matched control resolved (D2) | 0.15 |
| Material target met (D4) | 0.04 |
| Income view passes the recipient condition | 0.70 |
| The MODEL_FIT binding component is on the occupation side (men's guard 0.45) | 0.65 |
| Both bitwise checks hold | 0.85 |

**Most likely outcome (0.45):** feasible, pair 0.600–0.618, a point benefit of +0.007 to +0.025 over NEURAL_RAW, not separated from its matched control.

### Built and tested
- **`sar/fb/pfd.py`:** training under the torch venv. `predict` is a float64 numpy forward pass of the saved weights.
- **`sar/fb/__init__.py`:** exposes `predict` and `model_paths` for `sar.deploy` (your note), with deferred imports so `python -m sar.fb.pfd` does not load the module twice. **predict runs under python3 with no torch**; checked via `sar.deploy.family_module('FB')` (`check_deploy_python3.json`, all ok).
- **`selftest`:** all 18 checks pass on invented data. They cover numpy-vs-torch forward agreement, the D² estimate, the AUC proxy (0.6651 vs empirical 0.6645), affine invariance, the pair bounding each view, partial attribution, all three arms, the selection and matching rules, and CPU (0.0041 s per step at width 145).
- **`selftest_stage`:** an end-to-end run of the stage on invented labels attached to admitted MODEL_FIT features. All three configs were saved and each release equals `predict`'s output bitwise.
- **CPU per real fit:** about 3–5 CPU-min per version (upper estimate 10).
- Outputs are saved as JSON in `roles/investigator_b/`.

### Open questions and risks
1. Whether you accept an exact-best-response, whitened, single-strength objective as a structurally different construction. I argue yes, but it is still penalty training of the same networks.
2. The quadratic best response may be evaded through higher-order structure (that is R1's trigger).
3. The men's guard may bind early. NEURAL_RAW E2 already shows men +0.0015 / +0.0020 against an overall occupation Brier of +0.0003; I could not check the guards at its stronger points without labels.
4. Spend estimated on MODEL_FIT may not transfer to MODEL_SELECT.

### Notes on shared code (no changes required)
1. The fit reads `U0_MODEL.pt`, `STAGE_A_SEED.json` and `NEURAL_RAW__rho1.0__E2.pt` from the replication `.private/acs_replication/runs/` tree. It records the hashes of the first two but checks nothing against SOURCE_ADMISSION.json. Please add pins if you want a check.
2. The fit writes `results/pcrl_strong_anchor_release_v1/fb/FIT_PFD_V<v>.json` and must run under `<repository>/.venv/bin/python` inside a queue unit (C.load is called from sar.fb.pfd).
3. FB_PFD_RAWJ's wire uses the numpy law, so it differs from NEURAL_RAW's stored torch wire by about 1e-7 even when the weights are bitwise equal.

### Files (in `<repository>/.claude/worktrees/pcrl-strong-anchor-release-v1/`)
- `sar/fb/pfd.py`
- `sar/fb/__init__.py`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_b/PROPOSAL.md`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_b/CARD_DRAFT.md`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_b/check_release_geometry.py` / `.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_b/check_deploy_python3.py` / `.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_b/selftest_output.json`
- `results/pcrl_strong_anchor_release_v1/roles/investigator_b/selftest_stage_output.json`
