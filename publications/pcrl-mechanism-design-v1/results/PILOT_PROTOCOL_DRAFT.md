# Pilot protocol draft: MMP on the Florida ACS 2019 two-recipient service

**Status: prepared and rehearsed on synthetic stand-ins. Not run. Not recommended as a competitive test.** The 2026-10-16 correction pass does not change this status and authorises no run.

> **Corrected on 2026-10-16 (see `CORRECTIONS.md` row 8).**
> - The original text said "Fitted SEX-aware trees do not merge cells", and the D0 row said the synthetic rehearsal suggested injective releases. Both are unsupported. The rehearsal counts come from 800 invented rows, and population injectivity of real models is unknown.
> - A distinct-value count on MODEL_SELECT is descriptive only. It can reveal frequent merges in that sample. It cannot certify injectivity, rare merges, merges outside the sample, or whether merged inputs differ in SEX posterior.

MMP's own pre-registered prototype contradicted the learnability leg of its rationale (`prototype/PROTOTYPE_REPORT.md`). In a law whose release is one-to-one by construction, deleting the inferred-SEX term did not reduce what readers learn. Under the brief, MMP is therefore not described as pilot-ready.

The verifier's probe, registered before it ran and post-hoc relative to the prototype pre-registration, then showed that without pure-proxy fields (all ten fields task-relevant), MMP's net effect relative to its twin had no definite sign: −0.033 in one draw and about +0.02 in two. The protocol is delivered complete because the brief asks for it whenever a mechanism is selected. It also records exactly what would be tested if the advisor nevertheless wants the one cheap data answer: Stage 0 alone (§4).

**Exploratory label.** Any run on these people is an exploratory development pilot on repeatedly used Florida ACS 2019 DEVELOPMENT people. It is not confirmation, error rates are nominal, and a new lock does not erase their selection history. SEALED_ASSESSMENT stays closed.

**Refusal in this campaign.**
- `mechanism_design/pilot/run_pilot.py --source real` raises PermissionError. It requires a committed `pilot/PILOT_AUTHORIZATION.json` that names this protocol's hash and an explicit user authorisation. That file is never created here.
- SEALED_ASSESSMENT is refused unconditionally.
- DEV_REPORT truth is refused without a committed pilot REPORTING_LOCK.
- No real-person array was read in this campaign. The rehearsal used invented data under the `no_real_data` guard.

## 1. Preserved from the source

- **Cohort, fields, tasks and roles:** the Florida 2019 cohort; the six permitted fields; income above $50k and five occupation groups; the household roles MODEL_FIT, CALIBRATION, MODEL_SELECT, ATTACK_FIT, ATTACK_VALIDATION and DEV_REPORT.
- **References:** U0 and Ucal.
- **Utility caps:** accuracy loss ≤ 0.010, NLL excess ≤ 0.010, summed Brier excess ≤ 0.005, for both tasks, both references and both seeds, giving 24 clauses.
- **Selection rule:** the frozen source rule.
- **Release contract:** the primary contract.

## 2. Arms (equal exposure)

| Arm | Role in the test | Source |
|---|---|---|
| **MMP** (candidate) | SEX-aware CatBoost teachers (T_TREE registry; SEX as an extra input) on MODEL_FIT. Release m = (1 − π̄)q⁰ + π̄q¹ | new fit |
| **MMP_TWIN** (task-only twin) | The same components mixed with a cross-fitted π̂(x). Isolates the marginalisation step | new fit |
| NEURAL_RAW ρ1 E2 | Existing eligible neural benchmark | frozen outputs |
| LOCAL_TREE E20 | Strongest existing local (non-joint) protected twin | frozen outputs |
| TASK_REFIT E20, T_TREE | Tree-family task-only controls | frozen outputs |
| U0, Ucal | Utility references | frozen outputs |
| Published baselines | FAGTB and official FARE had no MODEL_SELECT-feasible configuration in this population, so they are reported as ineligible and not re-run. **Implementation gap:** no published attribute-inference defense implements marginalisation over a sensitive attribute; the device is known from fairness and causal work | — |
| Sequential orders | Not applicable: no joint-component claim | — |

**Equal exposure.**
- Every new arm uses MODEL_FIT features, task labels and SEX only for fitting, the same household internal split, and the same CatBoost registry (version 1.2.8).
- CALIBRATION is unused.
- Gates and selection use MODEL_SELECT only. Readers use ATTACK_FIT and ATTACK_VALIDATION only.
- The transmitted contract is identical for every arm.

## 3. Readers, controls and inference

- **Readers.** For every arm and seed, the fresh 7-reader slate (OUT_LR, OUT_MLP, OUT_HGB, LOG_LR, LOG_MLP, DEC_LR, CONF_HGB) is fitted on ATTACK_FIT with SEX, in 3 views.
  - Orientation and winner are chosen by ATTACK_VALIDATION AUC.
  - The pair bank holds **21 candidates**: 7 pair readers plus 14 ignore-one aliases.
  - DEV_REPORT is scored from releases only.
  - Reader seeds are MODEL_SEED_BASE + s.
- **Learning curve.** Pair readers are refitted on nested 25/50/100% household subsets of ATTACK_FIT for MMP, MMP_TWIN and NEURAL_RAW. A markedly steeper rise for one arm indicates obfuscation rather than removal.
- **Controls** on the MMP wire:
  - shuffled-SEX null, accepted if AUC is in [0.45, 0.55];
  - planted SEX bit, accepted if AUC ≥ 0.99.
- **Inference.**
  - Paired household bootstrap, B = 2048, new pilot seed 2026101701; Wald t(G − 1).
  - Clause bounds one-sided at α/3. Family K = 36.
  - Primary endpoints:
    - competitive vs NEURAL_RAW, both seeds (lower bound > 0, nested margin 0.020);
    - recipe vs MMP_TWIN, both seeds (> 0.020);
    - 4 recipient-worsening clauses (≤ 0.010);
    - 24 DEV utility clauses;
    - 4 within-SEX occupation-Brier guards (≤ 0.005), reported for **both** groups.
  - Single-recipient AUCs and pair-minus-best-single are reported beside the coalition endpoint.
- **Separate decisions:**
  - point feasibility;
  - utility confidence;
  - recovery reduction vs the twin;
  - competitive vs NEURAL_RAW;
  - subgroup;
  - novelty (none claimed).

## 4. Stage 0: the only part worth considering

Stage 0 uses MODEL_FIT and MODEL_SELECT only, with no ATTACK or DEV roles. If the advisor wants a data answer, this is the cheapest informative step. It costs about 0.5 CPU-hour and needs explicit authorisation. Any failed gate stops the pilot with `NOT_SUPPORTED_AT_GATE`, before any reader is fitted.

| Gate | Test | Why |
|---|---|---|
| D0 merge diagnostic (label-free; reported first, not a stopping rule) | Distinct MMP release pairs per distinct encoded input on MODEL_SELECT; split usage of the SEX-aware trees on each field | The prototype showed population removal only where the release merges inputs with different SEX posteriors. *(Corrected 2026-10-16.)* The count is descriptive: it shows frequent merges in MODEL_SELECT at stored precision and cannot certify injectivity. The original text inferred injectivity from the synthetic rehearsal, which is unsupported |
| G0 SEX gain | Held-out NLL gain from adding SEX to the teacher > 0 for at least one task, both seeds | Without it, Δ ≈ 0 |
| G1 utility (F3) | MMP passes all 24 MODEL_SELECT point clauses | Frozen selection rule |
| G2 premise (F1) | Frozen OUT_LR coalition proxy: MMP_TWIN − MMP ≥ 0.02, both seeds | Δ must carry learnable signal |
| G3 nonlinear premise | The same with OUT_HGB: ≥ 0.01, both seeds | Symmetric nonlinear gate (adversary reviewer's condition) |

**Nominee and stopping.** MMP is the single registered nominee: no grid, strength or checkpoint, and no rescue, retune or new nominee.

## 5. Units, runtime, commands and dependencies

**Units.**
- Stage 0: 14 CatBoost fits (2 seeds × (2 SEX-aware teachers + 5 propensity folds)), plus the SEX-blind gate teachers and the proxy fits.
- Stage 1: 84 new reader fits, 84 refits of the frozen comparators for equal treatment, 126 learning-curve fits and 4 control fits.
- Stage 2: one bootstrap evaluation.

**Runtime estimate:** Stage 0 about 0.5 CPU-h, Stage 1 about 1.0, Stage 2 about 0.1; about 2.5 CPU-h total with margin.

**Commands.**
```
# rehearsal (this campaign; synthetic stand-ins; gate override allowed only here)
PYTHONPATH=. python -m mechanism_design.jobqueue run --unit <ID> --stage pilot_rehearsal --rss-gib 1.5 -- \
  python -m mechanism_design.pilot.run_pilot --source synthetic --scale 0.1 --max-trees 60 --rehearsal-continue-after-gates --out <file>
# real data (refused here; a future authorised campaign only)
python -m mechanism_design.pilot.run_pilot --source real --out <file>
```

**Configuration:** `mechanism_design/pilot/config.py`. The protocol hash is the SHA-256 of `PILOT` (sorted JSON).

**Dependencies not held by this campaign:**
- the admitted Florida 2019 role exports and extraction receipt (source private tree);
- the frozen comparator outputs (frozen-hybrids bank and U0/Ucal);
- the source private pylib's CatBoost. This campaign installed the same version into its own `.private/` for rehearsal.
- the authorisation record.

## 6. Rehearsal evidence

Unit `COORD_PILOT_REHEARSAL_V1`, 28 CPU-seconds, wrote `pilot/PILOT_REHEARSAL_RESULTS.json`. The data were invented, at 10% role sizes with at most 60 trees, and the comparators were stand-ins.

**Every stage executed:**
- Stage-0 gates, clauses, proxies and the merge diagnostic;
- Stage-1 readers with 21 pair candidates, the learning curves and the controls;
- Stage-2 recovery endpoints (8) and utility plus subgroup endpoints (28), K = 36, with household bootstrap critical values.

**What the rehearsal does and does not show.** Its numbers are meaningless as evidence. On the stand-ins G1–G3 failed, and the override was used only to exercise the path. One seed's shuffled-null control fell outside [0.45, 0.55] at 300 validation rows, and the control logic flagged it correctly as invalid.
