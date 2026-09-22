# Method lineage notes: what was actually implemented

Scope: this is a read-only trace of committed source code for an independent claims audit, written 2026-09-22 and not committed. It did not open ACS row data, any 2016 data, predictions or scores. The only committed JSON it read was result summaries: `MATH_REPLAY_*_anchor*.json`, `SELECTION.json` route nominees, `CODE_CONSTRUCTION.json` and `RESULTS_AT_A_GLANCE.md`/`RESEARCH_DECISION.md` tables.

The question: a conversational narrative describes one combined framework ("purpose-specific representations" + "fixed prediction services H with auxiliary channel J" + "coalition/stochastic release Q"). **The code contains four separate method families. No pipeline composes the original purpose-specific LoRA/erase-layer encoder with the H/J/Q line.**

Refs (short SHAs):

| Name | SHA |
|---|---|
| main | `0eee48f99` |
| origin/main | `55e4cb1d1` |
| erase-layer-pilot-2026-05-17 | `65dd5c050` (ancestor of main) |
| erase-layer-vicreg-sweep-2026-05-18 | `39c5a84e6` (merge-base of main and the task-directed branch) |
| rebuttal-evidence | `5739d3bb9` |
| ablations-facct-2026-07-24 | `ad2c08872` |
| residual-spectral-20260910 | `349efa454` |
| research/pcrl-competitive-method-v1 | `7f961d5c7` |
| research/pcrl-invariant-baselines-v1 | `73903b7f2` |
| research/pcrl-nonlinear-rank-v1 | `c37807e4f` |
| research/pcrl-stochastic-channel-v1 | `cd895e429` |
| research/pcrl-stochastic-replacement-overnight-v1 | `e3415b94d` |
| research/pcrl-task-directed-release-v1 | `f4bdf4cd5` (this worktree) |

Family 2/3 source files are byte-identical across `f4bdf4cd5`, `349efa454`, `ad2c08872`, `c37807e4f`, `73903b7f2` and `7f961d5c7`. The one exception is an additive `extra_candidates` option in `acs_spectral_audits.py` relative to `349efa454`. Unless a line says otherwise, family 2–4 citations are at `f4bdf4cd5`.

---

## Family 1: shared encoder + per-purpose LoRA adapters / erase layer (the NeurIPS "One Encoder, Many Purposes" line)

- **Backbone.** `StandardEncoder` is at `pcrl/models/encoder.py:13@0eee48f`. Each hidden layer is Linear→BN→ReLU→Dropout (`:37-42`), followed by `repr_proj` (`:65`). The v2 runner builds it at `experiments/run_v2_dataset.py:243-246@0eee48f` as `[128,128]→64`, dropout 0.3.
- **Frozen, and never pretrained, in the v2 runner.** Evidence:
  - The backbone is built right after `torch.manual_seed` (`run_v2_dataset.py:217-246`).
  - `requires_grad_(False)` is set at `pcrl/models/lora.py:181-182` and again at `pcrl/training/v2_trainer.py:335-336`. BN is held in eval mode (`v2_trainer.py:346,509-514`).
  - The optimizer receives only LoRA and head parameters (`v2_trainer.py:494-495`).
  - The backbone reload at `run_v2_dataset.py:342` loads the run's own checkpoint, not a pretrained model.
  - So in v2 the "shared encoder" is a frozen randomly initialised MLP. The older v1 FiLM path (`configs/adult.yaml`, [256,256]→128) is a different model.
- **LoRA.**
  - `LoRAAdapter` is at `lora.py:51` and includes an adapter bias (`:98`). `PerPurposeLoRAEncoder` is at `lora.py:134`. There is one adapter per purpose per Linear (`:212-226`), attached through forward hooks (`:234-245`).
  - Rank and alpha come from `run_v2_dataset.py:93-101`: 8/16 for adult, hmda and folktables; 24/48 for diabetes.
  - The per-purpose representation is z_p = `encoder(x,p)`, 64-d (`v2_trainer.py:792-796`).
- **Erase layer.**
  - It is a frozen 128×128 Linear, initialised to identity, sitting between the backbone and `repr_proj` (`encoder.py:52-61,95-98`).
  - It is fitted by `V2Trainer.fit_erase_layer`, which calls `LeaceEraser.fit(H, A_oh)` at `v2_trainer.py:584`. A_oh is the one-hot union of all purposes' disallowed attributes (`:567-579`), and W=I−pl·pr is written into the layer at `:588-601`.
  - It is enabled with `--use-erase-layer`. When it is on, the per-purpose warm start is skipped (`run_v2_dataset.py:309-316`).
- **LEACE warm start (default).** `leace_warm_start` (`v2_trainer.py:619`) fits LEACE per purpose on the 64-d output against that purpose's disallowed attributes (`:671-684`). The result is loaded into the last-layer LoRA by a rank-r SVD (`lora.py:327-394`).
- **Training objective.** One joint loss over all purposes, with a single backward/AdamW step (`v2_trainer.py:792-944`). Its terms:
  - Task CE per purpose, on that purpose's first allowed task (`:798-809`).
  - VICReg (`:811-819`).
  - vCLUB (`:837`).
  - Per (purpose, protected attribute) **linear ridge R² constraints**, per-class one-vs-rest for high-K attributes (`:856,:880`). The R² computation is `VerificationRegularizer` at `pcrl/training/losses.py:475`, with the solve at `:496-518` and R² at `:538-541`/`:562-565`. τ=0.05 (`run_v2_dataset.py:278`).
  - The constraints are enforced by a proxy-Lagrangian (`v2_trainer.py:936`; `pcrl/training/proxy_lagrangian.py:133,153`), with λ_min=5 and Cotter best-iterate selection (`v2_trainer.py:1207`).
  - An optional concatenated cross-purpose R² constraint is at `v2_trainer.py:887-925`.
- **Labels.** Permitted and protected labels are defined per purpose: Adult `pcrl/data/adult.py:456-479`, HMDA `hmda.py:128-146`, Diabetes `diabetes.py:101-119`, Folktables `folktables.py:378-396`.
- **Data.** Adult, HMDA, Diabetes and Folktables through `run_v2_dataset.py:484`. There are also separate CelebA (`pcrl/vision/`) and BIOS (`pcrl/language/`) variants. **None of this is ACS 2018/2017 CA household-split work.**
- **Auditor/attacker.**
  - Audits receive continuous z_p per purpose (`pcrl/evaluation/certificates.py:331,491-509`).
  - The coalition attack concatenates all purposes' representations, `[h_p1|h_p2|h_p3]` = 192-d (`experiments/run_cross_purpose_attack_v2.py:10-11,388-389`). It uses LR, MLP, XGB, RF and SVM.
- **Reuse in ACS work: none.** A `git grep` for `lora|LoRA|EraseLayer|erase_layer|PCRLModel|PerPurposeLoRA|V2Trainer|StandardEncoder|leace_warm|fit_erase|pcrl.models|pcrl.training` over `experiments/acs_*`, `experiments/run_acs_*` and `experiments/pcrl_*_v1/*` returns no hits at `f4bdf4cd5`, `349efa454` or `ad2c08872`. `(from|import) pcrl\b` has no hits in the family-4 package either.
  - The only `pcrl/` import anywhere in the ACS code is `pcrl.baselines.splince.fit_splince`, imported lazily as a baseline (`pcrl_invariant_baselines_v1/erasure_baselines.py:222`, `pcrl_direct_adversarial_v1/erasure.py:113`, `transport.py:74`, `pcrl_competitive_method_v1/run_fit_e.py:319`).
  - The branch only touches `tests/test_lora.py` to update a parameter count for the ablations `LinearAdapter`.

## Family 2: fixed prediction services H + neural auxiliary channel J (ACS)

- **Data.** ACS **2018 California** 1-year (`psam_p06`), ages 19–34, PWGTP>0, capped at 30,000 persons (20,147 households). Seven household pools, .35/.10/.15/.10/.10/.10/.10 (`experiments/acs_transfer_data.py:22-25,56-72,88-96`; `results/redesign_20260907_acs_transfer_v1/config.json:8`).
  - 2017 CA is a separate transport cohort. Frozen 2018 maps are applied to it and nothing is refit (`acs_spectral_transport.py:15-16,29-33,62-77`; `pcrl_direct_adversarial_v1/transport.py:8`).
  - The cohort is development-exposed, and the seed variants reuse the same people (see the `ARTIFACT_DEFINITIONS.md` audit).
- **PCA32 inputs.**
  - Covariates: AGEP and WKHP (standardised, plus missing flags) and one-hot SCHL, MAR, RELP, CIT, DIS, DEAR, DEYE, DREM (`acs_transfer_data.py:15-17,147-183`).
  - PCA(32) is fit on representation_fit (`run_acs_transfer.py:152,181`).
- **H (frozen).**
  - Heads use PCA32 input and are trained on income>50k, ESR==1 and PUBCOV==1 (`run_acs_bottleneck.py:36-41`). Candidates are logistic or MLP[64,32], selected by validation log-loss on `downstream_fit` (`run_acs_protection.py:258-263`; `acs_transfer_heads.py:28-29,363-364`).
  - H_A = [income0, income1, employment0, employment1]; H_B = [coverage0, coverage1] (`results/redesign_20260909_acs_fixed_predictions_v1/INPUT_SCHEMA.json`).
  - Reproduced only through `predict_proba`, with bit-equality checks (`run_acs_fixed_predictions.py:46-63`).
- **J.**
  - **Definition.** `COEF['J']=(-.1,0,-.1)` at `experiments/acs_fixed_predictions_training.py:14`. `class Auxiliary` is at `:15-27`, wrapping a PurposeBranch mapper Linear(32,64)→ReLU→Linear(64,16) with income and employment source heads (3186 parameters). The mapper is initialised from PCA16 (`:135-136`).
  - **Input.** Standardised PCA32 **only**; H_A is not an input (`:22-26`).
  - **Wires.** `tensor_wires` builds A=[H_A,J] (20-d), B=H_B (2-d), AB=[A,B] (22-d) (`:41-43`).
  - **Loss.** Mapper loss = source − .1·individual − .1·coalition (`:84-93`), where source = .25·BCE_income + .25·BCE_employment (`:60-62`).
  - **Observer losses** (`experiments/acs_coalition_training.py:165-178`). Each is entropy-normalised observer CE:
    - individual = ½(mean over A roles + mean over B roles)
    - coalition = mean over AB roles
    - Roles: A:{public_coverage, SEX, RAC1P}; B:{income, employment, SEX, RAC1P}; AB:{SEX, RAC1P} (`:29-33`).
  - **Schedule.** 60 source epochs, 20 observer warm-up epochs, then 80 continuation epochs with 3 observer steps per mapper step (`acs_fixed_predictions_training.py:132,150-173`).
  - **B gets zero mapper gradient.** This is asserted at `:107-110`.
  - **Labels used to train J.** Task: income and employment (coverage is logged but not back-propagated). Protected: SEX and RAC1P, plus public_coverage as an A-view forbidden role.
- **Only A receives a new channel; B's H_B is fixed. A and AB observers are optimised jointly within one mapper.**
- **Constraint type.** An attacker penalty against trained observers. It is not R² and not CMI.
- **Attackers.**
  - They see continuous float64 H and J; nothing is binned.
  - The slate is logistic, 2×MLP, 2×HistGB, the saved observer, and catch-up retrains (`acs_fixed_predictions_audits.py:30-35,213-224,303-309`), plus RFF kernel ridge (`acs_spectral_audits.py:13,22-36`).
- **Direct-adversarial variant (`pcrl_direct_adversarial_v1`).**
  - The mapper is initialised from A0. It is trained with a β-weighted attacker penalty over policies L1/L2/C1 (`train.py:9-17,99-121`).
  - The service baseline sees H_A for A roles and [H_A,H_B] for AB roles, never Z (`attackers.py:42-48`).
  - Wires: A=[H_A,Z], B=H_B (`inputs.py:188-199`).
- **Competitive-method variant.** Affine partial projections of the frozen A0/J outputs (`pcrl_competitive_method_v1/run_fit_e.py:1-18,53`; `projection.py`).
- **Utility-extension variant.** Appends an R channel computed from PCA32 only: [H_A, Z_J, R] (`pcrl_utility_extension_v1/extension.py:1-10`).

## Family 3: residual spectral / erasure variants (ACS 2018, 2017 transport)

- **Features** (`experiments/acs_residual_spectral.py`):
  - V = [standardised PCA32, 96 RFF], 128-d (`:19,250-259`).
  - Least-squares residualised on qA(H_A[:,[1,3]]) (`:260-267`), then whitened (`:268-273`). Inference reads only PCA32 and H_A (`SpectralModel.features`, `:196-203`).
- **qA/qAB are not bins.** qA is a degree-≤2 polynomial basis in P(income=1) and P(employment=1) (`:60-91`). qAB adds H_B[:,1] (`:261-262`).
- **Teacher.** A **label-free** residual teacher: R = PCA32 minus its qA projection, centred (`:275-277`). Utility is U = trace-normalised (VᵀR/n)(VᵀR/n)ᵀ (`:278-279`).
- **Constraint.** A soft penalty on linear conditional first moments E[V⊗basis(H)·(onehot(s)−p̂(s|H))] (`:151-170`).
  - Nuisances p̂ come from household out-of-fold logistic regression on the basis (`:113-148`).
  - Local roles are {SEX, RAC1P, public_coverage} on qA; coalition roles are {SEX, RAC1P} on qAB (`:283-304`).
- **Map.** W = the top-16 eigenvectors of U−λP in whitened coordinates (`:306-316`), for arms S0, M, L, C (`:21-24`).
- **Recipients.** A=[H_A,Z16], B=H_B, AB=[A,B] (`run_acs_residual_spectral.py:62-65`). The map is fitted on representation_fit with SEX, RAC1P and PUBCOV (`:43,51-55`).
- **Related variants.**
  - Nonlinear-rank: Stiefel-manifold RFF/quadratic moment penalties at ranks 16/8, policies L1/L2/C1 (`pcrl_nonlinear_rank_v1/maps.py:65-136`, `nonlinear_moment.py`, `objective.py:31`).
  - Invariant baselines: LEACE on the A0 16-d channel via the external `concept_erasure` library, targets SEX+RAC1P+coverage (`pcrl_invariant_baselines_v1/erasure_baselines.py:171-212`); SPLINCE (`:216`); OptNet-ARL on V128 (`optnet_arl.py:44-80`).
- **Attackers.** Continuous H/Z. The spectral audits add RFF kernel-ridge candidates.

## Family 4: task-directed finite stochastic release Q (`experiments/pcrl_task_directed_release_v1/`, `f4bdf4cd5`)

### Precursors

- **`pcrl_stochastic_channel_v1@cd895e429`.**
  - The code T is **label-free** KMeans(64) on standardised PCA32 (`stage_b.py:16-20,37-54`).
  - The release augments J: S0=[H_A, Z_J, onehot(T)/R] (`stage_b.py:14`; `slate.py:1-19`).
  - CVXPY CMI solver (`channel.py:151-177`). The solver's duality gap is recorded as `optimality_gap` when `solver_stats` supplies it (`:170-176`).
  - Closed NEGATIVE at gate G2.
- **`pcrl_stochastic_replacement_overnight_v1@e3415b94d`.**
  - Label-free KMeans(64/256) codes on PCA32 or on Z_J (`replacement.py:9-13,96-100`).
  - Contract S1: A=[H_A,Z], J only a comparator (`replacement.py:1-6`).
  - Expected loss averages losses, not probabilities (`tokens.py:1-12,62`). The alternative is kept only as `loss_of_expected_prediction` at `:86`.
  - Closed NEGATIVE at the representation gate.

### Latest (task-directed) implementation

- **Inputs.** `RuntimeInputs(x_a, h_a)` accepts exactly the historical standardised PCA32 plus the 4 H_A columns (`data.py:23-36`). The PCA and standardiser are loaded from the saved `redesign_20260909_acs_fixed_predictions_v1` artifacts (`data.py:97-102`). Data is ACS 2018 CA `psam_p06` (`data.py:14`).
  - **No LoRA/erase-layer or original learned encoder is used.** Historical J and A0 outputs are loaded **read-only as comparators** (`data.py:100-118`).
- **Teachers (supervised; this is not a label-free code).** `fit_encoder` is at `encoding.py:211-257`.
  - `task` = teacher slate (logistic C∈{.03,.3,3}; MLP[64,32] with wd∈{0,1e-4}, 200 epochs; `config.py:71-74`; `audits.py:602-609`) on [PCA32, H_A], trained on **same_residence** labels (`encoding.py:215-222`).
  - `baseline` b = the same slate on H_A only, trained on same_residence.
  - Selection uses internal household validation; the teachers are frozen with no refit (`encoding.py:236-238`).
- **Residual.** r = logit p − logit b (`encoding.py:202,229`).
- **Code T = g(PCA32, H_A).** At runtime it is deterministic and reads no labels, but it is **fitted with labels** (`Codebook.fit`, `encoding.py:60-87`; `assign`, `:89-101`):
  - **T0** = 32 quantile cells of r (`:66`).
  - **Ttask** = each T0 cell split at the within-cell median of r, with ties broken by a fixed cosine projection of PCA32 (`:72-75,93`).
  - **Trisk** = each T0 cell split by 2-means on predicted [SEX2, RAC1P9] probabilities (`:68,76-77,97`). These come from `RiskPredictor`, a logistic model on [PCA32, H_A] trained on **protected SEX/RAC1P labels** (`encoding.py:109-141,224-226`).
- **Action dictionary.** 17 actions (33 reserved), `fit_dictionary` at `encoding.py:164-185`.
  - Offsets a_z are stationary roots of the soft-label CE, with target the **teacher p**, inside 16 residual-quantile groups; zero is action 0.
  - Decoder: g_z(H_A) = σ(logit b(H_A) + a_z) (`encoding.py:185`).
- **Cost table D.** `estimate_tables` (`mechanisms.py:72-147`) calls `finite.cost_table` (`finite.py:85-109`).
  - D[t,z] = ½d_U + ½d_W, with d_w[t,z] = Σ_{i:T_i=t} w_i·CE(Y_i, g_z(H_Ai)) / Σw. Y is the **same_residence label** on mechanism rows (`mechanisms.py:88,104-105,119`).
  - **D depends on the residence teacher** three ways: through T via r, through the offsets a_z via p, and through b. It also uses residence labels directly.
  - Teacher, code and mechanism rows are separate household roles (.48/.12/.40; `data.py:39-46`).
- **Conditioning views: coarsened.**
  - C_A = KMeans(2) on H_A[:,[1,3]] → **2 cells**.
  - C_AB = C_A × 1{H_B[:,1] > median} → **4 cells**.
  - Source: `ServicePartitions`, `encoding.py:144-161`; used at `mechanisms.py:93-96`; configured at `config.py:88-90`. A robustness variant crosses KMeans4 cells with the same H_B split.
- **Disclosure constraint.** Finite-table CMI I(S;Z|C) ≤ δ for S∈{SEX, RAC1P}, C∈{C_A, C_AB}, each under both unweighted (U) and PWGTP (W) weighting (`mechanisms.py:107-118`).
  - Budgets δ ∈ {.0005, .002, .01, 0} (`config.py:16`).
  - Policy **L** constrains only the `A/*` roles; **C** adds `AB/*`; **U** is unconstrained (`mechanisms.py:197-198`).
  - There is no R², covariance or attacker penalty inside Q fitting.
- **Solver** (`finite.py:273-431`):
  - δ=0 or unconstrained: scipy `linprog(method="highs")` with the exact independence equations (`:364-382`).
  - δ>0: **CVXPY** `rel_entr` cone, trying **CLARABEL**, then **SCS** as fallback (`:384-417`).
  - Every attempt records solver, status, reported objective, raw simplex error and minimum, repair size, CMI and residuals computed independently in NumPy/SciPy, warnings and time (`:337-362`).
  - A candidate is accepted only if it is feasible under the tolerances at `:25-34` and its status is optimal or optimal_inaccurate. A witness is labelled `feasible_witness`, never `optimal` (`:419-424`).
  - **No dual variables and no duality-gap bound are extracted or recorded.** The `HIGHS` "dual_feasibility_tolerance" setting (`:36`) is only a solver option. The precursor did record `optimality_gap`.
  - Local-vs-coalition ordering is asserted only when both solver statuses are optimal (`MATH_REPLAY_*`, `local_coalition_ordering`).
- **Release.** A receives [H_A, one sampled token Z]; B receives exactly H_B; AB receives [H_A, H_B, Z] (`RELEASE_CONTRACT.md:3`).
  - `build_release` (`mechanisms.py:251-349`) supplies `token_probs = Q[T]` for exact expectation. Only the token is on the wire.
  - **Only A gets an optimised channel. One Q, for recipient A, is fitted jointly under the A and AB constraints (policy C). B is never given a channel** (`evaluation.py:160-170`: the B view is always H_B only).
- **Expected loss.**
  - `audits.expected_token_loss` (`audits.py:167-180`) computes Σ_z Q[T_i,z]·(−log f(y_i|H_i,z)), the **expectation of the loss**, not −log Σ_z Q·f. Callers: `evaluation.py:210-212` and `audits.py:317-327`.
  - Q fitting uses the same form (`finite.cost_table`; objective Σ D·Q, `finite.py:261,400`). The only exception is the fixed-decoder diagnostic, which evaluates each token's decoder prediction under the same token-expectation formula (`evaluation.py:185-192`).
  - Training uses exact person-token expansion with conserved weights (`audits.py:155-164,201-206`).
- **Attack slate.**
  - Every role fits the `catchup` slate on the full continuous H plus the one-hot token, with no binning (`evaluation.py:296-310`; `audits.py:209-217`). Slate: logistic with H×onehot interactions; MLP[64,32] at 120/360 epochs; HistGB with leaf sizes 20 and 5; the same trees on one sampled token.
  - **H-only ancestors are included:** every model candidate from the H release is routed with `wire='H'` (`evaluation.py:316-327`).
  - **For AB roles, all A and B singleton candidates are added as well** (`:328-333`).
  - **J cannot be an ancestor:** `'Only H-only candidates are legal service ancestors; no J ancestor'` (`:269-270`). J features enter a wire only when the release being audited is J itself (`mechanisms.py:302-303`).
  - For residence utility, the deployment decoder and all H-only global-offset recalibrations are also candidates (`evaluation.py:334-344`).
- **Roles.**
  - Primary attacks: A/SEX, A/RAC1P, AB/SEX, AB/RAC1P.
  - Secondary: A/{public_coverage, commute}, B/*.
  - Utility: A/{same_residence, income, employment}, B/{coverage, commute} (`config.py:10-14`).
- **Evaluation data.** ACS 2018 CA, on the inherited historical `test` pool, gated behind the selection-freeze permit (`data.py:49-58,86-93`). There is **no 2017**, and 2016 stays sealed (`config.py:128`; `archive.py:31`).
- **Other controls in the same package.**
  - Supervised LEACE/SPLINCE on [PCA32, teacher logit], with SEX/RAC1P protected and residence, income and employment preserved (`baselines.py:1-23`; `run.py:176-180`).
  - Historical LEACE/SPLINCE/OptNet on A0; continuous_task; code; randomized-response; withhold; constant; independent_token (`config.py:138-144`; `mechanisms.py:251-337`).

### Selected releases and whether coalition constraints bind

The frozen nominees are `SELECTION.json` → `routes`:

- **utility_first → `T0_L_0.01_a17` (local).** In `MATH_REPLAY_FINAL_anchor{0,1,2}.json`:
  - A-role CMI sits at the 0.01 budget.
  - **AB CMI is 0.0177, 0.0165 and 0.0177 nats**, which exceeds δ. No coalition constraint is imposed; AB coalition leakage in the fitted finite model is above the budget the coalition policy would enforce.
- **protection_first → `Ttask_C_0.01_a17` (coalition).**
  - AB CMI sits at 0.0100, which is **active**. A-role maxima are 0.0077, 0.0081 and 0.0086, which are **slack**.
  - The objective is higher than the local map with the same input by 0.0047, 0.0035 and 0.0018 nats. Both statuses are optimal for all three anchors.
  - So the coalition constraint **binds** in this nominee.
  - In `MATH_REPLAY_PRIMARY_anchor0.json`, the pre-recovery Ttask_C objective equalled T0_C (0.469549). This is consistent with an embedded-witness solution that was later replaced by numerical recovery (FINAL gives 0.463005).
- **Adjusted claims.** No adjusted coalition-attribution claim passes (`RESULTS_AT_A_GLANCE.md`). Both routes fail their competitive conjunction (`RESEARCH_DECISION.md`).

---

## Implementation-to-claim table

| Claim (narrative) | Implementing object(s) | Evidence file | Tested? | Notes |
|---|---|---|---|---|
| Shared encoder + purpose-specific representations | `StandardEncoder` pcrl/models/encoder.py:13@0eee48f; `PerPurposeLoRAEncoder` pcrl/models/lora.py:134@0eee48f | experiments/run_v2_dataset.py:243-251@0eee48f | Adult/HMDA/Diabetes/Folktables (+CelebA/BIOS variants) | The backbone is frozen at random init in v2 (no pretraining). Not ACS. |
| Erase layer with LEACE | encoder.py:52-61,95-98; v2_trainer.py:584-601@0eee48f | results/rebuttal/erase_layer_pilot_aws (per memory) | Adult/HMDA/Diabetes | 128-d backbone space, union of all purposes' protected attributes |
| Linear-R² disclosure constraint | pcrl/training/losses.py:475-565; v2_trainer.py:856,880,936@0eee48f | — | same | Proxy-Lagrangian with τ=0.05. **This constraint appears in no ACS/Q code.** |
| Fixed prediction services H_A, H_B | run_acs_fixed_predictions.py:46-63; INPUT_SCHEMA.json | redesign_20260909_acs_fixed_predictions_v1 | ACS 2018 CA | PCA32 heads trained on income/employment/coverage; frozen, byte-identical |
| Auxiliary channel J | acs_fixed_predictions_training.py:14-27,84-93,41-43 | seed_s/training/J/{final.pt,releases.npz} | ACS 2018 (+2017 transport) | PCA32-only input; observer penalty −.1·individual −.1·coalition; A only |
| Residual spectral map | acs_residual_spectral.py:196-316 | results/redesign_20260910_acs_residual_spectral_v1 | ACS 2018 (+2017 transport) | Label-free residual teacher, linear-moment penalty, polynomial (not binned) conditioning |
| Label-free code T=g(X_A) | stochastic_channel_v1/stage_b.py:37-54@cd895e4; replacement.py:96-100@e3415b9 | precursor results | closed NEGATIVE | **Not the latest.** The latest T uses a residence teacher and a SEX/RAC1P risk model. |
| Task-directed code T0/Ttask/Trisk | encoding.py:53-141,211-257@f4bdf4c | CODE_CONSTRUCTION.json | ACS 2018 | Fitted with same_residence, SEX and RAC1P labels; runtime input is PCA32+H_A |
| Cost table D | mechanisms.py:72-147; finite.py:85-109 | MATH_REPLAY_*.json | yes (replayed) | Residence-label CE of the teacher-derived action decoders; ½U+½PWGTP |
| Stochastic release Q with CMI constraints | finite.py:273-431; mechanisms.py:186-237 | MATH_REPLAY_*.json, NUMERICAL_RECOVERY*.json | yes | CMI on a finite table with 2- or 4-cell coarse C; HiGHS/CLARABEL/SCS; no dual bounds |
| Coalition (AB) protection | mechanisms.py:197-198 (policy C) | MATH_REPLAY_FINAL_anchor*.json | yes | Binds in Ttask_C_0.01; **absent in the utility-first local nominee** (AB CMI ≈ .017 > .01) |
| B gets an optimised channel | none | evaluation.py:160-170 | n/a | **Not implemented.** B always receives H_B. |
| Expected loss Σ_z Q·loss | audits.py:167-180; finite.py:85-109 | METHOD.md | unit tests plus replay | Average of losses, not loss of the average |
| Attack slate with H-only ancestors, no J in non-J views | evaluation.py:269-270,316-333 | ATTACK_CALIBRATION.json | yes | Continuous H; AB inherits A and B singleton candidates |
| Composition of LoRA purposes with H/J/Q | none | git grep (no hits) | **no** | No code path exists |

## Actually implemented paths (per recipient)

```
FAMILY 1 (Adult/HMDA/Diabetes/Folktables; main 0eee48f)
 x ─► frozen random StandardEncoder ─[erase layer: LEACE on 128-d, all purposes]─► repr_proj
      └─ LoRA_p (per purpose, jointly trained: task CE + VICReg + vCLUB + ridge-R^2 ≤ τ proxy-Lagrangian)
 purpose p ─► z_p (64-d continuous)      coalition attack: [z_1|z_2|z_3] (192-d)

FAMILIES 2/3/4 (ACS 2018 CA; 2017 transport for 2/3 only). Shared frozen inputs:
 covariates(10 cols) ─► PCA32 ─► H heads (income, employment │ coverage)   [frozen, byte-identical]
                                   H_A = 4 probs                H_B = 2 probs

 Recipient A:
   F2  [H_A, J(PCA32→64→16; observer-penalized)]              J = neural aux channel
   F3  [H_A, Z16 = W^T whiten(resid_qA(PCA32⊕RFF96))]        spectral / LEACE / OptNet variants
   F4  [H_A, Z ~ Q[T(PCA32,H_A), ·]]   one sampled token of 17
         T: residence teacher p(PCA32,H_A), b(H_A) → r → 32 quantile cells (T0) [+median split (Ttask) | +SEX/RAC1P-risk 2-means (Trisk)]
         Q: min Σ D·Q  s.t. CMI(S;Z|C_A[2 cells]) ≤ δ  (L)   [+ CMI(S;Z|C_AB[4 cells]) ≤ δ  (C)]
 Recipient B:
   F2/F3/F4  H_B only (no new channel in any family)
 Coalition AB:
   F2  [H_A, J, H_B]    F3  [H_A, Z16, H_B]    F4  [H_A, H_B, Z]
 Attackers (F2-F4): continuous H (+J/Z16 or one-hot Z); for F4 plus H-only ancestors and A/B singletons; J never an ancestor of F4.

 NOT IMPLEMENTED: F1 LoRA/erase output ─► any of F2/F3/F4;  J ─► input of Q;  optimized channel for B.
```
