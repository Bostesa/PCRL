# Methodology findings: PCRL encoder lineage ("One Encoder, Many Purposes", NeurIPS 2026, withdrawn 2026-08-21)

Date: 2026-10-01. Role: METHODOLOGY (PCRL encoder). Phase: analysis-first preparation. No training, cloud
compute, or new labels were used.

**Manuscript cited.** "PDF(23) l.N" is the printed line number of
`~/Downloads/Formatting_Instructions_For_NeurIPS_2026 (23).pdf`, which is text-identical to `(24).pdf`. That
file is the latest pre-deadline build (see F0). Appendix tables carry no line numbers and are cited by table.

**Repository refs.** Unless stated otherwise, references are to `Bostesa/PCRL`.
- `origin/main` = 55e4cb1d1.
- The headline tabular code is b158abc54 (Round 7 tip). The Round 5 lineage is dbe0fdcc2 + b3f43c53c.
- The paths are unchanged on origin/main.

**Status labels.**
- [SRC]: source inspected.
- [RES]: stored results checked or recounted.
- [NR]: not independently rerun.
- [UNV]: unverified.

**Supporting files** (all in this directory):
- `ARCHITECTURE.md`
- `WORST_CLASS_AUDIT.md`
- `recounts.json` and `recount_headlines.py`
- `recount_cross_purpose.py`
- `worst_class_fixture.py`
- `accuracy_bound_counterexample.py`
- `crosswalk_rows.csv`, whose row ids are given in brackets after each finding
- `data_exposure_rows.csv`

**Severity scale.**
- critical: a headline claim is false as stated.
- high: a headline number or method claim depends on an undisclosed choice or artifact.
- medium: misdescription or a weak protocol.
- low: cosmetic.

---

## F0. Which manuscript is "final"? Severity: medium for provenance. [PE-M01]

**Problem.** The source directory `(21)` (main.tex, paper-body/) is **not** the final submitted text.
- `(21)`'s zip is dated 2026-05-07 03:37. Its text matches PDF `(22)` (03:25 EDT, 31 pp).
- PDF `(23)` (06:48 EDT, 26 pp) is later and was substantially rewritten:
  - the abstract now says "26 of 33" where `(21)` says "22 of 33";
  - propositions were renumbered: the DA identity went from Prop 4 to Prop 5, Linear Compliance from Prop 2
    to Prop 3, and Zhao–Gordon from Prop 5 to Prop 1;
  - the contributions paragraph was rewritten;
  - BIOS was dropped;
  - CelebA was folded into §5.5.
- `(24).pdf` (saved 2026-09-06, after the withdrawal) has text identical to `(23)` but a different sha256.
  That is consistent with a later re-download of the submitted PDF. [UNV]

**Evidence.**
- pdfinfo creation dates.
- `diff` of the pdftotext output of (22) against (23).
- sha256: (23) 4a705d5e…8172, (24) a9735896…ed9, (21) S5-experiments.tex ada6ce7f…e5.
- No PCRL ref has ever tracked `paper-body/*.tex`, only `paper-body/tables/` (`git log --all -- 'paper-body/*.tex'` is empty).

**Repair.** Treat PDF(23) as the citable final text. Mark `(21)` as near-final source. Obtain the
OpenReview copy if access becomes available.

**Verification.** The OpenReview PDF sha256 or text equals (23).

---

## A. Architecture and training (Q1)

### F1. The "pretrained" backbone is a frozen, seeded random MLP. Severity: critical. [PE-A01]

**Problem.** PDF(23) l.133-135 and Table 4 say the backbone is "pretrained on the union of allowed tasks"
with "BN frozen". The code does something else:
- `run_v2_dataset.py:214-216` constructs a fresh `StandardEncoder([128,128]→64, dropout 0.3)`, with
  Kaiming/Xavier initialisation (`encoder.py:47-56`).
- It is frozen immediately (`lora.py:175-176`).
- BN is put in eval mode before any forward pass (`v2_trainer.py:275-288, 392-397`). The running statistics
  therefore stay at their defaults, and BN is approximately the identity.
- No code path pretrains the backbone or loads external weights.
- Each seed draws a different backbone, so seed variance includes backbone variance.

**Evidence.** b158abc54 paths above [SRC]. This is consistent with
`research/pcrl-claims-foundation-v1:results/pcrl_claims_foundation_v1/CORRECTIONS.md:19` (T3-10).

**Repair.**
- In every reuse, describe the backbone as a "frozen seeded random ReLU feature map".
- Any pretrained-backbone arm is a new, registered experiment.

**Verification.** At the checkpoint level, reconstructing `StandardEncoder` under `torch.manual_seed(s)`
reproduces the stored backbone tensors bitwise. This needs the relocated checkpoints; not done this phase.

### F2. vCLUB is active (λ=1.0) in the headline loss, but the paper says 0.0. Severity: high. [PE-A02]

**Problem and evidence.**
- `run_v2_dataset.py:246` sets `lambda_vclub=1.0` in every round (dbe0fdcc2, b3f43c53c, f542c06, b158abc54,
  origin/main).
- `v2_trainer.py:602, 652-661` adds the term to the primal loss.
- The term is not detached (`vclub.py:100-116`).
- PDF(23) Table 4 and App. O l.711 instead say "λvCLUB (constrained phase) 0.0 (logged only)". [SRC]

**Repair.** Correct Table 4. Mark the HSIC/vCLUB ablation narrative (App. O) as describing a different
configuration.

**Verification.** `config.lambda_vclub == 1.0` inside the relocated `final.pt`/`best.pt`.

### F3. The training-time R² constraint is a biased in-batch estimator that τ cannot satisfy without collapse. Severity: high (hypothesis for "compliance via collapse"). [PE-A03, PE-B06]

**Problem.** The proxy-Lagrangian constrains ridge one-hot R² fit and scored **within each minibatch**
(B = 256, d = 64, float32; `losses.py:496-541`, `v2_trainer.py:619-650`).
- Under independence, its expectation is about rank/(B−1), roughly 0.25 at full rank. That is far above
  τ = 0.05.
- It can only be met by representations of batch rank up to about 12, or by shrinking below ridge scale.

This is an untested alternative to the paper's explanation of collapse, which attributes it to the λ_min floor
and Proposition 4 (PDF(23) l.283-291, 619-627).

**Evidence.**
- Synthetic: rank 64 gives 0.249–0.253, rank 12–13 gives about 0.05, rank 1 gives 0.0035
  (`worst_class_fixture.py` §C; ARCHITECTURE §1.7) [SRC + fixture].
- Stored effective ranks are 1.0–8.8.
- HMDA underwriting s0 at best.pt has per-dim std 1.2e-5, i.e. a numerically constant representation that
  "passes".

**Repair.** Constrain a cross-fitted or bias-corrected statistic: half-batch fit/score, a running-covariance
estimate, adjusted R², or a cross-covariance norm. Report the null floor.

**Verification.** No training is needed. On relocated checkpoints, compute in-batch R² with permuted A (the
null floor) and the Gram spectra per cell.

### F4. The LEACE guarantee holds only at initialisation, and Propositions 2 and 4 do not describe the trained model. Severity: medium. [PE-A06, PE-M02]

**Problem.**
- LEACE is fit on the 64-d `repr_proj` output on the train split. It is realised by rank-r SVD in the
  `repr_proj` LoRA (`v2_trainer.py:401-501`, `lora.py:236-304`).
- LoRA adapters sit on **every** backbone Linear, including hidden layers that feed BN, ReLU and Dropout, and
  they are trained (`lora.py:183-206`).
- No nonlinearity follows h_p. However, the hidden-layer adaptation changes f0, so:
  - covariance-zero is not maintained after epoch 0;
  - Proposition 4's "fixed f0, final-layer rank edit h = (I+BA′)f0" bound does not apply to the trained model;
  - the "joint LEACE optimality" (Prop 2) concerns only the initialiser.

**Repair.** Present the guarantees as initialisation-only. Restrict Prop 4 to last-layer-only LoRA, or
re-derive it for all-layer adaptation.

**Verification.** Code inspection (done). Optionally, compare epoch-0 and final R² from stored histories.

### F5. Gradients and other low-severity training details. Severity: low. [ARCHITECTURE §1.5]

- Joint gradient clipping couples the purposes (`v2_trainer.py:670-674`). This contradicts "drift on one
  adapter cannot propagate" (PDF(23) l.45) only weakly, because adapters are disjoint but clipping is shared.
- The dual update is projected gradient ascent on the same in-batch estimator, not Cotter et al.'s proxy
  two-player scheme.
- The warmup skip is unconditional whenever LEACE init is on, although the text says "when feasible".
- Dropout 0.3 inside the frozen backbone is active during training.
- The environment is unpinned: `requirements.txt` gives lower bounds only, and the EC2 user-data for R5/R7 is
  not committed (`a00d5749b:experiments/round5_orchestrator.py:41-44`). [SRC; argv UNV]

---

## B. Checkpoints, counts and selection (Q4)

### F6. The 56/60 headline is exact on final.pt, but the same runs give 54/60 on the best.pt reload. Severity: medium. [PE-C01, PE-A04]

**Evidence.**
- `dominant_axis_audit.json` (final.pt, epoch 199) gives 23+16+17 = 56/60 (sha256 in `recounts.json`
  a1_strict_56_60).
- `per_seed_results.json` `linear_r2`, written after `run_v2_dataset.py:336-339` reloads canonical > best >
  final, gives 21+16+17 = 54/60.
- `results/laftr_benchmark/PCRL_R5_RECONCILIATION.md` mislabels the latter as train-time.
- Later rebuttal assets quote 54/60.

[RES]

**Repair.** Declare one checkpoint rule in advance. Report both counts.

**Verification.** Every reported count is tied to the sha256 of one checkpoint file.

### F7. "7/60 cleanly compliant" mixes final.pt R² with best.pt health. Four of the seven are recovered by nonlinear auditors. Severity: high. [PE-B04, PE-C02]

**Evidence.**
- `scripts/identify_collapse_cells.py:58-62, 78-88, 104-115` joins final.pt R² with `per_purpose_health`
  from best.pt.
- Recount on a single checkpoint (best.pt) gives 5 clean / 49 collapse / 6 failed. No final.pt health
  metrics are stored.
- Of the 7 "clean" cells, 4 have Δaud ≥ 0.02 (HMDA underwriting/race s2: +0.283).
- R², health and Δaud together pass on only 3/60. That figure appeared in draft (22) Table 14 and in PDF(23)
  Table 14 App. R ("PCRL 3/60"), not in the main text. [RES]

**Repair.** Compute health on final.pt; this needs a CPU forward pass on relocated checkpoints in a later
phase. Report the three-bar table in the main text.

**Verification.** Health and R² share a checkpoint hash.

### F8. 46/60 → 56/60 depends on the checkpoint rule and bundles three changes. Severity: high. [PE-C03, PE-A08]

**Evidence.**
- Round 4 is 46/60 on final.pt and 33/60 on Cotter best.pt (Adult 0/24, HMDA 16/18, Diabetes 17/18).
- On HMDA and Diabetes, best.pt is better.
- final.pt was adopted for all datasets after inspecting Adult (`results/v2_adult_ROUND4/cotter_selection_bug.md`).
- The Diabetes headline 17/18 is Round 7 (per-class OvR + rank 24), not the λ-floor/warmup fix it is credited
  to (PDF(23) l.614-618). Diabetes R5 is 16/18 on final.pt and 18/18 on best.pt.
- All quoted mean R² values reproduce. [RES]

**Repair.** Build a rule × round × dataset table and attribute each patch separately.

**Verification.** The table is generated from the three `final_vs_best.json` files.

### F9. All tuning used the test split, and "held-out Adult seed 3" is a new seed on the same data. Severity: high. [PE-A07, PE-A08, PE-C11, PE-E07]

**Problem and evidence.**
- Splits are fixed (seed 42) across seeds 0–3 and Rounds 1–7: `adult.py:196-204`, prepare_hmda
  `split_indices`, `preprocess_diabetes.py:254-269`.
- The compliance metric is in-sample OLS **fit and scored on test-split representations**
  (`certificates.py:494-512`).
- These were all chosen from that metric: λ_min = 5, the warmup skip, rank 8→16→24, the K≥6 OvR gate, and
  final.pt over best.pt (`eval_round4_final_vs_best_v2.py:46-91`; `V2_ROUND5_SUMMARY.md`;
  `V2_DIABETES_ROUND7_VERDICT.md`).
- PDF(23) footnote 1 and l.248-257, 356-371 disclose post-hoc tuning "on the same 60-cell grid". They do not
  say that the grid is the test split, and they describe seed 3 as "untouched".
- Recounts reproduce 6/8 vs 7/8 and Δ = +0.0111. [SRC/RES]

**Repair.**
- Label seed 3 as a "seed-robustness check".
- Freeze the configuration, then confirm on a fresh partition or population registered before evaluation.
- The tabular test splits are **spent** (`data_exposure_rows.csv`).

**Verification.** Any new evaluation set has an empty intersection with the recorded test indices.

### F10. The erase-layer pilot "54/60 → 60/60" uses a different architecture, a different baseline and a different checkpoint, and it destroys purpose conditioning. Severity: high for any reuse. [PE-A10, PE-C05, PE-D07]

**Evidence.**
- One LEACE eraser is fit on the **union of all purposes' disallowed attributes** and placed in the shared
  trunk (`erase-layer-pilot-2026-05-17:pcrl/training/v2_trainer.py:520-604`). For Adult the union includes
  `income`, the income_prediction task label.
- Counts come from best.pt (Cotter fallback), and the baseline is the best.pt 54/60, not 56/60.
- Utility falls: Adult occupation 0.992→0.753, education 0.9995→0.834; HMDA loan_amount_band 0.498→0.377;
  Diabetes medication_change 0.999→0.764.
- Clean compliance goes from 5/60 to 0/60.
- Source data: local untracked `results/rebuttal/erase_layer_pilot_aws/` (provenance consistent with branch
  code). [SRC/RES]

**Repair.** Present it as "single-union LEACE + per-purpose heads", a different method. Report utility.

**Verification.** Recount on the same checkpoint (done for counts in `recounts.json` c1_erase_pilot).

### F11. Smaller provenance items. Severity: low–medium. [PE-C09, PE-C10, PE-C12, PE-C13]

- **INLP/LAFTR/PCRL (0.0136/0.0382/0.2702; 56, 43 and 15 of 60) reproduces.** However:
  - LAFTR per-seed HMDA/Diabetes metrics and the merge script are not committed;
  - "27 cells" means training cells, while the percentages are over 60 audit rows;
  - INLP and LAFTR **train** their encoder, whereas PCRL's backbone is random, so "the same backbone anchors all
    three" (PDF(23) l.730) holds for the architecture only [PE-A12].
- **Rank-8 18/18 [0.0043, 0.0087] reproduces.** Its "published Round 7" comparator is actually the
  erase-pilot rank-24 run (published R7 is 17/18). medication_change accuracy is 0.75 vs 0.999.
- **App. P "5/20" is 20 pairs at one seed** against a pre-v2 PCRL. LEACE-on-raw 0/20 exists only as markdown.
- **LAFTR-hard Adult 0/24 COLLAPSED is unverifiable locally.** Only a 1-seed, 5-epoch smoke run is committed.
  The full result is reportedly in the no-expiry S3 archive, which was not read this phase.

---

## C. Dominant-axis / worst-class auditing and thresholds (Q2, Q3)

### F12. "Dominant axis" is the maximum single-class one-vs-rest R², not the worst linear leakage. It misses class contrasts. Severity: high. [PE-B03, PE-B01]

**Definition.** R²_DA = max_k R²(H, 1[A=k]), where each term is ridge OLS fit and scored in-sample on the
test split (`origin/main:pcrl/evaluation/certificates.py:47-116`; PDF(23) l.199, 272-273). There is no search
over contrasts c⊤z.

**Fixture.** K=10 balanced classes with the signal on a grouping of classes (`worst_class_fixture.py` §B):
- one-hot R² = 0.045 and R²_DA = 0.046, both pass;
- the top squared canonical correlation (the leading generalised eigenvalue, i.e. the max over contrasts) is
  ρ₁² = 0.407;
- the binary grouping is recovered at 79.6% vs 50%.

In the noiseless limit, a "classes 1–5 vs 6–10" signal has every per-class R² equal to 0.111 while the contrast
R² is 1.

**Repair.**
- Report ρ₁² (the CCA/generalised-eigenvalue criterion) alongside one-hot and max-class.
- Call R²_DA "max single-class leakage".
- Use held-out probes on declared groupings.

**Verification.** The fixture reproduces the gap (done). Computing ρ₁² for all 33 multi-class cells needs
relocated representations.

### F13. Proposition 5 (the convex-combination identity) is correct but tautological. The "33/33 verified" check is a code-consistency check with too loose a tolerance. Severity: medium. [PE-B02]

**When it holds.** R²_onehot = Σ_k w_k R²_OvR,k with w_k = π_k(1−π_k)/Σ_j π_j(1−π_j). This is exact for the
pooled (variance-weighted) R² over all K columns, with a shared ridge and unclamped per-class scores, in-sample
or on a fixed scoring set (fixture residual about 1e-16).

**When it fails.**
- per-class clamping at 0 (residual 0.0695 in the fixture);
- sklearn `uniform_average`;
- K−1 encodings.

**The stored check.** The maximum residual of 0.0021 sits on Adult employment/age_group s0, where R²_onehot
is 0.0054, so the residual is 39% of the value. It points to two different solve paths:
float32 in `verification.py:77-102` vs float64 in `certificates.py:84-97`. The 0.01 tolerance
(`eval_round4_dominant_axis.py:272-273`) exceeds most audited values.

**Repair.** Present the identity as a definition with weights from the scored rows' priors. Compute both scores
from one float64 solve. Drop the word "verified".

**Verification.** A single-solve recomputation gives residual < 1e-10 on all 33 cells.

### F14. The "worst case 12.7×" amplification is a ratio of two sub-threshold values. The argmax class is "Joint" applications, not a race. Severity: medium. [PE-B03, PE-C04]

- HMDA underwriting/race s0: R²_onehot 0.0026, R²_DA 0.0332, both below τ.
- Class 4 = "Joint" (`prepare_hmda.py:109-118`), π_test = 0.0092.
- Only s1 crosses τ (0.0268 vs 0.288).
- The median of 1.4328 over 33 cells reproduces. [RES]

**Repair.** Report amplification only where R²_DA ≥ τ, or report absolute gaps with CIs. Name the class.

### F15. An absent class is scored R² = 1.0 in training per-class constraints. That artifact produces the Diabetes "λ≈420" used to justify rank 24. Severity: high. [PE-B05]

**Evidence.**
- `losses.py:562-565` (float32) scores a minibatch with no members of class k as R² = 1. Missing *top* classes
  are padded (`v2_trainer.py:859-867`).
- Diabetes age_bucket class 0 ([0-10)) has prior 0.00215 after patient deduplication. It is absent from about
  57.6% of B = 256 batches.
- Absence alone implies λ ≈ 430. The stored λ_k0 is 420.2/419.4/420.2 (R7) and 421–423 (R6, including R6 s1,
  which otherwise sat at the floor). The audit R²_k0 is 0.000.
- PDF(23) App. L l.663-666 attributes the saturation to classes 5–7 and uses it to justify rank 16→24.
- All R7 seeds have `cotter_selection.kind = "fallback"` and 0 feasible epochs. [RES + fixture]

**Repair.**
- Merge the fixed-schema NaN/skip semantics, which exist only on
  `research/pcrl-submission-finish-v1:tests/test_scoring_support.py:15-27, 62-74` and are unmerged to
  origin/main.
- Disclose the artifact.
- Re-derive the rank choice; that needs retraining in a future phase.

**Verification.** The post-fix unit tests pass. A replay of the λ trajectories, if history is stored, shows
class 0 rising only on absent batches.

### F16. Threshold provenance. Severity: medium. [PE-B01, PE-D10]

Chronology:
- τ = 0.05 and Δaud < 0.02 were fixed on 2026-03-25 (a96ee0e10), before the V2 results.
- The health thresholds (per_dim_std ≥ 0.5, eff_rank ≥ 2) were fixed on 2026-04-27 (53feba0f2), before R4/R5.
- The K≥6 per-class gate was added on 2026-04-30 (f542c064c), after R5. It was chosen to exclude Adult and
  HMDA, which leaves the 5-class HMDA race attribute, where the paper's own 0.288 case occurs, on the averaged
  path.
- Table 5's τ sweep re-thresholds τ = 0.05-trained models; it does not retrain. It reproduces exactly.

Problems:
- Nothing was pre-registered.
- per_dim_std ≥ 0.5 is scale-dependent.
- The paper justifies τ through the refuted Prop 3: "≈22pp for a binary balanced attribute" (PDF(23) l.202-205).

**Repair.**
- State the chronology.
- Treat τ as a convention with a sweep, and drop the Prop-3 rationale.
- Use a scale-free health metric.
- Apply per-class constraints uniformly, or register the gate in advance.

### F17. Handling of missing and constant classes at evaluation. Severity: low for published numbers, medium for the released code. [WORST_CLASS_AUDIT §5]

Before the fix, the evaluation code would score a missing middle class as R² = 1 and drop a missing top class.
This was not triggered in the stored R5/R7 audits: no zero priors and no per-class score ≥ 0.999. The post-fix
NaN/coverage semantics exist only on research branches.

---

## D. Utility protocol (Q8)

### F18. "Task accuracy within 1pp of the unconstrained backbone" is contradicted by the repository's own comparison file. Severity: critical. [PE-A05; recounts.json s1_within_1pp]

**Evidence.** `origin/main:results/v2_adult_ROUND5/task_acc_vs_unconstrained.json` (sha256 0b748c5a…62b0,
committed 135e440e6 on 2026-05-05, before the PDF) shows `within_1pp` = **2/7** comparable tasks.
- Gaps: Adult income −6.22pp, HMDA loan_decision −1.44, loan_amount_band −11.56, tract_denial_high −2.44,
  Diabetes primary_diagnosis −12.85.
- The two "passes" are uninformative:
  - readmission_outcome sits at about majority for both models;
  - medication_change is a deterministic recoding of the inputs.
- `results/laftr_benchmark/PCRL_R5_COLLAPSE_DIAGNOSIS.md` §Q4-Q5, at the same commit, already says the claim
  "does not hold".
- The comparator is a separately trained single-seed StandardEncoder with val early stopping, **not** a probe
  on PCRL's frozen random backbone.
- The PCRL accuracies come from best.pt, while compliance comes from final.pt (F6).
- The claim appears at PDF(23) l.55 (contributions paragraph); a commented-out alternative abstract in source (21) abstract.tex also carries it. [RES]

**Repair.** Withdraw the claim. For each task, report test accuracy from **one** checkpoint against all of:
- (a) majority;
- (b) a matched-seed unconstrained model;
- (c) a linear probe on the same frozen backbone.

The protocol should be common across methods, with heads refit on a common held-out split.

**Verification.** The recount gives 2/7 (done). A new table carries a checkpoint hash per cell.

### F19. Several tasks are deterministic recodings of inputs or carry leaky labels, and protected attributes are model inputs. Severity: medium (interpretation of both utility and erasure). [PE-A09, PE-E06]

**Deterministic tasks.**
- Adult occupation_group and education_level are recoded from the one-hot occupation and education inputs
  (`adult.py:108-117, 354-362`).
- Diabetes medication_change is `change == "Ch"` (`preprocess_diabetes.py:237`).

**Leaky label.** HMDA `tract_denial_high` uses denial outcomes of all rows, including val, test and the row
itself (`prepare_hmda.py:33-37, 395-408`).

**Protected attributes as inputs.** HMDA race, ethnicity and sex one-hots are inputs; on `train.npz` they agree
100% with the labels. The same holds for Adult race, sex, marital status and age, and for Diabetes age_bucket.
Erasure therefore removes an explicitly supplied input.

**Repair.** Drop or redefine these tasks. Compute tract aggregates on train only, leaving the row out. State
the input inclusion explicitly.

### F20. The manuscript's purpose registries and dataset descriptions do not match the code. The HMDA motivating conflict is not instantiated. Severity: high for framing. [PE-A09, PE-D11, PE-E06]

| Item | PDF(23) | Code |
|---|---|---|
| Cohort and HMDA tasks | | |
| Diabetes cohort | "101,766 encounters" (l.194, 578) | first encounter per patient, 71,506 rows (50,053/10,725/10,728) |
| HMDA pricing task | "interest rate" (App. H l.575) | `loan_amount_band` |
| HMDA fair_lending task | "denial reason" (App. H l.576) | `tract_denial_high` |
| Disallowed sets and class counts | | |
| HMDA underwriting disallowed set | {race, ethnicity(3), sex} | {race(5), ethnicity(2)} |
| HMDA fair_lending race | the auditor "must see race" (l.19-23, 88-92, 191-193) | race is **disallowed for every HMDA purpose** |
| Adult marital_status | 7 classes | binary |
| Diabetes quality_research task | "A1C result" | `readmission_outcome` |
| Table 2 task class counts | Adult occupation m=4, education m=5 | 6 and 4 |
| Cross-purpose text (l.325-326) | marital_status "allowed under employment_analysis" | disallowed there |

Sources: `pcrl/data/hmda.py:119-150`, `adult.py:447-485`, `diabetes.py:97-125`,
`preprocess_diabetes.py:220-222`. [SRC; counts from local processed data]

**Repair.** Regenerate App. H and Table 2 from the code. Restate the conflict example honestly, or add an HMDA
purpose that allows race.

---

## E. Cross-purpose / coalition analysis (Q5)

### F21. What the concatenation attack fed. Severity: medium (protocol). [PE-D01, PE-D02]

The attack, `run_cross_purpose_attack_v2.py:82-98, 167-218, 338-348, 388-426, 494`, was set up as follows:
- **Representations.** It always concatenates all three purposes' 64-d h_p (192-d), including purposes where
  the audited attribute is **allowed**.
- **Attributes.** The audited set is the union of the disallowed attributes.
- **Checkpoints.** R5/R7 final.pt.
- **Auditor protocol.**
  - Auditors (LR, sklearn MLP 256-256, XGB 100×d6) are fit on the encoder's own train split and scored on
    test.
  - Each cell takes the maximum test accuracy over auditor seeds {11, 22, 33}.
  - The majority baseline is computed on test.
- **Incremental comparator.** The same auditor family is fit on each single h_p, and the best of the three
  is taken. Nested singleton predictors are therefore **included**, using 9 test-selected draws against 3 for
  the concatenation.
- **Recount** (`recount_cross_purpose.py`): 26/33 absolute, 22/33 incremental, 25/33 best-single−majority.
  All match.

**Repair.**
- Use a three-way split: encoder-fit, attacker-train and attacker-test.
- Select attacker seeds on attacker-validation data.
- Restrict the "coalition gain" comparator to views each recipient is permitted.

### F22. The dominant leak is single-recipient nonlinear recovery, not composition. Severity: high. [PE-D01, PE-C06]

**Evidence.**
- In 17/33 cells, the representation of a purpose that **disallows** the attribute already beats majority by
  more than 1pp. For HMDA race, which is disallowed in every purpose, a single h_p gives MLP +26.6pp and
  XGB +28.2pp.
- This contradicts PDF(23) l.324-332 ("whenever an attribute is allowed under some purpose … recovers it
  through that purpose's representation"; "a guarantee against single-consumer access").

**Repair.** Make per-(purpose, disallowed attribute) nonlinear single-recipient audits primary, and report the
coalition gain second.

### F23. The rebuttal "22/33 → 8/33 unified protocol" is a different model plus a headline-criterion choice made after seeing results. It is not a measurement repair. Severity: high. [PE-D07, PE-D08, PE-C06]

**What changed** (d39211214, branch `cross-purpose-rebuttal-2026-05-18`, unmerged): the 8/33 comes from a
different model.
- It is the erase-layer union-LEACE architecture with a training-time concatenation constraint, 200 epochs,
  and best.pt.
- Diabetes best_epoch is 0, i.e. the LEACE initialisation, and n_feasible is 0/200 on every seed.

**Under the absolute criterion:**
- the count goes 26/33 → 19/33, matching local untracked `results/rebuttal/cross_purpose/comparison.json`
  (sha256 94edf36c…4812) and `attack_matrix.md`, whose provenance is consistent;
- absolute concatenation leakage **rose** on 20 of 22 MLP/XGB cells (mean +19.4 → +24.2pp; e.g. Diabetes
  age_bucket MLP +33.9 → +59.5pp);
- LR cells drop to exactly 0 because of the union erase.

**What did not change.** The criterion code and the auditor definitions are identical
(`run_eval_multi.py:208-236` vs `run_cross_purpose_attack_v2.py:167-215`).

**The headline choice.** PAPER_PASTE moves 19/33 to an appendix with the note "do NOT use as headline …
invites a reviewer to notice the protocol swap".

**Measurement caveats.**
- Concat and single-purpose accuracies come from two separate runs.
- `load_encoder` uses `strict=False` and forces `use_erase_layer=True`. [UNV]

**Repair.**
- Report both criteria for both models, together with per-recipient leakage.
- Label the criterion choice as post hoc.
- State that the retrain removes the union of disallowed attributes from every purpose.

**Verification.** The standalone recount reproduces 22→8 (incremental) and 26→19 (absolute) (done).

### F24. Other cross-purpose items. Severity: medium. [PE-D03, PE-D04, PE-D05, PE-D06]

- **Table 10 caption.** It says "gain over majority", but the values are concat − best single (`gain_mean_pp`).
  The absolute values are larger, e.g. age_group XGB +20.0 vs 9.28.
- **l.324-326.** The sentence mixes absolute (+38.2) and incremental (+9.28/+11.96) values.
- **Adult concat-constraint pilot (19/24, 12/15).** Both figures are confirmed, but the run used 75 epochs,
  canonical_iterate.pt, and constraints on {race, sex, age_group} only.
- **Prop 6 check.** The rows labelled "Adult Round-5" load **variance-constrained retrain** checkpoints
  (`verify_bound.py:150-160`). All moments come from the same test sample, so "the bound holds on every cell"
  is guaranteed by algebra rather than shown by evidence. Premise (C1) also fails on several rows.

---

## F. Accuracy-guarantee retirement (Q6)

### F25. origin/main still ships the refuted R²→accuracy guarantee. The retirement branch is unmerged. Severity: high (the public repo distributes a false guarantee). [PE-D09]

**What still ships.**
- `origin/main:pcrl/purposes/verification.py:203-354`: "Theorem: Linear Compliance Guarantee",
  `certified_accuracy_bound` = min(max(π + √(ε·k·π(1−π)), π), 1). It is exported in
  `pcrl/purposes/__init__.py:21, 47`.
- `pcrl/evaluation/certificates.py:23-25, 466-472, 526, 572, 589-613`:
  - `NonlinearComplianceCertificate` and randomized smoothing are built on the bound;
  - `nonlinear_bound` is stored in every `generate_report`, which `run_v2_dataset.py:64, 365` calls;
  - reports print "theoretical accuracy bound from the Linear Compliance Guarantee".
- `experiments/deployment_case_study.py:81, 255-271`.
- `tests/test_accuracy_bound.py` asserts the bound.
- README l.4 says "compliance certificate".

**What does not depend on it.** Pass/fail counts do not use the bound (`certificates.py:534`). The stored
`nonlinear_bound` fields are nevertheless invalid.

**The retirement branch.** `fix/retire-accuracy-guarantee` (5d4eda046) is contained only in itself and its
origin twin; its merge-base is 55e4cb1d1. It has defects of its own:
- the doc cites `tests/test_accuracy_bound.py`, which the same commit deletes;
- the validation after `raise` is unreachable.

`research/pcrl-submission-finish-v1` carries the same doc plus a fuller regression test, also unmerged.

**Formula mismatch.** The paper's Prop 3 is π + k·√(επ(1−π)); the code is π + √(εkπ(1−π)).

### F26. The 20-observation counterexample and the flawed proof step. Severity: high (theory). [PE-D09, PE-D10]

**Counterexample** (`accuracy_bound_counterexample.py`, reproduced). Take (A, h) = (1, 1)×9, (1, −9)×1,
(0, −1)×9, (0, 9)×1.
- E[h | A] = 0 for both classes, so Cov(h, A) = 0 and the affine LS R² = 0.
- The threshold rule "A=1 iff h > 0" scores 18/20 = 90%, against a 50% majority.
- The old bound (paper and code) is 0.50.

A K=3 analogue gives one-hot R² = 0 and argmax-affine accuracy 0.9 against a 1/3 majority.

**Flawed step.** PDF(23) App. A step (3), l.458-459, bounds cov(1{ŷ=j}, z_j) using the **linear** OvR R².
But 1{argmax_i(w_iᵀh+b_i) = j} is a nonlinear, piecewise-constant function of h.

**Claims that rest on it.**
- the τ rationale (l.202-205);
- "R² 0.034 on sex ⇒ near 67% majority" (l.102-103);
- the ΔDP–R² link (l.122-125);
- "regulator-facing certificate" (l.469-471);
- checklist items 3 and 10 (l.824-825, 878-880).

**Correction proposal (text only, not applied).**

*Code on origin/main:*
1. Merge the retirement, preferring the `research/pcrl-submission-finish-v1` regression test. Fix the stale
   doc reference and drop the unreachable validation. Keep `certified_accuracy_bound` only as a shim that
   raises `NotImplementedError("retired: invalid")`.
2. `generate_report` stops constructing `NonlinearComplianceCertificate`. It writes `nonlinear_bound=None` and
   `accuracy_guarantee_status="retired_invalid"`, and removes the "Bound"/"NL Bound" print columns.
3. `deployment_case_study.py` is marked legacy and fails loudly.
4. README: "evaluated with an empirical affine least-squares leakage score (one-hot R² ≤ 0.05 on the scoring
   sample); not an accuracy, independence, or nonlinear guarantee". Replace "certificate/certified" with
   "linear-score pass".
5. Add a results README note that pre-retirement `nonlinear_bound`/`accuracy_bound` fields are invalid.

*Manuscript errata:*
- withdraw Prop 3 and App. A;
- restate τ as a convention with a sweep;
- delete the 67% sentence;
- state the valid narrower fact: Cov(h, A) = 0 implies only that the best affine least-squares predictor of
  one-hot A is constant on that distribution or sample.

**Regression fixtures (description).**
- **F1 (binary, exact rationals).** The 20-row table above.
  - Expected: cov == 0 exactly in Fraction arithmetic; affine lstsq R² == 0 (abs 1e-14); accuracy of 1{h>0}
    == 9/10; majority == 1/2.
  - The old `certified_accuracy_bound(0.0, 0.5, 2)` returned 0.5 < 0.9. The corrected API must raise,
    matching "retired".
  - `LinearComplianceCertificate(epsilon=0.01).check` may return certified=True, but the report must label it
    an empirical least-squares pass only.
- **F2 (K=3, 30 rows).** Class-conditional means are 0. Expected one-hot R² == 0, argmax-affine accuracy 0.9
  vs majority 1/3, and the corrected API raises.
- **F3 (report schema).** `generate_report` on an identity encoder over F1 gives `nonlinear_bound is None` and
  `accuracy_guarantee_status == "retired_invalid"`, with no bound column.
  `git grep -n "certified_accuracy_bound(" -- pcrl experiments` returns only the shim.

---

## G. CelebA (Q7)

### F27. CelebA compliance is in-sample on the eraser's own fit set, and the held-out-style numbers fail τ. Severity: high. [PE-E01, PE-E02, PE-A11]

**The "train-set R² ≤ 0.005"** (PDF(23) l.338): ridge OLS fit and scored on the same 60K partition-0 subsample
that LEACE was fit on.
- Values are 0.0029–0.0036, which is zero by construction.
- The generating script is in no git ref.

**Per-epoch val R².** In-sample OLS on a 4,096-image partition-1 subsample (d = 512) gives 0.156–0.171 on every
seed and every epoch.
- The chance floor is about d/n = 0.125; adjusted R² is 0.035–0.052.
- The manuscript's "O(d/n_fit)" explanation is wrong: d/n_fit = 0.0085.
- No train-fit → held-out linear probe was ever computed. [RES]

**Repair.** Report a cross-fitted held-out probe on an identity-disjoint split (F30), with CIs.

### F28. The PCRL-V constraint machinery is inert, and the model is effectively LEACE + linear probe. Severity: high (method claim). [PE-E05]

**Evidence.**
- After the frozen erase layer comes `task_proj` (W = I + BA) and then a Linear head, so measured linear R²
  cannot change.
- Val R² moves by less than 1e-3 over 25 epochs while the duals reach about 220, and 0 epochs are feasible
  (`backbone.py:29-80`, `training_log.json`).
- The backbone is a frozen ImageNet ResNet-18 (IMAGENET1K_V1), not fine-tuned on CelebA.
- There is a single purpose and no concatenation, so App. N's "cross-purpose attack" is a nonlinear attack on
  one representation (Male R² 0.56–0.88 after training).

**Repair.** Describe PCRL-V honestly. Drop the proxy-Lagrangian/VICReg contribution claim for vision.

### F29. App. N's "trained ≤ LEACE-init without exception" is false, and "worst-case" labels the best seed. Severity: medium-high. [PE-E04, PE-C08]

- Per-attacker deltas include seed 2 Male DeepMLP +7.20pp, seed 0 Male MLP +0.41pp and LR +0.01pp, and seed 1
  Young LR +0.01pp.
- Table 12 compares the maximum over attackers on each side, and its "worst" row (−6.3/−8.2) is the most
  favourable seed. The true worst seed is −2.8 Male and −6.8 Young.
- "60K validation partition" is impossible, because the official val partition has 19,867 images. The attacker
  fit/score split is unknown, since the script is missing.
- Smiling 0.752 ± 0.010 is final-epoch accuracy on the same 4,096 monitoring subset. There is no
  test-partition number and no unconstrained comparator (E-F6). [RES]

### F30. CelebA reproducibility and an identity-disjoint split proposal. Severity: medium. [PE-E05]

**What can and cannot be reproduced.**
- All stored-table numbers reproduce from tracked JSONs.
- The model cannot be reconstructed bitwise:
  - the relocated `final.pt` files are 42 KB and **omit the erase layer**;
  - refitting LEACE needs a ResNet forward pass over 60K images, and that refit is non-deterministic because of
    random flips;
  - the evaluation scripts are missing.
- `identity_CelebA.txt` is unused and absent. Identity overlap between the fit, attack and monitoring sets was
  never controlled.

**Proposal (text only): `celeba_iddisjoint_v1`.** It is explicitly distinct from the historical splits:
- H1: the partition-0 60K Male×Young stratified subsamples, seeds 0–2;
- H2: the partition-1 4,096 subsamples, seeds s and s+7;
- H3: the `head()` prefixes of the 64px pipelines.

Construction:
- **Unit.** identity_id.
- **Assignment.** u = int(sha256("celeba_iddisjoint_v1|" + identity_id)[:8], 16)/2³². The bands are
  eraser-fit < 0.40, probe-train 0.40–0.60, probe/attack-test 0.60–0.80 and utility-test ≥ 0.80.
- **Scoring pool.** Partition-2 identities only, excluding identities touched by the H3 prefixes.

Checks:
- identity intersections between groups are empty;
- no scoring image appears in H1, H2 or H3;
- an image-list sha256 manifest is committed before any evaluation;
- per-group marginals are reported;
- the builder never reads attribute labels.

**Input needed.** `identity_CelebA.txt`, which is a download and so is out of scope this phase.

---

## H. Data exposure (Q9)

### F31. Exposure status by dataset. Severity: high for future claims. [`data_exposure_rows.csv`]

**Spent:**
- Adult, HMDA and Diabetes test splits: selection, tuning and every reported metric.
- CelebA partition 0 (60K subsamples) and the 4,096 partition-1 subsets.
- BIOS dev split.

**Untouched by the medium pipeline:** CelebA partition 2, except for the file-order prefixes used by the 64px
pipelines.

**Never loaded by stored code:** the BIOS test split.

**Cross-lineage exposure (EARLY FLAG).** The Folktables ACS 2018 CA test split (80/10/10, RandomState(42),
N = 310,901) was used in the encoder-lineage v2 Folktables Rounds 1–2. This predates the ACS release studies.
The ACS SaTML'27 manuscript was withdrawn on 2026-09-26.

---

## Verification status summary

| Finding | Status |
|---|---|
| F1, F2, F4, F20, F21, F25 | [SRC] verified at ref:line |
| F6–F11, F14, F18, F22–F24, F27, F29 | [RES] recounted from stored files with standalone scripts (sha256 in recounts.json) |
| F3, F15 | [SRC] + synthetic fixture. Checkpoint-level confirmation [NR] needs the relocated checkpoints |
| F12, F13, F26 | synthetic fixtures run this phase |
| F11 LAFTR-hard, F27/F29 generating scripts, R5/R7 EC2 argv | [UNV] |
