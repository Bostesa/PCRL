# Access contracts and attacker classification — durable-guarantees ("Outputs Leak What They Use")

Role: METHODOLOGY (durable). Phase: analysis-first, $0 compute. Status legend: **SI** = source inspected,
**SR** = stored results checked, **NR** = not independently rerun.

Source pins: repo `Bostesa/durable-guarantees@956f5c883f515646aa457db55ecbd74b913768b2` (read-only clone,
paths below are relative to its root, shown as `dg:path:line`). Manuscript = `paper.tex` / `appendix.tex`
from the Aug 30 zip (byte-identical to the Sep 17 zip; see SUMMARY.md for the submitted-version argument),
cited as `paper.tex:line`, `appendix.tex:line`.

## 1. What every attacker is actually fed (Q1)

All attackers share one data object per (cell, model seed): a row matrix over the **PCRL train partition only**
(`dg:utils/pcrl_io.py:150,188` — `split="train"`; no test/val split is ever loaded), on which the encoder,
the head, any learned projection/subspace, and the attacker splits all live. Attacker split = stratified
75/25 on the protected attribute with `random_state = probe seed` (`dg:experiments/diagnostic.py:133-136`,
`dg:utils/battery.py:50-57`). The encoder was trained on **all** rows, including every attacker-test row
(`dg:experiments/mi_ceiling.py:240-262`, `dg:experiments/end_to_end_surgical.py:142-175`). Noise channels
draw **one fresh Gaussian draw per row** at export (`mi_ceiling.py:256`, `end_to_end_surgical.py:164-165`);
that single draw is used for both attacker-train and attacker-test rows. Post-hoc channels draw one fresh
draw over all rows per probe seed (`dg:experiments/targeted_noise.py:95-105`, `dg:utils/battery.py:106-118`).

| Attacker (code) | Train-time input | Test-time input | Mechanism knowledge | Queries per target | Classification | Evidence |
|---|---|---|---|---|---|---|
| XGBoost / MLP(256-256) / rank-32 ReLU-LoRA ("Tier 1") | released (noised / erased) representation of 75% of rows **with their protected labels** | released representation of the held-out 25% (one draw each) | none (black box) | 1 | **standard recipient trained on labelled defended examples** (adaptive in the Elazar-Goldberg sense; not mechanism-aware) | `diagnostic.py:139-202`; `two_tier_certification.py:117-128` |
| Gaussian LRT ("Tier 2", `gaussian_lrt`) | **pre-noise** representation h of the 75% train rows + their s labels; exact noise covariance σ²I or σ²QQᵀ | single **noised** release of each held-out row; it never sees the target's clean h | full: noise law, σ, Q (subspace channels) | 1 | **white-box / pipeline-access defense-aware attacker** (needs encoder or vendor-side clean outputs on a labelled population sample). NOT a repeated-query attacker; NOT a target-insider | `dg:utils/battery.py:60-103`; `two_tier_certification.py:124,147,337` |
| Gaussian LRT on deterministic baselines (LAFTR, DANN-scrub, LEACE, VFAE-mean, Obliviator, FARE, FNF, Fair PCA) | the released rep itself (h_clean := P, Σ_noise = 0) | released rep | none beyond Tier 1 | 1 | **collapses to a QDA probe on the release = Tier-1 access** | `dg:experiments/baseline_gauntlet.py:52-56,329-333`; `fare_stage_b.py` docstring; `run_fairpca_gauntlet.py:1-10` |
| Gaussian LRT on VFAE sampled-z | posterior means μ of train rows (encoder internals) + diag(mean σ²) | one sampled z per held-out row | moment-matched approximation of VFAE's own noise | 1 | **white-box (encoder-internal μ)** — stronger access than VFAE's claim covers | `baseline_gauntlet.py:56-58,345-352` |
| Gaussian LRT on audit projections (expansion) | **pre-erasure** clean P of train rows, Σ_noise = 0 | erased H_er = P − PQQᵀ of held-out rows | knows the pre-erasure distribution (misspecified model) | 1 | insider side-information but a mis-specified (weaker) test; irrelevant to verdicts (projections already fail Tier 1) | `dg:experiments/expansion_pipeline.py:168` |
| Knows-Q attacker | release projected onto span(Q)^⊥ (recovers clean h⊥ exactly) | same projection of the target release | the learned basis Q (a model parameter) | 1 | **parameter-knowledge (white-box Q) attacker** | `dg:experiments/run_knows_q.py:67-77` |
| Averaging attacker | mean of N fresh draws per row (N ∈ {1,2,4,8,16}), labelled | mean of N fresh draws per target | none | N ≤ 16 | **repeated-query attacker** (black box) | `dg:experiments/averaging_attack.py:110-129`; middle/hard: `dg:results/tpr_capture.json` key `averaging` (post hoc, `docs/tpr_results.md:12-13`) |
| FNF "optimal adversary" (Lemma 5.1 of FNF) | exact log-densities under both trained flows | release | white-box flows | 1 | white-box, but within FNF's own claimed adversary class | `fnf_stage_b.py:1-12` |
| Prop-2 two-sheet witness | an oracle that knows the constructed sheet centres | synthetic | constructed distribution, not the data | — | **not an attack on any deployed system** (existence argument) | `appendix.tex:127-168`; `mi_ceiling.py` |
| Output-surface battery | own head's **logits / log-probas** (continuous score vector) on 75% rows | same on 25% | none | 1 | standard recipient of soft scores (more than the hard decision ŷ the paper describes) | `two_tier_certification.py` (not run for "ours"), `baseline_gauntlet.py:401-402`, `expansion_pipeline.py:100-130` |

### Query arithmetic (averaging)
Averaging N draws of h + σ·ε is distributionally h + (σ/√N)·ε (`docs/averaging_attack_prediction.md:31-34`).
To bring σ_eff to the BatchNorm unit scale (σ_eff ≤ 1): easy/middle/hard subspace σ = 16 → N = 256;
full-rank Tier-2 points σ = 20 / 24 / 64 → N = 400 / 576 / 4096; to σ_eff ≤ 0.25, multiply by 16.
**Maximum N used anywhere = 16** (subspace channel only; σ_eff = 4). No averaging attack was run on any
full-rank point or on any baseline. The clean-limit is therefore never approached experimentally; the
"converges to clean h, which the LRT reads at 0.84" statement (`averaging_attack_prediction.md:55-57`) is
an extrapolation.

### Where descriptions overstate or misstate access
1. README "Tier 2 ... has seen the representation before noise was added. This is the vendor, an auditor
   with pipeline access, or anyone who can query the same row repeatedly and average the answers"
   (`dg:README.md:62-66`) and the identical docstring (`dg:utils/battery.py:15-21`,
   `two_tier_certification.py:9-10`): the code's LRT never averages repeated queries and never sees the
   target's pre-noise vector; the repeated-query threat is a separate experiment (N ≤ 16, one cell + post-hoc
   two). Manuscript `paper.tex:251-254` ("knows the clean representation and the noise we added") is also
   ambiguous: the clean representation is known only for a labelled *training population*, not for the
   target. Because P|s = (h|s) ⊛ N(0,Σ_noise), the LRT's "side information" is, in the large-sample limit,
   the same information Tier 1 already has from labelled noised rows; its advantage is a correct parametric
   model of the noise, not additional information about the target. Tier 2 is therefore best described as
   *"mechanism-aware with white-box encoder access on a labelled population sample, single release"*.
2. Paper Setup says model parameters are not exposed (`paper.tex:203-205`); the LRT needs either the encoder
   or vendor-side clean vectors, so Tier 2 already steps outside the stated exposure model — the paper's
   Kerckhoffs framing is implicit, not stated.

### Stronger-access attacks used against claims that never assumed that access
- **Subspace channel vs knows-Q / LRT**: own method; the paper scopes this correctly (`paper.tex:565-575`,
  Table 1 footnote 5). No over-reach.
- **Baselines vs Tier 2**: for every deterministic baseline Tier 2 adds no access (QDA on the release), so
  Tier-2 ✗ marks carry no extra-access charge. The only exception is **VFAE sampled-z** (LRT gets encoder
  μ); Table 1 marks VFAE Tier 2 as "≈" (`paper.tex:618`). Low severity, but the Tier-2 cell for VFAE should be
  labelled white-box.
- **Tier 1 vs LEACE / Obliviator / LAFTR**: nonlinear probes exceed LEACE's (linear) and Obliviator's
  (its own probe) claimed scope; the paper's "Claimed scope" column and caption (`paper.tex:615-637`)
  disclose this. The framing "fail" is about the stronger test, not a broken claim.
- **FARE multiclass**: the paper says FARE's certificate "reads 0.000 on the multiclass cell while three of
  those configurations leak race pairs at 0.603–0.610" (`paper.tex:692-694`). FARE's dp_ub bounds downstream
  demographic parity, not attribute recovery (`docs/fare_results.md` claim 4, which states "not a
  contradiction"); the 0.603–0.610 are macro-OvR Tier-1 readings of non-certified sweep points. Using a
  property FARE never certifies to call it "silent" is an out-of-scope comparison; the paper's wording
  "honest in its own scope" partly concedes this.
- **LAFTR official on 5-class race**: the official binary-A code is run on a 5-class attribute it cannot
  represent (`dg:experiments/laftr_official.py:29-40`); its failure there is a misuse, not evidence against
  LAFTR's claim. Table 1 merges the official and re-implementation rows (`paper.tex:617`).

## 2. Access contract per method (Q2)

"Public state" = what the paper's exposure model releases; "Tier-2 extra" = what the LRT additionally gets.
Utility head and output head are the same object unless stated. All fits use all rows of the cell (PCRL
train partition, or 60k-capped external cell) unless stated.

| Method (row) | Provenance (code) | Uses s at train | Uses s at inference | Public release (rep surface) | Output surface attacked | Utility head / convention | Selection budget | Tier-2 extra | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| LAFTR official | VectorInstitute/laftr @a166ba3 via TF1 shim | yes (adversary) | **yes** (`use_attr=true` concatenates s onto x) | Z (zdim 8) for all rows | their Y_hat sigmoid → 2-col logits | max(own head in-sample, LR held-out 25%) / clean e2e in-sample lift | γ ∈ {0.5,1,2,4,8}, 1 sweep seed, ≤2 certified candidates × 3 seeds | none (QDA) | `laftr_official.py:1-45,144-193` |
| LAFTR re-impl (5-class adversary) | own | yes | s in x (PCRL features) | Z (64-d) | own head logits | same max(own,LR) | same γ grid | none | `baseline_gauntlet.py:16-21,362-380` |
| VFAE | own re-impl (no official code) | yes (q(z|x,s), MMD) | yes (q(z|x,s)) | sampled z **or** mean μ (two exposures per β) | own q(y|z) logits | max(own,LR) | β ∈ {1,10,100,1000} × 2 exposures | **encoder μ + diag σ²** (white-box) | `baseline_gauntlet.py:22-27,52-58,345-352`; `vfae_calibration.py` |
| DANN-scrub (generic GRL adversary) | own; not Jaiswal "adversarial forgetting" | yes | s in x | Z (64-d) | own head | max(own,LR) | λ ∈ {1,2,5,10,20} | none | `baseline_gauntlet.py:28-37` |
| LEACE | concept-erasure lib `LeaceEraser` | yes (fit on all rows incl. attacker test rows) | s in x | eraser(clean P) | LR head fit and scored on all rows (in-sample log-probas) | max(LR in-sample, LR held-out) | none (parameter-free), 3 P-training seeds | none | `baseline_gauntlet.py:38-41,309-323` |
| Obliviator | ramin-akbari/Obliviator @0f2233f (official, supervised mode) | yes + task label y | s in x | z_t at iterations {0,1,2,3,5,7,10,15,stop} | LR head on z_t | max(own LR in-sample, LR held-out) | 9 iteration indices | none | `obliviator_gauntlet.py:1-60,199-263` |
| FARE | eth-sri/fare @89cb1b6 (official, gate reproduced) | yes | s in x | leaf-median embedding z | LR head retrained on z | max(own,LR) | k ∈ {5,10,25,50,100} × α ∈ {0.5,0.75,0.9,0.975,0.999} = 25 | none | `fare_stage_a.py:1-10`; `docs/fare_results.md` |
| FNF | official (inapplicable on multiclass cells) | yes | **yes** (encoder chosen by the row's own s) | flow release z | LR head on z | max(own,LR) | γ grid (5 settings) | FNF optimal adversary (white-box flows, within FNF's claim) | `fnf_stage_a.py:21-22,141`; `fnf_stage_b.py:1-12` |
| Fair PCA | amazon-science/fair-pca (official) | yes | s in x | projection Z | LR head on Z | max(own,LR) | sweep incl. certified point | none | `run_fairpca_gauntlet.py:1-10,176-269` |
| MMD / HSIC targeted projections (audit) | own (QR-parametrised, Adam 500 steps) | yes | s in x | H_er = H − HQQᵀ (r ∈ {8,16,32}) | **not measured** in the audit | held-out LR (task_accuracy) | r grid | LRT on pre-erasure P (misspecified) | `smart_erasure.py`; `expansion_pipeline.py:150-186`; `paper.tex:278-308` |
| Full-rank noise ("ours, full-rank") | own | **no** (task CE only) | s in x | P = BN(E(x)) + σε, one draw per row | own linear head logits D(P) — **not attacked in `two_tier_certification`**; reported separately as the floor | own head **in-sample** / clean σ=0 (no-BN) in-sample lift, 5 seeds | σ grid 6/5/5 values per cell, escalation along grid | pre-noise h of train rows + σ²I | `mi_ceiling.py:209-262`; `two_tier_certification.py:117-231` |
| Isolate-then-noise ("ours, subspace", e2e) | own | **yes** (HSIC(h⊥, onehot s), λ ∈ {0,10,100}) | s in x | P = h + σ εQᵀ | own head logits (attacked in TPR/worst-pair runs, not in two-tier) | own head in-sample (paper headline uses fresh-partition 64/97/97) | λ pilot {0,10,100}; k × σ grid; σ doubling to 64 | h of train rows + σ²QQᵀ; Q | `end_to_end_surgical.py:120-175`; `appendix.tex:481-498` |
| Isolate-then-noise post-hoc | own | yes (HSIC projection fit on all rows) | s in x | frozen P + σ_rel·σ_h noise in span(Q) | LR head (lr_logits) | **held-out LR** (`utility_through`) | (rank, σ_rel) grid | h + σ²QQᵀ | `two_tier_certification.py:237-314`; `targeted_noise.py:108-120` |
| Clipped full-rank DP channel (Prop 3) | own | no | s in x | Π_C(h) + σε | own head | own head in-sample only; **no attacker run** | σ ∈ {8,32} × nominal ε ∈ {0.5,1,3,6} | n/a | `run_dp_fullrank.py:1-25,72-160` |

**Is output-side leakage measured from the deployed task head?** For e2e channels and LAFTR/VFAE/DANN, yes:
the head trained jointly with the encoder, evaluated on the same rows it was trained on, attacked through its
**full logit vector** (not ŷ). For LEACE, Obliviator, FARE, FNF, Fair PCA and post-hoc channels, the "output"
is a logistic-regression head the auditors fit on the release (all rows) — not a deployed head shipped by the
method. For the 67-configuration audit, the output surface is **not measured** at all
(`expansion_pipeline.py:150-186`, `honest_reaudit.py:62-80`), although the manuscript says the audit asks
about "either exposed surface" (`paper.tex:306-308`). For the gauntlet, the verdict is **representation-gated**
(`baseline_gauntlet.py:452`, `master_gauntlet_table.py:166` "Verdict on the representation at ≤0.55");
the output reading is reported but does not gate, contrary to the pass rule stated at `paper.tex:261-262`.
For "ours", `two_tier_certification.py` never attacks outputs; the paper discloses this for its own method
(`paper.tex:266-269`).

## 3. Proposed standardized contract for a revised benchmark
1. Name each tier by access, not by role: T1 = labelled released examples, 1 release/target;
   T1-Q(N) = T1 with N fresh releases per target (N on a grid up to σ²-matched clean limit);
   T2-MA = mechanism-aware (noise law, Q, public parameters) with labelled released examples;
   T2-WB = white-box encoder / clean vectors on a labelled population sample; Insider = target's clean vector.
2. One gating rule for every method: pass iff every attacker in the tier ≤ bar on **both** surfaces, with the
   output surface = the deployed head's released object (state whether ŷ or scores).
3. Fit encoders, erasers, heads, and attackers on disjoint partitions (see FINDINGS F-03).
