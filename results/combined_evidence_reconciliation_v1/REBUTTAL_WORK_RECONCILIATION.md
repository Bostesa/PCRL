# Rebuttal and later work: what was completed, and what it changes

Reconciled 2026-10-01.

## Sources

| Source | Location / pin |
|---|---|
| PCRL | GitHub; all remote branches inventoried (51 refs) |
| durable-guarantees | @956f5c88 |
| External relocation drive | YOTUO, 2026-09-30 and 2026-09-25 runs |
| Supplied venue PDFs | NeurIPS 397d85d5…; AAAI 2d86afad… |
| Official review exports | NeurIPS: 3 human reviews; AAAI: 2 human reviews + 1 AI review. Private, used locally only. |

## Status words

Each number below carries one of these statuses (VERIFICATION_REPORT.json):

| Status | Meaning |
|---|---|
| **recomputed** | independently recomputed from stored per-seed or per-person files |
| **checked** | checked against code and recorded aggregates |
| **reported** | reported but not independently reproduced |
| **unsupported** | no supporting artifact located |

## Chronology

Read every row with its timing in mind. An experiment can bear on a review concern without having been
done in response to it.

- **NeurIPS**
  - The reviews were written Jun 22–26 and released Jul 24. The export contains no author response.
  - All May 2026 "rebuttal" branches **predate** the reviews: erase-layer pilot, VICReg sweep, rank-8,
    LAFTR hard-R², cross-purpose rebuttal and constraint, BIOS.
  - The only work aimed at the real reviews is the FAccT ablation registration of Jul 24. It was
    **registered and smoke-tested, never executed**.
- **AAAI**
  - Every durable-guarantees experiment commit is dated on or before Aug 2, which is before the human
    reviews (Aug 23–30).
  - The late-July robustness runs answered a *simulated* local review package, not the venue reviews.

---

## 1. PCRL encoder lineage (NeurIPS paper)

### 1a. Checkpoint rule and compliance counts

| Count | Final iterate (final.pt) | Best validation (best.pt) | As printed in paper |
|---|---|---|---|
| Strict R² ≤ 0.05 | **56/60** (recomputed) | **54/60** (recomputed) | 56/60 (the paper states the final-iterate rule) |
| Clean (strict + per_dim_std ≥ 0.5 + eff_rank ≥ 2) | **6/60** (recomputed) | **5/60** (recomputed) | **7/60** (mixed rule) |

- **Where the final-iterate health comes from.** It is the SPLINCE benchmark's `pre_health`, which loads
  final.pt in eval mode on the test split.
- **Why 7/60 is a mixed count.** It pairs final.pt R² with best.pt health. Its three Adult s0
  employment cells are clean under neither single rule.
- **Two other counts in circulation:**
  - Round 4's 46/60 (final) is 33/60 on best.pt.
  - A further 47/60 appears at commit a00d5749b.
- **Task accuracies.** The paper reports them from best.pt while R² is from final.pt (checked).

**What this changes.** 56/60 and 54/60 are both correct, under different rules. The earlier assessment
presented 54/60 as a "corrected" 56/60, and that is withdrawn. A revised paper must declare one rule and
report strict and clean counts from that same checkpoint.

### 1b. Cross-purpose concatenation

The two criteria:
- **ABS:** the concatenation beats majority by more than 1 pp.
- **INCR:** the concatenation beats the strongest single-purpose release by more than 1 pp.

Both are computed as a mean over 3 PCRL seeds, taking the maximum over auditor seeds.

| Model | Checkpoint | ABS | INCR | Status |
|---|---|---|---|---|
| Submitted PCRL (R5 Adult/HMDA, R7 Diabetes) | final.pt | 26/33 | 22/33 | recomputed |
| Rebuttal: erase-layer union-LEACE + h_concat R² constraint (τ_cross = 0.10), 200 ep | best.pt | 19/33 | 8/33 | recomputed (laptop-only files, now preserved) |
| LAFTR (submitted appendix) | — | 29/33 | 16/33 | checked (aggregate only) |

**Neither criterion was registered** (checked, from commit history):
- The submission switched its headline from INCR (22) to ABS (26) hours before the final build. A
  HEADLINE "integration decision" chose the criterion to match existing wording.
- The rebuttal switched back to INCR after its results came in.
- Table 10's caption says "gain over majority" (ABS), but the values printed are INCR gains.

**On the rebuttal model specifically:**
- All 11 linear-auditor cells are exactly 0. This is expected, because the union eraser removes every
  disallowed attribute linearly from every purpose.
- On 20 of the 22 nonlinear cells, **absolute** leakage *rose* (mean +19.4 → +24.2 pp).
- 25 of 33 cells already leak from a single release (best single − majority > 1 pp).

**What this changes.**
- The rebuttal genuinely reduced *concatenation increments*, from 22 to 8 incremental flags.
- It did this with a different access policy: one eraser for all purposes. That policy also removes
  attributes a purpose is allowed to use.
- It did not reduce absolute nonlinear recovery.
- Report both criteria side by side, labelled post hoc. They are different criteria, not contradictory
  measurements.

### 1c. Erase layer, VICReg and LoRA rank

**Erase-layer pilot** (recomputed; per-seed files were laptop-only and are now preserved in this branch):
- Strict compliance is 60/60 against the like-for-like best.pt baseline of 54/60. Clean compliance is
  0/60 against 5/60.
- Utility is test-split task accuracy (stored per task).
- Strict compliance is **structural**:
  - A frozen union-LEACE layer sits upstream of every trainable map.
  - Test R² is within 1.2–1.8× the in-sample OLS null floor.
  - Every Cotter selection was a fallback (0 feasible epochs).
- **Nonlinear recovery got worse.** This overturns the earlier note that the "auditor channel was
  unchanged":
  - The auditor delta rose on 48 of 60 paired cells (means: Adult 0.068 → 0.160, HMDA 0.128 → 0.298,
    Diabetes 0.007 → 0.059).
  - Cells with delta ≥ 0.02 went from 27/60 to 44/60.
  - Adjusted passes fell from 32/60 to 16/60.

**VICReg ×5** (recomputed):
- Adult and HMDA: 42/42 strict, 0/42 clean, 0/42 adjusted passes.
- per_dim_std moved +0.002 (Adult) and −0.014 (HMDA).
- The Diabetes arm was cut by the run's 10-hour time cap, and no per-seed file exists.

**The "0.5 per_dim_std floor is architectural" claim, narrowed:**
- What the diagnostic measured: per-dim std of a *reconstructed, frozen, untrained* random map, in eval
  mode, at 9 dataset×seed points (checked).
- The synthetic fixture shows:
  - R², held-out linear AUC and kNN AUC are invariant to isotropic rescaling.
  - per_dim_std scales linearly with the rescaling.
  - eff_rank survives isotropic but not anisotropic rescaling.
- Scaling the stored pilot representations by about 1.15–1.41 would turn 0/60 "clean" into 60/60 with no
  change in leakage.
- So this is an observation about this backbone's scale under an absolute threshold, not an
  impossibility. A scale-free health metric is needed.

**LoRA rank:**
- "Rank-8 matches rank-24, because LEACE acts in the 128-d backbone space" is true only *by construction*
  in the erase-layer architecture: frozen erase layer, LoRA only on repr_proj.
- It does not test the Round 7 rank-floor rationale for the submitted all-layer LoRA (checked).
- The rank-8 HEADLINE's "rank-24" column mixes erase-pilot metrics with Round 7 accuracies. Against the
  matched rank-24 erase pilot, medication_change drops 1.1 pp, not 24.7 pp (recomputed).

### 1d. Baselines and criterion matching

- **No baseline was trained against R² ≤ 0.05** (checked).
  - INLP stops when validation LR accuracy is within 1 pp of majority.
  - LAFTR optimises adversarial cross-entropy.
  - SPLINCE and LEACE are closed-form.
  - INLP and LAFTR train their encoders, whereas the PCRL backbone is frozen and random.
- **LAFTR hard-R²: completed, with full outputs on the drive only** (recomputed). Location:
  `wt-PCRL-superpowers-laftr-hard-r2-2026-05-17.tar::…/results/laftr_hard_r2_{adult,hmda,diabetes}_LAFTR_HARD_R2/`.
  - Strict: Adult 0/24, HMDA 17/18, Diabetes 18/18, i.e. **35/60**. Adjusted 18/60; clean 0/60.
  - All 9 seeds used Cotter fallback checkpoints; Adult was scored at epochs 11–17 of 200.
  - It is PCRL's stack plus an adversary, i.e. the only R²-trained comparator, but not LAFTR proper.
  - Stored-file defects:
    - The HMDA per-seed `pass_count` field reads 0.
    - PAPER_PASTE says PCRL R7 Diabetes passes "without collapse", which contradicts the paper.
  - Its comparison table copies the LAFTR-Q HMDA/Diabetes rows from the paper rather than measuring
    them.
- **Original LAFTR (Appendix Table 13; 15/60 strict):**
  - The Adult rows are recomputed (0/24).
  - The 36 HMDA and Diabetes rows are **unsupported**. They were joined from an S3 CSV in the 7-day
    lifecycle bucket, and no per-seed file exists on any branch or on the drive.
  - "Discriminator at chance 56/60" is unsupported.
- **INLP:** stored 43/60; a float64 recompute from drive representations gives 42/60. Its utility is
  0.815 vs PCRL 0.779 (best.pt) on the same 27 cells, and INLP is more than 1 pp higher on 14 of them.
  "Comparable utility" therefore overstates PCRL.
- **SPLINCE (60/60 strict → 3/60 clean):** it post-processes PCRL's own final.pt representations, one
  attribute per cell. It is a post-processing comparison, not an independent pipeline.
- **LEACE on raw features ("0/20"):**
  - The linear failures sit in near-singular directions (relative singular values 1e-7 to 1e-4). With a
    1e-3 cutoff, 20/20 pass.
  - The robust failure is nonlinear (Δ ≥ 2 pp on 17/20).
  - Appendix P says "reduces accuracy to majority on all datasets". This contradicts stored per-task JSON:
    accuracy is above majority on 8 of 9 tasks.
- **Implementations not as named:**
  - The in-house `LEACEEraser` is not LEACE. It is used only by legacy pre-v2 scripts; v2 and
    durable-guarantees use the official `concept_erasure`.
  - "R-LACE" is an INLP-style loop.

### 1e. Held-out selection, ablations and scale

- **Held-out validation exists only for Adult seed 3**, which is a new seed on the same split (6/8).
  There is no HMDA or Diabetes held-out run on any branch, in the drive inventories, or in the checkpoints.
  This reviewer concern is **unresolved**.
- **FAccT ablations 1–3** (linear vs LoRA, no erase layer, held-out hyperparameter selection) were
  registered on Jul 24 and exist only as smoke runs. They are **not executed**. They also ablate the
  later erase-layer variant, not the submitted architecture.
- **Larger backbones:**
  - BIOS BERT layer-12: 5 launches, R² 0.946 stable. This is a closed negative result, dropped from the
    paper.
  - CelebA PCRL-V: compliance is measured in-sample on the eraser's fit set, with validation R² 0.156–0.171.
    Only linear layers follow the erasure.
  - Runtime numbers exist (1873 s / 3861 s), but there is no scalability evidence.

### 1f. Guarantees

- The joint-LEACE Proposition 2 linear covariance guarantee holds on the fit law:
  - at initialisation in Round 5/7;
  - throughout training in the erase-layer pilot, which is why strict R² is 60/60 there.
- A nonlinear attacker does not refute it.
- The R²→accuracy bound is **invalid** (20-row counterexample: R² = 0, accuracy 0.9 vs bound 0.5).
  - It is still live on origin/main@55e4cb1.
  - `fix/retire-accuracy-guarantee`@5d4eda0 retires it by raising before the old return. It is
    unmerged, and was not merged here.
  - 27 research branches carry a retired version.

---

## 2. Attribute-removal lineage (AAAI paper)

**The audit's encoders are not the NeurIPS headline encoders** (recomputed hashes).
- The tabular cells load PCRL **Round 4 seed 0** `final.pt` (adult 1cfc2fef…, hmda e29d0367…, both at
  epoch 204). Their lambdas equal the Round 4 per-seed JSON.
- CelebA uses an April 16 `celeba_v2` CNN (b0df3fb7…), not PCRL-V. Its training config was not located.
- No PCRL commit is pinned.
- Consequence: these are related fits from the same lineage, **not** the same fits as the NeurIPS
  headline. Do not count them as one result, and do not present them as independent replications.

### 2a. The 59/67 audit and thresholds (recomputed from stored held-out probabilities)

- Recomputed failures: 64, 59 and 51 out of 67 at AUC bars 0.52, 0.55 and 0.60. The largest difference
  from stored values is 1.7e-5.
- Three failures lie within 0.004 of the bar.
- The 0.52/0.60 sensitivity the second AAAI human reviewer asked for **was already in the 5-page
  supplement** (64/59/51; published methods 0 vs 4 of 36).
- Still missing:
  - CI-based bars;
  - per-bar reselection of operating points;
  - the isolate-then-noise advantage at other bars (now recounted; see 2c).
- **Only the representation surface was scored** in the audit: 0 of 67 configurations were scored on
  outputs. The AI review's "surface change" premise is therefore wrong in fact. The paper invited it by
  writing "either exposed surface".
- What *does* change between the check and the audit is the metric (R² → AUC) and the attacker family.
- **Survivors:** "5 of 8 survivors leak at Tier 2" becomes 8, 5 and 0 at the three bars. Two of the
  five exceed the bar by at most 0.005.

### 2b. Multiclass scoring

- The one multiclass audit row (HMDA race, σ = 8):
  - macro OvR 0.551, a fail by 0.0011;
  - the paper's supported-pair rule gives 0.524–0.531, a pass, so 59 becomes 58;
  - worst-class or all-pairs gives 0.64–0.65, driven by class 4 with 126 held-out rows.
- **Table 1 at 0.55** confirms 7/42, with or without the output surface.
  - At 0.52: 0 certified passes; 4 FARE combinations are undetermined.
  - At 0.60: 8 certified passes and 2 single-seed rows.
  - The appendix's "4 of 36 at 0.60" counts single-seed rows as passes; the certified count is 2.
- **Worst-pair:**
  - VFAE on the easy cell, at Tier 1, fails every worst-case criterion (supported pair 0.566 vs null
    0.522). Its arrays *do* exist on the drive, although the dual report says "not_available".
  - The "Ours, full-rank" checkmarks fail supported-pair on both 5-class cells.
  - FARE's middle cell flips to fail under supported-pair scoring. This was done before the reviews:
    DG `results/baseline_dual_score.json`.
  - "Passes every attacker" therefore does not survive worst-pair scoring.

### 2c. Utility and isolate-then-noise

- **Utility conventions are mixed.**
  - Baselines take max(own head in-sample, a fresh LR head on rows the encoder saw).
  - The authors' arm uses its own head in-sample.
  - Out-of-partition utility exists only for 9 of the authors' points, from the fresh-partition run:
    0 verdict flips, max |Δ AUC| 0.013.
- **Rankings move.** Putting every row on the LR-held-out convention changes the orderings on the easy
  and hard cells. For example, LAFTR-official on the easy cell reads 102% on its own head vs 47% under LR.
  The middle cell is unchanged.
- **Isolate-then-noise vs full-rank noise:**
  - The frontier advantage at 0.55 is +4.4 / +44.1 / +87.9 pp (easy / middle / hard). The paper's
    nearest-T1 matched figures are 99.7 / 61.8 / 84.2.
  - Out of partition, at the paper's own selected points, the easy-cell advantage **reverses** to −6.8 pp
    (full-rank 70.8% vs subspace 64.0%). Middle is +39.4 and hard +99.8.
  - The advantage is real on the middle and hard cells, and fragile on the easy one.

### 2d. Threat model and guarantees

- **Tier 2** is a Gaussian class-conditional LRT. It is fit on clean, pre-noise vectors of labelled
  attacker-train rows, with the noise covariance (and Q) known, and it scores **one** release.
  - It has no target clean vector, no noise realisation, no repeated releases and no model weights.
  - No neural or white-box adaptive attacker exists. The second AAAI human reviewer's scope concern
    stands.
  - Averaging is a separate run: one point, N ≤ 16, breaching at N = 16.
  - The knows-Q attack is separate (0.98 on the easy cell).
- **Scope of "certificate failures":**
  - The linear R² check, LEACE and Fair PCA fail only against nonlinear attackers, which is outside
    their scope. None was ever tested with a held-out linear probe.
  - paper.tex:429-430 labels attacked readings (a ReLU LoRA probe) as "the certificate".
  - FNF's failure is outside its density-estimate precondition.
  - FARE is coherent on the binary cell; on the multiclass cell it is outside its binary scope.
  - Proposition 3 covers only the clipped channel. The Table 1 and Fig. 4 full-rank points are
    unclipped, so their Tier-2 passes are empirical, not certified.

---

## 3. Net effect

| Area | Genuine improvement found | Still open |
|---|---|---|
| Linear compliance | Erase layer makes strict linear compliance structural (60/60) | Nonlinear leakage rose; clean 0/60; utility loss on some tasks |
| Concatenation | Incremental flags 22 → 8 with a cross-purpose constraint | Absolute nonlinear leakage rose; one union eraser changes the access policy; both criteria post hoc |
| Criterion-matched baseline | LAFTR hard-R² ran in full (35/60, all collapsed) | Outputs on drive only; original LAFTR HMDA/Diabetes unsupported |
| Collapse diagnosis | VICReg tested; floor diagnostic run | Health threshold is scale-dependent; no scale-free metric; no fix |
| Held-out selection | Adult seed 3 only | No HMDA/Diabetes held-out run; FAccT ablations not executed |
| AAAI thresholds | 0.52/0.55/0.60 sensitivity already in supplement | CI-based bars; per-bar reselection |
| AAAI multiclass | Supported-pair rescoring of FARE existed | VFAE and others not rescored in the paper; arrays exist on drive |
| AAAI utility | Fresh-partition check (9 points) | No common held-out protocol; orderings change |
| AAAI Tier 2 | Averaging and knows-Q side studies | No general adaptive/white-box attacker; LRT mis-described |
