# Analysis positioning: what the attribute-removal analysis adds, and what it does not

Written 2026-10-01; literature current to that date.

- **Literature sources:** PRIOR_ANALYSIS_MATRIX.csv has 39 rows: 8 reviewer-cited or mandatory primary
  texts read in full, and 31 papers from a systematic search with forward citations of 27 starting papers.
- **Repository sources:** the role FINDINGS files cited by key.
- **Section 6 is a recommendation, not a fact.**

The working question under assessment: *when do attribute-removal methods' stated protection tests
agree with sensitive-attribute recovery by defense-aware recipients, under consistent utility, including
task outputs and combinations of permitted releases?*

---

## 1. Already established observations (prior work; not ours to claim)

| Observation | Established by (location) |
|---|---|
| Linear erasure/certificates do not bound nonlinear recovery; RBF-SVM/MLP recover > 90% after R-LACE; fresh adversaries reach 99% | Ravfogel et al. ICML 2022 §5.1, App. B.1, Table 2 fn 9 |
| Nonlinear (kernel) erasure does not transfer to other attacker families (MLP 0.97) | Ravfogel et al. EMNLP 2022 |
| Training-time adversaries are fooled while post-hoc attackers recover the attribute | Elazar & Goldberg 2018; Barrett et al. 2019 (LB19); Gonen & Goldberg 2019 (LB20); FNF (LB12) |
| Standard-scaling or stronger attackers break adversarial fair representations; benchmarks with fixed downstream models exist | FCRL AAAI 2021 (LB13); Cerrato et al. (LB09); Reddy et al. (LB10); Pouget et al. (LB11) |
| Probing-based removal is unreliable | Kumar et al. (LB22) |
| Recoverability ≠ causal use | Elazar et al. TACL 2021 (amnesic probing) |
| Representations trained for one task reveal unrelated sensitive attributes (overlearning), and censoring can be undone (de-censoring) | Song & Shmatikov ICLR 2020 |
| Universal least privilege is incompatible with utility; label-only "fundamental leakage" is measurable before training | Stadler et al. ICML 2024 Thm 2, §4.1/Fig. 3 |
| **Task outputs leak more than the labels alone (model leakage − dataset leakage at matched accuracy)** | **Wang et al. ICCV 2019 (LB01)**; Hirota et al. (LB02) |
| Task predictions downstream of an erased representation can leak the attribute (small for honest binary heads; full for adversarial multiclass heads) | Ravfogel, Goldberg & Cotterell ACL 2023 (LB03) |
| Fairness constraints and predictions are exploitable side information for attribute inference; labels-only vs labels+predictions adversaries | Ferry et al. SaTML 2023 (LB05); Aalmoes et al. (LB04) |
| Attribute inference must be compared with an imputation baseline | Jayaraman & Evans (LB07) |
| Projection erasure fitted on the released rows can be inverted by anti-clustering | Johansson (LB25) |
| Stated fair-representation certificates are checked against held-out downstream classifiers (incl. one predicting S) | FARE (LB14); FRG NeurIPS 2025 (LB15) |
| Certification requires smoothing/noise; deterministic injective encoders cannot be certified from finite samples | Gitiaux & Rangwala AISTATS 2021 Thm 2.1 |
| Averaging repeated noise draws undoes isolated-subspace noise | FairNVT v2 Table 3 (TMLR 2026) |
| Combining a biased and a fair version of a model raises attribute inference | Tian et al. (LB06) |
| Colluding adversaries in ML pipelines are systematised | Duddu et al. SoK (LB32) |
| Sequential releases under collusion with finite alphabets have formal budgets | Taylor, Vippathalla & Coon 2026 |

## 2. Observations our existing evidence reproduces

These are reproductions and should be presented as such.

- **durable-guarantees: linear certificates pass, XGB/MLP recover (18 approved → 15 collapse).**
  - This reproduces §1 rows 1, 3 and 4.
  - The README count "21" is wrong; the correct count is 18 (DG-F24 R01; IC-F8).
- **durable-guarantees: published methods fail a common battery.**
  - This reproduces the FCRL/Cerrato/Reddy-type findings.
  - It has extra caveats: verdicts are gated on the representation only, utility is in-sample, and the
    access labels are wrong (DG-F01/F03/F05).
- **durable-guarantees: output leakage tracks AUC(s|y).**
  - The *measurement* is Stadler's fundamental leakage and Wang et al.'s dataset leakage.
  - What is not a reproduction is the cross-method *cost* relation (r = 0.80, n = 27). See §3.
- **durable-guarantees: averaging breaks the subspace noise channel at N = 16.**
  - This reproduces FairNVT's averaging result on a closely related design.
- **PCRL encoder: R² ≤ τ representations leak to nonlinear auditors.**
  - This reproduces §1 row 1.
  - The concatenation attack mostly measures *single-recipient* recovery (17/33 cells, PE-F22).
- **PCRL ACS release studies: coalition slates with nested references.**
  - These are methodologically sound (PA-F18).
  - They find small positive recovery beyond H on 8/8 endpoints (PA-F1).

## 3. Plausible additional empirical questions (hedged)

None of these is established as novel. Each is a place where the systematic search did not find a single
study doing the combination.

1. **A controlled agreement measurement** between each method's *own stated test* and defense-aware
   recovery, under the conditions below. FRG and FARE check *certificates* against held-out
   classifiers, but for fairness bounds and within their own method families.
   - One access contract (A1/A2, with A3/A4 as labelled stress tests).
   - One held-out utility contract.
   - Methods from four families: closed-form linear, iterative linear, stochastic noise, and
     certificate-based.

   The addition would be cross-family agreement rates at matched held-out utility. This is a
   measurement contribution, and it is modest.
2. **A cross-method cost relation.** Does label coupling AUC(s|y) predict the *utility cost* of reaching
   a given A2 protection level, across methods and cells?
   - Wang 2019 and Stadler measure the leakage floor, not the cost of removal.
   - The existing r = 0.80 comes from in-sample utility, a small n with 6 bootstrap clusters, and cells
     chosen after earlier results.
   - It is **unknown whether it survives held-out utility**. This is the most specific candidate
     contribution, and the most fragile.
3. **Combination of purpose-specific learned releases under a declared sensitive set**, with
   jointly-disallowed targets and nested references.
   - The search found only analogues: Tian et al. on model versions, the SoK, and Taylor et al.'s
     finite-alphabet theory.
   - **But our own evidence weakens the motivation.** In the encoder lineage the dominant leak is
     single-recipient. In the ACS lineage the combination increments are small, sit near a null-token
     floor of [−0.00036, +0.00077], and come from spent data.
   - This should be a secondary scenario, run only where the policy table creates jointly-disallowed
     targets. It is not the headline.
4. **The multiclass worst-case diagnostic.** Top canonical correlation vs max-per-class, with
   not-estimable handling.
   - This is a measurement repair (METHODOLOGY_IMPROVEMENTS §4), not a contribution on its own.
   - Ravfogel et al. 2023 (LB03) already shows multiclass heads can recover what binary probes miss.

## 4. Gaps requiring repaired evaluation before any of §3 can be claimed

(Details in METHODOLOGY_IMPROVEMENTS.md.)

1. Access-tagged attackers. The current "Tier 2" is A4 white-box population access, not
   insider/repeated-query (§1 there).
2. Held-out utility with one head rule and stable denominators (§2).
3. A gated output surface with label-only and clean-output references (§3).
4. Contrast-sensitive categorical diagnostic and unsupported-class handling (§4).
5. Registered thresholds (§5).
6. Nested slates and paired, refit-aware uncertainty (§6).
7. A declared sensitive set and coherent coalition targets (§7).
8. Official, pinned method implementations, and an S-in-X regime declared per arm (§8).
9. **Erasers fitted on rows disjoint from the released and attacked rows** (Johansson LB25; DG-F10).
10. Fresh evaluation data chosen blind (§11).

## 5. Claims that should be removed or rewritten

These are wording changes; they need no new experiments.

- **AAAI: "We contribute its measurement" of label-only leakage** (`paper.tex:171-175`). Remove.
  Stadler §4.1 and Wang 2019 already measure it. Cite both.
- **AAAI: "leakage floor established by Stadler"** (`paper.tex:474`). Restate with the theorem's actual
  scope: all attributes, finite discrete spaces, positive posterior, conditioning on true Y.
- **AAAI: Tier 2 = "knows the clean representation / anyone who can query the same row repeatedly"**
  (`paper.tex:251-254`; README). Replace with the A4 description.
- **AAAI: isolate-then-noise beats full-rank at matched protection (1.9/38.6%)** (`paper.tex:576-580`).
  Withdraw. With frontier matching the full-rank channel keeps about 55.5% and 56.6% (DG-F08).
- **AAAI: easy-cell supported-class 49.7%.** The registered 5-seed figure is 41.0%. Also: "no
  configuration can clear the null" quotes a max over 27 draws (DG-F13).
- **AAAI: Prop 2 main-text generalisation** ("no sound certificate ... can be useful"). Restrict it
  (DG-F16).
- **AAAI: Prop 3 covering Fig. 4 / full-rank points.** Those points are unclipped (DG-F17).
- **AAAI: the CelebA "frozen encoder we did not train".** It is PCRL's own `celeba_v2` network
  (DG-F19).
- **AAAI: Elazar et al. cited for the wrong point** (`paper.tex:156-158`). Its point is recoverability
  ≠ use. Separately, the title "Outputs Leak What They Use" implies use, but the paper measures
  recoverability.
- **PCRL: "pretrained backbone"** (PE-F1). Withdraw: the backbone is frozen and randomly initialised.
- **PCRL: "within 1pp of the unconstrained backbone"** (PE-F18). Withdraw.
- **PCRL: Dominant-axis as the "worst linear leakage"** (PE-F12) and the **"12.7× amplification"**
  headline (PE-F14). Withdraw or restate.
- **PCRL: "held-out seed 3"** (PE-F9). It is the same split.
- **PCRL: "cross-purpose attack confirms composition theorem".** Single-recipient leakage dominates
  (PE-F22).
- **PCRL: rebuttal 22/33 → 8/33 and 54/60 → 60/60 presented as PCRL improvements** (PE-F10/F23).
  These use a union eraser, a different model, and a post-hoc criterion.
- **PCRL: R²→accuracy guarantee** (shipped on public main). Retire it. A proposal exists and is not
  merged in this phase.
- **PCRL bibliography.** Fix the `ravfogel2022rlace` authors and the "R-LACE refines LEACE" sentence.
  Add Stadler, Wang 2019, Ravfogel 2023 and FairNVT.
- **ACS lineage** (withdrawn). Explain "passes all 8 sensitive clauses" as J-relative only (PA-F1).
  Correct "no earlier stage had touched 2016" (PA-F4). Correct the "capping added recovery" wording
  (PA-F2).

## 6. Recommendation: **narrow** the empirical analysis (and pilot before committing)

- **Do not continue it as currently framed.** Its central observations are already established (§1).
  Several of its distinctive numbers do not survive the audit: Tier 2 access, matched-utility
  comparison, in-sample utility, and stale counts.
- **Do not abandon it either.** Two questions remain plausibly open (§3.1, §3.2), and the repaired
  contract (§4) is itself useful for any future method comparison.
- **Proposed narrowing:**
  - **Primary question.** Cross-family *agreement* between each method's own stated test and
    A2-recipient recovery (representation and gated output surface), at matched held-out utility, under
    a declared finite sensitive set.
  - **Secondary question.** Whether label coupling predicts the held-out utility cost of reaching a
    fixed A2 protection level.
  - **Scenario, not headline.** The multi-recipient combination analysis runs only for
    jointly-disallowed attributes, with per-recipient LEACE as the baseline. Report honestly if Δ is
    small.
- **Reconsider with Dr. Yus** if the one-cell pilot (protocol §11) shows either of these:
  - (a) the cost relation disappears under held-out utility; or
  - (b) every method's own test already agrees with A2 recovery once utility is held out.

  In either case the remaining contribution would be methodological only. A smaller venue or a merged
  methods section could then be the better home; that is an author decision, not a fact.
- **Future algorithm work stays downstream.** Our own evidence does not yet demonstrate a gap that a
  joint multi-purpose method would close:
  - single-recipient leakage dominates;
  - combination increments are small;
  - per-recipient LEACE has not been compared under the repaired contract.
