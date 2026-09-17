# Novelty matrix: verified relation to the closest prior work

Terminal B, 2026-09-17. This extends and corrects
`results/redesign_20260917_acs_spectral_transport_v1/PRIOR_WORK_AND_NOVELTY.md`, which is a careful
document; the corrections here are two bibliographic errors, one sharpened mathematical
identification, and one misattribution that matters for the proposed follow-up.

**Verification status.** Rows marked *full text* were read beyond the abstract in this audit and the
quoted theorem or equation was located. Rows marked *abstract* were verified only at abstract or
proceedings-page level and their internal claims are labelled accordingly. Nothing below is taken
from a search snippet.

## 1. The matrix

| Work | Conditioned on | Encoder / channel | Protected target | Utility target | Optimiser | Rank rule | Guarantee scope | Preserves a fixed prior output? | Multiple recipients / coalition? | Verified |
|---|---|---|---|---|---|---|---|---|---|---|
| **SARL** (Sadeghi, Yu & Boddeti, ICCV 2019) | nothing (unconditional) | linear, then kernel | any vector attribute, via linear LS adversary | linear LS target prediction | closed form, eigenvectors of `B` | **Thm 3**: the γ = min{r, j} negative-eigenvalue directions, j = #negative eigenvalues | global optimum of the fixed matrix; adversary restricted to linear/kernel LS | no | no | full text |
| **OptNet-ARL** (Sadeghi, Wang & Boddeti, ECML PKDD 2021) | nothing | deep encoder, closed-form ridge players | multiple sensitive attributes, multiple λ | multiple targets | encoder by SGD through closed-form best responses | **Thm 4.1**: optimal dimension = #negative eigenvalues of `λS̃ᵀS̃ − (1−λ)ỸᵀỸ` | ridge best responses | no | no | full text |
| **K-TOpt** (Sadeghi, Dehdashtian & Boddeti, TMLR 2022) | nothing (explicitly unconditional) | RKHS encoder, random Fourier features | attribute via covariance-operator dependence (HSIC-like) | label dependence | closed form, generalised eigenproblem (**Thm 4**, r largest) | **Cor 4.1**: optimal dimension = #non-negative eigenvalues | near-optimal under an approximate-normality assumption on low-dimensional projections | no | no | full text |
| **U-FaTE** (Dehdashtian, Sadeghi & Boddeti, CVPR 2024) | **a discrete label stratum** `Y = y` (**Eq. 3**, `X_c ~ P(X\|Y=y)`); the text says this equals `Dep(f(X),S\|Y=y)` "when Y is not a continuous label" | RKHS encoder | sensitive attribute (age, on Folktables) | target label (employment) | closed form, generalised eigenvectors (**Thm 1**, *r largest*) | **r = c − 1**, from the number of target classes; **not** eigenvalue-sign based | as K-TOpt | no | no | full text |
| **LEACE** (Belrose et al., NeurIPS 2023) | nothing | affine edit of an owned embedding | one concept | minimal embedding change, broad norm class | closed form | n/a | **provably prevents all linear classifiers**; explicitly silent on nonlinear ones | no — it *edits* the representation | no | abstract |
| **SPLINCE** (Holstege, Ravfogel & Wouters, NeurIPS 2025) | nothing | oblique projection on an owned embedding | one linear concept direction | exact covariance with a target label | closed form (unique solution) | n/a | removes linear predictability, preserves target covariance | no | no | abstract |
| **KCI** (Zhang et al., UAI 2011) | `Z` (a test, not a mechanism) | n/a | n/a | n/a | kernel test statistic, asymptotic null | n/a | **Lemma 2** (Daudin 1980): `X ⊥ Y \| Z` ⟺ (v) `E(f̃g′)=0 ∀ f̃ ∈ E_XZ, g′ ∈ L²_Y` | n/a | n/a | full text |
| **RCIT / RCoT** (Strobl et al., JCI 2019) | `C` | n/a | n/a | n/a | **Eq. 26**: `S′ = n‖Σ̂_{AB·C}‖²_F`, ridge residualisation on random features, Lindsay–Pilla–Basak null | n/a | approximate KCIT | n/a | n/a | full text |
| **Multi-consumer IT privacy** (Sankar, Rajagopalan & Poor, TIFS 2013) | side information `Z^n` held by users | one sanitised database, rate-distortion encoder | identity/attributes | distortion-measured utility | asymptotic, known joint distribution | n/a | utility–privacy tradeoff regions | **no** | **"multiple legitimate information consumers" consume one common release**; no per-recipient release, no collusion, no complementary second release | full text |
| **TAPPFL** (Arevalo et al., AAAI 2024) | nothing | learned FL representation | sensitive attribute | task-agnostic information about the device data | mutual-information objectives, variational | n/a | claims guarantees against worst-case attribute inference | no | no | abstract |
| **Least-privilege limits** (Stadler et al., ICML 2024) | n/a | any feature mapping | any non-task attribute | intended task label | n/a | n/a | a **fundamental trade-off**: high task utility and prevention of all non-task inference are not simultaneously achievable, regardless of technique | no | no | abstract |
| **PCRL residual spectral channel** (this work) | **continuous released service probabilities** `H_A` (and `H_AB` for the coalition role), via cross-fitted logistic nuisances and a degree-2 basis | fixed random-feature map, whitened residual, linear projection | SEX (2), RAC1P (9), and nine further forbidden roles | least-squares reconstruction of the **residualised teacher** | closed form (Ky Fan), fixed r = 16 | **fixed r = 16**, deliberately width-matched to the neural baselines | global optimum of the fixed matrix only | **yes — bitwise, structurally** | **yes — separate local and coalition penalties; only recipient A receives a channel** | own code and fixtures |

## 2. Corrections to the existing prior-work document

| # | Original | Verified | Consequence |
|---|---|---|---|
| N1 | "Sadeghi, Wang & Boddeti (2021) … **Theorem 4**" | The result is **Theorem 4.1** | citation fix |
| N2 | "Our conditional moments are … one case of **Lemma 2(iii)/(v)**" | It is exactly **Lemma 2(v)**. Centring on the S side is algebraically equivalent to residualising the (Z,H) side: with `f̃ = f − E[f\|H]`, `E[f(Z,H)(1{S=c} − m_c(H))] = E[f̃·1{S=c}]`, because `E[f̃·m_c(H)] = 0` and `E[E[f\|H](1{S=c}−m_c(H))] = 0`. So the penalty is Lemma 2(v) restricted to `f ∈ {Z_l b_k(H)}` and `g′ ∈ {1{S=c}}` | sharper and correct identification |
| N3 | U-FaTE is implied to be the source of eigenvalue-sign rank selection (via `RESEARCH_DECISION.md`'s "select rank by the sign of the eigenvalues … adapt the U-FaTE-style measure") | U-FaTE Thm 1 takes the **r largest** eigenvalues and sets **r = c − 1**. Sign-based rank selection is SARL Thm 3, OptNet-ARL Thm 4.1 and K-TOpt Cor 4.1 | the follow-up's rank half is **already published**, three times; only the conditional-penalty half is an adaptation |
| N4 | U-FaTE described as the one evaluated on Folktables/ACS | **K-TOpt also evaluates on Folktables** (Washington and New York), alongside CelebA | ACS evaluation is not itself a distinguishing feature |
| N5 | "multiple-consumer information-theoretic privacy" cited for the multi-recipient setting | Sankar et al. model **one** sanitised release consumed by many users with private side information. No per-recipient release, no collusion, no complementary release, no preserved prior output; strictly asymptotic with a known joint distribution | the multi-recipient prior work is *weaker* prior art than the citation suggests — this strengthens the problem-formulation contribution |

## 3. Which components are not new

Stated plainly so the manuscript cannot be read as claiming them:

* **The spectral trace step.** Ky Fan (1949), applied exactly as SARL and K-TOpt apply it.
* **Random Fourier features inside a closed-form invariant-representation solver.** K-TOpt.
* **Multiple protected attributes combined by weighted penalties.** OptNet-ARL, explicitly.
* **Rank selection by eigenvalue sign.** SARL Thm 3, OptNet-ARL Thm 4.1, K-TOpt Cor 4.1.
* **A conditional dependence penalty inside such a solver.** U-FaTE, conditioned on a discrete label.
* **Conditional-independence moments of the form used here.** KCI Lemma 2(v) / RCoT Eq. 26.
* **Cross-fitting the nuisance.** Standard (Chernozhukov et al.).
* **The marginal arms M025/M1.** Up to parametrisation, SARL's linear construction on whitened
  random-feature residuals with a continuous vector target.
* **The observation that erasure certified against linear probes fails against nonlinear attackers.**
  LEACE scopes its guarantee to linear adversaries; Elazar & Goldberg showed the phenomenon for
  adversarial removal in 2018.
* **The existence of a task-utility / non-task-leakage floor.** Stadler et al. (2024).
* **An eigensolver, a reconstruction objective, or the word "coalition".** None of these is a
  contribution by itself.

## 4. Where a genuine difference remains, and how strong it is

| Candidate difference | Status | Why |
|---|---|---|
| **Conditioning on continuous released *predictions* rather than on a label stratum** | **adaptation, and a real one** | U-FaTE's construction is stratification over discrete `Y`; the paper itself scopes the equivalence to "when Y is not a continuous label". `H_A` is a pair of continuous probability vectors, so stratification is unavailable and a nuisance-residual formulation is required. Separately, `H` is a *released artifact*, not the label to be predicted — a different object even setting continuity aside. Not a new optimisation principle. |
| **Exact preservation of an already-published output, as a structural constraint** | **not found in the surveyed prior work** | LEACE and SPLINCE edit a representation the releaser owns; SARL/K-TOpt/U-FaTE design a representation from scratch; Sankar et al. design one sanitised release from scratch. None constrains a *previously published* vector to be reproduced bitwise while adding capability beside it. |
| **Recipient-specific and coalition-specific penalties, with only one recipient receiving a channel** | **not found in the surveyed prior work** | Verified negative: Sankar et al.'s "multiple consumers" share one release; OptNet-ARL's multiple λ are multiple attributes, not multiple recipients; no surveyed paper penalises the *joint view* of two colluding recipients. Eight targeted searches returned nothing on complementary/incremental releases with colluding recipients. |
| **Utility as reconstruction of a residualised teacher** | **adaptation** | A design choice, not a principle. It is what makes "beyond what H already explains" operational. |
| **The evaluation design** (fresh vs frozen attackers kept separate, absolute vs incremental recovery reported separately, unclipped negative increments, routed-H control, locked temporal transport) | **the strongest candidate contribution** | Not a method claim. See [`CONTRIBUTION_ASSESSMENT.md`](CONTRIBUTION_ASSESSMENT.md). |

## 5. Limits of this search

A bounded search cannot establish novelty and nothing here claims a first. "Not found" means these
searches did not surface it. Rows marked *abstract* have not been read in full; in particular, the
formal statements in Stadler et al. and TAPPFL are **unverified** at theorem level and are cited only
for the claims their abstracts make. U-FaTE's supplementary material, which the paper says carries
the per-criterion solutions, was not read.
