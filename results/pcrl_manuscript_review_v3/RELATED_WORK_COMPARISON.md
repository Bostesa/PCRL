# RELATED_WORK_COMPARISON — primary sources, and a feature table with its gaps marked

Scope rule for this file: **an entry is filled only from a text actually read.** Where a
claim could be checked only against an abstract page, the cell is marked
`UNVERIFIED-ABSTRACT`. Nothing is inferred from a title.

## 0. What was read, and how

| Source | Identifier | Verified from | Verified by |
|---|---|---|---|
| Madras, Creager, Pitassi, Zemel — *Learning Adversarially Fair and Transferable Representations*, ICML 2018 (PMLR v80, 3384–3393) | proceedings.mlr.press/v80/madras18a.html | landing page: title, authors, venue, abstract | this review |
| Belrose et al. — *LEACE: Perfect Linear Concept Erasure in Closed Form*, NeurIPS 2023 | arXiv:2306.03819 | abstract page (this review); **PDF body** (Terminal 1, `BASELINE_ADAPTATIONS.md` §3.1) | both |
| Holstege, Ravfogel, Wouters — *Preserving Task-Relevant Information Under Linear Concept Removal*, NeurIPS 2025 | arXiv:2506.10703 | abstract page (this review); **PDF body**, Thm 1 Eq. 4 and Thm 2 line by line (Terminal 1) | both |
| Sadeghi, Yu, Boddeti — *On the Global Optima of Kernelized Adversarial Representation Learning*, ICCV 2019 | arXiv:1910.07423 | abstract page (this review); **Appendix B eq. 24, Thm 3** (Terminal 1) | both |
| Sadeghi, Wang, Boddeti — *Adversarial Representation Learning with Closed-Form Solvers* (OptNet-ARL) | arXiv:2109.05535 | abstract page (this review); **Lemma 1, Eq. 3/11, Thm 4.1** (Terminal 1) | both |
| Dehdashtian, Sadeghi, Boddeti — *Utility-Fairness Trade-Offs and How to Find Them*, CVPR 2024 (U-FaTE) | arXiv:2404.09454 | abstract page + CVF listing | this review |
| Sadeghi, Dehdashtian, Boddeti — *On Characterizing the Trade-off in Invariant Representation Learning* (K-TOpt), TMLR 2022 | arXiv:2109.03386 | v2 audit | predecessor |
| Sankar, Rajagopalan, Poor — *Utility–Privacy Tradeoffs in Databases*, IEEE TIFS 2013 | arXiv:1102.3751 | abstract page only | this review |
| Stadler et al. — *The Fundamental Limits of Least-Privilege Learning*, ICML 2024 | PMLR v235, 46393–46411 | v2 audit | predecessor |
| Elazar & Goldberg — *Adversarial Removal of Demographic Attributes from Text Data*, EMNLP 2018 | aclanthology.org/D18-1002 | v2 audit | predecessor |

**Recorded hazard, carried from Terminal 1.** A first automated pass over LEACE returned a
**hallucinated** formula. It was discarded and every equation in the adaptation document
was taken from the PDF body. That is the reason this file records *where* each fact came
from rather than only *what* it says.

**Venue note.** OptNet-ARL's arXiv page states no venue; the repository README says ECML
2021 and dblp lists ECML PKDD 2021 pp. 731–748. The bibliography carries that with an
explicit note. This is the kind of entry that would otherwise be guessed.

---

## 1. The comparison table

Five features. `Y` = the text demonstrates it; `N` = the text's setting excludes it;
`ADAPT` = an adaptation supplies it but the source does not; `UNVERIFIED-ABSTRACT` = could
not be settled from what was read.

| Work | Fixed pre-existing output the method may not modify | Reusable extra features beside that output | Recipient-specific permissions | Coalition-aware training | Independent attack evaluation beyond the certified class |
|---|---|---|---|---|---|
| **This paper** | **Y** (bitwise, 210/210 views) | **Y** (16-d channel appended) | **Y** (A / B / AB roles, 11 forbidden) | **Y** (penalty from the AB joint view) | **Y** (5 attack families, fresh + frozen, never pooled) |
| Madras et al.\ 2018 (LAFTR) | N — representation learned from scratch | Y (transferable representation) | N — one representation, unknown third parties | N | Partial: adversarial objective + transfer tests |
| LEACE (Belrose et al.\ 2023) | N — edits the representation it owns | N — it transforms, it does not add | N | N | N — guarantee scoped to linear adversaries; authors conjecture nonlinear is intractable |
| SPLINCE (Holstege et al.\ 2025) | N | N | N | N | N — same kernel as LEACE, range differs |
| SARL (Sadeghi et al.\ 2019) | N | N | N | N | Partial — compares to iterative minimax |
| OptNet-ARL (Sadeghi et al.\ 2021) | N | N | N | N (multi-attribute asserted, not multi-recipient) | Partial |
| K-TOpt (Sadeghi et al.\ 2022) | N | N | N | N | Partial |
| U-FaTE (Dehdashtian et al.\ 2024) | N | N | N | N | Y — evaluates >1000 pre-trained models |
| Sankar et al.\ 2013 | N | `UNVERIFIED-ABSTRACT` | `UNVERIFIED-ABSTRACT` — "multiple legitimate information consumers" is in the abstract; whether they receive *different* releases is not settled by it | `UNVERIFIED-ABSTRACT` | N — information-theoretic, known joint distribution |
| Stadler et al.\ 2024 | N | N | N | N | N/A — impossibility result |
| Elazar & Goldberg 2018 | N | N | N | N | **Y** — the classic demonstration that adversarial removal leaves recoverable information |

**The one column that is ours.** No text we read *evaluated* a mechanism under the
constraint that a previously published output is immutable, with per-recipient and
coalition audit roles. Several of these methods can be **adapted** into that setting —
three were, in Study 3, and they beat our mechanism. The distinction between "can express
it", "supplies an algorithm for it", "evaluated it", and "demonstrates an effect beyond its
own controls" is maintained in the manuscript and answered separately.

**What we do not claim.** Not that the formulation is unprecedented; a bounded search
cannot establish that. Not that any listed method fails at this task — for most of them the
task was never attempted, and for LEACE the scoping is deliberate and stated by its authors.

---

## 2. Attribution ledger: what is reused, with the credit named

| Component in this work | Prior art | Status |
|---|---|---|
| Utility-minus-`λ`·dependence solved by a trace/eigenvector step | SARL (Sadeghi et al.\ 2019); the trace step is Ky Fan (1949) | **Reused.** The marginal arms rebuild SARL's construction bitwise |
| Eigenvalue-sign rank selection | SARL Thm 3; OptNet-ARL Thm 4.1; K-TOpt Cor 4.1 | **Reused**, and measured to be non-binding here (32 positive eigenvalues against `r = 16`) |
| Closed-form ridge players inside a learned encoder | OptNet-ARL | **Reused as a baseline**, with the multi-`λ` form declared an extrapolation |
| Conditional dependence penalty in this objective family on ACS/Folktables | U-FaTE | **Direct antecedent** |
| Squared Frobenius norm of a residualised cross-covariance | RCoT statistic (Strobl, Zhang & Visweswaran 2019) | **Reused as a penalty.** No null distribution or Type-I control inherited |
| Multiplying a residualised sensitive side by functions of `(Z,H)` | Daudin (1980) via KCI Lemma 2(v) (Zhang et al.\ 2011) | **Finite fitted subset** of the characterisation |
| Exact Gaussian kernels in place of random features | the standard kernel-vs-random-feature trade; Rahimi & Recht (2007) is what it replaces | **Reused** |
| Cross-fitted out-of-fold nuisances | standard practice; Chernozhukov et al.\ (2016) | **Reused.** No DML orthogonality or rate result inherited |
| Adversarially trained channel that transfers to new tasks | Madras et al.\ (2018) | **Prior art.** Not claimed as new |
| Linear concept erasure on the auxiliary channel | LEACE; SPLINCE | **Executed as adapted baselines** |

**Reusing a known solver, a conditional-moment characterisation or a rank rule is
attribution, not innovation**, and the manuscript says so in the body rather than in a
footnote.

---

## 3. Two corrections to the predecessor's baseline specification, and why they matter

Recorded because they change what a fair comparison *is*, not just how it is described.
Both originate in Terminal 1's `BASELINE_ADAPTATIONS.md` §1.1 and are adopted here.

1. **Do not apply LEACE/SPLINCE to the concatenated `[H_A, Z]` wire.** The v2 review's
   `CONTRIBUTION_ASSESSMENT.md` §6 specified exactly that. LEACE's `P*` is a single `d×d`
   oblique projection over **all** `d` coordinates and nothing constrains it to act as the
   identity on the first four. Applying it to the joint wire would generally change the
   published service output and break the structural guarantee that is this paper's first
   deliverable. The channel is therefore transformed alone and appended.
2. **SPLINCE must not preserve covariance with the residence label.** `same_residence` is
   the held-out task; preserving covariance with it during representation fitting would
   leak the held-out task into the representation and defeat the test. SPLINCE's
   preservation target here is the **authorised training tasks** `income_binary` and
   `civilian_at_work`. This is named an adaptation, and it remains an asymmetry in
   SPLINCE's favour on utility, since no other arm sees a task label during representation
   fitting.

## 4. A source hazard worth publishing

SARL's official code takes the `r` **algebraically smallest** eigenvectors with no sign
test, which its own Theorem 3 excludes. Measured here, ascending truncation selects a
subspace at projector Frobenius distance **5.657 of a maximum 5.657** — completely
orthogonal to the correct one. Any equivalence claimed in this work is to the **paper**,
never to that code path. This is the single most damaging bug a reimplementation of this
family can ship, and it is silent.
