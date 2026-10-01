# Gitiaux & Rangwala (2021): Learning Smooth and Fair Representations (AGWN)

## Citation and version read
- X. Gitiaux, H. Rangwala. *Learning Smooth and Fair Representations.* AISTATS 2021, PMLR 130:253–261 (PMLR date 2021-03-18). Landing page: https://proceedings.mlr.press/v130/gitiaux21a.html.
- **Read in full:**
  - main PDF, gitiaux21a.pdf (10 pp.);
  - **supplementary PDF**, gitiaux21a-supp.pdf (10 pp.): proofs for Thms 1–4, Lemmas 1.1–1.5, experimental details, architecture and hyperparameter tables.
  - Both fetched 2026-10-01.
- Reading caveats:
  - The supplement has unresolved LaTeX references ("?").
  - Some equation layout is garbled in extraction. I cross-checked the main statements against the layout extraction.
  - Main-text noise is N(0, σ²I_d) (§3). Supplement Lemma 1.5 writes N(0, σI_d), so σ may mean std or variance.
- **Code:** none linked in the paper or supplement. A GitHub search for "gitiaux fair" returned nothing. **Code availability not established.**

## Actual question (authors' scope)
- Can a data controller release a representation whose **demographic-parity certificate**, estimated from a finite sample, upper-bounds the DP of **every** downstream processor (abstract; §1; Fig. 1)?
- The main conditions:
  - **Necessary:** finite χ²-mutual information I_χ²(X,Z) is needed for finite-sample certification.
  - **Sufficient:** an additive Gaussian white noise (AGWN) channel achieves it.

## Exact objective
- Learn encoder t and decoder g that minimise reconstruction loss plus λ × fairness loss (eq. 15).
- The fairness loss is L̂_DP(µ_{n,σ}): a Monte-Carlo estimate of E_z|η_n(z,1) − η_n(z,0)|.
- The quantities are computed on Z_σ = t(X) + N(0, σ²I) (§3.2):
  - η_n is the posterior P(S=s | Z=z), from a Gaussian-mixture plug-in density built on half of each mini-batch (eq. 10);
  - m is the number of noise draws per point;
  - λ is swept from 0 to λ_max.
- σ is chosen so that the plug-in certificate on train matches validation within 0.025, starting from σ ≈ 0.005 (§3.2, "Choice of σ").
- Final σ values: Swiss roll 0.05, DSprites 0.05, Adult 0.02, Heritage 0.05 (supp. Table 2).
- The encoder's last layer is tanh, so the representation is bounded (supp. Table 1 caption).
- There is **no task label in training**. This is an autoencoder; task utility is evaluated downstream (Fig. 6).

## Certified quantity and all assumptions
- **Certified quantity:** Δ\*(t) = sup over **all** f: Z→{0,1} of |E[f(Z)|S=1] − E[f(Z)|S=0]| (Def. 2.2, eq. 2).
  - This equals TV(P(Z|S=0), P(Z|S=1)) and **1 − 2·BER\***, where BER\* is the optimal balanced error rate of predicting S from Z (supp. Lemma 1.1; §3.2).
  - So the certificate bounds **both** (a) the DP gap of any binary downstream classifier operating on Z and (b) the balanced advantage of **any** attribute-inference test on a single draw of Z.
- **Empirical certificate:** Δ_n = ∆(f_n^plug, t) = 1 − 2·BER(f_n^plug), computed leave-one-out on train or test (§3.2).
- **Thm 2.1 (necessary):**
  - Distribution-free worst case over µ: sup_µ E|Δ\* − ∆(f_n,t)| ≥ sup_{µx}(1 − 1/I_χ²(X,Z))^n.
  - Cor. 2.1: if I_χ² = ∞ for some feature distribution (e.g., any **injective deterministic** encoder, or a large image |t(X)|), finite-sample certificates have no meaningful rate; Δ\* = 1 while empirical certificates read 0.
  - Proof (supp. 1.1): constructs S as a function of t(X) through binary expansions.
- **Thm 2.2 (sufficient):** E|Δ\* − ∆(f_n,t)| ≤ 2Σ_s sqrt(I_χ²(X,Z|S=s)/n_s).
  - The statement says "for all f_n ∈ F_n", but the proof (supp. Lemmas 1.3–1.4) is for the **plug-in auditor**. I read this as a misstatement in the paper.
- **Thm 3.1 (AGWN):**
  - If ‖t‖_∞ := sup_x‖t(x)‖₂ < ∞, then I_χ²(X,Z|S=s) ≤ exp(‖t‖²_∞/σ²).
  - Hence inf_{f_n} sup_µ E[Δ\*(t_σ) − ∆(t_σ,f_n)] ≤ 2 exp(‖t‖²_∞/2σ²)(n₀^{-1/2} + n₁^{-1/2}).
  - The bound is **in expectation** over the sample, not a high-probability statement.
  - It is independent of dimension d and depends on the signal-to-noise ratio ‖t‖_∞/σ.
- **Thm 3.2 / supp. Thm 4:** the Monte-Carlo loss converges at the same rate. Lemma 1.5 gives MSE ≤ 4(2‖t‖²_∞ + σ²)/(σ²nm).
- **Assumptions:**
  - S is **binary**. The extension to richer S, and to EO/EOpp, is asserted but not shown (§1, §2.1).
  - X ⊂ [0,1]^D and the encoder is bounded.
  - Samples are i.i.d. from µ.
  - The noise is isotropic Gaussian and **freshly drawn per released sample**.
  - Each individual is certified for a **single draw** of Z.

## What it does not guarantee
- **Not repeated draws.** A recipient seeing multiple independent noisy releases of the same x effectively faces a lower σ. Nothing in the theorems covers this (my inference from the single-draw definition).
- **Not side information.** Attackers holding S-correlated information beyond Z are outside the certificate.
- **Task outputs:** a binary output computed **from Z alone** is a test function, so its DP is covered. Multi-class or real-valued outputs reduce to binary tests, which is my inference.
  - Outputs computed from X directly, or from unnoised t(X), are **not** covered.
  - The deployed task head here is trained downstream on Z_σ, so it is covered by post-processing.
- **Not individual-level or worst-case per-input privacy.** The certificate is a population TV.
- **Not training-set privacy.**
- **Not conditional fairness** (EO). Also not attribute inference when S is non-binary.
- **Not numerically meaningful at the paper's own operating points (my computation).**
  - With the tanh encoder, ‖t‖_∞ ≤ sqrt(d) (d = 10 for Adult, d = 24 for Heritage per supp. Table 1).
  - Even granting ‖t‖_∞ = 1, the constant exp(‖t‖²_∞/2σ²) is e^{1250} at σ = 0.02 and e^{200} at σ = 0.05.
  - If σ denotes the variance, it is still about e^{25}.
  - So certificate reliability in the experiments is shown **empirically** (Figs 2–4), not by the bound.

## Recoverability, causal use, fairness or privacy?
- Framed as fairness (DP). The certified quantity is mathematically a balanced **attribute-inference** bound: Δ\* = 1 − 2·BER\*. So it measures recoverability of S from a single Z by any test.
- No causal use.
- No privacy framing.

## Representation exposed; datasets
- Exposed: Z_σ samples. Downstream processors get 10,000 fresh forward passes of the frozen encoder on test data (supp. §2.4).
- Datasets (supp. §2.1):

| Dataset | Sensitive S | Encoder / processor / evaluation sizes |
|---|---|---|
| Swiss roll (synthetic, S shifts the roll) | synthetic | 20k / 10k / 10k |
| DSprites (S = shape, biased sampling) | shape | 600k / 20k / 20k |
| Adult | self-identified gender | 32k / 13k / 3k |
| Heritage Health | gender | 142k / 17k / 17k |

- Task utility: Adult income prediction with the income feature left out of encoder training (Fig. 6). Reconstruction loss is also reported (Fig. 5).

## Removal methods and attacker families
- Methods: AGWN (ours), AdvCE (Edwards & Storkey) and AdvL1 (LAFTR-style). Adversarial auditors are 3-layer MLPs (supp. §2.3).
- **Downstream processors = attackers:** fully connected networks with 2–4 hidden layers and 32–128 units, 400 epochs, predicting S from fresh representations (supp. §2.4).
- Each (method, λ) is repeated 50 times (supp. §2.5).

## Adaptive knowledge and query access
- Processors are trained on labelled fresh samples. They are not given t, σ or the clean t(X).
- In principle the theoretical certificate covers all test functions of one Z, including defense-aware ones.
- One noise draw per input per processor.

## Utility protocol and model selection
- Reconstruction loss against the **95th quantile** of processor disparity (Fig. 5).
- Accuracy–DP scatter for downstream income classifiers of varied depth and width (Fig. 6).
- σ is tuned on validation for certificate generalisation.
- No confidence intervals beyond the scatter of 50 repeats.

## Individual vs multiple recipients
- "Many downstream users" means **many independent processors**, each evaluated alone.
- No combination of processors' views, and no multiple draws per individual.

## Strongest results
- **Figs 2–4:** with AGWN at σ ≥ 0.05 (synthetic), the plug-in certificate upper-bounds the disparity of "any of the downstream processors we built".
- **Adversarial baselines' certificates underestimate:** on Swiss roll, "18.2% of near zero empirical certificates (∆(f_adv,t) ≤ 0.1) do not preclude a processor's disparity larger than 0.3" (§5).
- **Fig. 6:** only AGWN reaches processor DP ≤ 0.1 by varying λ. At DP 0.10–0.20 its accuracy trade-off equals AdvCE/AdvL1.

## Limitations relevant to our analysis
- Binary S only.
- The certificate is averaged in expectation, and its bound constant is vacuous at the σ used.
- Attackers are a family of MLPs only.
- No defense-aware or multi-draw attack is evaluated.
- No code.

## What our work repeats, extends or contradicts
**Durable-guarantees (AAAI)**
- **Not cited.** It is absent from references.bib, although the meeting summary reportedly names it as a missing method.
- **Closest overlaps:**
  1. **Certified channel.** AAAI's "certified full-rank channel" (appendix.tex Prop. 3: clip to a C-ball, add σN(0,I), giving μ-GDP and an AUC ceiling Φ(μ/√2)) is a **bounded-encoder + isotropic Gaussian** mechanism, the same construction as AGWN with a tanh-bounded encoder.
     - The certificates differ:
       - AAAI uses per-input indistinguishability (DP), so the ceiling is distribution-free and data-independent but loose.
       - Gitiaux uses a data-estimated population TV/BER certificate with a finite-sample rate.
     - Both bind only single-draw attackers with no S-informative side information.
     - AAAI's appendix states this scope correctly for its own channel.
  2. **No sound certificate from a deterministic encoder.**
     - AAAI Prop. 2 says second-moment statistics cannot identify recoverability.
     - Gitiaux Thm 2.1 / Cor. 2.1 is a more general distribution-free statement: no finite-sample auditor certifies an injective deterministic encoder.
     - The two results are related but distinct. AAAI should cite Gitiaux as prior art for "certificates on deterministic representations do not generalise".
  3. **The adversarial-certificate gap** (18.2% at Swiss roll) is a prior instance of AAAI's "approved configurations fail stronger attackers" (59/67).
     - **Differences:** AAAI uses a linear-R² certificate, tabular data, varied attacker families, a defense-aware tier, the output surface, and a utility bar.
- AAAI's statement that FARE, FNF and Fair PCA are "the field's … certified-guarantee methods" (paper.tex:709–717) omits AGWN, a certified method of the same mechanism type.
- Contradicts: nothing. Gitiaux's empirical claim "robust to many downstream users" is consistent with AAAI's finding that only full-rank noise passes both tiers.

**PCRL**
- PCRL h_p is a **deterministic** (LoRA-adapted) encoder output. By Thm 2.1 / Cor. 2.1, no finite-sample certificate over all test functions can be attached to it distribution-free.
- PCRL's per-purpose R² ≤ τ is a statistic over **linear** test functions only. Smoothing (AGWN-style noise) would be needed to upgrade it.
- Cross-purpose: Gitiaux has no multi-recipient analysis.
- Contradicts: nothing.

## Stated scope vs my inference
- **Stated by the paper:** all theorem content, the experimental setup, and the 18.2% figure.
- **My inference:** that the bound is vacuous at the paper's σ (my arithmetic), the multi-draw and side-information exclusions (they follow from the definitions), and that "for all f_n" in Thm 2.2 is a misstatement.
