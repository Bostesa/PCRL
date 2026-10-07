# Prior art and claim scope (hcal: held-out, shared calibration of frozen releases)

Role F, 2026-10-07, after SOURCE_ADMISSION_LOCK and before SCIENCE_LOCK. No calibration fit, label or assessment value
was run or read for this file. Read today: Guo, Pleiss, Sun and Weinberger, *On Calibration of Modern Neural Networks*,
ICML 2017 (PMLR 70), https://proceedings.mlr.press/v70/guo17a/guo17a.pdf, main text §1–§6. The supplement (§S1–§S4)
was not read.

## 1. What Guo et al. establish
- **§2, definitions.**
  - Perfect calibration: P(Ŷ = Y | P̂ = p) = p for all p in [0, 1] (eq. 1), where P̂ is the max-class confidence.
  - ECE (eq. 3) = Σ_m (|B_m|/n)·|acc(B_m) − conf(B_m)|, over M equal-width confidence bins I_m = ((m−1)/M, m/M].
  - MCE (eq. 5) = max_m |acc(B_m) − conf(B_m)|.
  - NLL (eq. 6) is a calibration measure; in expectation it is minimised iff π̂(Y|X) recovers π(Y|X).
- **§3.** Networks can "overfit to NLL without overfitting to the 0/1 loss".
- **§4 (intro).** All methods are post-processing steps. "Each method requires a hold-out validation set". Training,
  validation and test sets are assumed to come from the same distribution.
- **§4.1, binary.**
  - Histogram binning (eq. 7): each bin gets its own value from validation labels.
  - Isotonic regression.
  - BBQ: Bayesian averaging over binning schemes, with a Beta prior.
  - Platt scaling σ(az + b): a and b are fitted by NLL on the validation set, with the network fixed.
- **§4.2, multiclass.**
  - One-vs-all extension of binning.
  - Matrix scaling Wz + b, fitted by validation NLL; its parameter count grows quadratically in K.
  - Vector scaling: W is diagonal.
  - Temperature scaling (eq. 9): q̂ = max_k σ_SM(z/T)^(k), with one T > 0 for all classes, "optimized with respect to
    NLL on the validation set". T > 1 "softens" the softmax. "T does not change the maximum of the softmax function", so
    the class prediction and the accuracy are unchanged. Footnote 3 also writes it in terms of 1/T.
- **§5, results.**
  - Table 1 reports ECE with M = 15 bins.
  - Temperature scaling "outperforms all other methods on the vision tasks, and performs comparably" on NLP.
  - Vector scaling recovers "essentially the same solution"; miscalibration is "intrinsically low dimensional".
  - Matrix scaling performs poorly with hundreds of classes and fails to converge on ImageNet. "Any calibration model
    with tens of thousands (or more) parameters will overfit to a small validation set".
  - Binning methods "tend to change class predictions which hurts accuracy".
  - Fitting T is a "one-dimensional convex optimization problem".
- **§6.** Temperature scaling is the simplest, fastest and often most effective method.

## 2. How hcal's calibrators relate (none is a new algorithm)

| hcal | Established method | What differs here (not a novelty claim) |
|---|---|---|
| H-GLOBAL-TEMP | Temperature scaling (§4.2 eq. 9), with α = 1/T (footnote 3) | z = log q0_t, the log of the frozen smoothed token mean, so α = 1 is the identity. One α in [0.25, 4] per task, map and seed. Fitted by NLL on 2,000 held-out calibration representatives (§4 hold-out set), solved by bounded bisection. α > 0 keeps the argmax (§4.2) |
| H-CLASS-TEMP | Per-class parameterisations: vector scaling and one-vs-all binning (§4.2) | One α per **predicted** class (α = 1 below 50 representatives). This restriction of temperature scaling is not defined by Guo et al. It is one positive scalar per row, so unlike vector scaling it cannot change the argmax |
| H-TOKEN32 | Per-cell recalibration from validation labels, i.e. histogram binning (§4.1 eq. 7, tokens as bins), with prior shrinkage in the spirit of BBQ (§4.1) | Log-loss + ½ Brier per token, a κ = 32 KL prior to the frozen teacher mean mu_t, and a class-dominant constraint (no prediction change, unlike §5 binning). Many more parameters than one temperature: §5 already warns that these overfit small validation sets |
| T- vs H-TOKEN32 (D01–D04) | §4: fit on a hold-out set | Measures that effect on these frozen releases; it does not discover it |
| H-TOKEN32 vs H-GLOBAL-TEMP (D05–D08) | §5: simple calibrators beat high-parameter ones | Same: it measures the effect on these releases |

ECE (§2 eq. 3) is supplementary in hcal. NLL and Brier are the registered metrics (PROTOCOL §4).

## 3. Claim scope
- **No new calibration algorithm.** Temperature scaling and per-class / vector-scaling-like variants are established.
- **No privacy mechanism.** A calibrator is a deterministic decoder change that keeps the token identity.
  - Every public decoder table is computable from the token (PROTOCOL §1), so a decoder-only change is not information
    removal.
  - A finite attacker scoring lower after such a change is not protection. All decoder variants of a partition share one
    attack bank (PROTOCOL §5).
  - Guo et al. make no privacy claim.
- **Success is a pipeline improvement on development data.**
  - Guo's §4 hold-out recipe is followed, but the rows are reused, not fresh: CALIBRATION_HELDOUT is held out from the
    current teachers' fitting only, and AUDIT_FIT rows were used historically (PROTOCOL §2).
  - The results are exploratory development evidence (PROTOCOL §1 exposure statement), not confirmation and not
    generalisation.
  - If an original decoder wins: "the incumbent won; no calibration credit".
- **Not "joint beats sequential".** All partitions are frozen, and hcal registers no joint-versus-sequential
  comparison. A LOCAL or sequential map gets no joint-design credit (PROTOCOL §6).
- **No population privacy guarantee.** Privacy means recovery by the declared attacker slate. There is no
  population-MI, Bayes-optimal, training-data, repeated-query or unseen-attacker guarantee (PROTOCOL §1).
- **Never write:** "new calibration method", "calibration protects / removes information", "private", "certified",
  "confirmed", "joint beats sequential", or any population or MI guarantee.

## 4. PCRL lineage prior art
These are carried over from `results/pcrl_adult_learned_decoder_release_v1/PRIOR_ART_AND_CLAIM_SCOPE.md` and were not
re-read today. hcal inherits their maps frozen and re-tests neither paper.
- **Makhdoumi, Salamatian, Fawaz and Médard**, *From the Information Bottleneck to the Privacy Funnel*, arXiv:1402.1774
  (IEEE ITW 2014). It covers hard utility-constrained leakage minimisation, greedy partition search and non-convexity.
  The bank's constrained maps are adaptations, not the Privacy Funnel algorithm.
- **Taylor, Vippathalla and Coon**, *Adaptive Privacy of Sequential Data Releases Under Collusion*,
  arXiv:2601.21859v2. It covers sequential releases with per-party and collusion accounting. The LOCAL and SEQ maps are
  matched adaptations, not the Taylor solver.
