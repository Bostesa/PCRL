# Prior-art delta for the proposed full-view increment

This reuses the completed audit (`results/pcrl_claims_foundation_v1/PRIOR_ART_NOTES.md` @ 33124f96, where
Taylor et al., Calmon–Fawaz, Erdogdu–Fawaz and Rassouli–Gündüz were read at section or theorem level).
The four sources named in the assignment were re-read here at source.

| Source (verified) | What it establishes | Metric / assumptions | What it does not give us |
|---|---|---|---|
| Taylor, Vippathalla, Coon, *Adaptive Privacy of Sequential Data Releases Under Collusion*, arXiv:2601.21859 (v1 2026-01-29, v2 2026-07-10; title, authors and abstract checked at arXiv; sections per the audit notes) | Sequential releases to several parties. Per-party constraint I(R̂_k; X) ≤ ε_k and collusion constraint I(R̂_k, R̂^{k−1}; X) ≤ δ_k, with earlier releases fixed. Optimal for expected distortion (Blahut–Arimoto style); locally optimal for MI utility | Shannon MI; finite alphabets; the protected object is the whole database X | No separate sensitive S with a permitted task Y. Conditioning is on the releaser's own earlier outputs, not a third party's continuous service view |
| Diaz, Wang, Calmon, Sankar, *On the Robustness of Information-Theoretic Privacy Measures and Mechanisms* (author PDF, `tit20.pdf`; IEEE Trans. Inf. Theory, 2020 — volume and pages not re-verified) | Mechanisms designed on P̂_n: the discrepancy of leakage and utility is O(1/√n) for guessing probability, f-information (locally Lipschitz f), Arimoto and Sibson α-information and maximal α-leakage (Thms 1–5). "Uniform privacy mechanisms" protect every law within a large-deviation L1 radius of P̂ (§5) | **Finite alphabets.** The adversary "has no side information regarding the disclosed data" (§1). Shannon MI "appears to fall outside the scope" of the O(1/√n) technique (§1, citing O(log n/√n) for MI) | No conditional MI given a continuous published view. Its rates **must not** be transferred to I(S;Z\|H). Its uniform-mechanism idea is the direct antecedent of any calibrated-envelope construction |
| Lopuhaä-Zwakenberg, Goseling, *Mechanisms for Robust Local Differential Privacy*, Entropy 26(3):233, 2024, doi:10.3390/e26030233 (read via the PMC mirror) | (ε,F)-RLDP: P(Y=y\|S=s) ≤ e^ε P(Y=y\|S=s′) for all P in F = {P : D_α(P̂‖P) ≤ B}. Projections of Rényi balls are Rényi balls (Thm 1). Finitely many linear constraints (Thm 2). An LP over polyhedron vertices (Thm 3). SRR optimal under maximal uncertainty (Thm 4) | Finite alphabets; the mechanism input is X = (S, U), so it **sees S**; LDP-type max-divergence | Its "side information" is the public estimation sample, not an adversary's correlated continuous view. It does not guarantee against a continuous published H |
| van Erven, Harremoës, *Rényi Divergence and Kullback–Leibler Divergence*, arXiv:1206.2459v2 (IEEE Trans. Inf. Theory 60(7), 2014 — venue from the listing, not re-verified) | Thm 34: for a finite sample space, the Rényi capacity equals the minimax redundancy ("radius") for every order α ∈ [0,∞] and general input sets. Credits Gallager–Ryabko for α=1 and Csiszár 1995 for finite input sets | Finite output alphabet | It is the source of Theorem 1's tightness. The radius bound is theirs and earlier, not ours |
| Calmon–Fawaz 2012; Salamatian et al. 2015; Erdogdu–Fawaz 2015; Rassouli–Gündüz (JSAIT 2021) | Convex finite-alphabet leakage–distortion design; quantise-then-optimise; a new release beside fixed earlier releases, with incremental leakage (convex, Thm III.3); a perfect-privacy nullspace criterion, with an LP for MI utility | Finite alphabets | As in the audit |
| Polyanskiy, Wu, *Strong data-processing inequalities for channels and Bayesian networks*, arXiv:1508.06025 | η_KL = sup I(U;Y)/I(U;X) (eq. 17); η_f ≤ η_TV (Thm 1, from Cohen–Kemperman–Zbăganu 1998) | General | The source of Theorem 1's SDPI form |

## Smallest substantive distinction

Every distribution-free guarantee available for the proposed release (Theorem 1) is **existing
mathematics**. It never uses H, and it holds for any side information because it bounds all information
the release carries. Using a continuous H as the *conditioning* view does not by itself produce anything
new: I(S;Z|H) ≤ κ follows from the radius, and I(S;Z|H) = I(S;Z,H) − I(S;H), so a conditional budget is a
joint-view budget with a shifted constant, as in Erdogdu–Fawaz and Taylor et al.

The **only** increment I can identify that is not already covered is:

> an attribute-specific certificate of I(S;Z | H) at the level of the recipient's **continuous**
> published service view, for a mechanism that sees only (X_A, b(H_A)), under an explicitly stated
> assumption linking bin-level and H-level laws, with the worst case computed exactly (vertex
> enumeration) or bounded (Theorem 1), and an envelope calibrated with household-, weight- and
> selection-aware uncertainty.

That is an **adaptation** of Diaz et al.'s uniform mechanisms and of RLDP's robust polyhedral design to
conditional Shannon information with continuous side information. Two parts are genuinely new work and
are not in those sources: the assumption (Proposition 3's homogeneity, or an H-smoothness bound) and the
calibration.

"Ordinary robust optimisation plus a new name" is not enough. Without the assumption, the construction
reduces to either Theorem 1 (standard) or bin-level robustness (which G5c shows does not cover H).

## What empirical result would make the adaptation valuable

Registered in advance, on data not yet used:

1. At an equal **certified** level (for example a certified I(S;Z|H_AB) ≤ 0.026 nats), residence Bayes
   or held-out gain over H **larger than the best radius-κ kernel with the same certificate**. This is the
   test that the attribute-specific certificate buys utility the all-information bound cannot.
2. Measured recovery non-inferior to D17 and J, and a task gain beyond the registered margin. The 2016
   evaluation shows the margin is about the realised effect size, so precision planning must use the
   2016 size, not the 2018 development size.
3. Reporting κ and η_TV of the evaluated kernel beside every certified number, so readers can see whether
   the attribute-specific certificate beats the trivial all-information one.
