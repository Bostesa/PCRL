# Taylor, Vippathalla, Coon (2026). Adaptive Privacy of Sequential Data Releases Under Collusion

## Citation and version read
- **Authors and title:** S. Taylor, P. K. Vippathalla, J. P. Coon (University of Oxford). *Adaptive Privacy of Sequential Data Releases Under Collusion.* The title is verified on the PDF p.1 and on the arXiv abstract page.
- **arXiv record:** arXiv:2601.21859 [cs.IT].
  - v1: 29 Jan 2026 (2,613 KB).
  - **v2: 10 Jul 2026 (2,088 KB). This is the version read in full** (13 pp., §§I–IX plus references), fetched 2026-10-01.
  - No journal reference or comments field is listed. It is a preprint, and acceptance at any venue is not established.
- **Not read:** v1. Differences between v1 and v2 are not established.
- **Code:** no code link in the paper.

## Actual question (authors' scope)
- A data handler holds database X and receives **sequential requests** R_k from **m distinct parties**. Requests are online: future requests are unknown.
- For each request, choose the release channel p(r̂_k | r̂^{k−1}, x) to maximise utility U(R̂_k, R_k) subject to two mutual-information constraints (eq. 1, p.3):
  - an **individual** constraint I(R̂_k; X) ≤ ε_k;
  - a **collusion** constraint I(R̂_k, R̂^{k−1}; X) ≤ δ_k, with ε_k ≤ δ_k and δ_k non-decreasing.
- Utility is either −E[d(R̂,R)] (expected distortion) or I(R̂;R).

## Release design
- **Finite alphabets:** yes. The method optimises pmfs over discrete alphabets with a Blahut–Arimoto-style algorithm (§IV, Alg. 1–3). Complexity is O(T|R̂||Z||X|N_j) (p.9).
- **Known distribution:** the joint p(r, z, x) is assumed **known** to the handler. In experiments it is the empirical distribution (§VII-A).
- **Sequential and adaptive:** each release conditions on X and on **all previous releases** (Z = R̂^{k−1}). Releases are not jointly optimised. "She must apply an adaptive privacy scheme" (§II, p.2).
- **Ordering:** party 1 first, then 2, and so on. Earlier releases are fixed when later ones are designed. Budgets ε_k and δ_k are set per step (§II).
- **Guiding principle (§III):** if a request repeats an earlier one, the optimal release reuses the earlier release, so there is no additional collusion leakage.

## Formal guarantees and assumptions
- **Prop. 1:** a self-consistent sufficient condition for the optimum of the dual inner problem.
- **Lemma 1–2, Theorem 1:** an equivalence to alternating minimisations (p.6).
- **Expected-distortion problem:** convex, and the BA algorithm converges to the global optimum (Remarks 1–3, §IV-D).
- **MI-utility problem:** non-convex, so only local optima (Alg. 2) and a lower bound on the curve.
- **Leakage is MI with the whole X.** There is **no separate sensitive attribute S**, so the requested information itself counts as leakage. The guarantee is a constraint built into the mechanism. MI is an average-case measure, and there is no explicit attacker model.

## Attacker access and coalitions
- Party k receives R̂_k.
- **Collusion model (worst case):** "one malicious actor may gain access to the data supplied to all parties up to the most recent release" (p.2). This is all-prefix pooling. No knowledge of who colludes is assumed.
- **Multiple recipients:** yes; that is the core of the paper.
- **Named coalitions:** **§IX (p.13)** proposes constraints for subsets S_k^i ⊆ {1…k−1} of parties ("limit leakage should certain subsets of the parties collude"). The authors state the scheme "can be extended to this case". This is not implemented.
- The paper claims novelty for this scenario: "To the best of our knowledge, this scenario has not yet been considered" (§I-A, p.2).

## Recoverability, causal use, fairness or privacy?
Privacy, as information-theoretic leakage about X. No fairness and no causal use.

## What is exposed
Each party's released variable R̂_k. Colluders pool all of them.

## Datasets
- **§V:** a synthetic all-binary joint distribution (Table I) with Hamming distortion.
- **§VII (v2):** **UCI Adult**, discretised to X = (education 4 bins, income 2, age 4), |X| = 32.
  - Requests: (E, J, A, E) with ε ∈ {0.1, 0.3, 0.5} and δ_k = 0.2(k−1) + ε (Fig. 6).
  - Repeated-request curves for (E, E) (Figs 7–8).
- **Baselines:** non-adaptive (each release designed independently) and a symmetric channel.

## Utility protocol
- Expected Hamming distortion, computed exactly on the known distribution.
- No learned models, no held-out data and no statistical treatment (deterministic optimisation).

## Strongest results
- **Release 4 (repeated education request):** the non-adaptive mechanism incurs much higher cumulative leakage I(R̂^k;X) at the same distortion (Fig. 6, §VII-B).
- **Fig. 8:** the adaptive mechanism dominates on cumulative leakage versus distortion. Fig. 7 shows identical individual-leakage curves.
- **Fig. 8 also:** the history-aware BA updates reuse information even when δ = ∞ (§VII-C).

## Relation to a learned-representation pipeline
- **Stated by the authors:** §VIII links the dual inner problem to the information bottleneck and **progressive neural networks**, as a memory/compression analogy. It does not treat privacy of learned representations.
- **My inference:**
  - Applying it to learned representations requires (i) discretising representations to a finite alphabet, (ii) a known or estimated p(x, r), and (iii) treating each purpose-specific representation as a party's release.
  - It protects X as a whole, not a declared S with a permitted task.
  - Releases are designed sequentially and condition on previous releases. Per-purpose encoders trained independently are not.

## Prior PCRL notes — agreement check
Source: `research/pcrl-claims-foundation-v1@33124f965:results/pcrl_claims_foundation_v1/PRIOR_ART_NOTES.md` §15 (lines 296–322) and `PRIOR_ART_MATRIX.md` rows (b), (c), (e).

**Agree:**
- Title, authors and arXiv versions.
- Individual ε_k and collusion δ_k MI constraints, with previous releases fixed.
- Worst-case all-prefix pooling.
- Distortion convex and optimal; MI non-convex.
- §I-A novelty sentence.
- The four listed differences from PCRL: whole-X protection; unconditional per-party constraint; all-prefix pooling vs a named coalition including outputs the mechanism never produced; release may condition on previous outputs.

**Corrections or additions:**
1. The notes record "Opened: §§I–IV, VIII". **§VII (v2) contains a real-data experiment on UCI Adult (|X| = 32).** The matrix descriptor "theory + small numerics" is fair, but "synthetic" understates v2.
2. **§IX proposes subset (named-coalition) collusion constraints.** The distinction "all-prefix pooling rather than a named coalition" is accurate for what is solved, but the extension is anticipated by the authors. The remaining PCRL/ACS distinction should rest on the following:
   - the coalition view includes a third party's frozen output H_B that the mechanism never produced;
   - constraints are conditional on each view;
   - S is a declared attribute rather than all of X.
3. "Optimal … (Thm 1 and §IV)": Theorem 1 is the decomposition into alternating minimisations. Global optimality follows from convexity of the distortion problem and the convergence argument (§IV-D). This is a minor attribution nuance.

## What our work repeats, extends or contradicts
**Durable-guarantees (AAAI)**
- AAAI's "one release per input" condition and its **averaging attack** (N fresh noise draws; recovery grows as σ/√N; crosses 0.55 at 16 queries) are an instance of the sequential-release composition problem.
- Taylor's reuse principle (§III) implies a mitigation the AAAI paper does not discuss: release the same noisy copy for repeated requests about the same input. AAAI instead proposes a query cap or DP composition. That observation is my inference.
- AAAI has no multi-recipient or coalition evaluation. Taylor shows coalition-aware design is a formalised prior problem (for finite alphabets and MI to X).

**PCRL**
- PCRL's cross-purpose concatenation audit (S5-experiments.tex:157–174; 22 of 33 triples recover A) is an empirical **collusion** measurement. Taylor gives a formal collusion-constrained release framework (finite, MI to X, sequential).
- PCRL's explicit cross-purpose constraint trade-off (23/24 → 19/24 per-pair compliance) is a learned-representation analogue of an ε_k/δ_k trade-off.
- **For the ACS release lineage (SaTML candidate):** Taylor is the closest multi-recipient antecedent. This was already recorded in the claims-foundation notes.
- **Contradicts:** nothing.
