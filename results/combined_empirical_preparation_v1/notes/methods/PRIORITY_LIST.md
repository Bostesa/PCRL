# Priority list: methods for a first runnable comparison (methods role, 2026-10-01)

**Scope.** This list ranks methods for the first comparison. It does not design that comparison. All arms will share one access protocol and one utility protocol, and the protocol itself is the protocol and threat roles' job. The method-side constraints a protocol must fix are listed at the end.

**How the list was ranked.** Each method was scored on two things. First, how much scientific uncertainty it resolves. Second, whether it runs today from official or existing code with a pinnable version, at $0 to small compute, with no new data. Citations and paths are in `method_catalog.csv`.

## A. First runnable comparison (ranked)

| Rank | ID | Method (pinned code) | Uncertainty it resolves |
|---|---|---|---|
| 1 | BAS-03 | Anchors: clean task-only release, withhold/constant, and the label-only predictor of S | Fixes the utility denominator and the floor of output leakage that comes from coupling between S and the label. Durable-guarantees found this coupling predicts removal cost (r ≈ 0.80, continuous_cost.json). Without anchors, no method's number can be read. |
| 2 | BAS-01 / BAS-02 | LEACE, `concept-erasure==0.2.4` (pin it; it is unpinned in both repos), run two ways: separate per recipient, and joint over the union of sensitive sets | (a) Whether exact in-sample linear guardedness predicts recovery on held-out rows and by nonlinear attackers. (b) The combination question with the cheapest baseline. Per-recipient release covers attributes declared for every recipient; the union release covers the case where only some recipients declared an attribute. (c) The reference that any "purpose-conditioned" method must beat. Recheck Cov(r(X), Z) after every fit. |
| 3 | LIN-07 | SPLINCE, official fholstege/SPLINCE @ fced0d3 (no license: run and cite). Validate the PCRL reimplementation against it, or use the official code | It keeps Cov(X, y_k) exactly, so it tests most directly whether outputs leak what the task uses: protecting the task signal on purpose should raise output-surface recovery when S is coupled to y. The existing PCRL 60-cell result (60/60 R², 38/60 auditor) used an unvalidated reimplementation with LEACE fallbacks. |
| 4 | LIN-01 | INLP, official nullspace_projection @ e1edcc1 (MIT) | Whether a method that stops on an empirical criterion (linear probe at chance) agrees with stronger recovery attacks. It also tests whether stopping on held-out data, rather than training data, changes the verdict. Existing results come from PCRL reimplementations only. |
| 5 | OWN-06 | Isotropic full-rank noise channel (dg@956f5c8) | The only family that survived the durable-guarantees battery. It tests whether stochastic releases differ in kind from deterministic erasure under informed attackers, and how independent draws across recipients interact with averaging. Needs no protected labels. |
| 6 | VIN-05 | FARE, official eth-sri/fare @ 89cb1b6 with patched sklearn fd60379f (NO license: run and cite; one subprocess per fit) | The only finite-sample certificate over all downstream models. It measures how far a DP-distance certificate disagrees with attribute-recovery AUC; the known case is dp_ub = 0 configurations leaking at 0.603-0.610. |
| 7 | ADV-02 | Generic adversarial censoring (DANN-style; dg baseline_gauntlet.py:196-232) | The representative adversarial family for multiclass sensitive sets. It tests whether training-time adversaries agree with post-hoc attackers, the failure documented by Elazar & Goldberg. On binary cells, add a LAFTR-official check (VectorInstitute/laftr @ a166ba3, GPL-3.0) for fidelity only. |
| 8 | VIN-01 | VFAE (durable-guarantees reimplementation; no official code exists) | The only published baseline that certified in Table 1 (easy cell, Tier 1). It tests whether that result holds under the common protocol and on held-out rows. It already fails Tier 2. |
| 9 | VIN-09 | Obliviator, official @ 0f2233f (no license: run and cite) | A recent nonlinear-guardedness eraser. It shows whether such a method can reach its own stopping criterion on tabular inputs; it never did in durable-guarantees. It should be reported as "did not reach own criterion", not as "passed its own test and failed ours". |
| 10 | OWN-02 (+ OWN-04 no-LEACE arm) | PCRL erase layer, plus the counterfactual that drops LEACE (rebuttal-evidence@5739d3b) | The project's own method under the same protocol as everyone else, with no privileged metric. Its key ablation, LoRA training versus BAS-02 joint LEACE alone, was registered (FAccT ablations) but never run. Most expensive arm: about 2-3 GPU-hours per dataset. |

**If the budget allows only a core set, use ranks 1-6.** That set covers four families: closed-form linear, iterative linear, stochastic noise and certificate-based. It is cheap, almost entirely CPU, and every arm has official or existing pinned code.

### Notes that apply to all ranks

* **Methods that need s at deploy:** VFAE, FNF, LAFTR with use_attr, MiMiC, QLEACE, Xie et al., L-MIFR. The protocol must either give every method the same deployment access or report these methods separately.
* **Binary-only methods restrict the eligible cells:** LAFTR official, FNF, the Fair PCA repo API, CFair, FFVAE, Gitiaux, FairNVT, MiMiC, Kernel RLACE.

## B. Later benchmark (runnable or adaptable, lower priority)

* **RLACE (LIN-03).** Use the MIT standalone repo. It has never been run in either repository; PCRL's "R-LACE" is not RLACE.
* **SAL (LIN-05).** Its linear guarantee is equivalent to LEACE's.
* **Methods with a narrow or adapted setup:**
  * FNF (VIN-04): binary attributes only, and A is needed at encode time.
  * Efficient Fair PCA (VIN-06): binary attributes only in the repo.
  * CVIB and FCRL (VIN-02/03): FCRL has no license.
* **Methods that need a reimplementation:**
  * Gitiaux & Rangwala (VIN-11): no code; close to OWN-06 plus a certificate auditor.
  * FairNVT (VIN-12): no code; ViT/BERT on CelebA or BIOS only.
* **Recipient-subset methods:**
  * FFVAE (FLX-01): the only learned method with native release of attribute subsets. No official code.
  * Adversarial Forgetting (AFR-03): per-task gates are only sketched in the paper; no code.
* **Project methods:** PCRL v2 per-purpose (OWN-01), the cross-purpose constraint (OWN-03), isolate-then-noise (OWN-07), the DP channel (OWN-08), ACS release Q (OWN-09, finite-token track; ACS years are exhausted, see SUMMARY flags), and OptNet-ARL (REL-04).

## C. Related work only

* **Theory, or a different notion of fairness:** log-linear guardedness (LIN-09), Samadi fair PCA (VIN-07), Stadler et al. (EVL-02).
* **Analysis tools or attacks:** amnesic probing (ADV-04), Song & Shmatikov (EVL-01). Their de-censoring attacker should, however, be considered for the informed tier.
* **Legacy stacks:** Elazar & Goldberg (ADV-01, Python 2 / DyNet; covered by ADV-02).
* **Needs z at deploy:** MiMiC (LIN-06), QLEACE (LIN-08).
* **No code and covered by other rows:** Kernel RLACE (LIN-04), Xie et al. (ADV-03), CFair (AFR-04), L-MIFR (FLX-02).
* **Record-level LDP, not attribute protection:** DPNR (VIN-10).
* **Finite-alphabet information-theoretic release:** privacy funnel (REL-01), Liu & Wang (REL-02), Taylor/Vippathalla/Coon (REL-03). REL-03 is the closest prior art for sequential collusion, but it protects the whole database X under a known pmf. Applying it to learned representations or declared attributes changes its claim.
* **Closed negative ACS channel family:** OWN-10.
* **Not yet verified:** LEOPARD, KRaM, MANCE (LIN-10); Olfat & Aswani (VIN-08).

## D. Important missing implementations

1. **No published learned-representation method designs releases to resist collusion.** Native recipient structure exists only in FFVAE (subset release, no coalition model), in Adversarial Forgetting (a sketch), in finite-alphabet information theory (REL-03), and in the project's own OWN-03 (full-coalition linear R² only) and OWN-09 (AB coalition CMI on a fitted finite model). Even a simple sequential, collusion-aware baseline would be new engineering. An example: refit recipient k's LEACE on [previous releases, X], with the coalition's declared union as the target. That is not a published method and must be labelled as our construction.
2. **No official RLACE run** in either repository.
3. **No official code** for VFAE, FFVAE, Adversarial Forgetting, Gitiaux & Rangwala, FairNVT, Xie et al., or DANN-for-fairness. Every result for these would be a reimplementation, and its fidelity checks must be registered.
4. **LAFTR and FNF have no multiclass version.** Durable-guarantees fed 5-class race into binary LAFTR as a scalar, which is outside the paper's theory.
5. **No informed de-censoring attacker** (Song & Shmatikov Alg. 1) is implemented. Durable-guarantees' Tier 2 uses a Gaussian likelihood-ratio test.
6. **No pinned concept-erasure version** in either repository's requirements.
7. **PCRL's in-house `LEACEEraser`** (origin/main:pcrl/models/baselines.py:120-181) must not be used as "LEACE". It is a different, non-guarding projection.

## E. Method-side constraints the protocol must fix (handed to the protocol and threat roles)

* **Protected attribute as an input column.** Both repositories include the protected attribute in X. FNF, Fair PCA and LAFTR-official were therefore tested outside their papers' setting. Declare this per arm, or run both settings.
* **Fit split versus probe split.** In durable-guarantees, encoders and erasers are fit on all rows and probed on a 75/25 split of the same rows. PCRL's R² refits OLS on the test rows. LEACE's guarantee covers the fitting sample only.
* **Output surface.** It must be scored, and it must gate the verdict uniformly. In durable-guarantees it gated the verdict only for FNF and Fair PCA.
* **Certificate type differs by method:** R², DP distance, I(z;c), or none. Report each method's own certificate alongside the common recovery metric. Never treat one as the other.
* **Comparable stopping and selection.** Some methods select on validation data (RLACE, Obliviator, FARE); others have no selection step. Selection must use the same data split for every arm.
