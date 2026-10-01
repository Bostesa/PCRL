# One connected paper: provisional outline

Status: a draft for advisor review, written 2026-10-01. It contains no new algorithm.

The question of one paper versus two is deliberately left open. It should be decided after the evidence
map below has been reviewed. The advisor's concern about overlap between papers is recorded as a concern;
it is not treated as an established fact about acceptance or venue.

How to read the fit labels:

| Label | Meaning |
|---|---|
| **direct** | Usable as-is, with the corrected wording. |
| **rescore** | Recomputable from stored artifacts under repaired definitions; no training. |
| **rerun** | Needs new fits under the common contract (a later phase). |
| **appendix** | Historical: kept with its original definition, disclosed, and not part of the argument. |

Evidence IDs refer to EVIDENCE_CROSSWALK.csv. Findings refer to METHODOLOGY_IMPROVEMENTS.md.

---

## 1. The practical sharing problem and declared purpose permissions

**Content**
- A data holder serves several recipients, each with a declared task.
- There is a finite declared sensitive set S, and each recipient has an allowed subset of S.
- Deny-by-default applies only within S. Nothing is promised for undeclared attributes, which is where
  Stadler et al.'s result applies.

**Existing material**
- PCRL encoder §1 motivation: **rescore wording**. The HMDA underwriter/auditor conflict is not
  instantiated in code (PE-F20), so the example must either be implemented or dropped.
- ACS release framing (withdrawn SaTML): **direct** for the recipient/coalition vocabulary.

**Gap**
- The policy table in protocol §2 is new and needs advisor sign-off.

## 2. Existing attribute-removal guarantees and their precise scope

**Content**
- LEACE: exact linear guardedness on the fitting law.
- INLP / RLACE / SPLINCE: what each one guarantees.
- FARE: a demographic-parity certificate, not a recovery guarantee.
- Gitiaux & Rangwala: a single-draw certificate under smoothing.
- Gaussian-mechanism GDP: single release, clipped map.
- Composition: exact zero covariance composes; R² ≤ τ does not.
- The withdrawn R²→accuracy guarantee is retired; the counterexample can go in the appendix.

**Existing material**
- AAAI Props 1 and 3 (DG-C01, DG-C03): **direct**, with scope fixes.
- AAAI Prop 2 (DG-C02): **rescore wording**, restricted claim.
- PCRL convex-combination identity (PE-C04): **appendix**.
- ACS algebraic composition statements (PA-C01..03): **direct**.

## 3. What existing empirical analyses already establish

**Content** (ANALYSIS_POSITIONING §1)
- Linear tests miss nonlinear recovery (Ravfogel et al. 2022, ×2; Elazar & Goldberg 2018).
- Recoverability is not use (Elazar et al. 2021).
- Overlearning (Song & Shmatikov 2020).
- Label-only fundamental leakage (Stadler et al. 2024).
- Averaging undoes isolated noise (FairNVT).
- Finite-alphabet collusion (Taylor et al. 2026).
- Output leakage measured against matched label leakage (Wang et al. 2019, bias amplification).
- Erased representations leak through downstream predictions (Ravfogel, Goldberg & Cotterell 2023).
- Certificates checked against held-out classifiers (FARE; FRG 2025).
- Defense-exploiting attribute inference (Ferry et al. 2023; Johansson anti-clustering).
- Combined model versions (Tian et al.).
- The full list is in PRIOR_ANALYSIS_MATRIX.csv.

**Our own reproductions of known observations**

These are kept brief and presented as reproductions, not contributions.

- durable-guarantees: 18 certificate-approved configurations, 15 collapse under XGB/MLP. **rescore**:
  the README "21" is wrong (DG-C04).
- PCRL: nonlinear auditors recover what R² ≤ τ hides (PE-C07/C08). **direct**, as a reproduction.

## 4. A unified evaluation contract and the methodological repairs

**Content**
- Access tags A1–A5.
- Split roles grouped by linkage unit.
- Fixed-head and refit-head utility.
- Output-surface release object.
- Nested slates.
- Contrast-sensitive categorical diagnostic.
- Registered threshold grid.
- Inference rules.

**Existing material**
- Fixtures F1–F10 in `fixtures/` (this phase): **direct**, as validation of the evaluation itself.
- Historical metric definitions: **appendix**, kept as Track H.

## 5. Single-recipient representation and output analysis

**Content**
- For each method, compare its own stated test against A1/A2 recovery under the common utility
  contract.
- Measure the output surface against the label-only and clean-output references.

**Existing material**
- durable-guarantees gauntlet (1/36; 7/42): **rerun**. The verdicts are representation-gated, the
  utility is in-sample, and the access labels are wrong (DG-F01/F03/F05).
- The A4-tagged LRT values can be **rescored**, re-labelled only, as a stress test.
- AUC(s|y) cost predictor, r = 0.80 (DG-C05): **rescore**. Recompute utility on held-out rows. Its
  measurement is prior work (Stadler); the contribution is limited to the cross-method cost relation.
  Whether that relation survives held-out utility is open.
- PCRL encoder headline (56/60): **appendix**. It is in-sample on spent test splits, the backbone is
  random, and the utility claim is false (PE-F1/F9/F18).
- PCRL LEACE / INLP / LAFTR / SPLINCE comparisons: **rerun with official code**. The in-house
  `LEACEEraser` is not LEACE, and "R-LACE" is not RLACE.
- CelebA (both repos): **rerun**. Use an identity-disjoint split and a neutral backbone.

## 6. Multiple-purpose and coalition analysis

**Content**
- Per-recipient, other-recipient, union, output-only, and representation+output views, plus the
  same-information reference.
- Combination increment Δ on jointly-disallowed attributes only.
- Per-recipient LEACE as the baseline.

**Existing material**
- PCRL cross-purpose concatenation (26/33 absolute, 22/33 incremental): **rescore** with nested slates.
  The existing finding that **single-recipient leakage dominates** (17/33 cells, PE-F22) is itself an
  important result. It limits the empirical motivation for collusion-specific methods in these cohorts.
- Rebuttal 8/33 and erase-pilot 60/60: **appendix**. These use a union eraser, a different model, and a
  post-hoc criterion (PE-F10/F23).
- ACS coalition studies (Sept 2026): **appendix / development**. The slates there were correct (nested
  H, equal seeds), but:
  - recovery beyond H is positive on 8/8 endpoints;
  - the 2016–2018 data are spent;
  - catch-up slates are unequal in the redesign studies (PA-F1/F4/F6).

  These are the best existing evidence on combination effects, but they come from a finite-token release
  rather than learned representations.
- Taylor et al.: related work only. It uses finite alphabets and protects all of X.

## 7. Utility/protection trade-offs and failure explanations

**Content**
- Frontiers per method at each registered bar.
- Degenerate and collapse cases flagged.
- Cost relation to label coupling.
- Repeated-query (A3) curves for stochastic channels.

**Existing material**
- durable-guarantees isolate vs full-rank: **rescore** as a frontier comparison. The nearest-match rule
  is replaced, which shrinks the easy-cell advantage from about 100 pp to about 46 pp (DG-F08).
- Averaging attack: **rescore/rerun** to extend N and cover the full-rank channel (DG-F02).
- PCRL "compliance via collapse" and per_dim_std diagnostics: **appendix**. A hypothesis exists about an
  in-batch R² null floor near 0.25 (PE-F3), and it is unverified.

## 8. Discussion: what a future method must accomplish

Open requirements only; no evaluated architecture.

- Distinct access restrictions for recipients with conflicting permissions without destroying shared
  task signal.
- Protection that survives A2 and an explicit A3(N) budget.
- A composition statement that covers the released objects, including outputs.

The comparison that would show a method meets these:
- the same contract;
- against per-recipient official LEACE, the noise channel and FARE;
- with Δ and output increments as primary estimands;
- on a locked confirmation population.

## 9. Limitations and reproducibility

**Content**
- Spent data and development status.
- Single-state ACS.
- Survey weights.
- Linkage units.
- Attacker-family dependence (no recovery result certifies absence of information).
- Unavailable official reviews.
- Relocated archives.

**Artifacts**
- Pinned commits.
- `fixtures/run_all.sh`.
- Recount scripts.
- Manifests.

## Appendix (historical)

- PCRL v2 round history (R4→R5→R7 patches), with final vs best counts.
- The retired accuracy guarantee and its counterexample.
- Erase-layer pilot and VICReg sweep.
- BIOS negative result.
- The ACS release-study chain (closed negative and not-established outcomes).
- The original "Tier 1 / Tier 2" tables, kept with their original definitions.

## Overlap that needs an author decision

- **AUC(s|y) / label-only leakage.** It appears in AAAI as a contribution and overlaps Stadler §4.1.
  In the combined paper it should be presented as a reproduction plus a cost relation.
- **The same fits counted in more than one paper.** durable-guarantees Adult/HMDA cells reuse PCRL v2
  checkpoints (`v2_adult_s0/final.pt`, `v2_hmda_s0/final.pt`). These are the same underlying fits as
  PCRL encoder results and must be counted once. Which PCRL round produced them is unverified (DG-F21).
- **CA 2018 ACS.** It is used by both repositories: one population, not independent replications.
- **Venue history.** Both prior submissions are closed (NeurIPS withdrawn; AAAI Phase 1 decision), and
  the ACS SaTML submission was withdrawn. There is no live venue conflict on record. Nothing was
  submitted, withdrawn or changed in this phase.
