# Corrected contributions

For Terminal 2 to integrate into the manuscript at 30a6fd19e (`papers/pcrl_satml_final_v1/main.tex`).
Every item names its implementation and its evidence. Claim ids refer to CLAIM_MATRIX.csv.

## One sentence (for Nathan)

The paper sets up and honestly measures a real problem: adding useful information for one recipient
beside predictions that are already published and cannot change, while checking what that recipient,
alone or with another, newly learns about sex and race. The tools for building the release already exist.
Our tested release helps the permitted task, but has not been shown to beat a simpler, non-randomised
release on privacy.

## Short technical explanation (for the paper)

We formalise purpose-specific release beside immutable prediction services: per-recipient permitted
tasks and forbidden attributes, a coalition view that includes an output the releaser never controlled,
and a measurement contract in which every disclosure figure is an independently fitted attacker's gain
over an attacker restricted to the published outputs, reported beside absolute recovery. We instantiate
an established convex finite-alphabet leakage–distortion programme for one recipient and evaluate it on
reused ACS development data. It improves the permitted task over the strongest channel we had, with
bounds excluding zero, but a matched deterministic release performs comparably, and the registered
competitiveness conjunction is not established. We also report audit cautions and corrections from an
earlier purpose-conditioned encoder line.

## Recommended contribution list (four items)

1. **Problem and measurement contract for release beside immutable services** (C01, C02).
   *Implementation:* `experiments/pcrl_task_directed_release_v1/evaluation.py` roles A, B and AB;
   H-only ancestors; J excluded from non-J views; unclipped increments. *Evidence:* ATTACK_CALIBRATION,
   HEADLINE_EVIDENCE @ f4bdf4cd. *Closest prior:* Erdogdu & Fawaz 2015 (a release beside fixed prior
   releases, incremental leakage); Taylor et al. 2026 (per-party and collusion MI constraints). *Remaining
   distinction:* the fixed view is a third party's ML prediction service, which a coalition partner also
   holds, and the audit is empirical with independently fitted attackers. This is a formulation and
   methodology contribution, not an algorithm.
2. **An evaluated operating point, with its controls** (C10–C13). *Implementation:* T0_L_0.01_a17
   (local constraints only). *Evidence:* HEADLINE_EVIDENCE; Terminal 1 Tier A recomputation of 38/38
   endpoints. *Content:* permitted-task gain over J of 0.01625/0.01639 nats, bounds excluding zero; 8/8
   sensitive point estimates favourable, 7/8 unresolved; D17 comparable. *Closest prior:* none for this
   dataset and contract. It is an empirical finding on reused development data.
3. **Audit cautions for linear leakage reporting** (C05–C07). *Implementation:*
   `compute_dominant_axis_r2`, `LinearComplianceCertificate`. *Evidence:* DOMINANT_AXIS_REPLAY.json
   (numerical replay), fixtures F2, F2b, F3, F4, F11. *Content:* the pooled one-hot score is a
   variance-weighted average of per-class scores, so a rare class can hide (HMDA 0.027 vs 0.288, rarest
   class). Exact-zero composition holds but gives no independence. Approximate leakage composes with a
   1/λ_min penalty, and that penalty is tight. *Closest prior:* variance-weighted R² (a definition); LEACE
   Thm 3.4; Rayleigh–Ritz. *Distinction:* the empirical example and the audit practice, **not** the
   identity.
4. **Corrections to earlier claims** (C08, C09, C15, C16, C19). The R²→accuracy guarantee is refuted; the
   shared-union retention claim is narrowed; the LoRA floor is misapplied; the subsumption reading is
   withdrawn; the backbone is disclosed as random. *Evidence:* THEORY_REPAIRS; fixtures F1 and F12;
   patch 0001.

Merge item 2 into item 1 if a three-item list is preferred.

## What must not be claimed, with replacements

| Do not say | Say instead | Why |
|---|---|---|
| "a label-free code T=g(X_A)" (main.tex:191, :280) | "a code from 32 quantile cells of the residual residence logit of two residence-supervised teachers" | encoding.py @ f4bdf4cd (E1) |
| the evaluated release limits the coalition's inference (abstract) | "fitted under a budget for the recipient's own view; the coalition view is audited, and a coalition-constrained variant was also fitted" | the selected nominee is local; fitted AB CMI > δ (E2) |
| "a floor on what is recoverable" (:110) | "absolute recovery lower-bounds the view's information; the increment bounds neither way" | F9 (E4) |
| "the worst single direction across classes" (:129) | "the largest per-class one-versus-rest score, a lower bound on the best linear direction" | F3 |
| "we give the exact convex-combination identity" as a result (:75) | "we point out that the pooled score is variance-weighted" | definition of variance-weighted R² (M7) |
| "identity validated on every pair-seed, max residual 0.0021" (:132-133) | the exact identity plus the replay sentence (E7b) | float32 code path |
| "valid statement … squared loss only" (:150) | "any loss convex in the affine prediction; not thresholded accuracy" | LEACE Thm 3.1 (M6) |
| "the corresponding API now refuses to run" (:149), unqualified | "…in the evaluated code; the public default branch is being updated" (or apply patch 0001) | origin/main still ships it (E6) |
| Rassouli–Gündüz give "the LP form of the zero-budget linear-cost problem"; "the convexity facts are theirs" (:176-181, :203-204) | M1/M2 wording in PRIOR_ART_MATRIX | the LP is for MI utility; convexity is Calmon & Fawaz 2012 or standard |
| "U-FaTE, the conditional extension" (:171) | M3 wording | label-conditional dependence; no eigen-sign rank rule |
| Sankar/Liao as nearest multi-consumer prior art (:179-180) | cite Taylor et al. 2026, Erdogdu & Fawaz 2015 | M4, M5 |
| "joint design" of channels for several recipients | "a channel for one recipient, audited alone and with another recipient's fixed outputs" | only A is optimised |
| "shared frozen backbone … pretrained" (NeurIPS §4.1; any reuse) | "a frozen, randomly initialised MLP backbone" | run_v2_dataset.py @ dbe0fdc (E5) |
| "Proposition 4 explains collapse / sets the LoRA rank floor" | "a single rank-r multiplicative edit cannot erase below Σ_{i>r}σ_i²; multi-layer adapters are not bounded by it" | F12 |
| LoRA "is justified / necessary / efficient" beyond parameter count | "we did not compare against a plain per-purpose linear adapter" | ablation 1 registered, never run |
| "Zhao–Gordon near-optimality certificate" | "a lower bound on summed group error given the demographic-parity gap" | T5 |
| rebuttal "54/60 → 60/60" as the paper's headline improvement | report it against the same checkpoint rule (the paper's headline is 56/60 at the final iterate) | per-seed JSON (best checkpoint) = 54; final.pt = 56 |
| "we correct published claims of our own" (abstract) | "we correct claims made in an earlier version of this line of work" | the original was a submission, and the wording also bears on anonymity |

## About the proposed title

"Sharing What Helps, Limiting What Leaks" fits the problem. "Learning Representations for Specific
Purposes" describes F1, the encoder line, not the ACS release. The evaluated ACS mechanism is a finite
stochastic channel over a supervised code; it is not a learned representation. If the paper's empirical
centre is the release study, a subtitle such as "Purpose-Specific Releases Beside Fixed Prediction
Services" matches the implementation. The registered title is Terminal 2's and the author's decision;
Terminal 3 changes nothing.
