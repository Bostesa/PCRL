# Corrections to the mechanism-design campaign (correction pass, 2026-10-16)

**What this file is.** A dated correction record for the data-free mechanism-design campaign. It lists each corrected claim with its original location, a classification, the replacement wording, the evidence, and the effect on the recommendation.
- **Original record:** the pinned public package `e141330` and the research tip `12ffa8558`. Neither is changed.
- **This correction:** branch `research/pcrl-mechanism-design-correction-v1`.
- **Preserved without edits:** historical log entries, locks, numerical outputs, failed checks and decisions. The decision log and the selection decision receive dated notes. The current summaries were updated in place, each with a banner pointing here.
- **Data:** no real-person feature, prediction or label array was opened in any role. Nothing was fitted. The Florida sealed assessment stayed closed, and disclosure D2 still applies.

**Line references.** These are to the files as they stood at `12ffa8558`, which is the content of public `e141330`. The public package keeps `results/` at its top level.

**Roles in this pass.**
- **Coordinator and editor:** the main session.
- **Mathematical reviewer:** `roles/correction_v1/math_reviewer/MATH_REVIEW.md`.
- **Adversary reviewer:** `roles/correction_v1/adversary_reviewer/ADVERSARY_REVIEW.md`.
- **Independent verifier:** `roles/correction_v1/verifier/VERIFICATION.md`.

## Bottom line

1. **The injectivity claim is withdrawn.** The campaign said that every evaluated ACS release is essentially injective, so that the optimal attacker recovers exactly the raw-field level and every measured difference is a learnability difference. That claim does not follow from determinism or continuity, and nothing in the record establishes it.
   - Whether the actual ACS models are injective on their population support is **UNKNOWN**.
   - It is known that every model reads an encoding that pools rare category levels. So none of them can be one-to-one in the six raw fields on the declared domain.
2. **What remains valid.** By data processing, recovery from any release is bounded by recovery from the inputs. Equality needs the release to reconstruct the inputs, or at least to keep P(S | X), almost surely; neither is established for any ACS model.
   - Fixed-reader results remain what they always were: empirical evidence against the declared attacks.
3. **Information removal is possible inside the contract.** A deterministic many-to-one probability release can remove sensitive information within the existing output contract. Exact check B does this with no utility loss.
   - Whether any such release meets this benchmark's caps on the Florida data is an open feasibility question. Earlier coarsening failures concern specific mechanisms.
4. **MMP's scope is corrected.**
   - The population identities are correct under a known law.
   - Three phrases are corrected: "utility cost known exactly in advance", "if male / if female predictions", and "removes information only along pure-proxy directions".
5. **RCCS: one ground for rejection is corrected.**
   - The claim that it "only works against linear and quadratic attackers" is false as stated. In the verifier's two smooth synthetic laws, its fitted protection extended to tree and neural readers.
   - Its rejection as presented still stands on its feasibility evidence (S5 fails as presented, so rule 1 applies). A repaired version is unresolved, not promoted.
6. **The adversary text is corrected.** The rationale for excluding other access and the other overstatements listed in §6 are fixed.
7. **The status is unchanged.** `NO_DEFENSIBLE_NEW_MECHANISM_IDENTIFIED` (for a real-data pilot) is retained, because it rests on the prototype evidence, not on the withdrawn claim (§7).
   - No candidate acquires a competitive claim, and no real-data pilot is authorised.

## 1. The corrected mathematical statement

Notation: S is binary; X is the input; R is the released view (one recipient's output, or the linked pair); AUC\* is the area under the optimal ROC (likelihood-ratio ranking, ties counted ½). Proofs and conditions are in the mathematical reviewer's and verifier's files.

- **(a) Data processing: a bound, not an equality.** If R is generated from X with no extra information about S (any randomness independent of S given X, i.e. S → X → R), then the optimal ROC from R lies pointwise below the optimal ROC from X. Hence AUC\*(S; R) ≤ AUC\*(S; X).
- **(b) Reconstruction gives equality.** If some measurable g has g(R) = X almost surely under the population law of X, the optimal ROC and AUC from R equal those from X. Injectivity on a finite sample is neither necessary nor sufficient for this. Injectivity on a declared domain that contains the population support is sufficient but not necessary; on the encoded domain it gives AUC\*(S; E(X)), not AUC\*(S; X_raw). Conversely, non-injectivity on the declared domain, as for the pooled encoder, does not imply non-injectivity on the population support.
- **(c) Sufficiency gives equality without reconstruction.** If P(S = 1 | X) equals a measurable function of R almost surely, equality holds even though X cannot be recovered. Exact check D merges cells and keeps AUC\* = 4/5; check E shows equality for a non-injective partition.
- **(d) The converse holds under S → X → R, and without that condition it can fail.** Under that condition, equal AUC\* forces equal optimal ROC curves, hence equal laws of the posterior, hence P(S = 1 | X) = P(S = 1 | R) almost surely. Equality of the optimal AUC therefore happens exactly when the release keeps the sensitive posterior.
  - Without the Markov condition the converse fails: two independent noisy copies of S both give AUC\* = 4/5.
  - The converse is specific to the whole optimal ROC. A single operating point can be kept while AUC\* falls: 8 of law E's partitions keep the Bayes accuracy for S while AUC\* falls (10 keep it in all).
  - Equal finite-sample AUC estimates, or equal AUCs for a restricted reader class, do not license it.
  - The verifier proved (a)–(d) independently (`roles/correction_v1/verifier/VERIFICATION.md` §3). In check E, equality held exactly for the posterior-constant partitions, for all 52.
  - The mathematical reviewer adds a quantitative corollary for a deterministic R = f(X) (`roles/correction_v1/math_reviewer/MATH_REVIEW.md` §1(d)). Let M be the inputs whose populated fibre contains an input with a different posterior. Then 0 ≤ AUC\*(S; X) − AUC\*(S; R) ≤ P(X ∈ M | S = 1) + P(X ∈ M | S = 0).
    - Near-equality is guaranteed when the inputs in fibres that mix posteriors have small **population** mass. A sample collision count does not measure that mass. Large merged mass can also cost little when the merged posteriors are close.
- **(e) Determinism and continuity do not imply injectivity:**
  - tree ensembles are piecewise constant;
  - on a continuum of inputs, a continuous map from more input dimensions to fewer output dimensions cannot be injective. This matters for the continuous synthetic laws, such as the 10-field probe; the ACS inputs themselves are discrete;
  - pooled encodings merge raw inputs;
  - serialised precision can merge distinct real outputs;
  - a deterministic map can still merge cells.

  Conversely, merging inputs is not protection by itself: a non-injective map loses nothing when the merged cells share their SEX posterior (check D).

## 2. What the record shows about injectivity, level by level

| Level | What is established | What is unknown | Implication |
|---|---|---|---|
| Population law | AUC\* is defined under "the population law" (`PROBLEM_SPECIFICATION.md` §3) | Whether that means the 74,091 eligible people or a superpopulation; its support and posterior. Under a finite-population law, an input value held by one person has posterior 0 or 1, which inflates AUC\*(S; X) | Every "almost surely" is relative to this law. The record contains no estimate of AUC\*(S; X) at any level |
| Six raw fields | Discrete: AGEP is an integer in 25–64, WKHP an integer in 1–99, and COW, SCHL, MAR and POBP are codes (`acs_replication/schema.py:37, 39, 43, 55`) | AUC\*(S; X_raw) | "Continuous" does not apply to the inputs. Any statement is about a finite support |
| Encoded input (145 columns) | One encoder, fitted on MODEL_FIT and used by every evaluated model (`acs_replication/extract.py:109–133, 202, 210`; `acs_replication/data.py:56–71`; `acs_replication/deploy.py:29–41`). The workers load it at `stage_a.py:40–42`, `stage_b.py:43–48`, `fare_worker.py:42–43`, `fagtb_worker.py:43–44`, `neural_raw_worker.py:45–53` and `acs_hybrids/common.py:112`. Levels seen fewer than 10 times in MODEL_FIT go to one OTHER column per field (`acs_replication/schema.py:59`). Kept levels: COW 8, MAR 5, POBP 105, SCHL 21. SCHL codes 02, 03 and 05 share the OTHER column; input width 145 (`results/pcrl_acs_protected_tree_replication_v1/EXTRACTION_SUMMARY.json`, keys `category_levels_kept`, `feature_names`, `input_width`). AGEP and WKHP are z-scored, which is one-to-one on their integer grids. Stored as float32 | Whether two pooled levels co-occur with the same other five values in the population, and whether pooling loses SEX information | No model reading this encoding is one-to-one in the raw fields on the declared domain. The size of any AUC\* effect is unknown |
| One recipient's output | A deterministic function of the encoded input. The tree arms (T_TREE, TASK_REFIT, LOCAL_TREE, JOINT_TREE, the hybrids' tree parts, MMP and its twin) and FAGTB are piecewise constant, so not continuous (`acs_replication/trees.py:85–99`; `acs_replication/stage_b.py:1–9`). FARE takes at most as many values as it has cells (`acs_replication/fare_worker.py:63–69`). U0, Ucal and NEURAL_RAW are ReLU networks (`acs_replication/neural.py:21–24`) | Injectivity or sufficiency on the population support, for every ACS model (U0, Ucal, T_TREE, TASK_REFIT, LOCAL_TREE, JOINT_TREE, NEURAL_RAW, H_LJ) | AUC\*(S; R_i) ≤ AUC\*(S; X). Equality is UNKNOWN |
| Linked pair | A function of the encoded input. The pair can separate inputs that neither output separates alone | As above | AUC\*(S; V) ≤ AUC\*(S; X). Equality is UNKNOWN |
| A finite observed sample | Nothing about real ACS releases: this campaign opened no real array. The rehearsal counts (799 distinct encoded inputs, 797–799 distinct release pairs) come from **800 invented MODEL_SELECT rows**. They were compared bitwise. MMP had 2 and 1 fewer distinct pairs than inputs, so at least one merge per seed; the twin had none | — | A sample count cannot certify population injectivity (check C) |
| Declared domain vs population support | The declared domain is the product of the field domains; the population fills only part of it | The population support | Injectivity, if established at all, must hold almost surely under the population law |
| Real numbers vs stored precision | Inputs are float32 (`extract.py:133`). Tree routing compares float32 values with float32 borders, then sums float64 leaf values. Network forward passes run in float32 and are cast to float64 (`acs_replication/trees.py:57–59, 83–114`; `acs_replication/neural.py:72, 86`) | Collisions at transmitted precision | The optimal attacker sees the transmitted values. Arguments about real numbers do not carry over automatically |

**Existing proof of population injectivity for any actual model: none.**
- D3 (`prototype/DEMONSTRATIONS.json`) is a constructed injective finite law. It illustrates statement (b); it is not evidence about ACS.
- The adversary reviewer had flagged at the time that tree injectivity was unverified (`roles/adversary_reviewer/PROPOSAL_THREAT_ASSESSMENT.md:69–70`).
- The campaign's own records contain deterministic releases whose optimal AUC lies below AUC\*(S; X):
  - MMP in L_CONT_PROXY: Bayes AUC 0.7216 against 0.7878;
  - MMP's release in the verifier's 10-field probe P7: exact Bayes values 0.590–0.635 against AUC\*(S; X) = 0.6996. The twin's Bayes value was not computed.

## 3. Correction table

Classifications: **correct**, **conditional** (true under a stated premise), **unsupported** (not shown by the record), **false as stated**, **unresolved**.

| # | Original claim and location | Class | Replacement wording | Evidence or proof | Effect on recommendation |
|---|---|---|---|---|---|
| 1 | "Every evaluated ACS release is continuous and deterministic, hence almost surely injective on the populated inputs. Its population Bayes recovery is therefore AUC\*(S; X) whatever the defense." (`DECISION_LOG.md` 05:36Z entry, line 93) | **false as stated** (the implication), **unsupported** (the conclusion) | "Every evaluated ACS release is a deterministic function of the same pooled encoding E(X) of the six fields. The tree arms and FARE are piecewise constant. Its population Bayes recovery is at most AUC\*(S; E(X)) ≤ AUC\*(S; X), AUC\*(S; R) = AUC\*(S; E(X)) exactly when P(S \| E(X)) is almost surely a function of R (likewise with X for the second inequality). That has not been checked for any defense. No release has been shown to be one-to-one on the population, and none has been shown not to be." | §1 (a)–(e); §2; checks A–E | Removes the stated reason for the contract-change recommendation. The status is unchanged (§7) |
| 2 | "The measured differences between defenses (0.60–0.66) are differences in what a fixed slate of readers learns from 8,001 labelled releases." (same entry); "All measured differences between defenses, the neural control included, are differences in what a fixed slate of readers learns" (`START_HERE.md:48`); "Measured protection is learnability" (`ADVISOR_BRIEF.md:24–25`) | **conditional** (true if and only if the compared releases have the same optimal AUC; keeping P(S \| X) is sufficient. Neither is known) | "Measured differences are differences in what the declared reader bank recovers with the declared auxiliary sample. How much of each difference reflects information removed and how much reflects what readers fail to learn is unknown." | §1 (c)–(d); §2. The measured recovery splits into the optimal recovery, minus the gap of the selected reader (reader class plus finite sample), plus DEV_REPORT sampling error (`MATH_REVIEW.md` §3B). Nothing measures the first term's differences | The 0.020 margin remains a statement about the declared attacks |
| 3 | "(The verifier's probes confirm this independently.)" (`ADVISOR_BRIEF.md:24`) | **false as stated** | Delete. In the verifier's probe P7, MMP's releases (three draws) were non-injective, with exact Bayes AUC below AUC\*(S; X). Its injective law L_CONT_NOPROXY was constructed injective | `roles/math_verifier/PROTOTYPE_VERIFICATION.md` §3–§4 | None |
| 4 | "Under the deterministic contract, every evaluated release is essentially injective … AUC\*(S; V) = AUC\*(S; X). `prototype/DEMONSTRATIONS.json` D3 shows this." (`START_HERE.md:46–47`) | **unsupported** (injectivity); **false as stated** ("D3 shows this") | "D3 shows, on a constructed law, that an injective release keeps the optimal AUC. Whether the actual ACS releases are injective, or keep P(S \| X), is unknown." | §2 | None |
| 5 | "Under the current contract no deterministic release can beat the raw-field leakage against an optimal attacker" (`ADVISOR_BRIEF.md:56`) | **false as stated** | "A deterministic many-to-one release can lower the optimal attacker's recovery below the raw-field level while staying within the output contract. Whether any such release meets this benchmark's caps on the Florida data is an open feasibility question." | Check B (AUC\* 1 → 1/2 at full-input Bayes utility); MMP in L_PROXY, L_MIXED and L_CONT_PROXY; P7 draw 0 | The recommendation no longer rests on an impossibility |
| 6 | Option 2, "a contract change that makes attacker-independent protection possible: coarsened or randomised outputs" (`ADVISOR_BRIEF.md:61`); "coarsened outputs" listed as a contract change (`PROBLEM_SPECIFICATION.md:125`) | **false as stated** (a contract change is not necessary). The specification's §7 is **internally inconsistent** with its §2: it is right only for outputs outside the probability simplex, such as cell labels, set-valued outputs or abstention symbols. FARE and the tree arms are already probability releases that take finitely many or piecewise-constant values | "Deterministic coarsened probability outputs are inside the primary contract (`PROBLEM_SPECIFICATION.md` §2: any deterministic map of the fields into each task's simplex, plus its argmax). Randomised outputs, shared randomness or non-simplex symbols would be declared contract changes (§7)." | Specification §2 vs §7; check B | Option 2 is reframed in the advisor brief (§7 below) |
| 7 | "Reader-independent protection requires merging input cells, which the closed lanes found too costly at the caps." (`START_HERE.md:49`); "Earlier campaigns found that too costly at the occupation-confidence and income-Brier caps." (`ADVISOR_BRIEF.md:26`) | First half **correct** for deterministic releases, if read as "merging inputs with different SEX posteriors". Second half **unsupported** as a general limit | "Lowering the optimal recovery with a deterministic release requires merging inputs that differ in their SEX posterior. Earlier campaigns tried specific merges (decision-preserving codes, capacity-limited codes, 10-bit codes, mostly on Adult), and those failed the occupation-confidence or log-loss caps. That rules out those mechanisms, not every utility-feasible merge." | §1 (d); `CLOSED_LANES.md` rows "Latent bottlenecks / codes" | There is no universal privacy–utility limit; feasibility on Florida is open |
| 8 | "The synthetic pilot rehearsal … also showed fitted SEX-aware CatBoost releases to be essentially injective: 797–798 release pairs for 799 distinct inputs on stand-ins." (`DECISION_LOG.md:145`); the same reading at `prototype/PROTOTYPE_REPORT.md:95`, `METHOD_SPECIFICATION.md:143` ("injective in practice") and `PILOT_PROTOCOL_DRAFT.md:5, 79` | **unsupported** as a statement about any population, and **incomplete** as a description | "In the synthetic rehearsal (800 invented MODEL_SELECT rows, 799 distinct encoded inputs, compared bitwise), the MMP release pair took 797 and 798 distinct values on the two seeds and the twin took 799. This describes that sample only." | `pilot/PILOT_REHEARSAL_RESULTS.json` (banner, role_sizes, `stage0.per_seed.{0,1}.merge_diagnostic_mmp` and `merge_diagnostic_twin`); check C | The pilot's D0 count is relabelled as descriptive only |
| 9 | "Its learnability leg, needed wherever fitted teachers do not merge cells, which is the realistic case" (`DECISION_LOG.md:143`) | **unsupported** ("which is the realistic case") | "…needed wherever fitted teachers do not merge cells with different SEX posteriors. Whether that is the case for real data is unknown." | §2 | None |
| 10 | "Marginalising over SEX removes information only along pure-proxy directions." (`ADVISOR_BRIEF.md:28`); "only along directions where q_i^s(x) is constant, i.e. pure SEX proxies … everywhere else … m̂ is essentially injective" (`METHOD_SPECIFICATION.md:38–42`) | **false as stated** | "MMP lowers optimal recovery only where its release merges inputs that differ in SEX posterior. That happens along pure-proxy fields; more generally, along input directions on which the within-SEX task predictions do not depend; and wherever the mixture itself coincides for different pairs of components. On a continuum of inputs such directions exist whenever the inputs have more dimensions than the task-relevant part, even if every field matters for the tasks. On the discrete ACS support, dimension counting proves nothing either way. Where deleting the inferred-SEX term unmasks other SEX signal, or separates inputs the SEX-blind posterior had merged, MMP raises recovery." | `PROTOTYPE_VERIFICATION.md` §3 (P7, as twin − MMP: +0.033 / −0.018 / −0.019); `MATH_ASSESSMENT.md` M10. The same narrow reading appears at `prototype/PROTOTYPE_REPORT.md:48` ("i.e. along pure-proxy directions", correct for the finite laws it describes) | None. It strengthens the decline |
| 11 | "Its utility cost is known exactly in advance" (`ADVISOR_BRIEF.md:10`); "at an exactly known price" (`METHOD_SPECIFICATION.md:158`); "prices the deletion exactly" (`SELECTION_DECISION.md:51`) | **conditional** (exact only under a known law and relative to the SEX-blind posterior η) | "Under a known law its price relative to the SEX-blind posterior is an exact formula. For fitted models on real data the cost is unknown in advance: the conditionals are estimated, sex-atypical cells are extrapolated, NLL is clipped, the binding references are the fitted U0 and Ucal rather than η, and finite-sample uncertainty is separate." | §4 | None |
| 12 | "each model's 'if male' and 'if female' predictions" (`ADVISOR_BRIEF.md:10`); "the counterfactual law P̃" (`METHOD_SPECIFICATION.md:27`) | **unsupported** (implies individual counterfactuals or a causal intervention) | "the fitted model's predictions with its SEX input set to male and to female: evaluations of a fitted conditional model, not individual counterfactuals". P̃ is "the hypothetical product law P(x)P(s)P(y \| x, s)", a mathematical construct | §4 | None |
| 13 | "Cancelling SEX signal with a propensity-driven shift (RCCS) hides it from linear and quadratic attackers … and it never lowers the population-optimal attacker's recovery" (`ADVISOR_BRIEF.md:29`); "Its Bayes effect is ≥ 0" (`DECISION_LOG.md:161`); "its protection only works against the linear and quadratic attackers it optimises against" (the coordinator's closing message of the campaign session; not archived in the repository) | **false as stated** ("only"); **conditional** ("never lowers") | See §5 | `MATH_ASSESSMENT.md` 2.3, 2.7, 2.8; `DECISION_LOG.md` 06:12Z note | RCCS stays rejected as presented; a repaired RCCS is unresolved |
| 14 | RCCS fails S6, "protection defined against its own linear and quadratic critic class" (`SELECTION_DECISION.md:26, 35`; `DECISION_LOG.md:112`) | **unresolved** (law-dependent; the decision record is preserved) | "RCCS's objective controls only linear and quadratic critics. Whether its fitted protection extends to other readers is law-dependent: it did in two smooth Gaussian laws, and with a categorical driver an optimal reader recovers 21% to over 100% of the cancelled signal. Discrete-field laws and real data are untested." | §5 | None. S5 fails as presented, so under rule 1 MMP was the only eligible proposal. Even if both had been eligible, rule 2 would favour MMP (3 SERIOUS objections against 4) |
| 15 | Historical role statements: "A fitted CatBoost m̂ is constant only along columns no tree splits on" and "AGEP and WKHP are continuous" (`roles/math_verifier/MATH_ASSESSMENT.md:73`; **false as stated**: m̂ is constant on every cell of its split partition, and both fields are integers); "m̂ is essentially injective on the populated support" (same line); R7 "TRUE but incomplete" (`:189`); designer 2's "injective on the input support … equally true of NEURAL_RAW and the trees" (`roles/designer_2/PROPOSAL.md:52`); designer 1's "generic smooth … essentially injective" (`roles/designer_1/PROPOSAL.md:75`). Also: "The real-data effect MMP can deliver is therefore a fitted-reader learnability effect" (`MATH_ASSESSMENT.md:74`) and the "Bayes reduction only along exactly merged directions (pure proxies). Otherwise fitted-reader learnability" summary (`:34`); "Everywhere else … learnability" (`roles/adversary_reviewer/PROPOSAL_THREAT_ASSESSMENT.md:22`); "genuine where cells merge, learnability elsewhere" (`DECISION_LOG.md:100`); "the deployed release, which is injective" (`roles/designer_1/CROSS_CRITIQUE.md:13`) | **unsupported** for the actual models (designer 1's wording is **conditional** on genericity, which fitted trees do not have) | Not edited (role records). This row is the correction | §1–§2 | None |
| 16 | Adversary motivation: the exclusion rationale ("would measure a breach of the provider"), "without which no attacker can be calibrated", "a per-recipient policy cannot forbid", "None knows a defense's parameters or training objective" (`ADVERSARY_AND_MOTIVATION.md:19, 28, 32, 88`; `ADVISOR_BRIEF.md:45–50`; `MANUSCRIPT_ADVERSARY_PATCH.diff:55`) | See §6 | See §6 | `roles/correction_v1/adversary_reviewer/ADVERSARY_REVIEW.md` | The manuscript patch is replaced (v2); the access definition itself is unchanged |
| 17 | Fixed-reader recovery as protection evidence (all summaries) | **correct** as scoped | Fixed-reader results are empirical evidence against the declared attacks. They do not establish protection against arbitrary attackers, independently of the injectivity issue | `PROBLEM_SPECIFICATION.md` §3, §9.4 | None |
| 18 | `PROBLEM_SPECIFICATION.md` §3: "For a deterministic mechanism, AUC\*(S; V) ≤ AUC\*(S; X), by data processing" | **correct** | Unchanged. It also holds for randomised mechanisms whose randomness is independent of S given X; it is a bound, which the later equality claims ignored | §1 (a) | None |
| 19 | "The rationale is contradicted where it matters for real data" (`METHOD_SPECIFICATION.md:154`) | **unsupported** | "The rationale's learnability prediction failed in the pre-registered synthetic law, and the verifier's probe showed that MMP's effect can have either sign. Whether the Florida data have the merging structure in which MMP removes information is unknown." | `MATH_REVIEW.md` §3A (A20), §6 | None. The rule-4 outcome still follows |
| 20 | "…contradicted the part of its rationale that a real-data benefit needs / would need" (`START_HERE.md:5`, `ADVISOR_BRIEF.md:8`, `SELECTION_DECISION.md:70`, `prototype/PROTOTYPE_REPORT.md:95`, `HANDOFF.json` final_status) | **unsupported** (it rested on the withdrawn injectivity reading) | "Its pre-registered prototype contradicted the learnability leg of its rationale in a synthetic law, and the verifier's probe showed its effect can go either way. A real-data benefit would need either merging structure in the Florida fields or a learnability effect; which applies, if either, is unknown." | Rows 1, 8, 19; `MATH_REVIEW.md` §6 | None. The rule-4 outcome still follows from check (a) |
| 21 | "Without merging structure, MMP's effect … has no definite sign" (`START_HERE.md:6`, `METHOD_SPECIFICATION.md:154`, `SELECTION_DECISION.md:72`, `prototype/PROTOTYPE_REPORT.md:125` ("without such structure"), `PILOT_PROTOCOL_DRAFT.md:7`) | **false as stated** (P7's releases do merge, through dimension reduction) | "Without pure-proxy fields (all ten fields task-relevant), MMP's net effect relative to its SEX-blind twin had no definite sign." | `PROTOTYPE_VERIFICATION.md` §3 ("merging through dimension reduction is real") | None |
| 22 | RCCS's fitted readers "fell 0.019–0.045" (`MATH_ASSESSMENT.md` 2.7 and line 34; `DECISION_LOG.md:152`; `START_HERE.md:69`) | **false as stated** (a rounding slip) | "fell by 0.018–0.045". The smallest drop is 0.01827 (HGB, 100k rows, g = 0.35, B = 0.005) in `roles/math_verifier/V_RCCS_MAP_V1.json` | The verifier's direct lookup of the 28 stored differences (phase 2) | None |

**Files that needed no change:**
- `CLOSED_LANES.md` scopes every statement to the lanes' own reports.
- `CLOSEST_PRIOR_WORK.md` cites the adjustment formula as prior art only, and already warns against the Stadler over-application.
- `EVIDENCE_PACK.md` holds published numbers only.
- `PROBLEM_SPECIFICATION.md` is the campaign's pre-proposal problem statement. Rows 6 and 18 correct or confirm it here instead of editing it.

## 4. MMP: what holds and what was overstated

**Population identities.** These are correct, with explicit assumptions:
- q^s = P(Y | X, S = s), π = P(S = 1 | X), η = P(Y | X) = (1 − π) q^0 + π q^1, and m = (1 − w) q^0 + w q^1 for a fixed w (π̄ registered).
- Under the known law, with expectations over the same law that defines η:
  - summed Brier(m) − Brier(η) = E‖η − m‖² = E[(π − w)² ‖q^1 − q^0‖²];
  - NLL(m) − NLL(η) = E KL(η ‖ m).

  The second needs m_y > 0 wherever η_y > 0, which holds for 0 < w < 1.
- For any two predictors f and r that are functions of X, Brier(f) − Brier(r) = E‖f − η‖² − E‖r − η‖². So an excess against U0 or Ucal also depends on those references' own errors, which are unknown.

**Exact price vs real cost.** "Known exactly in advance" holds only for the known-law price relative to η (row 11). For a future fitted model, the cost against the binding references must be estimated, with finite-sample uncertainty reported separately. The verifier gives an illustration on its own small law (post-hoc): plugging fitted components into the closed form gave 0.02121, while the true Brier excess of the fitted release was 0.02541.

**Wording.** "If male / if female" means evaluating a fitted conditional model with its SEX input switched (row 12).
- A causal or individual-counterfactual reading would need a causal graph, identification assumptions and support (positivity) conditions.
- The resemblance to the adjustment formula gives an interventional reading only under a back-door condition that is not assumed here.

**Where information is removed.** See row 10. The observed effects stay as observed, and are not generalised:
- MMP lowered optimal recovery where it merged inputs (Bayes):
  - L_PROXY 0.767 → 0.500;
  - L_MIXED 0.717 → 0.650;
  - L_CONT_PROXY 0.788 → 0.722.
- It did nothing in L_NOPROXY (Bayes) and in L_CONT_NOPROXY (fitted pair AUC twin 0.7211, MMP 0.7207; the release there is injective by construction).
- It raised fitted and near-Bayes recovery in two of three draws of the verifier's probe P7 and lowered it in one (twin − MMP: −0.018, −0.019 and +0.033).
- These are statements about those laws. Three draws do not show that MMP generally fails or generally succeeds.

## 5. RCCS: reconciled with the independent review

| Question | What the record shows |
|---|---|
| Attacker class its objective controls | Linear and quadratic critics only: a closed-form ridge R² criterion (`MATH_ASSESSMENT.md` 2.1, 2.3) |
| Population-optimal effect in the specified laws | **Rose** every time: 0.615 → 0.644 in designer 2's finite law, and +0.021 to +0.053 in the verifier's smooth laws. The map injects the propensity residual. When the shifted release still determines the base pair, its optimal recovery cannot fall below the base's. "Never lowers" holds under that condition, not unconditionally |
| Fitted readers | In two smooth Gaussian laws, linear, quadratic, HGB (8k and 100k training rows) and MLP (8k) readers **all fell** by 0.018–0.045 (the source text says 0.019; its data, `V_RCCS_MAP_V1.json`, give a smallest drop of 0.01827), within all three binding clauses at B = 0.005 and within cap + headroom at B = 0.009. HGB and MLP fell at least as much as the linear readers (`MATH_ASSESSMENT.md` 2.7) |
| Beyond the matched local-targeting comparator (MIN) | Tied for linear readers (\|Δ\| ≤ 0.0012). **0.008–0.022 lower** for HGB and MLP pair readers. That is not the linear signature the proposal registered, and its own falsifier 1 would have rejected it there (2.7, 2.8) |
| Smooth laws vs discrete fields vs real data | Smooth Gaussian laws only. With a categorical driver, designer 1's exact check shows an optimal reader recovers 21% to over 100% of the cancelled signal. Discrete-field laws for the literal map and real data are untested |

**Adverse evidence retained:**
- inconsistent supporting accounting, which shrinks the RCCS-versus-MIN margin from up to 0.11 to 0.002–0.031, and a LOCAL replay that fails;
- a published-AUC reading that is model-dependent and non-specific;
- its own finite-law "feasible case" fails the binding clauses (the first reduction costs 3.75 accuracy points);
- falsifiers that are not decisive in either direction;
- population recovery rising in every tested law;
- no real-data validation.

**Conclusion.**
- The corrected record still supports **rejecting RCCS as presented** at selection: S5 fails as presented, so under rule 1 MMP was the only eligible proposal. Even if both had been eligible, rule 2 would favour MMP (3 SERIOUS objections against 4).
- The S6 judgment that its rationale "only holds against the training critic" is corrected to **unresolved (law-dependent)**.
- Whether a repaired RCCS would survive is unresolved. A repair would need consistent accounting, a literal-map feasible case within the binding clauses with discrete SEX-only fields, and decisive nonlinear falsifiers.
- RCCS is not fitted, changed or promoted here, and no method win follows.

## 6. Adversary motivation

**Kept unchanged: the concrete access definition.**
- Outputs only, for one recipient or a perfectly linked pair.
- A fixed SEX-labelled auxiliary sample from the same population (ACS: 8,001 people to fit, 3,000 to select and orient).
- Readers fitted on ATTACK_FIT and selected and oriented on ATTACK_VALIDATION.
- One fixed release per person.
- No target raw inputs, no target labels, no parameter- or gradient-specific attacks, and no interactive queries.
- Linkage, and the auxiliary data's availability, size and distributional match, are explicit assumptions.
- Identifiers are never reader features.
- The illustrative income/occupation scenario is still labelled as not a documented deployment.

**What was corrected.** The full 42-row table is in `roles/correction_v1/adversary_reviewer/ADVERSARY_REVIEW.md`.

| Original claim and location | Class | Replacement |
|---|---|---|
| Excluded access "would each measure either a breach of the provider or information already present in the inputs. A release policy controls neither." (`ADVERSARY_AND_MOTIVATION.md:32`; same reasoning at `:88` and `ADVISOR_BRIEF.md:50`) | **false as stated** | Target-input side information, public parameters, permitted repeated or chosen queries, objective- or parameter-aware attacks and insiders are distinct threat models outside this evaluation's scope. Several involve no breach: a recipient may legitimately hold a person's inputs, a model may be published, and a service may permit queries. Some are governed by release-policy choices (rate limits, one release per person, query auditing, whether to publish the model). They are untested, not dismissed |
| "Without them no reader can be calibrated" (`ADVERSARY_AND_MOTIVATION.md:28`); "the measuring instrument without which no attacker can be calibrated" (`ADVISOR_BRIEF.md:46`) | **false as stated** | Only this supervised reader bank uses the labelled sample, and its size is the declared data budget. Attacks built from a known population law or from analytic structure need no such sample; none was run |
| "Two capabilities the policy cannot take away" (`ADVERSARY_AND_MOTIVATION.md:19`); "pooling, something a per-recipient policy cannot forbid" (`ADVISOR_BRIEF.md:45`, `ADVERSARY_AND_MOTIVATION.md:88`) | **unsupported** | The evaluation grants linkage and does not assume that contracts or governance prevent it. Perfect linkage is the attacker-favourable limit. Pooling is not claimed to be unavoidable |
| "Perfect linkage is the attacker-favourable limit, so a failure to protect is not an artefact of linkage error" (`ADVERSARY_AND_MOTIVATION.md:27`) | **unsupported** (the reasoning runs backwards) | A low pair recovery cannot be blamed on poor linkage. A high one may overstate what a coalition with imperfect linkage would recover |
| "None knows a defense's parameters or training objective" (`MANUSCRIPT_ADVERSARY_PATCH.diff` v1, line 55) | **false as stated** | A recipient may know a defense's published algorithm and objective. Our attackers use no such knowledge beyond refitting to the defended releases, and none uses fitted parameters or gradients. Not implementing an objective-aware attack does not make recipients ignorant of the method |
| "The adversary is therefore drawn at the release boundary" (`ADVERSARY_AND_MOTIVATION.md:16`, `ADVISOR_BRIEF.md:43`) | **conditional** (a choice, not a consequence) | "We chose to evaluate an adversary that holds what is released" |
| A model "often releases" features and a prediction (manuscript abstract, `main.tex` line 29 at `3b5065805`; outside v1) | **unsupported** (no source; conflicts with the no-documented-deployment statement) | "may release". This is an optional, self-contained v2 hunk |
| An implied claim that this access is the only realistic threat | **removed** | The scope is stated positively (below) |

**Why this access, stated positively.** It isolates one narrow, well-posed question: what can be inferred about SEX from the delivered outputs alone, by one recipient or by a linked pair, under a declared labelled-data budget. The answer does not depend on assumptions about what else a recipient holds about the target. A stronger reader within the same access, including one that uses public knowledge of the algorithm and objective while seeing only the outputs, belongs to this threat model. That is why every recovery figure is a lower estimate, not a guarantee.

**Manuscript patch v2** (`MANUSCRIPT_ADVERSARY_PATCH.diff`, replacing v1).
- **Changes:** the Introduction paragraph, the "Adversary and access" paragraph and the Limitations sentence, plus the optional abstract hunk. Every other hunk is textually identical to v1.
- **Checks:** it applies cleanly to a fresh extraction of `3b5065805`, checked by the reviewer and independently by the coordinator. The result is `main.tex` `cf0b25f2…` and `references.bib` `b7043344…` (identical to v1's output; the patch still adds two bib entries to `3b5065805`). The reviewer's scratch build gives 11 pages with 0 LaTeX warnings.
- **For the manuscript owner:** the paper does not state the per-study `attacker_fit` / `attacker_val` sizes.

## 7. Status and recommendation

**Status: `NO_DEFENSIBLE_NEW_MECHANISM_IDENTIFIED` (for a real-data pilot), retained.** Under `roles/coordinator/SELECTION_CRITERIA.md` rule 4 it follows from the pre-registered learnability check (a) failing: L_CONT_NOPROXY, twin − MMP 0.0004 against ≥ 0.005.
- That was a low-power test, in a law with an injective release and readers within about 0.001 of Bayes.
- The verifier's probe P7 reinforces it: registered before it ran, post-hoc relative to the prototype pre-registration, and its sign was indeterminate (twin − MMP +0.033, −0.018, −0.019).
- The utility findings (men's within-SEX occupation increase at 2.2–2.6× the subgroup guard; the L_CONT_PROXY merge failing the strict caps) and the absence of novelty inform the label but are not rule-4 grounds.
- RCCS was rejected as presented under S5 and rule 1.

None of these depends on the withdrawn injectivity claim. What changes is the stated reason: it is no longer "real releases are injective". It is that the registered learnability prediction failed, the probe showed MMP's effect can go either way, and the real-data structure is unknown.

**The label, precisely.** The selection criteria define `NO_DEFENSIBLE_NEW_MECHANISM_IDENTIFIED` for the case where neither proposal is eligible (rule 3). The campaign applied it, with the qualifier "for a real-data pilot", to a rule-4 outcome. That is consistent in substance. The literal outcome is:
- MMP was selected but is not described as ready for a real-data pilot (rule 4);
- RCCS was rejected as presented (S5).

The qualified label is kept so that the record stays comparable. It must not be read as an impossibility result.

**Four statements that must not be conflated:**
- **No competitive method has been established** in this program. That remains true.
- **A candidate was declined under this campaign's criteria.** MMP was declined for a real-data pilot after its prototype; RCCS was rejected as presented at selection.
- **A specific hypothesis was contradicted under a specified synthetic law.** MMP's learnability leg failed in L_CONT_NOPROXY; "deleting Δ lowers recovery without pure proxies" failed in 2 of 3 P7 draws.
- **A universal impossibility claim would need a valid theorem.** None exists here. The injectivity-based impossibility is withdrawn, and check B refutes its general form.

**Recommendation, corrected.**
- The missing ingredient is still a decision about the target, but the choice is no longer forced.
- A declared attacker budget (reader bank, auxiliary sample size, learning curves) remains a reasonable and measurable target. It is a research choice, not a consequence of any impossibility.
- An information-removal target under the existing contract is not ruled out. Whether any deterministic many-to-one release meets the 24 caps on Florida is an open feasibility question that no study here has answered.
- Randomised releases would be a declared contract change.
- No real-data pilot is authorised by this correction. The prepared MMP pilot stays not recommended as a competitive test.

## 8. Exact checks (pre-registered) and their verification

- **Pre-registration:** `corrections/CHECKS_PREREGISTRATION.md`, committed at `b2958c76d` before any check code existed. Laws A–E and every expected outcome are fixed there.
- **Coordinator's run:** unit `CORR_EXACT_CHECKS_V1`, `corrections/EXACT_CHECKS.json`. **28 of 28 checks held,** including every hand-computed value.

| Check | Result (exact) |
|---|---|
| A, injective release | AUC\*(S; R) = AUC\*(S; X) = 3299/4606 ≈ 0.7162. A reader using the released probability as its score gets 1763/4606 ≈ 0.3828, and its reversal 2843/4606 ≈ 0.6172: both below AUC\* |
| B, useful deterministic removal | The release equals the full-input Bayes predictor in every cell (income accuracy 29/40, summed Brier 73/200; occupation 5/8, 53/100). AUC\*(S; X) = 1; AUC\* = 1/2 for each recipient and for the linked pair. Six inputs give three release pairs. I(S; X) = 0.971 bits; I(S; pair) = 0 |
| C1, sample uniqueness | Unique on the observed inputs {0, 2, 4} (the identity map passes too), 2-to-1 on the support, AUC\* 1091/1782 ≈ 0.612 against 487/594 ≈ 0.820 |
| C2, random samples | The closed form equals brute force at (3, 4) and (2, 5). A map merging all 2,000 cells in pairs shows no collision with probability 0.909 for n = 20 draws, 0.413 for n = 60, and 4.9 × 10⁻⁵ for n = 200 |
| D, merge without loss | 2-to-1 release, AUC\* = 4/5 = AUC\*(S; X) |
| E, exhaustive bound and equality | All 52 partitions ≤ AUC\*(S; X) = 569/833. Equality for exactly the two partitions with posterior-constant blocks (one injective, one not). The other 50 are strictly below |

- **Independent verification:** `roles/correction_v1/verifier/VERIFICATION.md` and `V_EXACT_CHECKS.json`, unit `V_CORR_EXACT_CHECKS_V1`, 2.3 CPU-seconds.
  - The verifier re-entered the laws by hand, wrote separate exact code, and opened the coordinator's results only after writing its own.
  - **All 16 registered expectations hold, and all 15 hand values agree exactly.** Every exact fraction matches the coordinator's, including the AUC\* of all 52 partitions. Logarithmic quantities differ by at most 2.2e-16.
  - C2 was confirmed by three further exact methods (a DP over draws, a Stirling-number sum and a DP over pairs) at all registered cases, including M = 1000.
  - The verifier's supplementary checks S1–S3 were not pre-registered and are disclosed as post-hoc: the extra C2 methods, the Bayes-accuracy count, and exact checks of the MMP identities on its own small law.
  - **What this validates.** The replay validates the arithmetic of the checks. The verifier separately assessed the mathematical statements in §1 and §4; its phase-2 review of the corrected summaries is summarised in §9.

## 9. Verification of this correction record

The independent verifier's phase 2 (`roles/correction_v1/verifier/VERIFICATION.md`, sections P2.1–P2.8) checked commit `717b566d8` against `12ffa8558`.
- **What it confirmed:**
  - the mathematics in §1–§2 and §4, apart from the items fixed below;
  - the status derivation from rules 1 and 4 without the withdrawn claim, and the four distinctions in §7;
  - all 47 line citations to the original files;
  - the quoted numbers, apart from the items fixed below;
  - the v2 patch (identical to the reviewer's file; applies cleanly at `3b5065805` with the stated hashes);
  - that no changed file implies the pilot is authorised or makes a competitive claim;
  - that no changed file claims the ACS models are non-injective on the population (only the encoder's declared-domain pooling is claimed).
- **What it found.** 2 MUST-FIX, 15 SHOULD-FIX and 21 MINOR items. Every MUST-FIX and SHOULD-FIX item, and the MINOR items in the current summaries, were addressed in the commit that follows `717b566d8`:
  - **MUST-FIX:**
    - the headline reason for the status still echoed the withdrawn logic (rows 20 and 21);
    - §1(b) wrongly called declared-domain injectivity insufficient.
  - **SHOULD-FIX:**
    - "near-equality only if" (now "guaranteed when");
    - "only if one-to-one" in METHOD_SPECIFICATION §3 (now "if one-to-one, and in general exactly when it keeps P(S | task-relevant fields)");
    - "without merging structure" for P7 (row 21);
    - rule-4 grounds stated precisely, with check (a)'s low power;
    - rule 1, not rule 2, for RCCS;
    - the scope of what the exact checks back;
    - the unscoped merge sentence;
    - resource and encoder citations;
    - "both releases" in P7;
    - the 0.018–0.045 rounding slip (row 22);
    - the SERIOUS-objection count pointer;
    - the handoff hashes;
    - this section;
    - uncovered historical lines (rows 10 and 15).
- **Replay versus interpretation.**
  - Numerical replay validated the exact checks A–E, C2 (by four exact methods), the verifier's supplementary checks and the patch hashes.
  - The following were validated by reading and argument only: the general theorems, the MMP identities in general, every ACS-specific fact (code reading), the status derivation, and the RCCS, prototype and P7 numbers (checked against their stored sources, not recomputed).
