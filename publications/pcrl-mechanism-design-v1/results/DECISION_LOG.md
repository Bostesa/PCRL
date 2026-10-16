# Decision log (coordinator-owned, append-only)

Times are UTC. Each entry records what was decided, by whom, and on what evidence. Earlier entries are never edited; corrections are new entries.

## 2026-10-16T04:35:52Z — campaign start
- First action: inspected worktrees, remotes and running processes from the frozen-hybrids worktree.
- Remotes: `github` = https://github.com/Bostesa/PCRL.git (the real GitHub); `origin` = the local mirror `<repository>/.` (a push there is not public registration).
- Running processes not owned by this campaign were left alone (an http.server, a `campus_lab.py web` process and a stale mock pipeline from another session).

## 2026-10-16T04:38Z — AGENTS.md
- No AGENTS.md (or CLAUDE.md) exists in the main checkout, in any branch's history (`git log --all --diff-filter=A`), or at home level. The binding instructions are therefore the campaign prompt and the repository's established practice (worktree per campaign, locks never edited, private data under `.private/`, sanitized publication branches).

## 2026-10-16T04:41Z — isolated worktree
- Created `<worktrees>/pcrl-mechanism-design-v1` on new branch `research/pcrl-mechanism-design-v1` from `54f089d3e` (frozen-hybrids research tip). Main, the source branches and every other worktree are untouched.

## 2026-10-16T04:42:51Z — ledger and data guard
- Campaign queue `mechanism_design/jobqueue.py` (copy of the frozen-hybrids queue): 8 CPU-hours, 6 elapsed hours from 04:35:52Z, 2 slots, one thread per job, 4 GiB combined RSS reservation, append-only ledger under `.private/mechanism_design/compute/`.
- `mechanism_design/no_real_data.py`: an audit hook refusing opens of earlier campaigns' private data, the private cache and backups, raw ACS files and array/model files outside this campaign's synthetic area. Self-test unit `SETUP_GUARD_SELFTEST_V1` passed.

## 2026-10-16T04:44Z — sources located
- Frozen-hybrid publication read from the local research branch `54f089d3e` (identical content to the public commit `3a1f509`): `START_HERE.md`, `ADVISOR_BRIEF.md`, `MANUSCRIPT_CLAIMS_HANDOFF.md`, `SERVICE_TABLE.csv`.
- Pinned source followed to the ACS replication (`research/pcrl-acs-protected-tree-replication-v1` @ `7eb9f2beb`, public `f4ea199`): `START_HERE.md`, `RESEARCH_DECISION.md`, `POPULATION_AND_SCHEMA.md`, `SCIENCE_LOCK.json` (release contract, claim rules, selection rule, model definitions), `SELECTION_DETAIL.json`.
- Original submitted papers found on disk as venue downloads: the NeurIPS submission "One Encoder, Many Purposes" (`<NeurIPS submission PDF>`, and an earlier download of the same submission number) and the AAAI submission "Outputs Leak What They Use" (`<AAAI submission PDF>`). Private review exports and the 2026-10-01 meeting transcript are in `<private review sources>` (private; never copied into public files; reviewer handles mapped to public pseudonyms there).
- The AAAI paper's code repository (`Bostesa/durable-guarantees`) has no local clone; it is public on GitHub and may be read from there.
- Combined empirical manuscript: `papers/combined_empirical_v1/main.tex` on `research/combined-analysis-paper-v1` @ `3b5065805` (latest manuscript `.tex` change in any branch since 2026-10-02). That branch is manuscript-owned and will not be modified; this campaign supplies a patch file instead.

## 2026-10-16T04:47Z — first agent wave
- Launched three agents in parallel:
  - a closed-lane evidence helper (read-only Explore agent; a coordinator aid, not one of the six roles);
  - the adversary and methodology reviewer;
  - the closest-prior-work reviewer (phase 1, literature base).
- Briefs are recorded in the agent prompts and summarised in `roles/ROLES_AND_OWNERSHIP.md`.

## 2026-10-16T04:50Z — problem fixed before any proposal
- Wrote `PROBLEM_SPECIFICATION.md`. Utility definitions are recovered from `acs_replication/data.py` (summed Brier, NLL with the 1e−12 floor, argmax accuracy, inclusive caps 0.010 / 0.010 / 0.005 vs U0 and Ucal, 24 clauses). Claim rules and the selection rule are recovered from the source SCIENCE_LOCK.
- Wrote `EVIDENCE_PACK.md` from published aggregates only (unit `COORD_EVIDENCE_SUMMARY_V1`). Rounding of the "pair minus best single" column was corrected from unrounded values before any designer saw the file.
- Wrote and self-tested a generic exact finite-law toolkit (`mechanism_design/prototype/finite_law.py`, unit `COORD_TOOLKIT_SELFTEST_V1`): exact Bayes AUC from a discrete view, mutual information, exact task metrics, posterior releases and set-partition enumeration. It is proposal-agnostic and contains no mechanism.

## 2026-10-16T04:55Z — designers launched in parallel, before CLOSED_LANES.md was finished
- Designers 1 and 2 were launched simultaneously with the identical brief `roles/coordinator/DESIGNER_BRIEF.md`. Each was told to read `CLOSED_LANES.md` once it appears and before finalising, so the ledger compilation does not serialise the campaign.
- **Disclosure: the launch messages differed by one sentence.** Designer 2's message added a nudge to "think from first principles about where the coalition's SEX signal lives … not only variations of the existing recipes". Designer 1's message stressed an evidence-backed reason why the neural control retains removable information. The difference was meant to diversify the two proposals. It is recorded here because the two inputs were not word-for-word identical.

## 2026-10-16T04:57Z — selection criteria fixed before any proposal exists
- `roles/coordinator/SELECTION_CRITERIA.md`, sha256 `330dfaf6224850c34df4383a49c0d1f114a273ef498450f5a62aa1b776a52765`. Both designer directories were empty at this time.
- The carried-forward frozen-hybrid conclusions were checked against files before quoting:
  - 24/24 point utility, with 10 PASS and 14 UNRESOLVED simultaneous clauses;
  - 0.022 / 0.024 benefit vs TASK_REFIT, with clause bounds 0.012 / 0.014;
  - pair AUC 0.641 / 0.643 vs NEURAL_RAW 0.640 / 0.638, with the nested 0.020 target an ESTABLISHED_SHORTFALL;
  - JOINT income fails utility;
  - the composition was post-hoc;
  - D2 applies;
  - the lane is closed.
- Sources: `results/pcrl_acs_frozen_hybrids_v1/START_HERE.md`, `PRIMARY_ENDPOINTS.csv` and `SERVICE_TABLE.csv`.

## 2026-10-16T05:07Z — CLOSED_LANES.md written; designers can now read it
- The closed-lane helper finished (read-only; summary documents only; about 20 minutes). Its report is saved, with paths made repository-relative, as `roles/coordinator/CLOSED_LANES_HELPER_REPORT.md`.
- The coordinator spot-checked six claims against the files before writing `CLOSED_LANES.md`:
  - the online-strength frontier quote and the allocation finding;
  - strength-matched REFRESHED − ONLINE −0.003 [−0.008, 0.003];
  - the minimal adapter's frozen-income sentence;
  - task-headroom 0.0505 [0.0257, 0.0754];
  - the separate-pool validation table;
  - confidence-capacity 0.034 and 0.0121.
- All six matched.

## 2026-10-16T05:28Z — adversary review received; designer 1 finished
- **Adversary and methodology reviewer finished.** Files are in `roles/adversary_reviewer/`; `PRIVATE_NOTES.md` is git-ignored there and stays out of every commit.
  - The coordinator re-checked three of its code claims at the cited lines: the NeurIPS cross-purpose attack's best-of-3 auditor seeds chosen by test accuracy; the AAAI `honest_stopped` rule on the representation only; race disallowed for all three HMDA purposes.
  - The coordinator also re-applied its manuscript patch to a fresh archive of `3b5065805`. It applies cleanly, and the output hashes match (`main.tex` fedd71ed…, `references.bib` b7043344…).
  - Curated into `ADVERSARY_AND_MOTIVATION.md` and `MANUSCRIPT_ADVERSARY_PATCH.diff`.
- **Designer 1 finished** (self-reported 05:26Z). Decision: `PROPOSAL`, "SEX-mixture marginalisation (MMP)".
  - `roles/designer_1/PROPOSAL.md` sha256 `2b7273ae01d6449036f25c7a6f5e74efb8d2843f1251cbceb8e4c7024c9ff368`, matching the designer's self-reported hash.
  - Its finite-law unit was `D1_MMP_LAWS_V2`. The designer discloses that V1 was superseded after law N was strengthened.
- The coordinator has not read designer 1's proposal text. It will not be shown to designer 2 or to the reviewers until designer 2's hash is logged.

## 2026-10-16T05:32Z — both proposals final; hashes logged before the reveal
- **Designer 1:** `roles/designer_1/PROPOSAL.md` sha256 `2b7273ae01d6449036f25c7a6f5e74efb8d2843f1251cbceb8e4c7024c9ff368`, finished 05:26Z. "SEX-mixture marginalisation (MMP)".
- **Designer 2:** `roles/designer_2/PROPOSAL.md` sha256 `83653f2db759bb8339edd96984ffa89683dc2f84478d37304b489894f4009f21`, finished 05:32:11Z. "Release-conditional counter-shift (RCCS)". This matches the designer's self-reported hash. Its synthetic units were `D2_GAUSS_V1` and `D2_FINITE_V1`–`V4`; the designer discloses that V1 was superseded after a float-rounding error.
- **Independence.** Each designer reports never opening the other's directory, and neither proposal was shown to anyone before this entry.
- **Closest-prior-work reviewer, phase 1, finished:** `roles/prior_work_reviewer/LITERATURE_BASE.md` and `README.md`, with 47 PDFs read in full text.
  - Key finding for the manuscript: the sentence attributing a label-implied disclosure claim to Stadler et al. over-applies their all-attribute theorem. A replacement sentence is provided.
  - `tian2025fairnessprivacy` is now published in IEEE TDSC, so the bib entry is outdated.
- **Next:**
  1. The mathematical verifier starts phase 1 on both proposals.
  2. The adversary reviewer and the prior-work reviewer receive both proposals for phase-2 assessments.
  3. Each designer receives the other's proposal for a short cross-critique, without rewriting its own.

## 2026-10-16T05:36Z — mechanism-independent demonstrations (expected outcomes written in the code before the first run)
- Unit `COORD_DEMONSTRATIONS_V1` ran `mechanism_design/prototype/demonstrations.py` and wrote `prototype/DEMONSTRATIONS.json`. All 10 checks passed, as expected:
  - **D1 (decision-preserving bound).** Every decision-preserving calibrated release has AUC* ≥ AUC*(S; d_ref). On law D1a the decision-preserving releases span 0.55 to 0.775 against the decision-only 0.55, and a decision-changing release reaches 0.5. On law D1b all decision-preserving releases equal the decision-only AUC.
  - **D2 (true labels are not a floor).** On law D2a, AUC*(S; X) = 1/2 for every release while the label-only AUC is 0.70. On law D2b the release reaches 0.825 against a label-only 0.52.
  - **D3 (fitted vs Bayes).** An injective release has AUC* = AUC*(S; X) = 0.789. Fitted logistic readers on that release average 0.566 (n_aux = 200) and 0.570 (n_aux = 5000).
- **Coordinator's reading.** Every evaluated ACS release is continuous and deterministic, hence almost surely injective on the populated inputs. Its population Bayes recovery is therefore AUC*(S; X) whatever the defense. The measured differences between defenses (0.60–0.66) are differences in what a fixed slate of readers learns from 8,001 labelled releases. Both designers state this limit themselves (designer 1 §3.5; designer 2 §3.5).

## 2026-10-16T05:50Z — cross-critiques and threat assessment received; dependency note
- **Designer cross-critiques** (`roles/designer_{1,2}/CROSS_CRITIQUE.md`; neither PROPOSAL.md was edited):
  - Designer 1 shows exactly (unit `D1_XC_DRIVER_V1`) that RCCS-style cancellation zeroes the linear and quadratic criteria while an optimal reader still recovers 21% to over 100% of the cancelled signal when the driver is categorical.
  - Designer 2 concedes that RCCS depends on reader limits more than stated, and that MMP has the cleaner removal interpretation and should be gated first. It proposes an MMP-income + RCCS-occupation combination. The coordinator does not treat that combination as a candidate, because it was formed after the reveal.
- **Adversary reviewer, phase 2** (`roles/adversary_reviewer/PROPOSAL_THREAT_ASSESSMENT.md`):
  - MMP keeps the contract, reads no SEX and fools no critic; it is "partly removal" (genuine where cells merge, learnability elsewhere).
  - RCCS is obfuscation: in its own exact law the deployed unpooled analogue raises Bayes pair recovery from 0.615 to 0.644.
  - Recommended pilot conditions: a label-free merge diagnostic, a learnability curve over auxiliary size, a symmetric nonlinear gate, single-recipient reporting, and an explicit 21-candidate pair bank.
- **Dependency.** CatBoost 1.2.8, the version pinned by the source registry, was not installed in the campaign interpreter (Python 3.13.7). The source used a private pylib under its own `.private/`, which this campaign does not open.
  - CatBoost 1.2.8 was installed from PyPI with `--no-deps` into `.private/mechanism_design/pylib`, for synthetic rehearsals only.
  - No real data is involved.

## 2026-10-16T06:04Z — SELECTION: MMP selected (conditional); RCCS rejected
- **Verifier phase 1** (`roles/math_verifier/MATH_ASSESSMENT.md`): both proposals `SOUND_WITH_FIXES`; no FATAL objection; MMP has 3 SERIOUS objections, RCCS 5. Every finite-law number in both proposals was reproduced with separate code.
- **Prior-work phase 2** (`roles/prior_work_reviewer/PROPOSAL_COMPARISON.md`): MMP is an established technique (Žliobaitė & Custers; Pearl's adjustment formula) applied to a new endpoint, with no new algorithmic ingredient. RCCS combines mostly established ingredients and overlaps closed lanes.
- **Decision.** Taken under the pre-registered rule (`SELECTION_CRITERIA.md`), objection by objection, with no vote; recorded in `SELECTION_DECISION.md`.
  - **MMP selected for prototyping**, as the one cheap, gated, critic-free test worth implementing next. This is not evidence of nearness to a competitive algorithm.
  - **RCCS rejected:** it fails S6 (protection defined against its own linear and quadratic critic class; the map injects SEX-propensity content; its Bayes recovery rises 0.615 → 0.644 in its own law) and S5 as presented (its feasible case violates the binding clauses; its Gaussian support is inconsistent).
- **Conditions adopted before any prototype run** (the verifier's fixes):
  - a law with no pure proxies;
  - a coalition required-disclosure law;
  - a matched-price local-masking comparison;
  - registered subgroup predictions;
  - the NLL approximation corrected in the specification;
  - w\* reported.
- Prototype pre-registration written as `prototype/PROTOTYPE_PREREGISTRATION.json`, with code hashes, before the first prototype run.

## 2026-10-16T06:12Z — prototype run once (COORD_PROTOTYPE_MMP_V1); pre-registered outcome: rationale partly contradicted
**Result.** 24 of 27 pre-registered checks held. The results are in `prototype/PROTOTYPE_RESULTS.json` (sha256 0ea7c567…).

**Held:**
- every exact identity;
- L_NEG: MMP infeasible; required disclosure kept the frontier at 0.772 at the cap;
- L_PROXY: MMP pair 1/2, feasible at the cap, equal to the global frontier;
- L_PROXY_FLIP: MMP fails accuracy;
- L_MIXED: MMP pair 0.65, feasible, equal to the frontier;
- L_NOPROXY: no Bayes effect, as expected;
- L_XOR: singles 1/2, pair 0.85 = AUC*(S;X);
- L_CONT_PROXY: fitted pair 0.779 → 0.723. This is the Bayes merge of the pure proxy; the occupation price exceeds the strict caps but sits within cap + headroom;
- all demonstrations.

**Failed:**
- **(a) Learnability leg.** L_CONT_NOPROXY fitted MMP below the twin by ≥ 0.005: observed 0.0004 (twin 0.7211, MMP 0.7207, Bayes 0.7222 for both), while MMP paid an occupation Brier price of 0.0050. Pre-registered as a contradiction of the rationale.
- **(b) Subgroup direction.** Women's occupation increase > men's in L_CONT: observed the reverse (men 0.0134 / 0.0115; women 0.0035 / −0.0023).
- **(c) Masking signature.** The matched-price local linear eraser had essentially no effect on any view, so the deletion-versus-masking comparison is **inconclusive**, not won.

**Coordinator's reading.**
- MMP's merging leg (exact removal along pure-proxy directions at the exact price) is confirmed.
- Its learnability leg, needed wherever fitted teachers do not merge cells, which is the realistic case, failed its pre-registered test.
- Per the brief, the candidate is therefore **not described as ready for a real-data pilot**.
- The synthetic pilot rehearsal (`COORD_PILOT_REHEARSAL_V1`) also showed fitted SEX-aware CatBoost releases to be essentially injective: 797–798 release pairs for 799 distinct inputs on stand-ins.
- Verifier phase 2 (independent recomputation and interpretation check) was requested.

## 2026-10-16T06:12Z — post-decision note on RCCS (evidence that arrived after the selection)
The verifier's synthetic test of the literal RCCS map (`V_RCCS_MAP_V1`) was not in `MATH_ASSESSMENT.md` when the selection was made: the section was still a placeholder. It reached the coordinator afterwards.

**What it found.** In a smooth Gaussian law with an oracle propensity:
- every reader, nonlinear ones included, fell 0.019–0.045 within the clauses;
- against MIN, linear readers tied and nonlinear readers favoured RCCS by 0.008–0.022;
- the verifier's final count is 4 SERIOUS objections for RCCS, not 5.

**Effect on the decision: none.**
- S5 still fails as presented: the feasible case violates the binding clauses, and the Gaussian support is inconsistent.
- The S6 concern is law-dependent rather than refuted. Designer 1's exact check shows linear and quadratic cancellation is complete only for location-family laws, as with a Gaussian driver; ACS propensities over categorical fields are lumpy.
- Under rule 2, MMP still has fewer unresolved SERIOUS objections (3 vs 4).

**For the record.** RCCS's synthetic success is a smooth-law learnability result. Its Bayes effect is ≥ 0.

## 2026-10-16T06:22Z — verifier phase 2 received; coordinator error disclosed; final status
- **Verifier phase 2** (`roles/math_verifier/PROTOTYPE_VERIFICATION.md`, separate code, predictions P1–P7 registered before its runs):
  - every prototype number reproduces exactly, and there is no bug in the coordinator's code;
  - check (a) is a genuine but low-power failure;
  - check (b) failed because of a mis-specified expectation from the verifier's own constant-gap formula. Men bear more because corr(π, ‖g‖²) < 0, and **men's within-SEX occupation increase is 2.2–2.6× the guard**.
- **Coordinator error, disclosed.** Check (c) used a mis-specified comparator: the matched-price local eraser (t < 1) is exactly invertible, so refitted readers undo it. Deletion versus masking at matched price is therefore untested. The pre-registration is not edited; this note records the error.
- **Registered post-hoc probe P7** (verifier): 10 task-relevant fields, 3-dimensional release. MMP's effect had no definite sign (+0.033, −0.018, −0.019, as twin − MMP) and raised recovery in 2 of 3 draws. This is not a reader artefact: a 1M-row reader was within 0.003 of exact Bayes. The merging case in L_CONT_PROXY fails all three occupation cap clauses at the strict cap.
- **Final campaign status: `NO_DEFENSIBLE_NEW_MECHANISM_IDENTIFIED` for a real-data pilot.** MMP was a design-stage selection that did not survive its prototype. The prepared pilot is delivered as code and protocol only, marked not recommended. No new method lane is opened.
