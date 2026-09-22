# Branch discovery: rebuttal and reviewer-response work across the PCRL refs

Audit date 2026-09-22. This pass was read-only: it made no checkout, commit, stash, reset, prune or edit in any worktree, and it wrote only this file and `REBUTTAL_WORK_INDEX.csv`. Every SHA below is the full 40-character value checked with `git rev-parse --verify <sha>^{commit}`. No ACS 2016 rows, predictions, labels or scores were opened. No review text is reproduced here; review-like files are identified only by path and type.

Target manuscript: `research/pcrl-submission-finish-v1` @ `30a6fd19e17453bc8c421a65491b5b482ab9291f`, `papers/pcrl_satml_final_v1/main.tex` (410 lines), with review notes in `results/pcrl_submission_review_v1/`. It is referred to below as **M**.

Original paper: "One Encoder, Many Purposes: Purpose-Conditioned Representation Learning with Per-Purpose Linear Leakage Bounds". The author's local copy of the submitted PDF (26 pages, created 2026-05-07) was used.
- §5.2 compliance: 56/60 cells pass strict R², of which 7 are cleanly compliant.
- §5.3 Dominant-Axis Auditing.
- §5.4 compliance via collapse.
- §5.5 cross-purpose concatenation attack: 26/33 cells above majority +1pp.
- §6 limitations, which include an Adult seed-3 held-out check.
- Propositions:
  - P1: Zhao–Gordon, restated.
  - P2: joint LEACE composition and optimality.
  - P3: Linear Compliance Bound, the R²-to-accuracy guarantee that has since been retired.
  - P4: LoRA Erasure Floor.
  - P5: convex-combination identity.
  - P6: cross-purpose bound kτ/λ₊(R), in Appendix C.

The submitted paper's §6 names its own follow-ups: a hard-R² LAFTR variant, held-out cross-validation on HMDA and Diabetes, a principled rank sweep, and a structural change that decouples erasure from representation geometry. The May 2026 rebuttal branches implement the first, third and fourth of these. The second was only registered, as FAccT Ablation 3, and never run.

## 1. Ref inventory (PCRL repository, common dir `<common git dir>`)

Dates are committer dates. "Anc(M)" means the ref tip is an ancestor of M, so its files already exist in M's tree. They are not necessarily cited by main.tex.

| ref | full SHA | date | Anc(M) | role / notes |
|---|---|---|---|---|
| claude/frosty-khayyam | 8fa4e900a85e6cf62b687b7f12c8962251aea142 | 2026-02-24 | yes | early pre-paper work; not rebuttal |
| overnight-bios-rank-mech-2026-05-04 (= origin) | 485734792fa626761451d01961797463ec21793b | 2026-05-04 | yes | pre-submission BIOS rank mechanism |
| laftr-benchmark-2026-05-05 (local) | e2f81eedd15be57389be50f83b6a2e996f3befb5 | 2026-05-05 | yes | local is 2 ahead of origin (135e440e6b16ca753465d0492193e33e63280dff anonymize + e2f81eedd15be57389be50f83b6a2e996f3befb5 variance-constrained script); pre-submission |
| origin/laftr-benchmark-2026-05-05 | ade5cd9f35dadec18e1bff76018167b1c8b3040a | 2026-05-05 | yes | ancestor of local; worker queue runner |
| bios-pcrl-layer12-2026-05-05 (= origin) | a967a4053464c238fe554b0c59331ab94ca8440a | 2026-05-06 | yes | pre-submission negative result (dropped from §5.6) |
| crosspurp-constraint-2026-05-06 (local) | ae63692cea9251d604e6722bce3c03380fb0b323 | 2026-05-07 | yes | local is 10 ahead of origin: 2026-05-07 cleanup, README, Prop-6 verifier (1ea908e981b6b6f000e69b4c3642cb31d151942b), paper-support artifacts (a3875c618962044b6de592511d96f196da84e92d) |
| origin/crosspurp-constraint-2026-05-06 | 93fdb9c26728b9c667797eca9780be605dbd3dd4 | 2026-05-06 | yes | ancestor of local; CROSSPURP eval scripts |
| erase-layer-pilot-2026-05-17 (= origin) | 65dd5c05063f01bcb9b0da7015595e1edc75dbaf | 2026-05-18 | yes | **NeurIPS rebuttal**: erase-layer architecture + aggregator |
| erase-layer-vicreg-sweep-2026-05-18 (= origin) | 39c5a84e69bcff54129ee2c16b17d36699dfef4d | 2026-05-30 | yes | **NeurIPS rebuttal**: VICReg sweep + per_dim_std diagnostic |
| diabetes-rank8-ablation-2026-05-30 (= origin) | 98c4d3ca2bee6e9d72a628b620b5883b08f4f43f | 2026-05-30 | **no** (1 ahead) | --lora-rank flag; patch-identical duplicate of 2787c83fc22b1b841b9d2e5715f9fd9f6e9e10a1, which is in M |
| rebuttal-evidence (= origin) | 5739d3bb94c7ab57afe190c8e2da704e1ee2dd5e | 2026-05-31 | yes | **NeurIPS rebuttal**: vicreg + diagnostic + rank-8 bundle |
| laftr-hard-r2-2026-05-17 (= origin) | 2037ad45287afcfa75a2be110b5ff84fc6b3c0e0 | 2026-05-31 | **no** (17 ahead) | **NeurIPS rebuttal**: hard-R² LAFTR baseline; code only, results uncommitted |
| main (local) | 0eee48f990949e522ecf1e3c251d0778e09b09ef | 2026-05-30 | **no** (19 ahead) | merges erase-pilot (rebased copies), vicreg sweep and laftr-hard-r2 up to 1c5e1aff4dfb0b6271ff52387fe01da863ef7762 |
| origin/main (= origin/HEAD) | 55e4cb1d1b603a52e5ae16fd5c71d41ebb9b3827 | 2026-08-19 | **no** (21 ahead) | local main + 962841995d6fedec255296437903730b4823f5ea and 55e4cb1d1b603a52e5ae16fd5c71d41ebb9b3827 (README de-submission) |
| cross-purpose-rebuttal-2026-05-18 (= origin) | d39211214314ec0027c345fafef035287fc98a2d | 2026-06-05 | **no** (8 ahead) | **NeurIPS rebuttal**: cross-purpose constraint on 3 datasets, 22/33 → 8/33 |
| ablations-facct-2026-07-24 (local, main worktree) | ad2c08872815185e4b63ae1d160e44f3aea1d5f4 | 2026-09-10 | yes | local is 5 **behind** origin; contains FAccT ablation registration (ecaeba46fcb921802087e792c5dbac2073e25347 to 15dbcc3c3338f6707e7a0b3901d738d99651a77d) |
| origin/ablations-facct-2026-07-24 | 349efa454afd907389760fd1f59fd8806a215efd | 2026-09-17 | yes | same tip as residual-spectral-20260910 |
| residual-spectral-20260910 | 349efa454afd907389760fd1f59fd8806a215efd | 2026-09-17 | yes | later redesign line |
| research/pcrl-evidence-paper-v1 (= origin) | 0d8f4b67b6d4961dfa133289d0167c874d2f4794 | 2026-09-17 | yes | evidence review v1 |
| research/pcrl-nonlinear-rank-v1 (= origin) | c37807e4f568ef38e5528fc09c1506083278bf4d | 2026-09-17 | yes | |
| research/pcrl-manuscript-integrated-v2 (= origin) | 3c6ada82e719489656da820b72ce8425d2079be0 | 2026-09-17 | yes | manuscript review v2 |
| research/pcrl-invariant-baselines-v1 (= origin) | 73903b7f28df68284285f0610a4036beb32b208f | 2026-09-18 | yes | |
| research/pcrl-manuscript-integrated-v3 (= origin) | ce5ba24a0a0ed1848c4029a922ac66fa629d71fc | 2026-09-18 | yes | |
| research/pcrl-direct-adversarial-v1 (= origin) | 106de9afa58cebbc26e34fb782e539e2a0881108 | 2026-09-18 | yes | |
| research/pcrl-manuscript-integrated-v4 (= origin) | 72a6383320e866cf5dbda1a8465c00c3fe068a3b | 2026-09-18 | yes | |
| research/pcrl-competitive-method-v1 (= origin) | 7f961d5c7f6f0562efcb25a27a77bb5221c279a7 | 2026-09-19 | yes | |
| research/pcrl-manuscript-integrated-v5 (= origin) | 1d18e898a552e920f3ccfc6b22e5f70e2ae734bb | 2026-09-19 | yes | |
| research/pcrl-utility-extension-aws-v1 (= origin) | 0517c06a79995dc574c857f890a2d6601e7d3c01 | 2026-09-20 | yes | |
| research/pcrl-manuscript-integrated-v6 (= origin) | eb4aa96685568426c098effb00b857f1ae934b0d | 2026-09-20 | yes | |
| research/pcrl-stochastic-channel-v1 (= origin) | cd895e4291b3dc2c0ed5e9c5482415d30f4e9f2a | 2026-09-20 | yes | |
| research/pcrl-manuscript-overnight-review-v1 (= origin) | 0176f149e91c02b8e2d202eb25ea9cba8ae019dc | 2026-09-21 | yes | user-pinned earlier review package |
| research/pcrl-stochastic-replacement-overnight-v1 (= origin) | e3415b94deb8d71c4870d4c392bb8ad4b464a847 | 2026-09-21 | yes | |
| research/pcrl-claims-foundation-v1 | f4bdf4cd5bf74c634feeec50aef78bff249667e4 | 2026-09-22 | yes | this audit's worktree (no remote) |
| research/pcrl-task-directed-release-v1 (= origin) | f4bdf4cd5bf74c634feeec50aef78bff249667e4 | 2026-09-22 | yes | holds ORIGINAL_WORK_HANDOFF.md |
| research/pcrl-submission-finish-v1 (= origin) | 30a6fd19e17453bc8c421a65491b5b482ab9291f | 2026-09-22 | = M | target manuscript |
| research/pcrl-final-prospective-v1 | df104bacd3f91c2d6abde119d5e6a50b30fe9aed | 2026-09-22 | no (descendant line) | not inspected: holds 2016 prospective material |
| refs/stash (stash@{0}) | 9e0b333daf3822c3c3044b4dc4e11a57bba96b67 | 2026-05-19 | no | "WIP NEXT_STEPS during cross-purpose branch setup" on erase-layer-vicreg-sweep; only change: +37/−1 lines of launch/monitoring notes in `results/rebuttal/erase_layer_pilot/NEXT_STEPS.md`. Index commit 892a9e77ac2bb9a588eb88b83b340919832cd269; untracked-files commit 350182059b5ce41749f7ce279585668e0164ef9e has the **empty tree** (4b825dc642cb6eb9a060e54bf8d69288fbee4904) |

There are no tags. Worktrees (from `git worktree list`), each at its branch tip above:
- `<main worktree>` (ablations-facct)
- `<superpowers worktrees>/laftr-hard-r2-2026-05-17`
- `<superpowers worktrees>/residual-spectral-20260910`
- `PCRL-t2-finish` (M)
- `PCRL-t2-overnight`
- `PCRL-terminal-1-{adversarial,competitive,invariant,replacement,stochastic,utility}`
- `PCRL-terminal-2-manuscript`
- `PCRL-terminal-3-claims`
- `PCRL-terminal-a`, `PCRL-terminal-b`
- `PCRL/.worktrees/{pcrl-final-prospective-v1,pcrl-task-directed-release-v1}`

A second, companion repository (another submission) was also inventoried. Its refs are listed only in the private handoff.

## 2. Search method and coverage limits

- `git log --all -i --grep` was run on the terms rebuttal, reviewer, review, NeurIPS, response, revision, dominant, "rare class", erasure, LoRA, collapse, theory, correction, certificate, composition, adapter, per_dim_std, cross-purpose, VICReg and rank. The two repos hold 329 unique commits. "rare class"/"rare-class" and "theory" match no commit message.
- File trees of every unique ref tip, plus the stash index and untracked commits, were scanned for paths matching rebuttal|review|headline|verdict|dropin|paper_paste|response|neurips|next_steps|mock. There were 260 unique paths. The hits under `results/rebuttal/**`, `results/reviewer_dropins/**` and `results/paper_critique_responses/**` were read. The September `results/pcrl_*review*` packages were grepped only.
- `git grep` was run for reviewer labels and OpenReview IDs across ref tips. `git merge-base --is-ancestor` and `git rev-list --count` gave ancestry against M. `git patch-id --stable` identified duplicates.
- Uncommitted work in other worktrees was read only through `git -C <wt> status --short [-uall]`. **Untracked file contents were not opened.**
- Author-local folders were searched by filename and content for venue-review markers. The details are in the private handoff.
- **Limits:**
  - rg does not look inside PDFs, so only named PDFs were opened.
  - Email, OpenReview itself, browser data and Claude session transcripts were not searched for review text. A browser session archive under `<private browser data> Support/Dia/` matched the OpenReview id string, but it is private browsing data and was not inspected.
  - Data in S3 (checkpoints, archives) was not checked for existence. Memory notes say the non-archive prefixes of bucket `pcrl-bios-overnight-20260504` expire after 7 days.
  - The recomputations reported below are light re-derivations of counts from committed JSON. No experiment was rerun.

## 3. Rebuttal work for the ORIGINAL NeurIPS paper

Scope note: `results/pcrl_submission_review_v1/REVIEW_TO_EVIDENCE.md` at M states that "`results/rebuttal/**` … [are] real rebuttal experiments, but addressed to blind-reviewer objections recorded for the AAAI paper". **That is incorrect.** Every directory under `results/rebuttal/` in the PCRL repo is work on the NeurIPS PCRL encoder paper:
- `erase_layer_pilot`
- `erase_layer_vicreg_sweep_aws`
- `erase_rank8_diabetes_cpu`
- `cross_purpose`
- `ablations_facct`

The AAAI rebuttal runs (fresh partition and FARE) live only in `<companion repository>` (§4). A related check: REVIEW_TO_EVIDENCE lists those two AAAI runs as being in "Appendix A" of the final paper, but main.tex contains neither (grep for FARE/fresh: no hits). The main.tex text is correct and the review note is wrong.

Where the "R1"/"R3"/"R1(c)" labels come from: they appear in May files (`results/rebuttal/cross_purpose/PLAN.md` "reviewer R3's strongest complaint", `docs/superpowers/plans/2026-05-18-laftr-hard-r2-pilot.md` "R1's complaint", and the rank-8 HEADLINE "R1(c)"). No file in any ref or home folder defines them. They predate the usual NeurIPS review release, so they may refer to internal or simulated critiques. The July FAccT registration uses different labels: anonymous reviewer handles and the AC (see §6; handles in the private handoff).

### Phase 0 (pre-deadline, 2026-05-05 to 05-07): critique-response work that shaped the submitted PDF

These answer numbered pre-submission critiques (#1 to #15), not venue reviews. They are all ancestors of M.
- `results/reviewer_dropins/*`: commits 72a2dc5238394d3e939bda72f5583bc341d0172d, a3875c618962044b6de592511d96f196da84e92d and 1ea908e981b6b6f000e69b4c3642cb31d151942b.
- `results/paper_critique_responses/PAPER_PASTE.md`: commit a3875c618962044b6de592511d96f196da84e92d.

Items:
- **Three-way compliance split** (clean / collapse / failed).
- **Term swap** from "certificate" to "bound".
- **Post-hoc tuning disclosure** (`PAPER_PASTE.md` lines 96–126). M uses it in Appendix A.
- **Adult seed-3 held-out check** (`PAPER_PASTE_heldout.md` + `heldout_s3_compare.json`): 6/8 vs 7/8. It appears in the submitted §6 but **not in M**.
- **Proposition 6 cross-purpose bound** (`prop6_bound_utilization/`). The bound holds on 18/18 Adult rows, with utilization 0.06–0.84 (my recount of `numerical_verification_v2.json`). The proof was marked "candidate; user review of proof required". It appears in the submitted PDF but **not in M**, and no September mathematical review covers it.
- **Proposition 5 (Zhao–Gordon)** table.
- **Tier-3 attempts**: Prop 7 shipped as "Tier 2.5", Prop 8 **retracted** 2026-05-07, TIER3 report abandoned.
- **Theory drop-in with the near-optimality certificate explicitly skipped** (`PAPER_PASTE_theory.md` lines 93–127). M uses it as Corrections item 4.
- **Adult-only cross-purpose constraint pilot** (`results/v2_adult_CROSSPURP/`, commit 5b0ec09b9005e0688f0768f29d47dac0670e348f): 14/15 → 12/15 flags at a per-pair cost of 23/24 → 19/24. It was a footnote in the submitted paper.

### Phase 1 (post-submission rebuttal, 2026-05-17 to 06-05)

**R-A. Erase-layer architecture ("§5.5 erase-layer for tabular").**
- Addresses: the submitted "compliance via collapse" and the strict-R² misses; the §6 follow-up "decouple concept erasure from representation geometry".
- Code: 96c03d01003a10ed1d797974d54d311b83b60456 on erase-layer-pilot. It adds a frozen joint-LEACE layer between the backbone and repr_proj, with LoRA on repr_proj only.
- Smoke: 04cb7efa31a7c1e77cc05325653662f6fbb1db38.
- Aggregator: 65dd5c05063f01bcb9b0da7015595e1edc75dbaf.
- Result: strict R² 54/60 → 60/60, eff_rank about 3.4 → 15.4, cleanly compliant 5/60 → **0/60**. The pilot improves strict R² and rank but loses every cell on the published per_dim_std ≥ 0.5 criterion.
- **Raw per-seed results are not committed in any ref.** They exist only as untracked files in `<main worktree>/results/rebuttal/erase_layer_pilot_aws/` (provisional). The committed λ=1.0 figures are re-quoted inside the VICReg HEADLINE.
- Caveat 1: the rebuttal baseline row (Round 5/7: 54/60 strict, 5/60 clean) does **not** match the submitted headline (56/60, 7/60). My recount of committed `results/v2_{adult,hmda}_ROUND5` and `v2_diabetes_ROUND7` per-seed `linear_r2` gives 21+16+17 = 54. This must be reconciled before any "before vs after" claim.
- Caveat 2: `run_v2_dataset.py` builds the StandardEncoder from `torch.manual_seed(seed)` and V2Trainer freezes it. The diagnostic script's docstring confirms it reconstructs the backbone bit-identically from the seed. With `lora_target=repr_proj_only`, the 128-d hidden features are therefore those of an untrained random network. This reading comes from the source; I did not run it.

**R-B. VICReg ×5 sweep and per_dim_std backbone diagnostic.**
- Addresses: whether the cleanly-compliant gap is caused by the erasure mechanism.
- Commits: d837248ba647cca8352802d76ec8c7bf8ed00e9e (flag), 67427e7bf0ac8d9372907011fb34e987275a003d (infra), 39c5a84e69bcff54129ee2c16b17d36699dfef4d (results).
- Files: `results/rebuttal/erase_layer_vicreg_sweep_aws/{HEADLINE.md,comparison_table.tex,diagnostic_perdim_std.json,v2_{adult,hmda}_ERASE_VICREG5/}`, `scripts/diagnose_perdim_std_floor.py`.
- My recount: Adult 24/24 (mean R² 0.0077), HMDA 18/18 (0.0056). The diagnostic maximum std is 0.289, below 0.5, in all 36 measurements.
- Diabetes λ=5 was cut off by the time cap and is absent.
- Overclaims to avoid, as ORIGINAL_WORK_HANDOFF already warns: "unreachable … by any mechanism", and the appeal to Proposition 4. P4 concerns LoRA doing the erasure, which is not the erase-layer setup.

**R-C. Diabetes LoRA rank-8 ablation.**
- Addresses: the justification for rank 24 (the §6 "principled rank sweep"), labelled "R1(c)".
- Commits: 5739d3bb94c7ab57afe190c8e2da704e1ee2dd5e (results); code flag 2787c83fc22b1b841b9d2e5715f9fd9f6e9e10a1, which is identical to 98c4d3ca2bee6e9d72a628b620b5883b08f4f43f.
- Files: `results/rebuttal/erase_rank8_diabetes_cpu/`.
- My recount: 18/18 pass, R² range 0.0043–0.0087, medication_change accuracy 0.798/0.767/0.692 (mean 75.2% vs 99.9% at rank 24).
- Only erase-layer architecture, Diabetes and 3 seeds were tested.

**R-D. Cross-purpose constraint on all three datasets under the erase-layer architecture.**
- Addresses: the §5.5 concatenation attack, described as the "R3" complaint.
- Commits: 4390835a197bf1b5cabad188872c72d7c0e04d54 (Phase A+B), f4813f348ce6475f20baec4a8df3815e9e968386 (infra), 01496a38346dc0911c73ee966aba4f3f9da7068c, 486645bc5ecc4494e4743eb0b689617b41d34415, 2b6defa51faed3408790ea50171e43e0b40a844b (eval fixes), d39211214314ec0027c345fafef035287fc98a2d (canonical result).
- Files: `results/rebuttal/cross_purpose/{PLAN.md,HEADLINE.txt,PAPER_PASTE.md,unified_protocol_results.json}`, `scripts/crosspurp/unified_protocol_reeval.py`.
- Result: the unified-protocol flag count (gain over the best single-purpose auditor above +1pp) falls from 22/33 to 8/33, with 60/60 per-pair and 27/27 h_concat R² ≤ 0.10.
- My recount: 8/33 from `unified_protocol_results.json`, and 22/33 from the first three columns of `paper-body/tables/cross_purpose_attack_extra.tex`.
- Caveats:
  - The two arms differ in **two** things at once, erase layer plus cross-purpose dual. There is no erase-only arm under the unified protocol.
  - Accuracy is the maximum over auditor seeds on the test split, as the submission also did.
  - All three Diabetes seeds have best_epoch=0, so the deployed model is the LEACE-projected initialization.
  - The in-loop feasibility count is 0/200.
  - Checkpoints are in S3 only.
- **Not an ancestor of M.**

**R-E. LAFTR with a hard linear-R² proxy-Lagrangian constraint.**
- Addresses: "R1's complaint" that LAFTR was audited under PCRL's criterion without being trained against it. The submitted §6 names this as the right future comparison.
- Commits: plan f9f037087ce23646c24975f8be751898358d7a9f (= 0ca91ee69ef76803464f6c07c3b951b9ae5cd4c4), 16 implementation/infra commits, tip 2037ad45287afcfa75a2be110b5ff84fc6b3c0e0.
- Files: `pcrl/training/laftr_proxy_trainer.py`, `experiments/run_laftr_hard_r2.py`, `scripts/aggregate_laftr_hard_r2.py`, `scripts/paper_baseline_numbers.json`. Only the Adult smoke is committed.
- **Full results are uncommitted**: untracked `results/laftr_hard_r2*/{HEADLINE.txt,PAPER_PASTE.md,comparison.*,per_seed/summary}` in the laftr-hard-r2 worktree (provisional).
- Commit messages and memory notes describe an Adult "compliance via deterministic recoding" (Q3) collapse framing and a Diabetes collapse caveat. I did not open or verify these.

**R-F. FAccT resubmission ablations 1–3.**
- Ablation 1: linear adapter instead of LoRA. Ablation 2: remove the erase layer. Ablation 3: held-out hyperparameter selection.
- The registration attributes each ablation to named anonymous reviewers and the AC (handles in the private handoff).
- Commits: ecaeba46fcb921802087e792c5dbac2073e25347 (predictions registered first), 71d6a0f9d2c0d7e7b3af5631959f78550f15d1e5 (code), 7530e9f7dc6c5ce9217a9bcdca1dac012ce1dbd1 (CPU smokes, Diabetes s0, 5 epochs; code-path only), 15dbcc3c3338f6707e7a0b3901d738d99651a77d (launch plan).
- **Never launched**: no results exist on any ref.
- The code is in M's tree (ancestor).

### Phase 2 (September): corrections to the original paper's claims (in M)

The following are the only original-line content M uses:
- 5f162ab37cc9e1caa391e6c964711fa85cc86a14: `docs/ACCURACY_CERTIFICATE_RETIREMENT.md` retires P3 (R²-to-accuracy) and scopes composition to exact zero. `results/redesign_20260907_gaussian_v1/TABLE.md` gives the union-eraser comparison.
- 135e440e6b16ca753465d0492193e33e63280dff: `results/V2_DOMINANT_AXIS_LATEST_SUMMARY.md`, the dominant axis.
- Together with the disclosure and skipped-certificate items, these are what M actually uses from the original line: §III, Table I, Appendix A and the Corrections section.

## 4. Work for the other submission (not this paper's reviews)

Rebuttal work for a different, concurrent submission lives in a separate repository: a fresh-partition
generalisation run and a certified-representation baseline, each registered before execution. It answers
that paper's reviewers, not this one's. It is indexed only in the private handoff, so that this public file
does not link the two submissions. Terminal 2's `REVIEW_TO_EVIDENCE.md` @ 30a6fd19e lists two of those runs
as "Appendix A" of this manuscript. `main.tex` contains neither, and they should not be cited as evidence
for this paper. That note also describes `results/rebuttal/**` as answering the other paper. In fact that
directory is this paper's (NeurIPS PCRL) rebuttal work; see §3.

## 5. Duplicates / superseded copies

Pairs were compared by patch-id and parent, not by timestamp.
- 96c03d01003a10ed1d797974d54d311b83b60456 ≡ dbb24f423de6ef6d8379f6b880bd14e394e80763. Same patch-id and same parent 17ef7d449c74da46659b8e7fa5ecb90d80d81d54. The second is the cherry-pick used on the cross-purpose branch and in local main's merge.
- 65dd5c05063f01bcb9b0da7015595e1edc75dbaf ≡ 7be65f9ab820faecec924e48caffb434bfe08f89. Same patch, rebased onto 17ef7d449c74da46659b8e7fa5ecb90d80d81d54; the aggregator is in main.
- 0ca91ee69ef76803464f6c07c3b951b9ae5cd4c4 ≡ f9f037087ce23646c24975f8be751898358d7a9f. LAFTR-hard-R² plan: the first copy is in M, the second is on laftr-hard-r2 and main.
- 2787c83fc22b1b841b9d2e5715f9fd9f6e9e10a1 ≡ 98c4d3ca2bee6e9d72a628b620b5883b08f4f43f. --lora-rank flag, identical patch and same parent 39c5a84e69bcff54129ee2c16b17d36699dfef4d. The branch `diabetes-rank8-ablation-2026-05-30` adds nothing beyond rebuttal-evidence.
- `results/rebuttal/erase_layer_vicreg_sweep_aws/HEADLINE_PARTIAL.md` is superseded by HEADLINE.md in the same directory, as HEADLINE.md itself states.
- `results/rebuttal/cross_purpose/PAPER_PASTE.md` (d39211214314ec0027c345fafef035287fc98a2d) supersedes an earlier "19/33 vs majority baseline" draft. That draft is probably the untracked `attack_matrix.md`/`comparison.*` in the main worktree; I inferred this from filenames and the PAPER_PASTE appendix without opening them. The canonical protocol number is 22/33 → 8/33.
- The Adult-only `results/v2_adult_CROSSPURP/` (5 May, no erase layer, 14/15 → 12/15) is a **different experiment** from R-D, not a copy.
- The 26/33 figure in the submitted paper (above majority) and 22/33 (incremental over best single purpose) are two criteria applied to the same submission data. See README commit a12153e753c0fb7012087f493303cb643fa7457b.
- Local main (0eee48f990949e522ecf1e3c251d0778e09b09ef) is not a superset of M: it lacks rebuttal-evidence (rank-8) and everything later. origin/main (55e4cb1d1b603a52e5ae16fd5c71d41ebb9b3827) adds only README removals. Neither holds unique rebuttal evidence beyond laftr-hard-r2 @ 1c5e1aff4dfb0b6271ff52387fe01da863ef7762.
- Local vs remote divergences: crosspurp-constraint local ae63692cea9251d604e6722bce3c03380fb0b323 is 10 ahead of origin 93fdb9c26728b9c667797eca9780be605dbd3dd4, and laftr-benchmark local e2f81eedd15be57389be50f83b6a2e996f3befb5 is 2 ahead of origin ade5cd9f35dadec18e1bff76018167b1c8b3040a. In both cases the extra commits are pre-submission cleanup and paper-support work, and all of it is in M. ablations-facct local is 5 behind origin.
- `<author-local file>` and `PCRL_final*` are zipped or unpacked copies of the anonymized supplementary repo from May 5–7, including `results/reviewer_dropins`. They duplicate the Phase 0 material.

## 6. Existing work NOT yet incorporated in manuscript 30a6fd19e (30a6fd19e17453bc8c421a65491b5b482ab9291f)

M deliberately limits original-line claims to the dominant-axis audit, the convex identity, the retired guarantee, composition at exact zero, and the post hoc disclosure. The items below are therefore candidates for the lineage appendix or a prior-review response, not for the main results.

| ref @ SHA | path | finding | suggested use |
|---|---|---|---|
| research/pcrl-submission-finish-v1 @ 30a6fd19e17453bc8c421a65491b5b482ab9291f | results/pcrl_submission_review_v1/REVIEW_TO_EVIDENCE.md | Misattributes `results/rebuttal/**` to the AAAI paper, and lists the AAAI fresh-partition/FARE runs as in "Appendix A" of M, where they are absent | Correct the review note: mark those two rows as other-paper and point the "genuine rebuttal work" row at R-A to R-F |
| rebuttal-evidence @ 5739d3bb94c7ab57afe190c8e2da704e1ee2dd5e | results/rebuttal/erase_rank8_diabetes_cpu/HEADLINE.md + per_seed_results.json | Under the erase layer, rank 8 keeps 18/18 strict R² on Diabetes but costs 24.7pp on medication_change. The rank-24 choice was about task capacity, not erasure | If the author judges this the "same paper", cite it in the prior-review response as the rank-sweep answer. Otherwise mention in one lineage-appendix sentence. Indexed but unused in M |
| erase-layer-vicreg-sweep @ 39c5a84e69bcff54129ee2c16b17d36699dfef4d | results/rebuttal/erase_layer_vicreg_sweep_aws/{HEADLINE.md,diagnostic_perdim_std.json} | VICReg ×5 leaves per_dim_std under 0.5 in 42/42 cells. Backbone features reach at most 0.289 in all 36 (dataset, seed, stage) measurements, so the 0.5 "clean" bar was out of reach for that backbone | Use as measured evidence that the old "cleanly compliant" threshold was mis-set for the architecture. Do not repeat "unreachable by any mechanism" or cite P4 |
| cross-purpose-rebuttal-2026-05-18 @ d39211214314ec0027c345fafef035287fc98a2d | results/rebuttal/cross_purpose/{PAPER_PASTE.md,unified_protocol_results.json} | Unified-protocol attack flags 22/33 → 8/33 with 60/60 per-pair compliance. **Not in M's history** | Relevant to M's composition section as empirical context only if framed as confounded (erase layer + dual together, test-set max over auditor seeds, Diabetes at the initialization iterate). Otherwise leave it in the prior-review response |
| erase-layer-pilot @ 65dd5c05063f01bcb9b0da7015595e1edc75dbaf (+ untracked raw) | results/rebuttal/erase_layer_pilot/*; raw only in untracked results/rebuttal/erase_layer_pilot_aws/ | Strict 54/60 → 60/60, clean 5/60 → 0/60. Backbone is an untrained random init (source-read) | Before any use: commit or archive the raw per-seed JSON and reconcile the 54 vs 56 baseline. Report the clean-count regression beside the strict gain |
| laftr-hard-r2 @ 2037ad45287afcfa75a2be110b5ff84fc6b3c0e0 (+ untracked results) | pcrl/training/laftr_proxy_trainer.py; untracked results/laftr_hard_r2*/ | The submitted paper's own "right future comparison" was implemented and run, but its results were never committed | Commit and verify the results first. Until then it stays PROVISIONAL and should not be cited |
| research/pcrl-manuscript-overnight-review-v1 @ 0176f149e91c02b8e2d202eb25ea9cba8ae019dc | results/reviewer_dropins/{PAPER_PASTE_heldout.md,heldout_s3_compare.json} | A single-fold Adult seed-3 held-out check (6/8 vs 7/8 strict) was in the submitted §6. M says only "neither was validated on a held-out subset" | Optionally add "apart from a single-fold, 8-cell Adult seed-3 check". Keep the no-grid-validation scope. The source's "defusing" wording is too strong |
| same @ 0176f149e91c02b8e2d202eb25ea9cba8ae019dc | results/reviewer_dropins/prop6_bound_utilization/{proof_draft.md,numerical_verification_v2.json,verify_bound_v2.py} | P6 (concat R² ≤ kτ/λ₊(R)) holds on 18/18 Adult rows. It is consistent with M's approximate-leakage counterexample, since λ₊ tends to 0 there. The proof was never independently reviewed | If M wants a quantitative complement to "exact-zero composition", audit the proof first. Otherwise state that P6 was not re-verified |
| ablations-facct @ ecaeba46fcb921802087e792c5dbac2073e25347 | results/rebuttal/ablations_facct/{PREDICTIONS.md,LAUNCH_PLAN.md} | The held-out selection protocol (Abl 3), the answer to the reviewers' tuning complaint, was registered and never run | Cite only as "registered, not executed". It is the natural route to replacing the post hoc disclosure with evidence |

## 7. Venue-review availability

**No genuine NeurIPS review text, meta-review, decision notice or posted rebuttal was found** in any ref of
this repository or of the companion repository. The committed file
`results/rebuttal/ablations_facct/PREDICTIONS.md` (ecaeba46fcb921802087e792c5dbac2073e25347, 2026-07-24)
attributes three concerns to anonymous reviewer handles and an area chair: LoRA versus a plain linear
adapter; necessity of the erase layer; held-out hyperparameter selection. This is indirect evidence that
genuine reviews exist. Their text is not in the repository, and only the author can retrieve it.

Author-local files were also classified. They include an internal pre-submission review, mock or simulated
reviews and agent review passes for the other (AAAI) paper, and unrelated projects. The classification,
with file locations, is in the private handoff (`pcrl_finish_handoff_v1/terminal_3/private/`), not in this
public file. None of those files is a venue review of the NeurIPS submission.

Consequence for the prior-review appendix: Terminal 2's `SUBMISSION_READY_CHECK.md` and
`REVIEW_TO_EVIDENCE.md` correctly say it cannot be completed without the author. The author should
retrieve the genuine reviews and any meta-review and posted rebuttal from the venue system, then decide
whether the rebuttal work indexed above counts as that paper's response.

## Amendment (Terminal 3, 2026-09-22): the 54/60 versus 56/60 baseline

Terminal 3 recounted both from committed files at 0176f149e91c02b8e2d202eb25ea9cba8ae019dc:
- `per_seed_results.json` (best/Cotter-selected checkpoint; best epochs 30–197) gives 21+16+17 = **54/60**
  with linear R² < 0.05.
- `dominant_axis_audit.json` (final.pt, epoch 199; in-sample on test representations) gives 23+16+17 =
  **56/60**.

The NeurIPS paper states the final-iterate rule, so its 56/60 is internally consistent. The rebuttal
"54/60 → 60/60" comparison uses the best-checkpoint baseline, so it is **not like-for-like** with the
paper's headline. Any before/after claim must fix one checkpoint rule for both arms.
