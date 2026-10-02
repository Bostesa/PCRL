# Updated meeting brief, after evidence reconciliation

2026-10-01. This brief supersedes `results/combined_empirical_preparation_v1/MEETING_BRIEF.md` where they
differ (see CORRECTIONS_TO_PREVIOUS_ASSESSMENT.md).

**Basis:**
- The actual venue reviews: NeurIPS (3 human reviews); AAAI (2 human reviews + 1 AI review, kept
  separate).
- The meeting transcript.
- All PCRL and durable-guarantees branches.
- The external drive (checkpoints, per-seed outputs, raw score arrays).
- 48 verification items: 23 independently recomputed, 25 checked against code and recorded aggregates.
- The owner reran the key recount scripts, and their outputs were byte-for-byte identical.

**Ground rules for this pass:**
- Nothing was trained or launched.
- No new data was opened.
- Nothing was merged into main.

**Advisor's direction, from the transcript (agreed, not tentative):**
- One coherent project first; decide one paper or two later.
- Empirical analysis first.
- Read the reviewer-cited and newer analyses before expanding.
- Survey techniques.
- Repair adversaries, access and evaluation settings.
- Include a multi-purpose setting where it helps show the limits of existing approaches.
- Run after the methodology is set.
- Quality over deadlines.
- Permissions should be explicit (deny by default, declared attribute set).

## 1. What do the two papers already establish?

**Attribute removal (AAAI paper).**
- **Linear checks miss nonlinear recovery.** Representations that pass a linear check are still
  recoverable by nonlinear attackers. 59 of 67 fail at AUC 0.55; the count is 64 and 51 at 0.52 and 0.60
  (recomputed from stored probabilities). Only noise-based configurations survive.
- **The published methods do poorly under one attacker battery.** At 0.55, 7 of 42
  method–cell–tier combinations pass, and all but one of those are FARE. FARE keeps only 0–39% of the
  task utility.
- **Label coupling tracks output leakage.** How strongly the task label is tied to the attribute tracks
  output-side leakage and removal cost *in aggregate* (r ≈ 0.80, 27 cells). That is an association, not
  a per-cell price.
- **Reviewers valued the audit itself.** Both human reviewers valued the systematic attack breadth and
  the defense-agnostic vs defense-aware distinction. One praised the robustness checks and disclosures.

**Purpose-conditioned representations (NeurIPS paper).**
- **The problem.** Conflicting purpose permissions are a real problem that a single shared
  representation cannot serve. All three reviewers accepted this.
- **The architecture.** Per-purpose adapters with LEACE initialisation achieve per-purpose *linear*
  compliance:
  - strict: 56/60 at the final iterate, 54/60 at the best-validation checkpoint;
  - only 6/60 (final) or 5/60 (best) stay "clean". The paper's 7/60 mixes the two checkpoint rules.
- **Dominant-axis audit.** It exposes rare-class leakage that one-hot R² hides. Two reviewers named it as
  a strength.
- **Per-purpose linear bounds do not compose.**

**Both papers together.** Across both, the strongest single observation is the gap between a method's own
(linear or in-scope) test and what a realistic nonlinear recipient recovers. Much of that observation
already exists in the literature (Ravfogel 2022, Elazar & Goldberg, Song & Shmatikov, Stadler, Wang 2019,
FARE/FRG, FairNVT).

**Not yet established, by either paper:**
- whether any certificate fails *within its own scope*. No held-out linear probe was ever run;
- general defense-aware (adaptive or white-box) robustness;
- a common held-out utility comparison;
- held-out hyperparameter selection beyond one Adult seed.

## 2. What did the rebuttal and later work genuinely improve?

Completed work that is real and supported by files:

1. **LAFTR hard-R² ran in full.** It is the only comparator trained against PCRL's R² criterion.
   - Result: 35/60 strict, all collapsed (Adult 0/24).
   - Outputs are on the external drive.
   - It shows the criterion-matched competitor does no better.
2. **Erase layer.** It makes strict linear compliance *structural*: 60/60 vs a like-for-like 54/60.
3. **Cross-purpose constraint.** It cuts *incremental* concatenation flags from 22/33 to 8/33.
4. **VICReg ×5, rank-8 and the per_dim_std diagnostic** were run and are reproducible from per-seed
   files.
5. **AAAI-side work already existed before the reviews:**
   - threshold sensitivity at 0.52 and 0.60;
   - supported-pair worst-pair rescoring;
   - a fresh-partition held-out check (0 verdict flips);
   - repeated-query averaging and a knows-the-basis attack.
6. **Raw score arrays** survive on the drive and allow exact recounts.

What the same work does **not** show:
- The erase layer **raised** nonlinear recovery on 48 of 60 cells. Adjusted passes fell from 32/60 to
  16/60.
- The cross-purpose result came from a different model with a single union eraser, which also removes
  attributes some purposes are allowed. Under absolute recovery, leakage on nonlinear cells *rose*
  (26/33 → 19/33 flags, but the mean leak went up).
- The "0.5 floor is architectural" claim depends on scale, not architecture.
- Rank-8's apparent independence from LoRA rank holds only by construction in the erase-layer variant.
- Almost all of this work **predates** the reviews. The one review-driven plan, the FAccT ablations
  (linear adapter vs LoRA, and held-out selection), was registered and smoke-tested but **never run**.

## 3. Which weaknesses remain?

Each item has artifact evidence; see the matrix and verification report.

1. **Held-out selection and held-out utility.**
   - NeurIPS tuned on its evaluation grid. Held-out validation is Adult seed 3 only, on the same split.
   - AAAI utility is in-sample with mixed conventions, and method orderings change on two of three cells
     under a common held-out convention.
2. **Threat model.**
   - AAAI "Tier 2" is a population likelihood-ratio test on one release, mis-described as an insider or
     repeated-query attacker.
   - No adaptive or white-box attacker exists.
   - The full-rank noise points rated as passing are unclipped, so they are empirical, not certified by
     Proposition 3.
3. **Multiclass criteria.**
   - Macro averaging flatters results: FARE, VFAE and "ours, full-rank" fail supported-pair or worst-class
     criteria.
   - The dominant axis misses class contrasts.
   - Absent classes were mis-scored in training.
4. **Criterion choices were post hoc.**
   - The cross-purpose headline switched criteria twice.
   - The 0.55 bar and τ were never registered.
   - The 7/60 count mixes checkpoint rules.
   - Two rebuttal headline tables use mixed comparators.
5. **Baseline fairness.**
   - No submitted baseline was trained to the R² criterion.
   - The original LAFTR HMDA/Diabetes rows (36 of 60) have no surviving files.
   - SPLINCE post-processes PCRL's own representations.
   - INLP's utility is higher than PCRL's on 14 of 27 cells.
6. **Claims that overreach the code.**
   - "Pretrained" backbone (it is random).
   - "Within 1pp" utility (2 of 7 tasks).
   - The R²→accuracy bound (refuted, and still shipped on public main).
   - CelebA: in-sample compliance; the AAAI vision encoder is PCRL's April 16 `celeba_v2` CNN, whose
     training config was not located.
   - The AAAI encoders are Round 4, not the NeurIPS headline models.
7. **Literature overlap.**
   - FairNVT and Gitiaux & Rangwala overlap the noise contributions.
   - Wang et al. 2019 measures output vs label leakage.
   - Stadler et al. already measures label-only leakage.

## 4. Which empirical question is still worth pursuing?

**One question, sharpened by the second AAAI human reviewer's (1)-vs-(2) distinction.**

The question: *For a declared sensitive set and purpose-permission table, how often do releases that pass
a method's own stated test remain recoverable, and for what reason?* The two possible reasons are:
- **(1)** a failure *within* the test's stated scope, for example a held-out linear probe against a
  linear certificate;
- **(2)** recovery by recipients *outside* that scope:
  - nonlinear attackers;
  - task outputs, measured against a label-only reference;
  - attackers who know the defense, with explicit access levels;
  - combinations of permitted releases.

All of this is measured at a common held-out utility.

**Why this question:**
- It answers both papers' reviewers directly.
- It uses the multi-purpose setting Dr. Yus asked for, as a scenario where per-recipient adaptations of
  existing methods are run rather than assumed to fail.
- Category (1) has **never been measured** in either repository.

**What would make it new.** The prior literature covers category (2) piecemeal. The contribution would be
the controlled decomposition under one access and utility contract. It would not be the observation that
nonlinear attackers succeed.

**Secondary question.** The label-coupling association, but only as an association tested on registered
cells under held-out utility.

## 5. What is the smallest next experiment justified by the reconciled evidence?

**A frozen-artifact re-evaluation pilot.** No defense is retrained. Development data only. To be run only
after Dr. Yus approves the protocol.

**Inputs** (stored representations and checkpoints already on the drive):
- the AAAI audit configurations, with their stored representations and score arrays;
- the three AAAI headline cells;
- PCRL Round 5/7 final.pt and best.pt;
- the LAFTR hard-R² per-seed outputs.

**What it does.** Apply the repaired evaluation:
- a held-out *linear* probe on the same rows, for the within-scope test (category 1);
- the nonlinear slate;
- output-surface scoring with a label-only reference;
- macro, supported-pair worst-class and top-canonical-correlation scores, with explicit "not
  estimable" counts;
- bars at 0.52 / 0.55 / 0.60 plus CI-based (UCB95) bars;
- access tags on every attacker;
- positive and null controls before any method row is read.

**What it would show:**
- the decomposition of 59/67 that the AI review asked for;
- the in-scope vs out-of-scope split the second AAAI human reviewer asked for;
- whether the repaired pipeline works end to end.

**Cost:** CPU-hours. It needs attacker fitting, so it requires approval and was not done here.

**What it cannot show:**
- held-out utility rankings, because the defenses were fit on spent splits;
- the cross-cell coupling–cost relation;
- whether a multi-purpose method is needed.

Those need the registered benchmark that follows.

**Not justified yet:**
- New method training.
- The FAccT ablations. They test the later erase-layer variant, and method work is deferred.
- Any fresh confirmation population.

## Decisions for Dr. Yus

1. Accept the single question in §4, including the scope (1)/(2) framing?
2. Approve the frozen-artifact pilot, development data only, after approving the access table, threshold
   grid and worst-class criterion?
3. Approve the purpose-permission table (deny by default) for the multi-purpose scenario?
4. Whether to merge the existing fix that removes the refuted guarantee from public main (pending; not
   done).
5. One paper or two: still deferred until a draft exists, per the meeting.
