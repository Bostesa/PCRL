# Shared context for preparation-phase roles (2026-10-01)

Execution date: 2026-10-01. Phase: **analysis-first preparation**. No new model fits, no cloud
instances, no new benchmark runs, no new evaluation labels/populations. $0 new compute.
Small synthetic fixtures and recomputation from existing stored results are allowed.

## Scope

Organizing (provisional) question: *When do attribute-removal methods' stated protection tests
agree with sensitive-attribute recovery by recipients who know the defense, after task utility is
measured consistently, including what recipients can infer from task outputs and from combining
permitted releases?* This is a candidate, not a claim of novelty. Do not manufacture novelty.
Exclude the LLM tracing / ledger / removal-pricing project ("What Can AI Hide?") entirely.

## Repositories (read-only for every role except owner/integration)

| Repo | Location | Pin | Notes |
|---|---|---|---|
| Bostesa/PCRL | `/Users/nathansamson/PCRL` (main checkout currently on `ablations-facct-2026-07-24` @ ad2c08872, **owned by another terminal; do not checkout/modify**) | origin/main 55e4cb1d1 | Read other branches with `git -C /Users/nathansamson/PCRL show <ref>:<path>`, `git grep <pattern> <ref> -- <paths>`, `git ls-tree`. Existing worktrees under `/Users/nathansamson/PCRL/.worktrees/` belong to other studies — read only. |
| Bostesa/durable-guarantees | read-only clone `/private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad/dg` | origin/main 956f5c883f515646aa457db55ecbd74b913768b2 (= the relocated local checkout's HEAD) | The original local checkout was relocated 2026-09-30 to an external drive (not mounted). 721 untracked local files (large score arrays/logs under `analysis/`) exist only there; file list + sha256 in `/Users/nathansamson/storage-relocation-20260930/inventories/tree-durable-guarantees.json.gz`. durable-guarantees imports PCRL via `PCRL_ROOT` (`utils/pcrl_io.py`). |

Important PCRL refs (not exhaustive): `origin/main`; `fix/retire-accuracy-guarantee` (5d4eda046, unmerged);
`research/pcrl-submission-finish-v1` (55c0c5a35; SaTML-candidate manuscript `papers/pcrl_satml_final_v1/`, `docs/PCRL_REVIEW_INDEX.md`, `docs/PCRL_PRIOR_WORK.md`);
`research/pcrl-claims-foundation-v1` (33124f965; `results/pcrl_claims_foundation_v1/PRIOR_ART_MATRIX.md`);
`research/pcrl-guarantee-review-v1` (1dfb17f2c); `research/pcrl-manuscript-integrated-v6`; `research/pcrl-evidence-paper-v1`;
`rebuttal-evidence` (5739d3bb9); `cross-purpose-rebuttal-2026-05-18` (d39211214); `erase-layer-pilot-2026-05-17`;
`erase-layer-vicreg-sweep-2026-05-18`; `laftr-hard-r2-2026-05-17`; `bios-pcrl-layer12-2026-05-05`;
`ablations-facct-2026-07-24` (local ad2c08872 vs origin 349efa454); research/pcrl-* study branches (ACS release studies, Sept 2026).

## Three manuscript lineages (verify, don't assume)

1. **PCRL encoder paper** — "One Encoder, Many Purposes: Purpose-Conditioned Representation Learning
   with Per-Purpose Linear Leakage Bounds" (earlier title: "...with Compliance Certificates"). NeurIPS 2026
   submission (withdrawn by authors 2026-08-21 per notification email). Local sources:
   `/Users/nathansamson/Downloads/Formatting_Instructions_For_NeurIPS_2026 (21)/` (May 7 source),
   `/Users/nathansamson/Downloads/neurips-pcrl/`, PCRL repo `paper-body/`. Methods: frozen backbone + per-purpose LoRA,
   LEACE warm-start / erase layer, proxy-Lagrangian, Dominant-Axis Auditing; Adult/HMDA/Diabetes, CelebA, BIOS.
2. **Attribute-removal analysis** — "Outputs Leak What They Use: Auditing Attribute Removal Beyond Its
   Certificates" (durable-guarantees). AAAI-27, did not advance past Phase 1 (2026-09-24). Sources:
   `/Users/nathansamson/Downloads/AAAI27___Outputs_Leak_What_They_Use.zip` (Aug 30 source: paper.tex, appendix.tex,
   AnonymousSubmission2027.tex, references.bib) and `... (1).zip` (Sep 17).
3. **ACS finite-token release studies** — "Sharing Data for Specific Predictions with Privacy Constraints"
   (PCRL `research/pcrl-submission-finish-v1:papers/pcrl_satml_final_v1/`, also
   `/Users/nathansamson/Downloads/PCRL_SaTML_Manuscript.pdf`). SaTML '27 candidate; [owner update: withdrawn by an author 2026-09-26] submission status NOT
   established — do not assert submitted/withdrawn.

Keep these three distinct. No code path implements a combined encoder/adapter/stochastic-release pipeline.

## Reviews

The **official** NeurIPS and AAAI review texts are **not available** locally or by email (OpenReview
requires login). Files in Downloads such as `reviews_outputs_leak_what_they_use.md`,
`reviewer-2-full-review.md`, `deep-review-round3.md`, `durable-guarantees-review.md`,
`simulated_reviews_v28.md`, `PCRL/MOCK_REVIEWS_2026-07-24.md`, `Review of Submission.docx` are
**simulated/self-generated reviews**, not official. Never present them as official reviewer text.
The meeting summary (not available as a file) reportedly cites: Ravfogel et al. 2022; Elazar et al.
2021; Song & Shmatikov 2020; Stadler et al. 2024; two unresolved references from a reviewer labelled
[handle withheld]; worst-class criterion; threshold sweep; common held-out utility; adaptive evaluation;
architecture; CelebA details; missing methods (Gitiaux & Rangwala AISTATS 2021; FairNVT). Mark
these as "known only from the meeting summary".

## Privacy

Do not copy reviewer handles, submission IDs beyond what is already in this file, private emails,
raw individual records, credentials, or account IDs into outputs.

## Rules for every role

- Write ONLY under your own `notes/<role>/` directory inside
  `/Users/nathansamson/PCRL/.worktrees/combined-empirical-preparation-v1/results/combined_empirical_preparation_v1/`
  (the verification role also owns `fixtures/`). Do not commit; the owner commits.
- Do not modify any other file, branch, worktree, stash, sparse-checkout config, S3 object or cloud resource.
- Do not launch training, cloud compute, or downloads of datasets. Small web reads of papers/code are fine.
- Every important statement needs a source: `repo@commit:path:line` or paper section/page/table.
- Distinguish: source inspected / stored results checked / not independently rerun / unverified.
- If you find something that changes another role's work (e.g., a threat model is mis-classified,
  a reviewer-cited paper is missing), put it at the TOP of your summary under "EARLY FLAGS".
- Final message: a concise summary (<= 600 words) listing files written and key findings; details
  belong in the files.
