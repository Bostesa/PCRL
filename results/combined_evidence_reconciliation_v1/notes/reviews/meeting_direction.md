# Meeting direction: agreed plan vs tentative statements

Role B (reviewer mapping), 2026-10-01. Source: the private automatic meeting transcript
(sha256 f09892a1…bffd9). The transcript has transcription errors and does not reliably label speakers.
This note paraphrases it; it quotes nothing and names no participant. The matrix rows M-01..M-12
(direction) and M-T1..M-T3 (tentative) in `review_response_matrix.csv` carry the status of each item.

## 1. Agreed direction (advisor's instructions, accepted by the author)

1. **Use the reviews to improve the work (M-01).** They show real promise and real weaknesses. A
   rejection is not a reason to abandon the work, and encouraging comments are not evidence of
   correctness or novelty.
2. **One coherent project (M-02).** Write the work as a single project. Decide later, from the evidence,
   the narrative and the length, whether it becomes one paper or two.
3. **Empirical analysis first (M-03).** Put new method development aside for now.
4. **Read before expanding (M-04).** Start with prior *empirical analyses*, including those the reviewers
   pointed to, plus newer work. Establish what our analysis adds. If it adds nothing, reconsider.
5. **Survey protection techniques (M-05).** Then update the comparison set with current techniques.
6. **Repair the methodology from the reviews (M-06).** Specify adversaries, attack access and evaluation
   settings carefully. A security venue would scrutinise the threat model above all.
7. **Include a multi-purpose setting (M-07).** Put a scenario with several purposes over the same
   attributes into the empirical methodology, where it helps expose limits of available approaches.
8. **Run only after the methodology is set (M-08).** Then let the findings drive the discussion and
   motivate any later method.
9. **If the work splits (M-09).** The empirical-analysis paper comes first; the method paper follows and
   cites it, with overlap disclosed (a preprint is an option). The advisor cautioned against submitting
   overlapping work in a way that could look like the same contribution twice.
10. **Strength over deadlines (M-10).** Do not pick a deadline first. Choose the venue later: an AI
    venue, a security/privacy venue, or an experiments-and-analysis track (a data-management conference
    was given as an example of such a track, while noting this work may not fit that venue).
11. **Next meeting deliverables (M-11).** The literature assessment and the methodology improvements.
    A competitive new algorithm is not the immediate deliverable.

## 2. Permission framing (M-12)

- The author described the setting as allow-by-default with explicit prohibitions per purpose.
- The advisor recommended the privacy convention instead: **deny by default; a purpose may use an
  attribute only when explicitly permitted.** The advisor said this changes little technically but is
  the stronger argument for a privacy-motivated system.
- Operationally (added in the author's instructions, not in the meeting): define a declared attribute
  set and an explicit purpose-permission table. Do not claim that everything other than the task can be
  suppressed universally.
- Status: a declared sensitive set and permission table exist only as a draft in the prior assessment
  (`EVALUATION_PROTOCOL_DRAFT.md` §2 on `research/combined-empirical-preparation-v1@f381bc266`), pending
  approval.

## 3. Tentative statements (discussion, not findings)

These must be verified against the literature, not adopted.

- **M-T1.** The author said existing methods cannot express several allowed and disallowed purposes for
  the same attribute, while acknowledging that some existing methods (one co-authored by a collaborator)
  do something similar for a single removal set. Not established. Per-recipient application of an
  existing eraser (e.g., per-recipient LEACE) is a viable adaptation, and coalition-budget formulations
  exist for finite alphabets (Taylor et al. 2026, in the prior literature notes).
- **M-T2.** The advisor expected existing methods to show nothing in a multi-purpose row, which would
  motivate the method. Do not predetermine this. The project's own encoder evidence shows that
  single-recipient leakage dominates (17/33 cells), and the ACS coalition effects were small.
- **M-T3.** The advisor relayed that a reviewer said empirical analyses of this kind already exist, and
  left open whether that is true. Four reviewer-named works were read in full by the prior assessment.
  The two references cited as [1,2] in one review cannot be identified from the export (see
  `source_gaps.md`).

## 4. Strengths reviewers credited (keep these while revising)

These come from the reviews, not the meeting. They are listed so revisions keep them, not as proof of
merit.

- **NeurIPS paper.**
  - Purpose-dependent protection is a realistic problem.
  - Explicit per-purpose views are a natural design.
  - The failure analysis is candid.
  - The dominant-axis (imbalanced-class) diagnostic. Note: it is max single-class R², which misses class
    contrasts; see prior-assessment encoder finding F12.
  - The joint-LEACE composition result (Prop 2).
  - Related work is well written.
  - The multi-purpose conflict problem matters.
- **AAAI paper.**
  - Breadth of attackers, and the distinction between defence-agnostic and defence-aware attackers.
  - The potential value of a systematic audit.
  - The representation-side vs output-side distinction.
  - Re-evaluation under a consistent protocol, and the robustness checks.
  - Disclosure of assumptions; informative figures.
  - The AI review credits Figures 1–2 as strong evidence that a low linear-probe score can coexist with
    high nonlinear recovery.
