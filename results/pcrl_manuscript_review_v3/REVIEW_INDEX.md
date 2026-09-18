# REVIEW_INDEX — v3 manuscript and review package

Branch `research/pcrl-manuscript-integrated-v3`, successor to
`3c6ada82e719489656da820b72ce8425d2079be0` on `research/pcrl-manuscript-integrated-v2`.
Every source branch and historical study directory is preserved unchanged.

## Deliverables

| File | What it is |
|---|---|
| [`papers/pcrl_manuscript_v3/main.tex`](../../papers/pcrl_manuscript_v3/main.tex) | the manuscript source |
| [`papers/pcrl_manuscript_v3/main.pdf`](../../papers/pcrl_manuscript_v3/main.pdf) | 18 pages: main text 1–11, appendices A–G 12–17, references on 18 |
| [`papers/pcrl_manuscript_v3/references.bib`](../../papers/pcrl_manuscript_v3/references.bib) | 23 entries; Madras et al. 2018 added in v3 |
| [`papers/pcrl_manuscript_v3/tables/`](../../papers/pcrl_manuscript_v3/tables/) | 22 generated tables + 8 caption-note macros |
| [`papers/pcrl_manuscript_v3/figures/`](../../papers/pcrl_manuscript_v3/figures/) | 9 generated figures |
| [`papers/pcrl_manuscript_v3/MANIFEST.json`](../../papers/pcrl_manuscript_v3/MANIFEST.json) | per asset: generating function + sha256 of every evidence file read |
| [`CLAIM_LEDGER.csv`](CLAIM_LEDGER.csv) | 28 claims with recomputed values and full provenance |
| [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) | pinned commits and every blob read, with sha256 |
| [`CORRECTIONS.md`](CORRECTIONS.md) | 14 new corrections (B1–B14), 17 carried (A1–A17) |
| [`MATHEMATICAL_REVIEW.md`](MATHEMATICAL_REVIEW.md) | M1–M8 at four levels each, with the defects found |
| [`RELATED_WORK_COMPARISON.md`](RELATED_WORK_COMPARISON.md) | primary sources read, five-feature table, attribution ledger |
| [`CONTRIBUTION_ASSESSMENT.md`](CONTRIBUTION_ASSESSMENT.md) | independent review memo: 3 contributions, 3 objections, the smallest gap |
| [`VALIDATION.md`](VALIDATION.md) | checks V1–V8, and what was deliberately not checked |
| [`VERIFICATION_V3.json`](VERIFICATION_V3.json) | machine-readable output of V1–V5 |
| [`PENDING_EXPERIMENT_INTEGRATION.md`](PENDING_EXPERIMENT_INTEGRATION.md) | the fourth study: protocol registered at `86f142c4` and checked (10/10 satisfied, 1 watch item), empty cells, 6 artifact checks, commands |
| [`REPRODUCE_PAPER.md`](REPRODUCE_PAPER.md) | every command, in order, with expected output |
| [`HANDOFF.json`](HANDOFF.json) | aggregate handoff record |

Generators: `experiments/pcrl_manuscript_v3/{make_assets_v3,build_claim_ledger_v3,recheck_study3}.py`,
plus `experiments/pcrl_manuscript_v2/make_assets_v2.py` reused unchanged for the Study 1 /
Study 2 assets.

## Read in this order

1. `CONTRIBUTION_ASSESSMENT.md` — what the paper does and does not support.
2. `CORRECTIONS.md` B1–B3 — the three claims that changed most.
3. `main.pdf` §8 and §9 — the external comparisons and the consolidated limits.
4. `CLAIM_LEDGER.csv` — to check any single number.

## What changed since v2

* **Study 3 integrated in full** at `73903b7f28df68284285f0610a4036beb32b208f`, including
  the first executed external-method comparisons. The predecessor's claim that Terminal 1's
  experiment did not exist was stale and is corrected.
* **Rebuilt around the supported contribution.** New plain-language opening, an explicit
  contribution paragraph naming the interface / the comparisons / the finding, a release-view
  diagram, a study chronology separating a diagnosed bug from a new formulation from a
  rerun, and a single consolidated limits section replacing scattered hedges.
* **Main text 22 pages → 11**, with the engineering narrative and redundant diagnostic
  tables moved to appendices. No negative result was deleted.
* **14 claim corrections**, four of which changed what the paper asserts (`dominates`,
  `indistinguishable`, `implicit regulariser`, and a wrong denominator).
* **One new finding contributed by this review:** every headline paired contrast is
  invariant to the attack scope, while `H`-relative levels move by a factor of five between
  scopes (`CORRECTIONS.md` B11, `VALIDATION.md` V3–V4).
* **The fourth study registered its protocol while this revision was being written.** Its
  protocol, method and matrix were read and all ten preconditions checked (10/10 satisfied,
  one watch item). **No result from it was read**, and no number from it appears anywhere
  here. Its own §1 carries corrections B1–B4, so the two records agree.

## Reviewer risks, as seen from here

| Risk | Where it is addressed |
|---|---|
| The confirmed effect validates a mechanism that is then beaten by a 2023 eraser | `CONTRIBUTION_ASSESSMENT.md` O1; stated in the manuscript's own conclusion |
| Three seeds, one state, two years, one of them spent | §9 "Sampling and design"; `CONTRIBUTION_ASSESSMENT.md` O2 |
| The externals are adaptations, not a benchmark | §8 "What each adaptation is, and what it is not"; `CORRECTIONS.md` B12; O3 |
| Study 3's `VALIDATION.md` contradicts its own `RUN_STATUS.md` on numerical faults | `CORRECTIONS.md` B10; `VALIDATION.md` V5; reported to Terminal 1 |
| The reporting scope was undocumented in the incoming package | `VALIDATION.md` V3; Appendix F |
| Residence is called held-out but has been inspected repeatedly | §2, §9; `CLAIM_LEDGER.csv` V23 |

## Not in this package

A private overlap note against the author's AAAI-27 draft was written **outside every
published path**, at `.git/pcrl_private_v3/PRIVATE_AAAI_OVERLAP.md`. It is not committed,
not pushed, and quotes no unpublished text. Its conclusion for the public manuscript is that
**no citation to that draft is added**, because doing so would link two anonymous
submissions; the shared premise (linear certificates do not survive nonlinear attackers) is
prior art in both and is cited here to LEACE's own scoping and to Elazar & Goldberg (2018).

**ACS 2016 is unscored and unreachable from anything in this directory.**
