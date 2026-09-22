# PCRL claims foundation — review index (Terminal 3)

Branch `research/pcrl-claims-foundation-v1`, based on f4bdf4cd5bf74c634feeec50aef78bff249667e4. The
manuscript target is Terminal 2's `research/pcrl-submission-finish-v1` @
30a6fd19e17453bc8c421a65491b5b482ab9291f.

## Final verdict (four separate judgments)

- **Correctness.** The current manuscript is mostly correct and admirably scoped. Two statements are
  wrong about the method's own objects:
  - the code is called "label-free" but is residence-supervised;
  - the coalition scope of the selected release is implied but not imposed.
  One definitional sentence is wrong: "a floor on what is recoverable". One code claim is false on the
  public default branch: the retired API still runs on main. Each has an exact repair in patch 0002, and
  patch 0001 fixes main's code. For the original encoder line:
  - P3 is refuted (already withdrawn).
  - P4 is misapplied to the trained architecture.
  - The "pretrained backbone" description is false; the backbone was a frozen random MLP.
  - P1, P2, P5 and P6 are correct.
- **Novelty.** No algorithmic novelty survives. The finite programme is established (Calmon & Fawaz 2012;
  Salamatian 2015). Releases beside fixed earlier releases (Erdogdu & Fawaz 2015) and per-party and
  collusion MI constraints (Taylor et al. 2026) are close antecedents. The one-hot identity is the
  definition of variance-weighted R². What survives is a problem setting and a measurement contract:
  conditioning on a *third party's immutable prediction service* that a coalition partner holds. It also
  keeps an honest empirical operating point and audit cautions.
- **Empirical competitiveness.** On reused 2018 development data the release beats J on the permitted
  task, with bounds excluding zero. It does not establish the registered conjunction, and it is matched by
  the deterministic D17. Randomisation and coalition conditioning earned no attribution. The fresh-year
  test belongs to Terminal 1.
- **Submission readiness.** Ready after patch 0002 (wording and attribution only; builds cleanly), subject
  to:
  - the author retrieving the genuine NeurIPS reviews for the prior-review appendix;
  - a decision on applying patch 0001 to main;
  - the optional abstract edits, depending on the venue's rules.

## Files

| File | Content |
|---|---|
| [CLAIM_MATRIX.csv](CLAIM_MATRIX.csv) | 37 claims: 2 blocking, 12 major, 15 minor, 8 none |
| [CORRECTED_CONTRIBUTIONS.md](CORRECTED_CONTRIBUTIONS.md) | 4-item list, explanations, a do-not-claim table, title note |
| [THEORY_REPAIRS.md](THEORY_REPAIRS.md) | P1–P6 and manuscript mathematics: proofs, counterexamples, attribution |
| [PRIOR_ART_MATRIX.md](PRIOR_ART_MATRIX.md) / [PRIOR_ART_NOTES.md](PRIOR_ART_NOTES.md) | matrix, novelty paragraph, M1–M7; primary-source notes |
| [METHOD_LINEAGE.md](METHOD_LINEAGE.md) / [METHOD_LINEAGE_NOTES.md](METHOD_LINEAGE_NOTES.md) | the four families, the implemented-path diagram, file:line evidence |
| [IMPLEMENTATION_CLAIM_ALIGNMENT.md](IMPLEMENTATION_CLAIM_ALIGNMENT.md) | 15 alignment rows; tests |
| [BRANCH_DISCOVERY.md](BRANCH_DISCOVERY.md) / [REBUTTAL_WORK_INDEX.csv](REBUTTAL_WORK_INDEX.csv) | refs, rebuttal work, duplicates, work not incorporated (public, sanitised; the full version is private) |
| [REVIEW_RESPONSE_EVIDENCE.md](REVIEW_RESPONSE_EVIDENCE.md) | R1–R10 concern → evidence → status |
| [CORRECTIONS.md](CORRECTIONS.md) | T3-1…T3-17, each marked number, interpretation or withdrawn |
| [MANUSCRIPT_PATCH.md](MANUSCRIPT_PATCH.md), `patches/0002-…`, `tex/` | integration patch against 30a6fd19e and fragments |
| `patches/0001-retire-accuracy-guarantee-on-main.patch` | separable code fix for public main |
| [SOURCE_MAP.json](SOURCE_MAP.json) | inspected, recomputed, reported, missing and superseded sources, with SHA-256 |
| [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) | commands, outcomes, limits |
| FIXTURE_OUTPUTS.json, DOMINANT_AXIS_REPLAY.json, HISTORICAL_PRECISION_DIAGNOSTIC.json | machine outputs |
