# CORRECTIONS_V5 — additive to CORRECTIONS_V4.md and the historical study records

Nothing here edits a historical study file. Each item names what was said, where, and what the
committed evidence supports. Study 5 evidence: `research/pcrl-competitive-method-v1` at
`a56bcc7ff7a122a6411ac3072b11b00b75f4cdad` (read at the final handoff `7f961d5c7f6f0562efcb25a27a77bb5221c279a7`).
Every number is checked in `STUDY5_VERIFICATION.json` (113 checks, all agreeing).

| # | Said | Where | Correction |
|---|---|---|---|
| C1 | "Study 5 has no committed protocol and no outcome" | manuscript v4 §10, Table 7, abstract | Stale. Protocol locked (`a44b42a1`) before any new outcome; all 372 slots fitted; verdict negative. Integrated in v5 §8. |
| C2 | "J-initialised fine-tuning remains untested" | manuscript v4 §9, correction list item 1 | True of Studies 1–4 only. Study 5 loaded J's actual weights (bitwise verified). Low-gamma selection returned J; gamma in {.1, 1} moved it toward A0. |
| C3 | The A0 k=2 coalition race effect is "candidate-wide significant under both weightings" | Study 5 `RESEARCH_DECISION.md` §4 | The significance is candidate-wide **within the exploratory family X** (m = 3900). No A0 release is in decision family P. `PAPER_ADDENDUM.md` item 4 states this correctly ("exploratory whole-grid level"). Not nominated. |
| C4 | `E_J_C_k8` matches `leace_J` "within ±0.0011 on every endpoint" | Study 5 `RESEARCH_DECISION.md` §2, `PAPER_ADDENDUM.md` item 3 | Unweighted point estimates only. Person-weighted AB/SEX differs by −0.0020. No equivalence or noninferiority test; residence interval [−0.0089, +0.0091]. Point proximity, not equivalence. |
| C5 | `E_J_C_k8` residence "significantly worse" vs J | Study 5 `RESEARCH_DECISION.md` §2 table | Correct unweighted (+0.0101 [+0.0023, +0.0179]); **unresolved** person-weighted (+0.0085 [−0.0009, +0.0178]). Both are now printed. |
| C6 | Study 3 repair: "4 cells significantly worse, 0 better" | manuscript v4 Table (claim_support), carried from v3 | Transcription error in the summary table. Ledger V06 and the v4 text both say **8 of 48** sensitive cells significantly worse, 0 better. v5 table regenerated with 8 of 48. |
| C7 | 2017 `E_J_C_k6` "J-level utility" | Study 5 `RESEARCH_DECISION.md` §7 ("closest thing to a J-level-utility… point") | Holds unweighted (0.0187 vs 0.0189). Person-weighted 0.0159 vs 0.0171: the 0.0012 deficit already exceeds the .001 margin at the point estimate; the per-anchor residence difference changes sign. Hypothesis only. |
| C8 | Stress interfaces "54" read as 54 distinct releases | Study 5 `HANDOFF.json` counts | 54 = (17 releases + the H-only view) × 3 anchors; 6 of the 17 are neural arms bitwise equal to J. |
| C9 | The pre-registered precondition "protocol committed BEFORE the first fit" (v4 PENDING_INTEGRATION item 1) | v4 review | Pre-lock **pilot** fits occurred (Track E pilot, refit after the P1 tolerance fix; Track N timing pilot in a deleted scratch directory; `ref_A0` seed-0 identity audit of an already-published channel). All are outcome-free on new releases and disclosed in `RUN_STATUS.md`. Recorded as a disclosure, not a violation. |
| C10 | Amendment 4 | Study 5 `RUN_STATUS.md` | Declared ~07:45Z **after** the 2018 panel intervals were computed. Both comparators reported; verdict identical. The timing is part of the record and is printed in the manuscript. |

Unchanged Study 5 conclusions (not re-litigated): mechanism works as a finite mechanism; no competitive
tradeoff; no coalition-specific benefit on J; stronger attack uninformative; 2017 no coalition-specific signal.
