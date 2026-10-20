# Engineering / claims reviewer, wave 2 part 2: claims review of the documents (saved verbatim with attribution)

Author: the engineering / claims reviewer of campaign pcrl_new_release_mechanisms_v1. Returned to the coordinator at about 17:03Z on
2026-10-20 (its own header), reviewing the documents at commit b2f0dccce. Read-only, except one label-free computation it charged.
Saved verbatim; nothing below the line was edited. Adjudication: roles/coordinator/ADJUDICATIONS.md, A6.

---

Engineering / claims reviewer, wave 2 part 2 (claims review of the documents at b2f0dccce), 17:03Z

None of the findings below is BLOCKING, and no finding changes the decision. Everything was read-only. I ran one label-free computation: `rules.reserve_status` on PRC_TWIN's saved evaluation record (3 CPU-s, charged).

**What checks out:**
- the headline is quoted verbatim;
- selection, F1, MEAN and FB2P agree with `claims/FINAL_V2` and `decisive/`;
- the twin, MINF, CTL and reference benefits and SEs agree with `TABLES.json`;
- the robustness numbers and the reader-seed qualifier agree with `PHASE2_REPORT`;
- the 36-40% / 11-14% fit-time figures agree with Designer B's closing report;
- the six packages with 19 refusals each agree with `DEPLOYMENT_CHECK`;
- off-device custody is recorded as pending;
- forbidden words appear only in negation.

**DOC2-01 RECORD: CTL_REM_1p00's failure is misdescribed.**
- Places: ADVISOR_BRIEF.md:40 and :77-78, RESEARCH_DECISION.md:32 and :69-70, CLOSEST_WORK.md:90-91.
- What the record shows (failed list in `dev/UTILITY_LEDGER.csv`; ratios from the evaluation record):
  - the 2.67 / 2.97 × cap is the men's guard;
  - the occupation NLL clauses are at 1.79-1.87 / 2.20-2.21 × cap;
  - the occupation Brier clauses pass (0.55-0.87).
- The documents say "2.7-3.0 times the occupation caps" or "occupation clauses and men's guard at 2.7-3.0".

**DOC2-02 RECORD: the headline says "about 0.011" for the SEX_S5 reduction against its base** (RESEARCH_DECISION.md:20, from CLOSING_STATEMENT.md:39). That is the base-instance value (0.6605 − 0.6496). Every other document cites the Monte Carlo pooled +0.0120 (TABLES §2, ADVISOR_BRIEF:43). Ask the statistical reviewer to confirm "about 0.012", or footnote which reading is meant.

**DOC2-03 RECORD: equivalence-type reading of "not separated"** (ADVISOR_BRIEF.md:75). "The SEX-blind channel did almost as well (difference +0.0019, se 0.0089)" reads "not separated" as near-equal. Suggested wording: "was not separated from it (+0.0019, se 0.0089)".

**DOC2-04 RECORD: interpretations stated as findings** (RESEARCH_DECISION.md:64 and :66-67).
- "The gap ... sits in what the task outputs themselves transmit" and "the readers use SEX information outside the single targeted direction" were never tested.
- "The readers' AUC barely moved" is wrong: it fell about 0.012, as much as under SEX-blind noise.
- Fix: phrase these as "consistent with", for this base and the declared readers.

**DOC2-05 RECORD: the evidence earlier work "repeats" is misattributed** (CLOSEST_WORK.md:32-35).
- "Repeats the earlier finding that finite many-to-one cells pay about 0.013 nats ... on this cohort" uses this campaign's own estimate (roles/designer_a/CLOSING_REPORT.md:36). The earlier finite-code studies were on Adult (about 0.02 nats; FAILURE_MAP row 7).
- The gate is also misdescribed: it was nested estimate + 2 SE ≤ 0.004 nats over the continuous teachers, not "room under the 0.010 limit".

**DOC2-06 RECORD: the citation instruction contradicts the documents** (HANDOFF.md:31-33). It says `claims/MAIN_V2` "is the one to cite", while START_HERE and RESEARCH_DECISION govern by `claims/FINAL_V2` (with the CTL path). Also, the verifier's 0-mismatch replay covers MAIN_V2 only; the FINAL_V2 replay is pending.

**DOC2-07 RECORD: custody predates the final records** (HANDOFF.md:37-38; START_HERE.md:37). The same-device backup in `custody/BACKUP_VERIFICATION.json` is at tip ae6aa602c (16:45Z). It predates 735cde202 and b2f0dccce (FINAL_V2, decisive FINAL_V2, tables, documents). Re-run it at the final tip, or state the tip.

**DOC2-08 RECORD: referenced files do not exist yet** (START_HERE.md:19, :35, :38). `VERIFICATION.md`, `EXPERIMENT_LEDGER.csv`, `CONFIG_LEDGER.csv`, `ACCESS_RECEIPT.json` and `RESOURCE_ACCOUNTING.json` are missing. Write them, or mark them pending, before publication.

**DOC2-09 RECORD: the QUICKSTART example silently mis-encodes** (QUICKSTART.md:48). `'POBP': '12'` has no encoder column: the encoder uses 'POBP=012', and SCHL codes are zero-padded ('SCHL=01'). `encode_raw` maps the unknown code to POBP=__OTHER__ without any refusal. Use '012', and say that unknown codes fall into OTHER.

**DOC2-10 MINOR: figure caption** (ADVISOR_BRIEF.md:14-15). The figure and its CSV show base-instance pair AUC: FB2_SEX_S5 0.6496, as the figure's y axis says. The text cites pooled values (0.6486), and the caption does not say which. Commit b2f0dccce says the figure uses pooled values; it does not.

**DOC2-11 MINOR: two seed labels appear without explanation** (ADVISOR_BRIEF.md:56-57; RESEARCH_DECISION.md:48-49). The `claims/FINAL_V2` records show A-SC1 grid validity `READER_SEEDS_EFFECTIVE`, while the documents say seeds were "partly effective" (the aggregate seed qualifier). Name both so the documents do not appear to contradict the claims.

**DOC2-12 MINOR: novelty label differs from the claims record** (CLOSEST_WORK.md:8; ADVISOR_BRIEF.md:65-66). The documents call family A `PRIOR_ART_FOUND` (the plan's label), but the claims record shows G7 NOT_CLAIMED for PRC_TWIN, because the grid-invalid branch forces it.

**DOC2-13 MINOR: PRC_TWIN's reserve cell is a placeholder** (TABLES.md:89). It shows "False / —" from that same grid-invalid branch. The rule computed directly gives RESERVE_ONE_SEED_ONLY, min ratios 2.31 / 1.29. The label happens to be right; show the computed ratio.

**DOC2-14 MINOR: citation** (CLOSEST_WORK.md:47-48). Dwork, McSherry, Nissim and Smith (2006) introduced the Laplace / sensitivity-calibrated mechanism, not the Gaussian mechanism. Cite Dwork, Kenthapadi, McSherry, Mironov and Naor (2006) or Dwork and Roth (2014) for the Gaussian mechanism. The other citations are correctly attributed.

**DOC2-15 MINOR: smaller wording and accuracy points.**
- ADVISOR_BRIEF.md:3: "about 3 hours"; the elapsed time is about 2.5 h since 14:29:43Z.
- START_HERE.md:26: the receipt was verified on the remote at 15:14:43Z, not "published" then.
- QUICKSTART.md:7-8: family B v1 was fitted under the torch venv, not python3.
- QUICKSTART.md:30: `deploy store` is write-once, so it fails on the existing private tree.
- CLOSEST_WORK.md:41: the temperature is applied before the noise (z/T + c·ε·v), not after.
