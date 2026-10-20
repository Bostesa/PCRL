# Handoff to the manuscript owner and the claims owner

Campaign `pcrl_new_release_mechanisms_v1` (2026-10-20). This file is a repository handoff; no message was sent to anyone.

## What can be said

Use the statistical reviewer's headline wording exactly (`RESEARCH_DECISION.md`, from `roles/statistical_reviewer/CLOSING_STATEMENT.md`).
In short: no new release mechanism met the frozen development criteria and no candidate was selected; family A closed at its capacity
stage; family B's post-hoc SEX-targeted Gaussian logit channel stayed within the original limits at two strengths but did not lower
linked-pair recovery beyond its same-form SEX-blind control (+0.0027 and +0.0019 AUC, each with SE 0.0089), so its falsifier F1 holds.

Supporting numbers: `TABLES.md` (generated), `figures/FRONTIER.png` and `.csv`, `claims/FINAL_V2/CLAIMS.json`, `decisive/`.

## What must not be said

- Not "targeting helps", "promising" or "useful component": G2 was not met anywhere.
- Not "post-hoc noise cannot work" or "the gap is unclosable": no general claim follows, and the coordinator's decision not to launch a
  replacement family (A4) is not evidence for one.
- Not "confirmed" or "replicated": everything is development evidence on adaptively used MODEL_SELECT records.
- Not "SEX targeting equals SEX-blind removal": the control-path result is "not separated" (standard errors are about 40% of the 0.020
  target).
- The partial remainder flattening's null effect is a measured result for the declared readers (pair recovery changed by about
  0.001), not a general invertibility principle.

## Corrections that touch earlier wording

- **Injectivity.** Adjudication A3 of this campaign repeated the claim that continuous deterministic releases are injective, which was
  withdrawn on 2026-10-16. It is corrected visibly (`roles/coordinator/ADJUDICATIONS.md`, erratum; `PROTOCOL_DEVIATIONS.md` item 12).
  Manuscript text should keep following the 2026-10-16 correction: determinism and continuity do not imply injectivity, and the
  population injectivity of the actual models is unknown.
- **Claims records to cite.** The decision is governed by `claims/FINAL_V2/` (both declared control paths). `claims/MAIN_V2/` is the
  registered-grid claims (MINF path only). Superseded and kept: `claims/MAIN/` mislabelled every grid as invalid because an input was
  dropped (`PROTOCOL_DEVIATIONS.md` item 10); `claims/FINAL/` differs from FINAL_V2 in one selection-record field (ENG2-02). The
  independent verifier replayed MAIN_V2, FINAL and FINAL_V2 with 0 mismatches (`VERIFICATION.md`).

## Open items

- **Off-device custody is pending.** The owner's drive (YOTUO) was not attached during the campaign (checked 16:46:48Z and 17:10:09Z);
  only same-device custody exists, verified from the backup alone at the final research tip 9974d9efe (`custody/BACKUP_VERIFICATION.json`).
  When the drive is attached, the safe copy and check is:
  `python3 -m nrm.custody offdevice <worktrees>/pcrl-new-release-mechanisms-v1-backup/pcrl_new_release_mechanisms_v1_20261020T170906Z <external drive>`
  A check right after copying is warm; a cold re-check needs an ordinary eject and re-attach first. Nothing is mounted, unmounted or
  forced by the tool.
- **Untested routes** (listed in `RESEARCH_DECISION.md`) remain open for a future campaign; the MMP mixture still awaits the target
  decision recorded on 2026-10-16.
- **Rule note for a future campaign** (statistical reviewer): A-SC1 clause (ii) gives no input-cardinality explanation on a finite-cell
  minimum set, so PRC_TWIN's grid is invalid on its own minimum set; this affected no decision here.
