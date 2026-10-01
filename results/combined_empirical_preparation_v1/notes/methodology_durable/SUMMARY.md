# SUMMARY — methodology role, durable-guarantees ("Outputs Leak What They Use")

(Saved by the owner from the role's final report; the role's own write was refused by the harness.)

Execution date 2026-10-01; analysis-first; $0 compute (no fits, downloads, training).
Repo: `Bostesa/durable-guarantees@956f5c883f515646aa457db55ecbd74b913768b2` (read-only clone).
Status tags: [SI] source inspected, [SR] stored results checked/recounted, [NR] not rerun, [UV] unverified.

## EARLY FLAGS

1. **Tier 2 is mis-described.** The "informed" attacker is a Gaussian LRT fit on the *pre-noise*
   representations of a labelled training population (attacker-train rows) plus the exact noise
   covariance; it scores a *single* noised release per target (`dg:utils/battery.py:60-103`). It never
   sees the target's clean vector and never averages repeated queries, although README/docstrings say
   "vendor / auditor / anyone who can query the same row repeatedly" (`dg:README.md:62-66`,
   `battery.py:15-21`). Correct classification: **white-box population access, one release**. Against
   deterministic baselines it collapses to QDA on the release (Tier-1 access). Repeated-query
   (averaging; subspace channel only, N ≤ 16) and Q-knowledge attacks are separate experiments; no
   averaging was run on the full-rank channel marked ✓/✓ in Table 1.
2. **Stated pass rule ≠ implemented rule.** Paper: both surfaces must clear the bar
   (`paper.tex:261`). The 67-configuration audit attacks representation only (XGB+MLP); gauntlet verdicts
   are representation-gated (`master_gauntlet_table.py:166`); the authors' channels are never
   output-attacked in `two_tier_certification.py`; where outputs are attacked, logits are read, not ŷ.
3. **Utility is in-sample.** Everything runs on the PCRL *train* partition; attackers held out 75/25
   inside it; utility measured on training rows. Adult clean lift 0.185 in-sample vs **0.069 held-out**
   (fresh partition). Easy-cell lift 0.022 (1 point = 45%); LSAC lift 0.017. Baselines get
   max(own head in-sample, LR held-out); "ours" gets own head in-sample. PCRL val/test splits and dg's
   20% external holdouts were never read.
4. **Matched-utility claim (`paper.tex:576-580`) rests on a nearest-T1 matching rule.** Stored results
   already contain full-rank points within 0.01 of the isolate T1 keeping **55.5% (easy, σ=20)** and
   **56.6% (middle, σ=12)**, not 1.9/38.6%.
5. **Stale number in submitted text.** Easy-cell supported-class utility quoted 49.7% (3-seed,
   `paper.tex:748`); the registered 5-seed recertification reads **41.0%** (1/5 seeds negative lift),
   committed 6a625e7 about 2 h before the submission build. "No configuration can clear" the all-pairs
   null quotes a *max* over 27 null draws (0.556; mean 0.536).
6. **CelebA "frozen encoder we did not train" is PCRL's own `celeba_v2` network**, whose purpose
   definitions list Young as disallowed; clean Young AUC 0.567/0.605. Checkpoint provenance unresolved
   (sha256 b0df3fb7… predates the only tracked writer, whose architecture differs). Image-level splits,
   no identity grouping.
7. **ACS exposure:** only **CA 2018 1-Year** (ACSIncome, ACSEmployment, ACSPublicCoverage). NY and TX
   appear in no revision. Person-level splits not grouped by household (SERIALNO). CA-2018 raw file
   shared with PCRL ACS studies.
8. **No PCRL commit pinned.** Checkpoints read: `v2_adult_s0/final.pt` 1cfc2fef…, `v2_hmda_s0/final.pt`
   e29d0367…, `celeba_v2/final.pt` b0df3fb7…; likely tree origin/main@962841995 or d39211214 [UV].
   739 local-only files (~964 MB, relocated drive) back FARE/FNF/TPR/worst-pair/max-seed/fleet aggregates.
9. **Thresholds not pre-registered.** 0.55 bar first gates on 2026-06-26 right after Exp 3 saw a
   residual of 0.55–0.58; τ = 0.05 inherited from PCRL; registrations date from 2026-07-23 onward; the
   ≥3000-row support rule chosen after the per-pair null table was seen.

## Submitted-version determination

- Aug 30 and Sep 17 zips differ in zip sha256 but members are byte-identical (`diff -r` empty).
- **Main text (high confidence):** `paper.tex` compiles to text identical to the 2026-07-29 04:57 EDT
  build — the last 9-page build before the deadline. Residual uncertainty: whether that exact file was uploaded.
- **Appendix (medium-low):** `appendix.tex` matches the 2026-08-02 build (after the Jul 31
  supplementary deadline). The appendix as submitted on Jul 31 is **not established**.
- The dg repo holds no manuscript (deleted `paper/paper.tex` 2026-07-18, 83739fd).

## Files

`ACCESS_CONTRACTS.md`, `crosswalk_rows.csv` (DG-01..DG-28), `data_exposure_rows.csv` (16 rows),
`recounts.json` (30 recounts: 21 match, 9 flagged), `FINDINGS.md` (F-01..F-24).

## Key findings by area

| Area | Main point | Severity |
|---|---|---|
| Q1 | Tier 2 = white-box population LRT, single release; averaging N ≤ 16 on subspace channel only; σ_eff ≤ 1 needs N = 256–4096 | High |
| Q2 | Verdicts representation-gated; post-hoc outputs are logits of an auditor-fit LR; LAFTR-official misapplied to 5-class race; VFAE T2 white-box | High/Medium |
| Q3 | In-sample utility vs held-out attackers; unstable denominators; best-of-two heads for baselines; selection and certification on same seeds/rows | High |
| Q4 | Train partition only; erasers fit on attacker-test rows; no household/patient/identity grouping; fresh-partition control shrinks attacker data to 37.5% | Medium |
| Q5 | CelebA uses a pre-protected PCRL encoder of unresolved provenance; official train partition only; image-level splits | High |
| Q6 | Macro OvR verdicts; worst-pair uncalibrated (max over pairs and seeds); post-hoc support rule; neither 0.55 nor τ registered | Medium–High |
| Q7 | 67/59/8, 7/42, r=.799/ρ=.828 reproduce; mismatches 49.7 vs 41.0%, VFAE 71 vs 72%, LRT 0.66–0.94 vs 0.66–0.86, Table-1 ranges, Prop-2 caption 0.016, constructed "clean accuracies" | Medium |
| Q8 | Prop 1 algebraic, sound (ridge approx.); Prop 2 population sketch with Y/s and moment inconsistencies, overgeneralized; Prop 3 exact but does not cover Fig-4 points | Medium |
| Q9 | CA 2018 only; PCRL holdouts unspent; headline cells chosen after earlier results; 14 lift-free diabetes_hospital audit rows | Medium |
| Q10 | Isolate vs full-rank nearest-match; post-hoc vs e2e differ in architecture/noise scale/head; fresh partition differs in attacker n and seeds | High (F-08) |
