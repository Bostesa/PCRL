# SUMMARY: methodology_pcrl_encoder (2026-10-01)

## EARLY FLAGS (for other roles)

1. **The backbone is random, not pretrained.**
   - The tabular headline uses a frozen seeded random MLP; BN runs at default statistics, so it is about
     the identity.
   - Nothing pretrains it (`b158abc54:experiments/run_v2_dataset.py:214-216`, `pcrl/models/lora.py:175-176`).
   - This confirms the claims audit (T3-10).
   - Do not repeat "pretrained on the union of allowed tasks" (PDF(23) l.133).
   - Only CelebA PCRL-V uses a pretrained network: a frozen ImageNet ResNet-18. (F1)
2. **The utility claim is false according to the repo's own file.** "Within 1pp of the unconstrained
   backbone" holds for 2/7 comparable tasks in `origin/main:results/v2_adult_ROUND5/task_acc_vs_unconstrained.json`.
   - The gaps reach −12.85pp.
   - Both passing tasks are uninformative.
   - The comparator is a differently trained model. (F18)
3. **The tabular test splits are spent, and the "held-out seed 3" holds out only the seed.**
   - Compliance R² is in-sample OLS fit and scored on test-split representations.
   - The λ floor, warmup skip, rank 24, the K≥6 gate and the final.pt rule were all chosen from those numbers.
   - Seed 3 uses the same split. (F9, F31)
4. **The HMDA benchmark does not contain the paper's motivating conflict.**
   - Race is disallowed under every HMDA purpose in code.
   - Appendix H's tasks and attributes do not match the code.
   - Protected attributes are model inputs in all three tabular datasets.
   - `tract_denial_high` is computed over all splits, including each row's own outcome.
   - Diabetes has 71,506 deduplicated rows, not 101,766. (F19, F20) This is relevant to the threat-model and
     claims roles.
5. **The rebuttal assets ("54/60→60/60" erase pilot, "22/33→8/33" cross-purpose) are not PCRL as described.**
   - Both fit one LEACE eraser on the **union** of every purpose's disallowed attributes. For Adult this erases
     even the income task label. They report best.pt.
   - The 8/33 is a different model plus a choice of criterion made after seeing results. Under the absolute
     criterion it is 19/33, and absolute nonlinear leakage rose on 20 of 22 cells. (F10, F23)
6. **Leakage reaches single recipients.** The dominant leak in the cross-purpose data is nonlinear recovery
   from a *single* recipient. In 17/33 cells, a purpose that disallows the attribute already leaks it; HMDA race
   reaches +26–28pp from one representation. This bears on the "adaptive evaluation" and "outputs leak" framing.
   (F22)
7. **The worst-class criterion misses contrasts.** "Dominant axis" = max over class indicators, which misses
   linear contrasts between groups of classes.
   - Fixture: one-hot 0.045, DA 0.046, top canonical ρ₁² 0.407.
   - The meeting-summary "worst-class criterion" should be defined via the leading generalized eigenvalue (CCA),
     not max-OvR. (F12)
8. **origin/main still ships the refuted R²→accuracy guarantee** (`certified_accuracy_bound`, `nonlinear_bound`
   in every report, a test asserting it, and README wording).
   - `fix/retire-accuracy-guarantee` is unmerged and has a stale doc reference.
   - The manuscript's Prop 3 proof step (3) treats an argmax indicator as linear. (F25, F26)
9. **CelebA PCRL-V results are in-sample and the training does nothing measurable.**
   - The R² ≤ 0.005 is in-sample on the eraser's fit set. Held-out-style R² is 0.156–0.171 on every epoch.
   - The trainable path after erasure is linear, so the constraint is inert; the model is effectively LEACE plus
     a linear probe.
   - There is a single purpose, so there is no cross-purpose attack.
   - "Trained ≤ init without exception" is false (+7.2pp).
   - `identity_CelebA.txt` was never used. (F27–F30)
10. **Undisclosed confounds.**
    - vCLUB λ = 1.0 was active, while the paper says 0.0. (F2)
    - The Diabetes "λ≈420 ⇒ rank 24" story comes from an absent-class R² = 1 artifact on age class 0. (F15)
    - The in-batch R² constraint has a null floor of about 0.25, well above τ. That is a candidate cause of
      "compliance via collapse" (hypothesis). (F3)
11. **Cross-lineage exposure.** The Folktables ACS 2018 CA test split was used in encoder-lineage v2 Folktables
    rounds before the ACS studies. The ACS SaTML'27 manuscript was withdrawn on 2026-09-26. (F31)
12. **Manuscript version.** The `(21)` source is not the final text. Cite PDF `(23)`, which is text-identical to
    `(24)`. The final .tex is not available anywhere. (F0)

## Headline recount status (details in recounts.json)

| Claim | Status |
|---|---|
| 56/60 strict; τ sweep 37/52/56/59/60; 1 hidden; Tables 7–9 | reproduce exactly on final.pt; 54/60 on best.pt |
| 7/60 cleanly compliant | reproduces only by mixing final.pt R² with best.pt health; single-checkpoint 5/60; R²+health+Δaud 3/60 |
| 46/60 → 56/60 | final.pt 46 vs best.pt 33 for R4; the Diabetes gain comes from R7 (OvR + rank 24) |
| Erase pilot 54→60/60 | reproduces from local untracked files; best.pt baseline; union eraser; utility losses |
| Cross-purpose 26/33 abs, 22/33 incr; rebuttal 8/33 | reproduce; 8/33 = different model, abs 19/33 |
| INLP/LAFTR/PCRL 0.014/0.038/0.270 | reproduce; LAFTR HMDA/Diabetes per-seed files missing |
| Diabetes rank-8 18/18 | reproduces; comparator mislabelled |
| LAFTR-hard Adult 0/24 | unverifiable locally (S3 archive only) |
| CelebA | stored numbers reproduce; claims partly false |
| 12.7× / median 1.43× | reproduce; 12.7× is the ratio of two sub-τ values, and the class is "Joint" |

## Files written (this directory)

| File | Contents |
|---|---|
| `FINDINGS.md` | F0–F31: problem, evidence, severity, repair, verification |
| `ARCHITECTURE.md` | per-implementation, verified call graphs, separate-implementation table |
| `WORST_CLASS_AUDIT.md` | definitions, formulas, code locations, identity check, contrast gap, unsupported classes, thresholds |
| `crosswalk_rows.csv` | 52 rows, PE-A/B/C/D/E/M, 19 columns |
| `data_exposure_rows.csv` | 10 rows, 13 columns |
| `recounts.json` | 15 entries + 6 supplementary entries, sha256 of sources |
| `recount_headlines.py`, `recount_cross_purpose.py` | standalone scripts, no repo imports |
| `worst_class_fixture.py` | standalone synthetic fixture |
| `accuracy_bound_counterexample.py` (+ `_out.json`) | standalone synthetic fixture |

## Method and limits

- **Sources.** Read-only `git show`/`grep` against refs. Local untracked rebuttal results and the relocation
  inventories were inspected.
- **No execution beyond small scripts.** No training, cloud, checkpoint re-evaluation or dataset download.
  Recounts use stored JSON; fixtures are synthetic numpy.
- **Not verified:**
  - EC2 argv for R5/R7, since the user-data is uncommitted;
  - checkpoint-level checks (backbone reconstruction, final.pt health, in-batch null floor on real
    representations, ρ₁² on real cells);
  - the generating scripts for CelebA `train_set_r2.json` and `cross_purpose.json`;
  - the full LAFTR-hard result;
  - the official OpenReview PDF.
- **Minor unresolved inconsistency.** The Adult train size is 24,129 in `data_exposure_rows.csv` (80% of 30,162 rows after dropna) but 24,145 in ARCHITECTURE §1.8 (from `results/adult_LEACE/leace_baseline.json`).
- **Reviews.** Official reviews are not available. Items attributed to the meeting summary are not treated as
  reviewer text.
