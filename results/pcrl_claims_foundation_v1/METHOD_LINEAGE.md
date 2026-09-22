# Method lineage: what was actually implemented

Terminal 3, 2026-09-22. Traced from source at pinned commits. File:line evidence for every row is in
[METHOD_LINEAGE_NOTES.md](METHOD_LINEAGE_NOTES.md), written by a delegated code trace. Terminal 3
re-verified the load-bearing items directly (marked ✔).

## Verdict

**No implemented pipeline combines the original purpose-specific encoder with the fixed-service release
line.** No ACS experiment under f4bdf4cd5, 349efa454 or ad2c08872 imports the LoRA, erase-layer or
purpose-encoder code. The only `pcrl/` import there is the SPLINCE baseline. The two lines share a
problem formulation and an auditing philosophy. They do not share an algorithm, data, or a trained object.
The latest method (task-directed Q) is **an alternative implementation of the same problem formulation**,
not a composition.

## The four families

| | F1 original encoder line | F2 fixed services + neural J | F3 residual spectral / erasure variants | F4 task-directed finite release Q |
|---|---|---|---|---|
| Data | Adult, HMDA, Diabetes (+Folktables, CelebA/BIOS variants) | ACS 2018 CA (2017 transport) | ACS 2018 CA (2017 transport) | ACS 2018 CA (development-exposed) |
| Inputs | raw tabular features | 10 allowlisted covariates → PCA32 | PCA32 ⊕ 96 RFF, residualised on H | PCA32 and H_A |
| Frozen / shared | backbone MLP [128,128]→64, **seeded random init, never pretrained** ✔ (run_v2_dataset.py `run_seed` @ dbe0fdc and 0eee48f; lora.py:182) | H_A (4 probs), H_B (2 probs): PCA32 heads for income/employment and coverage, byte-identical | H, PCA32 | H, PCA32, J (comparator only), both teachers, codebook, dictionary, cost table |
| Trained | per-purpose LoRA on every Linear ✔ (lora.py `_linear_modules`), task heads; optional frozen LEACE erase layer | J: PCA32→64→16, BCE on income/employment minus 0.1·(individual + coalition observer penalties) | closed-form spectral map W | residence teacher p(PCA32,H_A), baseline b(H_A) ✔; SEX/RAC1P risk model (Trisk); offsets a_z; then Q by convex solve |
| Permitted-task labels | each purpose's task | income, employment | residence (teacher) | **same_residence** (teachers, cost D) ✔ |
| Protected labels | per-purpose disallowed attributes (ridge-R² constraint, vCLUB) | SEX, RAC1P (observer penalties) | SEX, RAC1P (moment penalties) | SEX, RAC1P in the fitted CMI table; also in the Trisk risk model |
| Who gets a new channel | each purpose gets its own representation z_p | **A only** | **A only** | **A only** ✔ (B always receives H_B) |
| Joint multi-recipient optimisation | purposes trained jointly (shared backbone) | no (one channel; a coalition penalty in J's loss) | no | no. In policy C one Q for A carries both A and AB constraints |
| Disclosure constraint | in-sample ridge linear R² ≤ τ (proxy-Lagrangian) | trained observer penalties | linear moment penalties | CMI ≤ δ in a **fitted finite table** conditioned on 2 (A) or 4 (AB) KMeans/median cells of H ✔ |
| Auditor sees | z_p (continuous) and concatenations | continuous H (+J) | continuous H (+Z16) | continuous H + one-hot token; H-only ancestors in every augmented slate; J never an ancestor ✔ |
| Selected / evaluated | 60 cells, final iterate (56/60 strict at τ=0.05; 54/60 in the best-checkpoint per-seed JSON) ✔ | development, then locked 2017 transport | 2017 transport: C1 beats local controls (14/16 better, 0 worse, 2 unresolved), still worse than J | utility nominee **T0_L_0.01_a17 (local only)** ✔; protection nominee Ttask_C_0.01_a17 |

## Specific answers

1. **Does the latest release use the original encoder or LoRA?** No. It uses the historical PCA32 and H_A,
   plus two residence-supervised teachers. T0 is 32 quantile cells of logit p − logit b ✔. It is **not
   label-free**: the label-free KMeans codes were in the precursors cd895e4 and e3415b9, and both closed
   negative.
2. **Is only A optimised?** Yes ✔. B's view is H_B in every family. No channel for B is implemented.
3. **Is coalition protection imposed on the selected release?** No ✔. The utility nominee is local. Its
   fitted AB CMI is 0.0119–0.0177 > δ=0.01 in 12/12 role×weighting×anchor cells. In the protection
   nominee the AB constraint binds (0.0100) and the A constraints are slack.
4. **Expected loss.** Σ_z Q·(−log f) ✔, not −log Σ_z Q·f.
5. **Solver.** HiGHS (zero or unconstrained budget); CLARABEL, then SCS (positive budget). Status, residuals
   and recomputed CMI are recorded; no dual bound ✔.

## Code-backed diagram (implemented paths only)

```
F1  (Adult/HMDA/Diabetes)
    x ──► frozen RANDOM MLP backbone ──[optional frozen LEACE erase layer]──► repr_proj ──► z_p
          └── per-purpose LoRA on every Linear (trained)               one z_p per purpose
    audits: linear R² / DA / MLP probes on z_p; concatenation [z_1|z_2|z_3] for cross-purpose

F2–F4  (ACS 2018 CA)
    covariates ──► PCA32 ──► frozen heads ──► H_A (to A)      H_B (to B)      [never modified]
                     │
                     ├─F2─► J (neural, observer-penalised) ─────────────► A receives [H_A, J]
                     ├─F3─► spectral Z16 ───────────────────────────────► A receives [H_A, Z16]
                     └─F4─► p(PCA32,H_A), b(H_A) ─► r ─► T0 cell ─► Q[T0,·] ─► ONE token Z ─► A receives [H_A, Z]
                                        (Q fitted with CMI ≤ δ given 2 cells of H_A; policy C adds 4 cells of (H_A,H_B))
    B receives H_B only.  Coalition AB audit view: [H_A, H_B, (J | Z16 | Z)].
    NOT IMPLEMENTED: F1 output → F2/F3/F4;  a channel for B;  joint optimisation of several new channels.
```

## Corrected combined narrative (usable in the paper)

> This line of work studies one problem: releasing information that serves a permitted purpose while
> limiting what specified recipients, alone or together, can newly infer about protected attributes. Two
> implementations were studied, in different settings. When the releaser controls the representation
> (Adult, HMDA, Diabetes), per-purpose adapters on a shared frozen backbone were trained under a
> per-purpose linear-leakage constraint. That work contributes an auditing caution (aggregate one-hot
> scores can hide a rare-class direction) and several corrected guarantees. When published prediction
> outputs are immutable (ACS), a channel is appended for one recipient and audited against the published
> outputs, alone and with a second recipient's outputs. The two implementations were never composed, and no
> result for one is evidence for the other.
