# Methods catalog — summary

(Saved by the owner from the role's final report; the role's own write was refused by the harness.)

Files: `method_catalog.csv` (51 rows, 25 columns; status 23 runnable / 14 adaptation required / 11
unsuitable for this setting / 3 not yet verified; role 12 first runnable benchmark / 19 later benchmark
/ 20 related work only), `CAPABILITY_TABLE.md`, `PRIORITY_LIST.md`. No training; a few small synthetic
checks run from stdin.

## EARLY FLAGS
1. **PCRL's `LEACEEraser` is not LEACE** (origin/main `pcrl/models/baselines.py:120-181`): it projects
   out the ridge-coefficient span orthogonally. Synthetic check: in-sample R² 0.9995 → 0.9965 after it;
   the official `concept-erasure` package removes the cross-covariance to < 1e-14. PCRL "LEACE" numbers
   from this class are under-erased.
2. **PCRL's "R-LACE" is not RLACE** — `scripts/rlace_diagnostic.py` is an INLP-style loop. Official
   RLACE has never been run in either repo.
3. **PCRL "LEACE fails 0/20" (`results/LEACE_SUMMARY.md`) is unaudited** — R² 0.95–0.999 after LEACE,
   refit on test rows with the protected attribute as a literal, rank-deficient one-hot input. Chaining
   erasers cannot explain it (a later linear map keeps zero cross-covariance at zero).
4. **The protected attribute is an input feature in every tabular cell of both repos** — FNF, Fair PCA
   and official LAFTR were tested outside their papers' setting.
5. **durable-guarantees README "3 of 21" should be "3 of 18"** (honest_reaudit.json: 18 'stopped', 3 'breach').
6. **durable-guarantees gates inconsistent; evaluation in-sample** — output surface does not decide the
   verdict in four scripts (does for Fair PCA and FNF); encoders/erasers fit on the probed rows.
7. **Obliviator never reached its own stopping rule** in durable-guarantees — not a case of "passed its
   own test, failed ours".
8. **No registered predictions exist for the core Table-1 baselines** (LAFTR, VFAE, DANN, LEACE, Obliviator).
9. **No license** on FARE, Obliviator, SPLINCE, the RLACE paper repo, MiMiC, CFair, FCRL and others;
   FARE's only LICENSE file is the GPL-3.0 of its vendored LAFTR copy; LAFTR is GPL-3.0. Run-and-cite,
   do not redistribute.
10. **Bibliography:** Stadler et al. ICML 2024 authors = Stadler, Kulynych, Gastpar, Papernot, Troncoso;
    FairNVT = TMLR 2026 (also AFAA workshop at ICLR 2026), title changed in v2, no code found;
    Gitiaux & Rangwala: no code found.
11. **Taylor/Vippathalla/Coon** applies to finite alphabets only; known joint distribution; protects the
    whole database; collusion = "all releases so far"; no code; no venue. Using it as a baseline changes its claim.
12. **Combining releases:** exact linear guardedness survives concatenation when the attribute is erased
    in every release; PCRL's R² ≤ τ does not (hence the concatenation attack flags 26/33); per-recipient
    methods cannot protect an attribute only some coalition members declared; no published
    learned-representation method states a guarantee covering combined releases.
13. **ACS evaluation years are spent** (2018, 2017, 2016 now scored) — the finite-token track needs a new
    population (owner note: this conflicts with other ledgers that call 2016 "admitted but unscored";
    resolved in DATA_EXPOSURE_LEDGER.csv).
14. **`concept-erasure` unpinned in both repos** (0.2.4 installed = latest). PCRL counts disagree across
    files: 56 vs 54 of 60 strict; 7 vs 5 of 60 cleanly compliant.

## Recommended first comparison
Core (ranks 1–6): anchors (clean task-only release, withhold, label-only predictor); LEACE pinned 0.2.4,
per recipient and on the union of sensitive sets; official SPLINCE @fced0d3; official INLP @e1edcc1;
isotropic noise channel; official FARE @89cb1b6. Extended: DANN-style adversary + official-LAFTR fidelity
check on binary cells; VFAE; Obliviator; PCRL erase layer with a no-LEACE arm (key ablation — LoRA vs joint
LEACE alone — registered but never run).

## Missing implementations
No collusion-aware learned method exists (native recipient structure only in FFVAE, a sketch in
Adversarial Forgetting, finite-alphabet information theory, and the project's own OWN-03/OWN-09). No
official code for VFAE, FFVAE, Adversarial Forgetting, Gitiaux & Rangwala, FairNVT, Xie et al. No
multiclass LAFTR/FNF. No de-censoring attacker.

## Verification depth
Subagents inspected repo records and web sources; spot-checked durable-guarantees pins, honest re-audit
counts, PCRL eraser and LEACE baseline code, license + HEAD of 11 repos. No reported outcome rerun. Open
concern from reading the LEACE package trace-limiting code: untested (branch never triggered in 700
synthetic trials).
