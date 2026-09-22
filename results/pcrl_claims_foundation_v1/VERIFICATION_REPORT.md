# Verification report (Terminal 3)

Environment: macOS arm64, Python 3.14.3, NumPy 2.4.4, SciPy 1.17.1, pytest 9.1.1 (scratch venv). Compute
was CPU only. No ACS model fits, no 2016 data, no paid APIs and no cloud resources were used.

## What was run and its outcome

| Check | Command | Result |
|---|---|---|
| Exact fixtures F1–F12 | `python analysis/pcrl_claims_foundation_v1/exact_fixtures.py --json results/pcrl_claims_foundation_v1/FIXTURE_OUTPUTS.json` | all assertions pass (rational arithmetic; F10 is a float spot-check) |
| Test suite (owned) | `python -m pytest -q tests/pcrl_claims_foundation_v1` | 10 passed |
| Dominant-axis replay | `python analysis/pcrl_claims_foundation_v1/replay_dominant_axis.py > DOMINANT_AXIS_REPLAY.json` | deterministic (sha256 c6ddb934…a58); Table I means reproduce |
| Historical precision diagnostic | `python analysis/pcrl_claims_foundation_v1/historical_precision_diagnostic.py` | float32 residual up to 1.6e-2; float64 1e-15 (synthetic) |
| Committed separation fixture | `python experiments/pcrl_task_directed_release_v1/fixtures/pcrl_score_refinement_fixture.py` @ f4bdf4cd | byte-identical to committed JSON |
| Main-branch patch | 4 tests in the patch, run in a detached scratch checkout of origin/main | 4 pass patched; 3 of 4 fail unpatched (the counterexample test passes on both, by design) |
| Manuscript patch | `git apply --check`, then latexmk on the patched copy of 30a6fd19e | applies; 0 undefined refs or cites; 0 overfull; 6 pages |
| Strict-pass recount | from committed per_seed_results.json and dominant_axis_audit.json @ 0176f149 | 54/60 (best checkpoint), 56/60 (final.pt) |
| Selected-release solver and CMI | read from MATH_REPLAY_FINAL_anchor{0,1,2}.json | status optimal / optimal_inaccurate / optimal; AB CMI 0.0119–0.0177 |
| 2017 transport family | read from TRANSPORT_RESULTS.md | 14/16 better, 0 worse, 2 unresolved; max-\|t\| c=2.93 |
| Critical value | scipy `norm.isf(0.05/1412)` | 3.9735 (matches z=3.97) |
| Prior-art load-bearing items | arXiv abs pages for 2601.21859 and 1712.08500 (T3 direct), plus the delegated full-text pass | confirmed |

## Limits

- **Fixtures and proofs.** The fixtures instantiate or refute statements; universal claims rest on the
  written proofs in THEORY_REPAIRS.md. F10 (convexity) is a numerical spot-check of a standard fact.
- **Numerical replay versus model replay.** The dominant-axis and pass-count checks replay stored
  statistics. No checkpoint was loaded and no representation recomputed. The float32 explanation of the two
  residual cells is supported by source reading and a synthetic reproduction, not by re-running those cells.
- **Headline numbers not recomputed here.** The Q-vs-J bounds, D17 comparisons and byte parity are taken
  from Terminal 1's and Terminal 2's verification records (T1: 38/38 endpoints recomputed from archived
  per-person losses). Terminal 3 did not open per-person arrays.
- **Delegated passes.** The code-lineage and branch-discovery passes and the full-text prior-art pass were
  run by delegated agents. Every load-bearing item they reported and that is used in a repair was
  re-checked directly by Terminal 3:
  - T0 construction; partitions; nominee locality and AB CMI; the random backbone and LoRA targets; the
    54/56 recount; the public-main API.
  - Taylor et al. and Rassouli–Gündüz abstracts; the misattribution in REVIEW_TO_EVIDENCE.
  Recounts that only the agents performed (VICReg 24/24+18/18, rank-8 18/18, cross-purpose 22→8) are
  marked as such.
- **Review record.** No genuine review text was available. Concern attribution rests on a committed
  project paraphrase.
- **Not checked:** the 0.003-nat hardware-transport figure; the "three stronger-attacker searches"; C3
  (subsumption withdrawal); the NeurIPS 26/33 and 22/33 figures.
