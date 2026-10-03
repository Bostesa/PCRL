# Cost and storage

**Machine.** Laptop only (Apple M4 Pro); no cloud. Single-threaded workers (OMP/MKL/OPENBLAS/VECLIB = 1), with at
most 2 heavy workers at a time.

**Elapsed time.** About 1 h 05 m, from custody (08:44Z) to the verified backup (about 09:50Z), plus verification and
reporting. The ceiling was 10 h.

| Job | Wall s | CPU s (user + sys) | Peak RSS |
|---|---|---|---|
| Stages 2/3, Adult | 1,042 | 1,035 | 0.34 GB |
| Stages 2/3, HMDA | 805 | 800 | 0.37 GB |
| S4 coalition | 115 | 115 | 0.42 GB |
| Controls (Adult + HMDA) | 62 | 61 | 0.36 GB |
| Stage 5, conditional FARE (screen + cell) | 461 | 457 | 0.63 GB |
| Inference (2 runs), S5 inference, figures, usefulness, custody, repairs | about 300 | about 290 | 0.98 GB |
| **Scientific total** | | **≈ 2,760 CPU-s ≈ 0.77 CPU-h** | |
| Independent verifier (stages 1–4 replay + S5 + synthetic self-tests) and reviewer checks | | ≈ 0.3–0.4 CPU-h | < 1 GiB |
| **Aggregate** | | **≈ 1.1 CPU-h** (ceiling 12) | **< 1 GiB** (ceiling 6) |

**FARE environment and compilation.** Reused from the output-aware study; no new setup.

**Model fits.** 15,042, from the run ledgers: attackers including grids and seed refits, U2 probes, heads and FARE
trees.

**Units.** 559 unit directories:
- 24 alias records reusing hash-complete output-aware units, without copying them;
- 105 fit-free bank records;
- 87 stage-5 units;
- the rest new attack, probe and head units.

**Storage.**
- New private data: 629 MB, 3,487 files in `~/PCRL_eval_cache_private/odx_v1`.
- A drive copy, verified 3,487/3,487 with uncached reads.
- Laptop free space: about 127 GB (the floor was 5 GiB).
- Public package: aggregate tables, code, figures and reports only.
- Nothing was deleted. Disposable scratch was limited to synthetic FARE test units in the session scratchpad.
