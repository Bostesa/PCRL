# Tier A: source recovery and historical verification (2018 already-used data only)

All checks below read committed code, the pinned private archive and already-used 2018 data. No 2016
record was read.

- **Archive restore.**
  - Source: `pcrl_task_directed_release_v1/final-5a78a39412f7`. The manifest SHA-256 `5049448…e667`
    matches ARCHIVE_INDEX.
  - Eight parts (0, 1, 2, 5, 6, 11, 16, 17) were downloaded by version ID. Each compressed part hash was
    verified, and all all 95,290 members of those parts were hashed against the manifest.
  - Only 10,463 needed files were restored: encoders, the three maps, the mechanism40 erasers, the
    prepared caches, core evaluation arrays for all anchors, and core audits for anchor 0.
  - The archive itself is unchanged; attacker weights remain there.
- **Frozen objects.**
  - Each solution matches its accepted-map receipt.
  - The encoder hash matches the mechanism40 fitting receipt.
  - `T0_L_0.01_a17` (Q) has full 17-action support in every row. `T0_U_unconstrained_a17` (D17) and
    `T0_U_unconstrained_a33` (D33) are deterministic, with one action per code.
- **Service parity.** For all three anchors and all seven 2018 pools, PCA32, H_A, H_B and J recomputed
  from raw rows are byte-identical to the archived arrays (HISTORICAL_VERIFICATION_ADDENDUM.json).
- **Release reconstruction and predictor replay** (anchor 0 test pool, all 10 families × 5 roles):
  - Archived selected predictors, re-applied to releases rebuilt by this study's loader, reproduce the
    archived per-person losses to ≤ 1.6e-6.
  - The exception is 2 of 2,982 rows. On those rows the macOS-arm64 run of the frozen float32 b(H_A) MLP
    moves the residual across a T0 quantile cut (r = 1.3200093 against a cut of 1.3200094).
  - On the 2018 fitting pools the same effect flips 0.1–0.4% of codes on macOS.
  - On the Linux x86 execution host the encodings are **bitwise identical** to the archive: 0 flips; p, b
    and actions all equal (`private/ENCODING_PARITY_x86_64.json`).
  - Consequence: all 2016 work runs on that single host. Historical byte fingerprints are kept separate
    from cross-platform toleranced compatibility.
- **Headline distinctions.**
  - Recomputed from archived per-person losses with the predecessor bootstrap (10,000 draws, seed
    20260921). All 38 matching registered endpoints reproduce the archived estimates exactly, and their SEs
    to ≤ 4.3e-19.
  - Q minus J, task: −0.016251 unweighted, −0.016394 PWGTP.
  - D17 minus J, task (not a registered 2018 endpoint): −0.017205 unweighted, −0.017654 PWGTP.
  - D33 minus J, task: −0.017937 / −0.018278.
  - D17 minus Q, sensitive recovery: D17 is lower in 7 of 8 points; the exception is AB/SEX PWGTP. All 10
    Q-versus-D17 adjusted differences in the 706-endpoint family were unresolved.
  - Historical scores are retained verbatim; this is verification only.
