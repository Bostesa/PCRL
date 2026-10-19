# Errata (registered and campaign texts; the locked files themselves are never edited)

## E1 (written 00:33:42Z): RULES.md section 7, T0 wording

RULES.md section 7 (T0) says "all 28 conditions per version present and finite". The registered evaluator gives 14 conditions per
version (12 clauses + 2 guards), 28 over both versions; RULES.md section 1 and the code (`tsz.rules.trigger` through
`ohn.rules.conditions`, which requires the 14 conditions of one version) use 14 per version. Found by the independent verifier (V2,
00:29Z). No outcome depends on the wording: the trigger record checks 14 per version on both versions (28 in total).

## E2 (written 01:57Z): the A-SC1 disclosure sentence, wherever it is quoted

The required disclosure sentence of ADJUDICATION_SEEDCHECK.md section 5 ("... Amendment A-SC1, made after this check's outcome and
before any reader-stability label was computed or read, ...") is quoted verbatim in conditional/CLAIMS.json (reader_seed_check.
disclosure) and in the decision documents. It holds for the coordinator and the statistical reviewer only: the independent verifier
had already computed its own reader labels (unit VER_READERS_V1, 01:32:59-01:33:09Z, after the seed-check record of 01:31:29Z and
before the ruling at 01:36:00Z) and stored them unread until the claims were computed (PROTOCOL_DEVIATIONS items 11-13). The locked
records are not edited; this qualification applies to every quotation (claims review M5).
