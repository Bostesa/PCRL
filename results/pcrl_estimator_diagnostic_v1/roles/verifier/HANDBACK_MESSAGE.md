# Independent verifier, final message (verbatim covering text)

The verifier's final message arrived at about 18:06Z. It contained the covering text below, followed by the full texts of
HANDBACK_1.md and HANDBACK_2.md "verbatim". At 18:06:55Z the coordinator checked that both files in this folder carry the final
wording (HANDBACK_1.md includes the late N4 addition "one `ps` listing") and recorded their SHA-256:
- HANDBACK_1.md: bef95ab4422263049286ff091347d7083197cd9fd1a8324dc99a4600629ef746
- HANDBACK_2.md: 60063fe1af8c36cce877526159d11bf2284d6cab143b9f3133149bbef0719b76

> Both verifier rounds for EDG1 are finished. Round 1 found one study defect (D1). Round 2 reproduced every summary I checked exactly, and found two last-digit rounding mismatches in the current report (lines 48/124 and line 100).
>
> - **Round 1:**
>   - **Pass:** checks 1, 2, 4, 5, 6 and 7. Check 6 passes only after I corrected my own check.
>   - **Check 3:** merging and A's tie handling pass. The failing part is study defect D1: the B_same / B_indep known score orders posteriors that are exactly tied, using float rounding. As a result B's paired difference is not exactly zero in 15 of the 32 dA = 0 comparisons. You have already accepted D1.
>   - **My own errors:** two, both kept beside corrected runs: an exact-rational comparison of the float LOOKUP reader, and an eyeballed range in the first CHECKS.json.
> - **Round 2:**
>   - **Check 8:** every SUMMARY.json / POST_HOC.json entry I checked equals my own computation from the saved rows to 1e-12.
>   - **Check 9:** three re-run repetitions are byte-identical to the saved lines.
>   - **Check 10:** 62 of 64 report items match the current report (HEAD 523f0bec6). The two mismatches:
>     - lines 48 and 124 say "94.8–95.9%"; the data give 94.8–95.8%;
>     - line 100, the COMP pair row, says "0.092 | 0.075"; the data give 0.091 | 0.074.
>
>     Neither changes a conclusion. Three other slips I found in the 6ceaa0b40 text were already fixed in the current text.
> - **CPU:** about 55 CPU-s in total, one process at a time, one thread. Nothing was written outside `roles/verifier/`.

The coordinator corrected both rounding mismatches in ESTIMATOR_DIAGNOSTIC.md (commit d93eee3be).
