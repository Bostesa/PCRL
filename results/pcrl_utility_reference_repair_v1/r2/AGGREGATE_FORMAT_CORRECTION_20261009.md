# Aggregate CSV line ending correction, 2026-10-09

The first post-q22 aggregate build produced an uncommitted 120-row CSV with
CRLF line endings. `git diff --check` rejected its carriage returns as trailing
whitespace before the output commit. The uncommitted CSV SHA was
`a76fbd3f11f38b192062b3ace5a2d8912e40fd3140ad6507a5af97d9ad18855e`;
the paired uncommitted selection JSON SHA was
`35e044b1368816f5fb6491653240035c94a07a6b0c6850e0b5f15a3bcab25322`.

The builder now writes LF line endings. It reopened only the same sealed
aggregate JSON inputs and reran after R3's q22 verification passed. The new
CSV SHA is `7ce186b0181d379d382d0d27d5210b1afdfdc95a127cc8dda14b3f6c3ac2e990`;
the paired selection JSON SHA is
`03795df2c6ca7bbd7952e0f0aa6f9b886d4ee47ba3c249551eb1fdc6d59a857c`.
The code change affects only CSV line endings and the receipt's CSV hash.
No attacker fit, score, utility endpoint, control rule, or comparator changed.
