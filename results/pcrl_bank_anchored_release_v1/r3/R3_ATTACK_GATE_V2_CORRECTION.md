# Attack gate interpretation correction — 2026-10-10

The first write-once `*_TWO_SEED_POINT_GATE.json` records and
`R3_POINT_NOMINATION.csv` combined the 12 B0-relative FIT proxy clauses with
the 24 original U0/Ucal INNER clauses when setting attack eligibility. That
interpretation conflicts with `PROTOCOL.md` (Evaluation and nomination),
`r3/EVALUATION_CONTRACT.md` (Three separate cost roles) and the sealed
`CAMPAIGN_LOCK.json`: fresh attack admission and J nomination require all 24
INNER clauses alone. The FIT proxy is a separate formulation diagnostic.

The original gate files, nomination and all queued utility CSVs/receipts are
retained byte-for-byte. `*_TWO_SEED_ATTACK_GATE_V2.json` rechecks their pinned
source receipts and reports FIT and INNER counts separately. Its
`all_two_seed_task_reference_point_caps_pass` field means INNER 24/24 only.
The write-once `R3_ATTACK_NOMINATION_V2.csv` is the effective attack roster.
No threshold, score, source row, policy, realization or numerical tolerance
changed. R0 owns the dated campaign amendment before attack registration.

J has INNER 24/24 and FIT proxy 10/12; it is the locked nominee and enters
fresh attacks. S21 has INNER 22/24 and is excluded. B0, T, D, P, L and S12
have INNER 24/24 and enter attacks; W_Q and P_STAR retain their separate
historical original-INNER admission receipts. FIT proxy counts remain visible
in every V2 gate and the corrected nomination table.

Git custody note: while R3 had staged 65 aggregate utility CSV/receipt and
original combined-gate/nomination files, R5's concurrent shared-index commit
`9019442aa` included those exact staged R3 bytes with its independent replay.
R5 did not edit the R3 files. A second shared-index race occurred while R3
staged its V2 code and receipts: R4's concurrent commit `bbf17e618` included
those exact R3 files together with R4's attack builder correction. R4 did not
edit the R3 files. The records remain intact; no history rewrite was attempted.
