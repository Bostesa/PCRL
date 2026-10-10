# H1 attack queue admission correction

Date: 2026-10-10. This is an operational correction before any R4 attacker
opened ATT or INNER SEX. Preserve the original 38 R4 worker specs, queued
queue specs, public indexes and R0 attack-stage lock as failed admission
history. No old unit ID may be overwritten or enqueued with changed bytes.

The first R4 finalizer wrote every queue spec without `amendment_sha256`.
R0's global queue requires that field to equal the latest registered
amendment affecting `attacks`. The queue returned
`affected_stage_requires_latest_amendment_sha256` before running a child.
A read-only reproduction on the original B0 seed-0 queue spec returned the
same reason; a copied spec with the current latest attack-stage SHA returned
no admission reason. This identifies queue wiring, not a scientific fit,
utility result or attacker outcome. R5 withdrew its prior queue GO after the
admission check surfaced.

`ATTACK_H1_SOURCE_PLAN.json` SHA-256
`1b42e9a7e7355c27db684cac0a683d91aacd47fcb321da23b9ee44097c48e1bd`
lists exactly 38 successor IDs, each original arm, source seed, cached draw,
attacker seed, worker-spec SHA and queue-spec SHA. Every successor appends
`_H1` to its predecessor unit ID. The source plan is public and contains no
person-level values. It binds the original attack index SHA-256
`e07ae90de911d915941b133510c5012d93fe6eb066e32f13c3752a07f88c0696`,
queue index SHA-256
`bf1a7f9034e8db796b61ad93065cbe03b6b7622941ec3b5fa58fd5463c35d512`
and stage-lock SHA-256
`066dd017af280a1afb6ac661b0bb55bf4a70330f726b82e6b80c8a5dd07c96b6`.

The H1 path retains the same pinned release, public wire, attack roles,
reader slate, task gates, seeds, draws, resource ceilings and 1,140 planned
fits. Its builder must compare each successor worker spec with the predecessor
on every field except unit ID and output destinations. R0 first registers a
dated correction amendment binding this plan. Only then may R4 create H1
worker specs and a new public attack-unit index. R5 audits those bytes, R0
seals and registers a new exact H1 stage lock, and R4 finalizes fresh H1
queue specs with `amendment_sha256` set to that exact newly registered
stage-lock SHA. R5 repeats the queue admission check before any enqueue.
