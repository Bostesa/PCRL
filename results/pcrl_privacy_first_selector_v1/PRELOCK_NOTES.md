# Pre-lock notes (written before the lock and before any outer access)

Source: the three inner audit panels (`INNER_REPORT.json`, `inner_panels/a*/INNER_AUDIT.json` on the host). These are descriptive only and gate nothing (PROTOCOL §6).

## 1. Aliases on the audited rows

| Anchor | Aliases |
|---|---|
| 0 | D1 ≡ D17; TASK_SEL4 ≡ DET_SEL4 |
| 1 | Every release except NM4PF ≡ D17 (D4, R4, R1, D1, DET_SEL4, TASK_SEL4) |
| 2 | D1 ≡ D17; TASK_SEL4 ≡ DET_SEL4 |

TASK_SEL4 equals DET_SEL4 on all three anchors, so the lock merges the two as comparators.

## 2. Selected inner routes for AB/SEX

The routes were chosen on inner_selection:

| Anchor | D17 | D4 | R4 | DET_SEL4 | R1 | NM4PF |
|---|---|---|---|---|---|---|
| 0 | own token (AB view) | **H-only (AB)** | **H-only (AB)** | **H-only (AB)** | own token | own token |
| 1 | own token | = D17 | = D17 | = D17 | = D17 | own token |
| 2 | own token | **H-only (B view)** | **H-only (B view)** | **H-only (B view)** | **H-only (B view)** | **H-only (B view)** |

**What this means.**

- For D4, R4 and DET_SEL4, the fresh token-using AB/SEX attacker did not beat the ignore-channel H-only ancestor on inner_selection, on either anchor where they differ from D17. Their measured AB/SEX recovery therefore equals H's.
- Their AB/SEX contrast against D17 is the same number for all three. On inner_check it is −.0089 unweighted and −.0072 PWGTP (three-anchor means).
- It measures how much D17's own token adds over H. It is not a graded reduction.
- The prior lead (DET_SEL4, outer AB/SEX −.0036 / −.0034) came from this same route switch.

**Consequences, known before the lock and unchanged by it.**

1. **R vs D on AB/SEX is structurally zero.** On all three anchors R4 and D4 either share the same H-only route or are the same law. The lock generator flags these endpoints `exact_zero_same_route`.
2. **D4 vs DET_SEL4 on AB/SEX is structurally zero** for the same reason.
3. **Under this audit, the best a token release can do on AB/SEX is D17's measured gap to H.** Privacy-first selection cannot beat task-first selection on this endpoint once both reach the H route. They can differ only on task and on the other roles.

## 3. Inner_check task (descriptive)

Three-anchor means against D17, unweighted / PWGTP:

| Release | Task |
|---|---|
| D4 | +.0044 / +.0024 |
| R4 | +.0002 / −.0007 |
| DET_SEL4 | −.0003 / −.0009 |

- On the fixed bank, D4's fitted-decoder task sits within the +.001 cap by construction.
- The +.0044 unweighted value is the fresh utility probe on inner_check.

Nothing above changes an arm, endpoint, family, threshold or decision rule. Every registered comparison is scored on the outer role regardless.
