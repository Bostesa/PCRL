# RELEASE_CONTRACT — which contract this study tests, and on what evidence

**Date: 2026-09-21**, before any new outcome was inspected.

Two scenarios are distinguished. They are different questions, not two framings of one question.

---

## S1 — Primary design comparison: recipients who have never received J

* `H_A` and `H_B` are preserved **exactly**, byte-for-byte, on every pool.
* Release: `A = [H_A, Z]`, `B = H_B`, `AB = [H_A, Z, H_B]`, where `Z` is one sampled token.
* `Z_J` is **not** released. `[H_A, Z_J]` is an **alternative** a designer could have shipped, so J is
  a **comparator**, not an obligatory component.
* Attack slate for an `H + Z` candidate contains only **accessible** `H` ancestors. A `J`-only
  predictor is **never** injected into a release that neither exposes nor reconstructs `Z_J`.

This is the contract this study tests.

## S2 — Previously disclosed J: recipients retain J

* Recipients hold `J`, so `J` is in every conditioning view and every audit slate.
* A replacement **cannot undo** that disclosure. The best attainable is equality, because an attacker
  may ignore anything appended and re-run the `J` attack (`PRECURSOR_CORRECTIONS.md` C7).
* Every actual `J + R` extension must therefore include **accessible ignore-`R` `J` predictors** in
  its slate.
* Any result under S2 is a separate, **history-aware extension** question.

S2 is not tested tonight. The precursor's registration, which adopted S2, is preserved unchanged.

---

## Which scenario the evidence supports

I searched the repository for documentary evidence that a real external recipient operationally
received these people's `J` outputs: `results/**`, `docs/`, `paper-body/`, `*.tex`, experiment
docstrings, and `git log --all`, for recipient/disclosure/deployment/third-party/bank/agency/DUA
terms.

**Verdict: no documented external disclosure of `J`. Prior real deployment history is _not
established_.** The contract used for S1 is therefore explicitly **hypothetical and prospective**.

**The single piece of contrary text is my own precursor's registration**, which asserted it without a
source:

> `results/pcrl_stochastic_channel_v1/REGISTRATION.md:20-21` — "`Z_J` stays in the release and in
> every conditioning view. J was actually released to these recipients and remains accessible; the
> experiment may not pretend it was revoked."

Against it, the project's own records repeatedly describe a stipulated setting and local artifacts:

| evidence | file |
|---|---|
| "**None specifies a real data-holder/recipient agreement**" | `docs/PCRL_APPLICATION_SELECTION.md:229` |
| "a **hypothetical** data owner releasing a fixed per-record interface"; ACS records "**simulate** this interface" | `docs/PCRL_APPLICATION_SELECTION.md:513-517` |
| "**A hypothetical owner** makes one fixed per-record release" | `results/redesign_20260907_acs_transfer_v1/PROTOCOL.md:3-8` |
| "**released arrays** and person-level predictions **remain local**" | `results/redesign_20260909_acs_source_guard_v1/LOCAL_ARTIFACTS.md:3` |
| external-party sense of "release" reserved for describing **prior work** | `docs/PCRL_PRIOR_WORK.md:15-16` |
| "records **simulate** a release, not confidential Census deployment" | `results/redesign_20260908_acs_bottleneck_v1/RESEARCH_DECISION.md:12` |

The data are public US Census ACS PUMS records. The "recipients" `A`, `B`, `AB` are **wire roles** in
a stipulated policy — `REGISTRATION.md:40` defines them as `wire/A`, `wire/B`, `wire/AB` — not
organisations. No named external recipient, DUA, MOU or contract artifact exists anywhere.

**How the contrary sentence is read.** As a *registration boundary*: `Z_J` is inside the frozen
interface this study inherited, and that study forbade itself from quietly dropping `Z_J` to flatter
its own numbers. The operative clause is prescriptive — "the experiment may not pretend it was
revoked" — an anti-cherry-picking rule about an experimental interface, not a record of a transfer.

**Residual ambiguity, flagged rather than resolved.** The clause "and remains accessible" does not
reduce cleanly to interface bookkeeping, and no document explains who would hold it or explicitly
states that `J` never left the machine. So: **unknown-but-undocumented**, and S1 is scoped as
hypothetical rather than asserted as revocation.

**What is therefore claimed and not claimed.**

* Claimed: an S1 result is a statement about a **prospective** deployment in which only `H` was
  previously released, and about **recipients who have never received `J`**.
* Not claimed: that `J` was revoked. Nothing here revokes anything.
* Not claimed: that an S1 result transfers to S2. It does not; under S2 a strict disclosure
  improvement is unavailable by construction.
* The precursor's registration is **not** silently rewritten. It stands, with its S2 assumption, and
  `PRECURSOR_CORRECTIONS.md` records why that assumption made its selection target unreachable.

---

## The decisive toy — why "no added utility over J" does not settle a replacement

Constant `H`; independent fair bits `Y` (useful) and `S` (sensitive); `J = (Y, S)`; `T = Y`.

| release | `I(·;Y)` utility | `I(·;S)` disclosure |
|---|---|---|
| `J = (Y,S)` | `log 2` | `log 2` |
| `J + T = (Y,S,Y)` — the **extension** | `log 2` (**+0 over J**) | `log 2` (**unchanged**) |
| `T = Y` — the **replacement** | `log 2` (**full**) | **0** |

Adding `T` to `J` gains exactly nothing, because `J` already determines `Y`. Replacing `J` by `T`
preserves the useful bit **in full** and removes the sensitive bit **entirely**. So a screen that
measures "does `T` add utility to `J`?" can return zero while `T` is a strictly better release than
`J`. The precursor measured the former and concluded about the latter.

This is an **explanatory counterexample**. It is not an ACS finding, not evidence that an ACS
replacement will work, and not a novelty claim. Pinned in
`tests/.../test_precursor_corrections.py::test_replacement_can_strictly_remove_information_that_an_extension_cannot`.

---

## Routing rules, asserted per role in code

| release | eligible ancestors in its slate | forbidden in its slate |
|---|---|---|
| `H` only | — | anything containing `Z_J` or `Z` |
| `[H_A, Z_J]` (J comparator) | `H`-only predictors | `Z` (the new token) |
| `[H_A, Z]` (**S1 candidate**) | `H`-only predictors | **`J`-only predictors** — `Z_J` is not in this release and is not reconstructible from it |
| `[H_A, Z_J, R]` (S2 extension, not run tonight) | `H`-only **and** ignore-`R` `J` predictors | — |

Feature layouts and routing are asserted per role at audit time, not assumed.
