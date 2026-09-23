# Proposed registration wording — NOT applied, NOT submitted

The abstract and title are **registration text**. The venue bars substantial post-registration changes to
either, so nothing here is applied to `main.tex` or to any registration record. This file exists so the
author can decide with the venue's rules in hand.

## Status

`REGISTRATION_READY.md` holds the drafted title and 228-word abstract. Whether registration has happened
is **not known to this terminal**, and a local file is not evidence of it. If it has, treat everything
below as a request the venue must permit; if it has not, the edits can simply be folded in before
registering.

## Proposed edit 1 — scope of the constraint (from the claims-foundation audit)

> "…and use it to evaluate a task-directed finite release optimised under explicit conditional-disclosure
> constraints."

becomes

> "…and use it to evaluate a task-directed finite release fitted under a conditional-disclosure budget for
> that recipient's own view, with the coalition view audited."

**Why:** the selected candidate constrains the local view only; its coalition view is audited, not
constrained. Word count +3.

**If it cannot be changed:** the abstract is not false as written — the framework does offer
conditional-disclosure constraints, and the paper body now states the selected candidate's local scope
explicitly (§V). The audit reached the same conclusion.

## Proposed edit 2 — provenance wording

> "We also correct published claims of our own"

becomes

> "We also correct claims from an earlier version of this line of work"

**Why:** "published" overstates the status of a submitted-but-unpublished paper, and the replacement is
also safer for double-blind anonymity.

## Not proposed

No change to the title. No change to any number, result or conclusion. The abstract's substantive
claims — the task improvement, the unresolved sensitive bounds, the matched deterministic baseline, the
supervision asymmetry and the absent attribution — all survive the audit unchanged.

---

## Update after the prospective ACS 2016 result (2026-09-23)

The registered/drafted abstract in `REGISTRATION_READY.md` describes the development evidence only. The
paper now also carries a completed prospective evaluation, so the paper's abstract has been extended in
`main.tex`. **Whether that extension is permissible depends on the venue's post-registration rules and on
whether registration actually happened — neither is known to this terminal, and nothing has been
submitted or changed at the venue.**

**The as-drafted registration abstract is preserved verbatim in `REGISTRATION_READY.md` and is unchanged.**

### The added sentences, for the author to check against the rules

> We then evaluate two frozen operating points prospectively on a previously unused survey year under a
> lock created before any final label was read. Both pass all eight registered sensitive-recovery clauses
> against the prior channel and both fail both task clauses: their task upper bounds are below zero but
> above the registered 0.003-nat minimum, so each passes eight of ten clauses and neither establishes the
> full claim. Against matched randomisation and withholding controls the randomised release buys task loss
> at the price of significantly higher measured recovery on six of eight endpoints, so no frontier
> advantage for randomisation is established.

### How to judge it

* If the abstract was **not** registered: fold these sentences in and register the combined text.
* If it **was** registered and the venue treats added results as a substantial change: keep the registered
  abstract and leave the new evidence in the body, which is where it is fully reported. The registered text
  remains true — it claims no fresh-year outcome — so nothing in it becomes false by the addition.
* Either way the direction of the conclusion is unchanged: no competitive claim, in either year.
