# SaTML 2027 — registration-ready title and abstract

**Not submitted. Not registered. Nothing sent to anyone.** Prepared 2026-09-21 from committed evidence.
Substantial changes to either are barred after abstract registration (22 Sep 2026 AoE), so both are
written to survive every outcome that is still open.

## Title

**Releasing a Feature Channel Beside a Prediction Service You Cannot Change: An Evaluated Release Problem
and Six Negative Mechanism Results**

Shorter alternative, same commitments:

**An Immutable Published Service, a Second Channel, and Six Negative Mechanism Results**

## Abstract (233 words; counted on the body text alone, synchronised with papers/pcrl_satml_v1/main.tex)

> An organisation already publishes a prediction service: fixed probability vectors that downstream
> recipients consume and cannot be asked to re-accept. We formalise and evaluate what happens when it adds
> a second, reusable feature channel for one recipient without altering a published number, while limiting
> what that recipient, and that recipient colluding with another, can newly infer about sex and race. Our
> contribution is an implemented release interface and measurement contract: per-recipient and coalition
> views, independently fitted attackers, incremental disclosure reported beside absolute disclosure, and
> negative increments retained unclipped. On a locked, previously unused survey year, a penalty built from
> the coalition's joint view beats an equal-strength and an equal-total-mass local control on 14 of 16
> sensitive cells, none worse; its utility cost is bounded by a point rule, not by an interval. Six
> subsequent mechanisms fail to improve on the strongest channel we already had, under both registered
> senses of competitiveness: lower disclosure at comparable utility, and higher utility at bounded
> disclosure. Adapted published erasers beat our own mechanism. A capacity diagnostic then shows why the
> most recent attempt could not have succeeded as designed: a label-free code of the permitted inputs
> carries genuine signal for the held-out task, yet adds nothing measurable once the already-released
> service predictions are in the view---a result about appending, not about replacing. We report the release problem, the evaluation design, and every
> claim this revision withdraws.

---

## Commitments this abstract makes, and does not make

* **Does not** mention a stochastic or randomized mechanism as a result. The solver was never run on data;
  the study stopped at a capacity gate before any channel was constrained.
* **Does not** promise any ongoing or future result.
* Claims confirmation only for the locked-year coalition effect. Everything else is labelled development.
* "adds nothing measurable" is the fitted-probe result, not an information-theoretic subsumption claim,
  and it is stated as a result about appending rather than about replacing the channel.
* No bound on mutual information, no certificate, no survival against a stronger attack.

## Word count

Body text of the abstract only, excluding the title and the blockquote markers:

```
233
```

Re-run before registration: `sed -n '/^> /p' SATML_TITLE_ABSTRACT.md | sed 's/^> //' | wc -w`
