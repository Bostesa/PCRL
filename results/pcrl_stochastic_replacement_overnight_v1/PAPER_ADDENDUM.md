# PAPER_ADDENDUM — what Terminal 2 may and may not write

All of this is **2018 development** evidence on repeatedly used pools, validation split only,
task-informed throughout. Nothing below is a confirmation.

## 1. Two corrections that must be disclosed in the methodological description

**(a) The predecessor's audit slate.** `pcrl_utility_extension_v1`'s METHOD §1 promised untouched-J
predictor candidates in every comparison; the implementation routed **H-only** ancestors and audited
`ref_J` as a separate condition. Confirmed against source. Disclose; do **not** rerun, and infer no
historical win — the correction's direction only *raises* measured increments.

**(b) `role_gains`.** The training protection penalty mixes frozen-baseline approximation slack with
incremental disclosure; a constant extension still reports a positive gain. Audited leakage is
unaffected. The magnitude in that study is **unmeasured**, not estimated.

## 2. A precursor claim that must be withdrawn

`pcrl_stochastic_channel_v1`'s `RESEARCH_DECISION.md` asserted the code's residence signal was
"already subsumed by the released service predictions J". **Do not repeat this.** Three separate
problems, all in `PRECURSOR_CORRECTIONS.md`:

* its "J" was a **20-column** view (`H_A` + a 16-coordinate auxiliary channel), not the service
  predictions (C1);
* two finite probes failing establishes neither subsumption nor conditional independence nor
  exhaustion (C2);
* and this study **measured the opposite**: the unquantized A-side view attains strictly lower
  residence loss than `J`, −0.0153 unweighted / −0.0185 person-weighted, with adjusted one-sided
  upper bounds −0.0065 / −0.0079, both below zero after Bonferroni over a frozen family of 68.

Also withdraw the precursor's `probability_decile_partition` numbers outright: that partition was
built from a **label-dependent** quantity (C4).

## 3. What may be claimed from tonight

* A **replacement** contract is the design under which a strict disclosure improvement is even
  available; a `J`-retaining extension cannot deliver one, because an attacker may ignore the
  appended block (C7). This is structural and can be stated as such.
* Prior external disclosure of `J` is **not documented** anywhere in the project. Frame `J`-retention
  as a **chosen, registered threat-model assumption**, never as historical fact.
* The A-side inputs carry residence signal **beyond** `J`'s auxiliary channel (§2 numbers above).
  **Utility only** — no disclosure was measured.
* On a small finite model, a certified exhaustive comparison shows randomization buys **nothing**
  under local-only constraints and a large margin under **coalition** constraints. Scope it to that
  model.

## 4. What must not be claimed

* **No competitive operating point.** The programme closed at the representation gate, before any
  privacy constraint was applied. Nothing here measures a utility/disclosure tradeoff.
* Do **not** present the Stage-R failure as evidence against stochastic replacement — the mechanism
  was never constrained on ACS.
* Do **not** call it a capacity theorem: two code families at two resolutions, one probe family.
* Do **not** call the utility panel a frontier; there is no disclosure axis.
* The synthetic coalition advantage is **not** an ACS result and **not** a general superiority claim
  over deterministic mechanisms.
* No reserved-task claim: residence guided the screen and the (untriggered) nomination.

## 5. Prior-art corrections to carry forward

Established mathematics, to be cited rather than claimed: the single-role feasibility criterion is
Rassouli & Gündüz Proposition 1 verbatim; their Theorem 1 gives the `δ=0` LP including squared-error
and error-probability utilities; convexity follows from joint convexity of relative entropy composed
with affine maps; the privacy-funnel nonconvexity diagnosis (its utility *constraint*) is Makhdoumi
et al. SRLIP (Lopuhaä-Zwakenberg) already quantifies over **all** attribute-subset views, so a
"multi-view conditional constraints are new" claim will not survive review. "Conditional Privacy
Funnel" is an occupied name (Rodríguez-Gálvez et al., ITW 2021; de Freitas & Geiger, *Machine
Learning* 2025) — the object here is a **role-constrained stochastic release**. Optimal transport
(Xu & Strohmer) needs the sensitive attribute at inference and is ruled out on the **contract**.
Cite and distinguish Zamani et al. arXiv:2211.15525 and arXiv:2601.04815.

## 6. Figure

`UTILITY_PANEL.png` / `.pdf`, with its machine-readable twin `UTILITY_PANEL.json`. Caption it as a
utility comparison against same-host `J` with simultaneous intervals, explicitly not a frontier.

Do not expose credentials, bucket names, instance ids or private cloud details in the manuscript.
