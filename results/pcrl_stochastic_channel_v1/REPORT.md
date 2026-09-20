# REPORT — `pcrl_stochastic_channel_v1` (interim: stages 1–4 complete, G0 recorded, stage 5 blocked)

Branch `research/pcrl-stochastic-channel-v1`, based on `0517c06a7`. **No ACS pool has been read by
this study, no release has been fitted, no cloud compute has been created, and $0 has been spent.**
2016 is sealed and untouched. `main` and all unrelated work are untouched. No historical result or
decision has been revised.

## Headline

Three things were established, and one of them changes what this line of work can claim.

1. **Both of the review memo's code claims are true** (`VERIFICATION.md`), and seven of its
   literature characterizations are wrong or materially incomplete in ways that narrow the
   defensible contribution.
2. **The finite-channel machinery works and the randomization payoff is real in the model**: at
   exactly zero leakage the solver beats the best deterministic perfectly-private map by **6.5×**,
   and independently rediscovers the review's hand-constructed channel up to relabelling.
3. **The `.001`-nat operating point is not confirmable on this evidence base, by any mechanism.**
   This was found before any compute was spent, and it is the most consequential result here.

## 1. Verification (`VERIFICATION.md`, commit `dd7f8e1d2`)

**Confirmed.** The extension's audit slate routes H-only ancestors (`H_A_WIDTH = 4`) and contains
no predictor reading `[H_A, Z_J]` while ignoring `R`, although METHOD §1 promised one. Sharpened:
the *rationale* METHOD gave — that a larger input can make a finite learner worse — is served only
by an in-slate J candidate, not by auditing `ref_J` separately, because the differenced numbers come
from independently selected slates. The bias runs in the direction that flatters the release.

**Confirmed.** `role_gains` mixes baseline approximation slack with incremental disclosure. Executed
in `diagnostics.py`: a constant extension, where `I(S;R | H, Z_J) = 0` by construction, still
reports a positive gain, and the statistic is one-sided because `clamp(min=0)` is applied to an
`argmax`. **Scoped honestly**: the magnitude decays from 0.96 to 0.00004 nats as the frozen baseline
goes from 10 to 3000 full-batch steps, and the real `p0J` is fitted far past the top of that ladder,
so the size of the effect in the study is recorded as **unmeasured**, not estimated. It affects the
interpretation of `beta`, not the pilot's audited leakage.

Pilot numbers in the memo (`−0.0021` residence, `+0.0351` A/RAC1P for `U_r2`) reproduce exactly.

**Seven disagreements with the memo**, from the primary sources. The load-bearing ones: the
unconditional nullspace test is verbatim Rassouli & Gündüz Proposition 1 and the `δ = 0` linear-cost
problem already has a published LP, so that part is a citation not a contribution; SRLIP quantifies
over **all** attribute-subset views at once under a stronger pointwise metric, so a "multi-view
conditional constraints are new" claim will not survive review; "Conditional Privacy Funnel" is an
occupied name (Rodríguez-Gálvez et al., ITW 2021), so the object here is called a **role-constrained
stochastic release**; and optimal transport is ruled out on the *contract* — it needs the sensitive
attribute at inference to select the map — not on directness. Surviving defensible novelty: the
deployment contract and the nested local/coalition role structure solved as a convex program.

## 2. Stage A repairs (`STAGE_A.md`, commit `14b7528a2`)

In-slate untouched-J predictors added (`slate.py`), with the separate-comparator vs in-slate-
predictor distinction now explicit in candidate metadata. Hooked into `build_audits` through an
opt-in, keyword-only `extra_candidates` that defaults to `None`; historical behaviour is unchanged
and the historical tests pass. The routing validates itself: `build_audits` already re-scores every
routed candidate against its recorded validation metrics, which can only pass if the selected
columns reproduce the J wire bit for bit.

**Registered in advance:** this correction can only *raise* measured increments, so it makes the
screen harder. It is not a repair, and nothing historical is re-derived from it.

Review fixtures vendored byte-for-byte with sha256 pinned and independently recomputed in
`fractions`. 29 tests; full repository suite **772 passed**.

## 3. Registration (`REGISTRATION.md`, commit `c1d5bd188`)

Locked before machinery: contract, assumptions and their limits, per-arm label use, budget ladder,
selection and stopping rules, seven gates, and **eight directional predictions with confidences**,
including the statement on the record that *I expect this study to close negative*. `Z_J` stays in
the release and in every conditioning view because J was actually released. No rate constraint may
ever be added. The supervised residence diagnostic is separated from the label-free candidate, and a
supervised success can never become a claim about a task excluded from training.

## 4. Finite-channel machinery (`STAGE_4.md`, `VALIDATION_FINITE_CHANNEL.json`, commit `ccfe0b575`)

| validation | result |
|---|---|
| randomization payoff at `δ = 0` | cost **0.0512** vs **0.3333** for the only deterministic perfectly-private map — **6.5×**, at recomputed leakage `1.9e-9` |
| agreement with the review | solver found `P(Z=0\|T) = (0.999999, 0, 0.846256)`; review's hand construction `P(Z=1\|T) = (1, 0, 0.846154)` — the same channel up to relabelling, task info `0.485552` vs exact `0.4854855270534466` |
| nullspace test | feasible, `null_dim = 2`, spectrum `[0.2694, 0, 0]`, stable from tolerance `1e-14` to `1e-6`; correctly infeasible when `T` determines `S` |
| both roles necessary | XOR channel costs **0.000000** under the local constraint alone, **0.499957** once the coalition role binds; reverse counterexample has coalition `0` with local `log 2` |
| **coarse conditioning** | binning the coalition view drops its measured increment from **`log 2`** to **exactly `0`** |
| LP reference | at `ε = 0.05`, realised max log-ratio exactly `0.0500`; at matched achieved CMI the MI budget is the weaker constraint, as expected |

That fifth row is the executable form of why no fitted-model result may be called a certificate.

Solver honesty (G4): status strings are not accepted; every channel is re-verified by recomputing
`cmi`, a second solver is tried when verification fails, and all attempts are recorded.

## 5. G0 — precision, and why it matters more than anything else here

`gates/G0.json`. Source: `results/pcrl_nonlinear_rank_v1/PAIRED_INTERVALS.csv` — 2018 development
pools, paired-household bootstrap, the same four sensitive endpoints, five comparison families.

Adjusted one-sided half-width `z × SE`, median by endpoint (nats):

| endpoint | unweighted | person-weighted |
|---|---|---|
| `A/SEX` | 0.0072 | 0.0080 |
| `AB/SEX` | 0.0069 | 0.0077 |
| `A/RAC1P` | 0.0118 | 0.0136 |
| `AB/RAC1P` | 0.0103 | 0.0121 |

* **Fraction of sensitive rows with adjusted half-width above `.001`: 1.00.**
* Smallest adjusted half-width anywhere: **0.00220** nats.
* Smallest **unadjusted** one-sided-95% half-width anywhere: **0.00141** nats.

**The multiplicity adjustment is not the cause — the margin already fails unadjusted.** Even with a
point estimate of exactly zero, the one-sided upper bound on any sensitive endpoint lands at best at
1.4× the `.001` screen, and typically at 5–9×.

### What this does and does not mean

* A `.001`-nat **confirmation** claim is not demonstrable at this sample size, on any endpoint,
  under any weighting — **for any mechanism, not just this one.** More fitted channels do not supply
  more independent households. This is a property of the evidence base.
* A `.001` point-estimate **screen** remains runnable and meaningful; that is what the predecessor
  used and failed. The limit is on confidence statements.
* Per the registration, G0 constrains claims and does not fail the study. The budget ladder is to be
  set from this gate **in advance**, which is what is happening, rather than relaxed afterwards.

### Registered prediction P8: CONFIRMED, and more strongly than predicted

P8 (50%) said the precision check would show `z × SE > .001` on **at least one** endpoint. It holds
on **every** endpoint and every weighting, 544/544 rows, and also holds unadjusted. Recorded as
predicted-and-confirmed; the under-statement is recorded too.

## 6. Status of the other registered predictions

Unresolved, because Stage 5 has not run: P1, P2, P3, P4, P5, P6. P7 (J anchors raise measured
increments) is structurally argued in `VERIFICATION.md` §A but not yet measured.

## 7. Why stage 5 stopped, and what it needs

Two reasons, both reported rather than worked around.

**Access blocker.** `aws sts get-caller-identity` returns `Your session has expired. Please
reauthenticate using 'aws login'.` Stage B needs the stored release arrays (`H_A`, `Z_J`) and the
A-side inference inputs, which are **not present locally**: `anchors.npz` and `releases.npz` do not
exist in any worktree, having been archived to S3 and unlinked under `DELETION_LEDGER.md` after
verified read-back. The archive bucket has no lifecycle rule, so nothing is at risk of expiry; it is
purely a credentials problem.

**A decision that is not mine to make.** G0 changes what $50 of compute buys. It cannot buy a
confirmable `.001` result, because no amount of compute can. It can still buy a development answer
to a real question — *does randomization change the capability/disclosure tradeoff on point
estimates where the deterministic channel failed?* — which is what the registration scoped. That is
worth knowing and is bounded. But the budget should be committed with that constraint understood,
not discovered afterwards.

## 8. What is already usable, regardless

Even if stage 5 never runs, this branch carries:

* a confirmed, sourced correction to the predecessor's audit slate, with the fix implemented,
  tested, and its direction registered in advance;
* an executable demonstration that the training penalty was not measuring what its name says, with
  the magnitude honestly scoped as unmeasured in the study;
* validated finite-channel machinery, including the coarse-conditioning counterexample that bounds
  every claim this family of methods can make;
* a narrowed, source-checked contribution statement that will survive review, replacing three claims
  that would not have;
* **G0**, which is a result about the whole research programme and was obtained for free.

## 9. Reproduction

Python 3.13, cvxpy 1.9.3 (CLARABEL + SCS). Data-free:

```
python -m pytest tests/pcrl_stochastic_channel_v1/ -q      # 49 tests
python -m pytest tests/ -q                                  # 772 tests, full suite
```

`cvxpy` is a new dependency, used only by this study; no historical module imports it. The fixture
environment here is torch 2.14.0 / numpy 2.5.3, which is **not** the study environment
(`results/pcrl_utility_extension_v1/REPRODUCE.md`: torch 2.10.0, numpy 2.4.2). Every test added is
data-free and establishes a mechanism or an invariant, never a study number, so the difference is
immaterial; any reproduction of study numbers must use the pinned environment.
