# VERIFICATION — independent check of the 2026-09-20 mathematical review memo

Stage 1 of `pcrl_stochastic_channel_v1`. This document records what was checked against source,
what was confirmed, and **where this study disagrees with the memo**. Nothing here is a new
empirical result; no release was fitted, no ACS pool was read, 2016 remains sealed.

Reviewed at `ba531ab4` (pilot closing commit, branch `research/pcrl-utility-extension-aws-v1`).
Branch base for this study: `0517c06a7`.

---

## A. Code claim 1 — untouched-J predictors are NOT candidates in an extension's slate

**Memo says:** "The method says untouched-J predictors are candidates within every extended
release. The implementation routes H-only ancestors, while J is audited as a separate comparator."

**Verdict: CONFIRMED.**

Evidence chain:

| step | source | fact |
|---|---|---|
| wire layout | `experiments/pcrl_direct_adversarial_v1/inputs.py:57-59` | `H_A_WIDTH = 4`, `H_B_WIDTH = 2`, `A0_WIDTH = 16` |
| release | `results/pcrl_utility_extension_v1/METHOD.md` §1 | `wire/A = [H_A, Z_J, R]`, so `H_A` = cols 0-3, `Z_J` = cols 4-19, `R` = cols 20+ |
| confirms base | `experiments/pcrl_utility_extension_v1/program.py` `leace_on_extension` | `base = ext.HA + ext.Z_WIDTH` = 4 + 16 = 20 |
| ancestor source | `experiments/pcrl_direct_adversarial_v1/run_dev_2018.py` | `DEV_H_AUDITS = 'results/{DEV_NAME}/seed_{seed}/H/audits'`; `ancestor = audit.load_audits(resolve(DEV_H_AUDITS...))` — the ancestors are the **H** condition's candidates |
| routing | `experiments/acs_spectral_audits.py` `route_ancestor` | view `A` → `range(4)` = `H_A` only; view `AB` → `[0,1,2,3,width_a,width_a+1]` = `H_A + H_B` |

So the extension's candidate slate is: fresh five-candidate + kernel families fitted on the **full**
extended wire (which contains `R`), the canonical `B` candidates, and **H-only** ancestors routed to
columns 0-3. **No candidate anywhere in that slate sees `Z_J` while ignoring `R`.** `ref_J` is a
separate audited *condition*, compared at the metric level as `p['gains'][e] - j['gains'][e]`
(`program.py` `screen`), with its own independently selected slate.

### Sharpening the memo — this is not only a wording mismatch

METHOD §1 gives a *rationale* for the promise: "Finite fitted probes can move either way (a larger
input can make a finite learner worse), which is why untouched-J predictor candidates are part of
every comparison." That rationale is **only served by an in-slate J-only candidate.** Auditing
`ref_J` as a separate condition does not serve it, because the two numbers being differenced come
from two independently selected slates: if the extension's fresh learners are handicapped by the
wider input, nothing in the extension's own slate recovers the J-only predictor's performance, and
the measured increment is biased *downward* — i.e. in the direction that flatters the release.
The memo's parenthetical reading ("`ref_J` is audited under the same slate") is what the code does;
it is not what the stated rationale requires.

### Direction of the correction — confirmed, and it cannot rescue the pilot

`experiments/acs_fixed_predictions_audits.py` `select_pools` selects by
`min(ids, key=lambda cid: (validation_scores['log_loss'], cid))`. Adding a candidate to a pool
selected by argmin validation log loss can only **lower or leave equal** the selected validation
loss. Lower attacker loss → higher measured recovery for the extension → **larger** increment over
J → the `.001`-nat screen becomes *harder*, never easier. The memo's conclusion holds: no historical
win can be inferred from this correction and it does not justify a rerun. Test-split losses are not
monotone under validation selection, so the descriptive test table in `RESEARCH_DECISION.md` §3
could move either way; it carries no claim and is not re-derived here.

## B. Code claim 2 — `role_gains` can be positive for a constant R

**Memo says:** the training penalty "mixes baseline approximation error with incremental
disclosure"; a correction seeing H and J can beat the frozen baseline even when R is constant.

**Verdict: CONFIRMED, structurally and executably** — but the *magnitude* is baseline-dependent and
must not be transferred to the study. See the measured ladder below.

`experiments/pcrl_utility_extension_v1/extension.py` `role_gains`:

```
base  = fold['base'][role][ix]                       # frozen p0J logits
ref   = F.cross_entropy(base[valid], y[valid])
feats = torch.cat((fold['views'][role][ix], r), 1)   # views ALREADY contains Z_J
cand  = ref - F.cross_entropy((base + ens.nets[fam](feats))[valid], y[valid])
```

`fold['views'][role]` is built as `torch.cat((service_view(r, t_ha, t_hb), t_zj), 1)`, so the
correction network's input is `[H view, Z_J, R]`. The frozen baseline `p0J` is a finite-capacity
`ServiceBaseline` MLP trained with Adam for `config.p0_epochs` on the `p0_fit` fold — it is not the
Bayes predictor on its own inputs. A correction network on `[H view, Z_J, const]` therefore has
headroom to reduce cross-entropy with **zero** contribution from `R`.

Two amplifiers the memo does not mention:

* `torch.clamp(s[argmax], min=0.0)` — the gain is floored at zero, so baseline-approximation slack
  can only push the reported gain **up**, never down. The quantity is not mean-zero under the null.
* `int(torch.argmax(s.detach()))` selects the **largest** gain across families, which maximises the
  approximation-slack component as well as the disclosure component.

METHOD §3's "the incremental gain is zero at initialisation by construction" is true (the correction
nets are zero-initialised) but says nothing after the first optimizer step.

### Measured — `experiments/pcrl_stochastic_channel_v1/diagnostics.py`, artifact `CONSTANT_CHANNEL_DIAGNOSTIC.json`

A constant extension (`R ≡ 0`, column std exactly `0`, so `I(S;R | H, Z_J) = 0` by construction) is
run through the **shipped** `role_gains` after training the correction slate for 120 steps. Reported
gain, seed 0, synthetic fold of 512 rows:

| role | reported gain (nats) |
|---|---|
| `A/public_coverage` | 0.5041 |
| `A/SEX` | 0.4972 |
| `A/RAC1P` | 0.9048 |
| `AB/SEX` | 0.5250 |
| `AB/RAC1P` | 0.9568 |

**This magnitude is an artifact of a deliberately under-fitted baseline and does not transfer.**
The slack is the frozen baseline's distance from its own optimum, so it decays as the baseline is
better fitted (max over roles, seed 0):

| frozen-baseline full-batch steps | 10 | 50 | 200 | 800 | 3000 |
|---|---|---|---|---|---|
| max reported gain on a constant channel | 0.9568 | 0.7770 | 0.2680 | 0.0025 | 0.00004 |

The real study fits `p0J` for `p0_epochs = 60` minibatch epochs (`extension.py:70,185`, batch 256),
which is far more optimiser steps than the top of this ladder. **The honest conclusion is that the
mechanism is real and its size in the study is unmeasured**; quantifying it there requires the real
`seed_context`, which needs the 2018 pools and is not attempted in this stage. Note the scale it
would have to beat: the pilot's screen is `.001` nats, and the 800-step rung of this ladder is
`.0025`.

Gain is exactly `0` before the first optimiser step, confirming METHOD §3's initialisation claim
and its limits (`test_gain_is_zero_before_the_first_optimizer_step`).

**Scope limit (agrees with the memo):** `role_gains` is the *training* penalty only. The audited
leakage in `PILOT_SCREEN.json` / `metrics.json` comes from `run_dev_2018.evaluate_seed` →
`acs_spectral_audits.build_audits`, which fits attackers fresh. The pilot's measured increments are
**not** invalidated by this. What is affected is the interpretation of `beta`: the protection term
was partly paying to close the frozen baseline's own approximation gap, so the effective pressure on
`R` at a given `beta` was lower than the nominal penalty suggests.

## C. Numbers in memo §1 — CONFIRMED exactly

`RESEARCH_DECISION.md` §3, row `U_r2`: residence `-0.0021`, `A/RAC1P` `+0.0351`. Matches the memo's
".0021 nats" residence improvement and ".0351 nats of extra local race recovery". The memo's
characterisation of the pilot's strongest utility number as teacher-reconstruction (not residence)
matches §1's verdict row: "Yes, on the proxy (reconstruction), not demonstrated on residence".

## D. Fixtures — reproduced

`stochastic_channel_fixtures.py` (sha256 `998573c81c4b5558ea524a09eaddbc893a19ddd9f7f065949685336c4092c085`,
self-consistent with the value recorded inside its own JSON) runs on the standard library; all
assertions pass; regenerated JSON is byte-identical to the shipped artifact. The separation is real:
in the `P(S=1|T) = (1/10, 4/10, 9/10)` model all five set partitions of three states — every
deterministic map up to output relabelling — give either 0 nats of task information (the constant
map, the only exactly private one) or nonzero leakage, while `P(Z=1|T) = (1, 0, 11/13)` gives exact
independence checked in `fractions` at 0.4855 nats of task information.

---

# E. Disagreements with the memo, from the primary sources

These change the framing and, in two cases, the claim that can be made.

## E1. The nullspace test is not an adaptation — the unconditional form IS the cited proposition

Rassouli & Gündüz Proposition 1, verbatim: perfect privacy is feasible for `(X,Y,W)` iff
`dim(Null(P_{X|W}) \ Null(P_{Y|W})) != 0`. That is literally `ker(A) ⊄ ker(B)`. The memo presents
its §3 check as "a simultaneous-view adaptation of finite perfect-privacy algebra", which is fair
for the multi-role case, but the single-role form should be cited as the proposition, not derived.
Proposition 1 is itself credited in its own proof to a prior Theorem 4 — cite the origin.

**The memo understates the overlap on the objective too.** Their Theorem 1 already gives an **LP**
for the optimal perfectly-private mapping, and their Conclusions state the LP reduction also holds
when utility is mean-square error or probability of error. So *a fixed linear cost at `δ = 0` is
already solved in the literature.* What is not in that paper: the conditional constraint
`I(S_r;Z|C_r) = 0`, the intersection over several roles **and** over conditioning values, and any
treatment of `δ_r > 0`.

**Correction to a claim the memo does not make but a reader might infer:** there is no
cardinality impossibility theorem. `|T| > |S|` is *sufficient and generic*, not necessary — two
identical columns of `P_{S|T}` give feasibility with `|T| ≤ |S|`. Do not assert necessity.

**Useful and currently unused:** the paper's asymptotic result — when perfect privacy is infeasible
the slope of `g_ε` at the origin is finite, when feasible it is unbounded — makes the nullspace test
a sharp predictor of small-`δ` behaviour. That is a better argument for running the test first than
the one the memo gives, because our actual regime is `δ > 0`.

## E2. "Simultaneous multi-view conditional constraints are new" will not survive review

The memo's §7 already declines to call side-information constraints new, but its §2 row for
Lopuhaä-Zwakenberg ("finite mechanisms protecting against **specified** additional observations")
understates the source. Definition 3 of `ε`-SRLIP quantifies **over every subset `J ⊆ {1..m}` of
attributes and every `x^J`** simultaneously — a *superset* of our two-view family whenever `C_A` and
`C_AB` are subsets of observed attributes, under a strictly stronger pointwise likelihood-ratio
metric.

The memo also **overstates** that source in the other direction: its Theorem 3 is only a
*sufficient* product construction (per-attribute `ε/m` mechanisms compose to `Σε_j`-SRLIP); the
paper says explicitly it "does not give us the optimal `ε`-SRLIP protocol". There is no
optimal-SRLIP result.

Calmon et al. §2 likewise already proposes conditioning the fairness constraint on a named sub-view
`B` — from 2017 — which the memo's row for that paper does not mention.

**Consequence for the contribution statement:** of the four ingredients (average conditional-MI
constraints, heterogeneous per-role budgets, nested local-vs-coalition view structure, linear cost
solved as a convex program), only the **nested role structure and the deployment contract** are
individually unclaimed. Lead with the contract and the tractability result. Do not lead with
"multi-view" or with heterogeneous budgets (see E5).

## E3. Calmon is (quasi)convex, not convex

Their Proposition 1 is conditional: "(quasi)convex optimization if `Δ` is (quasi)convex and `J` is
quasiconvex in their respective first arguments", and their own chosen `J(p,q) = |p/q − 1|` under
the pairwise constraint yields **quasiconvex**, not convex. The memo's row says "convex". Two
further facts the memo omits and that matter for us: their mechanism **requires the protected
attribute `D` at apply time**, and the constraint is on the **outcome** `Ŷ`, not on a
representation. Both are the reasons that line of work does not transfer to our contract.

## E4. Makhdoumi — accurate, and sharper than the memo uses it

Verbatim: "the objective function `I(S;Y)` is convex in `P_{Y|X}`. However, because of the
constraint `I(X;Y) ≥ R`, the Privacy Funnel is not a convex optimization." **The nonconvexity comes
specifically from the utility constraint** — a convex function bounded *below*. Our linear cost
removes exactly that. That is the crisp reason our problem is tractable where the funnel is not, and
it is a better sentence than "replacing a neural penalty with the words mutual information does not
solve optimization".

Their Theorem 1 (inference-cost gain bounded by a constant × `sqrt(I(S;Y))` for any bounded cost) is
the best available justification for using MI as the leakage measure at all; the memo cites the
paper only for nonconvexity.

**Citation staleness:** "nonconvex, greedy merge only" is a 2014 state of the art. There is now a
difference-of-convex solver (arXiv:2403.04778) and an EM-relaxed method (arXiv:2405.00616).

## E5. Naming collision — rename the object

de Freitas & Geiger is **not a survey**. It proposes **CPFSI, "Conditional Privacy Funnel with
Side-information"** (published *Machine Learning*, Springer 2025, doi 10.1007/s10994-025-06924-9 —
cite that, not the preprint). The collision is nominal rather than technical: their leakage term is
plain `I(s;z)` in all three objectives, "conditional" refers to conditioning the *reconstruction and
utility* terms on `s`, there is no `I(S;Z|C) ≤ δ` constraint anywhere, no multiple views, no
constraints at all (pure Lagrangians), and they state "we do not claim that our method provides any
strict privacy guarantee for data publishing". The CPF/CFB names originate with Rodríguez-Gálvez et
al., ITW 2021, "A Variational Approach to Privacy and Fairness" — uncited by the memo.

**Action: this study must not call its object a "conditional privacy funnel."** Working name here:
**role-constrained stochastic release (RCSR)**.

The memo's judgement that another variational reconstruction penalty is not the clearest departure
is *supported* by that paper's own findings — "both CPF and CPFSI have little control on the
utility-privacy trade-off, while CFB has the best control", with leakage evaluated only by
trained-adversary accuracy. Honest counter-evidence: CPFSI does Pareto-dominate on their fairness
metric and reconstruction fidelity genuinely helps in the few-labels regime.

## E6. Xu & Strohmer is categorically incompatible, not merely "less direct"

The memo says optimal transport is "less direct under PCRL's ban on sensitive labels at inference".
Stronger: the representation is a **set of group-conditional transport maps** `T_x(·,z)`, `T_y(·,z)`
— the sensitive attribute is **required at inference to select the map**. The setting is also exact
or Wasserstein-relaxed *demographic parity* under `L2` loss only, and the guarantee is independence
of the *prediction* from `Z`, not a bound on leakage of `S` from a representation. Rank it out on
the contract, not on directness.

## E7. Recent work the memo missed, ranked by threat to a novelty claim

1. **Zamani, Oechtering, Skoglund, "Multi-User Privacy Mechanism Design with Non-zero Leakage"
   (arXiv:2211.15525).** `K` users, notation collision with our `C_r`. Distinguishable: their `C_i`
   is the sub-vector a user *wants*, not what they already know, and there is a **single aggregate**
   constraint `I(X;U) ≤ ε`. Closest multi-observer paper; must be cited and distinguished in one
   sentence.
2. **Zamani, Sadeghi, Skoglund, "Privacy-Utility Trade-offs Under Multi-Level Point-Wise Leakage
   Constraints" (arXiv:2601.04815, Jan 2026).** Explicitly claims to be the first with
   heterogeneous simultaneous budgets — but across **output realizations**, not across
   observers/views. **This preempts any novelty resting on per-role `δ_r`.**
3. **Zhong, Assaad, Sreekumar, "Information Bottleneck under Perfect Privacy" (arXiv:2608.11003,
   Aug 2026).** The LP characterization holds only when a **rate constraint is inactive**; in the
   active-rate regime the problem is nonconvex and needs ADMM.
   **Checked against our formulation:** we fix the output alphabet `|Z|` and impose no `I(T;Z) ≤ R`
   term. Fixing `|Z|` only fixes the dimension of `Q`; the feasible set stays a product of
   simplices, which is convex. **Registered constraint on this study: do not add a rate
   constraint.** If one is ever added, the convexity claim must be withdrawn.
4. arXiv:2601.07523, "Sparse Point-wise Privacy Leakage" (2026) — two simultaneous constraints per
   disclosed symbol.
5. Rodríguez-Gálvez et al., ITW 2021 — origin of the CPF name (see E5).
6. "Robustness of Maximal α-Leakage to Side Information" (arXiv:1901.07105) — the
   side-information-conditioned leakage line, adjacent to our `C_r`.

## E8. Convexity — independently re-derived, two routes, agree

The memo's route: `a^r_{scz}(Q) = Σ_t p_r(s,c,t) Q_tz = p(s,c,z)` and
`b^r_{scz}(Q) = p_r(s|c) Σ_t p_r(c,t) Q_tz = p(s|c) p(c,z)` are both **affine** in `Q`;
`I_Q(S_r;Z|C_r) = Σ a log(a/b)` is relative entropy, **jointly convex** in its two arguments;
convex ∘ affine is convex. Checked: the identification of `a` and `b` with `p(s,c,z)` and
`p(s|c)p(c,z)` is correct **given the Markov assumption `Z — T — (S,C)`**, which is the mechanism
definition; and `Σ_{s,c,z} p(s,c,z) log[p(s,c,z)/(p(s|c)p(c,z))]` is `I(S;Z|C)`.

Second route: `I(S;Z|C) = Σ_c p(c) I(S;Z | C=c)`; for each `c` the induced channel
`p(z|s,c) = Σ_t p(t|s,c) Q_tz` is linear in `Q` and the input law `p(s|c)` does not depend on `Q`;
mutual information is convex in the channel for a fixed input law; convex ∘ affine, then a
nonnegative combination. Same conclusion.

The memo's stated caveats are correct and are adopted here: optimizing the codebook, jointly
optimizing decoders, or **maximizing** a utility mutual information each destroy convexity.

The feasibility algebra also checks out. With `q_t = P(Z=1|T=t)`:
`p(s,c,Z=1) − p(s|c)p(c,Z=1) = (Aq)_{s,c}`, so conditional independence of a binary `Z` ⟺ `Aq = 0`
(the `Z=0` case follows because `A1 = 0`); `Bq ≠ 0` ⟺ `Z` adds task information given `C_u`; and
since `A1 = B1 = 0`, any `f ∈ ker(A)` with `Bf ≠ 0` scales into `q = 1/2 + αf ∈ [0,1]`.

---

## F. Net effect on the recommendation

Nothing found here refutes the memo's recommendation. Two things change:

* **The contribution statement must be narrowed** (E1, E2, E5, E7). The defensible claim is the
  deployment contract plus the nested local/coalition role structure solved as a convex program with
  a linear cost — not randomization, not conditional constraints, not multi-view, not heterogeneous
  budgets.
* **Two things get cheaper than the memo assumed**: the single-role feasibility test is an existing
  proposition to cite rather than derive, and the `δ = 0` linear-cost problem already has a
  published LP. The work is the conditional, multi-role, `δ > 0` case and the empirical question.

Registered constraints arising from this stage, carried into REGISTRATION.md:

1. No rate constraint `I(T;Z) ≤ R` (E7.3).
2. The object is named **role-constrained stochastic release**, never a conditional privacy funnel
   (E5).
3. Any in-slate J candidate correction is expected to move measured increments **up**, and is not a
   repair (A).
