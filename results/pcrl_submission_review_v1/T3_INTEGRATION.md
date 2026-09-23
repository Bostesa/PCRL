# Integration of the claims-foundation audit (Terminal 3)

**Audit source:** branch `research/pcrl-claims-foundation-v1` @
`33124f965861ccfcaa710aff4a56880c5ab3957e`, entry point `results/pcrl_claims_foundation_v1/REVIEW_INDEX.md`.
**Manuscript target:** this branch at `30a6fd19e17453bc8c421a65491b5b482ab9291f` — the exact commit the
patch was generated against, with `main.tex` SHA-256 `c603fad1…` and `references.bib` `d150c9b8…` matching
the patch header. There was no subsequent manuscript work to overwrite.

`git apply --check` passed; the patch applied cleanly; the paper rebuilds with 0 errors, 0 undefined
references and 0 undefined citations.

## What I verified myself before accepting it

I did not take the audit's word for the blocking items.

| claim | how I checked it | verdict |
|---|---|---|
| The code is residence-supervised, not label-free | read `encoding.py::fit_encoder` @ `f4bdf4cd5`: it fits two teachers on `rf['labels']['same_residence']` | **confirmed** — "label-free" was wrong |
| The selected release constrains local disclosure only | the nominee is `T0_L_…` (policy `L`); the audit's replay records AB fitted CMIs of 0.0119–0.0177 against a 0.01 budget, because `L` imposes no AB constraint | **confirmed** |
| The conditioning partition is 2 cells for A, 4 for AB | read `encoding.py::ServicePartitions.fit`: `n_local=2` k-means cells on `H_A[:, [1,3]]`, plus a median split of `H_B[:,1]` | **confirmed** |
| `results/rebuttal/**` is this paper's work, not the AAAI paper's | read all four headline files: per-dim standard deviation, the LoRA erasure floor, the published `[128,128]→64` backbone, the Adult/HMDA/Diabetes grid | **confirmed — my earlier attribution was wrong** |
| `origin/main` still ships the retired guarantee | `git show origin/main:pcrl/purposes/verification.py` still defines `certified_accuracy_bound` with its "Theorem: Linear Compliance Guarantee" docstring and proof, and it does not raise | **confirmed** |
| Reviews of the earlier submission exist | `results/rebuttal/ablations_facct/PREDICTIONS.md` @ `ecaeba46` registers ablations against "reviewers AC, AkJK Q1, NY7k Q2" and "AkJK Q2" — venue reviewer handles | **confirmed: reviews exist; their text does not** |

## Completed corrections

1. **"Label-free" withdrawn.** The construction paragraph now describes two residence-supervised teachers,
   a residual logit and its 32 quantile cells, noting that no protected label enters at runtime. The
   baselines paragraph says "the same residence-supervised code".
2. **Local versus coalition scope made explicit.** The paper now distinguishes the framework's available
   policies from the selected candidate's actual policy: the selected utility release is local, so its
   coalition view is **audited but not constrained**. Solver-reported optimality is labelled as such, with
   no duality gap recorded.
3. **Coarse conditioning stated with its limits** — two cells of $H_A$ for $A$, four of $(H_A,H_B)$ for
   $AB$ — and still not a population guarantee on continuous $H$.
4. **The increment no longer claimed as a bound.** Absolute recovery is, up to sampling error, a lower
   bound on the view's mutual information; the increment is a difference of two such lower bounds and
   bounds $I(S;Z\mid H)$ in neither direction.
5. **Separate implementations.** The lineage appendix now states that no combined encoder, adapter and
   stochastic-release pipeline has been built or evaluated, and that the two lines are reported as
   separate implementations.
6. **Rebuttal attribution corrected** in `REVIEW_TO_EVIDENCE.md`, with the two rows that wrongly cited
   "Appendix A" removed — `main.tex` contains neither of those runs, and they belong to the other paper.
7. **Original contributions preserved with their real scope.** The audit caution survives; the *identity*
   is not original (it is the variance-weighted multi-output $R^2$) and the dominant axis is a lower bound
   on the best linear direction. Checkpoint rules are now distinguished: final iterate 56/60 strict with
   7/60 clean, best-checkpoint 54/60, later erase-layer arm 60/60 strict but 0/60 clean. The registered
   adapter and erasure-layer ablations were never run, so no necessity is claimed for either.
8. **Prior art corrected.** Convexity is Calmon & Fawaz 2012 and Salamatian et al. 2015; Rassouli–Gündüz
   is cited for the nullspace feasibility criterion, with their LP noted as the mutual-information-utility
   case; Erdogdu & Fawaz 2015 for releases beside fixed earlier releases; Taylor et al. 2026 for per-party
   and collusion constraints; Calmon et al. 2017 and Creager et al. 2019 added. The novelty claim narrows
   to the setting plus the measurement contract, with no algorithmic novelty asserted.
9. **Approximate-composition lemma included.** $R^2(h_1,\dots,h_k)\le\sum_p R^2(h_p)/\lambda_{\min}(R)$
   for the block-whitened cross-view correlation, shown tight on the $\delta$ example, together with the
   fact that exact-zero-covariance views can still determine the attribute nonlinearly. It materially
   strengthens the composition paragraph and is attributed to the audit's repair of the original P6.

## The public-`main` repair, kept separate

Branch **`fix/retire-accuracy-guarantee`** @ `5d4eda04639aae10733e4d72c2ceaae0e849ede5`, pushed for review.
**Not merged. `origin/main` is unchanged at `55e4cb1d1`, and the public API is not fixed until it is.**

It applies the audit's patch 0001 and adds one thing the patch omitted: `tests/test_accuracy_bound.py`
(324 lines, 14 tests, all reaching the retired API through a shared helper) still asserted the refuted
bound, so `main`'s suite would have failed. That file is removed and superseded by
`tests/test_accuracy_bound_retired.py`.

Verification on the branch: the new retirement tests pass (4/4); `certified_accuracy_bound` raises
`NotImplementedError`; the 22 certificate/verification/compliance tests pass. The full suite is
**208 passed, 7 failed**, and **all 7 failures pre-exist on clean `origin/main`** — six in
`tests/test_folktables_fixes.py` (one torch in-place/autograd incompatibility) and one in
`tests/test_lora.py::test_n_trainable_matches_lora_count` (`assert 2208 == 1968`). I ran both on an
unmodified `origin/main` checkout to confirm. **The repair introduces no new failure**, and it does not
fix those seven, which are someone else's to triage.

The paper's wording was changed to match reality in the meantime: the API "refuses to run **in the
evaluated code**".

## Not adopted into the registered text

The audit's two optional abstract edits are recorded in `REGISTRATION_WORDING.md` and **not applied**,
because the abstract is registration text and the venue bars substantial post-registration changes. The
body now carries the local/coalition scope explicitly, so the abstract is not misleading as it stands.
