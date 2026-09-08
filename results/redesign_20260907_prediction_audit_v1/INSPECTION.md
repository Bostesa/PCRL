# Starting evidence and bounded verification

Starting branch ablations-facct-2026-07-24, commit
15dbcc3c3338f6707e7a0b3901d738d99651a77d. Existing tracked/untracked work was
inspected before editing; no AGENTS.md exists in the checkout or ancestors.
Read README, reproducibility instructions, PCRL_REDESIGN_STATUS, available
protocols/analyses/tables for all four redesign stages, and the relevant
checkpoint loading, release, preprocessing, fitting, scoring and selection code.
The Gaussian directory has PROTOCOL/TABLE and composition metrics, with its
interpretation in status rather than a separate ANALYSIS.md.

Before new fitting, historical_evidence_hashes.json recorded all2662 files in
the four previous result directories. starting_source_hashes.json recorded234
existing source/document files. Existing changes are not reset, merged or
committed. New experiments use this separate directory.

Confirmed starting observations:

* Separate continuous LEACE preserved the respective Gaussian task signals;
  shared-union erasure removed them. The post-erasure linear PCRL wiring control
  tied separate LEACE. This was not a nonlinear upstream protection result.
* The upstream pilot used pre-erasure differentiable affine ridge R² and
  projected dual weights. It found little LEACE utility loss and substantial
  independently recovered nonlinear leakage. No feasible PCRL advantage.
* The release pilot used actual nonlinear MSE adversaries against erased views.
  Its E arm changed fixed penalty weights to standard projected dual weights.
  Scalar releases came from the saved upstream C_task_only heads, without
  erasure. That exact scalar model, heads and standardizer are reused here.
* The final50 joint-training adversary updates used map375; encoder400 and
  map400 refresh followed, with no subsequent adversary update. Final erasers
  came from the separate calibration split. Last-map changes and fitting
  budgets were therefore different evaluation conditions.
* Frozen diagnostic seed2 E.01, identical P2→U validation target/examples/final
  release: saved training .047024004933238994, independent old audit
  .2524865780071607, continued saved weights .28737027831490314. This runner
  replays the saved training and continued predictions exactly in original
  coordinates. The diagnostic supports recoverability, not moving features
  as the unique cause. No novel protection advantage was established.

The present fitting API accepts only fitting/validation arrays, uses no
representation gradients, and fits scalar-width attackers from scratch.
The full positive control also reuses the compatible saved continued audit;
no full-representation weights are transferred to scalar inputs. Its final
calibrated erasers are loaded, never refitted.

Focused tests exercise full-size exposure accounting with small fitting
fixtures, validation checkpoint selection, target alignment, designated fitting
roles, frozen BatchNorm/dropout state, and refusal to generate a new test before
selection. Runtime integrity checks additionally compare all real scalar
parameter/buffer hashes, full-reference parameters/buffers/erasers, repeated
release outputs, historical split manifests and scalar validation replay.

All fixed-pilot target variances are supported and positive. A read-only review
noted that the small runner's cross-family selector assumes defined R², while
feasibility rejects undefined scores. No undefined score occurs in this pilot;
this runner is not an unqualified generic categorical/constant-target interface.
There was no invalid or superseded experimental run and no core-method repair.
The previously retired universal least-squares-to-classification certificate
remains disabled.
