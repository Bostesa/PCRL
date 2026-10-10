# R2 protected FIT engineering note, 2026-10-10

This note supplements the sealed `OPTIMIZATION_PROXY.md`; it does not change
the bank, objective, FIT caps, old INNER criteria, two starts, or update counts.

P and L retain every optimizer checkpoint and project each toward B0 under
the bank-relative FIT cost and local recovery constraints. Their release laws
remain products of two independent recipient marginals, sampled with separate
recipient HMAC domains. Sequential arms retain every first-stage checkpoint,
project each as `(1-eta_first) B0 + eta_first source`, select on FIT, and freeze
the resulting first law. The second stage trains against the actual cached
first FIT codes and its checkpoints are projected toward exact conditional B0.
Both stages retain the exact identity boundary. A source optimizer that stops
early remains `INCOMPLETE_SEARCH`, with its available checkpoints archived.

D runs a faithful copy of the inherited deterministic optimizer. The copy
adds private retention of initial, replay, accepted bias-pass and terminal or
stopped gate arrays and their exact FIT metrics. A small fixed-law parity test
checks that the selected metrics, start receipts and winner weights equal the
inherited trainer. The copied source preserves the original update and
polish logic. Its public receipt contains hashes and scalar metrics only.

One queue unit reserves up to 600 seconds of CPU and wall time to include
package writing and role materialization. The optimizer still has the sealed
300 CPU-second arm ceiling. Queue accounting charges actual use. B0 has the
same reservation for a matched unit envelope, though it performs no fit.

The new sequential private HMAC root is checked against the pinned public
source manifest before any FIT label read. Protected `P/L/J/S12/S21/D` workers
may open FIT SEX only after the queue gate; `T/B0` never open FIT SEX. The
worker never opens ATTACK_FIT_NEW or INNER_SELECTION labels, only their
admitted score/cell inputs after FIT-only selection is fixed.
