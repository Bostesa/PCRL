# PAPER_PASTE — Phase D additions

Two text additions, paste-ready.

## D.1 — "What would change our conclusions" paragraph (new §6 supplement block)

Paste at the END of the limitations paragraph (after the consolidated §6
limitations text, not replacing it).

```latex
\paragraph{What would change our conclusions.} Three findings would
materially weaken the claims in this paper. (i) A demonstration that a
nonlinear adversary can recover disallowed attributes from individual
per-purpose representations $h_p$ at substantially higher rates than
Section~\ref{sec:cross-purpose} reports for the linear-and-XGBoost
auditors we deploy; the per-purpose linear leakage bound bounds linear
adversaries by construction (Proposition~\ref{prop:linear-bound}) and
the empirical audit covers a finite class of nonlinear ones, but a
stronger nonlinear attacker could expose residual signal. (ii) A
deployment regime where the 7--18$\times$ training-cost overhead of
joint optimisation cannot be amortised by maintenance or update cost
savings; we frame this work as feasibility demonstration
(\S\ref{sec:compute_cost}) rather than as a deployment-ready
alternative, and the practical regime where joint training pays off
remains an open empirical question. (iii) Convergence of a held-out
cross-validation of the dual-update schedule on a substantially
different cleanly-compliant rate than $7/60$; the schedule was
developed by audit on the evaluation grid (\S\ref{sec:limitations}),
and a held-out fold could move the headline.
```

Verify the three labels exist in the source:
- `\ref{sec:cross-purpose}` — referenced from drop-in #2 and elsewhere
- `\ref{prop:linear-bound}` — must exist for the per-purpose bound proposition
- `\ref{sec:compute_cost}` — referenced from §C drop-in (Critique 14)
- `\ref{sec:limitations}` — referenced from drop-ins #6 and #11

If any label is named differently in the .tex source (e.g.,
`prop:per-purpose-linear` instead of `prop:linear-bound`), rewire on
paste; the prose itself is fixed.

## D.2 — Abstract CI hook (already applied to results/reviewer_dropins/PAPER_PASTE.md)

The new abstract opens with a contextual-integrity hook before the
shared-encoder framing. The patched text inserts one sentence after
the section-heading-style opening:

```
**Per-purpose linear leakage bounds for shared-encoder representation
learning.** Our framework operationalises contextual integrity
\citep{nissenbaum2004privacy} at the representation level: each declared
purpose receives its own representation with a quantitative bound on
what disallowed attributes it leaks. A single shared encoder produces
multiple purpose-conditional representations, each carrying a
per-purpose linear leakage bound $R^2(h_p, A) \leq \tau$ on declared
disallowed attributes. [...]
```

`\citep{nissenbaum2004privacy}` is a new bib key; entry is in
`results/reviewer_dropins/CITATIONS_TO_ADD.bib`.

## Word-count check on the patched abstract

Original Terminal-1 abstract was ~150 words. The CI hook adds ~30 words
(2 sentences). The patched abstract is ~180 words; comfortably within
the 200-word NeurIPS abstract limit, with margin for further edits.
