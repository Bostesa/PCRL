# Amendment to the claims-foundation audit (33124f965861ccfcaa710aff4a56880c5ab3957e), 2026-09-23

The claims audit's `THEORY_REPAIRS.md` §T3 and `CORRECTIONS.md` item T3-12 said that the Tikhonov form of
the approximate-composition lemma "needs an unregularised per-purpose premise", and that a per-purpose
ridge audit ≤ ε does not imply it. **That caution was wrong (too strong).**

With the same ridge τ in every score and R computed from the unregularised covariances with Σ_p ≻ 0:

  R²_τ(concat) ≤ Σ_p R²_τ(h_p) / λ_min(R).

The reason is that λ_min(R) ≤ 1, so R + τD⁻² ⪰ λ_min(R)(I + τD⁻²). The proof is in COMPOSITION_LEMMA.tex,
part (iii), and 20,000 random instances produced 0 violations.

Kind: an interpretation correction, making the statement *stronger*. No number changes. The manuscript
at 8df7527c does not contain the over-cautious sentence, so the paper needs no change. The historical
files are left unedited; this amendment supersedes that sentence.
