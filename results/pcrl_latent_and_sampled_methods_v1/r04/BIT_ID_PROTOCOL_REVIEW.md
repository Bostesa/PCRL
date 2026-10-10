# D code identity after the effective seal

The executed D protocol uses `c0` as the least significant bit and derives
the categorical identity as `sum(c_j * 2**j for j in 0..9)`. This is explicit
in the effective `CAMPAIGN_LOCK.json` D wire, `COMPLETE_VIEW_SPEC.md`, R07
service metadata, and R09 complete-view encoding. The R04 first paired-seed
replay independently reconstructed the same categorical identity and all
three saved attack matrices, then reproduced every frozen reader prediction
and AUC exactly. The earlier discussion of most-significant-bit numbering was
superseded before the effective seal. No executed post-lock deviation exists.

Changing only the integer naming to most-significant-bit numbering is a
bijection over the 1,024 ten-bit strings. The bits, code-head probabilities,
decisions, task losses, and information in a complete view are unchanged.
Categorical one-hot columns would need the corresponding permutation. An
ideal reader's attainable recovery is invariant under that relabeling;
bitwise equality of newly refitted finite LR/HGB readers under a different
feature-column ordering is not asserted without a new numerical replay. The
actual frozen LSB readers were replayed exactly.
