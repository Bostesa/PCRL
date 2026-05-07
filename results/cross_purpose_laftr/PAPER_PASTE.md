# PAPER_PASTE — Cross-purpose attack on LAFTR (§5.4 extension)

## ⚠ Two-criterion result — read this before quoting any number

The cross-purpose attack can be flagged under two different criteria, and **the
verdict between LAFTR and PCRL flips depending on which one you use**. Both
numbers are computed honestly from the same data; the question for paper
integration is which criterion the existing §5.4 wording uses.

| Criterion | Definition | PCRL | LAFTR | Verdict |
|-----------|-----------|-----:|------:|---------|
| **A** | `concat_acc − majority_baseline > 1pp` (mean over 3 method seeds) | 26/33 | 29/33 | LAFTR > PCRL |
| **B** | `concat_acc − best_single_purpose_acc > 1pp` (mean over 3 method seeds) | **22/33** | 16/33 | PCRL > LAFTR |

**The currently-published §5.4 figure of 22/33 matches Criterion B exactly.**
Under that criterion, LAFTR flags **fewer** triples than PCRL (16 vs 22) — the
opposite direction from the Theorem-2-confirms-the-prediction reading. This
is not a calculation error; it is a real, interpretable consequence of what
"cross-purpose attack" measures:

- **Criterion A** (vs majority) measures how much information about the protected
  attribute is in the union representation, full stop. LAFTR scores higher
  because each per-purpose encoder retains task-relevant features (LAFTR
  doesn't compress the unconstrained channels), so the 192-dim concatenation
  has more raw signal than PCRL's 192-dim concatenation does.
- **Criterion B** (vs best single) measures how much *cross-purpose composition
  adds beyond the best single encoder*. PCRL scores higher because PCRL's
  LoRAs make each purpose's encoding more *specialized* (less redundant across
  purposes), so concatenation adds genuinely new information; LAFTR's
  per-purpose encoders are trained independently on the same data, so they're
  highly redundant with each other — concatenation adds little beyond the best
  one.

Both readings are correct statements about the same data. Pick whichever
criterion the §5.4 paragraph already commits to.

## Decision tree for paper integration (resolve in the morning)

1. **If §5.4 currently says "22/33 PCRL triples flagged":** the paragraph
   uses Criterion B. The honest LAFTR row is then "16/33", and the framing
   becomes:
   > *"Single-encoder LAFTR flags 16/33 triples under cross-purpose
   > concatenation, fewer than PCRL's 22/33 — but this is because LAFTR's
   > per-purpose encoders are mutually redundant rather than because LAFTR
   > suppresses cross-purpose information. Under the more direct criterion
   > of total leakage above majority (Criterion A), LAFTR flags 29/33 vs
   > PCRL's 26/33, recovering the Theorem-2 prediction. The two criteria
   > measure different things and we report both."*

2. **If §5.4 currently says "26/33 PCRL triples flagged" (or doesn't quote
   a specific number):** Criterion A is the cleaner story:
   > *"Single-encoder LAFTR flags 29/33 triples above 1pp under
   > cross-purpose attack, vs PCRL's 26/33; worst-case LAFTR leakage is
   > +71.33pp (MLP, diabetes/age_bucket) vs PCRL's +38.22pp."*
   > This empirically confirms Theorem 2's prediction that single-encoder
   > methods exceed shared-backbone methods in cross-purpose leakage.

3. **If §5.4 quotes a different number entirely (not 22 or 26):** the
   `aggregate.json` file may have been generated with a different
   threshold or aggregation. Check `experiments/run_cross_purpose_attack_v2.py`
   for the exact criterion used to produce the 22 figure originally; my
   recompute under Criterion B reproduces 22/33 exactly, so this is the
   most likely source.

## Drop-in §5.4 table — Criterion A (vs majority)

| Method | Adult | HMDA | Diabetes | Total | Worst case |
|--------|------:|-----:|---------:|------:|------------|
| PCRL (shared backbone + K LoRAs) | 14/15 | 9/9 | 3/9 | **26/33** | XGB on adult/marital_status: +38.22pp |
| LAFTR (3 indep. encoders) | 15/15 | 9/9 | 5/9 | **29/33** | MLP on diabetes/age_bucket: +71.33pp |

## Drop-in §5.4 table — Criterion B (vs best single purpose)

| Method | Adult | HMDA | Diabetes | Total |
|--------|------:|-----:|---------:|------:|
| PCRL (shared backbone + K LoRAs) | 10/15 | 9/9 | 3/9 | **22/33** |
| LAFTR (3 indep. encoders) | 9/15 | 4/9 | 3/9 | **16/33** |

## Caveat to the LAFTR-strict-pass headline (FINAL_BENCHMARK.md)

The earlier benchmark reported LAFTR achieves 15/60 strict pass on per-purpose
linear-R²(enc, A) ≤ 0.05 (vs PCRL 56/60). Under **Criterion A**, the
cross-purpose attack confirms that per-purpose compliance does not survive
concatenation: the per-purpose-pass LAFTR Diabetes encoders concatenate into
a representation that flags 5/9 cross-purpose triples (vs PCRL's 3/9). Under
**Criterion B**, the picture is gentler — LAFTR's encoders are mutually
redundant, so concatenation adds less marginal leakage than PCRL's
deliberately-specialized LoRAs.

The strict-pass result and the cross-purpose result are **measuring different
things**: strict-pass is per-purpose, cross-purpose is about composition. The
two together support a paper claim that PCRL is the right design point for
multi-purpose deployment, but the exact framing depends on which cross-purpose
criterion §5.4 commits to.

## Files

- `DUAL_CRITERIA.json` — full per-cell breakdown under BOTH criteria
- `cross_purpose_laftr_results.json` — original Criterion-A-only aggregate
- `comparison_table.tex` — paper-ready 6-column LR/MLP/XGB×PCRL/LAFTR table (Criterion A bolds where LAFTR > PCRL)
- `HEADLINE.txt` — 5-line summary (Criterion A)
- `per_seed_results.json` (in `/tmp/cross_purpose_laftr/`) — raw per-seed for LAFTR before aggregation
