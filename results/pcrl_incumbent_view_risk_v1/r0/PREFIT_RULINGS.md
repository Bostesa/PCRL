# Prospective implementation rulings

Before policy fitting and before observing candidate results:

- Central O-J/V-J/V-P/V-D share the inherited 12→16→16 gate, 480 parameters. Product probabilities are factorized marginal gates with independent recipient coins. Hard trained deterministic checkpoints are separate.
- Sequential uses two 12→12→4 factors, 208 parameters each (416 total including frozen first). Local uses own-only 148+212=360. This gives the shared 480 parameter ceiling while disclosing architecture and input-scope differences. No trainable conditional first-code embedding is added.
- Eta .05 is the exploratory initialization, not a maximum change to the incumbent. Final feasibility projection starts at 1 and geometrically halves, with executable eta=0 archived separately. It stays in each declared family.
- Mac longdouble may not exceed float64. Independent Decimal.from_float and Decimal.ln will assess delicate numerical margins before scientific fits.
- The inherited empirical Bayes-accuracy objective already conditions on actual global public code tuples; new cross-fitted CE readers test a loss and reader generalization package. No prior mean-output omission is presumed.
