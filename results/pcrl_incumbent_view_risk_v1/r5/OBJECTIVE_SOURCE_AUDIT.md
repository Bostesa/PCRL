# Independent inherited-objective premise audit

Status: source and analytic review only. No new fixture algorithm, role-array read or fit has run under this audit. The local primary anchor is `dda8e450cdd398ec1153c69ae1a9dd732a35b955`; remote verification is pending.

The inherited finite-law objective is empirical binary SEX Bayes accuracy. For each public view v (recipient 1 code, recipient 2 code, or code tuple), its fitting-law joint masses are

`m_v(s,z) = (1/n) sum_i 1[SEX_i=s] sum_a q_i(a) 1[wire_code_v(a)=z]`.

It computes `G_v = sum_z max_s m_v(s,z)` and minimizes `G_pair + 0.5*(G_r1+G_r2)`. The source implements this in `_bayes_numpy`, `_torch_metrics`, `exact_metrics` and `bayes_recovery` in `sampled_channel_v1/impl/fit.py` and `channel.py`. This is the empirical Bayes accuracy of the finite fitting law; it is not a fresh reader AUC, a Shannon information quantity, or a population privacy guarantee.

The source aggregates the probabilities of actual emitted global code IDs and tuples. It does **not** evaluate a sensitive reader on the average task probability. When probability, decision and within-unit metadata are fixed deterministic functions of a global code, the code is sufficient to reconstruct the entire declared public view, so this finite-law objective covers that information on fitting rows. This conclusion requires the complete actual wire audit; person-varying fields not determined by code would invalidate the sufficiency premise.

The new complete-view cross-entropy/cross-fit reader design changes the loss, reader fitting and generalization treatment. It must not be described as evidence that the inherited objective used a mean-only service or omitted information that fixed global IDs already determine. Comparing old and new objectives in the shared P0 space can supply an objective-package diagnostic, with stronger formula attribution only if reader schedules are actually matched.

The previous aggregate frontier does not isolate causality: W_TSTAR input routing, output support, decoded bank, tight bank-relative utility, optimizer and empirical Bayes objective varied jointly relative to historical P*. The new reachability and proxy audit can demonstrate individual mechanical exclusions. An observed conflict rules out exact deterministic routing from those old inputs on those rows; an absent conflict does not establish population representability. A missing output state rules out its exact old-wire reproduction but does not establish impossibility of an equally useful alternative.

The old seven-boundary comparison is an arithmetic engineering problem, not grounds to change scientific caps. Independent stable sums and high-precision action costs should establish a prospective numerical safety rule; signed mathematical slack, estimated arithmetic error and development intervals remain separate. The old strict statuses and preserved failures remain unchanged.
