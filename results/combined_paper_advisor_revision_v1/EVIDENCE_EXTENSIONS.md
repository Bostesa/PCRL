# Possible evidence extensions (recorded separately, not in the paper, not authorised)

The revised manuscript keeps the evidence cutoff of the admitted version. The items below would add evidence. Each
would need its own decision, a registered protocol before any fit, and an explicit statement of whether it is
development evidence or confirmation. None is authorised by this revision, and none was run.

1. **Learning curves over the auxiliary budget** for the useful-head comparison and for the linked pair. Recovery at
   several `attacker_fit` sizes would show whether a treatment's advantage persists or shrinks as the attacker receives
   more labelled data. That is the most direct available evidence on whether a difference reflects information removed
   or slower learning. It requires new attacker fits on existing stored releases.
2. **Imperfect linkage.** The pair results assume perfect linkage. A sensitivity analysis with a declared linkage-error
   model would bound how much of the pair increment depends on that assumption.
3. **A second population with the corrected threat model.** The later census studies evaluated a two-recipient output
   service on Florida ACS 2019 development data under an explicitly specified adversary. Adding them would change the
   paper's scope and evidence cutoff. It would also need a reconciliation of release contracts and utility definitions,
   which differ from the stored-model benchmark.
4. **Fresh confirmation.** Every assessment row in the paper was used before. A confirmatory test would need data that no
   earlier analysis of these models has used, and no such data is identified here.
5. **Attacks outside the declared access** (target-input side information, published parameters, repeated or
   chosen-input queries, objective-aware attacks). Each would be a separate threat model, justified on its own terms
   rather than by being stronger.
