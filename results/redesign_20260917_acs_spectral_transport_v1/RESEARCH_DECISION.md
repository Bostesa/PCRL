# Research decision

## Disclosure: what was registered before the outcome

The protocol, the four comparison families, every endpoint, the uncertainty procedure and all decision rules were fixed, hashed and pushed before the 2017 final partition could be read (`PROTOCOL.md`, `COMPARISONS.json`, `TRANSPORT_LOCK.json`, commit `3e31f28`). **No directional prediction was registered**: the study did not record in advance which way the C1-versus-L2 comparison would go. The F1 pass should therefore be read as a prespecified-rule confirmation on a sealed evaluation, not as a called shot.

## Conclusion

The residual spectral channel is **not a competitive mechanism**, and the locked 2017 evaluation did not rescue it. It does two things well: it preserves the published service outputs exactly (structural, verified bitwise on every released view), and it supplies real reusable capability — about 0.025–0.030 nats of residence-prediction improvement over the service alone, in every seed, both weightings and both transfer modes, against 0.017–0.019 for the frozen adversarially trained channel J.

Its protection knob also works as designed, and the new year is what made that visible: coalition conditioning (C1) significantly lowers additional sex and race recovery against both its equal-strength local control (L1) and the equal-total-mass control (L2), under simultaneous 95% intervals, both weightings, and with the residence difference inside the prespecified 0.001 band. Development had the same signs on all 20 endpoints but could not resolve C1 from L2 on a 2,982-person evaluation pool; the 15,924-person locked partition could.

That is a mechanism result, not a method win. At its best setting the channel still lets attackers recover far more sensitive information than J: additional A-race recovery 0.045 against 0.008, additional AB/SEX recovery 0.009 against 0.002, with significant income- and employment-probe costs on top. The same ordering holds under frozen 2018 attacks. The honest description of this study is: **an evaluation finding plus a qualified tradeoff result, with a negative verdict on the tested design.**

## Where the design fails, using evidence already collected

* **Surrogate mismatch, not optimization failure.** The spectral step is a global optimum of its fixed matrix (checked against the eigenvalue sum and against direct least squares for every arm). The penalty it optimizes is a finite set of first moments, linear in Z over a degree-2 basis in the service probabilities. The attacks that defeat it are nonlinear (MLPs and boosted trees are selected in most roles). The mechanism's own fixtures show that zero first moments are compatible with perfect nonlinear recovery, and the frozen-nuisance moment diagnostics rise from training to held-out rows, so the gap is the surrogate, not the solver.
* **Capacity spent on leaky directions.** With r fixed at 16, the top-16 eigenvectors can include directions whose penalty exceeds their utility; SARL's analysis retains only the favorable ones. S0 (no penalty) has both the highest utility and the highest leakage, and the penalty ladder trades them off smoothly, which is what a capacity-limited spectral objective should do.
* **Not a tradeoff artifact of withholding.** Randomized withholding of a simpler channel does not dominate C1 in all seeds and weightings, and neither direction dominates in more than 3 of 180 fixed comparisons.

## One next decision

**Stop developing the fixed-rank residual spectral channel as a candidate mechanism.** The next study that would change a decision is not another spectral grid and not a new neural architecture: it is to test whether the *conditioning target* is what matters, by replacing the finite first-moment penalty with a penalty a nonlinear attacker cannot walk around, while keeping everything else in this frozen interface (same services, same recipients, same audits, same partitions). Concretely: adapt the U-FaTE-style conditional dependence measure (a kernel HSIC-type criterion conditioned on the released service probabilities, not on a discrete label stratum) and select rank by the sign of the eigenvalues rather than fixing r = 16. If that still leaks more than J at equal width, the conclusion generalizes from this design to the closed-form spectral family for this interface, which is a publishable negative result on its own.

Not authorized here and not started: any such study needs its own prospective protocol, and the 2017 final partition is now spent — it cannot serve as the confirmatory evaluation for a new mechanism. The remaining unused candidate year is 2016, whose admission has never been run.
