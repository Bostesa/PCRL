# Independent MIP accuracy review

The first seed-1 deterministic control was correctly refused as `INCOMPLETE_CONTROL`. Its solver reported optimality and zero gap, but its reported raw integer error (4.964079919573827e-7) and primal violation (4.964079762974328e-7) exceed the frozen executable acceptance tolerance of 1e-8. This is neither a finite-family infeasibility certificate nor a valid executable incumbent.

Static review found that the MIP call explicitly set one thread, time limit and relative gap, but omitted integrality and primal feasibility tolerances. The [official HiGHS options documentation](https://ergo-code.github.io/HiGHS/dev/options/definitions/) lists the integrality option `mip_feasibility_tolerance` with default 1e-6, and `primal_feasibility_tolerance` with default 1e-7. The refused error lies below that documented integrality default and above the campaign checker. This supports an optimization accuracy mismatch; it does not prove that a corrected call will succeed.

A single prospectively documented correction setting both solver tolerances at or below 1e-8 is an engineering accuracy repair. It must keep the checker, scientific caps, support, objective, seed and time ceiling unchanged. Preserve all original artifacts and charge the extra relaxation and MIP calls. Independent replay of the refused incumbent is queued; corrected execution and its independent replay remain required before any new feasibility claim.

The seed-0 integer incumbent is independently feasible. Its independently certified continuous relaxation gives a remaining global objective gap upper bound of 0.005113522935801029. This bound does not prove integer optimality. Product and sequential controls receive no global optimality certificate.
