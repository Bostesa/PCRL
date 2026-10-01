# SUMMARY — methodology_pcrl_acs (2026-10-01)

(Saved by the owner from the role's final report; the role's own write was refused by the harness.)
Lineage status: SaTML '27 registered 2026-09-23, withdrawn by an author 2026-09-26 — these are repairs
for any future reuse. Source inspected and stored results checked; nothing rerun; D17/Q map files and
private archives not verified.

## EARLY FLAGS
1. **Contract says "reported against the fixed service baseline" but the H-relative number is never
   shown.** Every sensitive figure is relative to J. Q's measured recovery beyond H is positive:
   development (2018) 8/8 endpoints positive (0.00086–0.00614); 2016 8/8 positive, 6/8 intervals exclude
   zero (0.0034–0.0060). "Passes all 8 sensitive clauses" means "leaks less than J", not "adds nothing" (F1).
2. **2016 clauses misdescribed in main.tex@55c0c5a35:365** ("capping added recovery"; they cap recovery
   in excess of J's). The Downloads PDF fixes this, but no git ref holds that PDF's source; its title is
   also in no ref (F2, F3).
3. **"No earlier stage had touched" 2016 overstates the record.** The 2016 admission read the whole file
   incl. label columns for schema checks and stored fit/validation class counts. CA 2016, 2017 and 2018
   are all spent; the eligible 2018 group (80,329 people / 53,907 households) is fully used. Texas 2018
   only planned. No New York use in PCRL or durable-guarantees (F4, F16).
4. **The C3/R1 correction cites intervals computed on the probe-selection pool.** The family of 68 tests
   it relies on was later declared indefensible by its own study. Direction holds at ~5.5 SE but these
   are not post-selection bounds (F8).
5. **The fitted 0.01 budget does not hold on held-out people.** Coarse conditional MI on separate 2018
   people is 0.0127–0.0247. A later diagnosis certifies solver optimality to 1.27e-7, so "not certified"
   in the paper is outdated (F9).
6. **Two older defects disclosed but never repaired:** Study 6 audit slate (no predictor reads the
   existing channel without the new one; H-only stands in) (F5); catch-up attackers exist only for some
   systems in the Sept 8–10 redesign studies, reversing coalition ordering in one study (F6).

## Verified OK
Coalition slates in the evidence the paper uses (task-directed and 2016 prospective) include every
single-recipient attacker and an H-only predictor (the complete ancestor, since J is a comparator, not on
the wire), with selection separate from the scored pool. Seeds depend only on anchor and role (identical
releases → identical attackers). Constant extension scores exactly 0. One persistent token per person;
scoring averages losses over the token law, never probabilities. No numerical clipping of increments (the
H-only route still floors at 0 when it wins on validation, F7). Precision-claim history correct
(G0 "for any mechanism" → v6 D10 → annotation N6/C8).
