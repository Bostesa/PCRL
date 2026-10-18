# Advisor brief: correction transfer, genuine refits and a stable method comparison (PCRL, 2026-10-18)

**What we asked.** The occupation-headroom campaign's coalition-aware occupation correction chose its strength from out-of-fold
costs inside the training set, and its realized men's cost missed the prediction. That campaign's NO_QUALIFYING_CONFIGURATION is
unchanged. We asked three questions:
- (A) can the incremental costs be predicted on excluded households;
- (B) does an improved strength rule keep utility while giving a repeatable recovery benefit beyond its strongest matched controls;
- (C) how much do the old recipient screen and unequal reserve requirements shape the reading.

**What we did.**
- **Data.** Florida ACS 2019 public-use development roles only; MODEL_SELECT was already used adaptively by earlier campaigns. No
  new data. The final ATTACK and DEV_REPORT labels stayed closed.
- **Refits.** We refitted the occupation side (witness, routing teacher, correction directions) on four overlapping training
  partitions that exclude GUARD and CHECK households, with H3's income frozen (Track F). The partitions are two diagnostic and
  two repair, the repair partitions with reserved CHECK panels. On the two repair partitions we also refitted the income lineage
  once each (Track R).
- **Rules and readers.** We compared the source rule with two GUARD-based rules frozen before any fit, and trained fresh
  development readers (classifiers estimating the recorded SEX column from released probability vectors) on MODEL_SELECT, with
  matched controls.
- **Checks.** An independent verifier replayed the numbers with its own code. We deployed ten packages from the six raw fields.

**What we found.**
1. **The old estimate missed in two ways.**
   - It over-predicts the NLL increase on MODEL_SELECT by 1.9-4.5e-3 on every partition. On D0, the one partition where nesting
     was run, nesting removed this against the CHECK panel but over-corrected against MODEL_SELECT (+2.9e-3, z 2.0). Non-nested
     cross-fitting is the likely cause, not an established one.
   - Its men's-cost error ranges from +2.1 to -5.1e-3 across partitions (about 0 on D0). That is more than the reserve slack
     (+0.5 to -0.8e-3).
   - An excluded-household (GUARD) estimate is unbiased within noise but as noisy as the slack. Under the registered test it never
     predicts better for the REFIT-routed coalition-aware or occupation-only directions.
   - Two R1 labels are scheme-dependent. An improvement appeared only under a superseded implementation and is not established
     under the registered scheme, which adjudication A5 makes primary. The correction was committed after the panels were read,
     but it was triggered by D-stage evidence only.
2. **Honest refitting removes the slack.**
   - The source witness's margin appears to come largely from training on all 20,000 people. This rests on an adaptive
     diagnostic, and selection is not excluded.
   - Refits on 42-53% leave men's slack of +0.5 to -0.8e-3.
   - In the full deployment every rule picks identity or at most 0.13; in the ensemble SIMPLE picks 0.2-0.6, and each of those
     crosses. Recipe selection gives NO_RECIPE.
   - A refit on 70% restores a positive men's reserve slack (+3.2 / +1.9e-3) and about half of the gap to the source witness.
3. **The correction lowers these development readers' recovery, at a price far outside the budget.**
   - The benefit is repeatable on both stages and both seeds: about +0.03 to +0.04 AUC against the witness and against a
     zero-critic continuation at full strength.
   - It is not stable against an occupation-only critic, which alone gives about +0.03.
   - At full strength the men's cost is 70 times the R0 slack and unbounded on R1. Already at s = 0.3 it is 14 times.
   - The effect survives a whole-service refit (Track R). The income side does not: H3's Brier / NLL headroom disappears when its
     recipe is refitted on about half of MODEL_FIT (training size not separated from recipe).
4. **The comparison design matters.**
   - H3's income view alone fails the old recipient gate, so no H3-anchored release can pass it.
   - Under simulation law L1 at two household repetitions, the old gate passes only about 19-31% of true-zero income differences.
     That is a property of the law.
   - Every required reference fails the reserve the candidates must meet.
   - Under one equal reserve, only the zero-critic continuation qualifies, and it does not lower the readers' recovery.

**Classification.** A replicated, independently replayed diagnosis and a negative method result (development evidence). There is
no competitive candidate and no useful protection component; the highest claimable rung is 0.

**Recommendation.**
- Do not pursue strength rules on held-out GUARD subsets of a small training set.
- If the line continues, consider nested cross-fitting on the full training set. It is untested as a remedy: on D0 it was no more
  accurate than GUARD.
- Size any subgroup-cost check by its SE. A 1e-3 reserve needs roughly 8-16 times the people of one GUARD or CHECK panel.
- Judge candidates and references under the same reserve, with a recipient screen whose zero-difference pass rate is known.
- A fresh population would be needed before any claim beyond development evidence.

Read next: `RESEARCH_DECISION.md` (the eight questions answered), then `START_HERE.md` for the records.
