# Advisor brief: what the released prediction gives away, and what it is worth

2026-10-03. Planned analysis of repeatedly used development data; this is not fresh confirmation. The protocol and
endpoints were locked and pushed before any new fit, and an independent replay recomputed every primary result. The
full account is in RESEARCH_DECISION.md.

## Five answers

1. **Are the decisions useful?**
   - **Adult income:** modestly. The frozen PCRL head beats a constant guess by 3–5 points, but catches only 14–22 % of
     high earners.
   - **HMDA underwriting:** usefulness is not established: about 1.6 points, and 0.6 on one seed. That head catches
     only 7–22 % of denials.
   - **HMDA fair_lending:** the frozen head is a constant: it always predicts the majority class.

   So "decisions leak little" partly means "these decisions say little".
2. **Can part of the score be dropped without changing the probabilities?**
   - Yes. Two-class logits are (margin, offset). The offset changes no probability, no decision and no task loss.
   - Yet an attacker reading the offset *alone* recovers sex or race about as well as from the margin (AUC 0.70 /
     0.74).
   - Removing it lowers measured recovery by **0.07 AUC** in both primary cells, with probabilities exactly unchanged.
     The result passed its pre-registered test and is attributable to the offset on every seed.
   - **Caveat:** a refitted head that is more useful leaks more through the margin it genuinely needs.
3. **Does it replicate across stored purposes?**
   - **Partly.** It holds for the binary income and underwriting heads and for HMDA pricing/race.
   - It does **not** hold for the multiclass employment and education heads: removing the offset changes nothing
     there.
   - It is a property of particular trained heads, not a general rule.
4. **Do two recipients together learn more?**
   - **Yes.** Combining the income and employment recipients' outputs raises race recovery above either alone in every
     contract.
   - That includes hard decisions only: 0.61 against 0.55 / 0.58.
5. **Does FARE help beyond compression on a useful task?**
   - **Not shown.** On a task FARE can keep (Adult occupation group; age protected), it lowers recovery far below LEACE
     (0.56 vs 0.71) at no accuracy cost.
   - But the same tree without its fairness term does just as well (0.57).
   - Its certificate is unavailable or vacuous at our sample sizes.

## Strongest comparisons

**Favourable.** Dropping the logit offset is a free, exact output change for these heads: same probabilities, about
0.07 less measured recovery.

**Adverse:**
- The protective-looking decisions are often near-useless.
- Coalitions recover more.
- FARE's advantage here is matched by plain compression.

## Limits

- Already-used rows.
- Intervals are conditional on the fitted attackers.
- Two primary cells × three encoder seeds.
- Rare HMDA race groups are not estimable.
- The stage-5 task is close to an input recoding.

## Decision for the meeting

Make the **output contract** a primary variable of the empirical paper. Report usefulness beside recovery for every
released output (logits, centred, probabilities, decision), and include the coalition check.

Repair the near-constant HMDA heads before calling their decisions protective.

No new algorithm is indicated. Testing the offset effect on an unopened cohort would need its own prospective
protocol.
