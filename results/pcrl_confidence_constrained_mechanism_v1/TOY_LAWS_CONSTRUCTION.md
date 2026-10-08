# Toy laws — construction notes (role F)

`TOY_LAWS.json` holds the four finite laws of PROTOCOL §6 plus JOINT_HEADROOM, added by amendment A1 at role A's
request (see the end of this file). It was written by role F before any candidate code (`ccm/guard.py`,
`ccm/geometry.py`, `ccm/oracle.py`) evaluated a law.

- Current sha256: `e6cee30524eb50192d3c4c7f1363f83198024451c2d913f408b9a845f1ad9746`.
- It is recorded in `DATA_ACCESS_MANIFEST.json` (`toy_laws.sha256`, with the history) and is bound by
  FEASIBILITY_LOCK.
- The laws are never changed after a candidate result.

## What the laws are, and what they are not

- **Capability, not prevalence.** These are purpose-built fixtures. Each one shows that the oracles can detect one
  situation: no headroom, removable sensitive distinction, coalition clue, no coalition clue, coordination headroom. They say nothing about how
  often any of these situations occurs in the Adult references, and no real-data claim may cite them.
- They test the oracle machinery (admissibility, partition enumeration, MI / Bayes accuracy, coalition measures). They
  are not evidence for or against the mechanism.

## Format

- Each law lists inputs `x` with `P(x)`, `P(SEX=1 | x)` and two reference vectors `p1(x)` and `p2(x)`.
  - All values are exact rationals (strings), with their float64 values alongside (`*_float = float(rational)`).
  - Every vector is strictly positive and has a strict argmax.
- **Recipients.**
  - Recipient 1 stands for income, K = 2.
  - Recipient 2 stands for occupation, with **K = 3 in every law**. The real occupation recipient has K = 6. K = 3 is
    the smallest K that exercises multi-coordinate NLL and Brier trade-offs.
- **Partition domain.** A release partitions the distinct reference values of a recipient, so inputs with equal vectors
  always share a token. NO_COALITION uses this: `p1` depends only on `a` and `p2` only on `b`.
- **Capacity.** No law has more than 6 inputs, so the i8o64 capacity (8 / 64 tokens per predicted class) never binds.
  Exhaustive enumeration is cheap: at most 10 admissible partitions per recipient and 100 admissible pairs.
- **Contract.** G(d = 1/200, b = 1/400), exactly as in PROTOCOL §3 and UTILITY_CONTRACT.md:
  - NLL: `q_k ≥ e^{-d} p_k`;
  - Brier: `Σ q² − Σ p² − 2(q_y − p_y) ≤ b` for every label y;
  - Class: strict argmax equal to the first-index argmax of p.

## Role F's construction check (`ccm.data.toy_law_check`)

The check is independent of every candidate module (it imports none). Reproduce it with
`PYTHONPATH=. python -m ccm.data toy-laws-check`. The stored `construction_check` block must be reproduced exactly; at
writing it is, and all four intended properties hold.

**Subset classification.** For every recipient, every subset of the distinct reference values gets exactly one status:
- **CERTIFIED.** An explicit q, stored in the JSON, is checked against every member and every label twice:
  - exactly in rationals, with a rigorous rational bracket of e^{−d} (Taylor partial sum ± twice the Lagrange
    remainder; the upper end of the bracket is used, so the check is sufficient);
  - again in float64 with no tolerance.
- **INFEASIBLE_CLASS.** The members disagree on the decision.
- **INFEASIBLE_NLL.** `Σ_k max p_k` exceeds the rational upper bound of e^d. This is the closed-form certificate of
  PROTOCOL §3.

A subset where the necessary condition holds but no certified q is found is UNDECIDED. Any UNDECIDED subset makes the
law invalid. **There are none.**

**Certificate margins.** Every admissible bin of every law has a certificate with slack on both conditions:
- NLL slack ≥ 2.49e-4;
- Brier slack ≥ 4.18e-4 (the tightest are the K = 3 chain pairs of POSITIVE).

**Margins on the necessary condition.** Mergeable pairs have `Σ max ≤ 1.003`, against e^d = 1.0050125. Every
certified-infeasible set has `Σ max ≥ 1.006` or a decision conflict. No law sits near the boundary, so a correct
candidate guard cannot disagree with this classification because of rounding.

**Measures.**
- Plug-in MI (nats) and exact Bayes accuracy of SEX are computed from t1, t2 and (t1, t2) for these arms: identity,
  decision-only, CLASS (eligibility), task-only (fewest tokens), local (own-MI minimum), sequential 1→2 and 2→1
  (non-adaptive), and joint (coalition-MI minimum).
- MI ties within 1e-12 are reported as ranges over every tied partition, so no number depends on a tie-break order.
- The argmin partitions are listed.
- The stochastic-LP arm is NOT computed by role F.

## The four laws

| Law | Inputs | I(S;X) | Identity MI (r1 / r2 / pair) | Local MI (r1 / r2 / pair) | Joint pair MI |
|---|---|---|---|---|---|
| NULL | 6 | 0.1450 | 0.1450 / 0.1450 / 0.1450 | 0.1450 / 0.1450 / 0.1450 | 0.1450 |
| POSITIVE | 6 | 0.1285 | 0.1285 / 0.1285 / 0.1285 | 0.0642 / 0.0642 / 0.0642 | 0.0642 |
| COMPLEMENTARY | 4 | 0.3681 | 0.3681 / 0.3681 / 0.3681 | 0 / 0 / 0.3681 | 0.3681 |
| NO_COALITION | 6 | 0.1330 | 0.1330 / 0 / 0.1330 | 0.0045 / 0 / 0.0045 | 0.0045 |
| JOINT_HEADROOM | 4 | 0.3230 | 0.3230 / 0.1853 / 0.3230 | 0.1390 / 0.1059 / 0.3230 | 0.1853 |

### NULL: privacy cannot usefully improve under G

**Construction.**
- x0/x1 (SEX law 4/5) and x2/x3 (SEX law 1/5) are mergeable pairs in both recipients, 0.001–0.002 apart.
- Every pair with different SEX laws either:
  - is at least 0.018 apart in total variation (`Σ max ≥ 1.018 > e^d`); or
  - has a different decision. x4 and x5 are class 1 for recipient 1 and class 2 for recipient 2, and they are 0.02
    apart from each other.

**Property.** Every admissible bin is SEX-homogeneous. So every admissible partition, in each recipient and for the
pair, has MI equal to the identity value, 0.1450 nats. The Bayes accuracy is 3/4 in every arm.

The same holds for stochastic channels. A G-feasible output q can be emitted only by inputs that q satisfies jointly,
and that set is an admissible bin, hence SEX-homogeneous. The posterior given any output is therefore the input's own
SEX law, and no channel lowers H(S | output) below H(S | X).

Admissible merges exist (task-only uses 4 tokens instead of 6). The law tests "merging is possible but useless", not
"nothing can be merged".

### POSITIVE: admissible coarsening removes an unnecessary sensitive distinction

**Construction.** In both recipients x0..x3 form a chain:
- consecutive vectors are 0.003 apart (`Σ max = 1.003`, certified mergeable);
- vectors two steps apart are 0.006 apart (`Σ max = 1.006`, certified infeasible).

The chain's admissible bins are therefore exactly the three adjacent pairs. SEX laws are 1/5, 1/5, 4/5, 4/5, so the
middle pair {x1, x2} is the sensitive but mergeable distinction. x4/x5 (SEX law 1/2, decision class 1) are a
SEX-neutral mergeable pair.

**Property.**
- The privacy-aware local partition {x0}, {x1, x2}, {x3} halves the MI in each recipient and for the pair:
  - 0.1285 → 0.0642 nats;
  - Bayes accuracy 7/10 → 3/5.
- The task-only (fewest-token) partition is UNIQUE: {x0, x1}, {x2, x3}, {x4, x5}. It is SEX-homogeneous on the chain and
  keeps the identity MI, so the comparison does not depend on any tie order.
- Joint equals local here (0.0642). CLASS is ineligible (each class is too wide).

### COMPLEMENTARY: combining the releases creates a coalition clue (XOR)

**Construction.**
- Inputs carry two bits: a = 0 for x0, x1 and b = 0 for x0, x2. The SEX law is 9/10 if a = b and 1/10 otherwise.
- Recipient 1 must disclose a: the two a-groups are at least 0.048 apart (certified infeasible), in one decision class,
  so CLASS is ineligible.
- Recipient 2 must disclose b: the two b-groups have different decisions, so CLASS is eligible.
- Inside each group the two inputs are mergeable, 0.001–0.002 apart.

**Property.**
- Each identity single release reveals SEX (0.3681 nats, Bayes 9/10).
- The unique local partition of each recipient merges inside the groups, and each single release becomes exactly
  uninformative (MI 0, Bayes 1/2).
- The pair (t1, t2) still determines (a, b), so the coalition MI stays 0.3681. The coalition gain over the better
  single local release is +0.3681.
- The clue is FORCED: for every admissible pair of partitions the coalition MI is 0.3681. Joint, both sequential arms
  and local all equal 0.3681.

### NO_COALITION: no extra coalition clue

**Construction.**
- Inputs are (a, b) with a ∈ {0, 1, 2} and b ∈ {0, 1}, uniform, so a and b are independent.
- SEX depends on a only (4/5, 1/5, 3/5).
- `p1` depends on a only: a = 0 and a = 1 are mergeable (0.002 apart); a = 2 is class 1.
- `p2` depends on b only: b = 0 is class 0 and b = 1 is class 2.

**Property.** For EVERY admissible pair of partitions, I(S; t1, t2) = I(S; t1) = the better single release, with
coalition gain exactly 0. This holds because t2 is a function of b, and b is independent of (a, SEX).
- Recipient 1 still has headroom: 0.1330 → 0.0045 nats; Bayes 11/15 → 8/15 = the base rate.
- Joint equals local.

### JOINT_HEADROOM: coordination headroom (amendment A1)

**Intended property.** The best admissible JOINT pair of partitions has strictly lower coalition MI I(S; t1, t2)
than all three of:
- the local design (each recipient minimises its own MI);
- sequential 1→2 (recipient 1 takes its own-MI minimum; recipient 2 then minimises I(S; t1, t2) given recipient 1's
  PARTITION, so the design is non-adaptive);
- sequential 2→1 (symmetric).

The claim holds for every tie choice.

**Construction.** Four inputs, each with P = 1/4. SEX laws: x0 1/10, x1 1/10, x2 4/5, x3 9/10.
- Recipient 1 (K = 2):
  - chain x2 – x1 – x3 at (0.900, 0.100), (0.903, 0.097), (0.906, 0.094);
  - consecutive pairs are mergeable (`Σ max = 1.003`); two steps apart is certified infeasible (1.006);
  - x0 at (0.950, 0.050) is certified infeasible with every chain value (`Σ max ≥ 1.044`).
- Recipient 2 (K = 3):
  - chain {x1} – {x0, x2} – {x3} at (0.800, 0.150, 0.050), (0.803, 0.147, 0.050), (0.806, 0.144, 0.050);
  - x0 and x2 share one reference vector;
  - consecutive values are mergeable; {x1} with {x3} is certified infeasible.
- **Capacity.** The registered i8o64 bound (8 tokens per predicted class for recipient 1, 64 for recipient 2). It never
  binds: recipient 1 has 4 values in one class and recipient 2 has 3.

**Why it works.**
- The only coalition-reducing move hides the x1/x2 distinction (SEX law 1/10 vs 4/5) from BOTH recipients:
  recipient 1 merges {x1, x2}, and recipient 2 merges {x1} with {x0, x2}.
- Each recipient's own optimum is a different merge, and each one is useless to the coalition:
  - recipient 1 prefers {x1, x3} (own MI 0.1390, versus 0.1853 for {x1, x2}), but recipient 2 can never merge x1
    with x3;
  - recipient 2 prefers {x0, x2} with {x3} (0.1059, versus 0.1332), but recipient 1 can never merge x2 with x3, and
    keeps x0 apart.
- Each first mover's own optimum keeps x1 and x2 apart, so the second mover cannot reduce the coalition MI at all.

**Verified values** (role F's independent exhaustive check; 9 admissible pairs):

| Arm | Coalition MI (nats) | Bayes accuracy |
|---|---|---|
| identity | 0.3230 | 7/8 |
| local | 0.3230 | 7/8 |
| sequential 1→2 | 0.3230 | 7/8 |
| sequential 2→1 | 0.3230 | 7/8 |
| joint (unique argmin) | 0.1853 | 29/40 |

The joint argmin is recipient 1 {x0}, {x1, x2}, {x3} with recipient 2 {x0, x1, x2}, {x3}. Joint is 0.1377 nats below
local and both sequential designs.

**Tie handling.** Own-MI argmins are compared within 1e-12, every tied choice is evaluated, and the claim must hold
against the most favourable choice for the competitor. Here every own-MI argmin and the joint argmin are unique, so no
tie-break is involved. The second mover's coalition-MI minimum is a single value whatever its tie-break. The task-only
fewest-token partition is tied (two partitions per recipient); it is reported as a range (pair 0.1853–0.3230) and is
not part of the claim.

**Purpose-built.** JOINT_HEADROOM shows that the oracles can detect coordination headroom. It says nothing about how
often such structure occurs in the Adult references.

## Known limits of this fixture set (stated before any candidate result)

1. **Coordination headroom is shown by one purpose-built 4-input law (JOINT_HEADROOM, amendment A1).**
   - In COMPLEMENTARY the coalition clue is forced, so joint equals sequential equals local there.
   - No law separates the two sequential orders from each other.
2. **Real dimensions are not used.** Recipient 2 has K = 3, not 6. The capacity bound never binds. No input sits near a
   decision tie or near the e^d boundary.
3. **The stochastic-LP arm has no role F reference value.** The only stochastic statement here is NULL's argument
   above.

## Amendment A1 (role F, at role A's request; before any candidate code evaluated any law)

- **Change.** Added the law JOINT_HEADROOM.
- **Reason.** The four original laws contain no fixture where the joint arm strictly beats the local and both
  sequential arms: COMPLEMENTARY's coalition clue is forced. The coalition-aware arms were therefore tested only as
  equalities.
- **Previous `TOY_LAWS.json` sha256:** `bdb8b0fe48ab2c19201c6545b9453f5c1ac3d35e0798c815d2fb1463c6b03c12`.
- **New `TOY_LAWS.json` sha256:** `e6cee30524eb50192d3c4c7f1363f83198024451c2d913f408b9a845f1ad9746`.
- **Unchanged content.** NULL, POSITIVE, COMPLEMENTARY and NO_COALITION are content-identical to the previous file, as
  are their construction-check blocks. The canonical sha256 of each is recorded in `TOY_LAWS.json` under
  `amendments[0]` and checked by `tests/…/test_data.py`.
- **Other file changes.** The file gains the `amendments` block and the JOINT_HEADROOM law and check. Top-level fields
  are otherwise unchanged.
- **Checker change.** `ccm.data.toy_law_check` gained an opt-in witness report (`report: ["joint_argmin_partitions"]`
  in a law) and the JOINT_HEADROOM predicates. The output for the four original laws is unchanged.
