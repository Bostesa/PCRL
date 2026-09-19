# RESEARCH_DECISION — `pcrl_competitive_method_v1`

All numbers are **2018 development** on repeatedly used pools (and exploratory 2017).
Residence is reserved-from-training capability, not a blind endpoint. Backing:
`DEVELOPMENT_2018.json`, `INTERVALS_P.csv` (decision family, m = 400, candidate-wide
z = 3.84), `INTERVALS_X.csv` (exploratory whole grid, m = 3900, z = 4.36),
`TRACK_N_FITS.json`, `TRACK_E_FITS.json`, `STRESS.json`, `EXPLORATORY_2017.json`,
`VALIDATION.json`, `COUNTS.json`. Increments are over the `H` view, unweighted seed means
unless stated; recovery differences are candidate-minus-comparator (negative = leaks
less); residence differences are loss differences (positive = less residence capability).

## 1. Verdict

**No competitive tradeoff was established.** No prospectively nominated panel release
meets the registered conjunction against J and a strong external comparator under
either weighting, and none shows a coalition-specific benefit over its matched local
controls. This holds with the decision comparator for J taken as either `ref_J` (exposure-
matched, amendment 4) or historical `J`.

Decision levels (PROTOCOL §6):

| level | result |
|---|---|
| Mechanism works | **Yes**, as a finite mechanism: projections remove the declared training moments as specified (full-span zeroes them exactly; ranks as requested; 33/33 serialisation replays exact); Track N selection reconstructs 168/168. |
| Competitive development tradeoff | **Not established.** |
| Coalition-specific benefit (on the panel) | **Not established.** |
| Candidate for confirmation | **Not reached.** Nothing was nominated at the level below, and the stronger attack was uninformative (§6). |

## 2. The panel (validation split only, committed `ad405fe5` before any panel test table)

| family | nominee | what it actually is |
|---|---|---|
| neural | `N_J_g000_C1_b100`, `N_J_g000_C1_b300` | **bitwise untouched J** (checkpoint selection returned step 0 on every anchor); their L1/L2 controls are also bitwise J |
| projection | `E_J_C_k8`, `E_J_C_k6` | coalition projection of J removing 8 / 6 of 16 whitened directions |

All 16 neural and all 10 projection configurations passed the source allowance; the
rule then minimised the worst normalised validation increment, which favours the most
protective configurations.

Family P, candidate-wide, unweighted (person-weighted in `INTERVALS_P.csv`):

| contrast | significantly better | residence difference [adj. interval] |
|---|---|---|
| `E_J_C_k8` vs `ref_J` | AB/SEX −0.0091 [−0.0158, −0.0025] (both weightings) | **+0.0101 [+0.0023, +0.0179] — significantly worse** |
| `E_J_C_k8` vs `leace_A0` | A/RAC1P −0.0108 (unweighted only) | +0.0106 [−0.0012, +0.0224] — not noninferior |
| `E_J_C_k8` vs `splince_A0` | A/RAC1P −0.0092 | +0.0053 [−0.0045, +0.0150] — not noninferior |
| `E_J_C_k8` vs `E_J_L_k8`, `E_J_LX_k8` | nothing | not noninferior |
| `E_J_C_k6` vs `ref_J` | nothing | +0.0049 [−0.0014, +0.0112] |
| `E_J_C_k6` vs `splince_A0` | A/RAC1P −0.0105, AB/RAC1P −0.0089 | +0.0001 [−0.0107, +0.0109] — not noninferior at .001 |
| `E_J_C_k6` vs `E_J_L_k6`, `E_J_LX_k6` | nothing | nothing |
| neural nominees vs anything | identical to J | — |

The lower-disclosure projections buy it with residence capability: `E_J_C_k8` retains a
residence gain of +0.0114 against J's +0.0215. **Lower recovery with lower residence
capability is a tradeoff, not dominance**, and it is no better than what ordinary LEACE
on J gives (`E_J_C_k8` vs `leace_J`: every endpoint within ±0.0011, residence +0.0001).

## 3. Track N — did utility weighting or initialisation matter?

* **Teacher strength mattered for A0-initialised training.** Lowering `gamma` from 1 to
  0 at `C1, beta = 1` cut A/RAC1P recovery +0.0201 → +0.0038 and AB/RAC1P +0.0182 →
  +0.0078 at an unchanged residence gain (~+0.023). Exploratory family X: 22 sensitive
  cells significantly better than `gamma = 1`, 1 worse. This is a gain from changing a
  coefficient in an existing objective, **not a new method**, and it still leaves every
  A0-initialised arm significantly worse than J on at least one endpoint.
* **J initialisation did not improve on J.** At `gamma in {0, .01}` the registered
  equal-budget fresh-probe selection returned untouched J (step 0) in **every** J-arm:
  fine-tuning moved the channel, but no moved checkpoint scored better on the monitor
  fold. At `gamma in {.1, 1}` the A0 teacher pulls J back toward A0 (J starts at
  distortion 0.63 from it), and the selected channels leak more than J. Family X shows
  J-init significantly better than A0-init in 36 sensitive cells — that is J's head
  start, not an improvement over J.
* **Coalition conditioning vs matched local training:** C1 beat the weak control L1 in
  21 of 128 sensitive cells and the strength-matched **L2 in 0 of 128** (X family). The
  predecessor's pattern repeats.
* The `beta = 0` continuations moved the channel only where the teacher term pulled it
  (J-init `gamma` .1/1); otherwise selection returned the start. Utility change alone
  did not produce protection.

## 4. Track E — did coalition projection matter?

* **On J (the strong channel): no coalition-specific effect.** At every rank, C versus
  L and versus LX has no candidate-wide significant difference in the panel family, and
  in the exploratory family only `E_J_C_k1` vs `E_J_LX_k1` on A/SEX (−0.0049). Removing
  any 8 directions from J (C, L or LX) lowers AB/SEX significantly versus J, as do
  ordinary LEACE and SPLINCE on J — always with a residence cost.
* **On A0 (the weak channel): a clean coalition-specific effect on race.** At `k = 2`,
  `E_A0_C_k2` beats **both** `E_A0_L_k2` and the feature-count-matched `E_A0_LX_k2` on
  A/RAC1P (−0.0202, −0.0182) and AB/RAC1P (−0.0253, −0.0217), candidate-wide significant
  under both weightings, with no significant residence difference. This includes a
  **local** race improvement from coalition conditioning. But A0-based releases remain
  far from J (e.g. `E_A0_C_k2` A/RAC1P +0.0223 vs J +0.0000), so the effect is a
  mechanism finding, not a competitive release. It was not prospectively nominated and
  is exploratory.
* **Against generic compression**: C beats rank-matched PCA in 28 and one random
  subspace in 25 sensitive cells (of 80) — directions matter, not just rank. Against
  marginal partial erasure: 5 of 80.
* Full-span projections are constant (the moment Gram is full rank on both channels),
  informationally equivalent to `H`: residence gain −0.0005.
* Descriptively (post hoc, not nominated, no claim): `E_J_C_k4` sits between J and
  LEACE-on-J (AB/SEX +0.0009 vs J +0.0080; residence +0.0185 vs +0.0215); its X-family
  intervals vs J are not significant on any endpoint.

## 5. Local race outcomes

No panel release shows a local race cost versus J (A/RAC1P differences −0.0011 and
−0.0024, not significant). Against `leace_A0` and `splince_A0` the J projections are
significantly **better** on A/RAC1P. On A0, coalition projection *reduces* local race
recovery relative to local-only projection (§4). Neural C1 vs L2: no race advantage.

## 6. Stronger attack — uninformative

The prespecified `MLP[256,256,128]` attack (frozen before any panel outcome) recovered
**less** than the same recipe on the `H`-only view on both SEX endpoints for **every**
interface, including untouched A0 (stress increment −0.092 on A/SEX where the standard
audit finds +0.032), and on the race endpoints it found only about half of A0's
standard-audit increment (+0.026 / +0.024 vs +0.053 / +0.047) and nothing positive on
any other interface. It was weaker than the standard audit everywhere it could be
checked, so it did not stress anything: the recipe overfits or underfits on 4096 rows
with 16 extra inputs. **No survival claim is made from it, and it was not re-tuned after its
outcome was seen.** The standard matched audit remains the only protection measurement.

## 7. 2017 exploratory transport

All 18 transported Track E units (panel + L/LX controls, 3 anchors) fitted and scored with
0 faults; neural nominees are J. Unweighted seed means (`EXPLORATORY_2017.md`):
J +0.0013 / +0.0069 / +0.0039 / +0.0015, residence +0.0189; `E_J_C_k6` −0.0011 / −0.0014 /
−0.0003 / +0.0018, residence +0.0187; `E_J_C_k8` residence +0.0116. Both projections leak
less than J on AB/SEX and A/RAC1P in all three anchors; the k6 residence difference changes
sign across anchors. The matched local controls behave the same, so **2017 shows no
coalition-specific signal**. This descriptive check does not rescue the 2018 verdict: the
panel's registered decision is made on 2018, and 2017 is exploratory with no interval.
`E_J_C_k6` on 2017 is the closest thing to a J-level-utility, lower-disclosure point in the
study; it is a **hypothesis for a new prospective evaluation**, not a result.

## 8. Explanations ruled out and still open

Ruled out (within this study's finite families):

* that previous training never moved (every trajectory moved; SELECTION_DIAGNOSIS);
* that the zero-gain clamp drove selection (0.23% measured);
* that the teacher coefficient 1.0 was harmless for A0-initialised training (lowering it
  helped materially);
* that J-initialised fine-tuning under this objective and selection rule improves on J;
* that coalition conditioning adds anything on the strong channel J at matched rank,
  whether in neural training (vs L2) or projection (vs L, LX).

Still open:

* whether a different selection yardstick (the monitor slate is weak by design) would
  find J-descendants better than J;
* whether the A0 coalition-race effect survives on a channel as strong as J — on J
  there is little residual race signal left to remove (J's race increments are ~0);
* stronger-attack survival: the registered instrument failed to be stronger.

## 9. Integrity

Exact counts in `COUNTS.json`: Track N 168 fitted, 102 distinct audited, 66 exact
duplicates of the untouched start; Track E 204 fitted, 192 distinct audited, 12
duplicates (constant full-span maps); 0 infeasible (SPLINCE on J was feasible). 0
quarantined audits. Amendments 1–4 and incident 1 in `RUN_STATUS.md`; the registered
reference-identity check **failed on 3 cells with an established cause** (amendment 4).
