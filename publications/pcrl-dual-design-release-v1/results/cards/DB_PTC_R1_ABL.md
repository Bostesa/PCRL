# Card DB_PTC_R1_ABL: component ablation of DB's strongest release (family DB, revision R1)

Committed before its fit. RULES.md 8 and PROTOCOL 6.3 name the strongest feasible nonzero release of each family as the one that
gets the matched classifier refits and the component ablation, and the task requires both for the strongest feasible comparison
"even if the first interval crosses zero". For family DB that release is now DB_PTC_R1: the rule `ddr.rules.strongest_nonzero`
gives a pair benefit against TWIN of +0.0239 on the registered cells, against DB_PTC's +0.0164 (`claims/STRONGEST.json`). The card
DB_PTC_R1 fitted no ablation ("Ablations are not refitted"). This card adds one. It is part of the ADAPTIVE_R1 reading, descriptive,
and changes no D label.

Code: `ddr/db/ptc_r1.py`, entry `python3 -m ddr.db.ptc_r1 ablate --version V --card <this file>`. Deployment goes through
`ddr.db.predict` (operation `ptc_r1_occ`).

## Operation

DB_PTC_R1_OCC is DB_PTC_R1 with its income component removed:
- The income recipient receives the twin's vector. This is bitwise the twin's, as for DB_PTC_OCC; it was checked on MODEL_SELECT
  features before the fit, with no label read.
- The occupation recipient receives DB_PTC_R1's tilted vector at its fitted alpha_O* (0.65 in both versions), with the same tilt
  weights, propensity model and twin. It is bitwise DB_PTC_R1's occupation output.
- Nothing is selected or refitted. The MODEL_FIT read builds only the nested out-of-fold release, for its spend record.

Reading: DB_PTC_R1 against DB_PTC_R1_OCC attributes the release's benefit to its income component, at fixed occupation output.
DB_PTC_R1_OCC against TWIN gives the occupation component alone. The registered pair DB_PTC vs DB_PTC_R1 already isolates the
occupation strength change, because DB_PTC_R1's income output is bitwise DB_PTC's.

## Evaluation

- MODEL_SELECT utility against the twin and DB_PTC_R1.
- Grid cells inside the robustness bank `DDR1_ROB_DB_R1`, as a candidate: R1-R4 x k0-k3, so the registered and the robustness
  cells come from the same unit.
- `ddr.claims d2r` reports the ablation against the release and against the twin, as for the registered ablations.
- No data-rich readers are trained for it.

## Budget and predictions

Configurations: 1 x 2 = 2 (DB_PTC_R1_OCC), bringing the total to 46 of 48. CPU: under 0.05 h for the fit and evaluation; the
robustness grid is shared with the release's refits.

Coordinator predictions, made before the fit:
- DB_PTC_R1_OCC vs TWIN: +0.010 to +0.020 (point; resolved p 0.4).
- DB_PTC_R1 vs DB_PTC_R1_OCC: +0.005 to +0.015 (resolved p 0.2).
- The income component's share of the release's benefit: 30-60%.
