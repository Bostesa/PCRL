# Methodology findings — durable-guarantees / "Outputs Leak What They Use"

(Saved by the owner from the role's final report; the role's own write was refused by the harness.)

Pins: code `Bostesa/durable-guarantees@956f5c883f515646aa457db55ecbd74b913768b2` (`dg:path:line`);
manuscript `paper.tex`/`appendix.tex` from the Aug 30 zip (identical to Sep 17); main text = Jul 29
submission build, appendix = Aug 2 revision; PCRL not pinned (F-21).
Tags: [SI] source inspected, [SR] stored results checked/recounted, [NR] not rerun, [UV] unverified.
Severity for a *revised benchmark*: High = headline or comparability can change; Medium = scope/wording
or one claim; Low = hygiene.

## A. Threat model and access

**F-01 Tier 2 described as repeated-query/insider, implemented as white-box population access with one release. High.**
- README: Tier 2 "has seen the representation before noise was added ... vendor, auditor with pipeline
  access, or anyone who can query the same row repeatedly" (`dg:README.md:62-66`; `battery.py:15-21`;
  `two_tier_certification.py:9-10`); paper: "knows the clean representation and the noise we added"
  (`paper.tex:251-254`).
- Code: per-class Gaussians fit on **pre-noise h of the 75% attacker-train rows** with s labels, plus exact
  noise covariance (σ²I or σ²QQᵀ); scores **one noised release** per held-out row (`battery.py:60-103`).
  Never the target's clean vector, never averaging. Since P|s = (h|s) ⊛ N(0,Σ), the asymptotic side
  information is a correct noise model, not extra target information. For deterministic baselines the LRT
  is QDA on the release (`baseline_gauntlet.py:52-56,329-333`). The paper says parameters are not exposed
  (`paper.tex:203-205`), so Tier 2 silently exceeds the stated exposure model.
- Repair: access-named tiers — T1 (labelled releases, 1/target); T1-Q(N) (N fresh releases); T2-MA
  (mechanism + public parameters, labelled releases); T2-WB (white-box/clean vectors on a population
  sample); Insider (the target's clean vector). Re-tag stored LRT verdicts as T2-WB; fix README/docstrings.
- Verification: every verdict carries an access tag; an LRT fit on noised rows only with σ²I subtracted
  (T2-MA) matching the stored LRT within seed noise confirms clean vectors do not drive Tier 2.

**F-02 Repeated-query exposure unpriced for noise defenses marked passing. High.**
- Noise is fresh per pass (`paper.tex:352-357`; `mi_ceiling.py:256`; `end_to_end_surgical.py:164-165`).
  Stored averaging covers only the subspace channel (easy, registered, N ≤ 16: 0.506 → 0.560, breaching at
  N=16; middle/hard post hoc in `tpr_capture.json` key `averaging`). Nothing covers the full-rank ✓/✓ row
  (`paper.tex:626`) or any baseline. σ_eff ≤ 1 needs N = 256 (σ=16) or 400/576/4096 (σ=20/24/64); max N = 16.
  "Effect is stronger on the other two" (`paper.tex:561`) holds for middle only (breach at N=4); hard
  0.559 at N=16 ≈ easy; not marked unregistered.
- Repair: report N* (releases to breach) per passing point; releases-per-target becomes an access axis.
- Verification: N* column; empirical full-rank N* agrees with σ/√N interpolation within one grid step.

**F-03 Output surface: wrong object and not gated. High.**
- Rule says every attacker ≤ bar on **both** surfaces (`paper.tex:261-262`; audit "either surface",
  `paper.tex:306-308`). Audit attacks representation only (`expansion_pipeline.py:150-186`;
  `honest_reaudit.py:62-80`) — the 8 survivors were never output-tested. Gauntlet representation-gated
  (`baseline_gauntlet.py:452`; `master_gauntlet_table.py:166`). Authors' channels not output-attacked in
  `two_tier_certification.py` (117-128, 147; disclosed `paper.tex:266-269`). The attacked object is the
  logit/log-prob vector, not ŷ (`paper.tex:197-200`). For LEACE, Obliviator, FARE, FNF, Fair PCA and
  post-hoc channels the "head" is an auditor-fit LR on all rows. No published-method verdict changes (all
  fail on the representation; VFAE-easy output 0.533); survivors and "ours" untested.
- Repair: one rule (both surfaces gate); state the released output object; attack ŷ and scores separately.
- Verification: re-gate gauntlet from stored `out_tier*_max` (no compute); audit survivors need refits (deferred).

**F-04 Out-of-scope or white-box comparisons. Medium.** VFAE sampled-z LRT gets encoder μ
(`baseline_gauntlet.py:345-352`). FARE's dp_ub bounds demographic parity; "leak race pairs at
0.603–0.610" are macro-OvR T1 AUCs of non-certified sweep rows (`paper.tex:692-694`;
`docs/fare_results.md` claim 4). LAFTR-official binary-A code run on 5-class race and a `band==1`
recoding (`laftr_official.py:29-40`), merged with the reimplementation in Table 1 (`paper.tex:617`).
Repair: tag VFAE T2 as T2-WB; LAFTR-official N/A on multiclass; reword FARE claim.

## B. Utility protocol

**F-05 In-sample utility vs held-out attackers; unstable denominators. High.**
- Only PCRL **train** split loaded (`pcrl_io.py:150,188`); encoders/erasers/heads fit on all rows incl.
  attacker-test rows; utility = own head on those rows (`diagnostic.py:205-206`;
  `two_tier_certification.py:150`; disclosed `paper.tex:263`, `appendix.tex:203-216`). Clean reference is
  σ=0 model **without BatchNorm** (`mi_ceiling.py:217`).
- Denominators: easy lift 0.0223 (1 pt = 45%, `paper.tex:219-223`), 0.0152 out of partition; Adult clean
  lift **0.069 out-of-sample vs 0.185 in-sample** (`fresh_partition_generalization.json`); >100% readings
  are in-sample artifacts (easy subspace 102.2% → 64.0% out of partition); LSAC clean lift 0.017 enters
  r=0.799. "Clean accuracies 0.915/0.619/0.895" are constructed, not measured (R27).
- Repair: disjoint encoder-fit / attacker-train / evaluation splits with one common held-out utility;
  absolute accuracy + lift with bootstrap CIs; pre-registered minimum clean-lift rule; same architecture for
  σ=0 reference.
- Verification: in-sample and held-out side by side; no ratio reported with CI width > 20 pp.

**F-06 Utility conventions differ inside Table 1. Medium.** Baselines `lift_best = max(own in-sample,
LR held-out)` (`baseline_gauntlet.py:380`); e2e in-sample own head; post-hoc held-out LR
(`targeted_noise.py:108-120`); subspace headline fresh partition; `appendix.tex:410-434` lists six
conventions. Hard subspace 84% (3-seed) vs 85.4% (5-seed); held-out 108.8% uses a different clean
reference (R18). Repair: one head rule and one denominator.

**F-07 Selection and certification on the same rows/seeds. Medium/High.** Sweep at train seed 0 with
probe seeds [0,1], best-utility candidate ≤ bar; certification reuses seed 0 and probe seeds [0,1,2] on the
same rows (`two_tier_certification.py:180-197`; `baseline_gauntlet.py` tier_pick); λ,(k,σ) tuned on same
rows (`appendix.tex:498`). Marginal passes: VFAE 0.535; hard subspace 0.536–0.543 (worst seed 0.561; 3/5
seeds breach); FARE hard 0.546. Aggregation order matters: middle full-rank T2 0.5496 (mean-then-max)
passes vs 0.553 (max-then-mean) fails (R16, R23). VFAE quoted 0.536/71% is the seed-0 sweep row; certified
point 0.5347/72.2% (R10). Repair: separate selection/certification splits with fresh seeds; gate on UCB95 ≤
bar; register one aggregation order. Verification: count verdict changes under a UCB gate from stored per_seed arrays.

**F-08 Isolate-vs-full-rank matching rule. High for `paper.tex:576-580`.**
`run_isolate_vs_fullrank.py:79-88` picks the full-rank σ whose seed-0 T1 is *closest* to isolate T1; near
chance T1 is flat in σ so an over-protected σ is chosen (32 → 1.9% easy; 16 → 38.6% middle). Stored
`two_tier_certification.json` has easy σ=20 at T1 0.509 (isolate 0.5094) with **55.5%** and middle σ=12 at
T1 0.541 (|Δ| 0.006) with **56.6%**. Easy-cell isolate advantage shrinks ~100 pp → ~46 pp; at the bar about
8/43/85 pp (R19); hard unaffected. Repair: compare frontiers (max full-rank utility s.t. T1 ≤ max(isolate
T1, bar), matched seeds). Verification: recompute from stored JSON (DG-12; R19).

**F-09 Degenerate passes counted as passes. Medium.** FARE easy passes at 0% (all 25 configs
majority-degenerate); full-rank hard passes at −0.5% and −7.0%; "seven of 42 pass" (`paper.tex:629`)
includes them. Repair: pass requires utility CI lower bound > 0; report degenerate passes separately.

## C. Splits, linkage, configuration matching

**F-10 Fit/attack overlap, no grouping. Medium.** Erasers fit on all rows incl. attacker-test rows: LEACE
(`baseline_gauntlet.py:309-323`), Obliviator (`obliviator_gauntlet.py:30-31`), projection Q
(`two_tier_certification.py:243`). ACS split by person without SERIALNO (`folktables_io.py:189-198`);
fairlearn diabetes encounter-level with repeat patients; CelebA image-level, no identity file. Fresh
partition covers memorization for 9 points (no flips, max |Δ| 0.013), not linkage. Repair: grouped splits;
fit erasers on encoder split only. Verification: zero group overlap; grouped vs ungrouped AUC deltas.

**F-11 Fresh-partition control changes more than the partition. Medium.** Encoder gets half the rows;
attacker training shrinks to 37.5% of rows; 3 seeds instead of 5; out-of-sample utility
(`run_fresh_partition.py:1-30,86-95,223-226`). "Flips no verdict" (`paper.tex:722-726`) confounds
memorization with attacker n. Repair: in-partition control with matched attacker n.

**F-12 Post-hoc and e2e arms not matched. Medium.** Post-hoc: σ=0 no-BN frozen P, relative σ_rel·σ_h,
held-out LR (`two_tier_certification.py:270-273`); e2e: BN h, absolute σ, in-sample own head. The
"end-to-end advantage" (`paper.tex:590-592`; Table 1) mixes these.

## D. Metrics and thresholds

**F-13 Multiclass handling. Medium/High (HMDA).** Verdicts use macro OvR AUC (`diagnostic.py:185`;
`battery.py:85`). Worst-pair = max over pairs of orientation-free pairwise AUC scored by p_j/(p_i+p_j),
skipping pairs < 10 rows (`run_worstpair_sweep.py:81-98`), then **max over 9 seeds**. HMDA race class 4 is
0.79% (126 test rows). Null 0.5556 is a max over 27 draws (mean 0.536), quoted as "no configuration can
clear it" (`paper.tex:749-750`). ≥3000-row support rule chosen after per-pair nulls; registered c084ad3
only for the follow-up sweep. 3-seed 49.7% quoted while 5-seed gate read 41.0% (R24). TPR@1%FPR
unregistered; its "worst class" = most-exposed class (as small as 126 rows). Repair: one pre-registered
worst-class criterion with permutation null, minimum support, seed averaging, applied to all methods.
Verification: null rejection ≤ 5% at real class sizes.

**F-14 Thresholds not pre-registered. Medium.** 0.55 gate first in Exp 5 (d8867f4, 2026-06-26) right
after Exp 3 saw 0.55–0.58 (55ca332); τ = 0.05 is PCRL's; registrations from 2026-07-23; bar/τ sensitivity
post hoc (`appendix.tex:387-403`); gauntlet points not reselected at alternative bars. Repair: curves and
utility-at-bar for a registered set of bars.

**F-15 Certificate is in-sample ridge R². Low.** PCRL Tikhonov one-hot `linear_r2` + dominant-axis R²
(`falsification_attack.py:127-146`) on the full matrix (`appendix.tex:237`); Prop 1 exact only for
unregularized OLS (measured |ΔR²| 1.7e-5).

## E. Mathematical claims

**F-16 Prop 2 inconsistencies, over-generalized. Medium.** Conditions on task label Y
(`appendix.tex:101-104`) though the claim is about s; moments said to be of the protected representation
(line 103) while the construction builds clean h (131-145); tail bound sketched; asymptotic regime not
reached (separation/σ 0.028–0.171; caption 0.016 from a mislabelled row, R26). Main text "no sound
certificate computed from these statistics alone can be useful" (`paper.tex:271-273,761-763`) exceeds the
proposition, which only forces a sound certificate to report at least the witness value (below the bar at
Tier-2 points). Classification: population existence claim under assumptions (sketch); main-text
generalization partly unsupported.

**F-17 Prop 3 correct, scope over-read. Medium.** Gaussian mechanism, replace-one adjacency, single
input's release: μ-GDP, μ = 2C/σ; AUC ≤ Φ(μ/√2) via G_μ tradeoff + joint convexity (`appendix.tex:256-320`).
Limits: single release (k releases → √k·μ); no auxiliary s-information; conditional on a frozen
non-privately trained map; input-level local indistinguishability, not dataset DP; DP does not lower the
population floor (`paper.tex:779-784`). Fig. 4/Table 1 full-rank points are **unclipped**
(`mi_ceiling.py:255-256`) — not covered. No attacker run on the clipped variant (`run_dp_fullrank.py`),
which keeps ~0% utility. "Certified full-rank channel ... under Tier 2" (`paper.tex:583-584`) conflates
certified and measured channels. `tab:dp` caption mixes δ = 1e-5 and 1e-6.

**F-18 Prop 1 sound. Low.** Algebraic (variational multiple R²); exact for scalar s and in-sample OLS;
held-out R² not covered; ridge only approximately.

## F. Data, CelebA, provenance

**F-19 CelebA vision check. High.** X is PCRL `celeba_v2` FiLM CNN output (`celeba_extract.py:8-20,54-69`);
both purposes list Young as disallowed (PCRL origin/main `pcrl/data/celeba.py:146-177`). Clean Young XGB
AUC 0.567/0.605 (`celeba_pipeline.json`) vs ~1.0 on tabular cells; paper calls it "a frozen encoder we did
not train" (`paper.tex:719-721`). Checkpoint sha256 b0df3fb7… (2026-04-16) predates and mismatches the
only tracked writer (`run_celeba_v2.py` trains `CNNPurposeProjectionEncoder` on 10k images) [UV]. Official
train partition only (162,770 images, unfiltered); image-level 75/25; no `identity_CelebA.txt`; in-sample
utility; cache local-only. Repair: documented CelebA-disjoint backbone, identity-grouped splits, recorded
hash. Verification: clean AUC ≥ 0.8, zero identity overlap, hash in JSON.

**F-20 Audit and cell composition. Medium.** 67 audit rows = adult 20, folktables 27, diabetes_hospital 14
(clean lift 0.0005, "no task lift" `appendix.tex:444`), lawschool 5, hmda 1, dutch 0. Headline cells chosen
after earlier results (`diagnostic.py:80-90`) though paper says "named by this predictor and chosen to span
its range" (`paper.tex:217`). 20 PCRL-source cost cells scouted from an unrecorded 30-pair pool;
readmission dropped as "known-degenerate" (`continuous_cost.py:63-86`). External rules R1–R4 label-only and
committed before results (d9bce6a < a537b15), but the 0.52–0.60 band came from earlier results; "seven
external" = 9 selected − 2 lift-free. `continuous_cost.json` `stats` holds n=20, not 27; bootstrap has 6
clusters; constructed cells from 3 base cells.

**F-21 No PCRL pin; local-only inputs. High (reproducibility).** `pcrl_io.py:30-53` imports `pcrl.*`;
checkpoints adult 1cfc2fef…, hmda e29d0367…, celeba b0df3fb7…; likely tree origin/main@962841995 or
d39211214 [UV]; 739 untracked files (~964 MB) hold raw scores behind FARE/FNF/TPR/worst-pair/max-seed/fleet.
Repair: store PCRL commit + checkpoint hashes in JSON; archive `analysis/`. Verification: re-hash against
inventory; reproduce falsification R² 0.0349.

**F-22 s is an encoder input everywhere. Medium (design).** `appendix.tex:481`; LAFTR `use_attr=true`.
Explains clean recovery ≈ 1.0 (`paper.tex:225`) and why non-noise methods fail. Repair: add a regime with s
excluded from the input.

**F-23 Untouched data. Info.** ACS CA 2018 1-Year only (`folktables_io.py:56-57`); NY/TX in no revision;
PCRL val/test and dg 20% external holdouts unread; CA-2018 shared with PCRL ACS studies, which treat 2018
as spent [UV].

## G. Recounts

**F-24** `recounts.json`: 30 entries, 21 match. Mismatches: R24 (49.7 vs 41.0%; null max vs mean); R10
(VFAE sweep row vs certified point); R11 (LAFTR "3–17%", Obliviator "85–99%" ranges follow no declared
rule); R25 (LRT range 0.66–0.94 should be 0.66–0.86); R26 (Prop-2 caption 0.016 from mislabelled row; README
"Table 2" command prints superseded witness); R27 (clean accuracies constructed); R18 (84 vs 85.4); R30
(FARE "pairs" are macro OvR); R01 (README 21/18/3 holds only under Partition A; file verdict field supports
3 of 18; 3 duplicate pairs; σ=0 included). Matches with caveats: 67/59/8 (representation-only, XGB+MLP;
64 distinct), 94/102 (not in paper), 1/36 and 7/42, r=0.799/ρ=0.828/n=27 + robustness correlations,
out-of-partition numbers, bar sensitivity, max-seed 6/7, worst-pair 0.01–0.11, TPR 1.4–3.7%, 5/8 survivors
leak at T2.
