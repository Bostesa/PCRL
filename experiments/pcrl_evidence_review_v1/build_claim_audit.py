"""Emit EVIDENCE_LEDGER.csv and CLAIM_AUDIT.csv for Terminal B's audit.

Every row carries the exact source wording, the file and location it comes from,
the split and inference scope it applies to, a verdict, and a command that
regenerates the number. The rows are held here as data so that the CSVs are a
build product rather than hand-typed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

STUDY = 'results/redesign_20260917_acs_spectral_transport_v1'
MINE = 'results/pcrl_evidence_review_v1'

R = f'python -m experiments.pcrl_evidence_review_v1.reanalyse_transport --study <local> --published {STUDY} --out {MINE}'
T = f'python -m experiments.pcrl_evidence_review_v1.tradeoff_tables --published {STUDY} --out {MINE}'
K = f'python -m experiments.pcrl_evidence_review_v1.recheck_lock --study <local>/{STUDY} --root <local> --out {MINE}'
A = 'python -m experiments.pcrl_evidence_review_v1.admit_acs_2016 --data data/acs_2016_admission --out ' + MINE
F = 'python -m pytest tests/pcrl_evidence_review_v1 -q'

# (claim_id, wording, source, split, inference_scope, verdict, finding, replacement, numeric_change, command)
CLAIMS = [
 ('C01',
  'Existing income, employment and coverage prediction vectors are released bitwise unchanged in every condition.',
  f'{STUDY}/METHOD_AND_SCOPE.md §1; TRANSPORT_RESULTS.md §8',
  'all 2017 release partitions', 'structural property of the construction',
  'supported',
  'Verified on 210 released views (3 seeds x 5 partitions x 14 interfaces); 0 violations. Independent of any statistical assumption.',
  'no change',
  'no', 'INDEPENDENT_VERIFICATION.json check source_output_identity'),

 ('C02',
  'Exact output preservation does not imply stable service accuracy after a year of shift.',
  f'{STUDY}/METHOD_AND_SCOPE.md §1; TRANSPORT_RESULTS.md §9',
  '2017 final partition', 'the three fixed fitted systems',
  'supported',
  'Employment log loss rises 0.0138 nats and income falls 0.0145 nats from 2018 to 2017, on identical released vectors.',
  'no change', 'no', T + ' (SERVICE_QUALITY_SUMMARY.csv)'),

 ('C03',
  'The A-only residual spectral channel adds residence capability beyond the fixed services.',
  f'{STUDY}/RESEARCH_DECISION.md; TRANSPORT_RESULTS.md §2, §8',
  '2017 final partition, Modes A and B', 'the three fixed fitted systems, under the tested probes',
  'supported with an omitted qualification',
  'The >=0.01-nat rule over H passes in every seed, weighting and mode. The report does not say beside it that the '
  'stronger half-headroom reference criterion (against PCA32 and the richer banks) fails in 2 or 3 of 3 seeds for '
  'every interface in every mode. Reporting only the H-relative rule flatters the capability claim.',
  'Add: the gain is real relative to the published service, but no interface comes close to the capability of the '
  'unrestricted PCA32/rich-bank references (half-headroom passes in 0-2 of 3 seeds).',
  'no', f'read {STUDY}/evidence/CRITERIA_SUMMARY.csv columns half_headroom_pass_seeds, source_allowance_pass_seeds'),

 ('C04',
  'spectral_C1 has a simultaneous-interval advantage over both spectral_L1 and spectral_L2 under both weightings, '
  'with the residence difference inside the prespecified 0.001 band.',
  f'{STUDY}/TRANSPORT_RESULTS.md §1, §4; PAPER_DRAFT.md abstract and §7',
  'Mode B, transport_all, budget 360, 2017 final partition',
  'people in the 2017 final partition, three fixed fitted systems, tested attack families',
  'supported as a rule outcome; overstated as an interval statement',
  'The advantage half reproduces exactly (8 of 8 sensitive endpoints better, 0 worse; c=2.930106). The registered '
  'residence rule is `residence_difference <= .001 + 1e-12` on the SEED-MEAN POINT estimate and is one-sided, not a '
  'two-sided band. No F1 residence interval lies inside +/-0.001: the simultaneous intervals reach 0.00176, 0.00175, '
  '0.00191 and 0.00172, and even a pointwise one-sided 95% bootstrap upper bound is above the margin in all four '
  '(0.00112, 0.00103, 0.00135, 0.00102).',
  'The registered rule passed. Retrospectively, the data do not establish noninferiority at 0.001: 0 of 4 '
  'comparisons have an upper confidence bound below the margin, under any of the procedures we computed.',
  'no (the rule outcome is unchanged; the interval claim is new and negative)',
  R + ' (EQUIVALENCE_F1.csv)'),

 ('C05',
  'Every spectral arm leaks several times more sex and race than a frozen adversarially trained channel of the same width.',
  f'{STUDY}/PAPER_DRAFT.md §7; TRANSPORT_RESULTS.md §1',
  'Mode B, transport_all, 360, unweighted', 'tested attack families on the 2017 final partition',
  'supported but scale-dependent and unlabelled',
  'The "several times" ratio is computed on H-relative INCREMENTS (C1 0.0453 vs J 0.0076 on A/race, a factor of 6.0). '
  'On ABSOLUTE recovery the same pair is 0.0617 vs 0.0241, a factor of 2.6. The manuscript does not say which scale '
  'the ratio uses. The paired DIFFERENCE is identical on both scales (0.0377), because the shared prior and H '
  'selection cancel inside a paired contrast.',
  'Report the difference (0.0377 [0.0294, 0.0459] nats on A/race), state both levels, and drop the unlabelled ratio.',
  'no (all three numbers verified)', R + ' (MODEB_TRANSPORT_TABLE.csv)'),

 ('C06',
  'spectral_C1 versus J fails the same criterion on transport (local race significantly worse).',
  f'{STUDY}/TRANSPORT_RESULTS.md §1, §5',
  'Mode B, transport_all, 360, family F3', 'as C04',
  'supported',
  'F3 fails on two counts, not one: every sensitive endpoint has an adjusted lower bound above zero (A/SEX, AB/SEX, '
  'A/RAC1P, AB/RAC1P, both weightings), so the "no endpoint significantly worse" clause fails eight times over, and '
  'A/RAC1P triggers the separate local-race failure. C1 is better only on residence (-0.0093 [-0.0150, -0.0036]).',
  'State that C1 is significantly worse on all four sensitive endpoints under both weightings, not only on local race.',
  'no', R + ' (INDEPENDENT_FAMILIES.csv, family F3_secondary)'),

 ('C07',
  'C1 has more residence utility than J, so the comparison is a tradeoff rather than a dominance result.',
  f'{STUDY}/RESEARCH_DECISION.md; PAPER_DRAFT.md abstract',
  'Mode B, transport_all, 360', 'as C04',
  'supported and strengthenable',
  'Correct, and the randomized-withholding family makes it sharper than the report says. Expected residence gain under '
  'the declared mechanism is exactly p times the full-release gain, so the matching probability is identified in closed '
  'form. J cannot be made as useful as C1 at any p: it would need p* = 1.49. Only A0 (p*=0.976) and spectral_S0 '
  '(p*=0.929) can be withheld down to C1s mean residence gain, and at those p C1 is better on all four sensitive '
  'endpoints. In the other direction C1 withheld to Js residence level (p*=0.670) still leaks far more '
  '(A/race 0.0303 vs 0.0076).',
  'Add the matched-utility calculation in both directions and label it post-hoc average-utility matching.',
  'no (new exploratory analysis, arithmetic exact given the declared mixture)',
  T + ' (WITHHOLDING_MATCHING.csv)'),

 ('C08',
  'Every F1 endpoint keeps its development sign (20/20). What changed is resolution, not direction.',
  f'{STUDY}/TRANSPORT_RESULTS.md §12; PAPER_DRAFT.md §7',
  'development 2018 evaluation pool vs 2017 final partition', 'seed-mean point estimates',
  'overstated',
  'The 20/20 count is over SEED-MEAN signs. Only 13 of 20 F1 endpoints have all three transport seeds matching the '
  'development sign, and across all 90 family endpoints only 48 of 90 have all three seeds agreeing even with their '
  'own seed mean. "Resolution, not direction" is also incomplete: F1 point estimates moved by factors between 0.054x '
  'and 14.8x between the years (C1-L2 AB/SEX went from -0.0003 to -0.0047). A larger sample shrinks standard errors; '
  'it does not move point estimates by an order of magnitude. Year, evaluation pool and sample size all changed '
  'together and are not separated.',
  'Say: all 20 seed-mean signs agree; 13 of 20 agree in every seed; magnitudes changed by factors of 0.05x to 14.8x, '
  'so sample size alone does not explain the change in significance.',
  'no (both the 20/20 and the per-seed counts verified)', T + ' (PER_SEED_DIRECTION.csv, DEV_VS_TRANSPORT.csv)'),

 ('C09',
  'Under fresh adaptation (Mode B) every interface passes the legacy source-probe allowance in all three seeds, '
  'while under frozen transfer (Mode A) every spectral arm fails in 0-1 of 3 seeds.',
  f'{STUDY}/TRANSPORT_RESULTS.md §8',
  'Modes A and B, 2017 final partition', 'per seed and weighting, never averaged',
  'contradicted (pass/fail inversion)',
  'The evidence column is source_allowance_PASS_seeds. Mode A spectral arms record 0 or 1, i.e. they PASS in 0-1 of 3 '
  'seeds and therefore FAIL in 2-3 of 3. The sentence states the pass count as a fail count and so reports the '
  'opposite of the data: as written it implies frozen transfer is nearly clean, when in fact the frozen 2018 source '
  'probes fail the allowance in most seeds for every spectral arm.',
  'Under frozen transfer every spectral arm PASSES in only 0-1 of 3 seeds (fails in 2-3 of 3); under fresh adaptation '
  'all 14 interfaces pass in 3 of 3.',
  'no (the table was right; the prose inverted it)',
  f'read {STUDY}/evidence/CRITERIA_SUMMARY.csv, column source_allowance_pass_seeds, mode A'),

 ('C10',
  'Two lock amendments are recorded, both code-only.',
  f'{STUDY}/TRANSPORT_RESULTS.md §14; VALIDATION.md §3 and §5; PAPER_DRAFT.md §10; REVIEW_INDEX.md',
  'lock chain', 'audit trail',
  'contradicted (count) / supported (substance)',
  'Three amendment files exist. Amendment 3 (2026-09-17T18:30:13Z) re-binds '
  'scripts/report_acs_spectral_transport.py for a CSV-to-gzip serialisation change. It is genuinely code-only and '
  'post-dates the last final-partition access (17:54:45Z), so it cannot have touched a score. But the COMMITTED '
  'INDEPENDENT_VERIFICATION.json lists only amendments 1 and 2: the verifier was re-run locally after amendment 3 and '
  'passes with all three, and that re-run was never committed. An independent recheck of all 17,639 hashed inputs '
  '(1.94 GB) with every amendment applied passes with 0 changed, 0 missing and 0 invalid entries.',
  'Three code-only amendments are recorded; all final reads precede the third; the published verification artifact '
  'must be regenerated so it lists all three.',
  'no (no score, selection or interval is affected)', K),

 ('C11',
  'Negative selected increments are retained as measurements and are not evidence of protection.',
  f'{STUDY}/TRANSPORT_RESULTS.md §10; PROTOCOL.md §5',
  'Mode B, transport_all and common_fresh, 360', 'validation-selected attacks only',
  'supported',
  '22 of 429 unweighted role-cells are negative in transport_all (26 of 429 in common_fresh). J\'s own A/SEX increment '
  'is negative under PWGTP with an interval excluding zero. Separately, in 8 of 312 sensitive cells the selected '
  'attack on the augmented wire does worse than H\'s own attack routed onto that wire (15 of 312 under common_fresh) '
  '- and that routed H attack remains executable, so the information is available regardless of what selection picked.',
  'no change', 'no',
  f'read {STUDY}/evidence/NEGATIVE_INCREMENTS.csv and SELECTED_VS_ROUTED_H.csv'),

 ('C12',
  'Adding frozen 2018 attackers lowers additional recovery because it strengthens the subtracted H baseline.',
  f'{STUDY}/TRANSPORT_RESULTS.md §10',
  'Mode B, common_fresh vs transport_all, 360', 'as C04',
  'supported',
  'Reproduced for every interface and endpoint. The change is a baseline effect, not a change in what the augmented '
  'wire discloses: absolute recovery on the augmented wire moves far less than the increment does.',
  'no change', 'no', T + ' (SCOPE_COMPARISON.csv)'),

 ('C13',
  'Race class 3 (Alaska Native alone) has one person in the final partition, so class-level race statements are withheld.',
  f'{STUDY}/TRANSPORT_RESULTS.md §13; DATA_ADMISSION.md',
  '2017 partitions', 'class-level statements only',
  'supported',
  'Confirmed: final-partition support vector is [9053, 837, 112, 1, 37, 2544, 68, 2399, 873] and 2017 attacker '
  'validation contains none of class 3. Full nine-class log loss is always scored with the 1e-12 floor, and the '
  'comparable-category diagnostic reproduces the F1 race differences. A class with no support is not "protected": '
  'it is unmeasured.',
  'no change (wording already avoids calling it protected)', 'no',
  f'read {STUDY}/seed_0/spectral_C1/mode_B/metrics.json support field; DATA_ADMISSION.md'),

 ('C14',
  'The spectral step is a global optimum of its fixed matrix, so the gap is the surrogate, not the solver.',
  f'{STUDY}/RESEARCH_DECISION.md',
  'construction', 'the fixed empirical matrix only',
  'supported in the first half, overstated in the second',
  'Global optimality for the fixed matrix is Ky Fan and is verified against direct least squares for every arm, so a '
  'failure to solve the stated problem is excluded. That leaves five candidate causes unseparated: the finite feature '
  'map, nuisance misspecification, the whitening/normalisation, the fixed rank 16, the reconstruction surrogate '
  'standing in for downstream utility, and finite-sample generalisation. Nothing in the ACS measurements distinguishes '
  'them. The counterexamples prove a linear-in-Z penalty CAN be defeated by a nonlinear attacker; they do not show '
  'that this is what happened here.',
  'Replace "the gap is the surrogate" with "consistent with a surrogate mismatch; the design does not separate it from '
  'four other candidate causes".',
  'no', f'{MINE}/METHOD_REVIEW.md §9; ' + F),

 ('C15',
  'If a conditional penalty that acts on nonlinear functions of Z also leaks more than J, the conclusion generalizes '
  'to the closed-form spectral family for this interface.',
  f'{STUDY}/RESEARCH_DECISION.md ("One next decision")',
  'proposed follow-up', 'family-level inference',
  'unsupported; removed',
  'Invalid twice. Two arms failing is evidence about two arms. And a penalty nonlinear in Z is provably not a trace '
  'form tr(W\'AW) for any W-independent A - it is not invariant under W -> WQ for orthogonal Q, while every trace form '
  'is - so the proposed successor is not in the closed-form family at all and its failure would say nothing about it.',
  'Removed from the corrected paper and flagged to Terminal A as finding A5.',
  'no', f'{MINE}/METHOD_REVIEW.md §4, §9; ' + F +
  ' (test_kernelised_output_features_break_the_trace_form)'),

 ('C16',
  'Selecting rank by the sign of the eigenvalues is the proposed improvement, adapted from U-FaTE.',
  f'{STUDY}/RESEARCH_DECISION.md',
  'proposed follow-up', 'novelty attribution',
  'misattributed',
  'Eigenvalue-sign rank selection is already the published result in this line, three times over: SARL Theorem 3 '
  '(gamma = min{r, j} with j the number of negative eigenvalues of B), OptNet-ARL Theorem 4.1 (optimal dimension = '
  'number of negative eigenvalues), and K-TOpt Corollary 4.1 (number of non-negative eigenvalues). U-FaTE is NOT the '
  'source: its Theorem 1 takes the r LARGEST eigenvalues and it sets r = c-1 from the number of target classes. '
  'U-FaTE contributes the conditional penalty, not the rank rule.',
  'Attribute sign-based rank selection to SARL/OptNet-ARL/K-TOpt and the conditional penalty to U-FaTE.',
  'no', f'{MINE}/NOVELTY_MATRIX.md rows SARL, OptNet-ARL, K-TOpt, U-FaTE'),

 ('C17',
  'The protocol, families, endpoints and decision rules were fixed and pushed before the final partition could be read.',
  f'{STUDY}/PROTOCOL.md; RESEARCH_DECISION.md; VALIDATION.md §3',
  'lock and access log', 'evidential status of the run',
  'supported',
  'The lock (17,639 inputs) was created 16:46:04Z and all 10 final accesses fall in 16:46:33-17:54:45Z under the same '
  'lock digest. No directional prediction was registered, and the study says so. Prespecified contrasts, endpoint '
  'families and rules still carry evidential value: they fix the multiplicity, the estimand and the stopping point '
  'before the outcome. The absence of a called shot does not make the run exploratory.',
  'no change (the disclosure is already correct and should be kept verbatim)', 'no',
  f'read {STUDY}/FINAL_ACCESS_LOG.json and TRANSPORT_LOCK.json created_utc'),

 ('C18',
  'No withholding mechanism dominates spectral_C1 across all seeds and both weightings; the frontier does not reduce '
  'to withholding a simpler channel.',
  f'{STUDY}/TRANSPORT_RESULTS.md §11; PAPER_DRAFT.md §7',
  'Mode B, transport_all, 360, fixed p grid', 'the specified routed family only',
  'supported but asymmetrically reported',
  'Both directions matter and only one is given prominence: withholding dominates a given spectral arm in 1-3 of 180 '
  'fixed comparisons and a spectral arm dominates withholding in 0-3 of 180. Neither family is robustly better. A '
  'finite list of controls failing to dominate is not evidence of efficiency.',
  'Report both counts and state explicitly that non-dominance by these controls does not establish that C1 is on any '
  'frontier.',
  'no', f'read {STUDY}/evidence/WITHHOLDING_DOMINANCE.csv'),

 ('C19',
  '2017 is admissible as a locked temporal evaluation and 2016 remains the unused candidate year.',
  f'{STUDY}/DATA_ADMISSION.md; RESEARCH_DECISION.md',
  'admission', 'provenance and schema only',
  'supported; 2016 now prepared',
  'The 2016 California one-year person file validates against the 2017 code ranges with zero violations: 376,035 raw '
  'rows, 79,298 eligible people in 53,093 published groups, 0 duplicate or conflicting person keys. All 16 of 17 '
  'feature/target/sensitive variables have identical dictionary code sets to 2017; PINCP differs only in its declared '
  'top-code ceiling (2..9999999 in 2016 vs 2..4209995 in 2017) and no observed 2016 value falls outside the 2017 '
  'range. ADJINC is 1.007588 (2017: 1.011189; 2018: 1.013097), so nominal income shifts the income estimand further. '
  'SERIALNO is 9 digits with no year prefix, so year-qualified keys are required. Every RAC1P class has support in '
  'every fitting and validation pool, including class 4 (Alaska Native) with 2 people in attacker validation - the '
  'support gap that limited 2017. No 2016 model was fitted, no row transformed, no prediction made and no final '
  'label read.',
  'Record 2016 as admissible on provenance and schema, with the income estimand change documented, not repaired.',
  'no', A),

 ('C20',
  'Uncertainty is household-resampling uncertainty for fixed fitted systems, not retraining or Census design variance.',
  f'{STUDY}/PROTOCOL.md §6; METHOD_AND_SCOPE.md §4',
  '2017 final partition', 'people in the final partition, three fixed systems',
  'supported',
  'Independently reconstructed: 2,000 replicates over 10,701 households, shared draws across every method, seed, task '
  'and weighting; ratio estimators recomputed inside each replicate; both rows of a household share a resample count; '
  'three seed predictions for one person are one person. All 90 family rows reproduce to 1.1e-15 on estimates and '
  '2.4e-15 on interval endpoints, with identical critical values.',
  'no change', 'no', R + ' (FAMILY_AGREEMENT.json)'),
]

LEDGER_HEADER = ['claim_id', 'claim_wording', 'source', 'data_split', 'inference_scope',
                 'verdict', 'finding', 'replacement_wording', 'numerical_result_changed',
                 'regenerating_command']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / 'EVIDENCE_LEDGER.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(LEDGER_HEADER)
        w.writerows(CLAIMS)

    # the claim audit is the ledger restricted to the decision-relevant columns,
    # ordered so that contradicted and overstated claims come first
    order = {'contradicted (pass/fail inversion)': 0, 'contradicted (count) / supported (substance)': 1,
             'unsupported; removed': 2, 'misattributed': 3, 'overstated': 4}
    rows = sorted(CLAIMS, key=lambda r: (order.get(r[5], 9), r[0]))
    with open(out / 'CLAIM_AUDIT.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['claim_id', 'verdict', 'claim_wording', 'source', 'finding',
                    'replacement_wording', 'numerical_result_changed', 'regenerating_command'])
        for r in rows:
            w.writerow([r[0], r[5], r[1], r[2], r[6], r[7], r[8], r[9]])

    counts = {}
    for r in CLAIMS:
        counts[r[5]] = counts.get(r[5], 0) + 1
    print(f'{len(CLAIMS)} claims written')
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f'  {v:>2}  {k}')


if __name__ == '__main__':
    main()
