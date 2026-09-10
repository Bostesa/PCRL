"""Publish explicit unavailable cells for the proven empty-household V1.

This report cannot turn historical replay into a new scientific result.
"""
from __future__ import annotations
import csv,io,itertools,json
from pathlib import Path
from experiments.acs_locked_evaluation import ARMS,ROLES,TASKS,SCOPES,atomic_json
from scripts.run_acs_locked_evaluation import OUT,verify


def write(path,text):
    if path.exists():
        assert path.read_text()==text,('Incompatible completed report',str(path));return
    path.write_text(text)


def csvout(path,rows):
    h=io.StringIO();w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows);write(path,h.getvalue())


def run(out=OUT):
    lock=verify(out);assert lock['status']=='blocked_no_unused_households' and lock['sample_rows']==0
    obj=json.loads((out/'FROZEN_OBJECTS.json').read_text());replay=json.loads((out/'HISTORICAL_REPLAY.json').read_text())
    reason='No unused eligible households; no new outcome scored'
    rows=[];availability=[]
    for z in obj['systems']:
        s,c=z['seed'],z['condition']
        availability.append({'seed':s,'condition':c,'frozen_objects_available':True,'sample_rows':0,'sample_households':0,'inference_status':'blocked','score_status':'unavailable','independent_evaluation_complete':False,'reason':reason})
        for kind,records in [('utility',z['utilities']),('audit',z['audits'])]:
            for r in records:
                for weight in ('unweighted','PWGTP'):
                    rows.append({'seed':s,'condition':c,'kind':kind,'endpoint':r['role'],'budget':r.get('budget'),'scope':r.get('scope'),
                                 'weight':weight,'candidate_id':r['candidate_id'],'candidate_path':r['path'],'value':None,'pass':None,'status':'unavailable','reason':reason})
        for t in TASKS[:3]:
            for weight in ('unweighted','PWGTP'):
                rows.append({'seed':s,'condition':c,'kind':'native_service','endpoint':t,'budget':None,'scope':None,'weight':weight,
                             'candidate_id':Path(z['anchors'][t]).name,'candidate_path':z['anchors'][t],'value':None,'pass':None,'status':'unavailable','reason':reason})
    csvout(out/'PER_SEED.csv',rows);atomic_json(out/'SYSTEM_AVAILABILITY.json',availability)
    comparisons=[]
    for s,c,w,b,scope,endpoint in itertools.product(range(3),('L025','L20','A0','E','H'),('unweighted','PWGTP'),(120,360),SCOPES,
                                                  [*TASKS,*[v+'/'+t for v,ts in ROLES.items() for t in ts]]):
        comparisons.append({'seed':s,'J':'J','comparator':c,'weight':w,'budget':b,'scope':scope,'endpoint':endpoint,'J_minus_comparator':None,'status':'unavailable','reason':reason})
    csvout(out/'PAIRED_COMPARISONS.csv',comparisons)
    source=[]
    for s,c,w,t in itertools.product(range(3),ARMS,('unweighted','PWGTP'),TASKS[:3]):
        source.append({'seed':s,'condition':c,'weight':w,'task':t,'native_loss':None,'selected_readout_loss':None,'fresh_PCA32_reference_loss':None,'allowance':.01,'pass':None,'reason':reason})
    csvout(out/'SOURCE_CRITERIA.csv',source)
    panels=json.loads((Path(__file__).resolve().parents[1]/'results/redesign_20260909_acs_fixed_predictions_v1/comparison_rules.json').read_text())['panels']
    matches=[]
    for s,c,w,b,scope,panel,delta,attribute in itertools.product(range(3),('L025','L20','A0','E','H'),('unweighted','PWGTP'),(120,360),SCOPES,panels,(0,.0005,.001,.002),('SEX','RAC1P')):
        matches.append({'seed':s,'J':'J','comparator':c,'weight':w,'budget':b,'scope':scope,'panel':panel,'delta':delta,'attribute':attribute,'both_source_feasible':None,'utility_close':None,'utility_directional':None,'lower_recovery':None,'qualifies_close':None,'qualifies_directional':None,'reason':reason})
    csvout(out/'UTILITY_MATCHES.csv',matches)
    uncertainty={'status':'unavailable_no_households','planned_replicates':2000,'executed_replicates':0,'households':0,
                 'contrasts':[{'name':n,'point':None,'seed_estimates':[None,None,None],'seed_sample_sd':None,'percentile_95':None,'simultaneous_95':None,'status':'unassessable','reason':'zero selected household groups'} for n in lock['bootstrap']['contrasts']],
                 'rng_seed':20260910,'quantile':'linear','new_model_inference_calls':0,'bootstrap_draws_created':False}
    atomic_json(out/'UNCERTAINTY.json',uncertainty)
    completion={'status':'provenance_blocker_verified_and_reported','provenance_audit_complete':True,'frozen_inventory_complete':True,
                'historical_plumbing_complete':True,'scientific_evaluation_complete':False,'scientific_inference_systems_completed':0,
                'scientific_scored_rows':0,'planned_systems':18,'blocked_systems':18,'utility_selections':90,'audit_selections':1188,
                'required_missing_objects':obj['required_missing'],'sample_rows':0,'sample_households':0,'scientific_model_fits':0,'optimizer_updates':0,
                'candidate_reselections':0,'bootstrap_replicates_executed':0,'historical_replay_is_independent_confirmation':False,
                'per_seed_unavailable_rows':len(rows),'paired_unavailable_rows':len(comparisons),'matching_unavailable_rows':len(matches),
                'pending_scientific_units':'All18 inference systems, fresh source/native/readout/reference/prior/diagnostic scores, contrasts, matching, uncertainty and observed tradeoff figure. Cannot execute on this file under required independence rule.'}
    atomic_json(out/'COMPLETION.json',completion)
    table='''# Locked untouched-household evaluation: unavailable

**0 unused eligible persons in0 households.** Earlier executed v2 Folktables research exhausts all80,329 persons in53,907 households of the required population. These are unavailable outcomes, not zero losses/gains or failed scientific comparisons.

| Fixed system (all three seeds) | Objects recovered | Fresh residence loss U / PWGTP | Fresh AB SEX gain U / PWGTP | Independent evaluation |
|---|---|---|---|---|
'''
    for c in ARMS:table+=f'| {"H+E_A" if c=="E" else "H" if c=="H" else "H+"+c} | Yes | Unavailable | Unavailable | Blocked: no households |\n'
    table+='''
All18 systems and their90 utility /1,188 attack selections are inventoried. [PER_SEED.csv](PER_SEED.csv) enumerates every unavailable selected endpoint and original native service, for both weightings. [Paired comparisons](PAIRED_COMPARISONS.csv), [source criteria](SOURCE_CRITERIA.csv), [matching cells](UTILITY_MATCHES.csv), and [completion](COMPLETION.json) keep failures of availability explicit. No fresh class-support, confusion, PWGTP effective-sample-size, baseline disclosure or means/SD can be estimated.

The historical parent table remains [development evidence](../redesign_20260909_acs_fixed_predictions_v1/RESEARCH_DECISION.md). Its values are not transplanted into this empty evaluation. See [DATA_INDEPENDENCE.md](DATA_INDEPENDENCE.md) for the decisive exclusion evidence.
'''
    write(out/'TABLE.md',table)
    write(out/'ANCHOR_PARITY.md',f'''# Anchor and release parity

Fresh-sample native anchor/readout scores are unavailable: the sample has0 rows. No vacuous empty-array equality is reported as scientific preservation success.

Historical plumbing replay independently reproduced **{replay['anchor_arrays']} original full anchor arrays**, **{replay['release_arrays']} release arrays**,21 teacher arrays and21 raw-preprocessor/PCA transformations exactly. All18 system release graphs preserve H's full anchor columns in float64. B selected outputs/scores pass960 historical parity checks across alternatives;1,050 raw H-projection witnesses and26 selected singleton witnesses match exactly.

[HISTORICAL_REPLAY.json](HISTORICAL_REPLAY.json) records exact output comparisons, selected saved states and hashes. The new graph itself was independently checked on all18 historical representation-fit fixtures (90 wire/derived arrays), with unchanged mapper/teacher state; see [CODE_REVIEW.md](CODE_REVIEW.md). Identical batching is required for bitwise float32 parity: a different GEMM batch shape produced rounding up to1.91e-6 in a preliminary small slice, within the original mapper tolerance1e-5. No tolerance was changed to affect a scientific comparison.

Fixed service preservation is a design property, distinct from accuracy on another population and from the selected source-readout PCA32+.01 allowance. Neither fresh quantity can be checked here.
''')
    write(out/'AUDIT_FINDINGS.md','''# Audit findings and access limits

No new recovery gain, additional recovery, race regression or opposing-task contrast was measured. All1,188 historical selections are frozen across11 roles×2 budgets×3 scopes×18 systems; the same IDs supply both weightings.990 requested H-selected projection slots are inventoried (aliases are retained as witnesses). The independent historical replay checks1,050 projection records because it covers the union of selected historical records, not only those990 named slots.

All486 candidate directories and all required core objects were recovered. Original audit priors retain their full class vectors and fitting provenance; they include original pseudocount1 and are distinct from observer entropy priors. Required core candidates,21 priors,42 selected exposed diagnostic budget/target controls and9 original PCA32/rich/tree residence context candidates are bound in [FROZEN_OBJECTS.json](FROZEN_OBJECTS.json). These counts overlap unique directories where aliases occur. Every MLP model.pt matches its selected_state_hash; terminal optimizer states are not evaluation checkpoints.

Historical race-code4 absence in attacker fitting/validation remains unresolved by this investigation; the9-category schema is preserved. No all-category protection pass is possible. Independent versus catch-up and unweighted versus weighted new contrasts are unavailable, rather than suppressed for unfavorable signs. A synthetic fixed-selected augmented attacker loses to its available H witness, and signed negative additional recovery remains negative without reselection or monotonicity repair.

Recipients receive only one system/seed's A/B or AB wire and legal public functions. Raw PCA/ACS, other conditions/seeds/versions and exposed-label controls are not recipient releases. Historical fidelity is not resistance to newly fitted or optimal attackers, public-ACS linkage, or deployed privacy. Missing untouched households prevent the requested out-of-sample fixed-audit assessment itself.
''')
    write(out/'MATCHING_ANALYSIS.md','''# Utility matching: no new comparisons are assessable

J versus L025 and J versus L20 are **unassessable under both weightings**, every scope/budget and every utility panel/delta. There are no qualifying/failed scientific seeds to count; all three seed cells are unavailable. H/A0/E remain mandatory fixed comparators and are likewise unavailable. Null flags in the machine-readable [UTILITY_MATCHES.csv](UTILITY_MATCHES.csv) must not be interpreted asFalse,0/3 or equivalence.

The four original panels and deltas0/.0005/.001 primary/.002 remain unchanged. Both methods must pass all three new-row frozen PCA32+.01 source allowances. Close checks each absolute task difference; directional checks each signed J−control loss. Roundoff1e-12 is separate. SEX and race recovery comparisons are distinct, as are A/B/AB views; no opposing-task threshold is invented.

The mature historical comparison verifier independently passed11,520 matching rows,34,560 paired comparisons and15,360 aggregate checks with maximum arithmetic discrepancy3.47e-18. It ran against a temporary aggregate-input symlink tree, writing only this new study's report. The synthetic fixture confirms that close matching can fail because J has better residence utility while directional matching succeeds. Neither a missing close match nor this unavailable evaluation proves inferiority or impossibility.

The original residential context formula is the positive PCA32-versus-stronger-rich-bank half-headroom, not improvement over H. Exact frozen PCA32/B/C residence reference identities are inventoried; no new-row denominator or fresh best-bank selection was computed. The context panel is unavailable, not silently replaced by historical losses.
''')
    write(out/'UNCERTAINTY.md','''# Household uncertainty: unavailable

All six declared paired contrasts are **unassessable** because there are0 selected household groups. No point estimate, seed SD, percentile interval or simultaneous band exists. Planned replicates:2,000; executed:0. No draws were made, replaced or discarded. [UNCERTAINTY.json](UNCERTAINTY.json) retains every unavailable contrast.

The implemented supplement uses household-level loss-difference sums and eligible count/PWGTP denominators, common multiplicities across all conditions/targets/three fitted seeds, and per-replicate ratios. It averages three per-seed metrics, never probabilities. RNG seed20260910 and linear/type7 quantiles are fixed. Ordinary95% percentile intervals and the exact six-member max-absolute-standardized centered-bootstrap family are implemented; zero variance is degenerate and undefined replicates make the affected interval unavailable. The full family is not silently reduced.

Synthetic heterogeneous-size household tests match explicit repeated-household arithmetic with missing masks, different weights and shared seed resampling. Tampered deterministic draw files are rejected. No neural inference or fitting occurs in bootstrap. These are implementation checks, not population intervals.

If usable data existed, intervals would describe household variability conditional on the fitted systems and benchmark cohort under a bootstrap approximation. They would not be official ACS survey-design intervals, exact guarantees, privacy certificates or new training-cohort replication. Favorable bands could not override failed source/utility matching, individual race regressions or missing category support. Here even that conditional assessment is unavailable.
''')
    write(out/'TRADEOFF.md','''# Residence versus coalition SEX figure: unavailable

No observed new-sample points exist for any of the six systems, either weighting or seed. Therefore no tradeoff figure is drawn. In particular, absolute H disclosure cannot be estimated on an empty cohort. Drawing historical points here would mislabel development evidence as untouched-household performance; interpolating a frontier would invent unobserved outcomes. This omission is an explicit consequence of the verified provenance blocker, not a missing rendering step.
''')
    print(json.dumps(completion,indent=2))

if __name__=='__main__':run()
