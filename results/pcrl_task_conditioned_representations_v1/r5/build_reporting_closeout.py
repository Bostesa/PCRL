"""Format completed public V2 aggregates only; no scientific recomputation."""
import csv,hashlib,json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ARMS=['U0','U_NEW','U_TASKPOWER','LOCAL_TWIN','LOCAL_CONDITIONAL','JOINT_CONDITIONAL','JOINT_UNCONDITIONAL','RAW_JOINT','SEQ12','SEQ21','H_RAW_CSTAR']
TASKS=['income','occupation']
METRICS=['accuracy','nll','brier']

def read(name):return json.loads((ROOT/name).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def interval_cell(a):
    c=a['interval'];return f"{c['point']:.6f} [{c['lower']:.6f}, {c['upper']:.6f}]"
def count_cell(values):
    c=Counter(e['status'] for e in values)
    return f"P{c['PASS']}/U{c['UNRESOLVED']}/V{c['VIOLATION']}/M{c['MISSING_DEPENDENCY']}"
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def pair_mean(records,task,key):return (records[0]['tasks'][task]['metrics'][key]+records[1]['tasks'][task]['metrics'][key])/2

def main():
    master=read('FINAL_PAIRED_INFERENCE_V2.json');utility=read('R5_ORIGINAL_UTILITY_ALL40_V2.json');legacy=read('LEGACY_ORIGINAL_UTILITY_V2.json')
    assert master['schema'].endswith('v2') and utility['schema'].endswith('v2') and legacy['schema'].endswith('v2')
    assert master['K']=={'primary':51,'utility':240,'secondary':33}
    means={(r['arm_id'],r['contract'],r['view']):r for r in master['absolute_recovery'] if r.get('aggregation')=='paired_seed_mean'}
    legacy_contract='legacy_complete_score_wire_codes_probabilities_decisions_metadata'
    records={a:sorted([r for r in utility['records'] if r['arm_id']==a],key=lambda r:r['seed']) for a in ARMS}
    full_rows=[]
    for arm in ARMS:
        r=records[arm];point=utility['family_eligibility'][arm];ends=[e for e in master['families']['utility'] if e['arm']==arm]
        gate=f"eligible {point['passed']}/24" if point['point_eligible'] else f"diagnostic: {point['passed']} pass/{24-point['passed']} fail"
        full_rows.append([arm,*[f'{pair_mean(r,t,m):.6f}' for t in TASKS for m in METRICS],*[interval_cell(means[(arm,'full',v)]) for v in ('r1','r2','pair')],gate,count_cell(ends) if ends else 'outside K240'])
    score_rows=[]
    for arm in ARMS+['P0','S12']:
        is_old=arm in ('P0','S12');contract=legacy_contract if is_old else 'output';point=(legacy if is_old else utility)['family_eligibility'][arm]
        r=sorted([r for r in (legacy if is_old else utility)['records'] if r['arm_id']==arm],key=lambda r:r['seed'])
        gate=f"eligible {point['passed']}/24" if point['point_eligible'] else f"diagnostic: {point['passed']} pass/{24-point['passed']} fail"
        score_rows.append([arm,'legacy complete codes + outputs + metadata' if is_old else 'new probabilities + decision',*[f'{pair_mean(r,t,m):.6f}' for t in TASKS for m in METRICS],*[interval_cell(means[(arm,contract,v)]) for v in ('r1','r2','pair')],gate])
    primary=master['families']['primary'];privacy=[e for e in primary if e['kind']=='privacy'];comparators=[]
    for control in ['U0','U_NEW','U_TASKPOWER','LOCAL_TWIN','LOCAL_CONDITIONAL','JOINT_UNCONDITIONAL','RAW_JOINT','SEQ12','SEQ21']:
        es={e['metric']:e for e in privacy if e['comparator']==control};p=es['pair_benefit'];one=es['r1_worsening'];two=es['r2_worsening']
        comparators.append([control,interval_cell(p),p['status'],interval_cell(one),one['status'],interval_cell(two),two['status']])
    primary_utility=[e for e in primary if e['kind']=='utility'];pu=Counter(e['status'] for e in primary_utility)
    component=[]
    for name,control in [('coalition coupling','LOCAL_TWIN'),('task conditioning','JOINT_UNCONDITIONAL'),('kernel versus raw adversary','RAW_JOINT'),('joint versus freeze-first1→2','SEQ12'),('joint versus freeze-first2→1','SEQ21')]:
        e=next(e for e in privacy if e['comparator']==control and e['metric']=='pair_benefit');ci=e['interval']
        effect='positive interval' if ci['lower']>0 else 'adverse interval' if ci['upper']<0 else 'unresolved direction'
        local=[x for x in privacy if x['comparator']==control and x['metric']!='pair_benefit']
        conjunction=ci['lower']>0 and all(x['status']=='PASS' for x in primary_utility+local) and e.get('point_eligible_comparison',False)
        component.append([name,control,interval_cell(e),effect,'supported component conjunction' if conjunction else 'component conjunction not established'])
    reports=[read(f'R5_REPORT_{a}_S{s}_V1.json') for a in ARMS for s in (0,1)]
    legacy_reports=[read(f'R5_LEGACY_REPORT_{a}_S{s}_V1.json') for a in ('P0','S12') for s in (0,1)]
    assert all(r['all_controls_pass'] for r in reports+legacy_reports)
    md=['# R5 corrected reporting closeout','',
        'This is an exploratory method search on repeatedly used Adult people. The V2 replay preserves exact original U0/Ucal references in separate REF_* keys. All V1 artifacts and the reference-collision diagnosis remain preserved. No models, readers, orientations, selections, caps, bootstrap draws or endpoint families changed.', '',
        'Paired means average the two fixed fitted source runs on the same people. Absolute AUC brackets are descriptive pointwise 95% intervals (K=1), computed from the shared 2048 duplicate-group resamples. They are not the simultaneous competitive conjunction. Task metrics are displayed means; point eligibility and every interval requirement remain separate for each task/seed/reference.', '',
        '## Primary full feature + own-output service','',
        table(['Arm','Income acc','Income NLL','Income Brier','Occupation acc','Occupation NLL','Occupation Brier','Income AUC [95%]','Occupation AUC [95%]','Pair AUC [95%]','Original point eligibility','Utility K240'],full_rows),'',
        'Brier is the sum across classes. P/U/V/M counts mean interval PASS/UNRESOLVED/VIOLATION/MISSING. Each primary-pool arm has 24 utility intervals in the separate K240 family. H_RAW_CSTAR has different historical training/search exposure and passes 16 clauses while failing 8: occupation NLL and Brier against both references in both seeds. It fails the original point gate and is descriptive outside the primary pool and K240. Crossing intervals remain unresolved; they did not prevent the point-qualified fresh attack comparison.', '',
        '## Secondary score contracts','',
        table(['Arm','Actual contract','Income acc','Income NLL','Income Brier','Occupation acc','Occupation NLL','Occupation Brier','Income AUC [95%]','Occupation AUC [95%]','Pair AUC [95%]','Original point eligibility'],score_rows),'',
        'Every new arm uses only its own probabilities and decision in this table. P0/S12 retain their literal complete legacy codes, probabilities, decisions and metadata, with freshly fitted legacy readers; these are different-capacity services and cannot establish a matched full-feature mechanism effect. Historical task metrics are exact finite-law expected per-action metrics, not metrics of averaged probabilities. Score comparisons use their own frozen K33 family.', '',
        '## Registered primary privacy contrasts','',
        'Pair benefit is comparator AUC minus JOINT AUC. Each local worsening is JOINT minus comparator. All brackets below are the original K51 simultaneous intervals with df2233; competitive pair target is lower ≥0.020 and local guard is upper ≤0.010. The full conjunction also requires all 24 JOINT original utility bounds, valid controls, independent verification and deployment.', '',
        table(['Comparator','Pair benefit [K51]','≥0.020','Income worsening [K51]','≤0.010','Occupation worsening [K51]','≤0.010'],comparators),'',
        f"JOINT primary utility interval statuses: PASS {pu['PASS']}/24, UNRESOLVED {pu['UNRESOLVED']}/24, VIOLATION {pu['VIOLATION']}/24. Its point gate passes 24/24. These are separate conclusions.", '',
        '## Component quantities without extra multiplicity','',
        table(['Claim','Fixed control','Pair benefit [K51]','Direction','Complete component conjunction'],component),'',
        'Component direction reuses the identical registered privacy quantity and requires a lower bound strictly above zero. A favorable pair interval alone cannot establish a component claim when its required utility or local conjunction remains unresolved. Sequential statements concern these bounded optimization procedures.', '',
        '## Verdict suggestions and limits','',
        'The useful baseline and all ten primary-pool arms pass all 24 original point clauses. U_NEW and U_TASKPOWER selected the zero-update source alias; their selected service adds no empirical mechanism change over U0. JOINT stays point useful, with paired-mean income/occupation accuracies 0.850112/0.493512, but its complete utility interval conjunction remains unresolved (13 PASS, 11 UNRESOLVED in K51). Occupation performance is uneven across classes; each seed has one reporting person in class 5 and zero recall there. The per-seed CSV preserves balanced accuracy, every recall and confidence, so overall utility is not mistaken for uniformly strong class performance.', '',
        'The strongest favorable primary pair comparison is versus the tied U0/U_NEW/U_TASKPOWER baselines: benefit 0.008252 [0.000111,0.016393]. It is positive on this approximate conditional-development interval and remains below the required 0.020 target. Every registered comparator is a SHORTFALL for that pair target. RAW_JOINT is the strongest matched feature-pair reader control (AUC 0.856824 versus JOINT 0.861891). The strongest adverse local comparison is income versus RAW: JOINT worsens AUC by 0.063252 [0.043198,0.083306], violating the 0.010 guard.', '',
        'Conditioning contributes no established pair gain over JOINT_UNCONDITIONAL, coalition coupling contributes no established gain over its exact LOCAL_TWIN, and joint training contributes no established gain over either sequential order: each component interval crosses zero and its complete conjunction is unestablished. The competitive method verdict is negative. RAW should remain the strongest eligible feature-pair control for the next decision; H_RAW_CSTAR has substantially lower observed recovery but fails the original utility gate and cannot be promoted.', '',
        'For the new output-only service, JOINT has the lowest observed pair AUC among the new arms (0.834821), while freshly scored literal P0/S12 have lower pair AUCs 0.802763/0.798701 under their separate legacy contract. These narrower/different interfaces do not establish reusable-feature privacy or a matched full-feature mechanism effect. Conditioning, MMD/block testing and end-to-end training are established ideas; this run supplies no empirical competitive or component result that warrants a new-method novelty claim. All required new families and available historical references were actually scored; an unavailable-reference or incomplete-control explanation is unnecessary.', '',
        'All 26 fresh banks passed their declared null, independently shuffled-label and planted-leak checks. The pair envelopes include every local reader as an ignore-other alias. R6 independently refitted four predeclared strongest ATT-validation reader types for the secondary sensitivity check; this does not mean every reader was independently refitted.', '',
        'Oracle label-only and true-class/pair conditional diagnostics are interpretation-only and were not independently numerically verified in this run. They never supply primary inputs or success criteria. Source initialization/training history is shared, no new confirmation population exists, and passed fixtures or low kernel losses do not establish unconditional privacy.', '',
        'Standalone aggregate files retain all per-seed task/balanced-accuracy/recall/confidence values, complete primary/utility/secondary endpoint intervals, per-seed absolute AUCs and reader-level recovery. Root owns the final scientific decisions, independent comparison, deployment/custody receipts and publication closeout.']
    (ROOT/'REPORTING_CLOSEOUT_V2.md').write_text('\n'.join(md)+'\n')
    all_records=utility['records']+legacy['records'];fieldnames=['arm_id','seed','task','accuracy','nll','brier','balanced_accuracy','confidence','class_order','class_counts','per_class_recall','gain_over_constant_accuracy','gain_over_constant_nll','gain_over_constant_brier','point_eligible_seed','diagnostic_fallback','metric_contract']
    with (ROOT/'REPORTING_CLOSEOUT_PER_SEED_TASKS_V2.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader()
        for r in all_records:
            for task in TASKS:
                m=r['tasks'][task]['metrics'];g=m.get('gain_over_constant',{})
                row={k:m.get(k,'') for k in ('accuracy','nll','brier','balanced_accuracy','confidence')}
                row.update(arm_id=r['arm_id'],seed=r['seed'],task=task,class_order=json.dumps(m.get('class_order',[])),class_counts=json.dumps(m.get('class_counts',[])),per_class_recall=json.dumps(m.get('per_class_recall',[])),gain_over_constant_accuracy=g.get('accuracy',''),gain_over_constant_nll=g.get('nll',''),gain_over_constant_brier=g.get('brier',''),point_eligible_seed=r['point_eligible_seed'],diagnostic_fallback=r['arm_id'].endswith('_DIAGNOSTIC_FALLBACK'),metric_contract='exact stochastic finite-law per-action expectation' if r['arm_id'] in ('P0','S12') else 'deterministic own-head probabilities/decision')
                w.writerow(row)
    with (ROOT/'REPORTING_CLOSEOUT_ENDPOINTS_V2.csv').open('w',newline='') as f:
        names=['family','id','kind','arm','seed','task','reference','comparator','metric','point','lower','upper','bootstrap_se','K','cap','target','status','point_eligible_comparison','diagnostic_only'];w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for family,entries in master['families'].items():
            for e in entries:
                row={k:e.get(k,'') for k in names};row['family']=family;c=e.get('interval') or {}
                row.update({k:c.get(k,'') for k in ('point','lower','upper','bootstrap_se','K')});w.writerow(row)
    with (ROOT/'REPORTING_CLOSEOUT_ABSOLUTE_AUC_V2.csv').open('w',newline='') as f:
        names=['arm_id','seed','aggregation','contract','view','point','lower','upper','bootstrap_se','K','interval_role'];w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for e in master['absolute_recovery']:
            row={k:e.get(k,'') for k in names};row.update({k:e['interval'][k] for k in ('point','lower','upper','bootstrap_se','K')});w.writerow(row)
    with (ROOT/'REPORTING_CLOSEOUT_READERS_V2.csv').open('w',newline='') as f:
        names=['arm_id','seed','contract','view','reader','auc','controls_pass','oracle_diagnostics_independently_verified'];w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for r in reports+legacy_reports:
            contract='legacy_complete_score_wire' if r['arm_id'] in ('P0','S12') else 'new_full_feature_and_output_envelope'
            for view,values in r['views'].items():
                for reader,value in values['reader_auc'].items():w.writerow({'arm_id':r['arm_id'],'seed':r['seed'],'contract':contract,'view':view,'reader':reader,'auc':value,'controls_pass':values['controls_pass'],'oracle_diagnostics_independently_verified':False})
    r5root=ROOT.parent;source=[ROOT/'FINAL_PAIRED_INFERENCE_V2.json',ROOT/'R5_ORIGINAL_UTILITY_ALL40_V2.json',ROOT/'LEGACY_ORIGINAL_UTILITY_V2.json',*[ROOT/f'R5_REPORT_{a}_S{s}_V1.json' for a in ARMS for s in (0,1)],*[ROOT/f'R5_LEGACY_REPORT_{a}_S{s}_V1.json' for a in ('P0','S12') for s in (0,1)]]
    out={'schema':'TCR1-R5-aggregate-reporting-closeout-v2','source_sha256':{str(p.relative_to(r5root)):sha(p) for p in source},'display_only_paired_task_means':True,'no_new_fit_attack_bootstrap_or_interval_computation':True,'primary_slots':len(primary),'utility_slots':len(master['families']['utility']),'secondary_slots':len(master['families']['secondary']),'fresh_banks_controls_passed':26,'all_new_PRIMARY_nominees_point_eligible':all(utility['family_eligibility'][a]['point_eligible'] for a in ARMS if a!='H_RAW_CSTAR'),'historical_H_point_eligible':utility['family_eligibility']['H_RAW_CSTAR']['point_eligible'],'legacy_point_eligible':legacy['family_eligibility'],'JOINT_primary_utility_interval_status':dict(pu),'competitive_primary_interval_conjunction_established':all(e['status']=='PASS' and e.get('point_eligible_comparison',False) for e in primary),'oracle_diagnostics_independently_verified':False,'independent_final_comparison':'R6/root supplies final receipt; this formatter makes no independent-verification claim'}
    (ROOT/'REPORTING_CLOSEOUT_MANIFEST_V2.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    old_master=read('FINAL_PAIRED_INFERENCE_V1.json');old_utility=read('R5_ORIGINAL_UTILITY_ALL40_V1.json');old_legacy=read('LEGACY_ORIGINAL_UTILITY_V1.json')
    status_changes={};deltas={}
    for family,entries in master['families'].items():
        old={e['id']:e for e in old_master['families'][family]};changed=[];maximum={k:0. for k in ('point','lower','upper','bootstrap_se')}
        for e in entries:
            previous=old[e['id']]
            if e['status']!=previous['status']:changed.append(e['id'])
            for k in maximum:
                if e.get('interval') and previous.get('interval'):maximum[k]=max(maximum[k],abs(e['interval'][k]-previous['interval'][k]))
        status_changes[family]=changed;deltas[family]=maximum
    def scalar_compare(old,new):
        previous={(r['arm_id'],r['seed']):r for r in old['records']};maximum=0.;changed=[]
        for r in new['records']:
            before=previous[(r['arm_id'],r['seed'])]
            if r['point_eligible_seed']!=before['point_eligible_seed'] or r['clauses_passed']!=before['clauses_passed']:changed.append([r['arm_id'],r['seed']])
            for a,b in zip(r['clauses'],before['clauses']):
                assert (a['task'],a['reference'],a['metric'])==(b['task'],b['reference'],b['metric'])
                maximum=max(maximum,abs(a['value']-b['value']))
                if a['pass']!=b['pass']:changed.append([r['arm_id'],r['seed'],a['task'],a['reference'],a['metric']])
        return {'family_eligibility_identical':old['family_eligibility']==new['family_eligibility'],'scalar_gate_changes':changed,'maximum_absolute_clause_value_change':maximum}
    diagnostic={'schema':'TCR1-actual-V1-V2-aggregate-reference-diagnostic','original_utility':scalar_compare(old_utility,utility),'legacy_utility':scalar_compare(old_legacy,legacy),'endpoint_status_changes':status_changes,'maximum_absolute_interval_changes':deltas,
                'absolute_recovery_records_exactly_identical':old_master['absolute_recovery']==master['absolute_recovery'],'bootstrap_weight_archive_sha256_identical':old_master['bootstrap_weight_sha256']==master['bootstrap_weight_sha256'],
                'source_manifest_sha256_identical':old_utility['source_manifest_sha256']==utility['source_manifest_sha256'],'nomination_lock_sha256_identical':old_utility['nomination_lock_sha256']==utility['nomination_lock_sha256'],
                'nominee_checkpoint_associations_identical':{(r['arm_id'],r['seed']):r['checkpoint_sha256'] for r in old_utility['records']}=={(r['arm_id'],r['seed']):r['checkpoint_sha256'] for r in utility['records']},
                'endpoint_registry_sha256_identical':old_master['endpoint_registry_sha256']==master['endpoint_registry_sha256'],'K_identical':old_master['K']==master['K'],
                'utility_archive_sha256_V1':old_utility['utility_input_sha256'],'utility_archive_sha256_V2':utility['utility_input_sha256'],'utility_archive_bytes_changed':old_utility['utility_input_sha256']!=utility['utility_input_sha256'],
                'byte_provenance_scope':'archive SHA change and unchanged original source/checkpoint pins are copied from aggregate receipts; R6 owns independent original-reference array-byte equality verification','no_person_arrays_read_by_formatter':True}
    (ROOT/'REPORTING_CLOSEOUT_REFERENCE_DIAGNOSTIC_V2.json').write_text(json.dumps(diagnostic,indent=2,sort_keys=True)+'\n')
    with (ROOT/'REPORTING_CLOSEOUT_V2.md').open('a') as f:
        f.write('\n## Actual V1/V2 correction comparison\n\n')
        f.write(f"Original scalar utility gate changes: {len(diagnostic['original_utility']['scalar_gate_changes'])}; legacy scalar gate changes: {len(diagnostic['legacy_utility']['scalar_gate_changes'])}; endpoint status changes: {sum(len(v) for v in status_changes.values())}. Maximum original scalar clause change: {diagnostic['original_utility']['maximum_absolute_clause_value_change']:.17g}; maximum legacy clause change: {diagnostic['legacy_utility']['maximum_absolute_clause_value_change']:.17g}. Absolute recovery records are exactly identical: {diagnostic['absolute_recovery_records_exactly_identical']}. Original source, nomination and checkpoint associations remain identical. The archive SHA changed because references now have separate keys and preserve the original reference values. R6 supplies the independent array-byte and final interval comparison receipts.\n\n")
        f.write(table(['Family','Max point change','Max lower change','Max upper change','Max SE change','Status changes'],[[k,*[f'{v[m]:.17g}' for m in ('point','lower','upper','bootstrap_se')],len(status_changes[k])] for k,v in deltas.items()])+'\n')
    print('Formatted V2 aggregate closeout; no scientific recomputation.')

if __name__=='__main__':main()
