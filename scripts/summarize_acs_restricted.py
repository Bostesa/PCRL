"""Read-only restricted-input comparisons; native heads stay separate from probes."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import summarize_acs_selective as shared
from scripts.summarize_acs_protection import *
from scripts.summarize_acs_pca16_init import utility_criteria

ROOT=Path(__file__).resolve().parents[1]
PREFIXES=tuple(f'{t}_{a}' for t in ('E','S') for a in ('F','K'))
FINAL=tuple(p+'_'+a for p in PREFIXES for a in ('C','D'))
BANKS=('E_bank','S_bank')
DIRECT=('E_direct','S_direct')
STAGES=tuple(p+'_W' for p in PREFIXES)
HISTORICAL=('E_pca','PCA16','E_pca_leace','B_rich_bank','C_tree_bank','C_init','D_init',
            'E_rho0_C','E_rho0_D','S_rho0_C','S_rho0_D','R_rho0p1_C','R_rho0p1_D')
RELEASES=(*DIRECT,*FINAL,*BANKS,*HISTORICAL,*STAGES)
LABELS={**shared.LABELS,**{r:r.replace('_',' ') for r in (*FINAL,*BANKS,*STAGES)}}

def table(headers,rows): return '\n'.join(shared.table(headers,rows))

def alias(prefix,name): return prefix if name=='B' else prefix+'_'+name

def load_evidence(out):
    cfg=read_json(out/'config.json')
    evidence,_=shared.load_evidence(ROOT/cfg['selective_reference_results'])
    evidence.units=[]
    evidence.inputs.add(out/'config.json')
    fresh=shared.Evidence()
    for teacher,access,seed in cfg['execution_order']:
        prefix=f'{teacher}_{access}';directory=out/prefix/f'seed_{seed}';path=directory/'metrics.json'
        if not path.exists():continue
        record=evidence.read(path);selection=evidence.read(directory/'selection_before_test.json')
        for raw in record['raw_metrics']:
            budgets=(raw['audit_budget'],) if raw['role']=='audit' else (120,360)
            for budget in budgets:
                meta=candidate_metadata(selection['budgets'][str(budget)] if raw['role']=='audit' else selection,raw)
                for destination in (evidence,fresh):
                    destination.add(raw,meta,budget,path,lambda n:alias(prefix,n),reused=False)
        evidence.units.append({'unit':prefix,'seed':seed,'metrics':str(path.relative_to(out)),'runtime':record['runtime'],'integrity':record['integrity']})
    return evidence,fresh,cfg

def comparisons():
    pairs=[]
    for t in ('E','S'):
        for arm in ('C','D'):
            pairs.append(('K_minus_F',f'{t}_K_{arm}',f'{t}_F_{arm}'))
        for access in ('F','K'):
            prefix=f'{t}_{access}'
            pairs.extend([('D_minus_C',prefix+'_D',prefix+'_C'),('W_minus_teacher_reference',prefix+'_W',t+'_direct')])
            for arm in ('C','D'):
                name=prefix+'_'+arm
                pairs.extend([('final_minus_W',name,prefix+'_W'),('representation_minus_source_bank',name,t+'_bank'),
                              ('new_minus_beta0',name,arm+'_init'),('new_minus_historical_selective',name,t+'_rho0_'+arm)])
                if access=='K':pairs.append(('K_minus_direct_teacher',name,t+'_direct'))
        pairs.append(('source_bank_minus_direct_teacher',t+'_bank',t+'_direct'))
    for access in ('F','K'):
        for arm in ('C','D'):pairs.append(('E_minus_S',f'E_{access}_{arm}',f'S_{access}_{arm}'))
    pairs.append(('E_minus_S_bank','E_bank','S_bank'))
    return pairs

def paired(seeds,indices):
    rows=[]
    for seed in seeds:
        for name,left,right in comparisons():
            for role,targets in (('transfer',UTILITY_TASKS),('audit',ATTRIBUTES)):
                if role=='audit' and (left in STAGES or right in STAGES):continue
                for budget in ((120,) if role=='transfer' else (120,360)):
                    for selector in (('primary',) if role=='transfer' else ('primary','pooled')):
                        for target in targets:
                            for raw_split,split in SPLITS.items():
                                a=indices[budget].get((seed,role,left,target,selector));b=indices[budget].get((seed,role,right,target,selector))
                                av=a.get(raw_split,{}).get('log_loss') if a else None;bv=b.get(raw_split,{}).get('log_loss') if b else None
                                rows.append(dict(seed=seed,comparison=name,left=left,right=right,role=role,target=target,
                                    audit_budget=budget if role=='audit' else None,selector=selector,split=split,left_value=av,right_value=bv,
                                    left_minus_right=av-bv if finite(av) and finite(bv) else None,
                                    signed_gain_left_minus_right=bv-av if role=='audit' and finite(av) and finite(bv) else None))
    return rows

def aggregates(seeds,indices):
    rows=[]
    for release in RELEASES:
        for role,targets in (('transfer',UTILITY_TASKS),('audit',ATTRIBUTES)):
            if role=='audit' and release in STAGES:continue
            for budget in ((120,) if role=='transfer' else (120,360)):
                for selector in (('primary',) if role=='transfer' else ('primary','pooled')):
                    for target in targets:
                        for raw_split,split in SPLITS.items():
                            values=[get_value(indices[budget],s,role,release,target,raw_split,selector=selector) for s in seeds]
                            common=dict(release=release,role=role,target=target,audit_budget=budget if role=='audit' else None,selector=selector,split=split)
                            rows.append({**common,'metric':'log_loss',**statistics(values)})
                            if role=='audit':
                                priors=[get_value(indices[budget],s,role,'prior',target,raw_split,selector='primary') for s in seeds]
                                rows.append({**common,'metric':'signed_prior_relative_gain',**statistics(p-v if finite(p) and finite(v) else None for p,v in zip(priors,values))})
    return rows

def criteria(seeds,indices,selections):
    utility_rows,policies,banks,coverage=[],[],[],[]
    for seed in seeds:
        for raw_split,split in SPLITS.items():
            util=lambda r:{t:get_value(indices[120],seed,'transfer',r,t,raw_split,selector='primary') for t in UTILITY_TASKS}
            bank_res={b:util(b)['same_residence'] for b in ('B_rich_bank','C_tree_bank')}
            for release in RELEASES:
                utility_rows.append({'seed':seed,'split':split,'release':release,**utility_criteria(util(release),util('E_pca'),bank_res)})
            for budget in (120,360):
                for selector in ('primary','pooled'):
                    idx=indices[budget]
                    attack=lambda r:{t:get_value(idx,seed,'audit',r,t,raw_split,selector=selector) for t in ATTRIBUTES}
                    cov=lambda r,t:audit_coverage(idx,selections[budget][seed],seed,r,t,raw_split,selector)
                    for release in (*FINAL,*BANKS,*DIRECT):
                        common=dict(seed=seed,split=split,release=release,audit_budget=budget,audit_selector=selector)
                        attr_cov={t:combined_coverage(cov(release,t),cov('E_pca',t)) for t in ATTRIBUTES}
                        policies.append({**common,'fixed_parent':'E_pca','parent_actual_audit_budget':120,
                            **parent_comparison(util('E_pca'),util(release),bank_res,attack('E_pca'),attack(release),attack('prior'),attr_cov)})
                        coverage.extend({**common,'target':t,**cov(release,t)} for t in ATTRIBUTES)
                        if release not in FINAL:continue
                        for bank in ('B_rich_bank','C_tree_bank',release[0]+'_bank'):
                            comp=feature_bank_comparison(util(release)['same_residence'],util(bank)['same_residence'],attack(release),attack(bank),attack('prior'),
                                {t:combined_coverage(cov(release,t),cov(bank,t)) for t in ATTRIBUTES})
                            banks.append({**common,'bank':bank,'numeric_joint_inequalities':joint_status([comp['utility']['pass'],*[v['numeric_inequality'] for v in comp['attributes'].values()]]),**comp})
    return utility_rows,policies,banks,coverage

def native(out,units,evidence):
    rows=[];exposure=[]
    for unit in units:
        directory=out/unit['unit']/f"seed_{unit['seed']}"
        record=evidence.read(directory/'native_source.json');training=evidence.read(directory/'training/training.json')
        exposure.append({'unit':unit['unit'],'seed':unit['seed'],'source_fit_coverage':training['source_fit_coverage'],
            'native_exposure':training['native_source_head_training_exposure'],'source_label_hashes':training['source_label_hashes'],
            'schedules':training['schedules'],'bank':training['bank'],'common_observer_exposure':training['common_adversary_row_exposures'],'observer_exposure_by_arm':{k:{'continuation':v['adversary_row_exposures'],'total':training['common_adversary_row_exposures']+v['adversary_row_exposures']} for k,v in training['arms'].items()},
            'initial_head_hash':training['initialization']['heads_initial_sha256']})
        for raw in record['raw_metrics']:
            for weighted,key in ((False,'score'),(True,'person_weighted')):
                rows.append({'seed':raw['seed'],'release':alias(unit['unit'],raw['release']),'target':raw['target'],
                    'split':('development_evaluation' if raw['split']=='test' else 'source_validation')+('_person_weighted' if weighted else ''),
                    **raw[key],'probability_sha256':raw['probability_sha256']})
    grouped=defaultdict(dict)
    for row in rows:grouped[row['release'],row['target'],row['split']][row['seed']]=row['log_loss']
    ag=[dict(zip(('release','target','split'),key),**statistics(values.get(s) for s in (0,1,2))) for key,values in grouped.items()]
    index={(r['seed'],r['release'],r['target'],r['split']):r['log_loss'] for r in rows};pairs=[]
    for t in ('E','S'):
        for access in ('F','K'):
            for arm in ('C','D'):
                for target in SOURCE_TASKS:
                    for split in ('source_validation','source_validation_person_weighted','development_evaluation','development_evaluation_person_weighted'):
                        for seed in (0,1,2):
                            left,right=f'{t}_{access}_{arm}',t+'_bank';a=index.get((seed,left,target,split));b=index.get((seed,right,target,split))
                            pairs.append(dict(seed=seed,comparison='native_main_minus_equal_exposure_bank',left=left,right=right,target=target,split=split,
                                left_minus_right=a-b if finite(a) and finite(b) else None,left_value=a,right_value=b))
    save_csv(out/'NATIVE_SOURCE.csv',rows);save_csv(out/'NATIVE_AGGREGATE.csv',ag);save_csv(out/'NATIVE_PAIRED.csv',pairs)
    save_csv(out/'NATIVE_PAIRED_AGGREGATE.csv',shared.aggregate_rows(pairs,('comparison','left','right','target','split')))
    save_json(out/'SOURCE_EXPOSURE.json',exposure)
    return ag,exposure

def geometry(out,evidence):
    # Mature fixed decoder exports, with I names corrected to each actual teacher/input.
    result=shared.mechanism_rows(out,evidence)
    scores=result['scores']
    for rows in (scores,result['direct'],result['gradients']):
        for row in rows:
            if row.get('release')=='I':row['release']=row['unit']+'_I'
    coordinate_path=out/'MECHANISM_COORDINATES.csv'
    with coordinate_path.open(newline='') as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;coordinate_rows=list(reader)
    for row in coordinate_rows:
        if row['release']=='I':row['release']=row['unit']+'_I'
    with coordinate_path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(coordinate_rows)
    cfg=read_json(out/'config.json')
    for seed in cfg['seeds']:
        origin=ROOT/cfg['selective_reference_results']/'static'/f'seed_{seed}'/'geometry.json'
        record=evidence.read(origin)
        for release,snapshot in record['snapshots'].items():
            for target,meta in snapshot['targets'].items():
                for split in ('fit','source_validation','development_evaluation'):
                    if split not in meta:continue
                    scores.append({'unit':'historical_static','seed':seed,'release':release,'teacher':snapshot['teacher'],'target':target,'split':split,
                        'origin_geometry':str(origin.relative_to(ROOT)),'reused_reference':True,
                        **{k:v for k,v in meta.items() if k not in ('fit','source_validation','development_evaluation')},
                        **{k:v for k,v in meta[split].items() if not isinstance(v,list)}})
    bank_path=out/'bank_geometry/geometry.json'
    if bank_path.exists():
        bank_record=evidence.read(bank_path)
        for snapshot in bank_record['snapshots'].values():
            for target,meta in snapshot['targets'].items():
                for split in ('fit','source_validation','development_evaluation'):
                    if split not in meta:continue
                    scores.append({'unit':snapshot['reported_release'],'seed':snapshot['seed'],'release':snapshot['reported_release'],
                        'teacher':snapshot['teacher'],'target':target,'split':split,'input_dimension':3,
                        'origin_geometry':str(bank_path.relative_to(ROOT)),**{k:v for k,v in meta.items() if k not in ('fit','source_validation','development_evaluation')},
                        **{k:v for k,v in meta[split].items() if not isinstance(v,list)}})
    save_csv(out/'MECHANISM.csv',scores);save_csv(out/'DIRECT_MATCHING.csv',result['direct'])
    return result

def budget_rows(indices,evidence):
    rows=[];caught=[]
    for seed in (0,1,2):
        for release in (*FINAL,*BANKS):
            for target in ATTRIBUTES:
                for raw_split,split in SPLITS.items():
                    for selector in ('primary','pooled',*('candidate:'+c for c in ('logistic','mlp_0','mlp_1','hist_gb_20','hist_gb_5','catchup','saved_adversary'))):
                        a=indices[120].get((seed,'audit',release,target,selector));b=indices[360].get((seed,'audit',release,target,selector))
                        av=a.get(raw_split,{}).get('log_loss') if a else None;bv=b.get(raw_split,{}).get('log_loss') if b else None
                        rows.append(dict(seed=seed,release=release,target=target,split=split,selector=selector,loss120=av,loss360=bv,
                            difference=bv-av if finite(av) and finite(bv) else None,candidate120=a['candidate_id'] if a else None,candidate360=b['candidate_id'] if b else None))
                    for budget in (120,360):
                        r={'seed':seed,'release':release,'target':target,'split':split,'audit_budget':budget}
                        for label,selector in [('independent','primary'),('pooled','pooled'),('catchup','candidate:catchup'),('saved','candidate:saved_adversary')]:
                            row=indices[budget].get((seed,'audit',release,target,selector));r[label+'_loss']=row.get(raw_split,{}).get('log_loss') if row else None
                            r[label+'_candidate']=row['candidate_id'] if row else None
                            meta=evidence.meta[budget].get((seed,'audit',release,target,row['candidate_id']),{}) if row else {}
                            r[label+'_epoch']=meta.get('selected_epoch')
                        caught.append(r)
    return rows,caught

def tables(out,ag,native_ag,utility,units):
    def value(release,target,split,role='transfer',selector='primary',metric='log_loss'):
        rows=[r for r in ag if r['release']==release and r['target']==target and r['split']==split and r['role']==role and r['selector']==selector and r['metric']==metric and (role!='audit' or r['audit_budget']==360)]
        return shared.mean_sd(rows[0]) if rows else 'unavailable'
    lines=['# Downstream probes and native source heads','',
        'DEVELOPMENT EVALUATION on the same cohort. Log losses and signed gains are nats; lower task loss and lower prior-relative attack gain are better. Cells show mean ± sample SD across three seeds. Unweighted selection is reused for PWGTP. [Every selected seed/task/scope](PER_SEED.csv), [every new candidate](PER_TARGET.csv), [every class](PER_CLASS.csv), [paired contrasts](PAIRED.csv), [native contrasts](NATIVE_PAIRED.csv).','',
        'F receives teacher16 plus rawPCA32; K receives teacher16 only; C has no protection gradient; D uses real-attribute adversarial protection. Banks publish only three source probabilities. Main preservation is1, reconstruction0. Static teachers/banks have independent audits only. Pooled means independent plus own observer catch-up for main releases; inherited exposure differs. Historical references keep their original source identities.','']
    for split in ('development_evaluation','development_evaluation_person_weighted'):
        lines += ['## '+split.replace('_',' '),'','New main releases, banks and direct teachers share the independent360 audit recipe. Original PCA32/PCA32+LEACE retain their historical120 audits; other named rows shown have compatible360 evidence.','']
        rows=[[LABELS.get(r,r),*[value(r,t,split) for t in UTILITY_TASKS],*[value(r,t,split,'audit',metric='signed_prior_relative_gain') for t in ATTRIBUTES]] for r in (*DIRECT,*FINAL,*BANKS,*HISTORICAL)]
        lines += [table(['Release','Residence','Commute','Income','Civilian at work','Public coverage','SEX gain, independent','Race gain, independent'],rows),'',
            'Catch-up-inclusive decisions (no catch-up available for direct teachers/banks):','',
            table(['Release','SEX gain, pooled','Race gain, pooled','All3 source margins pass /3'],[[r,*[value(r,t,split,'audit','pooled','signed_prior_relative_gain') for t in ATTRIBUTES],
                sum(x['source_preservation']['pass'] is True for x in utility if x['release']==r and x['split']==split)] for r in (*FINAL,*BANKS,*DIRECT)]),'']
        nrows=[]
        for release in (*FINAL,*BANKS):
            cells=[]
            for target in SOURCE_TASKS:
                item=next((r for r in native_ag if r['release']==release and r['target']==target and r['split']==split),None)
                cells.append(shared.mean_sd(item) if item else 'unavailable')
            nrows.append([release,*cells])
        lines += ['### Native source heads: no fitted downstream probe','',table(['Release','Income','Civilian at work','Public coverage'],nrows),'']
    lines += ['## Stage utility','', 'I is the numerical teacher-identity starting function; direct-teacher scores are reused references, not a claim of bitwise old I predictions or a new I audit. W has no protection audit.','']
    for split in ('development_evaluation','development_evaluation_person_weighted'):
        lines += [split.replace('_',' '),'',table(['Stage',*UTILITY_TASKS],[[r,*[value(r,t,split) for t in UTILITY_TASKS]] for r in (*DIRECT,*STAGES)]),'']
    lines += ['Original PCA32 remains the parent for every descriptive margin. Its original audit budget is120, explicitly distinct from new360 audits; rawPCA16 and unprotected rich banks have compatible360 evidence. RAC1P code4 is absent from attacker fitting/validation for every seed; the full nine-class protection assessment remains unassessable. Neither a small signed gain nor an unsuccessful finite attack proves privacy.']
    (out/'TABLE.md').write_text('\n'.join(lines)+'\n')

def plots(out,ag,geo,native_ag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    def val(release,target,split='development_evaluation',role='transfer',selector='primary',metric='log_loss'):
        rows=[r for r in ag if r['release']==release and r['target']==target and r['split']==split and r['role']==role and r['selector']==selector and r['metric']==metric and (role!='audit' or r['audit_budget']==360)]
        return rows[0]['mean'] if rows and rows[0]['mean'] is not None else np.nan
    for target in ATTRIBUTES:
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        for i,weight in enumerate(('', '_person_weighted')):
            for j,scope in enumerate(('primary','pooled')):
                ax=axes[i,j]
                for r in (*DIRECT,*FINAL,*BANKS,'PCA16','E_pca','B_rich_bank','C_tree_bank'):
                    x=val(r,'same_residence','development_evaluation'+weight);y=val(r,target,'development_evaluation'+weight,'audit',scope,'signed_prior_relative_gain')
                    color='tab:blue' if r in (*FINAL,*BANKS,*DIRECT) and r.startswith('E_') else 'tab:orange' if r in (*FINAL,*BANKS,*DIRECT) else 'gray'
                    marker='D' if r in DIRECT else '^' if r in BANKS else 's' if '_K_' in r else 'o' if r in FINAL else {'PCA16':'*','E_pca':'X','B_rich_bank':'P','C_tree_bank':'v'}[r]
                    hollow=r in FINAL and r.endswith('_D')
                    label=r.replace('_',' ') if r in (*FINAL,*BANKS) else {'E_direct':'E teacher','S_direct':'S teacher','PCA16':'PCA16','E_pca':'PCA32 (audit120)','B_rich_bank':'Rich neural bank','C_tree_bank':'Rich tree bank'}[r]
                    ax.scatter(x,y,marker=marker,s=55 if hollow else 28,facecolors='none' if hollow else color,edgecolors=color,linewidths=1.4,label=label)
                ax.set(xlabel='Residence log loss (lower better)',ylabel=f'{target} prior-relative gain (lower better)',title=('PWGTP' if weight else 'Unweighted')+'; '+('independent360' if j==0 else 'catch-up inclusive for main'))
                ax.grid(alpha=.2)
        fig.suptitle(f'{target}: frozen-release development tradeoffs\nPCA32 historical audit120; teacher/bank catch-up unavailable; all race categories retained')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=8,frameon=False)
        fig.tight_layout(rect=(0,.12,1,.94));fig.savefig(out/f'tradeoff_{target}.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for ax,target in zip(axes.flat,UTILITY_TASKS):
        for prefix in PREFIXES:
            for arm,style in [('C','-'),('D','--')]:
                ax.plot([0,1,2],[val(prefix[0]+'_direct',target),val(prefix+'_W',target),val(prefix+'_'+arm,target)],style,marker='o',label=prefix+' '+arm)
        ax.set(title=target,ylabel='Unweighted log loss, nats');ax.set_xticks([0,1,2],['Teacher reference','W','Final']);ax.grid(alpha=.2)
    axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',fontsize=8)
    fig.suptitle('Development task utility at fixed stages; C/D continuations are parallel')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'utility_stages.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.6))
    for ax,target in zip(axes,('raw','E','qE')):
        for prefix in PREFIXES:
            for arm,style in [('C','-'),('D','--')]:
                values=[]
                for stage in ('I','W',arm):
                    rr=[r for r in geo['scores'] if r['unit']==prefix and r['release']==prefix+'_'+stage and r['target']==target and r['split']=='development_evaluation']
                    values.append(statistics(r['mean_mse'] for r in rr)['mean'])
                ax.plot([0,1,2],values,style,marker='o',label=prefix+' '+arm)
        ax.set(title=target,ylabel='Original-scale standardized MSE');ax.set_xticks([0,1,2],['I','W','Final']);ax.grid(alpha=.2)
    axes[-1].legend(fontsize=6);fig.suptitle('Affine recovery after freezing: qE is not pure protected information')
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(out/'retained_removed_structure.png',dpi=170);plt.close(fig)

def native_plot(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),sharex=True)
    releases=(*FINAL,*BANKS)
    for i,split in enumerate(('development_evaluation','development_evaluation_person_weighted')):
        for j,target in enumerate(SOURCE_TASKS):
            ax=axes[i,j]
            for k,release in enumerate(releases):
                row=next((r for r in rows if r['release']==release and r['target']==target and r['split']==split),None)
                if row is not None and row['mean'] is not None:
                    ax.errorbar(k,row['mean'],yerr=row['sample_sd'],fmt='^' if release in BANKS else 'o',color='tab:blue' if release.startswith('E_') else 'tab:orange',capsize=2)
            ax.set(title=target,ylabel=('PWGTP' if i else 'Unweighted')+' native log loss, nats')
            ax.set_xticks(range(len(releases)),[r.replace('_',' ') for r in releases],rotation=65,fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Fixed native source heads: matched140-epoch source exposure; no downstream probes\nBars are descriptive sample SD across three seeds, not population uncertainty')
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(out/'native_source_controls.png',dpi=170);plt.close(fig)


def summarize(out):
    out=Path(out).resolve();evidence,fresh,cfg=load_evidence(out);seeds=cfg['seeds']
    indices={b:shared.index_records(evidence.records(b)) for b in (120,360)};sel={b:evidence.selections(b) for b in (120,360)}
    exports=shared.export_all(out,fresh);ag=aggregates(seeds,indices);pairs=paired(seeds,indices)
    per_seed=[]
    for release in RELEASES:
        for seed in seeds:
            for role,targets in (('transfer',UTILITY_TASKS),('audit',ATTRIBUTES)):
                if role=='audit' and release in STAGES:continue
                for selector in (('primary',) if role=='transfer' else ('primary','pooled')):
                    for target in targets:
                        for raw_split,split in SPLITS.items():
                            row=indices[360].get((seed,role,release,target,selector));prior=get_value(indices[360],seed,'audit','prior',target,raw_split,selector='primary') if role=='audit' else None
                            loss=row.get(raw_split,{}).get('log_loss') if row else None
                            per_seed.append(dict(seed=seed,release=release,role=role,target=target,selector=selector,split=split,
                                log_loss=loss,prior_relative_gain=prior-loss if finite(prior) and finite(loss) else None,candidate=row['candidate_id'] if row else None,
                                evidence_audit_budget=row.get('evidence_audit_budget') if row else None))
    save_csv(out/'PER_SEED.csv',per_seed)
    pagg=shared.aggregate_rows(pairs,('comparison','left','right','role','target','audit_budget','selector','split'))
    util,policies,banks,coverage=criteria(seeds,indices,sel)
    nag,exposure=native(out,evidence.units,evidence);geo=geometry(out,evidence);budget,caught=budget_rows(indices,evidence)
    for name,rows in [('AGGREGATE.csv',ag),('PAIRED.csv',pairs),('PAIRED_AGGREGATE.csv',pagg),('SUPPORT.csv',coverage),('AUDIT_BUDGET.csv',budget),('CATCHUP.csv',caught)]:save_csv(out/name,rows)
    save_json(out/'criteria.json',{'margins':MARGINS,'fixed_parent':'E_pca','parent_audit_budget':120,'per_seed_split_utility':util,'per_seed_split_budget_policy':policies})
    save_json(out/'bank_comparisons.json',{'margins':MARGINS,'per_seed_split_budget':banks})
    tables(out,ag,nag,util,evidence.units);plots(out,ag,geo,nag);native_plot(out,nag)
    result={'evaluation_status':cfg['evaluation_status'],'completed_units':evidence.units,'complete':len(evidence.units)==18,
        'new_final_models':sum(1 if u['unit'].endswith('bank') else 2 for u in evidence.units),'expected_final_models':30,
        'aggregate_metrics':ag,'paired_aggregate_metrics':pagg,'native_aggregate_metrics':nag,
        'selected_audit_comparisons_changed':sum(r['difference']!=0 for r in budget if r['selector'] in ('primary','pooled') and r['difference'] is not None),
        'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(evidence.inputs)},
        'report_source_sha256':sha(Path(__file__)),'row_counts':dict(zip(('scores','classes','fits','curves'),map(len,exports)))}
    save_json(out/'summary.json',result);print({k:result[k] for k in ('complete','new_final_models','row_counts','selected_audit_comparisons_changed')});return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path);summarize(parser.parse_args().out)
