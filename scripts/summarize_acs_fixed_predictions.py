"""Readable post-freeze summaries; numerical comparisons come from frozen reporter."""
from __future__ import annotations
import argparse,csv,json,itertools,statistics,time
from pathlib import Path
import numpy as np
from scripts.report_acs_fixed_predictions import report,read,CONDITIONS,SCOPES,csvout,ROOT
NAMES={'H':'H','E':'H+E_A','A0':'H+A0','L025':'H+L025','L20':'H+L20','J':'H+J'}
TASKS=('income_binary','civilian_at_work','public_coverage','same_residence','commute_over20')
ROLES={'A':('public_coverage','commute_over20','SEX','RAC1P'),'B':('income_binary','civilian_at_work','same_residence','SEX','RAC1P'),'AB':('SEX','RAC1P')}
def md(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(str(x) for x in r)+' |' for r in rows]])
def cell(values):return f'{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}'
def summarize(out):
    started=time.perf_counter();points,means,feas,inc,matches=report(out);cfg=read(out/'config.json');rules=read(out/'comparison_rules.json')
    def point(s,c,w='unweighted',scope='expanded_catchup',b=360,split='test'):return points[s,c,split,w,b,scope]
    def values(c,kind,ep,w='unweighted',scope='expanded_catchup',b=360,split='test'):return [point(s,c,w,scope,b,split)[kind][ep] for s in range(3)]
    def passes(c,w,split='test'):return sum(all(r['pass'] for r in feas if r['seed']==s and r['condition']==c and r['weight']==w and r['split']==split) for s in range(3))
    lines=['# Complete fixed-prediction comparison','', 'All values are nats. Lower task loss and lower prior-relative recovery gain are favorable. U and PWGTP use the same unweighted-validation-selected predictions. Means ± sample SD describe three fitting seeds on one reused cohort; these are DEVELOPMENT EVALUATION results. Native anchor identity is separate from the refitted readout criteria.','']
    primary=[]
    for c in CONDITIONS:
        primary.append([NAMES[c],f"{passes(c,'unweighted')}/3 / {passes(c,'person_weighted')}/3",*[f"{statistics.mean(values(c,k,e)):.6f} / {statistics.mean(values(c,k,e,'person_weighted')):.6f}" for k,e in [('utility','same_residence'),('gains','AB/SEX'),('gains','AB/RAC1P')]]])
    lines += [md(['System','Source readout pass U / PWGTP','Residence loss U / PWGTP','AB SEX gain U / PWGTP','AB race gain U / PWGTP'],primary),'','Recovery in this primary table uses expanded catch-up at360. Full race assessment remains unassessable: code4 lacks independent fitting/validation support.','']
    for w in ('unweighted','person_weighted'):
        lines += [f'## Five task losses: {w}','',md(['System',*TASKS],[[NAMES[c],*[cell(values(c,'utility',t,w)) for t in TASKS]] for c in CONDITIONS]),'']
        for scope in SCOPES:
            for view,targets in ROLES.items():
                lines += [f'## {view} recovery, {w}, {scope},360','',md(['System',*targets],[[NAMES[c],*[cell(values(c,'gains',view+'/'+t,w,scope)) for t in targets]] for c in CONDITIONS]),'']
    lines += ['[Every selected endpoint](PER_SEED.csv), [all fixed paired contrasts](PAIRED_AGGREGATE.csv), [utility matching and exclusions](UTILITY_MATCHES.csv), [incremental disclosure](INCREMENTAL.csv) and [every-source-floor records](SOURCE_FEASIBILITY.csv) retain all seeds, validation/development, both weights, three scopes and120/360 budgets. Unit `metrics.json` and audit selections preserve every candidate.','']
    (out/'TABLE.md').write_text('\n'.join(lines))
    lines=['# Every seed and selected endpoint','', 'Development scores; expanded catch-up360. Complete other-scope/budget/validation values remain in PER_SEED.csv.','']
    for w in ('unweighted','person_weighted'):
        lines += [f'## Utility: {w}','',md(['System','Seed',*TASKS],[[NAMES[c],s,*[f"{point(s,c,w)['utility'][t]:.9f}" for t in TASKS]] for c in CONDITIONS for s in range(3)]),'']
        for view,targets in ROLES.items():lines += [f'## {view} recovery: {w}','',md(['System','Seed',*targets],[[NAMES[c],s,*[f"{point(s,c,w)['gains'][view+'/'+t]:.9f}" for t in targets]] for c in CONDITIONS for s in range(3)]),'']
    (out/'PER_SEED.md').write_text('\n'.join(lines))
    frows=[[NAMES[c],*[f'{passes(c,w,split)}/3' for split in ('validation','test') for w in ('unweighted','person_weighted')]] for c in CONDITIONS]
    lines=['# Source services and separately fitted readouts','', 'All original native source services are exactly preserved. Readout probes are additional logistic/MLP fits on the fixed released information, using the unchanged 2,048-label pool and validation rule. Their errors do not redefine the original service function. These rows overlap upstream anchor fitting exposure; they are conditional readout diagnostics.','',md(['System','Validation U','Validation PWGTP','Development U','Development PWGTP'],frows),'','Each of income, employment and coverage must be ≤ its original same-seed/split/weight selected PCA32 probe loss + .01. No mean-source substitution or native/readout best-of selection.','',md(['System','Seed','Split','Weight','Task','Loss','PCA32+.01','Excess','Pass'],[[NAMES[r['condition']],r['seed'],r['split'],r['weight'],r['task'],f"{r['loss']:.9f}",f"{r['parent_loss']+.01:.9f}",f"{r['excess_over_allowance']:+.9f}",r['pass']] for r in feas]),'']
    (out/'SOURCE_FEASIBILITY.md').write_text('\n'.join(lines))
    transfer=[]
    for c in CONDITIONS:
        for w in ('unweighted','person_weighted'):
            ds=[point(s,'H',w)['utility']['same_residence']-point(s,c,w)['utility']['same_residence'] for s in range(3)]
            transfer.append({'condition':c,'weight':w,'mean_residence_improvement_over_H':statistics.mean(ds),'sample_sd':statistics.stdev(ds),'seeds_ge_0p01':sum(v>=.01-1e-12 for v in ds),'source_passes':passes(c,w),**{f'seed_{i}':v for i,v in enumerate(ds)}})
    csvout(out/'TRANSFER.csv',transfer)
    lines=['# Reserved-task transfer','', 'Positive residence improvement means H loss minus augmented loss. The .01 reference remains fixed. A mean benefit does not imply every seed passes it, the source readout criteria, or additional recovery references.','',md(['System','Weight','Improvement mean ± SD','Seeds ≥.01','Source pass','s0','s1','s2'],[[NAMES[r['condition']],r['weight'],f"{r['mean_residence_improvement_over_H']:.6f} ± {r['sample_sd']:.6f}",f"{r['seeds_ge_0p01']}/3",f"{r['source_passes']}/3",*[f"{r['seed_'+str(s)]:+.6f}" for s in range(3)]] for r in transfer]),'','B receives the same two-column coverage anchor in every system. Its coverage and commute readouts and legal B attacks are exact aliases after recipe/input checks. Fixing these outputs does not assert preservation of historical feature-based commute utility. Historical context is separately labeled in [CONTEXT.md](CONTEXT.md).','']
    (out/'TRANSFER_ANALYSIS.md').write_text('\n'.join(lines))
    mr=[r for r in matches if r['split']=='test' and r['budget']==360 and r['scope']=='expanded_catchup' and r['panel']=='residential_transfer' and r['delta']==.001 and r['attribute']=='SEX']
    lines=['# Fixed J comparisons and utility eligibility','', 'Primary comparison: all three source readout floors, source plus residence, δ=.001 per task, expanded catch-up360, development. Close requires every absolute difference within δ. Directional permits task improvements but no deterioration above δ. Lower coalition SEX recovery is a separate requirement. The same named comparator is retained across all three seeds; no qualifying-only means.','',md(['J comparator','Weight','Both source floors','Close eligible','Directional eligible','Close + lower SEX','Directional + lower SEX'],[[NAMES[r['comparator']],r['weight'],f"{r['both_source_feasible']}/3",f"{r['eligible_close']}/3",f"{r['eligible_directional']}/3",f"{r['qualifies_close']}/3",f"{r['qualifies_directional']}/3"] for r in mr]),'','[All fixed-pair exclusions](UTILITY_MATCHES.csv) show task differences and source failures for every seed. [Aggregates](UTILITY_MATCHES_AGGREGATE.csv) retain the four panels, four deltas, both attributes, weights, budgets, audit scopes and validation/development. All observed points are alternatives; no deployment-selection procedure or interpolated frontier is claimed.','']
    (out/'MATCHING_ANALYSIS.md').write_text('\n'.join(lines))
    # Absolute H leakage and per-view increments, with strongest scope and weighting beside independent scopes.
    lines=['# Anchor disclosure and additional measured recovery','', 'H is audited directly under the new complete-vector schema. Its nine stationary observers receive the representation-fitting label exposure before own saved-start catch-up. Every augmented pool contains the corresponding legal H candidates. A negative selected development increment does not remove information already available in H; validation selection can generalize differently.','']
    for w in ('unweighted','person_weighted'):
        for scope in SCOPES:
            for view in ROLES:
                lines += [f'## {view}, {w}, {scope},360','',md(['System','Absolute SEX','SEX increment over H','Absolute race','Race increment over H'],[[NAMES[c],*[v for t in ('SEX','RAC1P') for v in (cell(values(c,'gains',view+'/'+t,w,scope)),cell([point(s,c,w,scope)['gains'][view+'/'+t]-point(s,'H',w,scope)['gains'][view+'/'+t] for s in range(3)]))]] for c in CONDITIONS]),'']
    lines += ['The .005-nat additional-sensitive-recovery reference applies separately to each attribute/view; it is not an absolute privacy bound. All opposing-task gains and signed increments remain in [INCREMENTAL.csv](INCREMENTAL.csv). Race code4 support prevents a full-category assessment.','']
    (out/'ANCHOR_DISCLOSURE.md').write_text('\n'.join(lines))
    anchors=read(out/'ANCHOR_PARITY.json');lines=['# Exact original source-function preservation','', 'All nine pinned selected models reproduce their original complete class0/class1 vectors exactly on downstream validation and development. Their original unweighted/PWGTP source scores replay exactly. The wire uses a lossless float64 container; it does not replace a separately rounded softmax column by a complement. Prediction metrics apply the historical scorer to the authoritative vectors.','',md(['Seed','Task','Selected original model','Original fit labels','Checkpoint epoch','Validation U / PWGTP','Development U / PWGTP'],[[r['seed'],r['task'],r['candidate_id'],r['metadata']['fit_rows'],r['metadata'].get('selected_epoch','fixed'),*[f"{r['checks'][sp]['unweighted']['log_loss']:.9f} / {r['checks'][sp]['PWGTP']['log_loss']:.9f}" for sp in ('validation','test')]] for r in anchors]),'','[Exact identities, models, checkpoints and complete metric dictionaries](ANCHOR_PARITY.json), [prefit inputs](PREFIT_IDENTITY.json), [reuse manifest](REUSE_MANIFEST.json), and [schema](INPUT_SCHEMA.json) bind provenance. H and every augmented system expose those same values. Public source-anchor parameters do not provide raw person PCA inputs. Auxiliary native heads are separate diagnostics and are public compositions when their features are available.','']
    (out/'ANCHOR_PARITY.md').write_text('\n'.join(lines))
    # Audit counts, scope and budget changes; score signs never drive candidate selection.
    counts={'five_candidate_roles':0,'fresh_MLP360':0,'static_logistic_tree':0,'catchup360':0,'utility_candidates':0};winners=[];budget_rows=[];exposure=[]
    for s,c in itertools.product(range(3),CONDITIONS):
        dest=out/f'seed_{s}'/c;a=read(dest/'audits/audit_selection.json');n=a['counts'];counts['five_candidate_roles']+=n['new_five_candidate_roles'];counts['fresh_MLP360']+=n['new_fresh_mlp_trajectories'];counts['static_logistic_tree']+=n['new_static_candidates'];counts['catchup360']+=n['own_catchup_trajectories'];counts['utility_candidates']+=10 if c=='H' else 6
        with np.load(dest/'predictions.npz') as z:
            for role in a['selections']['360']:
                v,t=role.split('/')
                for scope in SCOPES:
                    ids=[a['selections'][str(b)][role][scope] for b in (120,360)];meta=a['candidates']['360'][role][ids[1]]
                    changes={sp:not np.array_equal(z[f'audit/{role}/120/{ids[0]}/{sp}'],z[f'audit/{role}/360/{ids[1]}/{sp}']) for sp in ('validation','test')}
                    budget_rows.append({'seed':s,'condition':c,'role':role,'scope':scope,'candidate120':ids[0],'candidate360':ids[1],'epoch360':meta.get('selected_epoch'),'catchup':'catchup' in ids[1],'epoch0_catchup':'catchup' in ids[1] and meta.get('selected_epoch')==0,**changes})
                    winners.append({'seed':s,'condition':c,'role':role,'scope':scope,'candidate_id':ids[1],'path':meta['base_candidate_directory'],'source_view':meta['source_view'],'epoch':meta.get('selected_epoch'),'origin':meta['candidate_origin'],'ancestor':'anchor__' in ids[1],'inherited_exposure':meta.get('inherited_exposure')})
    counts.update(total_new_auditor_fits=counts['fresh_MLP360']+counts['static_logistic_tree']+counts['catchup360'],named_systems=18,learned_final_continuations=12,shared_source_warmups=3,unique_final_observer_trajectories=102,named_observer_roles=162,independent_B_five_role_pools_aliased=75,B_catchup_roles_aliased=60,derived_B_role_duplicates_avoided=60,historical_models_refitted=0)
    (out/'FITTING_COUNTS.json').write_text(json.dumps(counts,indent=2)+'\n');csvout(out/'AUDIT_BUDGET.csv',budget_rows);csvout(out/'SELECTED_CANDIDATES.csv',winners)
    table=[]
    for c in CONDITIONS:
        rr=[r for r in budget_rows if r['condition']==c and r['scope']=='expanded_catchup'];table.append([NAMES[c],sum(r['catchup'] for r in rr),sum(r['epoch0_catchup'] for r in rr),*[sum(r['test'] for r in budget_rows if r['condition']==c and r['scope']==sc) for sc in SCOPES]])
    (out/'AUDIT_FINDINGS.md').write_text('\n'.join(['# Audit budgets, inherited exposure and selection','', 'All eleven roles per named system are represented at120/360 under three scopes. Fresh means attacker-fit exposure; static and learned observer catch-up additionally inherits260 representation-fit label passes. H/E have stationary A inputs; learned observers saw changing A features. B trajectories, inputs and fitting recipes are exact aliases across all six systems. Epoch0 winners reflect inherited exposure, not a gain caused by additional catch-up updates.','',md(['System','Pooled360 catch-up winners /33 roles','Epoch0 winners','Standard changed development predictions','Expanded independent changed','Pooled changed'],table),'',f"Unique new fitting: {counts['fresh_MLP360']} fresh MLP paths, {counts['static_logistic_tree']} logistic/tree candidates, {counts['catchup360']} saved-start paths, and {counts['utility_candidates']} utility candidates. Inherited/projection aliases are not extra fits. Pinned prior/exposed controls keep their original budgets and failures.",'','All legal A/B sensitive candidates project into the corresponding AB pool. Every augmented pool also contains H projections; raw columns are extracted before saved preprocessing. Auxiliary-head probability compositions keep all anchors and both class columns. No other condition’s auxiliary channel or hidden input enters. Candidate paths, selection lineage and late epochs are in [SELECTED_CANDIDATES.csv](SELECTED_CANDIDATES.csv) and [AUDIT_BUDGET.csv](AUDIT_BUDGET.csv). Validation-only pool containment does not guarantee monotone selected development recovery.','', 'See [absolute and incremental scope tables](ANCHOR_DISCLOSURE.md) and [all fixed-pair differences](PAIRED_AGGREGATE.csv) for any independent-versus-pooled or weighted reversal. Full nine-category race protection is unassessable because code4 is absent from independent fitting/validation; exposed-control failures are retained.','']))
    plot(out,points,feas)
    (out/'SUMMARY_RUNTIME.json').write_text(json.dumps({'seconds':time.perf_counter()-started},indent=2)+'\n')

def plot(out,points,feas):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures';folder.mkdir(exist_ok=True);colors={'H':'#333333','E':'#8860a7','A0':'#e58b25','L025':'#478a47','L20':'#2878b5','J':'#c44343'}
    def pt(s,c,w):return points[s,c,'test',w,360,'expanded_catchup']
    for target in ('SEX','RAC1P'):
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for ax,w in zip(axes,('unweighted','person_weighted')):
            for c in CONDITIONS:
                x=[pt(s,'H',w)['utility']['same_residence']-pt(s,c,w)['utility']['same_residence'] for s in range(3)];y=[pt(s,c,w)['gains']['AB/'+target]-pt(s,'H',w)['gains']['AB/'+target] for s in range(3)]
                ax.scatter(x,y,color=colors[c],alpha=.5,s=24);ax.scatter(np.mean(x),np.mean(y),color=colors[c],marker='D',s=65,label=NAMES[c])
            ax.axvline(.01,color='grey',ls='--');ax.axhline(.005,color='grey',ls=':');ax.axhline(0,color='black',lw=.6);ax.set(xlabel='Residence improvement over H (nats; right better)',ylabel=f'Additional AB {target} gain (nats; down better)',title=w);ax.grid(alpha=.2)
        axes[1].legend(fontsize=8);fig.suptitle('Fixed observed systems; expanded catch-up360'+(' — code4 support absent' if target=='RAC1P' else ''));fig.text(.5,.01,'Small points: all three seeds. Diamonds: descriptive means. No interpolated boundary or privacy guarantee.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(folder/f'residence_increment_{target}.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,w in zip(axes,('unweighted','person_weighted')):
        for j,t in enumerate(TASKS[:3]):
            for k,c in enumerate(CONDITIONS):
                rr=[r['excess_over_allowance'] for r in feas if r['condition']==c and r['weight']==w and r['split']=='test' and r['task']==t]
                ax.scatter(np.array([k-.05,k,k+.05])+(j-1)*.2,rr,s=18,color=('#1b9e77','#d95f02','#7570b3')[j],label=t if k==0 else None)
        ax.axhline(0,color='black');ax.axhline(-.01,color='grey',ls=':',label='Exact native anchors');ax.set_xticks(range(6),[NAMES[c] for c in CONDITIONS],rotation=30);ax.set(title=w,ylabel='Refitted source loss − original PCA32 − .01');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Exact native service parity coexists with separate readout outcomes');fig.tight_layout();fig.savefig(folder/'source_probes.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,w in zip(axes,('unweighted','person_weighted')):
        ep=[v+'/'+t for v in ROLES for t in ('SEX','RAC1P')]
        for j,key in enumerate(ep):
            vals=[pt(s,'H',w)['gains'][key] for s in range(3)];ax.scatter([j-.08,j,j+.08],vals,color='black',s=20);ax.bar(j,np.mean(vals),color='#a8b6c3',zorder=0)
        ax.set_xticks(range(len(ep)),ep,rotation=30);ax.axhline(0,color='black',lw=.6);ax.set(title=w,ylabel='Absolute H recovery gain (nats)')
    fig.suptitle('Anchor-only disclosure; expanded catch-up360; race support incomplete');fig.tight_layout();fig.savefig(folder/'absolute_H.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(18,8))
    endpoints=[('utility','same_residence'),('gains','A/SEX'),('gains','AB/SEX'),('gains','A/RAC1P'),('gains','AB/RAC1P')]
    for i,w in enumerate(('unweighted','person_weighted')):
        for j,comp in enumerate(('L025','L20','A0','E')):
            ax=axes[i,j]
            for n,(kind,ep) in enumerate(endpoints):
                vs=[pt(s,'J',w)[kind][ep]-pt(s,comp,w)[kind][ep] for s in range(3)];ax.scatter([n-.1,n,n+.1],vs,color='black');ax.bar(n,np.mean(vs),alpha=.5,color=colors[comp])
            ax.axhline(0,color='black',lw=.6);ax.set_xticks(range(len(endpoints)),[ep for _,ep in endpoints],rotation=25);ax.set(title=f'H+J minus {NAMES[comp]}; {w}',ylabel='Difference in nats (down favorable)')
    fig.suptitle('Utility and recovery components stay separate; pooled360, three seeds');fig.tight_layout();fig.savefig(folder/'J_local_components.png',dpi=170);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260909_acs_fixed_predictions_v1');a=p.parse_args();summarize(a.out)
