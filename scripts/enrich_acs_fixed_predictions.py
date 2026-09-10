"""Post-freeze exposure, native-head and unchanged historical-context diagnostics."""
import argparse,csv,json,math,time,itertools,statistics
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts.report_acs_fixed_predictions import ROOT,read,csvout,selected,CONDITIONS
from scripts.summarize_acs_fixed_predictions import md,NAMES,TASKS,cell
from scripts.summarize_acs_pca16_init import utility_criteria
from scripts import verify_acs_bottleneck_scores as check
from experiments.acs_transfer_data import load_cohort,split_households

def run(out):
    tick=time.perf_counter();cfg=read(out/'config.json');assert read(out/'RELEASE_MANIFEST.json')['all18_frozen'];points,_,_=selected(out);pcfg=read(ROOT/cfg['parent_results']/'config.json');frame,_=load_cohort(ROOT/pcfg['raw_path'],pcfg['sample_cap'],pcfg['sample_seed']);identity=read(out/'PREFIT_IDENTITY.json');exposure=[];native=[];context=[];half=[];grad=[];aliases=[];used={};actual_forward=0;actual_observer_calls=0
    for s in range(3):
        d=out/f'seed_{s}';pools=split_households(frame,s);training=read(d/'training/training.json');n=training['fit_rows'];b=math.ceil(n/256);actual_forward+=380*b;actual_observer_calls+=1500*b
        with np.load(d/'anchors.npz') as z:
            for pool in ('representation_fit','source_validation','downstream_validation','test'):
                f=frame.iloc[pools[pool]]
                for task,(view,offset) in {'income_binary':('A',0),'civilian_at_work':('A',2),'public_coverage':('B',0)}.items():
                    y,mask=check.labels(f,task);p=z[pool+'/'+view][mask,offset:offset+2]
                    for weight in (False,True):native.append({'seed':s,'condition':'ALL_H_ANCHORS','task':task,'pool':pool,'weight':'person_weighted' if weight else 'unweighted','scores':check.independent_scores(y[mask],p,2,f.PWGTP.to_numpy(float)[mask] if weight else None),'prediction_sha256':check.array_hash(p),'selection':'original pinned prediction function'})
        for target,prior in training['priors'].items():
            y,m=check.labels(frame.iloc[pools['representation_fit']],target);yy=np.where(m,y,-1);k=9 if target=='RAC1P' else 2;counts=np.bincount(y[m],minlength=k);p=counts/counts.sum();entropy=-sum(v*math.log(v) for v in p if v>0);assert counts.tolist()==prior['support'];assert abs(entropy-prior['entropy'])<1e-14
            exposure.append({'seed':s,'target':target,'representation_fit_rows':n,'valid_labels':int(m.sum()),'labels_sha256':check.array_hash(yy),'raw_row_sha256':check.array_hash(frame.iloc[pools['representation_fit']]._raw_row.to_numpy()),'support':counts.tolist(),'prior_entropy':entropy,'observer_passes':260,'observer_valid_label_presentations':260*int(m.sum()),'auxiliary_source_passes':140 if target in TASKS[:2] else 0,'anchor_labels':2048 if target in TASKS[:3] else 0,'source_probe_fitting_overlap_with_anchor':target in TASKS[:3]})
        for c in ('A0','L025','L20','J'):
            with np.load(d/'training'/c/'releases.npz') as z:
                for pool in ('source_validation','test'):
                    f=frame.iloc[pools[pool]]
                    for task,offset in [('income_binary',4),('civilian_at_work',6)]:
                        y,mask=check.labels(f,task);p=z['derived/A/'+pool][mask,offset:offset+2]
                        for weight in (False,True):native.append({'seed':s,'condition':c,'task':task,'pool':pool,'weight':'person_weighted' if weight else 'unweighted','scores':check.independent_scores(y[mask],p,2,f.PWGTP.to_numpy(float)[mask] if weight else None),'prediction_sha256':check.array_hash(p),'selection':'fixed final public auxiliary head; never best-of with anchor/probe'})
            for stage in ('fork_diagnostic','final_diagnostic'):
                rec=training['arms'][c][stage]
                for group,terms in rec['gradients'].items():
                    for term,values in terms.items():grad.append({'seed':s,'condition':c,'stage':stage,'parameter_group':group,'component':term,'coefficient':rec['coefficients'][term],**values})
        for c in CONDITIONS:
            aliases.append({'seed':s,'condition':c,'B_observer_identity':training['B_final_identity'][c],'B_trajectory_origin':'A0','B_source_and_commute_readout_origin':'H','B_audit_origin':'H','H_projection_is_identical_evidence':True,'A_stationary_observer':c in ('H','E'),'new_forward_continuation':c not in ('H','E')})
        oldpath=ROOT/cfg['reference_results']/f'seed_{s}'/'metrics.json';old=read(oldpath)['raw_metrics'];used[str(oldpath.relative_to(ROOT))]=check.sha(oldpath)
        index={(r['release'],r['target']):r for r in old if r['role']=='transfer' and r['selected']}
        for split,w in itertools.product(('validation','test'),('unweighted','person_weighted')):
            score=split+('_person_weighted' if w=='person_weighted' else '');parent={t:index['E_pca',t][score]['log_loss'] for t in TASKS};banks={c:index[c,'same_residence'][score]['log_loss'] for c in ('B_rich_bank','C_tree_bank')}
            for c in ('E_pca','E_pca_leace','B_rich_bank','C_tree_bank'):
                for t in TASKS:context.append({'seed':s,'reference':c,'task':t,'split':split,'weight':w,'loss':index[c,t][score]['log_loss'],'interface':'historical single release','source':str(oldpath.relative_to(ROOT)),'audit_scope':'original120; not a matching360 audit'})
            for c in CONDITIONS:
                criteria=utility_criteria(points[s,c,split,w,360,'expanded_catchup']['utility'],parent,banks);half.append({'seed':s,'condition':c,'split':split,'weight':w,**criteria})
        for study,cs in [(cfg['source_guard_reference'],('F_T','P_T','F_G_J','P_G_J')),(cfg['coalition_reference_results'],('F_J','P_J'))]:
            for c in cs:
                path=ROOT/study/f'seed_{s}'/c/'metrics.json';raw=read(path)['raw_metrics'];used[str(path.relative_to(ROOT))]=check.sha(path)
                for r in raw:
                    if r['role']!='utility' or not r['selected_scopes']:continue
                    for split,w in itertools.product(('validation','test'),('unweighted','person_weighted')):
                        score=split+('_person_weighted' if w=='person_weighted' else '');context.append({'seed':s,'reference':Path(study).name+'/'+c,'task':r['target'],'split':split,'weight':w,'loss':r['scores'][score]['log_loss'],'interface':'historical paired F/P, different source training and disclosure','source':str(path.relative_to(ROOT)),'audit_scope':'own historical scope; contextual only'})
    for name,rows in [('EXPOSURE',exposure),('NATIVE_SCORES',native),('CONTEXT',context),('ORIGINAL_CRITERIA',half),('GRADIENT_DIAGNOSTICS',grad)]:
        csvout(out/(name+'.csv'),rows)
    (out/'ALIASES.json').write_text(json.dumps(aliases,indent=2)+'\n');(out/'CONTEXT_REUSE_HASHES.json').write_text(json.dumps(used,indent=2)+'\n')
    counts=read(out/'FITTING_COUNTS.json');counts.update(actual_live_auxiliary_optimizer_steps=actual_forward,actual_observer_optimizer_calls=actual_observer_calls,auxiliary_source_gradient_support='3152mapper+34heads',protection_gradient_support='3152mapper only',B_source_gradient_support=0,fixed_anchor_parameters_per_seed={s:sum(r['parameters'] for r in ident['anchors'].values()) for s,ident in identity.items()});(out/'FITTING_COUNTS.json').write_text(json.dumps(counts,indent=2)+'\n')
    lines=['# Training exposure, immutable B and public auxiliary heads','', 'Each source anchor inherits its original2,048-label selected-predictor fit. A learned auxiliary channel receives140 source-label passes (60 shared warmup plus80 continuation), so comparisons with H are not a label-count-only intervention. Auxiliary heads are public functions, are never substituted for the immutable source service, and enter expanded audits.','', 'Nine observer roles per named system have260 representation-fit passes. Actual unique final trajectories are102 across18 named systems: B’s four stationary roles are shared within seed; each A/AB system has its own five roles. Static H/E A inputs remain fixed throughout20+240passes; learned A inputs change. Identical exposure counts do not equate those histories. A0 supplies the common B trajectory from the identical pre-warmup initialization; no warmed state receives a second20-pass prefix.','', 'The original nine-role CE denominator is retained even where B work is aliased. All named final B tensors and Adam slots agree; every intermediate B tensor state used in learned forward objectives follows A0 at the same step. Source gradients reach mapper and auxiliary heads; protection reaches mapper only. Raw and applied norms at the predeclared batch/fork/final points remain in [GRADIENT_DIAGNOSTICS.csv](GRADIENT_DIAGNOSTICS.csv).','', '[Label/mask/support hashes and presentations](EXPOSURE.csv), [aliases](ALIASES.json), [actual unique fit counts](FITTING_COUNTS.json) and [native scores](NATIVE_SCORES.csv) distinguish fixed anchors, public auxiliary-head diagnostics and selected readouts. No residence or commute supervision enters auxiliary fitting.','']
    for w in ('unweighted','person_weighted'):
        tab=[]
        for c in ('ALL_H_ANCHORS','A0','L025','L20','J'):
            vals=[]
            for t in TASKS[:3]:
                v=[r['scores']['log_loss'] for r in native if r['condition']==c and r['task']==t and r['pool']=='test' and r['weight']==w];vals.append(cell(v) if v else 'No auxiliary B head')
            tab.append([NAMES.get(c,c),*vals])
        lines += [f'## Native development predictions, {w}','',md(['Fixed function','Income','Employment','Coverage'],tab),'']
    (out/'EXPOSURE_AND_NATIVE.md').write_text('\n'.join(lines))
    lines=['# Historical utility context and unchanged residential headroom','', 'These historical interfaces differ in source-label exposure, architectures and audit access. They do not replace H or the five matched augmented comparisons. Their original source identities and scopes are retained. The original PCA32 audit is120epochs, not a new360audit.','']
    for w in ('unweighted','person_weighted'):
        refs=list(dict.fromkeys(r['reference'] for r in context));tab=[]
        for c in refs:
            tab.append([c,*[cell([r['loss'] for r in context if r['reference']==c and r['task']==t and r['split']=='test' and r['weight']==w]) for t in TASKS]])
        lines += [f'## Development task losses, {w}','',md(['Historical reference',*TASKS],tab),'']
    lines += ['## Original residential half-headroom','', 'The parent remains original PCA32, and the denominator remains its positive advantage over the better original unprotected rich bank. Nonpositive denominators are undefined; H is not substituted as the parent. Passing .01 improvement over H is a separate criterion.','',md(['System','Weight','Development half-headroom pass seeds'],[[NAMES[c],w,f"{sum(r['residential_retention']['pass'] is True for r in half if r['condition']==c and r['weight']==w and r['split']=='test')}/3"] for c in CONDITIONS for w in ('unweighted','person_weighted')]),'','[Per-seed original criteria](ORIGINAL_CRITERIA.csv), [context values](CONTEXT.csv), and [hash-bound original sources](CONTEXT_REUSE_HASHES.json) retain the exact values and undefined cases. B coverage and commute are identical in the new matrix; historical F/P figures show the context rather than an implied preserved feature capability.','']
    (out/'CONTEXT.md').write_text('\n'.join(lines));(out/'ENRICH_RUNTIME.json').write_text(json.dumps({'seconds':time.perf_counter()-tick},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/redesign_20260909_acs_fixed_predictions_v1');a=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):run(a.out)
