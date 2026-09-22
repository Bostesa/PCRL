"""Independent mathematical replay of frozen channels; no fitting or repairs.

CMI uses an explicit probability log ratio, costs/joints use independent
histograms of original mechanism people, and support rules are reconstructed
from unweighted counts. No finite.py numerical function or production table
estimator is called. Aggregate diagnostics are public-safe; full cell support
is written only to a new task-owned private replay directory.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import expit

from .config import INPUTS, configuration, digest


TOLERANCES={'simplex':1e-7,'nonnegative':1e-8,'support':1e-7,'cmi':1e-7,
            'zero_linear':1e-8,'objective_ordering':1e-7,'objective_replay':1e-10,
            'table':1e-12,'aggregation':1e-12,'encoder_probability':1e-7,
            'privacy_row_roundoff_relative':16*np.finfo(float).eps}
CLIP=1e-5


def _max(a):return float(np.max(np.abs(a),initial=0))


def _error(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape:raise ValueError('Replay array shapes differ')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite replay array')
    return _max(a-b)


def _joint(p):
    p=np.asarray(p,float)
    if p.ndim!=3 or min(p.shape)<1 or not np.isfinite(p).all() or np.any(p<0):
        raise ValueError('Invalid empirical joint law')
    if abs(float(p.sum())-1)>1e-10:raise ValueError('Empirical joint law is not normalized')
    return p


def conditional_information(p,q):
    """Sum p(s,c,z) log[p(s,c,z)p(c)/(p(s,c)p(c,z))], in nats."""
    p=_joint(p);q=np.asarray(q,float)
    if (q.ndim!=2 or q.shape[0]!=p.shape[2] or not np.isfinite(q).all()
            or np.any(q<0) or _max(q.sum(1)-1)>TOLERANCES['simplex']):
        raise ValueError('Invalid stochastic channel for CMI replay')
    # Explicit contextwise multiplication avoids the optimizer's cone/formula.
    total=0.
    for c in range(p.shape[1]):
        scz=p[:,c,:]@q;pc=float(p[:,c,:].sum())
        if pc==0:continue
        psc=p[:,c,:].sum(1);pcz=scz.sum(0)
        si,zi=np.nonzero(scz>0);positive=scz[si,zi]
        if np.any(psc[si]<=0) or np.any(pcz[zi]<=0):raise ValueError('Positive joint mass has zero marginal')
        # Separate logarithms avoid underflow in products of rare marginals.
        total+=float(np.sum(positive*(np.log(positive)+np.log(pc)-np.log(psc[si])-np.log(pcz[zi]))))
    return max(0.,total)


def _cells(partition,ha,hb):
    centers=np.asarray(partition.centers,float);ha=np.asarray(ha);hb=np.asarray(hb)
    if centers.ndim!=2 or centers.shape[1]!=2 or ha.shape[1]!=4 or hb.shape!=(len(ha),2):
        raise ValueError('Service partition schema mismatch')
    squared=((ha[:,None,[1,3]]-centers[None,:,:])**2).sum(2)
    ca=np.argmin(squared,axis=1)
    return {'A':(ca,len(centers)),
            'AB':(2*ca+(hb[:,1]>partition.b_median).astype(int),2*len(centers))}


def _histogram(s,c,t,shape,weights):
    flat=np.ravel_multi_index((s,c,t),shape)
    return np.bincount(flat,weights=weights,minlength=int(np.prod(shape))).reshape(shape)


def reconstruct_tables(prepared,family,actions,*,fine_partition=None):
    """Rebuild costs and eight/sixteen laws from original people, without smoothing."""
    data=prepared['ctx']['pools']['representation_fit'];enc=prepared['encoded']['representation_fit']
    rows=np.asarray(prepared['roles']['mechanism'],int);n=len(data['ha'])
    if not len(rows) or len(np.unique(rows))!=len(rows) or np.any((rows<0)|(rows>=n)):
        raise ValueError('Invalid frozen mechanism row indices')
    nt=prepared['encoder'].code.n_states(family)
    full_t=np.asarray(enc['codes'][family]);t=full_t[rows]
    if np.any(t!=np.floor(t)) or np.any((t<0)|(t>=nt)):raise ValueError('Invalid frozen code')
    t=t.astype(int);w=np.asarray(data['weights'],float)[rows]
    if not np.isfinite(w).all() or np.any(w<=0):raise ValueError('Invalid PWGTP')
    b=np.asarray(enc['b'],float)
    if b.shape!=(n,) or np.any((b<=0)|(b>=1)) or not np.isfinite(b).all():raise ValueError('Invalid frozen baseline probabilities')
    offsets=np.asarray(prepared['encoder'].dictionaries[actions]['offsets'],float)
    g=np.clip(expit((np.log(b)-np.log1p(-b))[:,None]+offsets[None,:]),CLIP,1-CLIP)
    decoder_error=_error(g,enc['actions'][actions])
    y=np.asarray(data['labels']['same_residence'])[rows];valid_y=np.isin(y,[0,1])
    if not np.any(valid_y):raise ValueError('No observed task labels')
    yy=y[valid_y];gg=g[rows][valid_y]
    losses=-yy[:,None]*np.log(gg)-(1-yy[:,None])*np.log1p(-gg)
    d_u=np.vstack([losses[t[valid_y]==state].sum(0)/valid_y.sum() for state in range(nt)])
    d_w=np.vstack([(losses[t[valid_y]==state]*w[valid_y][t[valid_y]==state,None]).sum(0)/w[valid_y].sum()
                   for state in range(nt)])
    joints={};support={}
    partitions=[('',prepared['encoder'].partitions)]
    if fine_partition is not None:partitions.append(('/fineC',fine_partition))
    for suffix,partition in partitions:
        cells=_cells(partition,data['ha'],data['hb'])
        for protected,ns in (('SEX',2),('RAC1P',9)):
            ss=np.asarray(data['labels'][protected])[rows]
            valid=np.isfinite(ss)&(ss==np.floor(ss))&(ss>=0)&(ss<ns)
            if not np.any(valid):raise ValueError('No observed protected labels')
            s=ss[valid].astype(int)
            for view,(context,nc) in cells.items():
                shape=(ns,nc,nt);c=context[rows][valid]
                count=_histogram(s,c,t[valid],shape,np.ones(valid.sum()))
                sumw=_histogram(s,c,t[valid],shape,w[valid])
                sumw2=_histogram(s,c,t[valid],shape,w[valid]**2)
                key=f'{view}/{protected}'
                joints[key+'/U'+suffix]=count/count.sum()
                joints[key+'/W'+suffix]=sumw/sumw.sum()
                support[key+suffix]={'count':count.astype(np.int64),'sumw':sumw,'sumw2':sumw2,
                    'ess':np.divide(sumw**2,sumw2,out=np.zeros(shape),where=sumw2>0)}
    return {'cost_U':d_u,'cost_W':d_w,'cost':.5*(d_u+d_w),
            'state_mass':np.bincount(t,minlength=nt),'roles':joints,'support':support,
            'decoder_error':decoder_error,'actions':actions}


def support_diagnostics(q,mass,parents,zero_action):
    q=np.asarray(q,float);mass=np.asarray(mass,float);parents=np.asarray(parents,int)
    if q.shape[0]!=len(mass) or parents.shape!=mass.shape:raise ValueError('Support shape mismatch')
    unsupported=np.flatnonzero(mass==0);errors=[];absent=set();tied=0
    for state in unsupported:
        siblings=np.flatnonzero((parents==parents[state])&(mass>0))
        if len(siblings):
            target=np.average(q[siblings],axis=0,weights=mass[siblings]);tied+=1
        else:
            target=np.zeros(q.shape[1]);target[zero_action]=1;absent.add(int(parents[state]))
        errors.append(_error(q[state],target))
    return {'maximum_error':max(errors,default=0.),'supported_states':int(np.sum(mass>0)),
            'unsupported_states':len(unsupported),'tied_children':tied,'absent_parents':len(absent)}


def deployment_support(codes,weights,mass,parents):
    codes=np.asarray(codes,int);weights=np.asarray(weights,float);mass=np.asarray(mass)
    parents=np.asarray(parents,int)
    if codes.shape!=weights.shape or np.any((codes<0)|(codes>=len(mass))):raise ValueError('Invalid deployment support inputs')
    if not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('Invalid deployment weights')
    unsupported=mass[codes]==0
    absent={int(parent) for parent in np.unique(parents) if mass[parents==parent].sum()==0}
    absent_rows=np.isin(parents[codes],list(absent))
    return {'people':len(codes),'unsupported_count':int(unsupported.sum()),
            'unsupported_fraction':float(unsupported.mean()) if len(codes) else None,
            'unsupported_weight_fraction':float(weights[unsupported].sum()/weights.sum()) if len(codes) else None,
            'absent_parent_count':int(absent_rows.sum()),
            'absent_parent_fraction':float(absent_rows.mean()) if len(codes) else None,
            'absent_parent_weight_fraction':float(weights[absent_rows].sum()/weights.sum()) if len(codes) else None}


def _zero_equations(p):
    p=_joint(p);out=np.zeros_like(p)
    for c in range(p.shape[1]):
        pc=p[:,c,:].sum()
        if pc==0:continue
        direct=p[:,c,:]*pc
        product=p[:,c,:].sum(1)[:,None]*p[:,c,:].sum(0)[None,:]
        difference=direct-product
        scales=np.max(np.abs(direct),axis=1)+np.max(np.abs(product),axis=1)
        difference[np.max(np.abs(difference),axis=1)<=TOLERANCES['privacy_row_roundoff_relative']*scales]=0
        out[:,c,:]=difference/pc
    return out.reshape(-1,p.shape[2])


def zero_diagnostics(joints,q):
    q=np.asarray(q,float);nt=len(q)
    blocks={name:_zero_equations(p) for name,p in joints.items()}
    stacked=np.vstack(list(blocks.values())) if blocks else np.zeros((0,nt))
    singular=np.linalg.svd(stacked,compute_uv=False) if stacked.size else np.zeros(0)
    threshold=float(singular[0]*max(stacked.shape)*np.finfo(float).eps) if len(singular) else 0.
    rank=int(np.sum(singular>threshold));scale=np.max(np.abs(stacked),axis=1,initial=0)
    nonzero=scale>0;normalized=stacked[nonzero]/scale[nonzero,None]
    return {'rank':rank,'nullity':nt-rank,'rank_threshold':threshold,
            'rank_interpretation':'numerical SVD rank at the displayed threshold; not an exact algebraic certificate',
            'nullity_scope':'input-space privacy equations only; simplex and support restrictions also apply',
            'role_ranks':{name:int(np.linalg.matrix_rank(a)) for name,a in blocks.items()},
            'normalized_equation_error':_max(normalized@q),
            'constant_vector_residual':_max(stacked@np.ones(nt)),
            'unscaled_equation_error':_max(stacked@q)}


def refinement_errors(coarse,fine):
    nc=len(coarse['state_mass']);errors={}
    if len(fine['state_mass'])!=2*nc:raise ValueError('Nonbinary refinement shape')
    for field in ('cost_U','cost_W','cost'):
        errors[field]=_error(fine[field].reshape(nc,2,-1).sum(1),coarse[field])
    errors['state_mass']=_error(fine['state_mass'].reshape(nc,2).sum(1),coarse['state_mass'])
    if set(coarse['roles'])!=set(fine['roles']):raise ValueError('Refinement role schemas differ')
    for name,p in fine['roles'].items():
        errors['joint/'+name]=_error(p.reshape(*p.shape[:2],nc,2).sum(3),coarse['roles'][name])
    return {'maximum_error':max(errors.values(),default=0.),'errors':errors}


def embedding_errors(coarse,fine,coarse_q,encoded,family,actions):
    coarse_q=np.asarray(coarse_q,float);expanded=np.repeat(coarse_q,2,axis=0)
    errors={'objective':abs(float(np.sum(coarse['cost']*coarse_q))-float(np.sum(fine['cost']*expanded)))}
    for role,p in fine['roles'].items():
        errors['joint/'+role]=_error(np.tensordot(p,expanded,axes=(2,0)),
                                     np.tensordot(coarse['roles'][role],coarse_q,axes=(2,0)))
    for pool,e in encoded.items():
        tf=np.asarray(e['codes'][family],int);tc=np.asarray(e['codes']['T0'],int)
        if not np.array_equal(tf//2,tc):raise ValueError('Deployment code lost its parent')
        actual=expanded[tf];expected=coarse_q[tc];g=e['actions'][actions]
        errors['token_law/'+pool]=_error(actual,expected)
        errors['prediction/'+pool]=_error(np.sum(actual*g,1),np.sum(expected*g,1))
        errors['positive_CE/'+pool]=_error(np.sum(actual*-np.log(g),1),np.sum(expected*-np.log(g),1))
    return {'maximum_error':max(errors.values(),default=0.),'errors':errors}


def inspect_channel(table,result,parents,zero_action):
    q=np.asarray(result['Q'],float);cost=np.asarray(table['cost'],float)
    if q.shape!=cost.shape or not np.isfinite(q).all():raise ValueError('Invalid saved Q shape/value')
    objective=float(np.sum(cost*q));objective_error=abs(objective-float(result['objective']))
    simplex=_max(q.sum(1)-1);negative=float(max(0.,-np.min(q)))
    info={role:conditional_information(p,q) for role,p in table['roles'].items()}
    selected=result['constrained_roles'];budget=result['budget']
    if not set(selected).issubset(info):raise ValueError('Saved constrained role absent from tables')
    excess=max((max(0.,info[role]-budget) for role in selected),default=0.) if budget is not None else 0.
    support=support_diagnostics(q,table['state_mass'],parents,zero_action)
    zero=zero_diagnostics({role:table['roles'][role] for role in selected},q)
    rank_difference=zero['rank']-int(result.get('rank',zero['rank']))
    violations=[]
    for key,value,tolerance in [('simplex',simplex,TOLERANCES['simplex']),
            ('nonnegative',negative,TOLERANCES['nonnegative']),('objective',objective_error,TOLERANCES['objective_replay']),
            ('CMI',excess,TOLERANCES['cmi']),('unsupported_state_affine',support['maximum_error'],TOLERANCES['support'])]:
        if value>tolerance:violations.append(key)
    if budget==0 and zero['normalized_equation_error']>TOLERANCES['zero_linear']:violations.append('zero_linear')
    return {'passed':not violations,'violations':violations,'objective':objective,'objective_error':objective_error,
            'CMI_nats':info,'maximum_cmi_excess':excess,'simplex_error':simplex,'negative_mass_error':negative,
            'support':support,'zero_geometry':zero,'reported_rank_difference':rank_difference,
            'solver_status':result['status'],'solver_optimal':bool(result.get('optimal',False)),
            'optimality_scope':'solver status retained; mathematical replay does not certify a new optimum'}


def ordering_check(local,coalition):
    gap=float(local['objective']-coalition['objective'])
    optimal=(local.get('status')=='optimal' and coalition.get('status')=='optimal'
             and local.get('optimal') is True and coalition.get('optimal') is True)
    inversion=gap>TOLERANCES['objective_ordering']
    return {'local_minus_coalition_objective':gap,'ordering_inversion':inversion,
            'solver_status_supports_optimum_comparison':optimal,'warning':bool(optimal and inversion),
            'qualification':'both solver statuses optimal' if optimal else 'approximate/feasible candidates; optimum ordering not asserted',
            'tolerance':TOLERANCES['objective_ordering']}


def h_parity(ctx,inputs_root):
    path=Path(inputs_root)/'results/redesign_20260909_acs_fixed_predictions_v1'/f"seed_{ctx['anchor']}"/'anchors.npz'
    report={}
    with np.load(path,allow_pickle=False) as archive:
        for pool,data in ctx['pools'].items():
            record={}
            for role,key in (('A','ha'),('B','hb')):
                saved=archive[pool+'/'+role];current=np.asarray(data[key])
                record[role+'_byte_identical']=(saved.shape==current.shape and saved.dtype==current.dtype
                                               and saved.tobytes()==current.tobytes())
            report[pool]=record
    return report


def _read_tables(path):
    with np.load(path,allow_pickle=False) as archive:
        table={key:archive[key] for key in ('cost_U','cost_W','cost','state_mass')}
        table['roles']={key[6:]:archive[key] for key in archive.files if key.startswith('joint/')}
        table['support']={}
        for key in archive.files:
            if key.startswith('support/'):
                role,field=key[8:].rsplit('/',1)
                table['support'].setdefault(role,{})[field]=archive[key]
    return table


def compare_tables(actual,expected):
    errors={field:_error(actual[field],expected[field]) for field in ('cost_U','cost_W','cost','state_mass')}
    if set(actual['roles'])!=set(expected['roles']) or set(actual['support'])!=set(expected['support']):
        raise ValueError('Saved/reconstructed table schemas differ')
    for role in expected['roles']:errors['joint/'+role]=_error(actual['roles'][role],expected['roles'][role])
    support_errors={}
    for role,fields in expected['support'].items():
        if set(actual['support'][role])!=set(fields):raise ValueError('Saved support schema differs')
        for field,a in fields.items():
            b=actual['support'][role][field];error=_error(a,b)
            support_errors[role+'/'+field]={'absolute':error,'scaled':error/max(1.,_max(a))}
    support_max=max((e['scaled'] for e in support_errors.values()),default=0.)
    maximum=max(errors.values(),default=0.)
    return {'passed':maximum<=TOLERANCES['table'] and support_max<=TOLERANCES['table'],
            'maximum_error':maximum,'errors':errors,'support_maximum_scaled_error':support_max,
            'support_errors':support_errors}


def _support_summary(table):
    report={}
    for role,fields in table['support'].items():
        counts=fields['count'];nonzero=counts>0;ess=fields['ess'][nonzero]
        report[role]={'nominal_cells':counts.size,'observed_cells':int(nonzero.sum()),
            'zero_cells':int((~nonzero).sum()),'observed_people':int(counts.sum()),
            'minimum_observed_cell_count':int(counts[nonzero].min()) if np.any(nonzero) else None,
            'minimum_observed_cell_ESS':float(ess.min()) if len(ess) else None,
            'median_observed_cell_ESS':float(np.median(ess)) if len(ess) else None}
    return report


def _encoder_replay(prepared,encoder):
    from .data import RuntimeInputs
    encoded={};reports={}
    for pool,data in prepared['ctx']['pools'].items():
        before=np.asarray(data['ha']).tobytes()
        fresh=encoder.encode(RuntimeInputs(data['x'],data['ha']));old=prepared['encoded'][pool]
        errors={key:_error(fresh[key],old[key]) for key in ('p','b','r','risk')}
        for actions,g in fresh['actions'].items():errors['actions/'+str(actions)]=_error(g,old['actions'][actions])
        if 'global_offsets' in fresh:errors['global_offsets']=_error(fresh['global_offsets'],old['global_offsets'])
        codes={family:int(np.count_nonzero(np.asarray(fresh['codes'][family])!=old['codes'][family])) for family in INPUTS}
        h_unchanged=before==np.asarray(data['ha']).tobytes()
        reports[pool]={'maximum_probability_error':max(errors.values(),default=0.),'errors':errors,
            'code_changed_rows':codes,'H_unmodified':h_unchanged,
            'passed':h_unchanged and max(errors.values(),default=0.)<=TOLERANCES['encoder_probability'] and not any(codes.values())}
        encoded[pool]=fresh
    return encoded,reports


def verify_anchor(anchor,include_evaluation=False,*,out_dir=None,public_path=None,inputs_root=None):
    """Read accepted frozen artifacts after resource/optional evaluation gates.

    A new private replay directory and aggregate JSON are the only writes. An
    integrity exception becomes a per-map diagnostic, never a fitting retry.
    The caller controls when this function runs; importing it opens no data.
    """
    from . import run
    from .data import RuntimeInputs,load_anchor,validate_evaluation_permit
    schedule_path=run.OUT/'RESOURCE_SCHEDULE.json'
    schedule=json.loads(schedule_path.read_text())
    if not schedule.get('frozen_before_comparative_outcomes') or schedule.get('config_hash')!=digest(configuration()):
        raise PermissionError('Mathematical replay requires the frozen resource schedule')
    permit=run.OUT/'SELECTION.json'
    if include_evaluation:
        validate_evaluation_permit(permit)
        selection=json.loads(permit.read_text())
        if 'config_hash' in selection and selection['config_hash']!=digest(configuration()):
            raise PermissionError('Evaluation selection configuration changed')
        if 'source_hashes' in selection and selection['source_hashes']!=run.source_fingerprint():
            raise PermissionError('Evaluation selection sources changed')
    p=run.prepare(anchor,allow_fit=False);receipt=run._prepared_receipt(anchor)
    if 'test' in p['ctx']['pools'] or set(p['encoded'])!=set(p['ctx']['pools']):
        raise PermissionError('Primary preparation must contain only its frozen fitting/validation pools')
    root=Path(inputs_root) if inputs_root is not None else run.OUT/'private/inputs'
    private=Path(out_dir) if out_dir is not None else run.OUT/'private/math_replay'/f'anchor_{anchor}'/str(time.time_ns())
    private=private.resolve()
    if not private.is_relative_to((run.OUT/'private').resolve()):raise ValueError('Full support output must remain task-owned private')
    private.mkdir(parents=True,exist_ok=False)
    try:
        fresh_encoder=joblib.load(run.anchor_dir(anchor)/'encoder/encoder.joblib')
        encoded,encoder_report=_encoder_replay(p,fresh_encoder)
    except Exception as error:
        run.atomic(private/'encoder_incident.json',{'type':type(error).__name__,'message':str(error)})
        # Continue the independent table arithmetic from frozen cached values
        # solely to diagnose other issues; the aggregate replay remains failed.
        fresh_encoder=p['encoder'];encoded=dict(p['encoded'])
        encoder_report={'frozen_model_replay':{'passed':False,'integrity_error_type':type(error).__name__,
                       'qualification':'table diagnostics use frozen cached predictions; fresh weights did not replay'}}
    replay={**p,'encoder':fresh_encoder,'encoded':encoded}
    if include_evaluation:
        evaluation=load_anchor(anchor,pools=('test',),inputs_root=root,evaluation_permit=permit)
        replay['ctx']={**p['ctx'],'pools':{**p['ctx']['pools'],**evaluation['pools']}}
        for pool,data in evaluation['pools'].items():
            encoded[pool]=fresh_encoder.encode(RuntimeInputs(data['x'],data['ha']))
    try:h_report=h_parity(replay['ctx'],root)
    except Exception as error:
        run.atomic(private/'H_parity_incident.json',{'type':type(error).__name__,'message':str(error)})
        h_report={'owned_anchor_comparison':{'A_byte_identical':False,'B_byte_identical':False}}
    report={'schema':1,'created_utc':run.now(),'anchor':anchor,'include_evaluation':include_evaluation,
        'resource_schedule_sha256':run.sha(schedule_path),'prepared_cache_sha256':receipt['cache_sha256'],
        'selection_sha256':run.sha(permit) if include_evaluation else None,
        'source_hashes':run.source_fingerprint(),'replay_source_sha256':run.sha(__file__),
        'tolerances':dict(TOLERANCES),'encoder_replay':encoder_report,'H_parity':h_report,
        'maps':{},'code_refinements':{},'local_coalition_ordering':[],'warnings':[],
        'scope':'empirical numerical replay only; no new population/privacy or optimization certificate',
        'repairs_performed':False}
    tables_cache={};accepted={};results={};specs={};condition_partitions={}
    maps_root=run.anchor_dir(anchor)/'maps'
    paths=sorted(maps_root.glob('*/ACCEPTED.json')) if maps_root.exists() else []
    for marker in paths:
        name=marker.parent.name
        try:
            spec=run.map_spec(name,anchor)
            if spec is None:raise ValueError('Accepted channel is unregistered')
            accepted[name]=run._map_receipt(anchor,name)
            result=joblib.load(marker.parent/'solution.joblib');saved=_read_tables(marker.parent/'tables.npz')
            family,actions=spec['input'],spec['max_actions'];fine=None
            condition=spec.get('conditioning_family','primary')
            if spec.get('branch')=='C':
                from .robustness import prepared_for_spec
                fine=prepared_for_spec(anchor,p,spec,allow_fit=False)['encoder'].partitions
            condition_partitions[condition]=fine
            def material(code):
                key=(code,actions,condition)
                if key not in tables_cache:tables_cache[key]=reconstruct_tables(replay,code,actions,fine_partition=fine)
                return tables_cache[key]
            rebuilt=material(family)
            expected_roles=sorted(role for role in rebuilt['roles'] if spec['policy']=='C'
                                   or (spec['policy']=='L' and role.startswith('A/')))
            if (sorted(result['constrained_roles'])!=expected_roles or result['budget']!=spec['budget']
                    or result['input']!=family or result['actions']!=actions):
                raise ValueError('Saved solver specification/constrained role schema differs from registration')
            if bool(result.get('optimal'))!=(result.get('status')=='optimal'):
                raise ValueError('Saved solver optimality label contradicts its status')
            for key in ('simplex','nonnegative','support','cmi','zero_linear','privacy_row_roundoff_relative'):
                if result.get('tolerances',{}).get(key)!=TOLERANCES[key]:
                    raise ValueError('Saved solver acceptance tolerance differs from registration')
            parents=np.arange(len(rebuilt['state_mass'])) if family=='T0' else np.arange(len(rebuilt['state_mass']))//2
            if not np.array_equal(parents,fresh_encoder.code.parents(family)):
                raise ValueError('Frozen parent map differs from the registered refinement')
            zero_action=fresh_encoder.dictionaries[actions]['zero_action']
            if fresh_encoder.dictionaries[actions]['offsets'][zero_action]!=0:raise ValueError('Absent-parent default is not zero offset')
            inspection=inspect_channel(saved,result,parents,zero_action)
            comparison=compare_tables(saved,rebuilt)
            inspection.update(table_reconstruction=comparison,decoder_error=rebuilt['decoder_error'],
                               support_aggregates=_support_summary(rebuilt))
            inspection['deployment_support']={pool:deployment_support(e['codes'][family],replay['ctx']['pools'][pool]['weights'],
                            rebuilt['state_mass'],parents) for pool,e in encoded.items()}
            if family!='T0':
                coarse=material('T0');inspection['refinement']=refinement_errors(coarse,rebuilt)
                coarse_name=accepted[name].get('coarse_configuration')
                if coarse_name is not None:
                    coarse_spec=run.map_spec(coarse_name,anchor)
                    if (coarse_spec['input']!='T0' or coarse_spec['policy']!=spec['policy']
                            or coarse_spec['budget']!=spec['budget'] or coarse_spec['max_actions']!=actions
                            or coarse_spec.get('conditioning_family','primary')!=condition):
                        raise ValueError('Declared coarse witness is not a matched T0 channel')
                    coarse_result=run._load_map(anchor,coarse_name)
                    inspection['embedding']=embedding_errors(coarse,rebuilt,coarse_result['Q'],encoded,family,actions)
                    embedded=np.repeat(coarse_result['Q'],2,axis=0)
                    embedded_info={role:conditional_information(rebuilt['roles'][role],embedded) for role in expected_roles}
                    inspection['embedding']['CMI_nats']=embedded_info
                    if spec['budget'] is not None and any(value>spec['budget']+TOLERANCES['cmi'] for value in embedded_info.values()):
                        inspection['violations'].append('coarse_witness_privacy')
                    inspection['refined_minus_embedded_objective']=float(result['objective']-np.sum(coarse['cost']*coarse_result['Q']))
                    if inspection['refined_minus_embedded_objective']>TOLERANCES['objective_ordering']:
                        inspection['violations'].append('refinement_witness_objective')
                else:inspection['embedding']={'status':'no coarse witness declared; no feasibility inclusion assertion'}
            arrays={f'{role}/{field}':a for role,fields in rebuilt['support'].items() for field,a in fields.items()}
            arrays['state_mass']=rebuilt['state_mass']
            for pool,e in encoded.items():arrays['deployment_counts/'+pool]=np.bincount(e['codes'][family],minlength=len(parents))
            support_file=private/(name+'_support.npz');np.savez_compressed(support_file,**arrays)
            inspection['private_support_sha256']=run.sha(support_file)
            if not comparison['passed']:inspection['violations'].append('table_reconstruction')
            if rebuilt['decoder_error']>TOLERANCES['encoder_probability']:inspection['violations'].append('fixed_decoder')
            if family!='T0' and inspection['refinement']['maximum_error']>TOLERANCES['aggregation']:
                inspection['violations'].append('refinement_aggregation')
            if inspection.get('embedding',{}).get('maximum_error',0)>TOLERANCES['aggregation']:
                inspection['violations'].append('coarse_embedding')
            inspection['passed']=not inspection['violations']
            report['maps'][name]=inspection;results[name]=result;specs[name]=spec
            if inspection['reported_rank_difference']:
                report['warnings'].append({'configuration':name,'kind':'numerical_rank_convention_difference',
                                            'difference':inspection['reported_rank_difference']})
        except Exception as error:
            # Do not leak paths/IDs or arbitrary private exception strings.
            report['maps'][name]={'passed':False,'integrity_error_type':type(error).__name__,
                                  'violations':['replay_integrity_exception']}
            run.atomic(private/(name+'_incident.json'),{'type':type(error).__name__,'message':str(error)})
    # Nested-code inclusion is checked for both real refinements even if only
    # T0 channels finished. Only action/conditioning settings with accepted
    # maps above are inspected; this does not activate an optional branch.
    for actions,condition in sorted({(s['max_actions'],s.get('conditioning_family','primary')) for s in specs.values()}):
        key=f'actions_{actions}/{condition}';report['code_refinements'][key]={}
        try:
            for family in INPUTS:
                cache_key=(family,actions,condition)
                if cache_key not in tables_cache:
                    tables_cache[cache_key]=reconstruct_tables(replay,family,actions,
                                                      fine_partition=condition_partitions[condition])
            coarse=tables_cache[('T0',actions,condition)]
            for family in ('Ttask','Trisk'):
                value=refinement_errors(coarse,tables_cache[(family,actions,condition)])
                mismatches={pool:int(np.count_nonzero(np.asarray(e['codes'][family])//2!=e['codes']['T0']))
                            for pool,e in encoded.items()}
                value['deployment_parent_mismatches']=mismatches
                value['passed']=value['maximum_error']<=TOLERANCES['aggregation'] and not any(mismatches.values())
                report['code_refinements'][key][family]=value
        except Exception as error:
            report['code_refinements'][key]['integrity']={'passed':False,'error_type':type(error).__name__}
            run.atomic(private/f'{actions}_{condition}_refinement_incident.json',
                       {'type':type(error).__name__,'message':str(error)})
    groups={}
    for name,spec in specs.items():
        if spec['policy'] not in ('L','C'):continue
        key=(spec['input'],spec['max_actions'],spec['budget'],spec.get('conditioning_family','primary'))
        groups.setdefault(key,{})[spec['policy']]=name
    for group,policies in groups.items():
        if set(policies)!=set(('L','C')):continue
        local,coalition=policies['L'],policies['C']
        value=ordering_check(results[local],results[coalition])
        value.update(local_configuration=local,coalition_configuration=coalition,
                     input=group[0],actions=group[1],budget=group[2],conditioning_family=group[3])
        if not report['maps'][local]['passed'] or not report['maps'][coalition]['passed']:
            value['solver_status_supports_optimum_comparison']=False;value['qualification']='a replay integrity check failed'
        report['local_coalition_ordering'].append(value)
        if value['warning']:report['warnings'].append({'kind':'local_coalition_ordering','local':local,'coalition':coalition})
    report['passed']=(bool(report['maps']) and all(v['passed'] for v in report['maps'].values())
                      and all(v['passed'] for setting in report['code_refinements'].values() for v in setting.values())
                      and all(v['passed'] for v in encoder_report.values())
                      and all(all(v.values()) for v in h_report.values())
                      and not any(x['warning'] for x in report['local_coalition_ordering']))
    report['private_replay_directory']=str(private.relative_to(run.OUT.resolve()))
    run.atomic(private/'aggregate.json',report)
    target=Path(public_path) if public_path is not None else run.OUT/f'MATH_REPLAY_anchor_{anchor}.json'
    run.atomic(target,report)
    return report
