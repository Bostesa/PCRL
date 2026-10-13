"""Fixed data-free bank-state shape preflight, with small exact equivalence."""
from __future__ import annotations
import argparse,json,resource,time
from pathlib import Path
import numpy as np
from .scheduler import assert_active_from_env
from .source import load_manifest,load_bank,write_once,sha_file
from .contexts import ContextMap
from .templates import build_template_law
from .preparation import task_costs,reference_costs,constraint_rows
from .memory_lp import LazyTaskCosts,protection_program,direct_bayes,solve_protection
from .lp import protection_program as original_program,direct_bayes as original_bayes,ConstraintRows,build_mixture_problem


def run(config):
    assert_active_from_env(stage='admission');cfg=json.loads(Path(config).read_text())
    # Locked tiny test exercises alias tuples, weighted SEX masses and epigraph
    # order. No Adult arrays or optimizer outcome determines fixture design.
    e1=np.array([[0,0,1],[1,0,1],[0,1,1],[1,0,0]],np.int64)
    e2=np.array([[0,1,0],[1,1,0],[1,0,1],[0,1,0]],np.int64)
    i1=e1[:,0];i2=e2[:,0];donor=np.full((4,3),1/3,np.float64)
    law=build_mixture_problem(e1,e2,i1,i2,donor,np.array([0,1,0,1],np.int64))
    c=ConstraintRows(['tiny_budget'],np.array([[.3,.1]]),np.array([.2]));sex=np.array([0,1,1,0],np.int64)
    old=original_program(law,sex,c);new=protection_program(law,sex,c)
    fields=('objective','inequality_rhs','equality_rhs','bounds')
    for field in fields:np.testing.assert_array_equal(getattr(old,field),getattr(new,field))
    for field in ('inequality_matrix','equality_matrix'):
        a,b=getattr(old,field),getattr(new,field)
        if (a!=b).nnz:raise AssertionError('low-memory matrix changed')
    if old.names!=new.names:raise AssertionError('epigraph ordering changed')
    for theta in (np.array([0.,0.]),np.array([.25,.75]),np.array([1.,1.])):
        np.testing.assert_allclose(original_bayes(law,theta,sex)[0],direct_bayes(law,theta,sex)[0],atol=0,rtol=0)
    del old,new,law
    manifest=load_manifest(cfg['source_manifest'],cfg['source_manifest_sha256'])
    bank=load_bank(manifest,1)
    from incumbent_view.r2_package import from_bank
    space=from_bank(bank)
    # Public-bank IDs repeated deterministically to the largest declared role
    # shape. They do not represent private people or their true labels.
    n=15434;c1=np.arange(n,dtype=np.int64)%len(space.probability1)
    c2=(np.arange(n,dtype=np.int64)//len(space.probability1))%len(space.probability2)
    report=json.loads(Path(cfg['context_report']).read_text());r=report['context_report']
    context=ContextMap(np.array(r['raw_to_context'],np.int64),np.array(r['raw_fit_counts'],np.int64),
                       np.array([d['fit_rows'] for d in r['contexts']],np.int64),tuple(d['hierarchy'] for d in r['contexts']))
    ids=context.apply(space.probability1[c1],space.probability2[c2])
    started=time.process_time()
    law,metadata=build_template_law(space,c1,c2,ids,'WIDE',1073741824)
    labels=(space.probability1[c1].argmax(1).astype(np.int64),space.probability2[c2].argmax(1).astype(np.int64))
    prepared={'code_r1':law.emitted1,'code_r2':law.emitted2}
    cost=LazyTaskCosts(space,prepared,labels)
    # Exact action-loss parity is checked on a bounded prefix, not on a giant
    # person×action×class temporary that the new adapter deliberately avoids.
    small={k:v[:11] for k,v in prepared.items()};oldcost=task_costs(space,small,tuple(y[:11] for y in labels))
    for key,value in oldcost.items():np.testing.assert_array_equal(value,cost[key][:11])
    refs={ref:{key:np.zeros(n,np.float64) for key in cost.tables} for ref in ('U0','Ucal')}
    constraints,_=constraint_rows(law,cost,np.arange(n,dtype=np.int64)%3,refs,{},'STRICT')
    synthetic_sex=(np.arange(n,dtype=np.int64)%2)
    # A full-shaped sparse program and a bounded solve expose solver workspace
    # too, rather than measuring only the small loss lookup table.
    result=solve_protection(law,synthetic_sex,constraints,time_limit=30.)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3
    receipt={'schema':'CBR1-exact-low-memory-preflight-v1','status':'PASS','unit_id':cfg['unit_id'],
             'tiny_original_program_exact_parity':True,'tiny_original_bayes_exact_parity':True,
             'action_cost_prefix_exact_parity':True,'adult_person_arrays_read':False,'adult_labels_read':False,
             'synthetic_public_bank_state_rows':n,'source_bank_seed':1,'context_rule_sha256':sha_file(cfg['context_report']),
             'support_metadata':metadata,'actual_max_rss_gib':rss,'preflight_cpu_seconds':time.process_time()-started,
             'full_shape_solver_success':bool(result['success']),'full_shape_solver_status':int(result['status']),
             'protected_class_fixture':'alternating row-index parity; no true labels',
             'scope':'engineering shape/matrix/memory check only; no Adult feasibility/privacy conclusion'}
    write_once(cfg['public_output'],receipt,False)
    print(json.dumps({'status':'PASS','max_rss_gib':rss,'maxactions':metadata['maximum_actions_per_row']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args();run(args.config)
