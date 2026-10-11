"""Queued data-free full-shaped WIDE control memory and custody preflight."""
from __future__ import annotations
import argparse,json,resource,gc,time
from pathlib import Path
import numpy as np
from .scheduler import assert_active_from_env
from .source import load_manifest,load_bank,write_once,sha_file
from .contexts import ContextMap
from .templates import build_template_law
from .wide_control_fit import coordinate_law,sequential_second_law,emitted_first_matrix,axis_valid,widths
from .controls import register_second_groups,apply_second_groups
from .preparation import constraint_rows
from .fit import archive_result
from . import memory_lp


def run(config):
    assert_active_from_env(stage='admission');cfg=json.loads(Path(config).read_text())
    manifest=load_manifest(cfg['source_manifest'],cfg['source_manifest_sha256'])
    from incumbent_view.r2_package import from_bank
    space=from_bank(load_bank(manifest,1));n=15434
    c1=np.arange(n,dtype=np.int64)%len(space.probability1)
    c2=(np.arange(n,dtype=np.int64)//len(space.probability1))%len(space.probability2)
    report=json.loads(Path(cfg['context_report']).read_text());r=report['context_report']
    context=ContextMap(np.array(r['raw_to_context'],np.int64),np.array(r['raw_fit_counts'],np.int64),
                       np.array([d['fit_rows'] for d in r['contexts']],np.int64),tuple(d['hierarchy'] for d in r['contexts']))
    contexts=context.apply(space.probability1[c1],space.probability2[c2]);groups=np.maximum(contexts,0)
    template,metadata=build_template_law(space,c1,c2,contexts,'WIDE',1073741824)
    labels=tuple(getattr(space,f'probability{r}')[c].argmax(1).astype(np.int64) for r,c in ((1,c1),(2,c2)))
    costs=memory_lp.LazyTaskCosts(space,{'code_r1':template.emitted1,'code_r2':template.emitted2},labels)
    refs={ref:{key:np.zeros(n,np.float64) for key in costs.tables} for ref in ('U0','Ucal')}
    private=Path(cfg['private_output']);private.mkdir(parents=True,exist_ok=False,mode=0o700)
    folds=np.arange(n,dtype=np.int64)%3;sex=np.arange(n,dtype=np.int64)%2
    from .reader_extension import losses_for_actions
    # Thirty-six public-bank synthetic local readers reproduce the actual
    # lazy guard matrix count/shape without loading any fitted private reader.
    tables={}
    for j in range(36):
        r=1+j%2;bank=getattr(space,f'probability{r}')
        probability=np.clip(.2+.6*np.arange(len(bank))/max(1,len(bank)-1),1e-12,1-1e-12)
        table=np.broadcast_to(np.stack((-np.log1p(-probability),-np.log(probability))),(3,2,len(bank))).copy()
        tables[f'synthetic_{j}']={'view':f'r{r}','table':table}
    guards=losses_for_actions(tables,template.emitted1,template.emitted2,folds,sex)
    width=widths(metadata);previous=None;records=[];started=time.process_time()
    for recipient in (1,2):
        fixed=np.zeros((n,width[2-recipient]));fixed[:,0]=1.
        law=coordinate_law(template,metadata,contexts,groups,fixed,recipient)
        rows,_=constraint_rows(law,costs,folds,refs,guards,'STRICT')
        result=memory_lp.solve_protection(law,sex,rows,time_limit=2.)
        archived=archive_result(private,f'COORD_{recipient}',result)
        records.append({'kind':'coordinate','recipient':recipient,'archive':archived,
                        'program_parameters':law.parameters,'solver_status':int(result['status'])})
        # Retain previous selected affine law while building the next, matching
        # the real alternating peak; immediately archived result is released.
        del result,rows,fixed
        previous=law;gc.collect()
    del previous,law;gc.collect()
    first=emitted_first_matrix(template,metadata,contexts,1);active=contexts>=0
    _,keys,_=register_second_groups(groups[active],first[active],axis_valid(contexts,metadata,1)[active])
    secondgroups=np.zeros(first.shape,np.int64)
    secondgroups[active]=apply_second_groups(groups[active],first[active],keys)
    frozen=np.zeros((n,width[0]));frozen[:,0]=1.
    law=sequential_second_law(template,metadata,contexts,secondgroups,frozen,1)
    rows,_=constraint_rows(law,costs,folds,refs,guards,'STRICT')
    program=memory_lp.protection_program(law,sex,rows)
    records.append({'kind':'sequential_second_builder','groups':len(keys),'parameters':law.parameters,
                    'program_rows':len(program.inequality_rhs)})
    del law,rows,program,secondgroups,frozen,first;gc.collect()
    rows,_=constraint_rows(template,costs,folds,refs,guards,'STRICT')
    program=memory_lp.protection_program(template,sex,rows)
    records.append({'kind':'binary_mip_epigraph_builder','parameters':template.parameters,
                    'program_rows':len(program.inequality_rhs)})
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3
    receipt={'schema':'CBR1-wide-control-memory-preflight-v1','status':'PASS' if rss<=2. else 'FAIL_MEMORY',
             'unit_id':cfg['unit_id'],'actual_max_rss_gib':rss,'preflight_cpu_seconds':time.process_time()-started,
             'adult_person_arrays_read':False,'adult_labels_read':False,'synthetic_public_bank_state_rows':n,
             'records':records,'support_metadata':metadata,'context_report_sha256':sha_file(cfg['context_report']),
             'scope':'engineering memory and immutable archival check; no Adult feasibility or privacy claim'}
    write_once(cfg['public_output'],receipt,False)
    print(json.dumps({'status':receipt['status'],'actual_max_rss_gib':rss}))
    if rss>2.:raise MemoryError('2GiB control reserve unsupported')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args();run(args.config)
