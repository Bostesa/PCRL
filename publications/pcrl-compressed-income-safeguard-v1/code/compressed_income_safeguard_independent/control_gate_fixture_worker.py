"""Queued finite RED/GREEN for ATT/INNER conjunction and gate vector replay."""
from __future__ import annotations
import argparse,importlib.util,json
from pathlib import Path


def fixture(*,att_good=True,inner_good=True,gate=True,nonconstant=False,float_gate=False,label_mismatch=False,freeze_mismatch=False):
    import numpy as np
    from .registry import SERVICES,score_keys
    from .replay import control_replay
    class FiniteInputs:
        def seal(self):return {'status':'ALL_READERS_FROZEN',
            'role_label_sha256':{'INNER_SELECTION':{t:('b'*64 if label_mismatch else 'a'*64) for t in ('income','occupation','sex')}},
            'reader_freeze_sha256':['b'*64 if freeze_mismatch else 'a'*64]}
        def json(self,item):return self.jsons[item['path'] if isinstance(item,dict) else item]
        def arrays(self,item,fields=None):
            self.arrays_called.append(item)
            return self.tables[item['path'] if isinstance(item,dict) else item]
    inp=FiniteInputs();inp.arrays_called=[];ids=np.arange(4,dtype=np.int64);people={'row_id':ids,'unit':ids}
    y=np.array([0,1,0,1]);occup=np.array([0,1,2,3]);p1=np.full((4,2),.5);p2=np.full((4,6),1/6)
    def pin(name):return {'path':name,'sha256':'a'*64}
    def scores(good):
        out={k:np.full(4,.5) for view in ('r1','r2','pair') for k in score_keys(view)}
        for view in ('r1','r2','pair'):
            out[view+'__SHUFFLED']=np.full(4,.5)
            out[view+'__PLANTED_0']=np.full(4,0. if good else .5)
            out[view+'__PLANTED_1']=np.full(4,1. if good else .5)
        return out
    wires={f'r{i}':np.column_stack((np.zeros((4,16)),np.zeros((4,k)),p,np.zeros(4))) for i,k,p in ((1,2,p1),(2,6,p2))}
    reference=people|{'U0_1':p1,'U0_2':p2,'Ucal_1':p1,'Ucal_2':p2,'d_1':np.zeros(4),'d_2':np.zeros(4)}
    att=people|wires|{'sex':y,'validation':ids,'train':ids}
    att_scores=scores(att_good);inner_scores=scores(inner_good)
    att_controls=control_replay(att_scores,y,ids,2026101206,ids)
    gates=dict(people)
    for service in ('U0_FULL','RAW_FULL'):
        for seed in (0,1):
            value=np.full(4,gate,dtype=float if float_gate else bool)
            if nonconstant:value[-1]=not gate
            gates[f'{service}__S{seed}']=value
    inp.tables={'income':people|{'income':y},'occupation':people|{'occupation':occup},'sex':people|{'sex':y},
                'release':people|wires,'scores':inner_scores,'att':att,'att_scores':att_scores,'gates':gates,'refs0':reference,'refs1':reference}
    inp.jsons={'manifest':{'roles':{'INNER_SELECTION':{'references_s0':'refs0','references_s1':'refs1'}}}}
    banks=[]
    for service in SERVICES:
        for seed in (0,1):
            name=f'freeze_{service}_{seed}'
            inp.jsons[name]={'status':'FROZEN_COMPLETE','source_seed':seed,'inner_release_sha256':'a'*64,'score_sha256':'a'*64,
                'att_release_sha256':'a'*64,'att_scores_sha256':'a'*64,'ATT_validation_controls':att_controls}
            banks.append({'arm_id':service,'seed':seed,'release':pin('release'),'scores':pin('scores'),'freeze':pin(name),
                          'att_release':pin('att'),'att_scores':pin('att_scores')})
    inp.cfg={'manifest':pin('manifest'),'truth':{t:pin(t) for t in ('income','occupation','sex')},'banks':banks,
             'runner_control_gates':pin('gates'),'reporting_lock':pin('synthetic_reporting_lock')}
    return inp


def main():
    from .replay import Inputs
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True);args=parser.parse_args();inputs=Inputs(args.spec);cfg=inputs.cfg
    if cfg['mode']=='red':
        path=inputs.path(cfg['preserved_worker']);spec=importlib.util.spec_from_file_location('compressed_income_safeguard_independent._before_control_gate_E',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        refused=False
        try:module.run(fixture(att_good=False,inner_good=True,gate=True))
        except (AssertionError,ValueError):refused=True
        result={'schema':'CIS1-control-gate-fixture-v1','mode':'RED','status':'PASS' if refused else 'RED_FAILED_AS_EXPECTED',
                'legacy_refused_false_positive_ATT_gate':refused,'engineering_only':True,'actual_person_arrays_opened':False}
        failed=not refused
    elif cfg['mode']=='green':
        from .final_worker import run
        cases=[('ATT failure cannot inherit true gate',{'att_good':False}),
               ('INNER failure cannot inherit true gate',{'inner_good':False}),
               ('nonconstant gate refused',{'nonconstant':True}),('float gate refused',{'float_gate':True})]
        cases += [('uncommitted truth pin refused before materialization',{'label_mismatch':True}),
                  ('uncommitted reader freeze refused before materialization',{'freeze_mismatch':True})]
        records=[]
        for name,kwargs in cases:
            refused=False;finite=fixture(**kwargs)
            try:run(finite)
            except (AssertionError,ValueError,PermissionError):refused=True
            if not refused:raise AssertionError(name)
            if kwargs.get('label_mismatch') or kwargs.get('freeze_mismatch'):
                assert not finite.arrays_called
            records.append({'test':name,'status':'PASS'})
        valid=run(fixture());assert all(valid['control_gate_validity'].values())
        records.append({'test':'true gate independently verifies both roles','status':'PASS'})
        invalid=run(fixture(att_good=False,inner_good=False,gate=False));assert not any(invalid['control_gate_validity'].values())
        records.append({'test':'valid constant false gates preserved for MISSING inference','status':'PASS'})
        result={'schema':'CIS1-control-gate-fixture-v1','mode':'GREEN','status':'PASS','tests':records,'test_count':len(records),
                'engineering_only':True,'actual_person_arrays_opened':False,'scientific_runner_imports':False}
        failed=False
    else:raise ValueError('registered fixture mode')
    output=Path(cfg['public_output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({'mode':cfg['mode'],'status':result['status']}))
    if failed:raise AssertionError('preserved verifier missed ATT/root real-control gate replay')

if __name__=='__main__':main()
