"""Focused integration risks: immutable historical recipe and held-out routing."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from experiments import run_acs_coalition as run


def config():return json.loads((run.DEFAULT_OUT/'config.json').read_text())


def test_new_contract_preserves_original_margins_and_rejects_matrix_drift():
    cfg=config();run.check_config(cfg)
    for key,value in [('condition_order',['F_I','F_J']),('extra_local_reduction','mean'),('maximum_scientific_seconds',7200)]:
        other=copy.deepcopy(cfg);other[key]=value
        with pytest.raises(AssertionError):run.check_config(other)
    other=copy.deepcopy(cfg);other['margins']['source_loss_increase']=.02
    with pytest.raises(AssertionError):run.check_config(other)


def test_authorized_heads_are_fitted_only_on_their_recipient(monkeypatch,tmp_path):
    calls=[]
    class Candidate:
        metadata={'validation_scores':{'log_loss':.4}}
    def fit(xf,yf,xv,yv,k,seed,**kw):
        calls.append((float(xf[0,0]),int(yf[0]),seed,kw));return {'candidates':{'logistic':Candidate()},'metadata':{}}
    monkeypatch.setattr(run,'fit_candidates',fit);monkeypatch.setattr(run,'save_candidates',lambda *a:None)
    wire={view:{pool:np.full((4,1),v) for pool in ('downstream_fit','downstream_validation')} for view,v in [('A',10),('B',20)]}
    labels={pool:{target:np.full(4,j%2) for j,target in enumerate(run.TASKS)} for pool in ('downstream_fit','downstream_validation')}
    indices={t:np.arange(4) for t in run.TASKS}
    result=run.fit_utilities(wire,labels,indices,0,tmp_path)
    assert set(result['selection'])=={'A/income_binary','A/civilian_at_work','A/same_residence','B/public_coverage','B/commute_over20'}
    assert [c[0] for c in calls]==[10,10,10,20,20]
    assert all(c[3]['budget']['mlp']['epochs']==40 for c in calls)


def test_score_preserves_one_selected_probability_array_for_both_weights():
    rows=[];pred={};prob=np.array([[.7,.3],[.25,.75]])
    run.add_scores(rows,pred,0,'fixture','audit','A','SEX',360,'fixed',['standard_independent'],lambda x,valid:prob,{'validation':None,'test':None},{'validation':{'SEX':np.array([0,-1,1])},'test':{'SEX':np.array([0,-1,1])}},{'validation':np.array([1.,999.,3.]),'test':np.array([1.,999.,3.])})
    s=rows[0]['scores'];assert s['test']['n']==2 and s['test_person_weighted']['weight_sum']==4
    assert s['test']['log_loss']==pytest.approx(-np.log([.7,.75]).mean())
    assert s['test_person_weighted']['log_loss']==pytest.approx(np.dot(-np.log([.7,.75]),[1,3])/4)
    assert np.array_equal(pred['audit/A/SEX/360/fixed/test'],prob)


def test_historical_metadata_uses_original_compact_anchor(monkeypatch,tmp_path):
    import hashlib
    parent=tmp_path/'parent';candidate=parent/'fitted/transfer/E_direct/income_binary/logistic';candidate.mkdir(parents=True)
    metadata={'family':'logistic','parameters':{'C':1}}
    (candidate/'metadata.json').write_text(json.dumps(metadata))
    (parent/'local_artifacts.json').write_text('[]')
    record={'fitting_records':{'transfer/E_direct/income_binary':{'candidates':{'logistic':metadata}}}}
    (parent/'selection_before_test.json').write_text(json.dumps(record))
    digest=hashlib.sha256((parent/'selection_before_test.json').read_bytes()).hexdigest()
    (parent/'completion.json').write_text(json.dumps({'sha256':{'selection_before_test.json':digest}}))
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run,'load_candidate',lambda p:'loaded')
    used={};assert run.checked_candidate(parent,candidate,used)=='loaded'
    assert 'parent/selection_before_test.json' in used
    (candidate/'metadata.json').write_text(json.dumps({'family':'changed'}))
    with pytest.raises(AssertionError):run.checked_candidate(parent,candidate,{})
