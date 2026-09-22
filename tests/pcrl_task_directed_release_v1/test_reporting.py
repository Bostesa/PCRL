import numpy as np
import pytest
from experiments.pcrl_task_directed_release_v1.reporting import paired_endpoint,evaluate_formula,check_bound


def fixture(loss):
 return {'ids':np.array(['p1','p2','p3']),'households':np.array(['h1','h1','h2']),
         'weights':np.array([1.,2.,7.]),'y':np.array([0,1,0]),'loss':np.asarray(loss,float)}


def test_recovery_sign_shared_baseline_cancels_and_identical_zero():
 arrays={'J':fixture([.6,.4,.5]),'Q':fixture([.5,.3,.8])}
 endpoint={'id':'sensitive','weighting':'PWGTP','anchors':[0,1,2],
  'terms':[{'configuration':'J','role':'attack:A/SEX','coefficient':1.,'predictor':'frozen_validation_selection'},
           {'configuration':'Q','role':'attack:A/SEX','coefficient':-1.,'predictor':'frozen_validation_selection'}]}
 result=paired_endpoint(endpoint,lambda anchor,name,role:arrays[name])
 assert np.array_equal(result['anchors'][0]['difference'],arrays['J']['loss']-arrays['Q']['loss'])
 assert np.array_equal(result['anchors'][0]['weights'],[1,2,7])
 endpoint['terms'][1]['configuration']='J'
 exact=paired_endpoint(endpoint,lambda anchor,name,role:arrays[name])
 assert np.array_equal(exact['anchors'][0]['difference'],np.zeros(3))


def test_pair_alignment_and_duplicate_people_are_rejected():
 left=fixture([.2,.3,.4]);right=fixture([.3,.2,.4]);right['ids']=right['ids'][::-1]
 endpoint={'id':'bad','weighting':'unweighted','anchors':[0,1,2],
  'terms':[{'configuration':'a','role':'attack:A/SEX','coefficient':1.},
           {'configuration':'b','role':'attack:A/SEX','coefficient':-1.}]}
 with pytest.raises(ValueError,match='alignment'):
  paired_endpoint(endpoint,lambda anchor,name,role:left if name=='a' else right)
 left['ids'][1]=left['ids'][0]
 with pytest.raises(ValueError,match='unique'):
  paired_endpoint(endpoint,lambda anchor,name,role:left)


def test_conjunction_uses_bounds_and_any_is_explicit():
 checks={('e1','NI'):True,('e2','strict'):False,('e3','strict'):True}
 formula={'kind':'all','clauses':[{'endpoint':'e1','check':'NI'},
          {'kind':'any','clauses':[{'endpoint':'e2','check':'strict'},{'endpoint':'e3','check':'strict'}]}]}
 assert evaluate_formula(formula,checks)['passed']
 assert not evaluate_formula({'kind':'blocked','reason':'no eligible comparator'},checks)['passed']
 assert not check_bound({'lower':-.1,'upper':.01},{'bound':'upper','operator':'<','threshold':0})
 assert check_bound({'lower':0,'upper':0},{'bound':'upper','operator':'<=','threshold':0})


def test_stricter_diagnostic_result_cannot_replace_passing_registered_claim():
 from experiments.pcrl_task_directed_release_v1.reporting import evaluate_claims
 endpoints=[{'id':'feasible','checks':[{'name':'gain','bound':'upper','operator':'<','threshold':0}]},
            {'id':'extreme','checks':[{'name':'gain','bound':'upper','operator':'<','threshold':0}]}]
 feasible={'endpoint':'feasible','check':'gain'}
 strict={'kind':'all','clauses':[feasible,{'endpoint':'extreme','check':'gain'}]}
 contrasts={'endpoints':endpoints,'family_size':2,
  'claim_formulas':{'utility_first':{'competitive':feasible}},
  'diagnostic_claim_formulas':{'utility_first':{'competitive_all_families':strict}}}
 bounds={'family_size':2,'bounds':{'feasible':{'lower':-.02,'upper':-.01},
                                 'extreme':{'lower':.01,'upper':.02}}}
 result=evaluate_claims(contrasts,bounds)
 assert result['claim_results']['utility_first']['competitive']['passed']
 assert not result['diagnostic_claim_results']['utility_first']['competitive_all_families']['passed']
 assert result['family_size']==2
