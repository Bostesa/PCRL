"""Synthetic report arithmetic and frozen-decision checks; no ACS scores read."""
import importlib.util
import json
import numpy as np
import pytest


def mod():
    assert importlib.util.find_spec('scripts.report_acs_residual_spectral') is not None, 'report generator must exist'
    from scripts import report_acs_residual_spectral
    return report_acs_residual_spectral


def test_loss_is_original_clip_renormalize_and_mask_aligned():
    m=mod()
    y=np.array([0,-1,1]); p=np.array([[1.,0.],[.2,.8]])
    losses,mask=m.person_losses(y,p)
    np.testing.assert_array_equal(mask,[True,False,True])
    np.testing.assert_allclose(losses,[-np.log(1/(1+1e-12)),-np.log(.8)],atol=1e-16)
    with pytest.raises(ValueError,match='valid'):
        m.person_losses(y,np.ones((3,2))/2)


def test_withholding_averages_person_losses_not_probabilities():
    m=mod(); h=np.array([.1,3.]); a=np.array([2.,.2]); weights=np.array([1.,4.])
    expected,sampled=m.withholding_losses(h,a,.25,np.array([.2,.8]))
    np.testing.assert_allclose(expected,[.575,2.3])
    np.testing.assert_allclose(sampled,[2.,3.])
    assert np.average(expected,weights=weights)==pytest.approx(1.955)
    rows=np.array([70,80,90]); u=m.branch_uniforms(rows,2,'E')
    np.testing.assert_array_equal(u[[2,0]],m.branch_uniforms(rows[[2,0]],2,'E'))
    np.testing.assert_array_equal(u,m.branch_uniforms(rows,2,'E'))


def test_completion_gate_never_decides_on_partial_matrix(tmp_path):
    m=mod()
    with pytest.raises(ValueError,match='freeze'):
        m.completion_status(tmp_path)
    (tmp_path/'RELEASE_MANIFEST.json').write_text(json.dumps({'all24_frozen':True}))
    status=m.completion_status(tmp_path,allow_incomplete=True)
    assert status['complete'] is False and status['completed_units']==0
    with pytest.raises(ValueError,match='42'):
        m.completion_status(tmp_path)


def test_directional_vector_does_not_cancel_local_race():
    m=mod(); u={t:.4 for t in m.UTILITY_TASKS}; gains={e:.1 for e in m.SENSITIVE}
    right={'utility':u,'gains':gains}; left={'utility':dict(u),'gains':dict(gains)}
    left['utility']['same_residence']-=.02
    left['gains']['AB/SEX']-=.03
    left['gains']['A/RAC1P']+=.002
    d=m.directional(left,right,m.UTILITY_TASKS,.001)
    assert d['utility_no_worse'] and d['strict_improvement']
    assert not d['sensitive_no_worse'] and not d['dominates']


def decision_fixture(m):
    points={}; parents={}
    for s in range(3):
        for split in ('validation','test'):
            for weight in m.WEIGHTS:
                parents[s,split,weight]={t:.5 for t in m.SOURCE_TASKS}
                for c in m.CONDITIONS:
                    utility={t:.5 for t in m.UTILITY_TASKS}
                    if c!='H':utility['same_residence']=.48
                    points[s,c,split,weight,360,m.MAIN_SCOPE]={'utility':utility,'gains':{e:.1 for e in m.SENSITIVE},'assessable':True}
    return points,parents


def test_nomination_requires_shared_strict_endpoint_across_every_seed_weight():
    m=mod();points,parents=decision_fixture(m)
    arm='spectral_C1'
    for s in range(3):
        for w in m.WEIGHTS:
            points[s,arm,'test',w,360,m.MAIN_SCOPE]['gains'][m.SENSITIVE[s]]-=.002
    d=m.nominate(points,parents,complete=True)
    assert d['nominee'] is None
    for s in range(3):
        for w in m.WEIGHTS:
            points[s,arm,'test',w,360,m.MAIN_SCOPE]['gains']['AB/SEX']-=.002
    d=m.nominate(points,parents,complete=True)
    assert d['nominee']==arm
    points[2,arm,'test','person_weighted',360,m.MAIN_SCOPE]['gains']['A/RAC1P']+=.0006
    assert m.nominate(points,parents,complete=True)['nominee'] is None
    assert m.nominate(points,parents,complete=False)['decision_status']=='incomplete_no_decision'


def test_point_extraction_rejects_duplicate_selected_candidate():
    m=mod();record={'role':'utility','target':'same_residence','selected_scopes':['utility'],'scores':{'test':{'log_loss':.5}}}
    with pytest.raises(ValueError,match='Duplicate'):
        m.point_from_records([record,record],{},'test','unweighted',360,m.MAIN_SCOPE)


def test_observed_logloss_gate_does_not_invent_absolute_full_class_admission():
    m=mod();points,parents=decision_fixture(m)
    for s in range(3):
        for w in m.WEIGHTS:
            for c in m.CONDITIONS:
                points[s,c,'test',w,360,m.MAIN_SCOPE]['assessable']=False
            points[s,'spectral_C1','test',w,360,m.MAIN_SCOPE]['gains']['AB/SEX']-=.002
    d=m.nominate(points,parents,complete=True)
    assert d['nominee']=='spectral_C1'
    assert d['checks']['spectral_C1']['versus_J']['full_schema_assessable'] is False


def test_unknown_fit_support_remains_unknown_not_zero_or_crash():
    m=mod()
    assert m.fit_class_support({'fit_support':None},8) is None
    assert m.fit_class_support({},0) is None
    assert m.fit_class_support({'fit_support':[4,0]},1)==0


def test_local_loss_archive_deduplicates_equal_budget_candidates_and_maps_changed_ones(tmp_path):
    m=mod(); path=tmp_path/'losses.npz'
    cache={('H','audit/A/SEX/120/logistic/test'):np.array([.1,.2]),
           ('H','audit/A/SEX/360/logistic/test'):np.array([.1,.2]),
           ('E','audit/A/SEX/360/mlp/test'):np.array([.2,.3])}
    record=m.save_loss_vectors(path,cache,{'test':np.array([.2,.3,.4])})
    assert record['unique_loss_vectors']==2 and record['loss_references']==3
    with np.load(path) as saved:
        lookup=dict(zip(saved['reference_keys'],saved['reference_loss_ids']))
        assert lookup['H/audit/A/SEX/120/logistic/test']==lookup['H/audit/A/SEX/360/logistic/test']
        np.testing.assert_array_equal(saved['loss/'+lookup['E/audit/A/SEX/360/mlp/test']],[.2,.3])
        np.testing.assert_array_equal(saved['branch/test'],[.2,.3,.4])


def test_surrogate_summary_uses_training_attribute_trace_and_fixed_denominators():
    m=mod(); arm='spectral_C1'
    def details(norm,trace=2.):
        return {'raw_trace':trace,'valid_rows':4,'unsupported_classes':[],
                'classes':[{'output_squared_norms':{arm:norm}}]}
    local={'SEX':details(4),'RAC1P':details(2),'public_coverage':details(0)}
    coal={'SEX':details(4),'RAC1P':details(0)}
    train={'penalties':{'local':{'denominator':3,'attributes':local},'coalition':{'denominator':2,'attributes':coal}}}
    held={'test':{'local':{'SEX':details(8,999),'RAC1P':details(2,999),'public_coverage':details(0,999)},'coalition':coal}}
    rows=m.surrogate_rows(train,held,0,arms=(arm,))
    lookup={(r['pool'],r['role'],r['attribute']):r for r in rows}
    assert lookup['representation_fit','local','__mean__']['normalized_projected_moment']==pytest.approx(1.)
    assert lookup['test','local','__mean__']['normalized_projected_moment']==pytest.approx(5/3)
    assert lookup['test','local','SEX']['training_attribute_trace']==2.


def test_negative_selected_attack_counts_keep_signed_values_and_separate_history():
    m=mod(); rows=[]
    for condition,value in [('spectral_C1',-.01),('spectral_L2',-1e-16),('J',-.02),('H',0.)]:
        rows.append({'condition':condition,'value':value,'kind':'additional_recovery','split':'test','weight':'unweighted','budget':360,'scope':m.MAIN_SCOPE,'seed':0,'endpoint':'AB/SEX'})
    counts=m.negative_increment_counts(rows)
    d={r['system_group']:r for r in counts}
    assert d['spectral']['negative_count']==2 and d['spectral']['below_minus_1e12_count']==1
    assert d['historical_augmented']['negative_count']==1 and d['historical_augmented']['total_count']==1
