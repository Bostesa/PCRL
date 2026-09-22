"""Synthetic aggregate-only figure schemas and render checks."""
import copy
import importlib
import json

import pytest


def module():return importlib.import_module('experiments.pcrl_task_directed_release_v1.figures')


ROLES=('utility:A/same_residence','attack:A/SEX','attack:A/RAC1P','attack:AB/SEX','attack:AB/RAC1P')


def grid(phase='validation'):
    records={};receipts=[]
    names=('H','J','continuous_task','T0_code','T0_U_unconstrained_a17',
           'T0_L_0.002_a17','T0_C_0.002_a17','leace_A0')
    for index,name in enumerate(names):
        records[name]={}
        for anchor in (0,1,2):
            roles={}
            for r,role in enumerate(ROLES):
                value=.3+.1*r+.01*anchor-.002*index
                scores={'unweighted':value,'weighted':value+.02,'balanced':value+.01}
                fixed=None if name in ('J','T0_code','leace_A0') else {k:v+.01 for k,v in scores.items()}
                independent={k:v+.03 for k,v in scores.items()}
                if phase=='validation':
                    roles[role]={'validation':scores,'fixed_decoder':fixed,'independent_validation':independent,
                                 'selection':'chosen','independent_selection':'own'}
                else:
                    roles[role]={'ce':scores,'independent_ce':independent,'selection':'chosen','independent_selection':'own'}
                    if fixed is not None:roles[role]['fixed_decoder_ce']=fixed
            records[name][str(anchor)]=roles if phase=='validation' else {'test':roles}
            receipts.append({'configuration':name,'anchor':anchor,'summary_sha256':str(index+1)*64,
                             'receipt_sha256':str(anchor+1)*64})
    result={'records':records,'receipts':receipts,'scope':'synthetic development grid'}
    if phase=='validation':result['evaluation_opened']=False
    else:result.update(phase='evaluation',pool='test',selection_frozen=True)
    return result


def test_table_uses_equal_anchor_means_sameH_recovery_and_both_weightings():
    m=module();g=grid();table=m.build_point_table(g)
    assert table['phase']=='validation'
    assert len(table['tradeoff_points'])==8*len(g['records'])
    row=next(r for r in table['tradeoff_points'] if r['configuration']=='T0_C_0.002_a17'
             and r['role']=='A/SEX' and r['weighting']=='PWGTP')
    assert row['task_ce']==pytest.approx(.318)
    assert row['sensitive_recovery']==pytest.approx(.012)
    assert row['H_sensitive_ce']==pytest.approx(.43)
    assert row['category']=='new_Q' and row['policy']=='C'
    assert len(row['evidence'])==3
    assert row['evidence'][0]['task_pointer'].endswith('/validation/weighted')
    assert row['evidence'][0]['summary_sha256']==g['receipts'][18]['summary_sha256']


def test_native_evaluation_summary_adapter_preserves_fixed_and_independent_metrics():
    m=module();table=m.build_point_table(grid('evaluation'))
    rows=[r for r in table['decomposition_points'] if r['configuration']=='continuous_task' and r['weighting']=='unweighted']
    assert {r['predictor'] for r in rows}=={'fixed','independent'}
    assert next(r for r in rows if r['predictor']=='fixed')['ce']==pytest.approx(.316)
    assert next(r for r in rows if r['predictor']=='independent')['ce']==pytest.approx(.336)
    assert all('/test/utility:A~1same_residence/' in r['evidence'][0]['metric_pointer'] for r in rows)


def test_direct_code_never_receives_a_fabricated_fixed_decoder():
    m=module();table=m.build_point_table(grid())
    rows=[r for r in table['decomposition_points'] if r['configuration']=='T0_code']
    assert {r['predictor'] for r in rows}=={'independent'}
    assert any(r['configuration']=='T0_code' and r['predictor']=='fixed'
               for r in table['unavailable_decomposition'])


def test_incomplete_config_is_explicit_and_never_averaged_over_remaining_anchors():
    m=module();g=grid();g['records']['T0_C_0.002_a17'].pop('2')
    table=m.build_point_table(g)
    assert not any(r['configuration']=='T0_C_0.002_a17' for r in table['tradeoff_points'])
    assert table['unavailable_configurations']['T0_C_0.002_a17']['missing_anchors']==[2]
    g['records']['H'].pop('1')
    with pytest.raises(ValueError,match='H'):m.build_point_table(g)


@pytest.mark.parametrize('fault',['missing_receipt','duplicate_receipt','nonfinite','negative','mixed_phase','unfrozen_eval'])
def test_unaccepted_or_mixed_aggregate_inputs_are_rejected(fault):
    m=module();g=grid('evaluation' if fault=='unfrozen_eval' else 'validation')
    if fault=='missing_receipt':g['receipts'].pop()
    elif fault=='duplicate_receipt':g['receipts'].append(copy.deepcopy(g['receipts'][0]))
    elif fault=='nonfinite':g['records']['H']['0'][ROLES[0]]['validation']['weighted']=float('nan')
    elif fault=='negative':g['records']['H']['0'][ROLES[0]]['validation']['weighted']=-.1
    elif fault=='mixed_phase':g['phase']='evaluation';g['selection_frozen']=True
    else:g['selection_frozen']=False
    with pytest.raises(ValueError):m.build_point_table(g)


def test_all_supplied_extra_maps_and_matched_controls_are_retained():
    m=module();g=grid()
    extra={'Trisk_C_0.002_a17_fineC':{'input':'Trisk','policy':'C','budget':.002,'max_actions':17,
              'conditioning_family':'primary_intersect_fineC4','branch':'C'},
           'T0_rr_0.5_a33':{'input':'T0','canonical_release':'T0_rr_0.5','max_actions':33,
              'kind':'control','baseline_family':'rr','label_matched':True,'branch':'A'}}
    for name in extra:
        g['records'][name]=copy.deepcopy(g['records']['T0_C_0.002_a17'])
        g['receipts'] += [{**r,'configuration':name} for r in g['receipts'] if r['configuration']=='T0_C_0.002_a17']
    table=m.build_point_table(g,metadata=extra)
    assert all(sum(r['configuration']==name for r in table['tradeoff_points'])==8 for name in extra)
    q=next(r for r in table['tradeoff_points'] if r['configuration']=='Trisk_C_0.002_a17_fineC')
    control=next(r for r in table['tradeoff_points'] if r['configuration']=='T0_rr_0.5_a33')
    assert q['conditioning_family']=='primary_intersect_fineC4' and q['category']=='new_Q'
    assert control['category']=='label_matched' and control['actions']==33


def test_render_writes_traceable_png_svg_json_and_no_inference(tmp_path):
    m=module();source=tmp_path/'accepted_grid.json';source.write_text(json.dumps(grid()))
    out=tmp_path/'figures';manifest=m.render_figures(source,out)
    for name in ('utility_recovery.png','utility_recovery.svg','task_path_decomposition.png','task_path_decomposition.svg','POINTS.json'):
        assert (out/name).is_file() and name in manifest['artifacts']
    assert (out/'utility_recovery.png').read_bytes().startswith(b'\x89PNG')
    svg=(out/'utility_recovery.svg').read_text()
    assert 'mutual information' in svg and 'PWGTP' in svg
    assert manifest['inference_performed'] is False
    assert manifest['complete_configurations']==len(grid()['records'])
    assert len(json.loads((out/'POINTS.json').read_text())['tradeoff_points'])==8*len(grid()['records'])
    points=json.loads((out/'POINTS.json').read_text())
    assert all('id="point-'+r['point_id']+'"' in svg for r in points['tradeoff_points'])
    with pytest.raises(FileExistsError):m.render_figures(source,out)


def test_complete_nominal_ledger_is_kept_without_outcome_filters():
    m=module();g=grid();template=copy.deepcopy(g['records']['H'])
    names=sorted({r['configuration'] for r in m.config.release_ledger()['records']})
    g['records']={name:copy.deepcopy(template) for name in names}
    g['receipts']=[{'configuration':name,'anchor':a,'summary_sha256':'a'*64,'receipt_sha256':'b'*64}
                   for name in names for a in (0,1,2)]
    table=m.build_point_table(g)
    assert table['complete_configurations']==names
    assert len(table['tradeoff_points'])==8*len(names)
    assert {r['configuration'] for r in table['tradeoff_points']}==set(names)
    assert len({r['point_id'] for r in table['tradeoff_points']})==len(table['tradeoff_points'])
    assert all(r['sensitive_recovery']==0 for r in table['tradeoff_points'])


def test_improper_branch_metadata_fails_before_rendering():
    m=module();g=grid();g['records']['custom']=g['records'].pop('T0_C_0.002_a17')
    for r in g['receipts']:
        if r['configuration']=='T0_C_0.002_a17':r['configuration']='custom'
    with pytest.raises(ValueError,match='explicit metadata'):m.build_point_table(g)
    with pytest.raises(ValueError,match='input code'):m.build_point_table(g,metadata={'custom':{'policy':'C'}})
