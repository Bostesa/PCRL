"""Focused replay guards; no fitting or scientific artifacts required."""
import copy
import pytest
from scripts.verify_acs_coalition_strength import (
    condition_identity,candidate_belongs_to_condition,expected_system_identities,verify_global_gate,
)


def test_only_predeclared_new_strengths_are_fitted_endpoints():
    for interface in ('F','P'):
        for regime in ('Iplus','J'):
            for encoded,value in [('0p025',.025),('0p05',.05),('0p2',.2)]:
                assert condition_identity(f'{interface}_{regime}_b{encoded}') == (interface,regime,value)
    for name in ('F_I','P_Iplus_b0p1','F_J_b0','F_J_b0p3','F_X_b0p05','AB_J_b0p2'):
        with pytest.raises(ValueError): condition_identity(name)


def test_singleton_projection_cannot_import_another_deployment(tmp_path):
    dest=tmp_path/'seed_0/F_J_b0p025'
    own=dest/'audits/fitted/wire/A/SEX/fresh/nested360/mlp_0'
    assert candidate_belongs_to_condition(own,dest)==own.resolve()
    for other in ('seed_0/F_J_b0p05','seed_0/P_J_b0p025','seed_1/F_J_b0p025'):
        with pytest.raises(ValueError):
            candidate_belongs_to_condition(tmp_path/other/'audits/fitted/wire/A/SEX/fresh/nested360/mlp_0',dest)


def test_path_alias_cannot_bypass_deployment_boundary(tmp_path):
    dest=tmp_path/'seed_0/F_J_b0p025';dest.mkdir(parents=True)
    outside=tmp_path/'other';outside.mkdir()
    (dest/'audits').symlink_to(outside,target_is_directory=True)
    # The condition itself may be stored through a declared directory alias.
    # A candidate escaping its canonical fitted subtree must still be rejected.
    with pytest.raises(ValueError):
        candidate_belongs_to_condition(dest/'audits/fitted/../../stolen',dest)


def gate_fixture():
    systems=[]
    for (seed,name),(interface,family,beta,reused,original) in expected_system_identities().items():
        systems.append(dict(seed=seed,condition=name,interface=interface,family=family,beta=beta,reused=reused,original_condition=original))
    freeze=dict(all_54_systems_frozen=True,new_reserved_fitting_started=False,historical_reserved_outcomes_previously_known=True,
                created_utc='2026-09-09T00:00:00+00:00',systems=systems)
    trained=[dict(training_complete=True,reserved_labels_accessed=False,completed_utc='2026-09-08T23:59:00+00:00') for _ in range(36)]
    starts=[dict(all_54_frozen=True,first_reserved_labels_after_global_freeze=True,started_utc='2026-09-09T00:01:00+00:00')]
    return freeze,trained,starts


def test_global_freeze_requires_all54_before_first_reserved_fit():
    freeze,trained,starts=gate_fixture();verify_global_gate(freeze,trained,starts)
    bad=copy.deepcopy(starts);bad[0]['started_utc']='2026-09-08T23:59:59+00:00'
    with pytest.raises(AssertionError):verify_global_gate(freeze,trained,bad)
    with pytest.raises(AssertionError):verify_global_gate(freeze,trained[:-1],starts)
    bad=copy.deepcopy(trained);bad[-1]['completed_utc']='2026-09-09T00:00:01+00:00'
    with pytest.raises(AssertionError):verify_global_gate(freeze,bad,starts)


def test_shared_I_aliases_cannot_be_counted_as_additional_systems():
    freeze,trained,starts=gate_fixture()
    bad=copy.deepcopy(freeze);bad['systems'][-1]=copy.deepcopy(bad['systems'][0])
    with pytest.raises(AssertionError):verify_global_gate(bad,trained,starts)
    bad=copy.deepcopy(freeze);next(e for e in bad['systems'] if e['beta']==.1)['reused']=False
    with pytest.raises(AssertionError):verify_global_gate(bad,trained,starts)
