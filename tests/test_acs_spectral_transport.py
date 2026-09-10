"""Admission checks do not invoke fitted models or final prediction scores."""
import numpy as np
import pandas as pd
import pytest
from experiments import acs_spectral_transport as t


def frame():
    n=100
    x={c:np.ones(n) for c in t.REQUIRED}
    x.update(SERIALNO=[f'2017{i//2:09d}' for i in range(n)],
             SPORDER=[str(i%2+1) for i in range(n)], ST=np.full(n,6),
             RT=['P']*n, AGEP=np.full(n,25), PWGTP=np.ones(n),
             PINCP=np.arange(n)*1000)
    return pd.DataFrame(x)


def test_household_partitions_complete_disjoint_and_row_order_invariant():
    f=frame(); p=t.partition_households(f,2017)
    assert set(p)==set(t.POOLS)
    assert sorted(np.concatenate(list(p.values())))==list(range(len(f)))
    sets=[set(f.iloc[rows].SERIALNO) for rows in p.values()]
    assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    g=f.sample(frac=1,random_state=8).reset_index(drop=True)
    q=t.partition_households(g,2017)
    assert all(set(f.iloc[p[k]].SERIALNO)==set(g.iloc[q[k]].SERIALNO) for k in p)


def test_schema_rejects_silent_category_remapping():
    f=frame(); f.loc[0,'RELP']=18
    with pytest.raises(ValueError,match='RELP'):t.validate_frame(f,2017)


def test_schema_rejects_wrong_release_fractional_key_and_missing_weight():
    for key,value in [('SERIALNO','2018000000001'),('SPORDER','1.5'),('PWGTP',np.nan)]:
        f=frame(); f.loc[0,key]=value
        with pytest.raises(ValueError): t.validate_frame(f,2017)


def test_conflicting_person_duplicate_rejected_and_identical_deduplicated():
    f=frame(); g=pd.concat([f,f.iloc[[0]]],ignore_index=True)
    a,m=t.eligible_cohort(g,2017)
    assert len(a)==len(f) and m['duplicate_rows_removed']==1
    g.loc[len(g)-1,'PINCP']=999
    with pytest.raises(ValueError,match='Conflicting'):t.eligible_cohort(g,2017)


def test_exact_original_labels_and_missing_masks():
    f=frame().iloc[:4].copy()
    f['PINCP']=[50000,50001,np.nan,-19999]
    f['ESR']=[1,2,6,np.nan]; f['MIG']=[1,2,3,np.nan]
    f['JWMNP']=[20,21,200,np.nan]; f['PUBCOV']=[1,2,1,np.nan]
    y=t.fixed_labels(f)
    assert y['income_binary'].tolist()==[0,1,-1,-1]
    assert y['civilian_at_work'].tolist()==[1,0,0,-1]
    assert y['same_residence'].tolist()==[1,0,0,-1]
    assert y['commute_over20'].tolist()==[0,1,1,-1]
    assert y['public_coverage'].tolist()==[1,0,1,-1]


def test_partitions_ignore_targets_weights_and_features():
    f=frame(); p=t.partition_households(f,2017)
    for c in set(t.REQUIRED)-{'SERIALNO','SPORDER'}:f[c]=0
    assert all(np.array_equal(v,t.partition_households(f,2017)[k]) for k,v in p.items())


def test_labels_match_original_pipeline_without_fitting():
    from experiments.acs_transfer_data import source_labels,heldout_labels,audit_labels
    f=frame(); y=t.fixed_labels(f)
    original=source_labels(f,np.array([0,50000]))
    assert np.array_equal(y['income_binary'],original['income_binary'])
    assert np.array_equal(y['civilian_at_work'],(original['esr']==0).astype(int))
    assert np.array_equal(y['public_coverage'],(original['pubcov']==0).astype(int))
    for key,v in {**heldout_labels(f),**audit_labels(f)}.items():assert np.array_equal(y[key],v)


def test_changed_history_cannot_be_auto_admitted():
    from scripts.audit_acs_spectral_transport import require_reviewed_history
    with pytest.raises(ValueError,match='independent review'):
        require_reviewed_history({'year_token_matches':[{'path':'config.json','line':1,'text':'"year": 2017'}]})
