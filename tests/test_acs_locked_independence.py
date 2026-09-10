import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def verifier():
    path=Path(__file__).parents[1]/'scripts/verify_acs_locked_independence.py'
    assert path.exists(), 'independent verifier is not implemented'
    spec=importlib.util.spec_from_file_location('locked_independence',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_exclusion_uses_older_household_members_and_string_keys():
    m=verifier()
    raw=pd.DataFrame({'SERIALNO':['001','001','002'],'SPORDER':['01','02','01'],
                      'AGEP':[70,22,23],'PWGTP':[1,1,1]})
    target, excluded, sample, ordering=m.independent_selection(raw,[0],30000)
    assert len(target)==2 and len(excluded)==1 and sample.SERIALNO.tolist()==['002']
    assert json.loads(sample.iloc[0].person_id)[-2:]==['002','01']


def test_whole_household_prefix_does_not_skip_large_group():
    m=verifier()
    ids=[m.canonical(m.NAMESPACE,s) for s in ('a','b','c')]
    ranked=sorted(ids,key=lambda h:(hashlib.sha256(m.canonical(m.SALT,h).encode()).digest(),h))
    serial=[json.loads(x)[1] for x in ranked]
    raw=pd.DataFrame({'SERIALNO':[serial[0]]+3*[serial[1]]+[serial[2]],
                      'SPORDER':['1','1','2','3','1'],'AGEP':5*[22],'PWGTP':5*[1]})
    _,_,sample,selected=m.independent_selection(raw,[],2)
    assert len(sample)==1 and set(sample.household_id)=={ranked[0]}
    assert len(selected)==1


def test_identical_duplicates_collapse_conflicts_fail():
    m=verifier()
    raw=pd.DataFrame({'SERIALNO':['001','001'],'SPORDER':['01','01'],'AGEP':[22,22],'PWGTP':[1,1]})
    assert len(m.independent_selection(raw,[])[0])==1
    raw.loc[1,'PWGTP']=2
    with pytest.raises(ValueError,match='Conflicting duplicate'):
        m.independent_selection(raw,[])
