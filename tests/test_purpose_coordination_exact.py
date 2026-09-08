"""Independent exact identities and access-boundary regressions; no fitting."""
from fractions import Fraction as F
from itertools import product

import pytest

from scripts.verify_purpose_coordination_exact import (
    PAIRS, evaluate, forward_joint, inverse_joint, marginals, bayes_accuracy,
    independent_full_joint, binomial_own_accuracy, count_vector_bayes,
)

INDEPENDENT = {pair: (F(3,4) if pair[0] == 1 else F(1,4))*(F(3,4) if pair[1] == 1 else F(1,4)) for pair in PAIRS}
COORDINATED = {(1,1): F(1,2), (-1,1): F(1,4), (1,-1): F(1,4), (-1,-1): F(0)}


@pytest.mark.parametrize('law,expected', [(INDEPENDENT, [F(5,8),F(5,8),F(377,512)]),
                                       (COORDINATED, [F(1,2),F(5,8),F(107,128)])])
def test_full_history_bayes_matches_fixed_rational_expectations(law, expected):
    for k, target in zip((1,2,4), expected):
        joint = forward_joint(law,k,'fresh')
        inverse = inverse_joint(law,k,'fresh')
        assert joint == {key: p for key,p in inverse.items() if p}
        assert bayes_accuracy(marginals(joint)['coalition']) == target
        assert count_vector_bayes(law,k)[0] == target


@pytest.mark.parametrize('policy', ['fresh','cached'])
def test_own_history_channel_equality_and_full_S_independence(policy):
    for k in (1,2,4):
        a,b = [marginals(forward_joint(law,k,policy)) for law in (INDEPENDENT,COORDINATED)]
        for purpose in (1,2):
            assert a[f'own{purpose}'] == b[f'own{purpose}']
            assert independent_full_joint(a[f'individual{purpose}_S']) == (True,F(0))
            assert independent_full_joint(b[f'individual{purpose}_S']) == (True,F(0))
            assert bayes_accuracy(a[f'own{purpose}']) == binomial_own_accuracy(k if policy=='fresh' else 1,F(1,4))


def test_cached_returns_same_pair_and_adds_no_information():
    for law in (INDEPENDENT,COORDINATED):
        baseline = evaluate(law,1,'cached')
        for k in (2,4):
            result = evaluate(law,k,'cached')
            assert result['accuracy'] == baseline['accuracy']
            assert all(len(set(history))==1 for (_,_,history),p in result['joint'].items() if p)


def test_repeated_coordination_fails_even_though_single_history_independence_holds():
    assert evaluate(COORDINATED,1,'fresh')['independence']['coalition'][0]
    assert not evaluate(COORDINATED,2,'fresh')['independence']['coalition'][0]
    assert evaluate(COORDINATED,4,'fresh')['accuracy']['coalition'] > evaluate(INDEPENDENT,4,'fresh')['accuracy']['coalition']


def test_disjoint_error_grid_attains_decoder_union_bound():
    checked=0
    for a,b in product(range(5),repeat=2):
        q1,q2=F(a,8),F(b,8)
        if q1+q2>F(1,2):continue
        law={(1,1):1-q1-q2,(-1,1):q1,(1,-1):q2,(-1,-1):F(0)}
        result=evaluate(law,1,'fresh')
        assert result['accuracy']['own1']==1-q1
        assert result['accuracy']['own2']==1-q2
        assert result['accuracy']['coalition']==max(F(1,2),1-q1-q2)
        checked+=1
    assert checked==15


def test_exact_tasks_force_exact_coalition_recovery():
    law={pair:F(pair==(1,1)) for pair in PAIRS}
    for k in (1,2,4):
        result=evaluate(law,k,'fresh')
        assert result['accuracy']['own1']==result['accuracy']['own2']==result['accuracy']['coalition']==1
        assert result['accuracy']['individual1_S']==result['accuracy']['individual2_S']==F(1,2)


def test_pairwise_covariance_cannot_replace_full_joint_independence():
    joint={(s,(r1,r2)):F(1,4) for r1,r2 in PAIRS for s in (r1*r2,)}
    assert sum(p*s*obs[0] for (s,obs),p in joint.items())==0
    assert sum(p*s*obs[1] for (s,obs),p in joint.items())==0
    assert not independent_full_joint(joint)[0]
    assert bayes_accuracy(joint)==1
