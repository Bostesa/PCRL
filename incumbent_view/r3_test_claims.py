"""Data-free fixed-family claim accounting fixtures; queue execution only."""
import unittest
from itertools import combinations
from incumbent_view.r3_family import ARMS, TASKS, REFERENCES, METRICS, VIEWS
from incumbent_view.r3_claims import all_claims, oriented_interval, utility_components, recovery_components


def records():
    slots={}
    for seed in (0,1):
        for arm in ARMS:
            for task in TASKS:
                for ref in REFERENCES:
                    for metric in METRICS:
                        key=f's{seed}:utility:{arm}:{task}:{ref}:{metric}'
                        slots[key]={'point':0.,'lower':0.,'upper':0.,'status':'PASS'}
    auc={"P0":.80,"O_J":.75,"V_J":.70,"V_P":.81,"V_D":.82,"V_L":.83,"V_S12":.84,"V_S21":.85}
    for a,b in combinations(ARMS,2):
        for view in VIEWS:
            xa=(auc[a] if view=='pair' else .6);xb=(auc[b] if view=='pair' else .6)
            delta=xa-xb
            slots[f'mean:recovery:{a}:{b}:{view}:auc']={'point':delta,'lower':delta-.001,'upper':delta+.001}
    return slots


def gates():
    return {arm:{'all_two_seed_task_reference_point_caps_pass':True} for arm in ARMS}


class ClaimTests(unittest.TestCase):
    def test_all_three_can_pass_with_complete_simultaneous_slots(self):
        claims=all_claims('V_J',gates(),records())
        self.assertEqual({name:claims[name]['status'] for name in ('A','B','C')},{'A':'PASSED','B':'PASSED','C':'PASSED'})
    def test_control_minus_candidate_and_local_sign(self):
        s=records();pair=oriented_interval(s,'V_J','P0','pair');local=oriented_interval(s,'V_J','P0','r1',benefit=False)
        self.assertAlmostEqual(pair['point'],.1);self.assertAlmostEqual(local['point'],0.)
        self.assertEqual(recovery_components(s,'V_J','O_J',0.,True)[0]['status'],'PASS')
    def test_each_required_comparator_bound_matters(self):
        s=records();s['mean:recovery:V_J:V_P:pair:auc']={'point':-.001,'lower':-.002,'upper':0.}
        self.assertEqual(all_claims('V_J',gates(),s)['C']['status'],'NOT_ESTABLISHED')
    def test_missing_sequential_blocks_joint_claim(self):
        g=gates();g['V_S21']['all_two_seed_task_reference_point_caps_pass']=False
        self.assertEqual(all_claims('V_J',g,records())['C']['status'],'UNTESTED')
    def test_unrun_utility_slot_is_incomplete(self):
        s=records();del s['s1:utility:V_J:income:U0:nll']
        self.assertEqual(all_claims('V_J',gates(),s)['B']['status'],'INCOMPLETE')
    def test_exact_p0_alias_cannot_claim_A(self):
        self.assertEqual(all_claims('V_J',gates(),records(),nominee_exact_p0_alias=True)['A']['status'],'UNTESTED')


if __name__=='__main__':unittest.main()
