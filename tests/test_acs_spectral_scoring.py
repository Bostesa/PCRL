"""Predictions must preserve the actual historical raw-column audit route."""
import numpy as np
from experiments import acs_fixed_predictions_audits as old
from experiments.acs_transfer_heads import fit_candidates
from experiments.run_acs_residual_spectral import score

def test_saved_projected_predictions_equal_authoritative_contiguous_route(tmp_path):
    rng=np.random.default_rng(175)
    train=rng.normal(size=(120,4));y=(train[:,0]+.2*train[:,2]>0).astype(int)
    raw=rng.normal(size=(309,20));vy=(raw[:,0]>.2).astype(int)
    base=fit_candidates(train,y,np.ascontiguousarray(raw[:,:4]),vy,2,12,families=('logistic',))['candidates']['logistic']
    c=old.AuditCandidate(base,'wire',(0,1,2,3),{'base_candidate_directory':'synthetic'})
    w={'A':{'attacker_validation':raw,'test':raw.copy()}}
    labels={p:{'SEX':vy} for p in ('attacker_validation','test')};weights={p:np.ones(len(vy)) for p in labels}
    attacks={'candidates':{360:{'A/SEX':{'anchor__logistic':c}}},'selection':{360:{'A/SEX':{'standard_independent':'anchor__logistic'}}}}
    score(tmp_path,0,'synthetic',w,None,{'candidates':{},'selection':{}},attacks,labels,weights)
    expected=c.predict_proba({'wire':raw})
    with np.load(tmp_path/'predictions.npz') as z:
        assert np.array_equal(z['audit/A/SEX/360/anchor__logistic/test'],expected)
