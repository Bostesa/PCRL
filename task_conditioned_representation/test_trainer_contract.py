"""Queued checks of own-head wiring, continuation parity, and FIT selection."""
import unittest
import torch
from task_conditioned_representation.models import own_wire
from task_conditioned_representation.trainer import Recipe, select_common, normalized_violation, calibrate_ratio

class TrainerContract(unittest.TestCase):
    def test_recipe_rejects_unregistered_checkpoint(self):
        with self.assertRaises(ValueError): Recipe(epochs=20, checkpoints=(0,21))

    def test_common_selection_never_chooses_seed_specific_config(self):
        rows = [dict(seed=s, family='JOINT_CONDITIONAL', strength=g, epoch=5,
                     violation=(0.2 if (s==0)==(g==.3) else .5), nll=1,
                     checkpoint=f'{s}-{g}', checkpoint_sha256='a'*64)
                for s in (0,1) for g in (.3,1.)]
        result = select_common(rows, 'JOINT_CONDITIONAL', strengths=(.3,1.))
        self.assertEqual(result['strength'], .3)
        self.assertEqual([r['seed'] for r in result['nominees']], [0,1])

    def test_brier_is_summed_and_violation_has_correct_sign(self):
        base = [dict(accuracy=.8,nll=.4,brier=.2)]*2
        changed=[dict(accuracy=.789,nll=.39,brier=.206),base[1]]
        self.assertAlmostEqual(normalized_violation(changed,base),1.2)

    def test_own_wire_reads_actual_z_and_contains_every_scalar(self):
        z=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)
        head=torch.nn.Linear(2,2,dtype=torch.float64)
        with torch.no_grad(): head.weight.copy_(torch.eye(2,dtype=torch.float64)); head.bias.zero_()
        wire=own_wire(z,head,[0,1])
        self.assertEqual(wire['full'].shape,(2,7))
        self.assertTrue(torch.equal(wire['full'][:,:2],z))
        self.assertTrue(torch.allclose(wire['probabilities'],torch.softmax(z,1)))
        with self.assertRaises(ValueError): own_wire(z,head,[0,1],clean_logits=z)

    def test_calibration_floor_and_cap_are_failures(self):
        self.assertFalse(calibrate_ratio([1.]*16,[0.]*16)['supported'])
        self.assertFalse(calibrate_ratio([1.]*16,[1e-8]*16)['supported'])
        out=calibrate_ratio([2.]*16,[4.]*16)
        self.assertEqual(out['coefficient_per_ratio'],.5)

    def test_zero_strength_is_exact_same_continuation(self):
        import copy,tempfile
        from task_conditioned_representation.trainer import fit
        class Small(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.enc=torch.nn.ModuleList([torch.nn.Linear(3,2) for _ in range(2)]); self.head=torch.nn.ModuleList([torch.nn.Linear(2,k,dtype=torch.float64) for k in (2,6)])
                self.class_orders=[list(range(2)),list(range(6))];self.names=('a','b','c')
            def encode(self,i,x):return self.enc[i](x)
            def logits(self,i,z):return self.head[i](z.double())
        torch.manual_seed(7);model=Small();x=torch.randn(20,3);y=(torch.arange(20)%2,torch.arange(20)%6)
        with tempfile.TemporaryDirectory() as tmp:
            recipe=Recipe(epochs=1,checkpoints=(0,1),batch_size=7)
            first=fit(copy.deepcopy(model),x,y,torch.arange(14),torch.arange(14,20),recipe,'U_NEW',0,0,tmp+'/u')
            second=fit(copy.deepcopy(model),x,y,torch.arange(14),torch.arange(14,20),recipe,'JOINT_CONDITIONAL',0,0,tmp+'/j',target_enabled=False)
            a=torch.load(first['checkpoints'][-1]['checkpoint'],weights_only=False)['model'];b=torch.load(second['checkpoints'][-1]['checkpoint'],weights_only=False)['model']
            self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
            self.assertEqual(first['updates_per_recipient'],second['updates_per_recipient'])

    def test_protected_selection_uses_sensitive_surrogate_only_within_utility_screen(self):
        rows=[]
        for seed in (0,1):
            for strength,violation,sensitive in ((.3,.2,.7),(1.,.8,.3),(3.,1.1,.01)):
                rows.append(dict(seed=seed,family='JOINT_CONDITIONAL',strength=strength,epoch=5,violation=violation,nll=1.,sensitive=sensitive,checkpoint='private',checkpoint_sha256='a'*64))
        chosen=select_common(rows,'JOINT_CONDITIONAL')
        self.assertEqual(chosen['strength'],1.)
        self.assertEqual(chosen['fallback']['strength'],.3)

    def test_exact_twin_forced_configuration_is_honored(self):
        rows=[dict(seed=s,family='LOCAL_TWIN',strength=g,epoch=e,violation=.1,nll=1.,sensitive=g,checkpoint='private',checkpoint_sha256='a'*64) for s in (0,1) for g in (.3,1.) for e in (5,10)]
        selected=select_common(rows,'LOCAL_TWIN',forced=(1.,10))
        self.assertEqual((selected['strength'],selected['epoch']),(1.,10))

    def test_raw_actual_head_changes_are_visible_to_critics(self):
        from task_conditioned_representation.useful_service import UsefulModel
        from task_conditioned_representation.controls import RawObjective
        torch.manual_seed(9);m=UsefulModel([str(i) for i in range(83)]);x=torch.randn(64,83)
        r=RawObjective(0,[0,1]*32);r.refresh_model_reference(m,x)
        before=r._views(m,x)[2].detach().clone()
        with torch.no_grad():m.head[0].weight[0,0]+=1.
        after=r._views(m,x)[2].detach()
        self.assertFalse(torch.equal(before,after))
        self.assertTrue(torch.equal(before[:,:16],after[:,:16]))

    def test_sequential_frozen_first_encoder_and_head_are_unchanged(self):
        import copy,tempfile
        from task_conditioned_representation.trainer import fit
        from task_conditioned_representation.useful_service import UsefulModel
        torch.manual_seed(11);m=UsefulModel([str(i) for i in range(83)]);x=torch.randn(20,83);y=(torch.arange(20)%2,torch.arange(20)%6)
        frozen={k:v.clone() for k,v in m.state_dict().items() if k.startswith(('enc.0.','head.0.'))}
        with tempfile.TemporaryDirectory() as tmp:
            fit(m,x,y,torch.arange(14),torch.arange(14,20),Recipe(epochs=1,checkpoints=(0,1),batch_size=7),'U_NEW',0,0,tmp,active=(1,))
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in frozen.items()))

    def test_sequential_all_nominee_diagnostics_include_both_locals_and_pair(self):
        import random,tempfile
        from task_conditioned_representation.trainer import fit,seed_for
        from task_conditioned_representation.objectives import KernelObjective,StrataRegistry
        from task_conditioned_representation.useful_service import UsefulModel
        torch.manual_seed(23)
        # Two disjoint synthetic roles, each with32 groups in every task-pair/SEX arm.
        n=768;rows=torch.arange(n);yi=(rows//64)//6;yo=(rows//64)%6;sex=(rows//32)%2
        x=torch.randn(2*n,83);labels=(yi.repeat(2),yo.repeat(2))
        m=UsefulModel([str(i) for i in range(83)]);objective=KernelObjective()
        reg=StrataRegistry(yi.tolist(),yo.tolist(),sex.tolist(),rows.tolist())
        train=torch.arange(n);val=torch.arange(n,2*n);recipe=Recipe(epochs=1,checkpoints=(0,1),batch_size=n,gradient_log_every=100)
        with tempfile.TemporaryDirectory() as tmp:
            first=fit(m,x,labels,train,val,recipe,'LOCAL_TWIN',0,.01,tmp+'/first',objective=objective,registry=reg,selection_registry=reg,active=(0,),selection_active=(0,1),selection_family='SEQ12')
            second=fit(m,x,labels,train,val,recipe,'SEQ12',0,.01,tmp+'/second',objective=objective,registry=reg,selection_registry=reg,active=(1,),selection_active=(0,1),selection_family='SEQ12',initial_updates=tuple(first['updates_per_recipient']))
            combined=(first['checkpoints'][0],second['checkpoints'][-1])
            for row in combined:
                payload=torch.load(row['checkpoint'],weights_only=False);m.load_state_dict(payload['model'])
                check=KernelObjective();prior=payload['reference']
                # Recover the exact detached fitting references saved for this candidate.
                from task_conditioned_representation.kernels import DiagonalReference
                check.references=tuple(DiagonalReference(r['mean'],r['scale'],r['initial_scale'],torch.tensor(r['distance'],dtype=torch.float64),r['floor_count']) for r in prior)
                draws=reg.draw(random.Random(seed_for('TCR1selectiondraw',0)),'SEQ12')
                with torch.no_grad():parts=check.loss(m,x[val],draws,'SEQ12',active=(0,1))
                expected=float(parts['local0']+parts['local1']+parts['pair'])
                self.assertAlmostEqual(row['sensitive'],expected,places=12)
                self.assertEqual(row['selection_active'],[0,1])
                self.assertEqual(row['selection_family'],'SEQ12')
                self.assertEqual(payload['selection_active'],[0,1])
                self.assertEqual(payload['selection_family'],'SEQ12')
            self.assertEqual(combined[0]['updates_per_recipient'],[0,0])
            self.assertEqual(combined[1]['updates_per_recipient'],[1,1])
